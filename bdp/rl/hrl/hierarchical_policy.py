import os.path as osp
from collections import defaultdict
from typing import Dict

import gym.spaces as spaces
import torch
import torch.nn as nn
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem

from bdp.common.baseline_registry import baseline_registry
from bdp.common.logging import baselines_logger
from bdp.common.tensor_dict import TensorDict
from bdp.rl.hrl.gt_hl import GtHighLevelPolicy  # noqa: F401.
from bdp.rl.hrl.high_level_policy import HighLevelPolicy
from bdp.rl.hrl.nn_hl import NnHighLevelPolicy  # noqa: F401.
from bdp.rl.hrl.skills import (ArtObjSkillPolicy,  # noqa: F401.
                               NavSkillPolicy, NoopSkillPolicy,
                               OracleNavPolicy, PickSkillPolicy,
                               PlaceSkillPolicy, ResetArmSkill, SkillPolicy,
                               TurnSkillPolicy, WaitSkillPolicy)
from bdp.rl.hrl.utils import find_action_range
from bdp.rl.ppo.policy import Policy
from bdp.utils.common import get_num_actions


@baseline_registry.register_policy
class HierarchicalPolicy(nn.Module, Policy):
    def __init__(
        self,
        config,
        full_config,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        num_envs: int,
    ):
        self.is_fake = False
        self._action_space = action_space
        self._obs_space = observation_space
        self._num_envs: int = num_envs * config.batch_dup
        self._action_size = get_num_actions(action_space)

        # Maps (skill idx -> skill)
        self._skills: Dict[int, SkillPolicy] = {}
        self._name_to_idx: Dict[str, int] = {}

        for i, (skill_id, use_skill_name) in enumerate(
            full_config.RL.USE_SKILLS.items()
        ):
            if use_skill_name == "":
                # Skip loading this skill if no name is provided
                continue
            skill_config = full_config.RL.DEFINED_SKILLS[use_skill_name]
            skill_config.merge_from_other_cfg(full_config.RL.SKILL_OVERRIDES)

            cls = eval(skill_config.skill_name)
            skill_policy = cls.from_config(
                skill_config,
                observation_space,
                action_space,
                self._num_envs,
                full_config,
            )
            self._skills[i] = skill_policy
            self._name_to_idx[skill_id] = i
        self._skill_idx_to_name = {v: k for k, v in self._name_to_idx.items()}

        self._call_high_level: torch.BoolTensor = torch.ones(
            self._num_envs, dtype=torch.bool
        )
        self._cur_skills: torch.Tensor = torch.full(
            (self._num_envs,), -1, dtype=torch.long
        )
        super().__init__()

        self._policy_cfg = config
        high_level_cls = eval(config.high_level_policy.name)
        self._robot_id = self._policy_cfg.robot_id

        task_spec_file = osp.join(
            full_config.TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH,
            full_config.TASK_CONFIG.TASK.TASK_SPEC + ".yaml",
        )
        domain_file = full_config.TASK_CONFIG.TASK.PDDL_DOMAIN_DEF

        self._pddl_problem = PddlProblem(
            domain_file,
            task_spec_file,
            config,
        )

        self._high_level_policy: HighLevelPolicy = high_level_cls(
            config.high_level_policy,
            self._pddl_problem,
            self._num_envs,
            self._name_to_idx,
            observation_space,
            action_space,
            self._robot_id,
        )

        for skill in self._skills.values():
            skill.set_pddl_problem(self._pddl_problem)

        self._storage = self._high_level_policy.get_storage_wrapper()
        self._stop_action_idx, _ = find_action_range(action_space, "REARRANGE_STOP")

        self._num_hl_calls = torch.zeros(self._num_envs)

        self._skill_redirects = {}
        first_idx = None
        for i, skill in self._skills.items():
            if self._skill_idx_to_name[i] == "noop":
                continue
            if isinstance(skill, NoopSkillPolicy):
                if first_idx is None:
                    first_idx = i
                else:
                    self._skill_redirects[i] = first_idx

    @property
    def storage(self):
        return self._storage

    @property
    def obs_space(self):
        return self._obs_space

    @property
    def action_space(self):
        return self._action_space

    def eval(self):
        pass

    def set_robot_id(self, robot_id: str):
        self._robot_id = robot_id
        self._high_level_policy.set_robot_id(robot_id)

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def num_recurrent_layers(self):
        # return self._skills[0].num_recurrent_layers
        return 1

    @property
    def should_load_agent_state(self):
        return False

    @property
    def can_learn(self):
        return False

    def parameters(self):
        raise NotImplementedError

    def to(self, device):
        for skill in self._skills.values():
            skill.to(device)

    def _broadcast_skill_ids(self, skill_ids, sel_dat, should_adds=None):
        """
        :param skill_ids: Iterable[int]
        :param should_adds: Iterable[bool]
        """
        # skill id -> [batch ids]
        grouped_skills = defaultdict(list)
        if should_adds is None:
            should_adds = [True for _ in range(len(skill_ids))]
        for i, (cur_skill, should_add) in enumerate(zip(skill_ids, should_adds)):
            if should_add:
                cur_skill = cur_skill.item()
                if cur_skill in self._skill_redirects:
                    cur_skill = self._skill_redirects[cur_skill]
                grouped_skills[cur_skill].append(i)
        for k, v in grouped_skills.items():
            grouped_skills[k] = (
                v,
                {dat_k: dat[v] for dat_k, dat in sel_dat.items()},
            )
        return dict(grouped_skills)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        self._high_level_policy.apply_mask(masks)
        log_info = [{} for _ in range(self._num_envs)]

        should_terminate: torch.BoolTensor = torch.zeros(
            (self._num_envs,), dtype=torch.bool
        )
        bad_should_terminate: torch.BoolTensor = torch.zeros(
            (self._num_envs,), dtype=torch.bool
        )
        if not isinstance(observations, TensorDict):
            observations = TensorDict(observations)

        actions = torch.zeros((self._num_envs, self._action_size), device=masks.device)

        hl_says_term = self._high_level_policy.get_termination(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            self._cur_skills,
            log_info,
        )

        grouped_skills = self._broadcast_skill_ids(
            self._cur_skills,
            sel_dat={
                "observations": observations,
                "rnn_hidden_states": rnn_hidden_states,
                "prev_actions": prev_actions,
                "masks": masks,
                "actions": actions,
                "hl_says_term": hl_says_term,
            },
        )
        for skill_id, (batch_ids, dat) in grouped_skills.items():
            if skill_id == -1:
                # Policy has not prediced a skill yet.
                should_terminate[batch_ids] = 1.0
                continue

            (
                should_terminate[batch_ids],
                bad_should_terminate[batch_ids],
                actions[batch_ids],
            ) = self._skills[skill_id].should_terminate(
                **dat,
                batch_idx=batch_ids,
                log_info=log_info,
                skill_name=[
                    self._skill_idx_to_name[self._cur_skills[i].item()]
                    for i in batch_ids
                ],
            )
        self._call_high_level = should_terminate

        self._num_hl_calls *= masks.view(-1).cpu()

        # Always call high-level if the episode is over.
        self._call_high_level |= (~masks).view(-1).cpu()

        self._num_hl_calls += self._call_high_level
        for env_i in range(self._num_envs):
            log_info[env_i]["hl_calls"] = self._num_hl_calls[env_i].item()
        prev_skills = self._cur_skills.clone()

        # If any skills want to terminate invoke the high-level policy to get
        # the next skill.
        if self._call_high_level.sum() > 0:
            baselines_logger.info(f"Getting new skills for {self._call_high_level}")
            (
                new_skills,
                new_skill_args,
                hl_terminate,
                add_hl_info,
            ) = self._high_level_policy.get_next_skill(
                observations,
                rnn_hidden_states,
                prev_actions,
                masks,
                self._call_high_level,
                deterministic,
                log_info,
            )

            sel_grouped_skills = self._broadcast_skill_ids(
                new_skills,
                sel_dat={},
                # sel_dat={
                #     "observations": observations,
                #     "rnn_hidden_states": rnn_hidden_states,
                #     "prev_actions": prev_actions,
                # },
                should_adds=self._call_high_level,
            )
            for skill_id, (batch_ids, _) in sel_grouped_skills.items():
                self._skills[skill_id].on_enter(
                    [new_skill_args[i] for i in batch_ids],
                    batch_ids,
                    observations,
                    rnn_hidden_states,
                    prev_actions,
                )
                rnn_hidden_states[batch_ids] *= 0.0
                prev_actions[batch_ids] *= 0

            self._cur_skills = ((~self._call_high_level) * self._cur_skills) + (
                self._call_high_level * new_skills
            )
        else:
            hl_terminate = torch.zeros(self._num_envs, dtype=torch.bool)
            add_hl_info = {}
        if (prev_skills != self._cur_skills).all():
            grouped_skills = self._broadcast_skill_ids(
                self._cur_skills,
                sel_dat={
                    "observations": observations,
                    "rnn_hidden_states": rnn_hidden_states,
                    "prev_actions": prev_actions,
                    "masks": masks,
                },
            )
        for skill_id, (batch_ids, batch_dat) in grouped_skills.items():
            actions[batch_ids], rnn_hidden_states[batch_ids] = self._skills[
                skill_id
            ].act(
                observations=batch_dat["observations"],
                rnn_hidden_states=batch_dat["rnn_hidden_states"],
                prev_actions=batch_dat["prev_actions"],
                masks=batch_dat["masks"],
                cur_batch_idx=batch_ids,
                full_action=actions[batch_ids],
            )

        final_should_terminate = torch.zeros_like(
            bad_should_terminate, dtype=torch.bool
        )
        if self._policy_cfg.should_call_stop:
            final_should_terminate |= bad_should_terminate
        final_should_terminate |= hl_terminate
        if final_should_terminate.sum() > 0:
            # End the episode where requested.
            for batch_idx in torch.nonzero(final_should_terminate):
                baselines_logger.info(
                    f"Calling stop action for batch {batch_idx}, {bad_should_terminate}, {hl_terminate}"
                )
                actions[batch_idx, self._stop_action_idx] = 1.0

        per_step_log = []
        # for env_i in range(self._num_envs):
        #     per_step_log.append(
        #         {"called_hl": self._call_high_level[env_i].item()}
        #     )

        return (
            None,
            actions,
            None,
            rnn_hidden_states,
            {
                "called_hl": self._call_high_level,
                **add_hl_info,
                "log_info": log_info,
                # "per_step_log": per_step_log,
            },
        )

    @classmethod
    def from_config(cls, config, policy_cfg, observation_space, action_space):
        return cls(
            policy_cfg,
            config,
            observation_space,
            action_space,
            config.NUM_ENVIRONMENTS,
        )

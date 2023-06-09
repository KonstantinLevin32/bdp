import os.path as osp
from dataclasses import dataclass

import numpy as np
import torch
from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem
from habitat.tasks.rearrange.multi_task.rearrange_pddl import RIGID_OBJ_TYPE
from habitat.tasks.rearrange.oracle_nav_action import \
    get_possible_nav_to_actions
from habitat.tasks.rearrange.rearrange_sensors import (
    LocalizationSensor, TargetGoalGpsCompassSensor,
    TargetStartGpsCompassSensor)

from bdp.common.logging import baselines_logger
from bdp.rl.hrl.skills.nn_skill import NnSkillPolicy
from bdp.rl.hrl.utils import find_action_range


class OracleNavPolicy(NnSkillPolicy):
    @dataclass
    class OracleNavActionArgs:
        action_idx: int

    def __init__(
        self,
        wrap_policy,
        config,
        action_space,
        filtered_obs_space,
        filtered_action_space,
        batch_size,
        pddl_domain_path,
        pddl_task_path,
        task_config,
    ):
        super().__init__(
            wrap_policy,
            config,
            action_space,
            filtered_obs_space,
            filtered_action_space,
            batch_size,
        )

        self._pddl_problem = PddlProblem(
            pddl_domain_path,
            pddl_task_path,
            task_config,
        )
        self._poss_actions = get_possible_nav_to_actions(self._pddl_problem)
        self._oracle_nav_ac_idx, _ = find_action_range(
            action_space, "ORACLE_NAV_ACTION"
        )
        self._prev_pos = [None for _ in range(self._batch_size)]

    def on_enter(
        self,
        skill_arg,
        batch_idx,
        observations,
        rnn_hidden_states,
        prev_actions,
    ):
        for i in batch_idx:
            self._prev_pos[i] = None
        return super().on_enter(
            skill_arg, batch_idx, observations, rnn_hidden_states, prev_actions
        )

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        filtered_action_space = ActionSpace(
            {config.NAV_ACTION_NAME: action_space[config.NAV_ACTION_NAME]}
        )
        baselines_logger.debug(
            f"Loaded action space {filtered_action_space} for skill {config.skill_name}"
        )
        return cls(
            None,
            config,
            action_space,
            observation_space,
            filtered_action_space,
            batch_size,
            full_config.TASK_CONFIG.TASK.PDDL_DOMAIN_DEF,
            osp.join(
                full_config.TASK_CONFIG.TASK.TASK_SPEC_BASE_PATH,
                full_config.TASK_CONFIG.TASK.TASK_SPEC + ".yaml",
            ),
            full_config.TASK_CONFIG.TASK,
        )

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        ret = torch.zeros(masks.shape[0], dtype=torch.bool)

        cur_pos = observations[LocalizationSensor.cls_uuid].cpu()

        for i, batch_i in enumerate(batch_idx):
            prev_pos = self._prev_pos[batch_i]
            if prev_pos is not None:
                movement = torch.linalg.norm(prev_pos - cur_pos[i])
                ret[i] = movement < self._config.STOP_THRESH
            self._prev_pos[batch_i] = cur_pos[i]

        return ret

    def _parse_skill_arg(self, skill_arg):
        marker = None
        if len(skill_arg) == 2:
            targ_obj, _ = skill_arg
        else:
            raise ValueError(f"Unexpected number of skill arguments in {skill_arg}")

        targ_obj = self._pddl_problem.get_entity(targ_obj)

        match_i = None
        for i, action in enumerate(self._poss_actions):
            match_obj = action.get_arg_value("obj")
            if match_obj is None:
                match_obj = action.get_arg_value("marker")

            if match_obj != targ_obj:
                continue
            match_i = i
            break
        if match_i is None:
            raise ValueError(f"Cannot find matching action for {skill_arg}")

        return OracleNavPolicy.OracleNavActionArgs(match_i)

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        full_action,
        deterministic=False,
    ):
        full_action = self._keep_holding_state(full_action, observations)
        action_idxs = torch.FloatTensor(
            [self._cur_skill_args[i].action_idx + 1 for i in cur_batch_idx]
        )

        full_action[:, self._oracle_nav_ac_idx] = action_idxs

        return full_action, rnn_hidden_states

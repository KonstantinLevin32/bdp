from typing import Any, List, Tuple

import torch
import yaml
from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func

from bdp.common.logging import baselines_logger
from bdp.rl.hrl.high_level_policy import HighLevelPolicy
from bdp.task.sensors import CollInfoSensor
from bdp.utils.common import get_num_actions


class GtHighLevelPolicy(HighLevelPolicy):
    def __init__(
        self,
        config,
        pddl_problem,
        num_envs,
        skill_name_to_idx,
        observation_space,
        action_space,
        robot_id,
    ):
        super().__init__(
            config,
            pddl_problem,
            num_envs,
            skill_name_to_idx,
            observation_space,
            action_space,
            robot_id,
        )

        self._solution_actions = []
        self._config = config
        agent_sol_list = self._config.solution

        for i, sol_step in enumerate(agent_sol_list):
            sol_action = parse_func(sol_step)
            # for _ in range(3):
            #     self._solution_actions.append("))

            self._solution_actions.append(sol_action)
            if i < (len(agent_sol_list) - 1) and self._config.add_resets_between:
                self._solution_actions.append(parse_func("reset_arm(0)"))
        # Add a wait action at the end.
        if self._config.add_wait_at_end:
            self._solution_actions.append(parse_func("wait(30)"))
        self._next_sol_idxs = torch.zeros(num_envs, dtype=torch.int32)

    def _get_noop(self):
        return [parse_func("noop(ROBOT_1)") for _ in range(self._config.num_waits)]

    def get_termination(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_skills,
        log_info,
    ):
        term = torch.zeros(self._num_envs, dtype=torch.bool)
        coll_info = observations[CollInfoSensor.cls_uuid]
        cur_dist = coll_info[:, 0]
        for i, skill_i in enumerate(cur_skills):
            skill_i = skill_i.item()
            if skill_i == -1:
                continue
            skill_name = self._skill_idx_to_name[skill_i]
            if "nav" not in skill_name:
                continue
            term[i] = cur_dist[i] < self._config.replan_dist

        for (
            i,
            term_i,
        ) in enumerate(term):
            if term_i:
                cur_idx = self._next_sol_idxs[i].item()
                adds = self._get_noop()
                for add in adds:
                    self._solution_actions.insert(cur_idx, add)
            log_info[i]["hl_term"] = term_i

        return term

    def apply_mask(self, mask):
        self._next_sol_idxs *= mask.cpu().view(-1)

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        deterministic,
        log_info,
    ):
        next_skill = torch.zeros(self._num_envs, dtype=torch.long)
        skill_args_data = [None for _ in range(self._num_envs)]
        immediate_end = torch.zeros(self._num_envs, dtype=torch.bool)
        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan == 1.0:
                if self._next_sol_idxs[batch_idx] >= len(self._solution_actions):
                    baselines_logger.info(
                        f"Calling for immediate end with {self._next_sol_idxs[batch_idx]}"
                    )
                    immediate_end[batch_idx] = True
                    use_idx = len(self._solution_actions) - 1
                else:
                    use_idx = self._next_sol_idxs[batch_idx].item()

                skill_name, skill_args = self._solution_actions[use_idx]
                if skill_name not in self._skill_name_to_idx:
                    raise ValueError(
                        f"Could not find skill named {skill_name} in {self._skill_name_to_idx}"
                    )
                next_skill[batch_idx] = self._skill_name_to_idx[skill_name]
                skill_args_data[batch_idx] = skill_args

                self._next_sol_idxs[batch_idx] += 1

        return next_skill, skill_args_data, immediate_end, {}

    # def get_termination(
    #     self,
    #     observations,
    #     rnn_hidden_states,
    #     prev_actions,
    #     masks,
    #     cur_skills,
    #     log_info,
    # ):
    #     return torch.zeros(self._num_envs, dtype=torch.bool)

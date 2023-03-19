import abc
from collections import defaultdict
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from habitat.tasks.rearrange.multi_task.pddl_domain import PddlProblem

from bdp.common.logging import baselines_logger
from bdp.common.storage_wrapper import StorageWrapper
from bdp.rl.hrl.utils import find_action_range
from bdp.utils.common import get_num_actions


class HighLevelPolicy(nn.Module, abc.ABC):
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
        super().__init__()
        self._config = config
        self._num_envs = num_envs
        self._robot_id = robot_id
        self._pddl_problem = pddl_problem

        self._entities_list = self._pddl_problem.get_ordered_entities_list()
        self._action_ordering = self._pddl_problem.get_ordered_actions()
        self._skill_name_to_idx = skill_name_to_idx
        self._skill_idx_to_name = {v: k for k, v in skill_name_to_idx.items()}

        self._ac_start, _ = find_action_range(action_space, "PDDL_APPLY_ACTION")

    def set_robot_id(self, robot_id: str):
        self._robot_id = robot_id

    def get_storage_wrapper(self):
        return StorageWrapper()

    def get_next_skill(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        plan_masks,
        log_inof,
    ) -> Tuple[torch.Tensor, List[Any], torch.BoolTensor]:
        """
        :returns: A tuple containing the next skill index, a list of arguments
            for the skill, and if the high-level policy requests immediate
            termination.
        """
        raise NotImplementedError

    def apply_mask(self, mask: torch.Tensor) -> None:
        pass

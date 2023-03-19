from typing import Any

import gym.spaces as spaces
import torch

from bdp.rl.hrl.skills.skill import SkillPolicy
from bdp.utils.common import get_num_actions


class NoopSkillPolicy(SkillPolicy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
    ):
        super().__init__(config, action_space, batch_size, False)

    def _parse_skill_arg(self, *args, **kwargs) -> Any:
        pass

    def _is_skill_done(
        self, observations, rnn_hidden_states, prev_actions, masks, batch_idx
    ) -> torch.BoolTensor:
        return torch.zeros(masks.size(0), dtype=torch.bool)

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
        return full_action, rnn_hidden_states

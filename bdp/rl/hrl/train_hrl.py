import os.path as osp
from typing import Dict

import gym.spaces as spaces
import torch
import torch.nn as nn
from bdp.common.baseline_registry import baseline_registry
from bdp.common.logging import baselines_logger
from bdp.rl.hrl.gt_hl import GtHighLevelPolicy  # noqa: F401.
from bdp.rl.hrl.hierarchical_policy import HierarchicalPolicy
from bdp.rl.hrl.high_level_policy import HighLevelPolicy
from bdp.rl.hrl.nn_hl import NnHighLevelPolicy  # noqa: F401.
from bdp.rl.hrl.skills import (  # noqa: F401.
    ArtObjSkillPolicy,
    NavSkillPolicy,
    OracleNavPolicy,
    PickSkillPolicy,
    PlaceSkillPolicy,
    ResetArmSkill,
    SkillPolicy,
    WaitSkillPolicy,
)
from bdp.rl.ppo.policy import Policy
from bdp.utils.common import get_num_actions


@baseline_registry.register_policy
class TrainableHierarchicalPolicy(HierarchicalPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self._high_level_policy, NnHighLevelPolicy):
            raise TypeError(
                f"Learnable HL policy does not support {self._high_level_policy}"
            )

    def parameters(self):
        return self._high_level_policy.parameters()

    @property
    def num_recurrent_layers(self):
        return self._high_level_policy.num_recurrent_layers

    def to(self, device):
        self._high_level_policy.to(device)
        super().to(device)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        return self._high_level_policy.get_value(
            observations, rnn_hidden_states, prev_actions, masks
        )

    def evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        (
            distribution,
            values,
            rnn_hidden_states,
            aux_out,
        ) = self._high_level_policy.get_pi_and_value(
            observations, rnn_hidden_states, prev_actions, masks
        )
        action_log_probs = distribution.log_probs(action)
        distribution_entropy = distribution.entropy()

        return (
            values,
            action_log_probs,
            distribution_entropy,
            rnn_hidden_states,
            aux_out
        )


    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        deterministic=False,
    ):
        _, actions, _, _, add_info = super().act(
            observations, rnn_hidden_states, prev_actions, masks, deterministic
        )

        batch_size = masks.shape[0]
        use_values = add_info.get("values", None)
        if use_values is None:
            use_values = torch.zeros((batch_size, 1), device=masks.device)
        if 'values' in add_info:
            del add_info['values']

        use_action_log_probs = add_info.get("action_log_probs", None)
        if use_action_log_probs is None:
            use_action_log_probs = torch.zeros(
                (batch_size, 1), device=masks.device
            )
        if 'action_log_probs' in add_info:
            del add_info['action_log_probs']

        use_rnn_hxs = add_info.get("rnn_hxs", None)
        if use_rnn_hxs is None:
            use_rnn_hxs = torch.zeros_like(rnn_hidden_states)
        if 'rnn_hxs' in add_info:
            del add_info['rnn_hxs']

        return (
            use_values,
            actions,
            use_action_log_probs,
            use_rnn_hxs,
            add_info,
        )

    @property
    def can_learn(self):
        return True

    @property
    def should_load_agent_state(self):
        return True

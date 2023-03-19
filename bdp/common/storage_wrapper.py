from typing import Optional, Tuple

import gym.spaces as spaces
import torch
from habitat.core.spaces import ActionSpace

from bdp.common.ma_helpers import *
from bdp.common.rollout_storage import RolloutStorage
from bdp.common.tensor_dict import TensorDict
from bdp.utils.common import get_num_actions


class StorageWrapper:
    def __init__(
        self,
        filter_obs_keys=None,
        filter_ac_keys=None,
        override_ac_space=None,
        extract_actions_fn=None,
    ):
        self._filter_obs_keys = filter_obs_keys
        self._filter_ac_keys = filter_ac_keys
        self._override_ac_space = override_ac_space
        self._extract_actions_fn = extract_actions_fn

    def is_ready_for_update(self):
        return True

    def setup(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        device,
        num_recurrent_layers=1,
        is_double_buffered: bool = False,
        discrete_actions: bool = False,
    ):
        if self._filter_obs_keys is not None:
            observation_space = spaces.Dict(
                {k: observation_space[k] for k in self._filter_obs_keys}
            )

        if self._override_ac_space is not None:
            action_space = self._override_ac_space

        if self._filter_ac_keys is not None:
            action_space = ActionSpace(
                {k: action_space[k] for k in self._filter_ac_keys}
            )

        self._storage = RolloutStorage(
            numsteps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
            num_recurrent_layers,
            (get_num_actions(action_space),),
            is_double_buffered,
            discrete_actions,
        )
        self._storage.to(device)

    def advance_rollout(self, buffer_index):
        self._storage.advance_rollout(buffer_index)

    @property
    def wrapped(self):
        return self._storage

    def _filter_obs(self, obs):
        return TensorDict({k: obs[k] for k in self._filter_obs_keys})

    def insert_first(self, batch):
        self._storage.buffers["observations"][0] = self._filter_obs(batch)

    def insert_policy(
        self,
        next_recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        buffer_index,
        add_info,
    ):
        if self._extract_actions_fn is not None:
            actions = self._extract_actions_fn(add_info, actions)
        self._storage.insert(
            next_recurrent_hidden_states=next_recurrent_hidden_states,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            buffer_index=buffer_index,
        )

    def insert_obs(self, next_observations, rewards, next_masks, buffer_index):
        next_obs = self._filter_obs(next_observations)

        self._storage.insert(
            next_observations=next_obs,
            rewards=rewards,
            next_masks=next_masks,
            buffer_index=buffer_index,
        )

    def compute_returns(self, policy, ppo_cfg):
        with torch.no_grad():
            step_batch = self._storage.buffers[self._storage.current_rollout_step_idx]
            obs = filter_ma_keys(step_batch["observations"], policy.robot_id)
            next_value = policy.get_value(
                obs,
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self._storage.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

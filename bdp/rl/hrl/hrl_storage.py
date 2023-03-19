import warnings
from collections import defaultdict
from typing import Iterator, Optional, Tuple

import gym.spaces as spaces
import numpy as np
import rl_utils.common as cutils
import torch

from bdp.common.rollout_storage import RolloutStorage
from bdp.common.storage_wrapper import StorageWrapper
from bdp.common.tensor_dict import TensorDict
from bdp.rl.hrl.hrl_logger import hrl_logger
from bdp.utils.common import get_num_actions

EPS_PPO = 1e-5


class HrlRolloutStorage(RolloutStorage):
    def __init__(
        self,
        numsteps,
        num_envs,
        observation_space,
        action_space,
        recurrent_hidden_state_size,
        num_recurrent_layers=1,
        action_shape: Optional[Tuple[int]] = None,
        is_double_buffered: bool = False,
        discrete_actions: bool = True,
        use_normalized_advantage: bool = True,
    ):
        self._use_normalized_advantage = use_normalized_advantage
        self.buffers = TensorDict()
        self.buffers["observations"] = TensorDict()

        for sensor in observation_space.spaces:
            self.buffers["observations"][sensor] = torch.from_numpy(
                np.zeros(
                    (
                        numsteps + 1,
                        num_envs,
                        *observation_space.spaces[sensor].shape,
                    ),
                    dtype=observation_space.spaces[sensor].dtype,
                )
            )

        self.buffers["recurrent_hidden_states"] = torch.zeros(
            numsteps + 1,
            num_envs,
            num_recurrent_layers,
            recurrent_hidden_state_size,
        )

        self.buffers["rewards"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["value_preds"] = torch.zeros(numsteps + 1, num_envs, 1)
        self.buffers["returns"] = torch.zeros(numsteps + 1, num_envs, 1)

        self.buffers["action_log_probs"] = torch.zeros(numsteps + 1, num_envs, 1)

        if discrete_actions:
            action_shape = (1,)
        else:
            action_shape = action_space.shape

        self.buffers["actions"] = torch.zeros(numsteps + 1, num_envs, *action_shape)
        self.buffers["prev_actions"] = torch.zeros(
            numsteps + 1, num_envs, *action_shape
        )
        if discrete_actions:
            assert isinstance(self.buffers["actions"], torch.Tensor)
            assert isinstance(self.buffers["prev_actions"], torch.Tensor)
            self.buffers["actions"] = self.buffers["actions"].long()
            self.buffers["prev_actions"] = self.buffers["prev_actions"].long()

        self.buffers["masks"] = torch.zeros(numsteps + 1, num_envs, 1, dtype=torch.bool)

        self.is_double_buffered = is_double_buffered
        self._nbuffers = 2 if is_double_buffered else 1
        self._num_envs = num_envs

        assert (self._num_envs % self._nbuffers) == 0
        # Start at 1 because we already wrote the first observation.
        self._cur_step_idxs = torch.zeros(self._num_envs, dtype=torch.long)

        self.numsteps = numsteps
        self.current_rollout_step_idxs = [0 for _ in range(self._nbuffers)]

    def to(self, device):
        self.buffers.map_in_place(lambda v: v.to(device))

    def insert(
        self,
        next_observations=None,
        next_recurrent_hidden_states=None,
        actions=None,
        action_log_probs=None,
        value_preds=None,
        rewards=None,
        next_masks=None,
        buffer_index: int = 0,
        should_inserts: torch.BoolTensor = None,
    ):
        # hrl_logger.debug(f"Writing? {should_inserts} @ {self._cur_step_idxs}")
        # hrl_logger.debug(f"         Reward: {rewards}")
        if not self.is_double_buffered:
            assert buffer_index == 0

        next_step = dict(
            observations=next_observations,
            recurrent_hidden_states=next_recurrent_hidden_states,
            prev_actions=actions,
            masks=next_masks,
        )

        current_step = dict(
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            # rewards=rewards,
        )

        next_step = TensorDict({k: v for k, v in next_step.items() if v is not None})
        current_step = TensorDict(
            {k: v for k, v in current_step.items() if v is not None}
        )

        env_idxs = torch.arange(self._num_envs)
        if rewards is not None:
            # Accumulate rewards between updates.
            reward_write_idxs = torch.maximum(
                self._cur_step_idxs - 1, torch.zeros_like(self._cur_step_idxs)
            )
            self.buffers["rewards"][reward_write_idxs, env_idxs] += rewards.to(
                self.buffers["rewards"].device
            )

        # Make sure we are not writing past the end of the buffer.
        if should_inserts.sum() == 0:
            return

        if len(next_step) > 0:
            self.buffers.set(
                (
                    self._cur_step_idxs[should_inserts],
                    env_idxs[should_inserts],
                ),
                next_step[should_inserts],
                strict=False,
                alternate_assign=True,
            )

        if len(current_step) > 0:
            self.buffers.set(
                (
                    self._cur_step_idxs[should_inserts],
                    env_idxs[should_inserts],
                ),
                current_step[should_inserts],
                strict=False,
                alternate_assign=True,
            )

    def advance_rollout(self, should_inserts, buffer_index: int = 0):
        self.current_rollout_step_idxs[buffer_index] += 1
        self._cur_step_idxs += should_inserts.long()

        is_past_buffer = self._cur_step_idxs >= self.numsteps
        if is_past_buffer.sum() > 0:
            self._cur_step_idxs[is_past_buffer] = self.numsteps - 1
            env_idxs = torch.arange(self._num_envs)
            self.buffers["rewards"][
                self._cur_step_idxs[is_past_buffer], env_idxs[is_past_buffer]
            ] = 0.0

        # hrl_logger.debug(
        #     f"Advancing rollout? {should_inserts}. Now cur step = {self._cur_step_idxs}"
        # )

    def after_update(self):
        env_idxs = torch.arange(self._num_envs)
        self.buffers[0] = self.buffers[self._cur_step_idxs, env_idxs]
        self.buffers["masks"][1:] = False
        self.buffers["rewards"][1:] = 0.0

        self.current_rollout_step_idxs = [0 for _ in self.current_rollout_step_idxs]
        self._cur_step_idxs[:] = 0

    @property
    def _preds(self):
        return self.buffers["observations"]["all_predicates"]

    def compute_returns(self, use_gae, gamma, tau):
        if not use_gae:
            raise ValueError()

        assert isinstance(self.buffers["value_preds"], torch.Tensor)
        gae = 0.0
        for step in reversed(range(self._cur_step_idxs.max() - 1)):
            delta = (
                self.buffers["rewards"][step]
                + gamma
                * self.buffers["value_preds"][step + 1]
                * self.buffers["masks"][step + 1]
                - self.buffers["value_preds"][step]
            )
            gae = delta + gamma * tau * gae * self.buffers["masks"][step + 1]
            self.buffers["returns"][step] = (  # type: ignore
                gae + self.buffers["value_preds"][step]  # type: ignore
            )

    def recurrent_generator(self, advantages, num_batches) -> Iterator[TensorDict]:
        num_environments = advantages.size(1)
        assert num_environments >= num_batches, (
            "Trainer requires the number of environments ({}) "
            "to be greater than or equal to the number of "
            "trainer mini batches ({}).".format(num_environments, num_batches)
        )
        if num_environments % num_batches != 0:
            warnings.warn(
                "Number of environments ({}) is not a multiple of the"
                " number of mini batches ({}).  This results in mini batches"
                " of different sizes, which can harm training performance.".format(
                    num_environments, num_batches
                )
            )
        for inds in torch.randperm(num_environments).chunk(num_batches):
            batch = self.buffers[0 : self.numsteps, inds]
            batch["advantages"] = advantages[: self.numsteps, inds]
            batch["recurrent_hidden_states"] = batch["recurrent_hidden_states"][0:1]
            batch["loss_mask"] = (
                torch.arange(self.numsteps, device=advantages.device)
                .view(-1, 1, 1)
                .repeat(1, len(inds), 1)
            )
            for i, env_i in enumerate(inds):
                # The -1 is to throw out the last transition.
                batch["loss_mask"][:, i] = (
                    batch["loss_mask"][:, i] < self._cur_step_idxs[env_i] - 1
                )
            # hrl_logger.debug(f"Apply loss mask {batch['loss_mask']}")

            yield batch.map(lambda v: v.flatten(0, 1))


class HrlStorageWrapper(StorageWrapper):
    def __init__(
        self,
        ac_space,
        use_normalized_advantage,
        policy_cfg,
        filter_obs_keys=None,
        filter_ac_keys=None,
    ):
        super().__init__(filter_obs_keys, filter_ac_keys)
        self._action_space = ac_space
        self._use_normalized_advantage = use_normalized_advantage
        self._write_later = {}
        self._step_batch = {}
        self._policy_cfg = policy_cfg

    def is_ready_for_update(self):
        return (
            self._storage._cur_step_idxs.sum().item() >= self._policy_cfg.min_batch_size
        )

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
        discrete_actions: bool = True,
    ):
        if self._filter_obs_keys is not None:
            observation_space = spaces.Dict(
                {k: observation_space[k] for k in self._filter_obs_keys}
            )
        action_space = self._action_space

        self._storage = HrlRolloutStorage(
            numsteps,
            num_envs,
            observation_space,
            action_space,
            recurrent_hidden_state_size,
            num_recurrent_layers,
            (get_num_actions(action_space),),
            is_double_buffered,
            discrete_actions,
            self._use_normalized_advantage,
        )
        self._storage.to(device)
        self._called_hl = torch.zeros(num_envs, dtype=torch.bool)

    def advance_rollout(self, buffer_index):
        self._storage.advance_rollout(self._called_hl, buffer_index)
        self._called_hl = torch.zeros_like(self._called_hl).bool()

    def to(self, device):
        self._storage.to(device)
        self._step_batch = TensorDict(self._step_batch)
        self._step_batch["observations"] = TensorDict(self._step_batch["observations"])
        self._step_batch.map_in_place(lambda v: v.to(device))

    def insert_policy(
        self,
        next_recurrent_hidden_states,
        actions,
        action_log_probs,
        value_preds,
        buffer_index,
        add_info,
    ):
        # hrl_logger.debug("Inserting")

        called_hl = add_info["called_hl"]
        # Don't use the LL actions, but instead use the HL actions.

        # hrl_logger.debug(f"Insert got action {cutils.tensor_hash(actions)}")
        actions = add_info.get("actions", None)
        # hrl_logger.debug(f"Insert: HL Action {actions}")

        if len(self._write_later) > 0:
            # hrl_logger.debug(
            #     f"Insert: NEXT obs {self._write_later['next_observations']['all_predicates'][:, 0]}"
            # )
            if actions is not None:
                # Action will only be non-none when the HL policy acts, which
                # is not every step.
                self._step_batch["prev_actions"] = actions
                self._step_batch[
                    "recurrent_hidden_states"
                ] = next_recurrent_hidden_states

        self._storage.insert(
            next_recurrent_hidden_states=next_recurrent_hidden_states,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=value_preds,
            buffer_index=buffer_index,
            should_inserts=called_hl,
            **self._write_later,
        )
        # hrl_logger.debug(
        #     f"After insert obs: {self._storage._preds[:, :, 0].view(-1)}"
        # )
        # hrl_logger.debug(
        #     f"actions: {self._storage.buffers['actions'][:, 0].view(-1)}"
        # )
        # hrl_logger.debug(
        #     f"rewards: {self._storage.buffers['rewards'][:, 0].view(-1)}"
        # )
        self._called_hl = called_hl

    def insert_obs(self, next_observations, rewards, next_masks, buffer_index):
        next_obs = self._filter_obs(next_observations)
        next_obs = TensorDict(next_obs)
        # hrl_logger.debug(
        #     f"Called insert obs with {next_obs['all_predicates'][:, 0]}"
        # )
        self._step_batch["observations"] = next_observations
        self._step_batch["masks"] = next_masks
        self._write_later = dict(
            next_observations=next_obs,
            rewards=rewards,
            next_masks=next_masks,
        )

    def get_cur_batch(self, buffer_index, env_slice):
        return self._step_batch

    def insert_first(self, batch):
        filtered_batch = self._filter_obs(batch)
        self._storage.buffers["observations"][0] = filtered_batch
        # hrl_logger.debug(f"First obs {filtered_batch['all_predicates'][:, 0]}")
        self._step_batch = self._storage.buffers[0]
        self._step_batch["observations"] = batch

    def compute_returns(self, policy, ppo_cfg):
        self._storage.compute_returns(ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)

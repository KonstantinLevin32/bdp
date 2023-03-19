from typing import Any, List, Tuple

import gym.spaces as spaces
import torch
from habitat.tasks.rearrange.rearrange_sensors import IsHoldingSensor

from bdp.common.logging import baselines_logger
from bdp.rl.hrl.utils import find_action_range
from bdp.rl.ppo.policy import Policy
from bdp.utils.common import get_num_actions


class SkillPolicy(Policy):
    def __init__(
        self,
        config,
        action_space: spaces.Space,
        batch_size,
        should_keep_hold_state: bool = False,
    ):
        """
        :param action_space: The overall action space of the entire task, not task specific.
        """
        self._config = config
        self._batch_size = batch_size

        self._cur_skill_step = torch.zeros(self._batch_size)
        self._should_keep_hold_state = should_keep_hold_state

        self._cur_skill_args: List[Any] = [None for _ in range(self._batch_size)]
        self._raw_skill_args = [None for _ in range(self._batch_size)]
        self._action_size = get_num_actions(action_space)

        self._grip_ac_idx = 0
        found_grip = False
        for k, space in action_space.items():
            if k != "ARM_ACTION":
                self._grip_ac_idx += get_num_actions(space)
            else:
                # The last actioin in the arm action is the grip action.
                self._grip_ac_idx += get_num_actions(space) - 1
                found_grip = True
                break
        if not found_grip:
            raise ValueError(f"Could not find grip action in {action_space}")

        self._pddl_ac_start, _ = find_action_range(action_space, "PDDL_APPLY_ACTION")
        self._delay_term = [None for _ in range(self._batch_size)]

    def set_pddl_problem(self, pddl_prob):
        self._pddl_problem = pddl_prob
        self._entities_list = self._pddl_problem.get_ordered_entities_list()
        self._action_ordering = self._pddl_problem.get_ordered_actions()

    def _internal_log(self, s, observations=None):
        baselines_logger.debug(
            f"Skill {self._config.skill_name} @ step {self._cur_skill_step}: {s}"
        )

    def _get_multi_sensor_index(self, batch_idx: int, sensor_name: str) -> int:
        """
        Gets the index to select the observation object index in `_select_obs`.
        Used when there are multiple possible goals in the scene, such as
        multiple objects to possibly rearrange.
        """
        return self._cur_skill_args[batch_idx]

    def _keep_holding_state(
        self, full_action: torch.Tensor, observations
    ) -> torch.Tensor:
        """
        Makes the action so it does not result in dropping or picking up an
        object. Used in navigation and other skills which are not supposed to
        interact through the gripper.
        """
        # Keep the same grip state as the previous action.
        is_holding = observations[IsHoldingSensor.cls_uuid].view(-1)
        # If it is not holding (0) want to keep releasing -> output -1.
        # If it is holding (1) want to keep grasping -> output +1.
        full_action[:, self._grip_ac_idx] = is_holding + (is_holding - 1.0)
        return full_action

    def _apply_postcond(self, actions, log_info, skill_name, env_i, idx):
        skill_args = self._raw_skill_args[env_i]
        action = self._pddl_problem.actions[skill_name]

        entities = [self._pddl_problem.get_entity(x) for x in skill_args]

        baselines_logger.debug(
            f"Trying to apply action {action} with arguments {entities}"
        )
        ac_idx = self._pddl_ac_start
        found = False
        for other_action in self._action_ordering:
            if other_action.name != action.name:
                ac_idx += other_action.n_args
            else:
                found = True
                break
        if not found:
            raise ValueError(f"Could not find action {action}")

        entity_idxs = [self._entities_list.index(entity) + 1 for entity in entities]
        if len(entity_idxs) != action.n_args:
            raise ValueError(
                f"Inconsistent # of args {action.n_args} versus {entity_idxs} for {action} with {skill_args} and {entities}"
            )

        actions[idx, ac_idx : ac_idx + action.n_args] = torch.tensor(
            entity_idxs, dtype=actions.dtype, device=actions.device
        )
        baselines_logger.debug(
            f"Forcing action by assigning {entity_idxs} to action {skill_name} at index {ac_idx}"
        )
        apply_action = action.clone()
        apply_action.set_param_values(entities)

        log_info[env_i]["pddl_action"] = apply_action.compact_str
        return actions

    def _should_apply_post_conds(self, sensors):
        return [self._config.apply_postconds for _ in range(self._batch_size)]

    def should_terminate(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
        actions,
        hl_says_term,
        log_info,
        skill_name,
    ) -> Tuple[torch.BoolTensor, torch.BoolTensor, torch.Tensor]:
        is_skill_done = self._is_skill_done(
            observations, rnn_hidden_states, prev_actions, masks, batch_idx
        )
        # if is_skill_done.sum() > 0:
        #     self._internal_log(
        #         f"Requested skill termination {is_skill_done}",
        #         observations,
        #     )

        cur_skill_step = self._cur_skill_step[batch_idx]
        bad_terminate = torch.zeros(
            cur_skill_step.shape,
            dtype=torch.bool,
        )
        if self._config.MAX_SKILL_STEPS > 0:
            over_max_len = cur_skill_step >= self._config.MAX_SKILL_STEPS
            if self._config.FORCE_END_ON_TIMEOUT:
                bad_terminate = over_max_len
            else:
                is_skill_done = is_skill_done | over_max_len
            # if over_max_len.sum() > 0:
            #     self._internal_log(
            #         f"Skill exceeded max steps, terminating {over_max_len}"
            #     )

        # if bad_terminate.sum() > 0:
        #     self._internal_log(
        #         f"Bad terminating due to timeout {cur_skill_step}, {bad_terminate}",
        #         observations,
        #     )
        apply_post_conds = self._should_apply_post_conds(observations)

        for i, env_i in enumerate(batch_idx):
            if self._delay_term[env_i]:
                self._delay_term[env_i] = False
                is_skill_done[i] = 1.0
            elif (
                apply_post_conds[i]
                and is_skill_done[i] == 1.0
                and hl_says_term[i] == 0.0
            ):
                actions = self._apply_postcond(
                    actions, log_info, skill_name[i], env_i, i
                )
                self._delay_term[env_i] = True
                is_skill_done[i] = 0.0

        is_skill_done |= hl_says_term

        return is_skill_done, bad_terminate, actions

    def on_enter(
        self,
        skill_arg: List[str],
        batch_idxs: List[int],
        observations,
        rnn_hidden_states,
        prev_actions,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Passes in the data at the current `batch_idxs`
        :returns: The new hidden state and prev_actions ONLY at the batch_idxs.
        """
        self._cur_skill_step[batch_idxs] = 0
        for i, batch_idx in enumerate(batch_idxs):
            self._raw_skill_args[batch_idx] = skill_arg[i]
            self._cur_skill_args[batch_idx] = self._parse_skill_arg(skill_arg[i])
            self._delay_term[batch_idx] = False

            # self._internal_log(
            #     f"Entering skill with arguments {skill_arg[i]} parsed to {self._cur_skill_args[batch_idx]}",
            #     observations,
            # )

        # return (
        #     rnn_hidden_states * 0.0,
        #     prev_actions * 0,
        # )

    @classmethod
    def from_config(
        cls, config, observation_space, action_space, batch_size, full_config
    ):
        return cls(config, action_space, batch_size)

    def act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        full_action,
        deterministic=False,
    ):
        """
        :returns: Predicted action and next rnn hidden state.
        """
        self._cur_skill_step[cur_batch_idx] += 1
        action, hxs = self._internal_act(
            observations,
            rnn_hidden_states,
            prev_actions,
            masks,
            cur_batch_idx,
            full_action,
            deterministic,
        )

        if self._should_keep_hold_state:
            action = self._keep_holding_state(action, observations)
        return action, hxs

    def to(self, device):
        self._device = device

    def _select_obs(self, obs, cur_batch_idx):
        """
        Selects out the part of the observation that corresponds to the current goal of the skill.
        """
        for k in self._config.OBS_SKILL_INPUTS:
            cur_multi_sensor_index = self._get_multi_sensor_index(cur_batch_idx, k)
            if k not in obs:
                raise ValueError(
                    f"Skill {self._config.skill_name}: Could not find {k} out of {obs.keys()}"
                )
            entity_positions = obs[k].view(
                1, -1, self._config.get("OBS_SKILL_INPUT_DIM", 3)
            )
            obs[k] = entity_positions[:, cur_multi_sensor_index]
        return obs

    def _is_skill_done(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        batch_idx,
    ) -> torch.BoolTensor:
        """
        :returns: A (batch_size,) size tensor where 1 indicates the skill wants to end and 0 if not.
        """
        return torch.zeros(masks.shape[0], dtype=torch.bool).to(masks.device)

    def _parse_skill_arg(self, skill_arg: str) -> Any:
        """
        Parses the skill argument string identifier and returns parsed skill argument information.
        """
        return skill_arg

    def _internal_act(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_batch_idx,
        full_action,
        deterministic=False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError()

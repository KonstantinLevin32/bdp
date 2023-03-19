from itertools import chain
from typing import Any, List, Tuple

import gym.spaces as spaces
import numpy as np
import torch
import torch.nn as nn
import yaml
from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func

from bdp.common.logging import baselines_logger
from bdp.common.storage_wrapper import StorageWrapper
from bdp.rl.ddppo.policy import resnet
from bdp.rl.ddppo.policy.resnet_policy import ResNetEncoder
from bdp.rl.hrl.high_level_policy import HighLevelPolicy
from bdp.rl.hrl.hrl_storage import HrlStorageWrapper
from bdp.rl.models.rnn_state_encoder import build_rnn_state_encoder
from bdp.rl.ppo.policy import CriticHead
from bdp.task.sensors import CollInfoSensor, DidRobotsCollide, RobotsDistance
from bdp.utils.common import (CategoricalNet, CustomFixedCategorical,
                              get_num_actions)


class CustomPolicyNet(nn.Module):
    def __init__(self, num_inputs: int, hidden_dim: int, num_outputs: int) -> None:
        super().__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight, gain=0.01)
                m.bias.data.fill_(0.0)

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_outputs),
        )

        self.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return CustomFixedCategorical(logits=x)


class CustomCriticNet(nn.Module):
    def __init__(self, input_size, hidden_dim: int):
        super().__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
                m.bias.data.fill_(0.0)

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(init_weights)

    def forward(self, x):
        return self.fc(x)


class MlpPassThroughRnnHxs(nn.Module):
    def __init__(self, obs_dim, hidden_size):
        super().__init__()

    @property
    def num_recurrent_layers(self):
        return 1

    def forward(self, X, rnn_hxs, masks):
        return X, rnn_hxs


class PrefDiscrimNet(nn.Module):
    def __init__(self, obs_space, config, pref_dim):
        super().__init__()
        self._config = config
        self.pref_dim = pref_dim
        self.input_size = sum(obs_space[k].shape[0] for k in self._config.in_keys)
        self.net = nn.Sequential(
            nn.Linear(self.input_size, self._config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self._config.hidden_dim, self._config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self._config.hidden_dim, pref_dim),
        )

    def extract_obs(self, obs):
        return torch.cat([obs[k] for k in self._config.in_keys], -1)

    def pred(self, obs):
        obs = torch.cat([obs[k] for k in self._config.in_keys], -1)
        return self.net(obs)

    def forward(self, obs):
        return self.net(obs)


class NnHighLevelPolicy(HighLevelPolicy):
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

        self._all_actions = None
        self._n_actions = None

        self.set_robot_id(robot_id)
        self._n_actions = len(self._all_actions)
        self._prev_term = torch.zeros((self._num_envs, 1), dtype=torch.bool)

        self.pref_discrim_net = None
        n_agents = self._config.get("N_AGENTS", -1)
        if self._config.use_pref_discrim and (
            self._config.PREF_DIM != -1 or n_agents != -1
        ):
            if self._config.PREF_DIM != -1:
                dim = self._config.PREF_DIM
            elif n_agents != -1:
                dim = n_agents
            else:
                raise ValueError()
            self.pref_discrim_net = PrefDiscrimNet(
                observation_space,
                self._config.pref_discrim,
                dim,
            )

        self._use_obs_keys = self._config.use_obs_keys
        use_obs_space = spaces.Dict(
            {
                k: v
                for k, v in observation_space.spaces.items()
                if k in self._use_obs_keys
            }
        )
        self._im_obs_space = spaces.Dict(
            {k: v for k, v in use_obs_space.items() if len(v.shape) == 3}
        )

        state_obs_space = {k: v for k, v in use_obs_space.items() if len(v.shape) == 1}
        if self._config.PREF_DIM != -1:
            state_obs_space["pref_latent"] = spaces.Box(
                shape=(self._config.PREF_DIM,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32,
            )
        self._state_obs_space = spaces.Dict(state_obs_space)
        self.aux_pred_net = None

        if len(self._im_obs_space) > 0 and self._config.backbone != "NONE":
            resnet_baseplanes = 32
            self._hidden_size = self._config.hidden_dim
            self._visual_encoder = ResNetEncoder(
                self._im_obs_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, self._config.backbone),
                normalize_visual_inputs=self._config.normalize_visual_inputs,
            )
            self._visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self._visual_encoder.output_shape),
                    self._hidden_size,
                ),
                nn.ReLU(True),
            )
            rnn_input_size = sum(v.shape[0] for v in self._state_obs_space.values())

            print("-" * 20)
            print(f"Input size {self._im_obs_space}")
            print(f"Output size {self._n_actions}")
            print("-" * 20)

            self._state_encoder = build_rnn_state_encoder(
                self._hidden_size + rnn_input_size,
                self._hidden_size,
                rnn_type=self._config.rnn_type,
                num_layers=self._config.num_rnn_layers,
            )
            self._policy = CategoricalNet(self._hidden_size, self._n_actions)
            self._critic = CriticHead(self._hidden_size)
            use_pref_dim = self._config.OTHER_PREF_DIM
            if use_pref_dim == -1:
                use_pref_dim = self._config.PREF_DIM
            if self._config.use_aux_pred and use_pref_dim != -1:
                self.aux_pred_net = nn.Sequential(
                    nn.Linear(self._hidden_size, self._config.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self._config.hidden_dim, self._config.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(self._config.hidden_dim, use_pref_dim),
                )
        else:
            self._visual_encoder = nn.Sequential()
            self._visual_fc = nn.Sequential()
            obs_dim = sum(observation_space[k].shape[0] for k in self._use_obs_keys)
            hidden_dim = self._config.hidden_dim

            print("-" * 20)
            print(f"Input size {obs_dim}")
            print(f"Output size {self._n_actions}")
            print("-" * 20)

            if self._config.use_rnn:
                self._state_encoder = build_rnn_state_encoder(
                    hidden_dim,
                    hidden_dim,
                    rnn_type=self._config.rnn_type,
                    num_layers=self._config.num_rnn_layers,
                )
            else:
                self._state_encoder = MlpPassThroughRnnHxs(
                    obs_dim, self._config.hidden_dim
                )
            self._policy = CustomPolicyNet(
                obs_dim, self._config.hidden_dim, self._n_actions
            )
            self._critic = CustomCriticNet(obs_dim, self._config.hidden_dim)

    def set_robot_id(self, robot_id: str):
        if robot_id != self._robot_id or self._all_actions is None:
            robot_id = "ROBOT_" + robot_id.split("_")[1]
            robot_entity = self._pddl_problem.get_entity(robot_id)
            self._all_actions = self._pddl_problem.get_possible_actions(
                filter_entities=[robot_entity]
            )
            if not self._config.allow_other_place:
                self._all_actions = [
                    ac
                    for ac in self._all_actions
                    if (
                        ac.name != "place"
                        or ac.param_values[0].name in ac.param_values[1].name
                    )
                ]

            if (
                self._n_actions is not None
                and len(self._all_actions) != self._n_actions
            ):
                raise ValueError("Action size changed")

            self._predicates_list = self._pddl_problem.get_possible_predicates()
        super().set_robot_id(robot_id)

    @property
    def num_recurrent_layers(self):
        return self._state_encoder.num_recurrent_layers

    def parameters(self):
        return chain(
            self._visual_encoder.parameters(),
            self._visual_fc.parameters(),
            self._policy.parameters(),
            self._state_encoder.parameters(),
            self._critic.parameters(),
        )

    def forward(self, obs, rnn_hidden_states, masks):
        hidden = []
        if len(self._im_obs_space) > 0:
            im_obs = {k: obs[k] for k in self._im_obs_space.keys()}
            visual_features = self._visual_encoder(im_obs)
            visual_features = self._visual_fc(visual_features)
            hidden.append(visual_features)

        if len(self._state_obs_space) > 0:
            hidden.extend([obs[k] for k in self._state_obs_space.keys()])
        hidden = torch.cat(hidden, -1)

        return self._state_encoder(hidden, rnn_hidden_states, masks)

    def get_pi(self, obs, rnn_hidden_states, prev_actions, masks):
        state, rnn_hidden_states = self.forward(obs, rnn_hidden_states, masks)
        distrib = self._policy(state)
        return distrib

    def get_pi_and_value(self, obs, rnn_hidden_states, prev_actions, masks):
        state, rnn_hidden_states = self.forward(obs, rnn_hidden_states, masks)
        distrib = self._policy(state)
        values = self._critic(state)
        aux_pred = None
        if self.aux_pred_net is not None:
            aux_pred = self.aux_pred_net(state)

        return distrib, values, rnn_hidden_states, aux_pred

    def to(self, device):
        self._device = device
        self._prev_term = self._prev_term.to(device)
        return super().to(device)

    def get_value(self, observations, rnn_hidden_states, prev_actions, masks):
        state, _ = self.forward(observations, rnn_hidden_states, masks)
        return self._critic(state)

    def get_storage_wrapper(self):
        use_keys = self._config.use_obs_keys[:]
        if self._config.use_pref_discrim:
            use_keys.extend(self._config.pref_discrim.in_keys)
        return HrlStorageWrapper(
            spaces.Discrete(self._n_actions),
            self._config.use_normalized_advantage,
            self._config,
            filter_obs_keys=use_keys,
        )

    def get_termination(
        self,
        observations,
        rnn_hidden_states,
        prev_actions,
        masks,
        cur_skills,
        log_info,
    ):
        self._prev_term *= masks
        term = torch.zeros(self._num_envs, dtype=torch.bool)
        if CollInfoSensor.cls_uuid in observations:
            coll_info = observations[CollInfoSensor.cls_uuid]
            cur_dist = coll_info[:, 0]
            for i, skill_i in enumerate(cur_skills):
                skill_i = skill_i.item()
                if skill_i == -1:
                    continue
                skill_name = self._skill_idx_to_name[skill_i]
                if "Nav" not in skill_name:
                    continue
                term[i] = cur_dist[i] < self._config.replan_dist
                if self._config.replan_once:
                    term[i] = term[i] and (not self._prev_term[i])

                self._prev_term[i] = term[i]

        for (
            i,
            term_i,
        ) in enumerate(term):
            log_info[i]["hl_term"] = term_i

        return term

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

        state, rnn_hidden_states = self.forward(observations, rnn_hidden_states, masks)
        distrib = self._policy(state)
        values = self._critic(state)
        if deterministic:
            skill_sel = distrib.mode()
        else:
            skill_sel = distrib.sample()
        action_log_probs = distrib.log_probs(skill_sel)

        for batch_idx, should_plan in enumerate(plan_masks):
            if should_plan != 1.0:
                continue
            use_ac = self._all_actions[skill_sel[batch_idx]]
            # baselines_logger.info(f"HL predicted {use_ac}")
            next_skill[batch_idx] = self._skill_name_to_idx[use_ac.name]
            skill_args_data[batch_idx] = [entity.name for entity in use_ac.param_values]
            log_info[batch_idx]["nn_action"] = use_ac.compact_str

        return (
            next_skill,
            skill_args_data,
            immediate_end,
            {
                "action_log_probs": action_log_probs,
                "values": values,
                "actions": skill_sel,
                "rnn_hxs": rnn_hidden_states,
            },
        )

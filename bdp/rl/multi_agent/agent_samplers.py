import os
import os.path as osp
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import torch
import torch.nn as nn

from bdp.common.ma_helpers import create_ma_policy, filter_ma_keys
from bdp.rl.ddppo.algo import DDPPO
from bdp.rl.ppo import PPO
from bdp.rl.ppo.policy import Policy


def get_ckpt_idxs(load_ckpt):
    # This is specifying a group of checkpoints.
    try_paths = ["YOUR CHECKPOINT PATH HERE"]
    full_path = None
    for try_path in try_paths:
        try_full_path = osp.join(
            try_path,
            load_ckpt,
        )
        if osp.exists(try_full_path):
            full_path = try_full_path
            break
    return [
        int(fname.split(".")[1])
        for fname in os.listdir(full_path)
        if "resume-state" not in fname
    ], full_path


@dataclass
class AgentInfo:
    policy: Policy
    updater: PPO
    # Index of the agent in the agent list
    idx: int

    def to(self, device):
        self.policy.to(device)

    def make_non_updatable(self):
        self.updater = None
        return self

    def clone(self):
        return AgentInfo(self.policy, self.updater, self.idx)

    def copy_weights(self, other):
        """
        Assign weights from `other` to `self`
        """
        self.policy.load_state_dict(other.policy.state_dict())

    def print_weights(self):
        print(self.policy._high_level_policy._policy.fc[0].weight[0])


class AgentSampler:
    def __init__(self, config, run_config, device, policy_action_space, is_distrib):
        self.config = config
        self.run_config = run_config
        self.device = device
        self.policy_action_space = policy_action_space
        self.is_distrib = is_distrib
        self._agents = None
        self._prev_policy_ids = None

    @property
    def should_force_cpu(self):
        return False

    @classmethod
    def from_config(cls, run_config, device, policy_action_space, is_distrib):
        return cls(
            run_config.RL.AGENT_SAMPLER,
            run_config,
            device,
            policy_action_space,
            is_distrib,
        )

    @property
    def updaters(self) -> Iterable[PPO]:
        """
        Return ALL updaters.
        """
        return [agent.updater for agent in self._agents if agent.updater is not None]

    def _get_policy_cfgs(self):
        ret = []
        ret = [
            self.run_config.RL.POLICIES[k]
            for k in sorted(list(self.run_config.RL.POLICIES.keys()))
        ]
        if len(ret) == 1:
            ret.append(ret[0].clone())
        return ret

    def setup_storage(self, ppo_cfg, num_envs):
        self._num_envs = num_envs
        for agent in self._agents:
            policy = agent.policy
            agent.policy.storage.setup(
                ppo_cfg.num_steps,
                num_envs,
                policy.obs_space,
                policy.action_space,
                ppo_cfg.hidden_size,
                self.device,
                num_recurrent_layers=policy.num_recurrent_layers,
                is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            )

    def set_policy(self, policy, i):
        """
        Called during eval when loading weights
        """
        self._agents[i].policy = policy

    def insert_first(self, batch):
        for agent in self._agents:
            filtered_batch = filter_ma_keys(batch, agent.policy.robot_id)
            agent.policy.storage.insert_first(filtered_batch)

    def setup(self, ppo_cfg, observation_space):
        self._agents = []
        for i, policy_cfg in enumerate(self.run_config.RL.POLICIES.values()):
            agent_info = self.create_agent(policy_cfg, ppo_cfg, observation_space, i)
            self._agents.append(agent_info)

    @property
    def num_evals(self):
        return 1

    def create_agent(
        self,
        policy_cfg,
        ppo_cfg,
        observation_space,
        idx,
        agent_idx: Optional[int] = 0,
        device=None,
    ):
        if device is None:
            device = self.device
        if agent_idx is not None:
            policy_cfg.defrost()
            policy_cfg.robot_id = f"AGENT_{agent_idx}"
            policy_cfg.freeze()

        actor_critic = create_ma_policy(
            policy_cfg,
            self.run_config,
            device,
            observation_space,
            self.policy_action_space,
            filter=agent_idx is not None,
        )

        if (
            self.run_config.RL.DDPPO.pretrained_encoder
            or self.run_config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.run_config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.run_config.RL.DDPPO.pretrained:
            actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.run_config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.run_config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.run_config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(actor_critic.critic.fc.weight)
            nn.init.constant_(actor_critic.critic.fc.bias, 0)

        updater = None
        if actor_critic.can_learn:
            updater = (DDPPO if self.is_distrib else PPO)(
                actor_critic=actor_critic,
                clip_param=ppo_cfg.clip_param,
                ppo_epoch=ppo_cfg.ppo_epoch,
                num_mini_batch=ppo_cfg.num_mini_batch,
                value_loss_coef=ppo_cfg.value_loss_coef,
                entropy_coef=ppo_cfg.entropy_coef,
                lr=ppo_cfg.lr,
                eps=ppo_cfg.eps,
                max_grad_norm=ppo_cfg.max_grad_norm,
                use_normalized_advantage=ppo_cfg.use_normalized_advantage,
            )
        return AgentInfo(actor_critic, updater, idx)

    def _ret_agents(self, agents):
        policies = []
        updaters = []
        is_new = []
        pairing_id = []
        for i, agent in enumerate(agents):
            policy = agent.policy
            policy.set_robot_id(f"AGENT_{i}")
            pairing_id.append(agent.idx)

            if self._prev_policy_ids is None:
                is_new.append(False)
            else:
                is_new.append(id(policy) not in self._prev_policy_ids)
            policies.append(policy)
            updaters.append(agent.updater)
        self._prev_policy_ids = [id(policy) for policy in policies]
        return policies, updaters, is_new, tuple(pairing_id)

    def _get_policies(self, is_eval, total_num_env_steps) -> List[AgentInfo]:
        assert len(self._agents) == 1

        return [self._agents[0]]

    def get_policies(self, is_eval, total_num_env_steps):
        use_agents = self._get_policies(is_eval, total_num_env_steps)
        if use_agents is None:
            return None, None, None, None
        return self._ret_agents(use_agents)

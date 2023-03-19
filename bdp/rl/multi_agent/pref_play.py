import random
from collections import defaultdict
import torch.distributions as pyd
import torch
import os.path as osp

import numpy as np

from bdp.rl.multi_agent.agent_samplers import (
    AgentInfo,
    AgentSampler,
    get_ckpt_idxs,
)
from bdp.rl.multi_agent.pop_play import (
    PopulationPlayAgentSampler,
)


class PrefPlayAgentSampler(PopulationPlayAgentSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def setup(self, ppo_cfg, observation_space):
        self._policy_cfg = list(self.run_config.RL.POLICIES.values())[0]
        self._ppo_cfg = ppo_cfg
        self._obs_space = observation_space
        self._visual_encoder = None
        self._agents = [
            self._create_agent(0),
            self._create_fake_agent(1),
        ]
        self._agents.append(self._create_holdout_agent(2))
        self._pref_latent = pyd.Categorical(
            probs=torch.ones(self.config.PREF_DIM, device=self.device)
            / self.config.PREF_DIM
        )
        if self.config.LOAD_POP_CKPT != "":
            ckpt_idxs, full_path = get_ckpt_idxs(self.config.LOAD_POP_CKPT)
            last_idx = max(ckpt_idxs)
            full_path = osp.join(full_path, f"ckpt.{last_idx}.pth")
            ckpts = torch.load(full_path, map_location="cpu")["state_dict"]
            # Only load the behavior policy
            self._agents[0].updater.load_state_dict(ckpts[0])
        self._is_2nd_stage = False

    def _create_holdout_agent(self, idx):
        fake_cfg = self._policy_cfg.clone()
        fake_cfg.defrost()
        fake_cfg.high_level_policy.PREF_DIM = -1
        fake_cfg.high_level_policy.OTHER_PREF_DIM = (
            self._policy_cfg.high_level_policy.PREF_DIM
        )
        fake_cfg.high_level_policy.N_AGENTS = -1
        fake_cfg.batch_dup = 1
        fake_cfg.freeze()
        agent = self.create_agent(
            fake_cfg, self._ppo_cfg, self._obs_space, idx
        )
        if self.config.REUSE_VISUAL_ENCODER:
            # Reuse the visual encoder from the original agents.
            agent.policy._high_level_policy._visual_encoder = self._agents[
                0
            ].policy._high_level_policy._visual_encoder
        return agent

    def _create_fake_agent(self, idx):
        # Make the fake policy take up no room.
        fake_cfg = self._policy_cfg.clone()
        fake_cfg.defrost()
        fake_cfg.high_level_policy.backbone = "NONE"
        fake_cfg.high_level_policy.hidden_dim = 2
        fake_cfg.freeze()
        agent = self.create_agent(
            fake_cfg, self._ppo_cfg, self._obs_space, idx
        )
        agent.policy.is_fake = True
        agent.make_non_updatable()
        return agent

    @staticmethod
    def _assign_latent(agent, latent_id, latent):
        agent.idx = latent_id.item()
        agent.policy.pref_latent = latent

    def _on_enter_2nd_stage(self):
        print("Entering 2nd stage")
        # Remove double batching in the policy network
        self._policy_cfg.defrost()
        self._policy_cfg.batch_dup = 1
        self._policy_cfg.freeze()
        old_agent = self._agents[0]
        self._agents[0] = self._create_agent(0)
        self._agents[0].policy.load_state_dict(old_agent.policy.state_dict())

        self._agents[0].policy._storage = old_agent.policy._storage

    def _get_policies(self, is_eval, total_num_env_steps):
        latent_ids = self._pref_latent.sample((2,))
        if self.config.PREF_SELF_PLAY:
            latent_ids[0] = latent_ids[1]
        latents = torch.zeros((2, self.config.PREF_DIM), device=self.device)
        latents[torch.arange(2), latent_ids] = 1.0

        if (
            self.config.SECOND_STAGE_START != -1
            and self.config.SECOND_STAGE_START <= total_num_env_steps
        ):
            if not self._is_2nd_stage:
                # Called when entering 2nd stage.
                self._on_enter_2nd_stage()
                self._is_2nd_stage = True
            ret = [self._agents[2], self._agents[0]]
            ret[1] = ret[1].clone().make_non_updatable()
        else:
            ret = self._agents[:2]
            ret[0].idx = latent_ids[0].item()
            PrefPlayAgentSampler._assign_latent(
                ret[0], latent_ids[0], latents[0]
            )

        # 2nd agent is always latent conditioned agent.
        PrefPlayAgentSampler._assign_latent(ret[1], latent_ids[1], latents[1])
        return ret

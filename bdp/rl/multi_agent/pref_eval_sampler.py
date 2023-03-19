import random
from collections import defaultdict

import numpy as np
import torch
import itertools

from bdp.rl.multi_agent.agent_samplers import (
    AgentInfo,
    AgentSampler,
)


class PrefEvalSampler(AgentSampler):
    def setup(self, ppo_cfg, observation_space):
        self._policy_cfg = list(self.run_config.RL.POLICIES.values())[0]
        self._ppo_cfg = ppo_cfg
        self._obs_space = observation_space
        self._agents = []

        ckpt_a = torch.load(self.config.EVAL_CKPT_A, map_location="cpu")[
            "state_dict"
        ]

        ckpt_b = torch.load(self.config.EVAL_CKPT_B, map_location="cpu")
        self._pref_dim = ckpt_b["config"].RL.AGENT_SAMPLER.PREF_DIM

        ckpts = [
            ckpt_a[self.config.EVAL_IDX_A],
            ckpt_b["state_dict"][0],
        ]

        pref_agent = self.create_agent(
            self._policy_cfg, self._ppo_cfg, self._obs_space, i
        )
        pref_agent.updater.load_state_dict(ckpt_b["state_dict"][0])

        fake_cfg = self._policy_cfg.clone()
        fake_cfg.defrost()
        fake_cfg.high_level_policy.PREF_DIM = -1
        fake_cfg.high_level_policy.use_pref_discrim = False
        fake_cfg.freeze()
        other_agent = self.create_agent(
            fake_cfg, self._ppo_cfg, self._obs_space, i
        )
        other_agent.policy.is_fake = True

        # Make non-updatable so we load the weights here.
        self._agents = [pref_agent, other_agent]

        for i, agent in enumerate(self._agents):
            self.set_policy(agent.updater.actor_critic, i)
            agent.make_non_updatable()

    def _get_policies(self, is_eval, total_num_env_steps):
        latents = torch.zeros((2, self.config.PREF_DIM), device=self.device)
        latents[:, self.config.EVAL_IDX_B] = 1.0
        ret = self._agents[:2]
        ret[0].idx = self.config.EVAL_IDX_B
        ret[1].idx = self.config.EVAL_IDX_A
        for i, ret_agent in enumerate(ret):
            ret_agent.policy.pref_latent = latents[i]
        return ret

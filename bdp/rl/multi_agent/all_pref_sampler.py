import random
from collections import defaultdict

import numpy as np
import torch
import itertools

from bdp.rl.multi_agent.agent_samplers import (
    AgentInfo,
    AgentSampler,
)


class AllPrefSampler(AgentSampler):
    def setup(self, ppo_cfg, observation_space):
        self._policy_cfg = list(self.run_config.RL.POLICIES.values())[0]
        self._ppo_cfg = ppo_cfg
        self._obs_space = observation_space
        self._agents = []

        # TODO: This code can be refactored to handle both checkpoints in one list.
        ckpt = torch.load(self.config.EVAL_CKPT_A, map_location="cpu")
        self._pref_dim = ckpt["config"].RL.AGENT_SAMPLER.PREF_DIM

        pref_policy_params = ckpt["state_dict"][0]
        holdout_policy_params = ckpt["state_dict"][1]

        def setup_policy(params, agent_idx, pref_dim):
            if pref_dim is not None:
                policy_cfg = self._policy_cfg.clone()
                policy_cfg.defrost()
                policy_cfg.high_level_policy.PREF_DIM = pref_dim
                policy_cfg.freeze()
            else:
                policy_cfg = self._policy_cfg

            agent = self.create_agent(
                policy_cfg, self._ppo_cfg, self._obs_space, agent_idx
            )
            agent.updater.load_state_dict(params, strict=False)
            # Make non-updatable so we load the weights here.
            self._agents.append(agent)
            self.set_policy(agent.updater.actor_critic, agent_idx)
            agent.make_non_updatable()
            return agent

        self._pref_agent = setup_policy(pref_policy_params, 0, self._pref_dim)
        self._holdout_agent = setup_policy(holdout_policy_params, 1, None)

        idxs = np.arange(self._pref_dim + 1)
        if self.config.LIMIT_AGENT_SAMPLES != -1:
            idxs = idxs[: self.config.LIMIT_AGENT_SAMPLES]

        if self.config.ALLOW_SELF_SAMPLE:
            self._pairing_order = list(
                itertools.combinations_with_replacement(idxs, 2)
            )
        else:
            self._pairing_order = list(itertools.combinations(idxs, 2))

        if self.config.FIX_AGENT_A != -1:
            self._pairing_order = [
                x for x in self._pairing_order if self.config.FIX_AGENT_A in x
            ]
        if self.config.FIX_AGENT_B != -1:
            self._pairing_order = [
                x for x in self._pairing_order if self.config.FIX_AGENT_B in x
            ]
        self._cur_pairing = 0

    @property
    def num_evals(self):
        return len(self._pairing_order)

    def _get_policies(self, is_eval, total_num_env_steps):
        if self._cur_pairing >= len(self._pairing_order):
            pair_order = self._pairing_order[-1]
        else:
            pair_order = self._pairing_order[self._cur_pairing]
        self._cur_pairing += 1

        latent_ids = pair_order
        # The holdout agent might be here.
        latents = torch.zeros((2, self._pref_dim + 1), device=self.device)
        latents[torch.arange(2), latent_ids] = 1.0

        ret = []
        for latent_id in latent_ids:
            if latent_id >= self._pref_dim:
                agent = self._holdout_agent.clone()
                agent.idx = self._pref_dim
                ret.append(agent)
            else:
                agent = self._pref_agent.clone()
                agent.idx = latent_id.item()
                # Don't include the last index which is for the holdout agent
                agent.policy.pref_latent = latents[:, :-1]
                ret.append(agent)

        return ret

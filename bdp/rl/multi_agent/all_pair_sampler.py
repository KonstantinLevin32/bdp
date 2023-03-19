import random
from collections import defaultdict

import numpy as np
import torch
import itertools

from bdp.rl.multi_agent.agent_samplers import (
    AgentInfo,
    AgentSampler,
)


class AllPairSampler(AgentSampler):
    def setup(self, ppo_cfg, observation_space):
        self._policy_cfg = list(self.run_config.RL.POLICIES.values())[0]
        self._ppo_cfg = ppo_cfg
        self._obs_space = observation_space
        self._agents = []

        ckpts = torch.load(self.config.EVAL_CKPT_A, map_location="cpu")[
            "state_dict"
        ]
        n_policies = len(ckpts)

        for i in range(n_policies):
            agent = self.create_agent(
                self._policy_cfg, self._ppo_cfg, self._obs_space, i
            )
            agent.updater.load_state_dict(ckpts[i])
            # Make non-updatable so we load the weights here.
            self._agents.append(agent)
            self.set_policy(agent.updater.actor_critic, i)
            agent.make_non_updatable()

        idxs = np.arange(len(ckpts))
        if self.config.ONLY_SELF_SAMPLE:
            self._pairing_order = [(i, i) for i in idxs]
        elif self.config.ALLOW_SELF_SAMPLE:
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
        return [self._agents[i] for i in pair_order]

import random
from collections import defaultdict

import numpy as np
import os
import os.path as osp

from bdp.rl.multi_agent.agent_samplers import (
    AgentInfo,
    AgentSampler,
    get_ckpt_idxs
)
import torch



class PopulationPlayAgentSampler(AgentSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fetch_counts = defaultdict(lambda: 0)
        self._old_agents = []
        self._is_2nd_stage = False

    def setup(self, ppo_cfg, observation_space):
        self._policy_cfg = list(self.run_config.RL.POLICIES.values())[0]
        self._ppo_cfg = ppo_cfg
        self._obs_space = observation_space
        self._agents = []

        ckpts = []
        use_num_agents = self.config.NUM_AGENTS
        if self.config.LOAD_POP_CKPT != "":
            if ".pth" not in self.config.LOAD_POP_CKPT:
                # This is specifying a group of checkpoints.
                ckpt_idxs, full_path = get_ckpt_idxs(self.config.LOAD_POP_CKPT)
                add_idxs = [
                    min(ckpt_idxs),
                    int(np.median(ckpt_idxs)),
                    max(ckpt_idxs),
                ]

                for idx in add_idxs:
                    fname = osp.join(full_path, f"ckpt.{idx}.pth")
                    idx_ckpts = torch.load(fname, map_location="cpu")[
                        "state_dict"
                    ]
                    # Don't add the holdout agent.
                    ckpts.extend(idx_ckpts[1:])
                print(f"Loaded {len(ckpts)} agents")
                # We also have the holdout agent.
                use_num_agents = len(ckpts) + 1
            else:
                ckpts = torch.load(
                    self.config.LOAD_POP_CKPT, map_location="cpu"
                )["state_dict"]

        self._visual_encoder = None
        for agent_i in range(use_num_agents):
            self._agents.append(self._create_agent(agent_i))
        # The 0th agent is the holdout agent.

        if self.should_force_cpu:
            self.device = "cpu"

        if len(ckpts) != 0:
            # DO NOT load the holdout agent.
            for i in range(1, len(ckpts)):
                if self.should_force_cpu:
                    self._agents[i].to("cpu")
                self._agents[i].updater.load_state_dict(ckpts[i])

    @property
    def should_force_cpu(self):
        return self.config.FORCE_CPU

    def _create_agent(self, idx, create_storage=False):
        agent = self.create_agent(
            self._policy_cfg,
            self._ppo_cfg,
            self._obs_space,
            idx,
        )
        if self._visual_encoder is None and self.config.REUSE_VISUAL_ENCODER:
            self._visual_encoder = (
                agent.policy._high_level_policy._visual_encoder
            )
        elif self.config.REUSE_VISUAL_ENCODER:
            agent.policy._high_level_policy._visual_encoder = (
                self._visual_encoder
            )
        if create_storage:
            agent.policy.storage.setup(
                self._ppo_cfg.num_steps,
                self._num_envs,
                agent.policy.obs_space,
                agent.policy.action_space,
                self._ppo_cfg.hidden_size,
                self.device,
                num_recurrent_layers=agent.policy.num_recurrent_layers,
                is_double_buffered=self._ppo_cfg.use_double_buffered_sampler,
            )
        return agent

    def _get_policies(self, is_eval, total_num_env_steps):
        # population play
        n_agents = len(self._agents)
        idxs = list(range(n_agents))
        single_update = False

        if self.config.SECOND_STAGE_START != -1:
            idxs = idxs[1:]
            sample_idxs = [
                0,
                *random.sample(idxs, self.config.NUM_SAMPLE_AGENTS - 1),
            ]
            if self.config.SECOND_STAGE_START <= total_num_env_steps:
                single_update = True
                if not self._is_2nd_stage:
                    print(f"Entering 2nd stage @ {total_num_env_steps}")
                    self._is_2nd_stage = True
            else:
                sample_idxs[0] = random.sample(idxs, k=1)[0]
        else:
            if self.config.SAMPLE_INTERVAL == -1:
                # Fix the ordering if we will not resample.
                sample_idxs = idxs[: self.config.NUM_SAMPLE_AGENTS]
            else:
                sample_idxs = random.sample(
                    idxs, self.config.NUM_SAMPLE_AGENTS
                )

        if self.config.SELF_PLAY and not self._is_2nd_stage:
            for i in range(self.config.NUM_SAMPLE_AGENTS):
                sample_idxs[i] = sample_idxs[0]
            single_update = True

        ret_agents = [(self._agents[i], i) for i in sample_idxs]

        old_sample_prob = len(self._old_agents) / (
            len(self._old_agents) + len(self._agents)
        )
        for i in range(self.config.NUM_SAMPLE_AGENTS - 1):
            if random.random() < old_sample_prob:
                # Replace with old checkpoint
                old_idx = np.random.randint(len(self._old_agents))
                ret_agents[i + 1] = (self._old_agents[old_idx], None)
                assert ret_agents[i + 1][0].updater is None

        if single_update or self.config.SINGLE_UPDATE:
            # Do not update the second agent.
            ret_agents[1] = (
                ret_agents[1][0].clone().make_non_updatable(),
                ret_agents[1][1],
            )

        return [agent for agent, _ in ret_agents]

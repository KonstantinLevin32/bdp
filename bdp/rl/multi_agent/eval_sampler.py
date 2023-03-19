import random
from collections import defaultdict

import numpy as np
import torch

from bdp.rl.multi_agent.agent_samplers import (
    AgentInfo,
    AgentSampler,
)


class EvalAgentSampler(AgentSampler):
    def setup(self, ppo_cfg, observation_space):

        self._policy_cfgs = self._get_policy_cfgs()
        self._ppo_cfg = ppo_cfg
        self._obs_space = observation_space
        self._agents = []

        ckpt_paths = [self.config.EVAL_CKPT_A, self.config.EVAL_CKPT_B]

        ckpt_idxs = [
            self.config.EVAL_IDX_A,
            self.config.EVAL_IDX_B,
        ]

        for i, ckpt_path, ckpt_idx, policy_cfg in zip(
            range(2), ckpt_paths, ckpt_idxs, self._policy_cfgs
        ):
            if ckpt_path.endswith(".pth"):
                ckpt_data = torch.load(ckpt_path, map_location="cpu")
                ckpt = ckpt_data["state_dict"]
                if ckpt_idx >= len(ckpt):
                    raise ValueError(
                        f"Requested checkpoint for agent {i} in {ckpt_idx}, but only have {len(ckpt)} checkpoints"
                    )
                ckpt = ckpt[ckpt_idx]
                load_cfg = ckpt_data["config"].RL.POLICIES.POLICY_0
                policy_cfg.merge_from_other_cfg(load_cfg)

                # Do not use any latent preferences.
                policy_cfg.defrost()
                policy_cfg.high_level_policy.PREF_DIM = -1
                policy_cfg.high_level_policy.N_AGENTS = -1
                policy_cfg.high_level_policy.use_pref_discrim = False
                policy_cfg.freeze()
            else:
                # This is a scripted agent.
                policy_cfg.merge_from_file(ckpt_path)
                ckpt = None

            agent = self.create_agent(
                policy_cfg, self._ppo_cfg, self._obs_space, i
            )

            if ckpt is not None:
                if not agent.updater.actor_critic.should_load_agent_state:
                    raise ValueError()
                try:
                    agent.updater.load_state_dict(ckpt, strict=False)
                except Exception as e:
                    raise ValueError(
                        f"Could not load model for agent index={i} with model {ckpt_path}"
                    ) from e

            # Make non-updatable so we load the weights here.
            self._agents.append(agent)
            # self.set_policy(agent.updater.actor_critic, i)
            agent.make_non_updatable()

    def _get_policies(self, is_eval, total_num_env_steps):
        return self._agents

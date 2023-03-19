from typing import Dict

import gym.spaces as spaces
import numpy as np
import torch

from bdp.common.baseline_registry import baseline_registry
from bdp.common.obs_transformers import ObservationTransformer


@baseline_registry.register_obs_transformer()
class AddVirtualKeys(ObservationTransformer):
    def __init__(self, virtual_keys):
        super().__init__()
        self._virtual_keys = virtual_keys

    def transform_observation_space(self, observation_space: spaces.Dict, **kwargs):
        for k, obs_dim in self._virtual_keys.items():
            observation_space[k] = spaces.Box(
                shape=(obs_dim,),
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                dtype=np.float32,
            )
        return observation_space

    @torch.no_grad()
    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        first_obs = next(iter(observations.values()))
        device = first_obs.device
        batch_dim = first_obs.shape[0]
        for k, obs_dim in self._virtual_keys.items():
            observations[k] = torch.zeros((batch_dim, obs_dim), device=device)
        return observations

    @classmethod
    def from_config(cls, config):
        return cls(config.RL.POLICY.OBS_TRANSFORMS.ADD_VIRTUAL_KEYS)

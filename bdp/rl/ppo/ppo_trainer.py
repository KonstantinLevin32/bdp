import contextlib
import os
import pickle
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import rl_utils.common as cutils
import torch
import torch.nn.functional as F
import tqdm
from gym import spaces
from habitat import Config, VectorEnv, logger
from habitat.core.environments import get_env_class
from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat.utils import profiling_wrapper
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

import bdp.rl.hrl.obs_transforms  # noqa: F401.
from bdp.common.base_trainer import BaseRLTrainer
from bdp.common.baseline_registry import baseline_registry
from bdp.common.construct_vector_env import construct_envs
from bdp.common.ma_helpers import *
from bdp.common.obs_transformers import (
    apply_obs_transforms_batch, apply_obs_transforms_obs_space,
    get_active_obs_transforms)
from bdp.common.rollout_storage import RolloutStorage
from bdp.common.tensorboard_utils import (TensorboardWriter,
                                                           get_writer)
from bdp.rl.ddppo.algo import DDPPO
from bdp.rl.ddppo.ddp_utils import (EXIT, add_signal_handlers,
                                                     get_distrib_size,
                                                     init_distrib_slurm,
                                                     is_slurm_batch_job,
                                                     load_resume_state,
                                                     rank0_only, requeue_job,
                                                     save_resume_state)
from bdp.rl.ddppo.policy import \
    PointNavResNetPolicy  # noqa: F401.
from bdp.rl.hrl.hierarchical_policy import \
    HierarchicalPolicy  # noqa: F401.
from bdp.rl.hrl.hrl_logger import hrl_logger
from bdp.rl.hrl.train_hrl import \
    TrainableHierarchicalPolicy  # noqa: F401.
from bdp.rl.multi_agent.agent_samplers import AgentSampler
from bdp.rl.multi_agent.agent_tracker import AgentTracker
from bdp.rl.multi_agent.all_pair_sampler import AllPairSampler
from bdp.rl.multi_agent.all_pref_sampler import AllPrefSampler
from bdp.rl.multi_agent.eval_sampler import EvalAgentSampler
from bdp.rl.multi_agent.pop_play import \
    PopulationPlayAgentSampler
from bdp.rl.multi_agent.pref_eval_sampler import \
    PrefEvalSampler
from bdp.rl.multi_agent.pref_play import PrefPlayAgentSampler
from bdp.rl.ppo import PPO
from bdp.rl.ppo.policy import NetPolicy
from bdp.task.sensors import *
from bdp.utils.common import (ObservationBatchingCache,
                                               batch_obs, generate_video,
                                               get_num_actions,
                                               is_continuous_action_space)


@baseline_registry.register_trainer(name="ddppo")
@baseline_registry.register_trainer(name="ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv

    def __init__(self, config=None):
        super().__init__(config)
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1
        self._obs_batching_cache = ObservationBatchingCache()

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.obs_space, self.obs_transforms
        )

        self._agent_sampler.setup(ppo_cfg, observation_space)

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            workers_ignore_signals=is_slurm_batch_job(),
        )

    def _setup_agent_sampler(self):
        agent_sampler_cls = eval(self.config.RL.AGENT_SAMPLER.TYPE)
        self._agent_sampler = agent_sampler_cls.from_config(
            self.config,
            self.device,
            self.orig_action_space,
            self._is_distributed,
        )

    def _init_train(self):
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config: Config = resume_state["config"]

        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        action_space = self.envs.action_spaces[0]
        self.policy_action_space = action_space
        self.orig_action_space = self.envs.orig_action_spaces[0]

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_agent_sampler()

        self._setup_actor_critic_agent(ppo_cfg)
        if self._is_distributed:
            for updater in self._agent_sampler.updaters:
                updater.init_distributed(find_unused_params=True)  # type: ignore
                logger.info(
                    "agent number of parameters: {}".format(
                        sum(param.numel() for param in updater.parameters())
                    )
                )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1

        self._agent_sampler.setup_storage(ppo_cfg, self.envs.num_envs)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self._agent_sampler.insert_first(batch)

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )
        if "AGENT_TRACKER" in self.config.RL:
            self._agent_tracker = AgentTracker(
                self.config.RL.AGENT_TRACKER,
                self.config.VIDEO_DIR,
                self.config.WRITER_TYPE == "wb",
                self.config.RL.AGENT_SAMPLER,
            )
        else:
            self._agent_tracker = None

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": [
                updater.state_dict() for updater in self._agent_sampler.updaters
            ],
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict[str, Any]) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if not isinstance(k, str) or k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(v).items()
                        if isinstance(subk, str)
                        and k + "." + subk not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)
            elif isinstance(v, list) and isinstance(v[0], list):
                result[k] = np.stack(v, 0)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:
        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _compute_actions_and_step_envs(self, policies, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # Sample actions
        with torch.no_grad():
            profiling_wrapper.range_push("compute actions")
            ac_result = compute_ma_action_from_batch(policies, buffer_index, env_slice)

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        actions = ac_result.take_actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if is_continuous_action_space(self.policy_action_space):
                # Clipping actions to the specified limits
                act = np.clip(
                    act.detach().cpu().numpy(),
                    self.policy_action_space.low,
                    self.policy_action_space.high,
                )
            else:
                act = act.cpu().item()

            self.envs.async_step_at(index_env, act)

        self.env_time += time.time() - t_step_env

        # For logging policy output stats.
        # for i, agent_info in enumerate(ac_result.add_info):
        #     for env_i, env_info in enumerate(agent_info["per_step_log"]):
        #         for k, v in env_info.items():
        #             stat_k = f"agent_{i}.{k}"
        #             if k not in self.running_episode_stats:
        #                 self.running_episode_stats[stat_k] = torch.zeros_like(
        #                     self.running_episode_stats["count"]
        #                 )
        #             if isinstance(v, float) or isinstance(v, int):
        #                 self.running_episode_stats[stat_k][env_i] += v

        for i, policy in enumerate(policies):
            policy.storage.insert_policy(
                next_recurrent_hidden_states=ac_result.rnn_hxs[i],
                actions=ac_result.actions[i],
                action_log_probs=ac_result.action_log_probs[i],
                value_preds=ac_result.values[i],
                buffer_index=buffer_index,
                add_info=ac_result.add_info[i],
            )

    def _collect_environment_result(
        self, policies, buffer_index: int = 0, pairing_id=None
    ):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._agent_tracker is not None:
            self._agent_tracker.log_obs(batch, dones, pairing_id, infos)

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)
        # hrl_logger.debug(
        #     f"Trainer: Got from env obs: {batch['all_predicates'][:, 0]}, reward: {rewards}"
        # )

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        not_done_masks = not_done_masks.to(self.device)

        for policy in policies:
            policy_batch = filter_ma_keys(batch, policy.robot_id)
            policy.storage.insert_obs(
                next_observations=policy_batch,
                rewards=rewards,
                next_masks=not_done_masks,
                buffer_index=buffer_index,
            )

            policy.storage.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self, updaters, policies, lr_schedulers, pairing_id):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()
        all_updater_log = {}

        if len(policies) > 1 and policies[1].is_fake:
            all_updater_log = ma_batched_ppo_update(
                ppo_cfg, updaters, policies, lr_schedulers
            )
        else:
            for agent_i, (
                updater,
                policy,
                lr_scheduler,
                agent_id,
            ) in enumerate(zip(updaters, policies, lr_schedulers, pairing_id)):
                # if not policy.storage.is_ready_for_update():
                #     continue

                if updater is not None:
                    # policy.storage.compute_returns(policy, ppo_cfg)
                    updater.train()

                    latent_ids = None
                    if len(self._agent_sampler._agents) > 1:
                        latent_ids = []
                        for policy in policies:
                            if (
                                hasattr(policy, "pref_latent")
                                and policy.pref_latent is not None
                            ):
                                latent_id = torch.argmax(policy.pref_latent)
                            else:
                                latent_id = -1
                            latent_ids.append(latent_id)
                        latent_ids = torch.tensor(latent_ids, device=self.device)

                    updater_log = updater.update(
                        [policy.storage.wrapped],
                        ppo_cfg=ppo_cfg,
                        pref_latents=latent_ids,
                        agent_sampler=self._agent_sampler,
                    )
                    all_updater_log.update(
                        {f"agent_{agent_i}_{k}": v for k, v in updater_log.items()}
                    )
                    policy.storage.wrapped.after_update()
                    if ppo_cfg.use_linear_lr_decay:
                        lr_scheduler.step()
                else:
                    all_updater_log.update(
                        {
                            f"agent_{agent_i}_{k}": 0.0
                            for k in [
                                "value_loss",
                                "action_loss",
                                "entropy",
                                "aux_loss",
                                "discrim_loss",
                                "discrim_acc",
                                "bonus_reward",
                            ]
                        }
                    )

        self.pth_time += time.time() - t_update_model
        return all_updater_log

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack([self.running_episode_stats[k] for k in stats_ordering], 0)

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {k: stats[i].item() for i, k in enumerate(loss_name_ordering)}

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self,
        writer,
        losses: Dict[str, float],
        prev_time: int = 0,
        pairing_id=None,
    ):
        deltas = {
            k: ((v[-1] - v[0]).sum().item() if len(v) > 1 else v[0].sum().item())
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )
        if self._agent_tracker is not None:
            tracker_log = self._agent_tracker.training_log(
                deltas, pairing_id, self.num_updates_done
            )
            for k, v in tracker_log.items():
                writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"losses/{k}", v, self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)
        writer.add_scalar("metrics/fps", fps, self.num_steps_done)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step >= self.config.RL.PPO.num_steps * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.RL.DDPPO.sync_frac * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """
        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_schedulers = [
            LambdaLR(
                optimizer=updater.optimizer,
                lr_lambda=lambda _: 1 - self.percent_done(),
            )
            for updater in self._agent_sampler.updaters
        ]

        resume_state = load_resume_state(self.config)
        resume_run_id = None
        if resume_state is not None:
            assert len(self._agent_sampler.updaters) == len(resume_state["optim_state"])
            for updater, optim_state, state_dict in zip(
                self._agent_sampler.updaters,
                resume_state["optim_state"],
                resume_state["state_dict"],
            ):
                updater.load_state_dict(state_dict)
                updater.optimizer.load_state_dict(optim_state)

            for lr_scheduler, lr_sched_state in zip(
                lr_schedulers, resume_state["lr_sched_state"]
            ):
                lr_scheduler.load_state_dict(lr_sched_state)

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats["_last_checkpoint_percent"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(requeue_stats["window_episode_stats"])
            resume_run_id = requeue_stats["run_id"]

        ppo_cfg = self.config.RL.PPO

        with (
            get_writer(
                self.config,
                resume_run_id=resume_run_id,
                flush_secs=self.flush_secs,
            )
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            policies: Optional[List[HierarchicalPolicy]] = None
            updaters: Optional[List[PPO]] = None
            pairing_id = None

            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if ppo_cfg.use_linear_clip_decay:
                    for updater in self._agent_sampler.updaters:
                        updater.clip_param = ppo_cfg.clip_param * (
                            1 - self.percent_done()
                        )

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                        run_id=writer.get_run_id(),
                    )

                    save_resume_state(
                        dict(
                            state_dict=[
                                updater.state_dict()
                                for updater in self._agent_sampler.updaters
                            ],
                            optim_state=[
                                updater.optimizer.state_dict()
                                for updater in self._agent_sampler.updaters
                            ],
                            lr_sched_state=[
                                lr_scheduler.state_dict()
                                for lr_scheduler in lr_schedulers
                            ],
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                if policies is None or (
                    self.config.RL.AGENT_SAMPLER.SAMPLE_INTERVAL != -1
                    and (
                        self.num_updates_done
                        % self.config.RL.AGENT_SAMPLER.SAMPLE_INTERVAL
                        == 0
                    )
                ):
                    all_prev_obs = None
                    if policies is not None:
                        all_prev_obs = [
                            policy.storage.wrapped.buffers[0] for policy in policies
                        ]
                        if self._agent_sampler.should_force_cpu:
                            # Put the old checkpoints on the CPU.
                            for i in range(len(policies)):
                                policies[i].to("cpu")
                                policies[i].storage.to("cpu")
                                if updaters[i] is not None:
                                    updaters[i] = updaters[i].to("cpu")
                    (
                        policies,
                        updaters,
                        all_is_new,
                        pairing_id,
                    ) = self._agent_sampler.get_policies(False, self.num_steps_done)
                    if self._agent_sampler.should_force_cpu:
                        # Bring the new checkpoints to the GPU.
                        for i in range(len(policies)):
                            policies[i].to(self.device)
                            policies[i].storage.to(self.device)
                            if updaters[i] is not None:
                                updaters[i] = updaters[i].to(self.device)

                    if all_prev_obs is not None:
                        for prev_obs, policy, is_new in zip(
                            all_prev_obs, policies, all_is_new
                        ):
                            if not is_new:
                                continue
                            policy.storage.wrapped.buffers[0] = prev_obs

                for updater in updaters:
                    if updater is not None:
                        updater.eval()

                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(policies, buffer_index)

                for step in range(ppo_cfg.num_steps):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == ppo_cfg.num_steps
                    )
                    # hrl_logger.debug("")
                    # hrl_logger.debug(f"Step {step}")

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            policies, buffer_index, pairing_id
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                profiling_wrapper.range_push("_collect_rollout_step")

                            self._compute_actions_and_step_envs(policies, buffer_index)

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                updater_log = self._update_agent(
                    updaters, policies, lr_schedulers, pairing_id
                )

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    updater_log,
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time, pairing_id)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.EVAL.SHOULD_LOAD_CKPT:
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        else:
            ckpt_dict = {}

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        # Add additional eval metrics
        is_multi_agent = len(config.TASK_CONFIG.SIMULATOR.AGENTS) > 1
        if "AGENT_BLAME" not in config.TASK_CONFIG.TASK.MEASUREMENTS and is_multi_agent:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("AGENT_BLAME")
        if "COMPOSITE_STAGE_GOALS" not in config.TASK_CONFIG.TASK.MEASUREMENTS:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COMPOSITE_STAGE_GOALS")
        if "GLOBAL_PREDICATE_SENSOR" not in config.TASK_CONFIG.TASK.SENSORS:
            config.TASK_CONFIG.TASK.SENSORS.append("GLOBAL_PREDICATE_SENSOR")
        if config.TASK_CONFIG.SIMULATOR.HEAD_DEPTH_SENSOR.WIDTH < 10:
            # This was a state only run.
            config.TASK_CONFIG.SIMULATOR.HEAD_DEPTH_SENSOR.WIDTH = 256
            config.TASK_CONFIG.SIMULATOR.HEAD_DEPTH_SENSOR.HEIGHT = 256
        config.TASK_CONFIG.SIMULATOR.DEBUG_RENDER = True
        config.RL.PPO.hidden_size = 128
        config.freeze()

        if len(self.config.VIDEO_OPTION) > 0 and self.config.VIDEO_RENDER_TOP_DOWN:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            # config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

        if "AGENT_TRACKER" in self.config.RL:
            self._agent_tracker = AgentTracker(
                self.config.RL.AGENT_TRACKER,
                self.config.VIDEO_DIR,
                False,
                self.config.RL.AGENT_SAMPLER,
            )
        else:
            self._agent_tracker = None

        action_space = self.envs.action_spaces[0]
        self.orig_action_space = self.envs.orig_action_spaces[0]
        self.policy_action_space = action_space

        self._setup_agent_sampler()
        self._setup_actor_critic_agent(ppo_cfg)

        for i, updater in enumerate(self._agent_sampler.updaters):
            if updater.actor_critic.should_load_agent_state:
                updater.load_state_dict(ckpt_dict["state_dict"][i])
            self._agent_sampler.set_policy(updater.actor_critic, i)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        policies, _, _, pairing_id = self._agent_sampler.get_policies(True, 0)
        num_hxs = sum(policy.num_recurrent_layers for policy in policies)

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            num_hxs,
            ppo_cfg.hidden_size,
            device=self.device,
        )

        if "MA_VIS" in self.config.RL and self.config.RL.MA_VIS.LOG_INTERVAL != -1:
            ma_vis = MaVisHelper(
                self.obs_space,
                self.orig_action_space,
                self.config.VIDEO_DIR,
                policies,
                self.config.RL.AGENT_SAMPLER,
                self.config.RL.MA_VIS,
                len(policies),
            )
        else:
            ma_vis = None

        all_prev_actions = []
        for policy in policies:
            filtered_ac = ActionSpace(
                filter_ma_keys(self.orig_action_space, policy.robot_id)
            )

            all_prev_actions.append(
                torch.zeros(
                    self.config.NUM_ENVIRONMENTS,
                    get_num_actions(filtered_ac),
                    device=self.device,
                    dtype=torch.float,
                )
            )

        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        all_infos = [[] for _ in range(self.config.NUM_ENVIRONMENTS)]
        os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT

        resample_interval = self._agent_sampler.num_evals
        number_of_eval_episodes *= resample_interval

        replay_factor = 1
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                # logger.warn(
                #     f"Config specified {number_of_eval_episodes} eval episodes"
                #     ", dataset only has {total_num_eps}."
                # )
                # logger.warn(f"Evaluating with {total_num_eps} instead.")
                # number_of_eval_episodes = total_num_eps
                replay_factor = number_of_eval_episodes // (
                    total_num_eps // self.config.NUM_ENVIRONMENTS
                )

        num_ep_plays = defaultdict(lambda: replay_factor)

        pbar = tqdm.tqdm(total=number_of_eval_episodes)

        for policy in policies:
            policy.eval()

        while len(stats_episodes) < number_of_eval_episodes and self.envs.num_envs > 0:
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                action_result = compute_ma_action_eval(
                    batch,
                    not_done_masks,
                    all_prev_actions,
                    test_recurrent_hidden_states,
                    policies,
                )
                for agent_i in range(action_result.actions.size(0)):
                    all_prev_actions[agent_i].copy_(action_result.actions[agent_i])
                test_recurrent_hidden_states = action_result.rnn_hxs
                actions = action_result.take_actions

            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.

            if is_continuous_action_space(self.policy_action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.detach().cpu().numpy(),
                        self.policy_action_space.low,
                        self.policy_action_space.high,
                    )
                    for a in actions
                ]
            else:
                step_data = [a.item() for a in actions.cpu()]

            outputs = self.envs.step(step_data)

            if ma_vis is not None:
                ma_vis.log_step(batch, action_result)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore
            if self._agent_tracker is not None:
                self._agent_tracker.log_obs(batch, dones, pairing_id, infos)

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                ep_idx = int(next_episodes[i].episode_id) + (
                    total_num_eps * num_ep_plays[next_episodes[i].episode_id]
                )
                if (
                    next_episodes[i].scene_id,
                    ep_idx,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                if len(self.config.VIDEO_OPTION) > 0:
                    # TODO move normalization / channel changing out of the policy and undo it here
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    if not not_done_masks[i].item():
                        # The last frame corresponds to the first frame of the next episode
                        # but the info is correct. So we use a black frame
                        frame = observations_to_image(
                            {k: v[i] * 0.0 for k, v in batch.items()}, infos[i]
                        )

                    render_metrics = {
                        k: v
                        for k, v in infos[i].items()
                        if k not in [GfxReplayMeasure.cls_uuid]
                    }
                    for agent_i, add_info in enumerate(action_result.add_info):
                        to_log = add_info["log_info"][i]
                        render_metrics.update(
                            {f"A{agent_i}_{k}": str(v) for k, v in to_log.items()}
                        )
                    rgb_frames[i].append(frame)
                    all_infos[i].append(render_metrics)

                    if self.config.VIDEO_RENDER_ALL_INFO:
                        frame = overlay_frame(frame, render_metrics)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}

                    agent_policy_info = {
                        f"{agent_idx}_{k}": v
                        for agent_idx, ac_info in enumerate(action_result.add_info)
                        for k, v in ac_info["log_info"][i].items()
                    }

                    episode_stats.update(
                        self._extract_scalars_from_info(
                            {**infos[i], **agent_policy_info}
                        )
                    )
                    current_episode_reward[i] = 0
                    cur_ep_id = current_episodes[i].episode_id

                    if self._agent_tracker is not None:
                        self._agent_tracker.training_log(episode_stats, pairing_id, 0)

                    # use scene_id + episode_id as unique id for storing stats
                    ep_idx = int(cur_ep_id) + (total_num_eps * num_ep_plays[cur_ep_id])
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            ep_idx,
                        )
                    ] = episode_stats
                    pairing_str = "_".join([f"A{aid}" for aid in pairing_id])
                    if ma_vis is not None:
                        ma_vis.on_episode_done(
                            i,
                            f"{pairing_str}_{cur_ep_id}_{(total_num_eps * num_ep_plays[cur_ep_id])}",
                            infos[i],
                        )

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=f"{pairing_str}_{cur_ep_id}_{num_ep_plays[cur_ep_id]}",
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.VIDEO_FPS,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.EVAL_KEYS_TO_INCLUDE_IN_NAME,
                        )
                        rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        task_config = self.config.TASK_CONFIG.TASK
                        write_gfx_replay(
                            gfx_str,
                            task_config,
                            cur_ep_id,
                        )
                        filepath = osp.join(
                            task_config.GFX_REPLAY_DIR,
                            f"episode{cur_ep_id}_info.pickle",
                        )
                        with open(filepath, "wb") as f:
                            pickle.dump([current_episodes[i].scene_id, all_infos[i]], f)
                        print("Wrote to ", filepath)
                    all_infos[i] = []

                    num_ep_plays[current_episodes[i].episode_id] -= 1
                    num_ep_plays[current_episodes[i].episode_id] = max(
                        num_ep_plays[current_episodes[i].episode_id], 1
                    )

                    if len(stats_episodes) % self.config.TEST_EPISODE_COUNT == 0:
                        # Resample a new agent.
                        (
                            policies,
                            _,
                            _,
                            pairing_id,
                        ) = self._agent_sampler.get_policies(True, 0)
                        for policy in policies:
                            policy.eval()
                        print(f"Resample pair {pairing_id} at {len(stats_episodes)}")

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                all_prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                all_prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values()) / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )
        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        if self._agent_tracker is not None and is_multi_agent:
            self._agent_tracker.save_results()
            add_stats = self._agent_tracker.display(step_id, True)
            for k, v in add_stats.items():
                writer.add_scalar(f"eval_metrics/{k}", float(v), step_id)
                logger.info(f"Average episode {k}: {v:.4f}")

        self.envs.close()

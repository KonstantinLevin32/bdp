from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from habitat.utils import profiling_wrapper
from torch import Tensor
from torch import nn as nn
from torch import optim as optim

from bdp.common.ma_helpers import recur_stack
from bdp.common.rollout_storage import RolloutStorage
from bdp.rl.hrl.hrl_logger import hrl_logger
from bdp.rl.ppo.policy import NetPolicy

EPS_PPO = 1e-5


class DiscrimBuffer:
    def __init__(self, discrim_net, device):
        self._discrim_net = discrim_net
        self._device = device
        self.all_obs = torch.zeros(
            discrim_net._config.buffer_size, discrim_net.input_size
        )
        self.all_target = torch.zeros(
            discrim_net._config.buffer_size, discrim_net.pref_dim
        )

        self._cur_i = 0
        self._is_full = False

    def get_batches(self):
        batch_size = self._discrim_net._config.batch_size
        if batch_size == -1:
            batches = [torch.randperm(len(self))]
        else:
            batches = torch.randperm(len(self)).chunk(max(len(self) // batch_size, 1))
        for batch_idx in batches:
            yield self.all_obs[batch_idx].to(self._device), self.all_target[
                batch_idx
            ].to(self._device)

    def __len__(self):
        if self._is_full:
            return self.all_obs.shape[0]
        else:
            return self._cur_i

    def insert(self, rollouts, pref_latent):
        obs = self._discrim_net.extract_obs(rollouts.buffers["observations"])

        n_steps, n_envs, obs_dim = obs.shape
        for cur_step_idx in rollouts._cur_step_idxs:
            obs = obs[:cur_step_idx].view(-1, obs_dim)
            target = pref_latent
            target = target.view(1, -1).repeat(obs.shape[0], 1)

            write_end = min(self._cur_i + obs.shape[0], self.all_obs.shape[0])
            write_len = write_end - self._cur_i
            self.all_obs[self._cur_i : write_end] = obs[:write_len]

            self.all_target[self._cur_i : write_end] = target[:write_len]

            self._cur_i += obs.shape[0]
            if self._cur_i >= self.all_obs.shape[0]:
                self._cur_i = 0


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic: NetPolicy,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: Optional[float] = None,
        eps: Optional[float] = None,
        max_grad_norm: Optional[float] = None,
        use_clipped_value_loss: bool = True,
        use_normalized_advantage: bool = True,
    ) -> None:
        super().__init__()

        self.actor_critic = actor_critic

        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.optimizer = optim.Adam(
            list(filter(lambda p: p.requires_grad, actor_critic.parameters())),
            lr=lr,
            eps=eps,
        )
        self.device = next(actor_critic.parameters()).device
        self.use_normalized_advantage = use_normalized_advantage

        hl_policy = self.actor_critic._high_level_policy
        if hl_policy.pref_discrim_net is None:
            self.discrim_opt = None
            self.discrim_buffer = None
        else:
            self.discrim_opt = optim.Adam(
                hl_policy.pref_discrim_net.parameters(),
                lr=hl_policy.pref_discrim_net._config.lr,
                eps=eps,
            )
            self.discrim_buffer = DiscrimBuffer(hl_policy.pref_discrim_net, self.device)
        self.aux_loss_coef = hl_policy._config.aux_pred.weight
        self.discrim_reward_coef = hl_policy._config.pref_discrim.reward_weight
        self.aux_reward_coef = hl_policy._config.get("aux_reward_weight", 1.0)
        self.rolling_ep_rewards = defaultdict(lambda: 0.0)

    def forward(self, *x):
        raise NotImplementedError

    def get_advantages(self, all_rollouts: List[RolloutStorage]) -> Tensor:
        all_advantages = []
        for rollouts in all_rollouts:
            advantages = (
                rollouts.buffers["returns"][:-1]  # type: ignore
                - rollouts.buffers["value_preds"][:-1]
            )
            if not self.use_normalized_advantage:
                all_advantages.append(advantages)
            else:
                raise NotImplementedError()
        return torch.cat(all_advantages, 0)
        # return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def infer_reward(self, rollout, pref_latent):
        hl_policy = self.actor_critic._high_level_policy

        logits = hl_policy.pref_discrim_net.pred(rollout.buffers["observations"])
        score = F.log_softmax(logits, dim=-1)
        latent_idx = torch.argmax(pref_latent)
        log_prob = score[:, :, latent_idx, None]
        return self.discrim_reward_coef * log_prob

    def infer_div_reward(self, rollout, agent_sampler):
        policies = [agent.policy._high_level_policy for agent in agent_sampler._agents]
        cur_policy = self.actor_critic._high_level_policy

        masks = rollout.buffers["masks"]
        n_steps, n_envs, _ = masks.shape
        inf_rewards = torch.zeros((n_steps, n_envs, 1), device=masks.device)

        for inds in torch.randperm(n_envs).chunk(self.num_mini_batch):
            batch = rollout.buffers[0 : rollout.numsteps, inds]
            batch["recurrent_hidden_states"] = batch["recurrent_hidden_states"][0:1]
            n_steps = batch["masks"].shape[0]

            batch = batch.map(lambda v: v.flatten(0, 1))

            batch["loss_mask"] = (
                torch.arange(n_steps, device=masks.device)
                .view(-1, 1, 1)
                .repeat(1, len(inds), 1)
            )
            for i, env_i in enumerate(inds):
                # The -1 is to throw out the last transition.
                batch["loss_mask"][:, i] = (
                    batch["loss_mask"][:, i] < rollout._cur_step_idxs[env_i] - 1
                )

            all_dists = [
                policy.get_pi(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["masks"],
                )
                for policy in policies
            ]

            cur_dist = cur_policy.get_pi(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
            )
            # Get the average policy
            avg_dist = torch.mean(
                torch.stack([dist.probs for dist in all_dists], -1), -1
            )
            div = F.kl_div(cur_dist.probs, avg_dist, reduction="none").mean(-1)
            div = div.view(n_steps, -1, 1) * batch["loss_mask"]
            inf_rewards[:n_steps, inds, :] = div
        return self.discrim_reward_coef * inf_rewards

    def update_reward(self, log_stats):
        hl_policy = self.actor_critic._high_level_policy
        if hl_policy.pref_discrim_net is None:
            log_stats["discrim_loss"].append(0.0)
            log_stats["discrim_acc"].append(1.0)
            return
        pref_cfg = hl_policy.pref_discrim_net._config

        for epoch_i in range(pref_cfg.num_epochs):
            for obs, target in self.discrim_buffer.get_batches():
                logits = hl_policy.pref_discrim_net(obs)
                self.discrim_opt.zero_grad()
                loss = F.cross_entropy(logits, target, reduction="none")
                acc = torch.argmax(logits, -1) == torch.argmax(target, -1)
                n_steps = target.size(0)

                wait_ratio = pref_cfg.get("wait_ratio", None)
                if wait_ratio is not None:
                    is_valid = wait_ratio >= obs[:, 0]
                    loss *= is_valid
                    acc *= is_valid
                    n_steps -= (~is_valid).sum()

                if n_steps == 0:
                    log_stats["discrim_loss"].append(0.0)
                    log_stats["discrim_acc"].append(1.0)
                else:
                    loss = loss.sum() / n_steps
                    loss.backward()
                    self.discrim_opt.step()

                    acc = acc.sum() / n_steps

                    log_stats["discrim_loss"].append(loss.item())
                    log_stats["discrim_acc"].append(acc.item())

    def infer_aux_reward(self, my_pref_latent, other_pref_latent, rollout):
        masks = rollout.buffers["masks"]
        n_steps, n_envs, _ = masks.shape
        inf_rewards = torch.zeros((n_steps, n_envs, 1), device=masks.device)

        for inds in torch.randperm(n_envs).chunk(self.num_mini_batch):
            batch = rollout.buffers[0 : rollout.numsteps, inds]
            batch["recurrent_hidden_states"] = batch["recurrent_hidden_states"][0:1]
            n_steps = batch["masks"].shape[0]

            batch = batch.map(lambda v: v.flatten(0, 1))
            batch_size = batch["masks"].shape[0]

            batch["loss_mask"] = (
                torch.arange(n_steps, device=masks.device)
                .view(-1, 1, 1)
                .repeat(1, len(inds), 1)
            )
            for i, env_i in enumerate(inds):
                # The -1 is to throw out the last transition.
                batch["loss_mask"][:, i] = (
                    batch["loss_mask"][:, i] < rollout._cur_step_idxs[env_i] - 1
                )

            if not (my_pref_latent == -1).all():
                batch["observations"]["pref_latent"] = my_pref_latent.view(
                    1, -1
                ).repeat(batch_size, 1)

            (
                _,
                _,
                _,
                _,
                aux_pred,
            ) = self._evaluate_actions(
                batch["observations"],
                batch["recurrent_hidden_states"],
                batch["prev_actions"],
                batch["masks"],
                batch["actions"],
            )
            score = F.log_softmax(aux_pred, dim=-1)
            latent_idx = torch.argmax(other_pref_latent)
            log_prob = score[:, latent_idx]

            inf_rewards[:n_steps, inds, :] = log_prob.view(n_steps, -1, 1)
        return self.aux_reward_coef * inf_rewards

    def compute_aux_loss(self, aux_pred, batch, pref_latents):
        if "pref_latent" in batch["observations"]:
            target = torch.argmax(batch["observations"]["pref_latent"], dim=-1).long()
        else:
            target = torch.full(
                (aux_pred.shape[0],),
                pref_latents[-1].item(),
                device=self.device,
                dtype=torch.long,
            )
        return F.cross_entropy(aux_pred, target)

    def log_finished_rewards(self, inferred_rewards, masks, cur_steps):
        num_steps, num_envs = masks.shape[:2]
        done_episodes_rewards = []
        for env_i in range(num_envs):
            for step_i in range(cur_steps[env_i] - 1):
                self.rolling_ep_rewards[env_i] += inferred_rewards[step_i, env_i].item()
                if masks[step_i + 1, env_i].item() == 0.0:
                    done_episodes_rewards.append(self.rolling_ep_rewards[env_i])
                    self.rolling_ep_rewards[env_i] = 0.0
        return done_episodes_rewards

    def _log_reward(self, rollout, inferred_rewards, log_stats):
        rollout.buffers["rewards"] += inferred_rewards
        done_rewards = self.log_finished_rewards(
            inferred_rewards,
            rollout.buffers["masks"],
            rollout._cur_step_idxs,
        )
        if len(done_rewards) > 0:
            log_stats["bonus_reward"].extend(done_rewards)
        else:
            log_stats["bonus_reward"].append(0.0)

    def update(
        self,
        all_rollouts: List[RolloutStorage],
        pref_latents=None,
        ppo_cfg=None,
        agent_sampler=None,
    ) -> Tuple[float, float, float]:
        hl_policy = self.actor_critic._high_level_policy
        log_stats = defaultdict(list)

        div_reward = hl_policy._config.get("div_reward", False)
        aux_reward = hl_policy._config.get("use_aux_reward", False)

        with torch.no_grad():
            if hl_policy.pref_discrim_net is not None:
                for pref_latent, rollout in zip(pref_latents, all_rollouts):
                    inferred_rewards = self.infer_reward(rollout, pref_latent)
                    self._log_reward(rollout, inferred_rewards, log_stats)
            elif aux_reward:
                for i, rollout in enumerate(all_rollouts):
                    my_pref_latent = pref_latents[i]
                    other_pref_latent = pref_latents[(i + 1) % 2]
                    inferred_rewards = self.infer_aux_reward(
                        my_pref_latent, other_pref_latent, rollout
                    )
                    self._log_reward(rollout, inferred_rewards, log_stats)
            elif div_reward:
                for rollout in all_rollouts:
                    inferred_rewards = self.infer_div_reward(rollout, agent_sampler)
                    self._log_reward(rollout, inferred_rewards, log_stats)

        if ppo_cfg is not None:
            with torch.no_grad():
                for rollout in all_rollouts:
                    rollout.compute_returns(
                        ppo_cfg.use_gae,
                        ppo_cfg.gamma,
                        ppo_cfg.tau,
                    )
        advantages = self.get_advantages(all_rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        if self.discrim_buffer is not None:
            for rollouts, pref_latent in zip(all_rollouts, pref_latents):
                self.discrim_buffer.insert(rollouts, pref_latent)
        self.update_reward(log_stats)

        for _e in range(self.ppo_epoch):
            profiling_wrapper.range_push("PPO.update epoch")

            data_gens = [
                rollouts.recurrent_generator(advantages, self.num_mini_batch)
                for rollouts in all_rollouts
            ]

            for batches in zip(*data_gens):
                batch_size = batches[0]["masks"].shape[0]
                if len(batches) == 2:
                    for i in range(2):
                        batches[i]["observations"]["pref_latent"] = (
                            pref_latents[i].view(1, -1).repeat(batch_size, 1)
                        )
                    batch = dict(recur_stack(batches[0], batches[1]))
                else:
                    batch = batches[0]
                # need to add eps in case there are no samples.
                n_samples = max(batch["loss_mask"].sum(), 1)

                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    aux_pred,
                ) = self._evaluate_actions(
                    batch["observations"],
                    batch["recurrent_hidden_states"],
                    batch["prev_actions"],
                    batch["masks"],
                    batch["actions"],
                )

                if aux_pred is not None:
                    aux_loss = self.compute_aux_loss(aux_pred, batch, pref_latents)
                else:
                    aux_loss = torch.tensor(0.0)

                ratio = torch.exp(action_log_probs - batch["action_log_probs"])
                surr1 = ratio * batch["advantages"]
                surr2 = (
                    torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                    * batch["advantages"]
                )

                action_loss = torch.min(surr1, surr2)
                action_loss *= batch["loss_mask"]
                action_loss = -(action_loss.sum() / n_samples)

                if self.use_clipped_value_loss:
                    value_pred_clipped = batch["value_preds"] + (
                        values - batch["value_preds"]
                    ).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - batch["returns"]).pow(2)
                    value_losses_clipped = (value_pred_clipped - batch["returns"]).pow(
                        2
                    )
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped)
                else:
                    value_loss = 0.5 * (batch["returns"] - values).pow(2)

                value_loss *= batch["loss_mask"]
                dist_entropy *= batch["loss_mask"].view(-1)

                value_loss = value_loss.sum() / n_samples
                dist_entropy = dist_entropy.sum() / n_samples

                self.optimizer.zero_grad()
                total_loss = (
                    value_loss * self.value_loss_coef
                    + action_loss
                    - dist_entropy * self.entropy_coef
                    + aux_loss * self.aux_loss_coef
                )

                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)

                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += value_loss.item()
                action_loss_epoch += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()
                log_stats["aux_loss"].append(aux_loss.item())

            profiling_wrapper.range_pop()  # PPO.update epoch

        num_updates = self.ppo_epoch * self.num_mini_batch

        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        log_stats = {k: np.mean(v) for k, v in log_stats.items()}

        return {
            "value_loss": value_loss_epoch,
            "action_loss": action_loss_epoch,
            "entropy": dist_entropy_epoch,
            **log_stats,
        }

    def _evaluate_actions(
        self, observations, rnn_hidden_states, prev_actions, masks, action
    ):
        r"""Internal method that calls Policy.evaluate_actions.  This is used instead of calling
        that directly so that that call can be overrided with inheritance
        """
        return self.actor_critic.evaluate_actions(
            observations, rnn_hidden_states, prev_actions, masks, action
        )

    def before_backward(self, loss: Tensor) -> None:
        pass

    def after_backward(self, loss: Tensor) -> None:
        pass

    def before_step(self) -> None:
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

    def after_step(self) -> None:
        pass

import os.path as osp
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import gym.spaces as spaces
import matplotlib.pyplot as plt
import numpy as np
import rl_utils.common as cutils
import rl_utils.plotting.utils as putils
import seaborn as sns
import torch
from habitat.core.spaces import ActionSpace
from habitat.tasks.rearrange.multi_task.composite_sensors import \
    GlobalPredicatesSensor
from habitat.tasks.rearrange.multi_task.rearrange_pddl import parse_func

from bdp.common.baseline_registry import baseline_registry
from bdp.common.tensor_dict import TensorDict
from bdp.rl.hrl.hrl_logger import hrl_logger
from bdp.utils.common import get_num_actions

try:
    import wandb
except:
    pass

FONT_SIZE = 6


@dataclass(frozen=True)
class AgentActions:
    """
    P - number of policies
    :param take_actions: Shape (N, A*P)
    :param actions: Shape (P, N, A)
    :param add_info: List of size P.
    """

    rnn_hxs: torch.Tensor
    actions: torch.Tensor
    take_actions: torch.Tensor
    action_log_probs: Optional[torch.Tensor]
    values: Optional[torch.Tensor]
    add_info: Optional[List[Dict[str, Any]]]


def filter_ma_keys(d, robot_id: str):
    """
    :param robot_id: Something like `AGENT_0`
    """
    new_d = {}
    for k, v in d.items():
        if k.startswith(robot_id):
            new_d[k[len(robot_id) + 1 :]] = v
        elif not k.startswith("AGENT_"):
            new_d[k] = v
    return new_d


def create_ma_policy(
    policy_cfg,
    cfg,
    device,
    obs_space: spaces.Dict,
    action_space: ActionSpace,
    filter=True,
):
    policy = baseline_registry.get_policy(policy_cfg.name)

    if filter:
        filtered_ac = filter_ma_keys(action_space.spaces, policy_cfg.robot_id)
        filtered_obs = filter_ma_keys(obs_space.spaces, policy_cfg.robot_id)
    else:
        filtered_ac = action_space.spaces
        filtered_obs = obs_space.spaces

    action_space = ActionSpace(filtered_ac)
    observation_space = spaces.Dict(spaces=filtered_obs)

    actor_critic = policy.from_config(cfg, policy_cfg, observation_space, action_space)
    actor_critic.to(device)
    return actor_critic


def recur_stack(d0, d1):
    for k in d0.keys():
        v0 = d0[k]
        v1 = d1[k]

        if isinstance(v0, dict):
            yield k, dict(recur_stack(v0, v1))
        else:
            if v0.shape[1:] != v1.shape[1:]:
                raise ValueError(f"Problem with {k} {v0.shape}, {v1.shape}")
            yield k, torch.cat([v0, v1], dim=0)


def _get_policy_pref_latent(policy, n_envs, sel_agent=None):
    if sel_agent is not None and len(policy.pref_latent.shape) > 1:
        return policy.pref_latent[sel_agent].view(1, -1).repeat(n_envs, 1)
    else:
        return policy.pref_latent.view(1, -1).repeat(n_envs, 1)


def ma_batched_actions_from_batch(policies, buffer_index, env_slice):
    batch0 = policies[0].storage.get_cur_batch(buffer_index, env_slice)
    batch1 = policies[1].storage.get_cur_batch(buffer_index, env_slice)
    comb_batch = dict(recur_stack(batch0, batch1))
    n_envs = batch0["masks"].shape[0]

    pref_latent = torch.cat(
        [_get_policy_pref_latent(policy, n_envs) for policy in policies],
        0,
    )
    comb_batch["observations"]["pref_latent"] = pref_latent
    policy = policies[0]

    (
        values,
        actions,
        action_log_probs,
        rnn_hxs,
        add_info,
    ) = policy.act(
        comb_batch["observations"],
        comb_batch["recurrent_hidden_states"],
        comb_batch["prev_actions"],
        comb_batch["masks"],
        deterministic=False,
    )

    def split_batch(x):
        return torch.stack([x[:n_envs], x[n_envs:]], 0)

    split_add_info = []
    split_add_info.append({k: v[:n_envs] for k, v in add_info.items()})
    split_add_info.append({k: v[n_envs:] for k, v in add_info.items()})

    take_actions = torch.cat([actions[:n_envs], actions[n_envs:]], 1)
    return AgentActions(
        rnn_hxs=split_batch(rnn_hxs),
        actions=split_batch(actions),
        take_actions=take_actions,
        action_log_probs=split_batch(action_log_probs),
        values=split_batch(values),
        add_info=split_add_info,
    )


def ma_batched_ppo_update(ppo_cfg, updaters, policies, lr_schedulers):
    updater = updaters[0]
    policy = policies[0]
    lr_scheduler = lr_schedulers[0]

    rollouts = []
    for policy in policies:
        storage = policy.storage
        # storage.compute_returns(policy, ppo_cfg)

        rollouts.append(storage.wrapped)

    pref_latent = torch.stack([policy.pref_latent for policy in policies], 0)
    updater.train()

    updater_log = updater.update(rollouts, pref_latents=pref_latent, ppo_cfg=ppo_cfg)
    updater_log.update({f"agent_0_{k}": v for k, v in updater_log.items()})
    for policy in policies:
        policy.storage.wrapped.after_update()

    if ppo_cfg.use_linear_lr_decay:
        lr_scheduler.step()
    return updater_log


def compute_ma_action_from_batch(policies, buffer_index, env_slice):
    next_rnn_hxs = []
    all_actions = []
    all_action_log_probs = []
    all_values = []
    all_add_info = []
    if len(policies) > 1 and policies[1].is_fake:
        return ma_batched_actions_from_batch(policies, buffer_index, env_slice)
    for i, policy in enumerate(policies):
        step_batch = policy.storage.get_cur_batch(buffer_index, env_slice)
        n_envs = step_batch["masks"].shape[0]

        if hasattr(policy, "pref_latent") and policy.pref_latent is not None:
            step_batch["observations"]["pref_latent"] = _get_policy_pref_latent(
                policy, n_envs
            )

        (
            values,
            actions,
            action_log_probs,
            agent_rnn_hxs,
            add_info,
        ) = policy.act(
            step_batch["observations"],
            step_batch["recurrent_hidden_states"],
            step_batch["prev_actions"],
            step_batch["masks"],
            deterministic=False,
        )

        all_add_info.append(add_info)
        next_rnn_hxs.append(agent_rnn_hxs)
        all_actions.append(actions)
        all_values.append(values)
        all_action_log_probs.append(action_log_probs)
    return AgentActions(
        rnn_hxs=torch.stack(next_rnn_hxs, dim=0),
        actions=torch.stack(all_actions, dim=0),
        take_actions=torch.cat(all_actions, 1),
        action_log_probs=torch.stack(all_action_log_probs, dim=0),
        values=torch.stack(all_values, dim=0),
        add_info=all_add_info,
    )


def compute_ma_action_eval(
    batch,
    not_done_masks,
    all_prev_actions,
    test_recurrent_hidden_states,
    policies,
):
    next_rnn_hxs = []
    all_actions = []
    all_add_info = []
    prev_num_hxs = 0
    for i, policy in enumerate(policies):
        agent_batch = filter_ma_keys(batch, policy.robot_id)
        num_hxs = policy.num_recurrent_layers
        if hasattr(policy, "pref_latent") and policy.pref_latent is not None:
            # Preference conditioned policy.
            n_envs = not_done_masks.shape[0]
            agent_batch["pref_latent"] = _get_policy_pref_latent(
                policy, n_envs, sel_agent=i
            )

        agent_rnn_hxs = test_recurrent_hidden_states[
            :, prev_num_hxs : prev_num_hxs + num_hxs
        ]
        prev_num_hxs += num_hxs

        (_, actions, _, agent_rnn_hxs, add_info) = policy.act(
            agent_batch,
            agent_rnn_hxs,
            all_prev_actions[i],
            not_done_masks,
            deterministic=False,
        )
        all_add_info.append(add_info)
        next_rnn_hxs.append(agent_rnn_hxs)
        all_actions.append(actions)
    return AgentActions(
        rnn_hxs=torch.cat(next_rnn_hxs, dim=1),
        actions=torch.stack(all_actions, 0),
        take_actions=torch.cat(all_actions, dim=1),
        action_log_probs=None,
        values=None,
        add_info=all_add_info,
    )


class MaVisHelper:
    def _get_ac_key_start_end(self, search_k):
        cur_i = 0
        for k, v in self._ac_space.items():
            if k != search_k:
                cur_i += get_num_actions(v)
            else:
                return slice(cur_i, cur_i + get_num_actions(v))
        raise ValueError()

    def __init__(
        self,
        obs_space,
        ac_space,
        save_dir,
        policies,
        sampler_cfg,
        config,
        num_agents,
    ):
        self._obs_space = obs_space
        self._ac_space = ac_space
        self._num_agents = num_agents

        self._agent_names = [
            f"{sampler_cfg.AGENT_A_NAME}{sampler_cfg.EVAL_IDX_A}_A0",
            f"{sampler_cfg.AGENT_B_NAME}{sampler_cfg.EVAL_IDX_B}_A1",
        ][: self._num_agents]

        self._action_names = []
        for agent_i in range(self._num_agents):
            self._action_names.append(
                [
                    x.compact_str
                    for x in policies[agent_i]._high_level_policy._all_actions
                ]
            )

        poss_preds = policies[
            0
        ]._high_level_policy._pddl_problem.get_possible_predicates()

        pal = sns.color_palette()
        action_names = policies[0]._high_level_policy._pddl_problem.actions.keys()
        self._name_to_col = {name: pal[i] for i, name in enumerate(action_names)}
        self._name_to_col[""] = pal[len(self._name_to_col)]

        self._pred_names = [x.compact_str for x in poss_preds]

        self._all_obs = defaultdict(list)
        self._all_ac = defaultdict(list)
        self._save_dir = save_dir
        self._config = config
        self._num_saved = 0

    def log_step(self, obs, action_result):
        add_info = action_result.add_info

        n_envs = action_result.take_actions.size(0)
        n_agents = len(add_info)
        for env_i in range(n_envs):
            # self._all_obs[env_i].append(
            #     obs[GlobalPredicatesSensor.cls_uuid][env_i]
            # )
            ac = []
            for i in range(n_agents):
                if "actions" in add_info[i]:
                    ac.append(add_info[i]["actions"][env_i].item())
                else:
                    ac.append(-1)
            self._all_ac[env_i].append(ac)

    def _render_timeless_plot(self, ac, ep_idx, title):
        H = ac.shape[0]

        width = 0.35
        prev_t = np.zeros(self._num_agents)
        fig, ax = plt.subplots()
        fig.set_size_inches(18, 3, forward=True)
        ax.set_xlabel("HL Timestep")
        ax.set_title(title)
        for t in range(H):
            n_steps = np.zeros(self._num_agents)
            action_names = []
            action_ids = []
            ac_t = ac[t]
            for agent_i in range(self._num_agents):
                if ac_t[agent_i] == -1:
                    action_names.append("")
                    action_ids.append("")
                else:
                    n_steps[agent_i] = 1
                    action_names.append(self._action_names[agent_i][ac_t[agent_i]])
                    action_ids.append(parse_func(action_names[-1])[0])

            rects = ax.barh(
                y=self._agent_names,
                width=n_steps,
                height=width,
                left=prev_t,
                color=[self._name_to_col[name] for name in action_ids],
                edgecolor="black",
            )

            for action_name, rect in zip(action_names, rects):
                if action_name == "":
                    continue
                action_name = self._get_action_plot_disp(action_name)

                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() / 2,
                    action_name,
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE,
                )
            prev_t += n_steps
        putils.fig_save(self._save_dir, f"{ep_idx}_timeless", fig, False, clear=True)

    def _render_time_plot(self, ac, ep_idx, title):
        H = ac.shape[0]

        width = 0.35
        prev_t = np.zeros(self._num_agents)
        fig, ax = plt.subplots()
        fig.set_size_inches(18, 3, forward=True)
        ax.set_xlabel("LL Timestep")
        ax.set_title(title)

        for t in range(H):
            n_steps = np.zeros(self._num_agents)
            action_names = []
            action_ids = []
            ac_t = ac[t]
            for agent_i in range(self._num_agents):
                if ac_t[agent_i] == -1:
                    action_names.append("")
                    action_ids.append("")
                else:
                    for t_i in range(H - t):
                        if ac[t + t_i][agent_i] == -1 or t_i == 0:
                            n_steps[agent_i] += 1
                        else:
                            break
                    action_names.append(self._action_names[agent_i][ac_t[agent_i]])
                    action_ids.append(parse_func(action_names[-1])[0])

            rects = ax.barh(
                y=self._agent_names,
                width=n_steps,
                height=width,
                left=prev_t,
                color=[self._name_to_col[name] for name in action_ids],
                edgecolor="black",
            )

            for action_name, rect in zip(action_names, rects):
                if action_name == "":
                    continue
                action_name = self._get_action_plot_disp(action_name)

                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    rect.get_y() + rect.get_height() / 2,
                    action_name,
                    ha="center",
                    va="center",
                    fontsize=FONT_SIZE,
                )
            prev_t += n_steps
        putils.fig_save(self._save_dir, f"{ep_idx}_time", fig, False, clear=True)

    def _get_action_plot_disp(self, action_name):
        action_id, args = parse_func(action_name)

        action_name = action_id + "(\n" + ",\n".join(args[:-1]) + ")"
        action_name = action_name.replace("TARGET_", "G_")
        action_name = action_name.replace("_target", "")
        return action_name

    def on_episode_done(self, done_idx, ep_idx, metrics):
        # obs = torch.stack(self._all_obs[done_idx], 0).cpu().numpy()
        ac = np.array(self._all_ac[done_idx])

        self._all_obs[done_idx].clear()
        self._all_ac[done_idx].clear()
        self._num_saved += 1
        if (
            self._config.LOG_INTERVAL == -1
            or self._num_saved % self._config.LOG_INTERVAL != 0
        ):
            return
        num_collide = metrics["num_distinct_collide"]
        success = metrics["composite_success"]

        title = f"Success {success}, Collide {num_collide}"

        self._render_time_plot(ac, ep_idx, title)
        self._render_timeless_plot(ac, ep_idx, title)

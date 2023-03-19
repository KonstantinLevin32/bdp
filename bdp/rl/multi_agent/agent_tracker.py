import os.path as osp
from collections import defaultdict, deque
from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import rl_utils.plotting.utils as putils
import torch
import matplotlib
import seaborn as sns
from bdp.task.sensors import AgentBlameMeasure
import pickle
import torch.distributions as pyd
import copy


def avg_pairwise_kl(probs: Dict[Any, Dict[str, float]]) -> float:
    """
    Takes as input dict mapping pairing ID to probability of events occuring.
    """
    EPS = 1e-3

    def get_bern(p):
        return pyd.Bernoulli(probs=np.clip(p, 0 + EPS, 1 - EPS))

    pairing_names = list(probs.keys())
    event_names = list(probs[pairing_names[0]].keys())
    tmp = [
        (
            name0,
            name1,
            event,
            pyd.kl_divergence(
                get_bern(probs[name0][event]),
                get_bern(probs[name1][event]),
            ),
        )
        for name0 in pairing_names
        for name1 in pairing_names
        if name0 != name1
        for event in event_names
    ]

    return np.mean(
        [
            pyd.kl_divergence(
                get_bern(probs[name0][event]),
                get_bern(probs[name1][event]),
            )
            for name0 in pairing_names
            for name1 in pairing_names
            if name0 != name1
            for event in event_names
        ]
    )


class AgentTracker:
    def __init__(self, config, save_dir, use_wb, agent_sampler_cfg):
        self._config = config
        self._save_dir = save_dir
        self._use_wb = use_wb

        self._fix_agent_a = None
        if agent_sampler_cfg.FIX_AGENT_A != -1:
            self._fix_agent_a = agent_sampler_cfg.FIX_AGENT_A

        self._all_occupancy = defaultdict(
            lambda: defaultdict(
                lambda: deque(maxlen=self._config.POINTS_WINDOW_SIZE)
            )
        )

        self._traj_buffer = [defaultdict(list) for _ in range(2)]
        self._traj_cur_pairing = None

        self._all_traj = defaultdict(
            lambda: defaultdict(
                lambda: deque(maxlen=self._config.TRAJ_WINDOW_SIZE)
            )
        )

        self._stats = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=self._config.WINDOW_SIZE))
        )

    def load_results(self, load_dir):
        save_path = osp.join(load_dir, f"plotdat.pickle")
        with open(save_path, "rb") as f:
            self._stats, self._all_traj, self._all_occupancy = pickle.load(f)

    def save_results(self):
        def make_reg_dict(d):
            return {k: dict(v) for k, v in d.items()}

        save_path = osp.join(self._save_dir, f"plotdat.pickle")
        with open(save_path, "wb") as f:
            pickle.dump(
                [
                    make_reg_dict(x)
                    for x in [self._stats, self._all_traj, self._all_occupancy]
                ],
                f,
            )

    def display(self, num_updates, verbose=False):
        max_agent_idx = max([max(a, b) for a, b in self._stats.keys()]) + 1
        for k in self._config.LOG_KEYS:
            X = np.zeros((max_agent_idx, max_agent_idx))
            plt.imshow(X)
            for agent_pair, values in self._stats.items():
                if k != "count":
                    use_val = sum(values[k]) / sum(values["count"])
                else:
                    use_val = sum(values[k])
                X[agent_pair[0], agent_pair[1]] = use_val

            fig, ax = plt.subplots(1)

            if self._config.RENDER_SELF:
                mask = np.tri(X.shape[0], k=-1)
            else:
                mask = np.tri(X.shape[0], k=0)
            X = np.ma.array(X, mask=mask)
            cmap = copy.copy(matplotlib.cm.get_cmap(self._config.SUCC_CMAP))
            cmap.set_bad("w")

            X *= 100.0

            neg = ax.imshow(
                X, interpolation="none", cmap=cmap, vmin=0.0, vmax=100.0
            )
            disp_k = self._config.RENAME_MAP.get(k, k)
            ax.set_title(disp_k)
            ax.set_xticks(np.arange(max_agent_idx))
            ax.set_yticks(np.arange(max_agent_idx))
            ax.set_xlabel("Agent A Index")
            ax.set_ylabel("Agent B Index")

            self._save_legend_variations(
                f"{k}_{num_updates}",
                f"agent_{k}",
                "cbar_succ",
                verbose,
                fig,
                ax,
                neg,
            )

        # DISABLED TRAJECTORY PLOTTING.
        # if len(self._all_occupancy) > 0:
        #     self._plot_occupancies(num_updates, verbose)
        # if len(self._all_traj) > 0:
        #     self._plot_trajs_ind(num_updates, verbose)
        return self._plot_blames(num_updates, verbose)

    def _plot_blames(self, num_updates, verbose):
        def convert_disp_names(names):
            return [self._config.RENAME_MAP.get(x, x) for x in names]

        def plot_hist(ax, blames, title, event_name_ordering):
            name_disp = convert_disp_names(event_name_ordering)
            plot_dat = [blames[k] for k in event_name_ordering]
            ax.bar(np.arange(len(plot_dat)), plot_dat)
            ax.set_title(title)
            ax.set_xticks(np.arange(len(plot_dat)))
            ax.set_xticklabels(name_disp, rotation=65)
            ax.set_ylim(0.0, 1.0)
            ax.set_ylabel("Blame Ratio")

        # {agent_id -> {subgoal_name -> prob}}
        overall_agent_blames = defaultdict(lambda: defaultdict(list))

        event_name_ordering = None
        for pairing_id, pairing_dat in self._stats.items():
            blames = [{} for _ in range(2)]
            for k, v in pairing_dat.items():
                if not k.startswith(AgentBlameMeasure.cls_uuid):
                    continue
                if any(
                    ignore_k in k for ignore_k in self._config.IGNORE_SUBGOALS
                ):
                    continue
                k = k.split(".")[1]
                parts = k.split("_")
                agent_idx = int(parts[0])
                goal_k = "_".join(parts[1:])
                blames[agent_idx][goal_k] = np.mean(v)

            for agent_id, blame in zip(pairing_id, blames):
                for k, v in blame.items():
                    overall_agent_blames[agent_id][k].append(v)

            event_name_ordering = list(blames[0].keys())
            if len(event_name_ordering) == 0:
                return {}

            # fig, axs = plt.subplots(2, 1, squeeze=False)
            # for agent in range(2):
            #     ax = axs[agent][0]
            #     plot_hist(
            #         ax,
            #         blames[agent],
            #         f"Agent {pairing_id[agent]}",
            #         event_name_ordering,
            #     )
            # putils.fig_save(
            #     self._save_dir,
            #     f"A{pairing_id[0]}_A{pairing_id[1]}_blame_{num_updates}",
            #     fig,
            #     False,
            #     wandb_name=f"agent_{k}",
            #     verbose=verbose,
            #     clear=True,
            #     log_wandb=self._use_wb,
            # )

        # for agent_id, blames in overall_agent_blames.items():
        #     event_name_ordering = list(blames.keys())
        #     fig, axs = plt.subplots(1, 1, squeeze=False)
        #     plot_hist(
        #         axs[0][0],
        #         {k: np.mean(v) for k, v in blames.items()},
        #         f"Agent {agent_id}",
        #         event_name_ordering,
        #     )
        #     putils.fig_save(
        #         self._save_dir,
        #         f"overall_A{agent_id}_blame_{num_updates}",
        #         fig,
        #         False,
        #         wandb_name=f"overall_blame_agent_{k}",
        #         verbose=verbose,
        #         clear=True,
        #         log_wandb=self._use_wb,
        #     )

        # for agent_id, blames in overall_agent_blames.items():
        #     event_name_ordering = list(blames.keys())
        #     fig, axs = plt.subplots(1, 1, squeeze=False)
        #     plot_hist(
        #         axs[0][0],
        #         {k: np.mean(v) for k, v in blames.items()},
        #         f"Agent {agent_id}",
        #         event_name_ordering,
        #     )
        #     putils.fig_save(
        #         self._save_dir,
        #         f"overall_A{agent_id}_blame_{num_updates}",
        #         fig,
        #         False,
        #         wandb_name=f"overall_blame_agent_{k}",
        #         verbose=verbose,
        #         clear=True,
        #         log_wandb=self._use_wb,
        #     )

        agent_names = list(overall_agent_blames.keys())
        all_blame_data = np.zeros((len(event_name_ordering), len(agent_names)))
        for i, agent_id in enumerate(agent_names):
            for j, event_name in enumerate(event_name_ordering):
                all_blame_data[j, i] = np.mean(
                    overall_agent_blames[agent_id][event_name]
                )
        cmap = matplotlib.cm.get_cmap(self._config.CMAP)

        fig, ax = plt.subplots()
        neg = ax.imshow(
            all_blame_data, interpolation="none", cmap=cmap, vmin=0.0, vmax=1.0
        )
        ax.set_xticks(np.arange(len(agent_names)))

        ax.set_yticks(np.arange(len(event_name_ordering)))
        ax.set_yticklabels(convert_disp_names(event_name_ordering))

        ax.set_ylabel(self._config.EVENT_NAME)
        ax.set_xlabel("Agent IDX")

        self._save_legend_variations(
            f"overall_blame_{num_updates}",
            f"overall_blame",
            "cbar",
            verbose,
            fig,
            ax,
            neg,
        )

        return self._compute_stats(
            event_name_ordering, agent_names, overall_agent_blames
        )

    def _save_legend_variations(
        self, fig_name, wb_name, bar_name, verbose, fig, ax, neg
    ):
        putils.fig_save(
            self._save_dir,
            fig_name,
            fig,
            # Plot gets distorted with high quality mode on (likely due to DPI
            # option)
            is_high_quality=False,
            wandb_name=wb_name,
            verbose=verbose,
            clear=True,
            log_wandb=self._use_wb,
        )

        fig.colorbar(neg, ax=ax)
        putils.fig_save(
            self._save_dir,
            f"legend_{fig_name}",
            fig,
            # Plot gets distorted with high quality mode on (likely due to DPI
            # option)
            is_high_quality=False,
            wandb_name=wb_name,
            verbose=verbose,
            clear=True,
            log_wandb=self._use_wb,
        )
        ax.remove()
        putils.fig_save(
            self._save_dir,
            bar_name,
            fig,
            is_high_quality=True,
            verbose=False,
            clear=True,
            log_wandb=False,
        )

    def _compute_stats(
        self, event_name_ordering, agent_names, overall_agent_blames
    ):
        EPS = 1e-6
        adjusted_blames = {
            agent_name: {
                event_name: np.mean(
                    overall_agent_blames[agent_name][event_name]
                )
                + EPS
                for event_name in event_name_ordering
            }
            for agent_name in agent_names
        }

        # Compute P(E)
        p_E = {
            event_name: np.mean(
                [
                    adjusted_blames[agent_name][event_name]
                    for agent_name in agent_names
                ]
            )
            for event_name in event_name_ordering
        }

        def entropy(p):
            return (-(1 - p) * np.log(1 - p)) - (p * np.log(p))

        coverage = np.mean([entropy(p) for p in p_E.values()])
        avg_div = avg_pairwise_kl(adjusted_blames)

        return {"coverage": coverage, "diversity": avg_div}

    def _plot_trajs_ind(self, num_updates, verbose):
        for pairing, traj_dat in self._all_traj.items():
            trajs = {
                agent_id: [x[0] for x in agent_traj]
                for agent_id, agent_traj in traj_dat.items()
            }
            trajs_info = {
                agent_id: [x[1] for x in agent_traj]
                for agent_id, agent_traj in traj_dat.items()
            }
            n_trajs = min(len(trajs[pairing[0]]), len(trajs[pairing[1]]))

            sep_trajs = [
                [trajs[agent_id][i] for agent_id in pairing]
                for i in range(n_trajs)
            ]
            sep_infos = [
                [trajs_info[agent_id][i] for agent_id in pairing]
                for i in range(n_trajs)
            ]

            colors = ["Reds", "Blues"]

            for traj_i, (sep_traj, sep_info) in enumerate(
                zip(sep_trajs, sep_infos)
            ):
                # OLD: Two figures
                # fig, axs = plt.subplots(2, 1, squeeze=False)
                # fig.set_size_inches(4.0, 8.0)

                fig, ax = plt.subplots()
                fig.set_size_inches(4.0, 4.0)

                assert len(sep_traj) == 2

                for traj, color in zip(sep_traj, colors):
                    # ax = ax[0]
                    # Last point is from next episode.
                    traj = traj[:-1]
                    if len(traj) <= 1:
                        continue
                    traj = torch.stack(traj, 0).cpu()
                    pal = sns.color_palette(color, as_cmap=True)
                    traj_len = traj.shape[0]

                    ax.scatter(
                        traj[:, 0],
                        traj[:, 1],
                        color=[
                            pal((i + 1) / (traj_len - 1))
                            for i in range(traj_len)
                        ],
                    )
                    ax.set_xlim(-5.0, 10.0)
                    ax.set_ylim(-5.0, 10.0)

                info_s = ",".join([f"{k}={v}" for k, v in sep_info[0].items()])
                ax.set_title(f"{pairing}: Traj {traj_i}, {info_s}")
                putils.fig_save(
                    self._save_dir,
                    f"traj_{num_updates}_A{pairing[0]}_A{pairing[1]}_{traj_i}",
                    fig,
                    False,
                    wandb_name=f"agent_occ",
                    verbose=verbose,
                    clear=True,
                    log_wandb=self._use_wb,
                )

    # def _plot_trajs(self, num_updates, verbose):
    #     n_draws = len(self._all_traj)
    #     fig, axs = plt.subplots(2, n_draws, squeeze=False)
    #     fig.set_size_inches(n_draws * 4.0, 8.0)

    #     for plot_i, (pairing, (trajs, ep_dat)) in enumerate(
    #         self._all_traj.items()
    #     ):
    #         colors = ["hot", "cool"]
    #         cur_axs = [ax[plot_i] for ax in axs]

    #         for ax, agent_id, color in zip(cur_axs, pairing, colors):
    #             pal = sns.color_palette(color, as_cmap=True)
    #             agent_trajs = [
    #                 torch.stack(traj, 0).cpu()
    #                 for traj in trajs[agent_id]
    #                 if len(traj) != 0
    #             ]
    #             for traj in agent_trajs:
    #                 traj_len = traj.shape[0]
    #                 ax.scatter(
    #                     traj[:, 0],
    #                     traj[:, 1],
    #                     color=[
    #                         pal((i + 1) / (traj_len - 1))
    #                         for i in range(traj_len)
    #                     ],
    #                     # s=10,
    #                 )
    #                 ax.set_xlim(-5.0, 10.0)
    #                 ax.set_ylim(-5.0, 10.0)
    #         ax.set_title(f"{pairing}")
    #     putils.fig_save(
    #         self._save_dir,
    #         f"traj_{num_updates}",
    #         fig,
    #         False,
    #         wandb_name=f"agent_occ",
    #         verbose=verbose,
    #         clear=True,
    #         log_wandb=self._use_wb,
    #     )

    def _plot_occupancies(self, num_updates, verbose):
        n_draws = len(self._all_occupancy)
        fig, axs = plt.subplots(2, n_draws, squeeze=False)
        fig.set_size_inches(n_draws * 4.0, 8.0)

        for plot_i, (pairing, occupancies) in enumerate(
            self._all_occupancy.items()
        ):
            colors = ["hot", "cool"]
            cur_axs = [ax[plot_i] for ax in axs]

            for ax, agent_id, color in zip(cur_axs, pairing, colors):
                points = torch.stack(list(occupancies[agent_id]), 0).cpu()
                ax.hexbin(
                    points[:, 0],
                    points[:, 1],
                    gridsize=50,
                    extent=(-5.0, 10.0, -5.0, 10.0),
                    cmap=color,
                    mincnt=1,
                )
            ax.set_title(f"{pairing}")
        putils.fig_save(
            self._save_dir,
            f"occ_{num_updates}",
            fig,
            False,
            wandb_name=f"agent_occ",
            verbose=verbose,
            clear=True,
            log_wandb=self._use_wb,
        )

    def log_obs(self, obs, dones, pairing_id, infos):
        if (
            self._config.TRAJ_WINDOW_SIZE <= 0
            and self._config.POINTS_WINDOW_SIZE <= 0
        ):
            return

        if self._config.LOG_INTERVAL == -1:
            return
        id0, id1 = pairing_id
        if self._fix_agent_a is not None:
            other_id = id1 if id0 == self._fix_agent_a else id0
            pairing_id = (self._fix_agent_a, other_id)
        elif id0 > id1:
            pairing_id = (id1, id0)

        if (
            self._config.TRAJ_WINDOW_SIZE > 0
            and self._traj_cur_pairing != pairing_id
        ):
            self._traj_cur_pairing = pairing_id
            for i in range(2):
                self._traj_buffer[i].clear()

        for i, agent_id in enumerate(pairing_id):
            pos = obs[f"AGENT_{i}_localization_sensor"]
            n_envs = pos.shape[0]
            xy_pos = pos[:, [0, 2]]
            self._all_occupancy[pairing_id][agent_id].extend(xy_pos)

            if self._config.TRAJ_WINDOW_SIZE > 0:
                for env_i in range(n_envs):
                    self._traj_buffer[i][env_i].append(xy_pos[env_i])
                    if dones[env_i] == 1.0:
                        info = infos[env_i]
                        ep_dat = {
                            k: info[k]
                            for k in self._config.LOG_KEYS
                            if k != "count"
                        }
                        self._all_traj[pairing_id][agent_id].append(
                            (self._traj_buffer[i][env_i], ep_dat)
                        )
                        self._traj_buffer[i][env_i] = []

    def training_log(self, deltas, pairing_id, num_updates):
        pairing_id = tuple(sorted(pairing_id))
        for k in self._config.LOG_KEYS:
            if k == "count" and k not in deltas:
                self._stats[pairing_id][k].append(1)
            elif k in deltas:
                self._stats[pairing_id][k].append(deltas[k])
        for k, v in deltas.items():
            if k.startswith(AgentBlameMeasure.cls_uuid):
                self._stats[pairing_id][k].append(v)

        if (
            self._config.LOG_INTERVAL != -1
            and (num_updates + 1) % self._config.LOG_INTERVAL == 0
        ):
            return self.display(num_updates)
        else:
            return {}

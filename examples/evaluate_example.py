import json
import math
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from evaluate_policy import load_and_evaluate_one_episode, effort_and_energy_based_criterion_function

mpl.rcParams.update({
    "font.family": "Helvetica",
    "font.sans-serif": ["Helvetica"],
    "pdf.use14corefonts": True,
    "ps.useafm": True,

    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 22,
})

FIG_TITLE_FONTSIZE = 24
SUBFIG_TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 22
LEGEND_FONTSIZE = 20
TICK_FONTSIZE = 20
LINEWIDTH = 3
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "evaluate_example_config.json")

def plot_deformation(ax, title, obs, terminal_index):
    time = np.arange(obs.shape[0])
    # Plot deformation curves
    ax.plot(time, obs[:, 1], label='Stretch Deformation', color="#5db847", linewidth=LINEWIDTH, alpha=0.8)
    ax.plot(time, obs[:, 4], label='Push Deformation', color="#ee7c8b", linewidth=LINEWIDTH, alpha=0.8)

    # Vertical marker
    ax.axvline(x=terminal_index, color='black', linestyle='-.', label='Finish', linewidth=1)

    # Labels / formatting
    ax.set_xlabel('Time (ms)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Deform (m)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    # ax.set_title(title)
    # ax.legend()
    ax.grid()

def plot_force(ax, title, obs, act, terminal_index):
    time = np.arange(obs.shape[0])
    # Plot force curves
    ax.plot(time[:terminal_index], act[:terminal_index, 0], label='FF Stretch', color='#5db847', linestyle='--', linewidth=LINEWIDTH)
    ax.plot(time[:terminal_index], act[:terminal_index, 1], label='FF Push', color="#ee7c8b", linestyle='--', linewidth=LINEWIDTH)
    ax.plot(time, obs[:, 0], label='Ext Stretch', color='#5db847', linewidth=LINEWIDTH, alpha=0.8)
    ax.plot(time, obs[:, 3], label='Ext Push', color="#ee7c8b", linewidth=LINEWIDTH, alpha=0.8)

    # Vertical marker
    ax.axvline(x=terminal_index, color='black', linestyle='-.', label='Finish', linewidth=1)

    # Labels / formatting
    ax.set_xlabel('Time (ms)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Force (N)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    # ax.set_title(title)
    # ax.legend()
    ax.grid()

def load_episode(base_dir, traj_id_str):
    traj_id_str = str(traj_id_str)
    traj_path = os.path.join(base_dir, traj_id_str)
    group_key, obs, acts, _, terminals_buffer, _, _ = load_and_evaluate_one_episode(
        traj_id_str,
        traj_path,
        effort_and_energy_based_criterion_function,
        None, 1, False
    )
    idxs = np.nonzero(terminals_buffer)[0]
    terminal_index = idxs[0] if idxs.size else -1
    plot_range = terminal_index + 500
    return obs, acts, terminal_index, plot_range


def format_dlo_label(dlo_key):
    return dlo_key.replace("_", "")


def format_clip_label(clip_key):
    parts = clip_key.split("_")
    if len(parts) == 3:
        return f"{parts[0]}{parts[1]}-{parts[2]}"
    return clip_key.replace("_", "-")


def load_trial_configs(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        configs = json.load(f)["configs"]

    trial_configs = []
    for dlo_key, clip_configs in configs.items():
        for clip_key, trial_config in clip_configs.items():
            if not trial_config:
                continue
            trial_configs.append(
                {
                    "dlo": format_dlo_label(dlo_key),
                    "clip": format_clip_label(clip_key),
                    "trained": trial_config["trained"],
                    "random": trial_config["random"],
                }
            )
    return trial_configs


if __name__ == "__main__":
    trained_base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/evaluation_trained_with_encoder"
    random_base_dir  = "/home/tp2/Documents/kejia/clip_fixing_dataset/evaluation_random"
    trial_configs = load_trial_configs(CONFIG_PATH)

    for fig_index in range(0, len(trial_configs), 2):
        pair = trial_configs[fig_index: fig_index + 2]
        if len(pair) < 2:
            break

        fig = plt.figure(figsize=(24, 5.7))
        outer = fig.add_gridspec(
            1, 2,
            wspace=0.18,
            left=0.06, right=0.99,
            top=0.85, bottom=0.25
        )

        def draw_one_trial(gs_cell, random_obs, random_acts, random_terminal, random_range,
                           trained_obs, trained_acts, trained_terminal, trained_range,
                           trial_title):
            inner = gs_cell.subgridspec(
                2, 2,
                width_ratios=[random_range, trained_range],
                wspace=0.05,
                hspace=0.12
            )

            ax00 = fig.add_subplot(inner[0, 0])
            ax01 = fig.add_subplot(inner[0, 1])
            ax10 = fig.add_subplot(inner[1, 0])
            ax11 = fig.add_subplot(inner[1, 1])

            plot_force(ax10, "Random", random_obs, random_acts, random_terminal)
            ax10.set_xlim(0, random_range)

            plot_force(ax11, "Trained", trained_obs, trained_acts, trained_terminal)
            ax11.set_xlim(0, trained_range)

            plot_deformation(ax00, "Random", random_obs, random_terminal)
            ax00.set_xlim(0, random_range)

            plot_deformation(ax01, "Trained", trained_obs, trained_terminal)
            ax01.set_xlim(0, trained_range)

            for ax in (ax00, ax01):
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            for ax in (ax01, ax11):
                ax.set_ylabel("")
                ax.tick_params(labelleft=False)

            ax00.set_title("Random", fontsize=SUBFIG_TITLE_FONTSIZE)
            ax01.set_title("Trained", fontsize=SUBFIG_TITLE_FONTSIZE)

            p0 = ax00.get_position()
            p1 = ax01.get_position()
            x_center = (p0.x0 + p1.x1) / 2
            y_top = max(p0.y1, p1.y1) + 0.08
            fig.text(x_center, y_top, trial_title, ha="center", va="bottom", fontsize=FIG_TITLE_FONTSIZE)

            return ax10, (ax00, ax10)

        left_axes = []
        ax_for_legend = None
        output_tokens = []
        for panel_index, trial in enumerate(pair):
            trained_obs, trained_acts, trained_terminal, trained_range = load_episode(
                trained_base_dir, trial["trained"]
            )
            random_obs, random_acts, random_terminal, random_range = load_episode(
                random_base_dir, trial["random"]
            )
            ax_for_legend, axes_to_align = draw_one_trial(
                outer[panel_index],
                random_obs, random_acts, random_terminal, random_range,
                trained_obs, trained_acts, trained_terminal, trained_range,
                trial_title=f"Fixing {trial['dlo']} into {trial['clip']}"
            )
            left_axes.extend(axes_to_align)
            output_tokens.append(f"{trial['dlo']}_{trial['clip']}")

        fig.align_ylabels(left_axes)

        handles, labels = ax_for_legend.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(labels),
            frameon=False,
            bbox_to_anchor=(0.5, 0.005)
        )

        output_name = f"example_plot_{'_and_'.join(output_tokens)}.pdf"
        plt.savefig(os.path.join(trained_base_dir, output_name))
        plt.show()
        plt.close(fig)

    print("done")

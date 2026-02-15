import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math
import os
from evaluate_policy import load_and_evaluate_one_episode, effort_and_energy_based_criterion_function
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "Times New Roman",

    "axes.labelsize": 22,
    "axes.titlesize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 22,
})

FIG_TITLE_FONTSIZE = 28
SUBFIG_TITLE_FONTSIZE = 26
LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 18
TICK_FONTSIZE = 18
LINEWIDTH = 3

def plot_deformation(ax, title, obs, terminal_index):
    time = np.arange(obs.shape[0])
    # Plot deformation curves
    ax.plot(time, obs[:, 1], label='Stretch Deformation', color="#5db847", linewidth=LINEWIDTH, alpha=0.6)
    ax.plot(time, obs[:, 4], label='Push Deformation', color="#ee7c8b", linewidth=LINEWIDTH, alpha=0.6)

    # Vertical marker
    ax.axvline(x=terminal_index, color='black', linestyle='-.', label='Finish', linewidth=1)

    # Labels / formatting
    ax.set_xlabel('Time (ms)', fontsize=LABEL_FONTSIZE)
    ax.set_ylabel('Deformation (m)', fontsize=LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    # ax.set_title(title)
    # ax.legend()
    ax.grid()

def plot_force(ax, title, obs, act, terminal_index):
    time = np.arange(obs.shape[0])
    # Plot force curves
    ax.plot(time[:terminal_index], act[:terminal_index, 0], label='FF Stretch', color='#5db847', linestyle='--', linewidth=LINEWIDTH)
    ax.plot(time[:terminal_index], act[:terminal_index, 1], label='FF Push', color="#ee7c8b", linestyle='--', linewidth=LINEWIDTH)
    ax.plot(time, obs[:, 0], label='Ext Stretch', color='#5db847', linewidth=LINEWIDTH, alpha=0.6)
    ax.plot(time, obs[:, 3], label='Ext Push', color="#ee7c8b", linewidth=LINEWIDTH, alpha=0.6)

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


if __name__ == "__main__":
    trained_base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/evaluation_trained_with_encoder"
    random_base_dir  = "/home/tp2/Documents/kejia/clip_fixing_dataset/evaluation_random"

    # ------------------------
    # Trial A: trained 3704, random 1272
    # Trial B: trained 3686, random 3916
    # ------------------------
    trained_obs_A, trained_acts_A, trained_terminal_A, trained_range_A = load_episode(trained_base_dir, "3697")
    random_obs_A,  random_acts_A,  random_terminal_A,  random_range_A  = load_episode(random_base_dir,  "3823")

    trained_obs_B, trained_acts_B, trained_terminal_B, trained_range_B = load_episode(trained_base_dir, "3686")
    random_obs_B,  random_acts_B,  random_terminal_B,  random_range_B  = load_episode(random_base_dir,  "3916")

    # ------------------------
    # Big figure: two panels side-by-side (each panel is your original 2x2)
    # ------------------------
    fig = plt.figure(figsize=(24, 5.7))

    # 外层：左右两块区域（两个 trial）
    outer = fig.add_gridspec(
        1, 2,
        wspace=0.18,     # ✅ 只调这个决定两张图之间距离（想更近就减小，比如 0.10）
        left=0.06, right=0.99,
        top=0.85, bottom=0.25
    )

    def draw_one_trial(gs_cell, random_obs, random_acts, random_terminal, random_range,
                       trained_obs, trained_acts, trained_terminal, trained_range,
                       trial_title):
        # 内层：完全复刻你原来的 2x2
        inner = gs_cell.subgridspec(
            2, 2,
            width_ratios=[random_range, trained_range],  # ✅ 保留你原来的比例效果
            wspace=0.05,
            hspace=0.12
        )

        ax00 = fig.add_subplot(inner[0, 0])  # deformation random
        ax01 = fig.add_subplot(inner[0, 1])  # deformation trained
        ax10 = fig.add_subplot(inner[1, 0])  # force random
        ax11 = fig.add_subplot(inner[1, 1])  # force trained

        # Bottom row: Force
        plot_force(ax10, "Random", random_obs, random_acts, random_terminal)
        ax10.set_xlim(0, random_range)

        plot_force(ax11, "Trained", trained_obs, trained_acts, trained_terminal)
        ax11.set_xlim(0, trained_range)

        # Top row: Deformation
        plot_deformation(ax00, "Random", random_obs, random_terminal)
        ax00.set_xlim(0, random_range)

        plot_deformation(ax01, "Trained", trained_obs, trained_terminal)
        ax01.set_xlim(0, trained_range)

        # 复刻你原来的处理：去掉顶行 x label
        for ax in (ax00, ax01):
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)

        # 复刻你原来的处理：去掉右列 y label
        for ax in (ax01, ax11):
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)

        # 列标题（跟你原来一样）
        ax00.set_title("Random", fontsize=SUBFIG_TITLE_FONTSIZE)
        ax01.set_title("Trained", fontsize=SUBFIG_TITLE_FONTSIZE)

        # Trial 标题：放在该 trial 的两列上方居中
        p0 = ax00.get_position()
        p1 = ax01.get_position()
        x_center = (p0.x0 + p1.x1) / 2
        y_top = max(p0.y1, p1.y1) + 0.08
        fig.text(x_center, y_top, trial_title, ha="center", va="bottom", fontsize=SUBFIG_TITLE_FONTSIZE)

        return ax10  # 返回一个带 legend 的轴（用来抽 handles/labels）

    ax_for_legend_A = draw_one_trial(
        outer[0],
        random_obs_A, random_acts_A, random_terminal_A, random_range_A,
        trained_obs_A, trained_acts_A, trained_terminal_A, trained_range_A,
        trial_title="Fixing BMM into CL-2"
    )

    draw_one_trial(
        outer[1],
        random_obs_B, random_acts_B, random_terminal_B, random_range_B,
        trained_obs_B, trained_acts_B, trained_terminal_B, trained_range_B,
        trial_title="Fixing BLL into CL-2"
    )

    # ------------------------
    # One shared legend at the bottom (only share legend)
    # ------------------------
    handles, labels = ax_for_legend_A.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        # fontsize=LEGEND_FONTSIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005)  # ✅ legend 在图内底部，不会把图挤乱
    )

    plt.savefig(os.path.join(trained_base_dir, "example_plot_two_trials.pdf"))
    plt.show()

    print("done")
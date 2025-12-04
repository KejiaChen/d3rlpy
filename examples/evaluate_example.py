import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import math
import os
from evaluate_policy import load_and_evaluate_one_episode, effort_and_energy_based_criterion_function
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['mathtext.fontset'] = 'custom'
mpl.rcParams['mathtext.rm'] = 'Times New Roman'
mpl.rcParams['mathtext.it'] = 'Times New Roman:italic'
mpl.rcParams['mathtext.bf'] = 'Times New Roman:bold'

mpl.rcParams['pdf.fonttype'] = 42  
mpl.rcParams['ps.fonttype'] = 42

FIG_TITLE_FONTSIZE = 28
SUBFIG_TITLE_FONTSIZE = 22
LABEL_FONTSIZE = 20
LEGEND_FONTSIZE = 18
TICK_FONTSIZE = 18
LINEWIDTH = 3

def plot_deformation(ax, title, obs, terminal_index):
    time = np.arange(obs.shape[0])
    # Plot deformation curves
    ax.plot(time, obs[:, 1], label='Stretch Deformation', color='#86bc79', linewidth=LINEWIDTH, alpha=0.6)
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
    ax.plot(time[:terminal_index], act[:terminal_index, 0], label='FF Stretch', color='#86bc79', linestyle='--', linewidth=LINEWIDTH)
    ax.plot(time[:terminal_index], act[:terminal_index, 1], label='FF Push', color="#ee7c8b", linestyle='--', linewidth=LINEWIDTH)
    ax.plot(time, obs[:, 0], label='Ext Stretch', color='#86bc79', linewidth=LINEWIDTH, alpha=0.6)
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


if __name__ == "__main__":
    # load trained
    trained_base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/evaluation_trained_with_encoder"
    trained_traj_dir = str(3686) # 3704 #3697 #3686
    trained_traj_path = os.path.join(trained_base_dir, trained_traj_dir)
    trained_group_key, trained_obs, trained_acts, _, trained_terminals_buffer, _, _ = load_and_evaluate_one_episode(trained_traj_dir, 
                                                                                                            trained_traj_path, 
                                                                                                            effort_and_energy_based_criterion_function, 
                                                                                                            None, 1, False)
    trained_idxs = np.nonzero(trained_terminals_buffer)[0]
    trained_fixing_terminal_index = trained_idxs[0] if trained_idxs.size else -1
    # plot_example(trained_obs, trained_acts, trained_base_dir, trained_traj_dir, trained_fixing_terminal_index, trained_obs.shape[0]-1)
    trained_range = trained_fixing_terminal_index + 500

    # load random
    random_base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/evaluation_random"
    random_traj_dir = str(3916) # 1272 #3823 #3916
    random_traj_path = os.path.join(random_base_dir, random_traj_dir)
    random_group_key, random_obs, random_acts, _, random_terminals_buffer, _, _ = load_and_evaluate_one_episode(random_traj_dir, 
                                                                                                            random_traj_path, 
                                                                                                            effort_and_energy_based_criterion_function, 
                                                                                                            None, 1, False)
    random_idxs = np.nonzero(random_terminals_buffer)[0]
    random_fixing_terminal_index = random_idxs[0] if random_idxs.size else -1
    random_range = random_fixing_terminal_index + 500

    # Plotting
    fig, axs = plt.subplots(
                    2, 2,
                    figsize=(12, 5.5),
                    gridspec_kw={'width_ratios': [random_range, trained_range]},
                    sharey='row'
                )

    plot_force(axs[1, 0], "Random", random_obs, random_acts, random_fixing_terminal_index)
    axs[1, 0].set_xlim(0, random_range)
    plot_force(axs[1, 1], "Trained", trained_obs, trained_acts, trained_fixing_terminal_index)
    axs[1, 1].set_xlim(0, trained_range)

    plot_deformation(axs[0, 0], "Random", random_obs, random_fixing_terminal_index)
    axs[0, 0].set_xlim(0, random_range)
    plot_deformation(axs[0, 1], "Trained", trained_obs, trained_fixing_terminal_index)
    axs[0, 1].set_xlim(0, trained_range)

    # remove x labels on top row
    for ax in axs[0, :]:
        ax.set_xlabel("")    # remove top x labels
        ax.tick_params(labelbottom=False)   # hide x tick labels

    # remove y labels on right column
    for ax in axs[:, 1]:
        ax.set_ylabel("")    # remove right y labels
        ax.tick_params(labelleft=False)     # hide y tick labels

    # subtitles
    axs[0, 0].set_title("Random", fontsize=SUBFIG_TITLE_FONTSIZE)
    axs[0, 1].set_title("Trained", fontsize=SUBFIG_TITLE_FONTSIZE)

    # ONE ROW LEGEND AT THE BOTTOM
    handles, labels = axs[1, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=len(labels),
        fontsize=LEGEND_FONTSIZE,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02)
    )
    plt.tight_layout(rect=[0, 0.2, 1, 1])  # leave 15% of the figure height at the bottom
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(trained_base_dir, "example_plot.png"))
    plt.show()

    print("done")
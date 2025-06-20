import numpy as np
import os
from d3rlpy.dataset import MDPDataset, EpisodeDataset
from fixing_detection_no_smoothing import FFAnalyzerRTSlide 

# def success_threshold(criterion, threshold):

from torch.utils.data import Dataset, DataLoader
import torch

def check_success(traj_path, obs_buffer, acts_buffer):
    analyzer = FFAnalyzerRTSlide()

    # Configure it
    analyzer.set_window_size(30)          # Example window size
    analyzer.set_crt_type("dep")          # Use "dep" for dfext/dff analysis
    analyzer.set_stat_type("roll_slope")  # or "roll_zscore", "roll_slope"
    analyzer.set_z_score(0.5)             # Set z-score threshold

    push_ext = obs_buffer[:, 3]  # Assuming first column is fext
    push_ff = acts_buffer[:, 1]  # Assuming second column is ff

    fixing_success = False

    terminal_buffer = np.zeros(obs_buffer.shape[0], dtype=bool)
    # if traj_path contains unfixed or detectfailure in its name, then the trajectory is not successful
    if "unfixed" in traj_path or "detectfailure" in traj_path:
        terminate_index = obs_buffer.shape[0]
        print(f"\033[91mTrajectory {traj_path} is not successful, skipping fixing check.\033[0m")
    else:
        for i in range(obs_buffer.shape[0]):
            analyzer.update_sensor_data(push_ext[i], push_ff[i])
            if analyzer.get_contactlos():
                print(f"Finish fixing at step {i} out of {obs_buffer.shape[0]} steps.")
                terminate_index = i
                fixing_success = True
                break

        if not fixing_success:
            print(f"\033[91mFixing did not succeed in total number of {obs_buffer.shape[0]} steps.\033[0m")
            terminate_index = obs_buffer.shape[0]

        terminal_buffer[terminate_index:] = True  # last step is always terminal
    return terminal_buffer, terminate_index

def preprocess_action(action_buffer, env_step=5):
    downsampled_action_buffer = np.concatenate([
        action_buffer[::env_step],
        action_buffer[-1:] if (action_buffer.shape[0]) % env_step != 0 else np.empty((0, action_buffer.shape[1]))
    ])

    # delta force or abosolute force?
    # absolute force
    stretch_force_ff = downsampled_action_buffer[:, 0]
    # shift the force to calculate delta force
    stretch_force_ff_shifted = stretch_force_ff
    stretch_force_ff_shifted[0] = stretch_force_ff[0]
    stretch_force_ff_shifted[1:] = stretch_force_ff[:-1]
    # calculate delta force
    delta_stretch_force_ff = stretch_force_ff - stretch_force_ff_shifted

    push_force_ff = downsampled_action_buffer[:, 1]
    push_force_ff_shifted = push_force_ff
    push_force_ff_shifted[0] = push_force_ff[0]
    push_force_ff_shifted[1:] = push_force_ff[:-1]
    delta_push_force_ff = push_force_ff - push_force_ff_shifted

    return np.stack([stretch_force_ff, push_force_ff], axis=1)

def load_and_preprocess_terminal(traj_path, obs_buffer, acts_buffer):
    terminal_file = os.path.join(traj_path, "terminal.txt")
    # if not os.path.exists(terminal_file):
    start_check_index = 30 # due to communication latency, first few steps are not reliable
    terminal_buffer = np.zeros(obs_buffer.shape[0], dtype=bool)
    terminals_from_start, terminate_index_from_start = check_success(traj_path, obs_buffer[start_check_index:], acts_buffer[start_check_index:])
    terminal_buffer[start_check_index:] = terminals_from_start
    # write terminal to terminal file
    with open(terminal_file, 'w') as f:
        np.savetxt(f, terminal_buffer, fmt='%.6f')
    # else:
    #     terminal_buffer = np.loadtxt(terminal_file)
    return terminal_buffer, terminate_index_from_start + start_check_index

def preprocess_observation(obs_buffer, env_step=5):
    downsampled_obs_buffer = np.concatenate([
        obs_buffer[::env_step],
        obs_buffer[-1:] if (obs_buffer.shape[0]) % env_step != 0 else np.empty((0, obs_buffer.shape[1]))
    ])

    stretch_force_ext = downsampled_obs_buffer[:, 0]
    stretch_distance = downsampled_obs_buffer[:, 1]
    stretch_velocity = downsampled_obs_buffer[:, 2]
    push_force_ext = downsampled_obs_buffer[:, 3]
    push_distance = downsampled_obs_buffer[:, 4]
    push_velocity = downsampled_obs_buffer[:, 5]
            
    # distance with refernce to starting point
    stretch_distance_from_start = stretch_distance - stretch_distance[0]
    push_distance_from_start = push_distance - push_distance[0]

    return np.stack([
        stretch_force_ext,
        stretch_distance_from_start,
        stretch_velocity,
        push_force_ext,
        push_distance_from_start,
        push_velocity,
    ], axis=1)


def return_function(obs):
    # === 1. Extract Metrics ===
    stretch_force = np.max(np.abs(obs[:, 0])) # np.mean
    push_force = np.max(np.abs(obs[:, 3]))
    deformation = np.max(np.abs(obs[:, 4])) 
    duration = obs.shape[0]  # number of steps

    # === 2. Define Ideal Reference Values (based on expert or dataset stats) ===
    REF_STRETCH_FORCE = 7.0
    REF_PUSH_FORCE = 7.0
    REF_DEFORMATION = 0.01
    REF_DURATION = 1000  

    # === 3. Compute Positive Reward by Deviation ===
    stretch_score = 1.0 - (stretch_force - REF_STRETCH_FORCE) / REF_STRETCH_FORCE
    push_score = 1.0 - (push_force - REF_PUSH_FORCE) / REF_PUSH_FORCE
    deform_score = 1.0 - (deformation - REF_DEFORMATION) / REF_DEFORMATION
    time_score = 1.0 - (duration - REF_DURATION) / REF_DURATION

    # Clip scores to [0, 1]
    # stretch_score = np.clip(stretch_score, 0.0, 1.0)
    # push_score = np.clip(push_score, 0.0, 1.0)
    # deform_score = np.clip(deform_score, 0.0, 1.0)
    # time_score = np.clip(time_score, 0.0, 1.0)

    ep_return = 10*stretch_score + 10*push_score + 10*deform_score + 10*time_score

    return ep_return

def load_trajectories(base_dir, env_step=5, reload_data=True):
    if not reload_data and os.path.exists(os.path.join(base_dir, "episode_dataset.npz")):
        print(f"Loading existing episode dataset from {os.path.join(base_dir, 'episode_dataset.npz')}")
        data = np.load(os.path.join(base_dir, "episode_dataset.npz"), allow_pickle=True)
        return EpisodeDataset(
            observations=data["observations"],
            actions=data["actions"],
            terminals=data["terminals"],
            returns=data["returns"]
        )
    else:
        all_observations = []
        all_actions = []
        all_returns = []
        all_terminals = []

        for traj_dir in sorted(os.listdir(base_dir)):
            traj_path = os.path.join(base_dir, traj_dir)
            if not os.path.isdir(traj_path):  # Skip files like "episode_dataset.npz"
                continue
            obs_buffer_raw = np.loadtxt(os.path.join(traj_path, "observations.txt"))
            acts_buffer_raw = np.loadtxt(os.path.join(traj_path, "actions.txt"))
            # check success
            terminals_raw, terminal_index_raw = load_and_preprocess_terminal(traj_path, obs_buffer_raw, acts_buffer_raw)
            # downsample
            obs_buffer_downsampled = preprocess_observation(obs_buffer_raw, env_step=env_step)
            acts_buffer_downsampled  = preprocess_action(acts_buffer_raw, env_step=env_step)
            terminals_downsampled = np.concatenate([
                                        terminals_raw[::env_step],
                                        [terminals_raw[-1]] if (terminals_raw.shape[0]) % env_step != 0 else np.empty(0, dtype=terminals_raw.dtype)
                                    ])
            terminal_index_downsampled = terminal_index_raw // env_step
            print(f"Trajectory {traj_dir} has {obs_buffer_downsampled.shape[0]} steps after downsampling.")

            obs_buffer = obs_buffer_downsampled[:]
            acts_buffer = acts_buffer_downsampled[:]
            terminals_buffer = terminals_downsampled[:]

            # calcuate reward
            ep_return = return_function(obs_buffer)
            # save reward to a file
            np.savetxt(os.path.join(traj_path, "return.txt"), np.array([ep_return ]), fmt='%.6f')
            print(f"\033[93mExpected return for trajectory {traj_dir}: {ep_return} with {obs_buffer.shape[0]} steps.\033[0m")
            plot_force(obs_buffer, acts_buffer, base_dir, traj_dir)
            
            all_observations.append(obs_buffer)
            all_actions.append(acts_buffer)
            all_returns.append(ep_return)
            all_terminals.append(terminals_downsampled)

            # check if their size is consistent
            if not (obs_buffer.shape[0] == acts_buffer.shape[0]== terminals_buffer.shape[0]):
                raise ValueError(f"Size mismatch in trajectory {traj_dir}: "
                                f"observations {obs_buffer.shape[0]}, "
                                f"actions {acts_buffer.shape[0]}, "
                                f"terminals {terminals_buffer.shape[0]}.")
        # save to local files
        np.savez(
            os.path.join(base_dir, "episode_dataset.npz"),
            observations=np.array(all_observations, dtype=object),
            actions=np.array(all_actions, dtype=object),
            terminals=np.array(all_terminals, dtype=object),    
            returns=np.array(all_returns, dtype=np.float32)
        )
        print(f"Reloaded episode dataset saved to {os.path.join(base_dir, 'episode_dataset.npz')}")

        return EpisodeDataset(observations=all_observations,
                            actions=all_actions,
                            terminals=all_terminals,
                            returns=all_returns,
                            timeouts=None)  # timeouts are not used in this case

def plot_force(obs, act, base_dir, traj_dir):
    import matplotlib.pyplot as plt

    # if force_type == 'stretch':
    #     force_ext = obs[:, 0]
    #     distance = obs[:, 1]
    #     force_ff = act[:, 0]
    # elif force_type == 'push':
    #     force_ext = obs[:, 3]
    #     distance = obs[:, 4]
    #     force_ff = act[:, 1]
    # else:
    #     raise ValueError("force_type must be either 'stretch' or 'push'.")

    # Generate a linearly increasing time array
    time = np.arange(obs.shape[0])

    # Create subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # Plot force vs distance
    axs[0].plot(time, obs[:, 1], label=f'Stretch Distance', color='orange')
    axs[0].plot(time, obs[:, 4], label=f'Push Distance', color='blue')
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Distance from Start (m)')
    axs[0].set_title(f'Distance vs Time')
    axs[0].legend()
    axs[0].grid()

    # Plot force vs time
    axs[1].plot(time, obs[:, 0], label=f'Ext Stretch Force', color='orange')
    axs[1].plot(time, act[:, 0], label=f'FF Stretch Force', linestyle='--', color='orange')
    axs[1].plot(time, obs[:, 3], label=f'Ext Push Force', color='blue')
    axs[1].plot(time, act[:, 1], label=f'FF Push Force', linestyle='--', color='blue')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Force (N)')
    axs[1].set_title(f'Force vs Time')
    axs[1].legend()
    axs[1].grid()

    # Adjust layout and show the plot
    plt.tight_layout()
    # plt.show()
    plt_file = os.path.join(os.path.join(base_dir, traj_dir), f"{traj_dir}_force_plot.png")
    plt.savefig(plt_file)
    plt.close(fig)

# dataloader
class EpisodeBCDataset(Dataset):
    def __init__(self, episode_dataset: EpisodeDataset):
        self.episodes = episode_dataset.get_all_episodes()
        self.total_returns = np.array([ep.episode_return for ep in self.episodes], dtype=np.float32)
        self.total_returns = (self.total_returns - self.total_returns.min()) / max(1e-8, (self.total_returns.max() - self.total_returns.min()))  # normalize to [0, 1]

    def __len__(self):
        return len(self.episodes)

    def __getitem__(self, idx):
        ep = self.episodes[idx]
        return (
            torch.tensor(ep.observations, dtype=torch.float32),
            torch.tensor(ep.actions, dtype=torch.float32),
            torch.tensor(self.total_returns[idx], dtype=torch.float32)
        )


if __name__ == "__main__":
    base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/"
    dataset = load_trajectories(base_dir, env_step=5)
    print(f"Dataset loaded.")

    
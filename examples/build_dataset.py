import numpy as np
import os
from d3rlpy.dataset import MDPDataset

# def success_threshold(criterion, threshold):

def check_success(trail_length):
    terminal_buffer = np.zeros(trail_length, dtype=bool)
    terminal_buffer[-1] = True  # last step is always terminal
    return terminal_buffer

def load_and_preprocess_action(act_file, env_step=5):
    action_buffer = np.loadtxt(act_file)
    downsampled_action_buffer = np.concatenate([
        action_buffer[::env_step],
        action_buffer[-1:] if (action_buffer.shape[0] - 1) % env_step != 0 else []
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

def load_and_preprocess_terminal(terminal_file, trial_length, env_step=5):
    if not os.path.exists(terminal_file):
        terminal_buffer = check_success(trial_length)
        # write terminal to terminal file
        with open(terminal_file, 'w') as f:
            np.savetxt(f, terminal_buffer, fmt='%.6f')
    else:
        terminal_buffer = np.loadtxt(terminal_file)
    return terminal_buffer

def load_and_preprocess_observation(obs_file, env_step=5):
    obs_buffer = np.loadtxt(obs_file)
    downsampled_obs_buffer = np.concatenate([
        obs_buffer[::env_step],
        obs_buffer[-1:] if (obs_buffer.shape[0] - 1) % env_step != 0 else []
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


def reward_function(observation, terminal):
    reward = 0.0

    stretch_force_ext = observation[0]
    stretch_distance_from_start = observation[1]
    stretch_velocity = observation[2]
    push_force_ext = observation[3]
    push_distance_from_start = observation[4]
    push_velocity = observation[5]

    reward = reward - abs(stretch_force_ext*(stretch_distance_from_start*1))
    reward = reward - abs(push_force_ext*(push_distance_from_start*1))

    if terminal:
        reward = reward + 200

    return reward

def load_trajectories(base_dir, env_step=5):
    all_observations = []
    all_actions = []
    all_rewards = []
    all_terminals = []

    for traj_dir in sorted(os.listdir(base_dir)):
        traj_path = os.path.join(base_dir, traj_dir)
        obs_buffer = load_and_preprocess_observation(os.path.join(traj_path, "observations.txt"), env_step=env_step)
        acts_buffer = load_and_preprocess_action(os.path.join(traj_path, "actions.txt"), env_step=env_step)
        terminals = load_and_preprocess_terminal(os.path.join(traj_path, "terminates.txt"), obs_buffer.shape[0], env_step=env_step)

         # calcuate reward
        rewards = np.zeros(obs_buffer.shape[0])
        for i in range(1, obs_buffer.shape[0]):
            rewards[i] = reward_function(obs_buffer[i, :], terminals[i])
        # save reward to a file
        np.savetxt(os.path.join(traj_path, "rewards.txt"), rewards, fmt='%.6f')
        print(f"Expected return for trajectory {traj_dir}: {np.sum(rewards)/rewards.shape[0]}")
        plot_force(obs_buffer, acts_buffer, traj_path)
        
        all_observations.append(obs_buffer)
        all_actions.append(acts_buffer)
        all_rewards.append(rewards)
        all_terminals.append(terminals)

    # Concatenate everything
    observations = np.concatenate(all_observations, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    terminals = np.concatenate(all_terminals, axis=0)

    return MDPDataset(observations, actions, rewards, terminals)

def plot_force(obs, act, traj_path):
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
    plt_file = os.path.join(traj_path, f"force_plot.png")
    plt.savefig(plt_file)

if __name__ == "__main__":
    base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/"
    dataset = load_trajectories(base_dir, env_step=5)
    print(f"Dataset loaded.")

    
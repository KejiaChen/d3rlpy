import numpy as np
import os
from d3rlpy.dataset import MDPDataset, EpisodeDataset
from fixing_detection_no_smoothing import FFAnalyzerRTSlide 
from return_functions import *
import json
from data_processing.utils.datasets_utils import Dataloader
# def success_threshold(criterion, threshold):

from torch.utils.data import Dataset, DataLoader
from build_episode_dataset import preprocess_observation, preprocess_action, find_stable_start_index, load_and_preprocess_fixing_result, load_and_preprocess_terminal
import torch

ONLINE_SOURCES = {"ForceControl":["f_ext", "dx", "f_ext_sensor", "ff", "x"]}

# SETUP_LABELS = {"R_L_H_1_C_L_1": 1,
#                 "R_L_H_1_C_L_2": 2,
#                 "R_L_H_1_C_S_2": 3,
#                 "R_M_L_1_C_L_1": 4,
#                 "R_M_L_1_C_L_2": 5,
#                 "R_M_L_1_C_S_2": 6,
#                 "R_M_M_1_C_L_1": 7,
#                 "R_M_M_1_C_L_2": 8,
#                 "R_M_M_1_C_S_2": 9,
#                 "W_S_M_1_C_L_1": 10,
#                 "W_S_M_1_C_L_2": 11,
#                 "W_S_M_1_C_S_2": 12,
#                 "W_S_M_1_C_S_1": 13}

def normalize_obs(obs):
    force_scale = 30.0
    distance_scale = 0.1
    velocity_scale = 0.2

    if obs.dim() == 2:
        obs = obs.clone()
        # Force: [0, 30] → scale to [0, 1]
        obs[:, 0] /= force_scale
        obs[:, 3] /= force_scale

        # Distance: normalize using global mean/std or empirical range
        # Estimate these from your dataset (preprocess)
        obs[:, 1] /= distance_scale  # stretch displacement
        obs[:, 4] /= distance_scale  # push displacement
        # # Velocity: estimate from data (e.g., max ~0.05 m/s)
        obs[:, 2] /= velocity_scale
        obs[:, 5] /= velocity_scale
    elif obs.dim() == 3:
        obs = obs.clone()
        # Force: [0, 30] → scale to [0, 1]
        obs[:, :, 0] /= force_scale  # stretch force
        obs[:, :, 3] /= force_scale  # push force
        # Distance: normalize using global mean/std or empirical range
        obs[:, :, 1] /= distance_scale  # stretch displacement
        obs[:, :, 4] /= distance_scale  # push displacement
        # Velocity: estimate from data (e.g., max ~0.05 m/s)
        obs[:, :, 2] /= velocity_scale
        obs[:, :, 5] /= velocity_scale
    else:
        raise ValueError(f"Expected obs to be 2D or 3D, got {obs.shape}")
    return obs

def unnormalize_obs(normalized_obs):
    force_scale = 30.0
    distance_scale = 0.1
    velocity_scale = 0.2

    if normalized_obs.dim() == 2:
        obs = normalized_obs.clone()
        # Force: [0, 1] → scale to [0, 30]
        obs[:, 0] *= force_scale
        obs[:, 3] *= force_scale
        # Distance: scale back using global mean/std or empirical range
        obs[:, 1] *= distance_scale  # stretch displacement
        obs[:, 4] *= distance_scale
        # Velocity: scale back using estimated max (e.g., 0.1 m/s)
        obs[:, 2] *= velocity_scale
        obs[:, 5] *= velocity_scale
    elif normalized_obs.dim() == 3:
        obs = normalized_obs.clone()
        # Force: [0, 1] → scale to [0, 30]
        obs[:, :, 0] *= force_scale
        obs[:, :, 3] *= force_scale
        # Distance: scale back using global mean/std or empirical range
        obs[:, :, 1] *= distance_scale  # stretch displacement
        obs[:, :, 4] *= distance_scale
        # Velocity: scale back using estimated max (e.g., 0.1 m/s)
        obs[:, :, 2] *= velocity_scale
        obs[:, :, 5] *= velocity_scale
    else:
        raise ValueError(f"Expected obs to be 2D or 3D, got {obs.shape}")
    return obs

def normalize_acts(acts):
    if acts.dim() == 2:
        acts = acts.clone()
        acts[:, 0] /= 30.0  # stretch force ff
        acts[:, 1] /= 30.0  # push force ff
    elif acts.dim() == 3:
        acts = acts.clone()
        acts[:, :, 0] /= 30.0
        acts[:, :, 1] /= 30.0
    else:
        raise ValueError(f"Expected acts to be 2D or 3D, got {acts.shape}")
    return acts

def unnormalize_acts(normalized_acts):
    if normalized_acts.dim() == 2:
        acts = normalized_acts.clone()
        acts[:, 0] *= 30.0  # stretch force ff
        acts[:, 1] *= 30.0  # push force ff
    elif normalized_acts.dim() == 3:
        acts = normalized_acts.clone()
        acts[:, :, 0] *= 30.0
        acts[:, :, 1] *= 30.0
    else:
        raise ValueError(f"Expected normalized_acts to be 2D or 3D, got {normalized_acts.shape}")
    return acts

def read_group_type(traj_dir, parameters):
    cable_id = parameters.get("cable_id", None)
    clip_id = parameters.get("clip_id", None)
    fixing_pose_name = parameters.get("fixing_pose_name", None)
    fixing_pose_id = fixing_pose_name.split("_")[-1] if fixing_pose_name else None
    if cable_id is None or clip_id is None:
        print(f"\033[91mSkipping {traj_dir} due to missing cable_id or clip_id.\033[0m")
        return None
    insertion_sign = load_insertion_sign(parameters)
    if insertion_sign is None:
        print(f"\033[91mSkipping {traj_dir} due to missing insertion_sign.\033[0m")
        return None
    
    # if f"{cable_id}_{clip_id}" in SETUP_LABELS:
    #     setup_label = SETUP_LABELS[f"{cable_id}_{clip_id}"]
    # else:
    #     # raise ValueError(f"Unknown setup label for {cable_id}_{clip_id}")
    #     print(f"\033[93mUnknown setup label for {cable_id}_{clip_id}. Assigning default label -1.\033[0m")
    #     setup_label = -1
    # parameters["setup_label"] = setup_label
    # json.dump(parameters, open(os.path.join(mios_traj_dir, "parameters.json"), 'w'), indent=4)

    group_key = f"{cable_id}_{clip_id}_pose_{fixing_pose_id}_{insertion_sign}" if fixing_pose_id else f"{cable_id}_{clip_id}"

    return group_key, cable_id, clip_id, fixing_pose_id, insertion_sign

def compute_group_typed_statistics(base_dir, env_step=5):
    grouped_refs = {}
    grouped_trajs = {}

    # stretch_forces, stretch_effort, stretch_energy, push_forces, push_effort, push_energy, deformation_values, durations = [], [], [], [], [], [], [], []
    # group_trajs = []
    dt = 0.001*env_step  # each env_step is 1 ms
    
    traj_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    for idx, traj_dir in enumerate(traj_dirs):
        traj_path = os.path.join(base_dir, traj_dir)
        if not os.path.isdir(traj_path):
            continue
        
        # load parameters from the mios dir with the largest counter
        subfolders = [int(d) for d in os.listdir(traj_path) if os.path.isdir(os.path.join(traj_path, d)) and d.isdigit()]
        latest_mios_counter = max(subfolders)
        mios_traj_dir = os.path.join(traj_path, str(latest_mios_counter))

        parameters = json.load(open(os.path.join(mios_traj_dir, "parameters.json"), 'r'))
        group_key, cable_id, clip_id, fixing_pose_id, insertion_sign = read_group_type(traj_dir, parameters)
        if group_key is None:
            continue
        
        if group_key not in grouped_trajs:
            grouped_trajs[group_key] = {
                "stretch_force_list": [],
                "stretch_effort_list": [],
                "stretch_energy_list": [],
                "push_force_list": [],
                "push_effort_list": [],
                "push_energy_list": [],
                "deformation_list": [],
                "duration_list": [],
            }
            grouped_refs[group_key] = {"dirs": [],
                                        "refs": {}}

        # obs_raw = np.loadtxt(os.path.join(traj_path, "observations.txt"))
        # obs = preprocess_observation(obs_raw, env_step=env_step)
        # if obs.shape[0] == 0:
        #     continue  # skip empty

        obs, acts, _, terminals, _, _ = load_and_evaluate_one_episode(traj_dir, traj_path, None, None, env_step)
        if obs is None:
            continue  # skip empty

        terminal_index = min(np.where(terminals == 1)[0])

        grouped_trajs[group_key]["stretch_force_list"].append(np.max((obs[:terminal_index, 0])))
        grouped_trajs[group_key]["stretch_effort_list"].append(np.sum((obs[:terminal_index, 0])) * dt)
        grouped_trajs[group_key]["stretch_energy_list"].append(utils_integral(obs[:terminal_index, 0], obs[:terminal_index, 1]))
        grouped_trajs[group_key]["push_force_list"].append(np.max((obs[:terminal_index, 3])))
        grouped_trajs[group_key]["push_effort_list"].append(np.sum((obs[:terminal_index, 3])) * dt)
        grouped_trajs[group_key]["push_energy_list"].append(utils_integral(obs[:terminal_index, 3], obs[:terminal_index, 4]))
        grouped_trajs[group_key]["deformation_list"].append(np.max((obs[:terminal_index, 4])))
        grouped_trajs[group_key]["duration_list"].append(terminal_index)
        grouped_refs[group_key]["dirs"].append(int(traj_dir))

    for group_key, group_traj in grouped_trajs.items():
        ref_values = {}
        for statistics_list, values in group_traj.items():
            if len(values) == 0:
                raise ValueError(f"No values collected for {statistics_list} in group {group_key}")
            statistics_key = statistics_list.replace("_list", "")
            arr = np.asarray(values, dtype=np.float64)
            ref_values[statistics_key] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max())
            }
        grouped_refs[group_key]["refs"] = ref_values

        print(f"\033[92mComputed {len(grouped_refs[group_key]['dirs'])} reference values from demos of config type {group_key}: {ref_values}\033[0m")

    return grouped_refs

def load_and_evaluate_episodes(base_dir, criterion_fn=None, criterion_reference:dict=None, env_step=5, plot=True):
    ep_returns = []

    for traj_dir in sorted(os.listdir(base_dir)):
        traj_path = os.path.join(base_dir, traj_dir)
        if not os.path.isdir(traj_path):  # Skip files like "episode_dataset.npz"
            continue
        obs_buffer, acts_buffer, ep_return, terminals_buffer, fixing_success, insertion_sign = load_and_evaluate_one_episode(traj_dir, traj_path, criterion_fn, criterion_reference, env_step, plot)
        ep_returns.append(ep_return)

    return ep_returns

def load_and_evaluate_one_episode(traj_dir, traj_path, criterion_fn=None, criterion_reference:dict=None, env_step=5, plot=False):    
    obs_buffer_raw = np.loadtxt(os.path.join(traj_path, "observations.txt"))  # the last column is supposed to be the terminal, but was not stored correctly in the past
    acts_buffer_raw = np.loadtxt(os.path.join(traj_path, "actions.txt"))
    # terminals_buffer_raw = obs_buffer_raw[:, -1].astype(bool)  # last column is terminal, but was not stored correctly in the past

    # print(f"Loaded fixing observation with {obs_buffer_raw.shape[0]} steps.")

    # load parameters from the mios dir with the largest counter
    subfolders = [int(d) for d in os.listdir(traj_path) if os.path.isdir(os.path.join(traj_path, d)) and d.isdigit()]
    latest_mios_counter = max(subfolders)
    mios_traj_dir = os.path.join(traj_path, str(latest_mios_counter))

    if not os.path.isdir(mios_traj_dir):
        raise ValueError(f"Expected mios traj dir at {mios_traj_dir}, but not found.")
    else:
        fixing_success = load_and_preprocess_fixing_result(os.path.join(mios_traj_dir, "fixing_result.json"))
        parameters_path = os.path.join(mios_traj_dir, "parameters.json")
        parameters = json.load(open(parameters_path, 'r'))

        group_key, cable_id, clip_id, fixing_pose_id, insertion_sign = read_group_type(traj_dir, parameters)
        if group_key is None:
            raise ValueError(f"Cannot read group type for {traj_dir} in {parameters_path}")
        print(f"Evaluating trajectory {traj_dir} of group type {group_key}")
        
        # setup_label = load_setup_label(parameters)
        # if setup_label is None:
        #     raise ValueError(f"Missing setup_label in {parameters_path}")
        
    fixing_terminal_index = -1
    if fixing_success:
        acts_mask = np.all(acts_buffer_raw == 0, axis=1)      # True for rows that are all zeros
        idxs = np.nonzero(acts_mask)[0]
        fixing_terminal_index = idxs[0] if idxs.size else -1  # first all zero index, -1 means "not found"
    terminals_buffer_raw = load_and_preprocess_terminal(traj_path, obs_buffer_raw, acts_buffer_raw, fixing_terminal_index)

    # load from policy log and downsample
    obs_buffer_downsampled = preprocess_observation(obs_buffer_raw, env_step=env_step)
    acts_buffer_downsampled  = preprocess_action(acts_buffer_raw, env_step=env_step)
    terminals_downsampled = np.concatenate([
        terminals_buffer_raw[::env_step],
        [terminals_buffer_raw[-1]] if (terminals_buffer_raw.shape[0]) % env_step != 0 else np.empty(0, dtype=terminals_buffer_raw.dtype)
    ])

    if fixing_success:
        terminal_index_downsampled = np.where(terminals_downsampled == True)[0][0] if np.any(terminals_downsampled) else terminals_downsampled.shape[0] - 1
    loading_end_index_downsampled = obs_buffer_downsampled.shape[0] - 1

    # print(f"Trajectory {traj_dir} has {obs_buffer_downsampled.shape[0]} steps after downsampling.")

    # find start index where forces only rise
    start_index = find_stable_start_index(obs_buffer_downsampled)
    obs_buffer = obs_buffer_downsampled[start_index:]
    acts_buffer = acts_buffer_downsampled[start_index:]
    terminals_buffer = terminals_downsampled[start_index:]
    terminal_index = terminal_index_downsampled - start_index
    ending_index = loading_end_index_downsampled - start_index

    print(f"Trajectory {traj_dir} starts from index {start_index} after stable start index detection")
    print(f"Trajectory {traj_dir} has {obs_buffer.shape[0]} steps after trimming.")

    # load reference for this episode
    ep_return = None
    if criterion_fn is not None and criterion_reference is not None:
        ep_ref_values = {"ref_stretch_force": 7.0, 
                        "ref_stretch_effort": 21.0, 
                        "ref_stretch_energy": 0.02, 
                        "ref_push_force": 7.0, 
                        "ref_push_effort": 7.0, 
                        "ref_push_energy": 0.1, 
                        "ref_deformation": 0.01, 
                        "ref_duration": 1000, 
                        "dt": 0.005}
        group_ref = criterion_reference.get(group_key, None)
        if group_ref is None:
            print(f"\033[93mWarning: No reference values found for group {group_key}. Using default references.\033[0m")
        else:
            for ref_key, _ in ep_ref_values.items():
                stats_key = ref_key.replace("ref_", "")
                if stats_key in group_ref["refs"]:
                    ep_ref_values[ref_key] = group_ref["refs"][stats_key]["mean"]
        # calculate reward only BEFORE the terminal index
    
        ep_return = criterion_fn(obs_buffer[:terminal_index], **ep_ref_values)
        # save reward to a file
        np.savetxt(os.path.join(traj_path, "return.txt"), np.array([ep_return ]), fmt='%.6f')
        print(f"\033[93mExpected return for trajectory {traj_dir}: {ep_return} with {obs_buffer.shape[0]} steps.\033[0m")

    # push or pull for insertion
    insertion_sign = +1 if insertion_sign.startswith("+") else -1
    obs_buffer[:, 3:] = insertion_sign * obs_buffer[:, 3:]  # push force, distance, velocity
    acts_buffer[:, 1] = insertion_sign * acts_buffer[:, 1]  # push force

    # plotting
    if plot:
        plot_force(obs_buffer, acts_buffer, traj_dir, traj_path, terminal_index, ending_index)
    return obs_buffer, acts_buffer, ep_return, terminals_buffer, fixing_success, insertion_sign

# def load_setup_label(parameters):
#     # parameters = json.load(open(parameters_path, 'r'))
#     return parameters.get("setup_label", None)

def load_insertion_sign(parameters):
    insertion_sign = "+1"
    insertion_force = parameters["p2"]["f_push"]

    # ensure insertion_force is array-like (handle 0-d scalars)
    insertion_force = np.asarray(insertion_force)
    insertion_force_1d = np.atleast_1d(insertion_force)
    non_zero_mask = insertion_force_1d != 0
    non_zero_insertion_force = insertion_force_1d[non_zero_mask]
    non_zero_insertion_index = np.nonzero(non_zero_mask)[0]
    if non_zero_insertion_force.size == 0:
        insertion_sign = None
    elif non_zero_insertion_force[0] < 0:
        insertion_sign = "-"+str(non_zero_insertion_index[0])
    else:
        insertion_sign = "+"+str(non_zero_insertion_index[0])

    return insertion_sign

def plot_force(obs, act, traj_dir, traj_path, terminal_index, ending_index):
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
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Plot distance vs time
    axs[0].plot(time, obs[:, 1], label=f'Stretch Distance', color='orange')
    axs[0].plot(time, obs[:, 4], label=f'Push Distance', color='blue')
    axs[0].axvline(x=terminal_index, color='red', linestyle='-.', label='Finish')
    axs[0].axvline(x=ending_index, color='black', linestyle='--', label='End of Dataset')
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Distance from Start (m)')
    axs[0].set_title(f'Distance vs Time')
    axs[0].legend()
    axs[0].grid()

    # Plot velocity vs time
    axs[1].plot(time, obs[:, 2], label=f'Stretch Velocity', color='orange')
    axs[1].plot(time, obs[:, 5], label=f'Push Velocity', color='blue')
    axs[1].axvline(x=terminal_index, color='red', linestyle='-.', label='Finish')
    axs[1].axvline(x=ending_index, color='black', linestyle='--', label='End of Dataset')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_title(f'Velocity vs Time')
    axs[1].legend()
    axs[1].grid()

    # Plot force vs time
    axs[2].plot(time, obs[:, 0], label=f'Ext Stretch Force', color='orange')
    axs[2].plot(time, act[:, 0], label=f'FF Stretch Force', linestyle='--', color='orange')
    axs[2].plot(time, obs[:, 3], label=f'Ext Push Force', color='blue')
    axs[2].plot(time, act[:, 1], label=f'FF Push Force', linestyle='--', color='blue')
    axs[2].axvline(x=terminal_index, color='red', linestyle='-.', label='Finish')
    axs[2].axvline(x=ending_index, color='black', linestyle='--', label='End of Dataset')
    axs[2].set_xlabel('Time (ms)')
    axs[2].set_ylabel('Force (N)')
    axs[2].set_title(f'Force vs Time')
    axs[2].legend()
    axs[2].grid()

    # Adjust layout and show the plot
    plt.tight_layout()
    # plt.show()
    plt_file = os.path.join(traj_path, f"{traj_dir}_force_plot.png")
    plt.savefig(plt_file)
    plt.close(fig)


def effort_and_energy_based_criterion_function(obs, include_deformation=True, ref_stretch_force=7.0, ref_stretch_effort=21.0, ref_stretch_energy=0.02, ref_push_force=7.0, ref_push_effort=7.0, ref_push_energy=0.1, ref_deformation=0.01, ref_duration=1000, dt=0.005):
    # 1. Total force (integrated effort)
    stretch_effort = np.sum(obs[:, 0]) * dt
    push_effort = np.sum(obs[:, 3]) * dt
    stretch_energy = utils_integral(obs[:, 0], obs[:, 1]) # integral of f_s *delta_x_s
    push_energy = utils_integral(obs[:, 3], obs[:, 4])
    total_energy = stretch_energy + push_energy

    # 2. Max deformation and duration
    deformation = np.max(np.abs(obs[:, 4]))
    duration = obs.shape[0]

    # 3. Scores
    ref_total_energy = ref_stretch_energy + ref_push_energy

    print(f"Stretch effort: {stretch_effort}, Push effort: {push_effort}, Total energy: {total_energy}")

    stretch_effort_score = 1.0 - (stretch_effort - ref_stretch_effort) / ref_stretch_effort
    stretch_energy_score = 1.0 - (stretch_energy - ref_stretch_energy) / ref_stretch_energy
    push_effort_score    = 1.0 - (push_effort - ref_push_effort) / ref_push_effort
    push_energy_score    = 1.0 - (push_energy - ref_push_energy) / ref_push_energy
    total_energy_score = 1.0 - (total_energy - ref_total_energy) / ref_total_energy
    # deform_score  = 1.0 - (deformation - ref_deformation) / ref_deformation
    # time_score    = 1.0 - (duration - ref_duration) / ref_duration

    # 4. Clip to [0, 1]
    # stretch_score = np.clip(stretch_score, 0.0, 1.0)
    # push_score = np.clip(push_score, 0.0, 1.0)
    # deform_score = np.clip(deform_score, 0.0, 1.0)
    # time_score = np.clip(time_score, 0.0, 1.0)

    # 5. Final return
    if include_deformation:
        ep_return = 15 * np.max(stretch_effort_score, 0) + 15 * np.max(push_effort_score, 0) + 10 * np.max(total_energy_score, 0)
    else:
        ep_return = 20 * np.max(stretch_effort_score, 0) + 20 * np.max(push_effort_score, 0)

    if ep_return < 0:
        ep_return = 0.0
    return ep_return


def effort_and_energy_based_exponential_criterion_function(obs, include_deformation=True, ref_stretch_force=7.0, ref_stretch_effort=21.0, ref_stretch_energy=0.02, ref_push_force=7.0, ref_push_effort=7.0, ref_push_energy=0.1, ref_deformation=0.01, ref_duration=1000, dt=0.005):
    stretch_force = obs[:, 0]
    stretch_displacement = obs[:, 1]
    stretch_velocity = obs[:, 2]
    push_force = obs[:, 3]
    push_displacement = obs[:, 4]
    push_velocity = obs[:, 5]
    
    # 1. Compute efforts and energies
    stretch_effort = np.sum(obs[:, 0]) * dt
    push_effort = np.sum(obs[:, 3]) * dt
    stretch_energy = utils_integral(obs[:, 0], obs[:, 1])
    push_energy = utils_integral(obs[:, 3], obs[:, 4])
    total_energy = stretch_energy + push_energy

    # 2. Deformation and duration
    deformation = np.max(obs[:, 4])
    duration = obs.shape[0]

    print(f"Stretch effort: {stretch_effort}, Push effort: {push_effort}, Stretch energy: {stretch_energy}, Push energy: {push_energy}, Total energy: {total_energy}")

    # 3. Exponential scores (bounded in (0,1])
    stretch_effort_score = np.exp(-(stretch_effort / ref_stretch_effort))
    push_effort_score    = np.exp(-(push_effort / ref_push_effort))
    total_energy_score   = np.exp(-(total_energy / (ref_stretch_energy + ref_push_energy)))

    # Optional scores if you want them later
    # stretch_energy_score = np.exp(-(stretch_energy / ref_stretch_energy))
    # push_energy_score    = np.exp(-(push_energy / ref_push_energy))
    # deform_score         = np.exp(-(deformation / ref_deformation))
    # time_score           = np.exp(-(duration / ref_duration))

    # 4. Weighted final return (bounded above)
    if include_deformation:
        ep_return = (
            30 * stretch_effort_score +
            30 * push_effort_score +
            20 * total_energy_score
        )
    else:
        ep_return = (
            40 * stretch_effort_score +
            40 * push_effort_score
        )

    # Since all scores are > 0, ep_return never goes negative
    return float(ep_return)

if __name__ == "__main__":
    # trained_with_encoder_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/evaluation_trained_with_encoder"
    # # dataset = load_trajectories(base_dir, max_based_return_function, env_step=5)
    # trained_ref_values = compute_group_typed_statistics(trained_with_encoder_dir, env_step=5)
    # with open(os.path.join(trained_with_encoder_dir, "criterions.json"), 'w') as f:
    #     json.dump(trained_ref_values, f, indent=4)

    random_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/evaluation_random"
    random_ref_values = compute_group_typed_statistics(random_dir, env_step=5)
    with open(os.path.join(random_dir, "criterions.json"), 'w') as f:
        json.dump(random_ref_values, f, indent=4)

    normal_returns = load_and_evaluate_episodes(random_dir, effort_and_energy_based_criterion_function, random_ref_values, env_step=5, plot=False)

    exp_returns = load_and_evaluate_episodes(random_dir, effort_and_energy_based_exponential_criterion_function, random_ref_values, env_step=5, plot=False)

    # plot two returns for comparison
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(normal_returns, label='Normal Return', marker='o')
    plt.plot(exp_returns, label='Exponential Return', marker='x')
    plt.xlabel('Episode Index')
    plt.ylabel('Return')
    plt.title('Comparison of Return Functions')
    plt.legend()
    plt.grid()
    plt.show()

    print("Evaluation completed.")
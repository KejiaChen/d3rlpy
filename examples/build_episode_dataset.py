import numpy as np
import os
from d3rlpy.dataset import MDPDataset, EpisodeDataset
from fixing_detection_no_smoothing import FFAnalyzerRTSlide 
from return_functions import *
import json
from data_processing.utils.datasets_utils import Dataloader
# def success_threshold(criterion, threshold):

from torch.utils.data import Dataset, DataLoader
import torch

ONLINE_SOURCES = {"ForceControl":["f_ext", "dx", "f_ext_sensor", "ff", "x"]}

SETUP_LABELS = {"R_L_H_1_C_L_1": 1,
                "R_L_H_1_C_L_2": 2,
                "R_L_H_1_C_S_2": 3,
                "R_M_L_1_C_L_1": 4,
                "R_M_L_1_C_L_2": 5,
                "R_M_L_1_C_S_2": 6,
                "R_M_M_1_C_L_1": 7,
                "R_M_M_1_C_L_2": 8,
                "R_M_M_1_C_S_2": 9,
                "W_S_M_1_C_L_1": 10,
                "W_S_M_1_C_L_2": 11,
                "W_S_M_1_C_S_2": 12}

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

def compute_average_reference_values(group_references):
    stretch_forces, stretch_effort, stretch_energy, push_forces, push_effort, push_energy, deformation_values, durations = [], [], [], [], [], [], [], []
    for key, ref_values in group_references.items():
        stretch_forces.append(ref_values["ref_stretch_force"])
        stretch_effort.append(ref_values["ref_stretch_effort"])
        stretch_energy.append(ref_values["ref_stretch_energy"])
        push_forces.append(ref_values["ref_push_force"])
        push_effort.append(ref_values["ref_push_effort"])
        push_energy.append(ref_values["ref_push_energy"])
        deformation_values.append(ref_values["ref_deformation"])
        durations.append(ref_values["ref_duration"])

    average_ref_values = {
        "ref_stretch_force": np.mean(stretch_forces),
        "ref_stretch_effort": np.mean(stretch_effort),
        "ref_stretch_energy": np.mean(stretch_energy),
        "ref_push_force": np.mean(push_forces),
        "ref_push_effort": np.mean(push_effort),
        "ref_push_energy": np.mean(push_energy),
        "ref_deformation": np.mean(deformation_values),
        "ref_duration": np.mean(durations),
        "dt": group_references[list(group_references.keys())[0]]["dt"]  # assuming all groups have the same dt
    }

    return average_ref_values

def compute_group_typed_reference_values(base_dir, env_step=5):
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
        
        for mios_dir in sorted(os.listdir(traj_path)):
            mios_traj_dir = os.path.join(traj_path, mios_dir)
            if not os.path.isdir(mios_traj_dir):
                continue

            parameters = json.load(open(os.path.join(mios_traj_dir, "parameters.json"), 'r'))
            cable_id = parameters.get("cable_id", None)
            clip_id = parameters.get("clip_id", None)
            fixing_pose_name = parameters.get("fixing_pose_name", None)
            fixing_pose_id = fixing_pose_name.split("_")[-1] if fixing_pose_name else None
            if cable_id is None or clip_id is None:
                print(f"\033[91mSkipping {mios_traj_dir} due to missing cable_id or clip_id.\033[0m")
                continue

            if f"{cable_id}_{clip_id}" in SETUP_LABELS:
                setup_label = SETUP_LABELS[f"{cable_id}_{clip_id}"]
            else:
                raise ValueError(f"Unknown setup label for {cable_id}_{clip_id}")
            parameters["setup_label"] = setup_label
            json.dump(parameters, open(os.path.join(mios_traj_dir, "parameters.json"), 'w'), indent=4)

            group_key = f"{cable_id}_{clip_id}_pose_{fixing_pose_id}" if fixing_pose_id else f"{cable_id}_{clip_id}"

            if group_key not in grouped_trajs:
                grouped_trajs[group_key] = {
                    "stretch_forces": [],
                    "stretch_effort": [],
                    "stretch_energy": [],
                    "push_forces": [],
                    "push_effort": [],
                    "push_energy": [],
                    "deformation_values": [],
                    "durations": [],
                }
                grouped_refs[group_key] = {"dirs": [],
                                           "refs": {}}

            obs_raw = np.loadtxt(os.path.join(traj_path, "observations.txt"))
            obs = preprocess_observation(obs_raw, env_step=env_step)

            if obs.shape[0] == 0:
                continue  # skip empty

            grouped_trajs[group_key]["stretch_forces"].append(np.max(np.abs(obs[:, 0])))
            grouped_trajs[group_key]["stretch_effort"].append(np.sum(np.abs(obs[:, 0])) * dt)
            grouped_trajs[group_key]["stretch_energy"].append(utils_integral(obs[:, 0], obs[:, 1]))
            grouped_trajs[group_key]["push_forces"].append(np.max(np.abs(obs[:, 3])))
            grouped_trajs[group_key]["push_effort"].append(np.sum(np.abs(obs[:, 3])) * dt)
            grouped_trajs[group_key]["push_energy"].append(utils_integral(obs[:, 3], obs[:, 4]))
            grouped_trajs[group_key]["deformation_values"].append(np.max(np.abs(obs[:, 4])))
            grouped_trajs[group_key]["durations"].append(obs.shape[0])

            grouped_refs[group_key]["dirs"].append(int(traj_dir))

    # Once we hit a group boundary or end of data, save the group's reference stats
    for group_key, group_traj in grouped_trajs.items():
        ref_values = {
            "ref_stretch_force": np.mean(group_traj["stretch_forces"]),
            "ref_stretch_effort": np.mean(group_traj["stretch_effort"]),
            "ref_stretch_energy": np.mean(group_traj["stretch_energy"]),
            "ref_push_force": np.mean(group_traj["push_forces"]),
            "ref_push_effort": np.mean(group_traj["push_effort"]),
            "ref_push_energy": np.mean(group_traj["push_energy"]),
            "ref_deformation": np.mean(group_traj["deformation_values"]),
            "ref_duration": np.mean(group_traj["durations"]),
            "dt": dt
        }

        grouped_refs[group_key]["refs"] = ref_values

        print(f"\033[92mComputed {len(grouped_refs[group_key]['dirs'])} reference values from demos of config type {group_key}: {ref_values}\033[0m")

    return grouped_refs


def compute_group_sized_reference_values(base_dir, env_step=5, group_size=30):
    grouped_refs = {}
    group_key = 0

    stretch_forces, stretch_effort, stretch_energy, push_forces, push_effort, push_energy, deformation_values, durations = [], [], [], [], [], [], [], []
    group_trajs = []
    dt = 0.001*env_step  # each env_step is 1 ms
    
    traj_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

    for idx, traj_dir in enumerate(traj_dirs):
        traj_path = os.path.join(base_dir, traj_dir)
        if not os.path.isdir(traj_path):
            continue

        obs_raw = np.loadtxt(os.path.join(traj_path, "observations.txt"))
        obs = preprocess_observation(obs_raw, env_step=env_step)

        if obs.shape[0] == 0:
            continue  # skip empty

        stretch_forces.append(np.max(np.abs(obs[:, 0])))
        stretch_effort.append(np.sum(np.abs(obs[:, 0])) * dt) 
        stretch_energy.append(utils_integral(obs[:, 0], obs[:, 1]))
        push_forces.append(np.max(np.abs(obs[:, 3])))
        push_effort.append(np.sum(np.abs(obs[:, 3])) * dt)
        push_energy.append(utils_integral(obs[:, 3], obs[:, 4]))
        deformation_values.append(np.max(np.abs(obs[:, 4])))
        durations.append(obs.shape[0])

        group_trajs.append(traj_dir)

        # Once we hit a group boundary or end of data, save the group's reference stats
        if len(group_trajs) >= 2:
            if (int(group_trajs[-1])-int(group_trajs[0])+1) % group_size == 0 or (idx + 1) == len(traj_dirs): # group_trajs might not be continuous, as we delete some trajectories manually
                ref_values = {
                    "ref_stretch_force": np.mean(stretch_forces),
                    "ref_stretch_effort": np.mean(stretch_effort),
                    "ref_stretch_energy": np.mean(stretch_energy),
                    "ref_push_force": np.mean(push_forces),
                    "ref_push_effort": np.mean(push_effort),
                    "ref_push_energy": np.mean(push_energy),
                    "ref_deformation": np.mean(deformation_values),
                    "ref_duration": np.mean(durations),
                    "dt": dt
                }

                grouped_refs[group_key] = {"dirs": [], "refs": {}}
                grouped_refs[group_key]["dirs"] = group_trajs
                grouped_refs[group_key]["refs"] = ref_values

                print(f"\033[92mComputed {len(grouped_refs)} reference values from demos {group_trajs[0]} to {group_trajs[-1]}: {ref_values}\033[0m")

                # Reset for next group
                stretch_forces.clear()
                stretch_effort.clear()
                stretch_energy.clear()
                push_forces.clear()
                push_effort.clear()
                push_energy.clear()
                deformation_values.clear()
                durations.clear()

                group_trajs.clear()   

                group_key += 1

    return grouped_refs


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
        # TODO@KejiaChen: the analyzer is not as accurate as its cpp version
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

def load_and_preprocess_fixing_result(result_path):
    result = json.load(open(result_path, 'r'))
    if result == 100:
        return True
    else:
        return False

def load_setup_label(parameters_path):
    parameters = json.load(open(parameters_path, 'r'))
    setup_label = parameters.get("setup_label", None)
    if setup_label is None:
        raise ValueError(f"Missing setup_label in {parameters_path}")
    return setup_label

def load_and_preprocess_terminal(traj_path, obs_buffer, acts_buffer, fixing_index=-1):
    terminal_file = os.path.join(traj_path, "terminal.txt")
    terminal_buffer = np.zeros(obs_buffer.shape[0], dtype=bool)
    if fixing_index != -1:
        terminal_buffer[fixing_index:] = True
        print(f"Fixing success at step {fixing_index}, setting terminal from this step.")
    else:
        terminal_buffer[-1] = True  # last step is always terminal
        print(f"Fixing did not succeed, setting terminal at the end of the trajectory.")
    
    with open(terminal_file, 'w') as f:
        np.savetxt(f, terminal_buffer, fmt='%.6f')

    return terminal_buffer

# def load_and_preprocess_terminal(traj_path, obs_buffer, acts_buffer):
#     terminal_file = os.path.join(traj_path, "terminal.txt")
#     # if not os.path.exists(terminal_file):
#     start_check_index = 30 # due to communication latency, first few steps are not reliable
#     terminal_buffer = np.zeros(obs_buffer.shape[0], dtype=bool)
#     terminals_from_start, terminate_index_from_start = check_success(traj_path, obs_buffer[start_check_index:], acts_buffer[start_check_index:])
#     terminal_buffer[start_check_index:] = terminals_from_start
#     # write terminal to terminal file
#     with open(terminal_file, 'w') as f:
#         np.savetxt(f, terminal_buffer, fmt='%.6f')
#     # else:
#     #     terminal_buffer = np.loadtxt(terminal_file)
#     return terminal_buffer, terminate_index_from_start + start_check_index

def find_stable_start_index(obs_buffer, window_size=5):
    """
    Returns the index after the initial dip in both dim 0 and dim 3 where values start increasing.
    Uses a simple windowed slope check to avoid spurious noise.
    """
    dim0 = obs_buffer[:, 0]  # stretch_force_ext
    dim3 = obs_buffer[:, 3]  # push_force_ext

    def find_increasing_start(signal):
        # smooth using a moving average
        smoothed = np.convolve(signal, np.ones(window_size)/window_size, mode='valid')
        for i in range(1, len(smoothed) - window_size):
            if all(smoothed[i + j] > smoothed[i + j - 1] for j in range(1, window_size)):
                return i + window_size  # account for the convolution lag
        return 0  # fallback if no increasing pattern found

    idx0 = find_increasing_start(dim0)
    idx3 = find_increasing_start(dim3)

    return max(idx0, idx3)

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

def load_post_observation_and_action(mios_traj_dir, obs_buffer_raw, acts_buffer_raw, fixing_success):
    data_loader = Dataloader(disabe_filter=True) # for real-time analysis, filters have to be disabled
    # start to contact
    sorted_record = data_loader.load_one_trail(sources=ONLINE_SOURCES,
                                            data_folder=mios_traj_dir,
                                            full_range=False, 
                                            start="finished", 
                                            end="ended")
    # load post fixing sensing data
    post_finish_steps = 1000
    # end_index = -(10*1000 - post_finish_steps)  # 10 seconds before the end

    # pushing
    external_force_y = sorted_record["ForceControl"]['Force']["proj"]
    external_force_y_sensor = sorted_record["ForceControl"]['Force_sensor']["proj"]
    feedforward_force_y = np.zeros_like(external_force_y_sensor)
    linear_velocity_y = sorted_record["ForceControl"]['LinearVelocity']["proj"]
    distance_y = sorted_record["ForceControl"]['Distance']["proj"]

    # stretching
    external_force_x = sorted_record["ForceControl"]['Force']["x"]
    external_force_x_sensor = sorted_record["ForceControl"]['Force_sensor']["x"]
    feedforward_force_x = np.zeros_like(external_force_x_sensor)
    linear_velocity_x = sorted_record["ForceControl"]['LinearVelocity']["x"]
    # TODO: fake distance x because we don't have it in the mios log
    # distance_x = np.ones_like(distance_y) * obs_buffer_raw[-1, 1]
    distance_x = sorted_record["ForceControl"]['Distance']["x"]

    post_obs_buffer_raw = np.stack([
        external_force_x_sensor[:post_finish_steps],
        distance_x[:post_finish_steps],
        linear_velocity_x[:post_finish_steps],
        external_force_y_sensor[:post_finish_steps],
        distance_y[:post_finish_steps],
        linear_velocity_y[:post_finish_steps],
    ], axis=1)

    post_acts_buffer_raw = np.stack([
        feedforward_force_x[:post_finish_steps],
        feedforward_force_y[:post_finish_steps],
    ], axis=1)

    # load fixing result
    fixing_terminal_index = -1
    if fixing_success:
        fixing_terminal_index = data_loader.get_timestamps()["finished"] - data_loader.get_timestamps()["sensed"]
        loading_end_index = fixing_terminal_index + post_finish_steps

    return post_obs_buffer_raw, post_acts_buffer_raw, fixing_terminal_index, loading_end_index

def load_one_episode(traj_path, env_step=5, post_sensing=True, include_deformation_in_return=True, return_fn=None, return_kwargs=None):
    base_dir = os.path.dirname(traj_path)
    traj_dir = os.path.basename(os.path.normpath(traj_path))

    obs_buffer_raw = np.loadtxt(os.path.join(traj_path, "observations.txt"))  # the last column is supposed to be the terminal, but was not stored correctly in the past
    acts_buffer_raw = np.loadtxt(os.path.join(traj_path, "actions.txt"))
    terminals_buffer_raw = obs_buffer_raw[:, -1].astype(bool)  # last column is terminal, but was not stored correctly in the past

    print(f"Loaded fixing observation with {obs_buffer_raw.shape[0]} steps.")

    for mios_dir in sorted(os.listdir(traj_path)): # load from raw mios logs when post sensing is not stored correctly in policy log
        mios_traj_dir = os.path.join(traj_path, mios_dir)
        if not os.path.isdir(mios_traj_dir):
            continue
        else:
            fixing_success = load_and_preprocess_fixing_result(os.path.join(mios_traj_dir, "fixing_result.json"))
            setup_label = load_setup_label(os.path.join(mios_traj_dir, "parameters.json"))

    #         if post_sensing:
    #             post_obs_buffer_raw, post_acts_buffer_raw, fixing_terminal_index, loading_end_index = load_post_observation_and_action(mios_traj_dir, obs_buffer_raw, acts_buffer_raw, fixing_success)
    #             # if fixing_terminal_index != -1:
    #             #     fixing_terminal_index = fixing_terminal_index - 30 # Assuming 10 steps were used for communication latency

    #             obs_buffer_raw = np.concatenate([obs_buffer_raw[:, :6], post_obs_buffer_raw], axis=0)
    #             acts_buffer_raw = np.concatenate([acts_buffer_raw, post_acts_buffer_raw], axis=0)
    #             # pad terminals_raw to match the new obs_buffer_raw length with the last terminal
    #             terminals_raw = np.zeros(obs_buffer_raw.shape[0], dtype=bool)
    #             terminals_raw[:terminals_buffer_raw.shape[0]] = terminals_buffer_raw

    #             # terminals_raw = load_and_preprocess_terminal(traj_path, obs_buffer_raw, acts_buffer_raw, fixing_terminal_index)
    #         else:
    #             terminals_raw = terminals_buffer_raw
    #             terminals_raw = terminals_raw.reshape(-1)



    # load from policy log and downsample
    obs_buffer_downsampled = preprocess_observation(obs_buffer_raw, env_step=env_step)
    acts_buffer_downsampled  = preprocess_action(acts_buffer_raw, env_step=env_step)
    terminals_downsampled = np.concatenate([
        terminals_buffer_raw[::env_step],
        [terminals_buffer_raw[-1]] if (terminals_buffer_raw.shape[0]) % env_step != 0 else np.empty(0, dtype=terminals_buffer_raw.dtype)
    ])
    # if post_sensing:
    #     terminal_index_downsampled = fixing_terminal_index // env_step
    #     loading_end_index_downsampled = loading_end_index // env_step
    # else:
    #     terminal_index_downsampled = np.where(terminals_downsampled)[0][0] if np.any(terminals_downsampled) else terminals_downsampled.shape[0] - 1
    #     loading_end_index_downsampled = obs_buffer_downsampled.shape[0] - 1

    if fixing_success:
        terminal_index_downsampled = np.where(terminals_downsampled == True)[0][0] if np.any(terminals_downsampled) else terminals_downsampled.shape[0] - 1
    loading_end_index_downsampled = obs_buffer_downsampled.shape[0] - 1

    print(f"Trajectory {traj_dir} has {obs_buffer_downsampled.shape[0]} steps after downsampling.")

    # find start index where forces only rise
    start_index = find_stable_start_index(obs_buffer_downsampled)
    obs_buffer = obs_buffer_downsampled[start_index:]
    acts_buffer = acts_buffer_downsampled[start_index:]
    terminals_buffer = terminals_downsampled[start_index:]
    terminal_index = terminal_index_downsampled - start_index
    ending_index = loading_end_index_downsampled - start_index

    print(f"Trajectory {traj_dir} starts from index {start_index} after stable start index detection")
    
    # calculate reward only BEFORE the terminal index
    # ep_return = return_function(obs_buffer, ref_stretch_force=10.0, ref_push_force=5.0, ref_deformation=0.005, ref_duration=500)
    if return_fn is not None:
        if return_kwargs is None:
            return_kwargs = {}
        ep_return = return_fn(obs_buffer[:terminal_index], include_deformation_in_return, **return_kwargs)
    else:
        raise ValueError("return_fn must be provided to calculate the return.")

    # save reward to a file
    np.savetxt(os.path.join(traj_path, "return.txt"), np.array([ep_return ]), fmt='%.6f')
    print(f"\033[93mExpected return for trajectory {traj_dir}: {ep_return} with {obs_buffer.shape[0]} steps.\033[0m")

    # concatenate post-fixing data
    # if post_sensing:
    #     post_obs_buffer = preprocess_observation(post_obs_buffer_raw, env_step=env_step)
    #     post_acts_buffer = preprocess_action(post_acts_buffer_raw, env_step=env_step)
    #     # concatenate the post-fixing data to the original buffers
    #     obs_buffer = np.concatenate([obs_buffer, post_obs_buffer], axis=0)
    #     acts_buffer = np.concatenate([acts_buffer, post_acts_buffer], axis=0)

    # plotting
    plot_force(obs_buffer, acts_buffer, base_dir, traj_dir, terminal_index, ending_index)
    return obs_buffer, acts_buffer, ep_return, terminals_buffer, fixing_success, setup_label

def load_trajectories(base_dir, return_function, env_step=5, reload_data=True, load_terminal_in_obs=False):
    if not reload_data and os.path.exists(os.path.join(base_dir, "episode_dataset.npz")):
        print(f"Loading existing episode dataset from {os.path.join(base_dir, 'episode_dataset.npz')}")
        data = np.load(os.path.join(base_dir, "episode_dataset.npz"), allow_pickle=True)
        grouped_ref_values = json.load(open(os.path.join(base_dir, "reference_values.json"), 'r'))
        return EpisodeDataset(
            observations=data["observations"],
            actions=data["actions"],
            terminals=data["terminals"],
            returns=data["returns"],
            load_terminals=load_terminal_in_obs,
            labels=data["label"]
        )
    else:
        # === Step 1: compute reference values across all episodes ===
        grouped_ref_values = compute_group_typed_reference_values(base_dir, env_step=env_step)
        # save as json
        with open(os.path.join(base_dir, "reference_values.json"), 'w') as f:
            json.dump(grouped_ref_values, f, indent=4)

        all_observations, all_actions, all_returns, all_terminals, all_labels = [], [], [], [], []

        for traj_dir in sorted(os.listdir(base_dir)):
            traj_path = os.path.join(base_dir, traj_dir)
            if not os.path.isdir(traj_path):  # Skip files like "episode_dataset.npz"
                continue
            
            # get reference values for this trajectory
            traj_ref_values = {}
            for group_key, ref_values in grouped_ref_values.items():
                if int(traj_dir) in ref_values["dirs"]:
                    traj_ref_values = ref_values["refs"].copy()
                    break

            obs_buffer, acts_buffer, ep_return, terminals_buffer, fixing_result, setup_label = load_one_episode(traj_path, 
                                                                                                    env_step=env_step,
                                                                                                    post_sensing=True,
                                                                                                    include_deformation_in_return=True,
                                                                                                    return_fn=return_function,
                                                                                                    return_kwargs=traj_ref_values)
            
            # filter out outliers with flat curves
            # if np.max(obs_buffer[:, 3]) - obs_buffer[0, 3] < 3:
            #     print(f"\033[91mSkipping trajectory {traj_dir} due to flat pushing with difference {np.max(obs_buffer[:, 3]) - obs_buffer[0, 3]}.\033[0m")
            #     continue
            
            all_observations.append(obs_buffer)
            all_actions.append(acts_buffer)
            all_returns.append(ep_return)
            all_terminals.append(terminals_buffer)
            all_labels.append(setup_label)

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
            returns=np.array(all_returns, dtype=np.float32),
            label=np.array(all_labels, dtype=object)  # save the setup label for each trajectory
        )
        print(f"Reloaded episode dataset saved to {os.path.join(base_dir, 'episode_dataset.npz')}")

        return EpisodeDataset(observations=all_observations,
                            actions=all_actions,
                            terminals=all_terminals,
                            returns=all_returns,
                            timeouts=None,
                            load_terminals=load_terminal_in_obs,
                            labels=all_labels
                            )  # timeouts are not used in this case

def plot_force(obs, act, base_dir, traj_dir, terminal_index, ending_index):
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
    dataset = load_trajectories(base_dir, max_based_return_function, env_step=5)
    print(f"Dataset loaded.")

    
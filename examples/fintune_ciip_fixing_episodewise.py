import argparse
import copy
import time

import d3rlpy
print("d3rlpy version:", d3rlpy.__version__)
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
import torch.nn.functional as F
from policies import MLPPolicy, StochasticMLPPolicy, export_to_onnx
from train_ciip_fixing_episodewise import train_weighted_bc, evaluate_imitation_policy

from build_episode_dataset import load_trajectories, load_one_episode
import wandb
import numpy as np
import os
import matplotlib.pyplot as plt
from d3rlpy.dataset import EpisodeWindowDataset

def reinforce_update(policy, optimizer, obs_seq, act_seq, episode_return, baseline=None,  num_epochs=50, lr=1e-3, gamma=0.99):
    """
    One-step REINFORCE update using a single trajectory.

    Args:
        policy: stochastic PyTorch policy (returns a Normal distribution)
        optimizer: torch optimizer
        obs_seq: list or tensor of observations (T, obs_dim)
        act_seq: list or tensor of actions (T, act_dim)
        episode_return: scalar return from this trajectory
        baseline: running average return (float)
        gamma: discount factor (not used here, but could be for future time weighting)
    Returns:
        updated baseline
    """
    policy.train()

    obs = torch.tensor(obs_seq, dtype=torch.float32)        # (T, obs_dim)
    acts = torch.tensor(act_seq, dtype=torch.float32)       # (T, act_dim)

    dist = policy.get_dist(obs)                             # Normal(mean, std)
    log_probs = dist.log_prob(acts).sum(dim=-1)             # (T,)
    log_prob_sum = log_probs.sum()

    # Compute baseline (EMA)
    if baseline is None:
        baseline = episode_return
    else:
        baseline = 0.9 * baseline + 0.1 * episode_return

    advantage = episode_return - baseline
    loss = -advantage * log_prob_sum

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return policy, baseline

def find_current_dir(base_dir):
    """
    Find the current directory with the highest index in the base directory.
    """
    dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.isdigit()]
    if not dirs:
        raise FileNotFoundError("No directories found in the base directory.")    
    
    # the current dir is the one with the highest index
    current_dir_index = max(int(d) for d in dirs)

    if os.path.exists(os.path.join(base_dir, str(current_dir_index))):
        # If the directory with the current index exists, return it
        return os.path.join(base_dir, str(current_dir_index))
    else:
        # If neither exists, raise an error
        raise FileNotFoundError(f"directory {current_dir_index} doens't exist in {base_dir}.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="REINFORCE episode-wise training script")
    parser.add_argument("--bc_log", type=str, default="20250702_190857", help="Base log directory for the pre-trained BC policy")
    parser.add_argument("--update_step", type=int, default=5, help="Update step for the policy output (default: 5,  i.e. 200Hz for the 1000Hz robot control frequency)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer (default: 1e-3)")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs for each training update (default: 4)")
    args = parser.parse_args()

    '''Load the policy'''
    bc_log_base_dir = "/home/tp2/Documents/kejia/d3rlpy/d3rlpy_logs/Clip_Weighted_BC/"
    bc_log_dir = os.path.join(bc_log_base_dir, args.bc_log)
    bc_policy_path = os.path.join(bc_log_dir, "trained_BC_policy.pth")

    # load config file
    if os.path.exists(os.path.join(bc_log_dir, "config.txt")):
        with open(os.path.join(bc_log_dir, "config.txt"), "r") as f:
            config = f.readlines()
        obs_dim = int(config[0].split(":")[1].strip())
        act_dim = int(config[1].split(":")[1].strip())
        # num_epochs = int(config[2].split(":")[1].strip())
        # batch_size = int(config[3].split(":")[1].strip())
        # lr = float(config[4].split(":")[1].strip())
        policy_type = config[5].split(":")[1].strip()
    else:
        obs_dim = 6
        act_dim = 2
        policy_type = "stochastic"

    if policy_type != "stochastic":
        raise ValueError("This script is designed for stochastic policies. Please use a stochastic policy.")

    policy = StochasticMLPPolicy(obs_dim, act_dim)
    policy.load_state_dict(torch.load(bc_policy_path))
    optimizer = optim.Adam(policy.parameters(), lr=args.lr)

    rl_log_base_dir = os.path.join(bc_log_dir, "finetune_reinforce")
    os.makedirs(rl_log_base_dir, exist_ok=True)
    rl_log_dir = os.path.join(rl_log_base_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(rl_log_dir, exist_ok=True)

    '''Training'''
    wandb.init(project="REINFORCE_episodewise", name=f"run_{time.strftime('%Y%m%d_%H%M%S')}",
               config={"learning_rate": args.lr, "epochs": args.num_epochs})
    baseline = None  # running average return

    # flag to control training loop
    ort_log_base_dir = "/home/tp2/Documents/kejia/ort/policy_log/"
    
    trial_id = 0
    while True:
        latest_ort_dir = find_current_dir(ort_log_base_dir)
        flag_path = os.path.join(latest_ort_dir, "done.flag")

        if os.path.exists(flag_path):
            print(f"[✓] Found {flag_path}. Starting training...")

            try:
                '''load the data'''
                obs_buffer, acts_buffer, ep_return, terminals_downsampled, terminals_buffer = load_one_episode(latest_ort_dir, env_step=args.update_step)  # load one episode

                '''train the policy'''
                policy, baseline = reinforce_update(policy, optimizer, obs_buffer, acts_buffer, ep_return, num_epochs=args.num_epochs, lr=args.lr, baseline=baseline)
                print(f"[✓] Training complete for trial_{trial_id}")

                '''Save the policy and export to ONNX'''
                rl_policy_path = os.path.join(rl_log_dir, f"trained_reinforce_policy_{str(trial_id)}.pth")
                torch.save(policy.state_dict(),rl_policy_path)
                export_to_onnx(policy, obs_dim, export_path=os.path.join(rl_log_dir, f"trained_reinforce_policy_{str(trial_id)}.onnx"))
                with open(os.path.join(rl_log_dir, f"config_{str(trial_id)}.txt"), "w") as f:
                    f.write(f"bc_policy: {args.bc_log}\n")
                    f.write(f"rl_data_dir: {latest_ort_dir}\n")

                wandb.log({"epoch": trial_id, "return": ep_return, "baseline": baseline})
            except Exception as e:
                print(f"[✗] Error during training trial_{trial_id}: {e}")
                # Optionally keep the flag file for debugging
                continue

            # Remove or rename flag to avoid retriggering
            try:
                os.remove(flag_path)
                print(f"[✓] Remove {flag_path}")
                time.sleep(1)  # wait for the flag file to appear

                if os.path.exists(flag_path):
                    os.remove(flag_path)
                    print(f"[✓] Remove {flag_path} again")
                    time.sleep(1)  # wait for the flag file to appear
                else:
                    print(f"[✓] {flag_path} already removed")

            except Exception as e:
                print(f"[✗] Failed to remove {flag_path}: {e}")

            trial_id += 1

        else:
            time.sleep(0.1)  # wait for the flag file to appear
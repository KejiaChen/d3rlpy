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
from policies import MLPPolicy, MLPSeqPolicy, StochasticMLPPolicy, StochasticMLPSeqPolicy, export_to_onnx
from return_functions import *

from build_episode_dataset import load_trajectories
import wandb
import numpy as np
import os
import matplotlib.pyplot as plt
from d3rlpy.dataset import EpisodeWindowDataset
from functools import partial

def collate_variable_episodes(batch):
    """
    Pads episodes in a batch to the same length.

    Args:
        batch: List of tuples (obs: Tensor(T_i, obs_dim), act: Tensor(T_i, act_dim), weight: float)

    Returns:
        obs_padded:  Tensor(B, T_max, obs_dim)
        act_padded:  Tensor(B, T_max, act_dim)
        weights:     Tensor(B,)
        mask:        BoolTensor(B, T_max) — True for valid steps
        lengths:     Tensor(B,) — lengths before padding
    """
    observations, actions, weights = zip(*batch)

    lengths = torch.tensor([obs.shape[0] for obs in observations])
    obs_padded = pad_sequence(observations, batch_first=True)  # (B, T_max, obs_dim)
    act_padded = pad_sequence(actions, batch_first=True)       # (B, T_max, act_dim)

    max_len = lengths.max()
    batch_size = len(batch)

    # Create mask
    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)  # shape: (B, T_max)

    return obs_padded, act_padded, torch.tensor(weights), mask, lengths

def collate_variable_episodes_windowed(batch, seq_len, include_terminal=False):
    """
    Args:
        batch: List of (obs: Tensor(T, obs_dim), act: Tensor(T, act_dim), weight: float)

    Returns:
        obs_seq_tensor: (B, T - seq_len + 1, seq_len, obs_dim)
        act_target_tensor: (B, T - seq_len + 1, act_dim)
        weights: (B,)
        mask: (B, T - seq_len + 1)
        lengths: (B,)
    """
    obs_sequences = []
    act_targets = []
    masks = []
    lengths = []
    weights_out = []

    for data in batch:
        if include_terminal:
            obs, act, weight, terminals = data
            terminals = terminals.view(-1, 1)                    # (T, 1)
            obs = torch.cat([obs, terminals], dim=-1)           # (T, obs_dim + 1)
        else:
            obs, act, weight = data

    # for obs, act, weight in batch:
        T = obs.shape[0]
        if T < seq_len:
            continue

        obs_seq = torch.stack([obs[i:i+seq_len] for i in range(T - seq_len + 1)])  # (T - seq_len + 1, seq_len, obs_dim)
        act_seq = act[seq_len - 1:]  # next action target (T - seq_len + 1, act_dim)
        mask = torch.ones(obs_seq.shape[0], dtype=torch.bool)

        obs_sequences.append(obs_seq)
        act_targets.append(act_seq)
        masks.append(mask)
        lengths.append(obs_seq.shape[0])
        weights_out.append(weight)

    # Pad sequences across batch
    obs_padded = pad_sequence(obs_sequences, batch_first=True)         # (B, T*, seq_len, obs_dim)
    act_padded = pad_sequence(act_targets, batch_first=True)           # (B, T*, act_dim)
    mask_padded = pad_sequence(masks, batch_first=True, padding_value=False)  # (B, T*)

    return obs_padded, act_padded, torch.tensor(weights_out), mask_padded, torch.tensor(lengths)

def compute_weighted_masked_bc_loss(obs_padded, act_padded, weights, mask, policy, policy_type="deterministic"):
    """
    Args:
        obs_seq:      (B, T, S, obs_dim)
        act_padded:   (B, T, act_dim)
        weights:      (B,)
        mask:         (B, T)
        policy:       Sequence-based policy taking (B*T, S, obs_dim)
    """
    if obs_padded.dim() == 3:
        # obs_padded: (B, T, obs_dim)
        B, T, obs_dim = obs_padded.shape
        obs_flat = obs_padded.view(B * T, obs_dim)
    elif obs_padded.dim() == 4:
        # obs_padded: (B, T, seq_len, obs_dim)
        B, T, S, obs_dim = obs_padded.shape
        obs_flat = obs_padded.view(B * T, S, obs_dim)      # (B*T, S, obs_dim)

    # B, T, obs_dim = obs_padded.shape
    _, _, act_dim = act_padded.shape

    # obs_flat = obs_padded.view(B * T, obs_dim)
    act_flat = act_padded.view(B * T, act_dim)
    mask_flat = mask.view(B * T)

    if policy_type == "stochastic":
        # For stochastic policy, we need to sample actions
        pred_dist = policy.get_dist(obs_flat)
        log_probs = pred_dist.log_prob(act_flat).sum(dim=-1)  # (B*T,)
        loss_per_step = -log_probs  # negative log likelihood
    elif policy_type == "deterministic":
        pred_flat = policy(obs_flat)  # (B*T, act_dim)
        loss_per_step = F.mse_loss(pred_flat, act_flat, reduction='none').sum(dim=-1)  # (B*T,)

    # Apply mask
    loss_masked = loss_per_step[mask_flat]  # (valid_steps,)
    
    # Expand weights: (B,) → (B*T,) → masked
    weight_per_step = weights.unsqueeze(1).repeat(1, T).view(-1)
    weight_masked = weight_per_step[mask_flat]

    eps = 1e-8
    weighted_loss = (loss_masked * weight_masked).sum() / (weight_masked.sum() +eps)
    return weighted_loss, weight_masked.sum().item()

def train_weighted_bc(dataset, obs_dim, act_dim, seq_len=10, num_epochs=50, batch_size=8, lr=1e-3, policy_type="deterministic", log_dir=None, save_epoch=50, include_terminal_in_obs=False):
    run_name = f"Clip_BC_episodewise_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="BC_episodewise", name=run_name, config={
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": num_epochs
    })
    
    if seq_len is not None and seq_len > 1:
        # Use the windowed dataset if seq_len is specified
        collate_function = partial(collate_variable_episodes_windowed, seq_len=seq_len, include_terminal=include_terminal_in_obs)
    else:
        collate_function = collate_variable_episodes

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
    
    if policy_type == "stochastic":
        if seq_len is not None and seq_len > 1:
            policy = StochasticMLPSeqPolicy(obs_dim, act_dim, seq_len=seq_len)
        else:
            policy = StochasticMLPPolicy(obs_dim, act_dim)
    elif policy_type == "deterministic":
        if seq_len is not None and seq_len > 1:
            policy = MLPSeqPolicy(obs_dim, act_dim, seq_len=seq_len)
        else:
            policy = MLPPolicy(obs_dim, act_dim)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Use 'stochastic' or 'deterministic'.")
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        total_weight = 0.0
        for obs_batch, act_batch, weights, mask, _ in dataloader:
            optimizer.zero_grad()
            weighted_loss, batch_weight = compute_weighted_masked_bc_loss(obs_batch, act_batch, weights, mask, policy, policy_type)
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            total_weight += batch_weight

        avg_loss = epoch_loss / max(total_weight, 1e-8)
        print(f"Epoch {epoch}: Epoch loss {epoch_loss:.4f}, Weighted Avg Loss = {avg_loss:.4f}")
        wandb.log({"epoch": epoch, "weighted_bc_loss": avg_loss})

        if epoch % save_epoch == 0:
            if log_dir:
                policy_path = os.path.join(log_dir, f"policy_epoch_{epoch}.pth")
                torch.save(policy.state_dict(), policy_path)
                export_to_onnx(policy, obs_dim, seq_len, export_path=os.path.join(log_dir, f"trained_BC_policy_{epoch}.onnx"))
                print(f"Saved policy at {policy_path}")

    return policy

def evaluate_imitation_policy(policy, dataset, policy_type="deterministic"):
    policy.eval()
    total_mse = 0.0
    with torch.no_grad():
        for obs, gt_act in zip(dataset.observations, dataset.actions):
            if policy_type == "stochastic":
                # For stochastic policy, sample actions
                pred_act = policy(torch.tensor(obs).float())
            elif policy_type == "deterministic":
                # For deterministic policy, get the mean action
                pred_act = policy(torch.tensor(obs).float())
            else:
                raise ValueError(f"Unknown policy type: {policy_type}. Use 'stochastic' or 'deterministic'.")
            
            mse = F.mse_loss(pred_act, torch.tensor(gt_act).float()).item()
            total_mse += mse
            
    avg_mse = total_mse / len(dataset)
    print(f"Average MSE over dataset: {avg_mse:.4f}")
    return avg_mse

if __name__ == "__main__":
    train = True # Set to False to skip training and only evaluate
    reload_data= False # Set to True to reload the dataset from raw txt files, False to use the cached npz dataset stored from previous runs
    load_fixing_terminal_in_obs = True # Set to True to load the fixing terminal as an extra input dimension in the observation, False to ignore it
    update_step = 5  # update the policy output every 5 robot control loops, i.e. 200Hz for the 1000Hz robot control frequency
    policy_type = "stochastic"  # "stochastic" for stochastic policy, "deterministic" for MLPPolicy
    train_epochs = 50
    learning_rate = 1e-3
    batch_size = 1
    seq_window_len = 30  # sequence length for the episode window dataset. Corresponding number of control loops: seq_window_len*update_step

    '''prepare the dataset'''

    dataset_base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/off_policy_3/"
    return_function = effort_and_energy_based_return_function  # or effort_based_return_function
    dataset = load_trajectories(dataset_base_dir, return_function, env_step=update_step, reload_data=reload_data, load_terminal_in_obs=load_fixing_terminal_in_obs) # observation (B, D)
    # seq_dataset =  EpisodeWindowDataset(dataset, seq_len=seq_window_len)  # observation (B,T,D)
    
    # if seq_window_len is None or seq_window_len <= 1:
    #     dataset = single_dataset
    # else:
    #     dataset = seq_dataset


    from torch.utils.data import random_split   
    # 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    obs_dim = dataset[0][0].shape[1]
    if load_fixing_terminal_in_obs:
        obs_dim += 1
    act_dim = dataset[0][1].shape[1]

    print(f"Training using dataset with {dataset.__len__()} episodes, "
          f"obs_dim={obs_dim}, act_dim={act_dim}")

    '''train the policy'''
    log_base_dir = "/home/tp2/Documents/kejia/d3rlpy/d3rlpy_logs/Clip_Weighted_BC/"
    if train:
        log_dir = os.path.join(log_base_dir, time.strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # save the training config to log_dir
        with open(os.path.join(log_dir, "config.txt"), "w") as f:
            f.write(f"obs_dim: {obs_dim}\n")
            f.write(f"act_dim: {act_dim}\n")
            f.write(f"seq_window_len: {seq_window_len}\n")
            f.write(f"num_epochs: {train_epochs}\n")
            f.write(f"batch_size: {batch_size}\n")
            f.write(f"lr: {learning_rate}\n")
            f.write(f"policy_type: {policy_type}\n")
            f.write(f"return_function: {return_function.__name__}\n")

        policy = train_weighted_bc(
                                train_dataset,
                                obs_dim=obs_dim,
                                act_dim=act_dim,
                                seq_len=seq_window_len,
                                num_epochs=train_epochs,
                                batch_size=batch_size,
                                lr=learning_rate,
                                policy_type=policy_type,
                                log_dir=log_dir,
                                save_epoch=50,
                                include_terminal_in_obs=load_fixing_terminal_in_obs
                            )
    else:
        log_stored_folder = "20250727_182152" #20250629_170553 #20250702_160901 20250707_215329 #20250709_141454 # 20250725_162934
        log_dir = os.path.join(log_base_dir, log_stored_folder)

    policy_path = os.path.join(log_dir, "trained_BC_policy.pth")

    if train:
        # save the policy
        torch.save(policy.state_dict(), policy_path)
        export_to_onnx(policy, obs_dim, seq_window_len, export_path=os.path.join(log_dir, "trained_BC_policy.onnx"))

    '''Evaluate the policy'''
    # load the policy
    if policy_type == "stochastic":
        if seq_window_len is not None and seq_window_len > 1:
            reload_policy = StochasticMLPSeqPolicy(obs_dim, act_dim, seq_len=seq_window_len)
        else:
            reload_policy = StochasticMLPPolicy(obs_dim, act_dim)
    elif policy_type == "deterministic":
        if seq_window_len is not None and seq_window_len > 1:
            reload_policy = MLPSeqPolicy(obs_dim, act_dim, seq_len=seq_window_len)
        else:
            reload_policy = MLPPolicy(obs_dim, act_dim)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Use 'stochastic' or 'deterministic'.")
    reload_policy.load_state_dict(torch.load(policy_path))
    reload_policy.eval()
    # offline imitation evaluation
    with torch.no_grad():
         for idx, item in enumerate(val_dataset):  # obs_seq: (T, obs_dim)
            if load_fixing_terminal_in_obs:
                obs_seq, act_seq, weight, terminals = item
                terminals = terminals.view(-1, 1)                    # (T, 1)
                obs_seq = torch.cat([obs_seq, terminals], dim=-1)    
            else:
                obs_seq, act_seq, weight = item
            # # evaluate the policy episode-wise
            # pred_seq = reload_policy(obs_seq)  # (T, act_dim), works for MLP
            # mse = F.mse_loss(pred_seq, act_seq).item()
            # print(f"Evaluation MSE: {mse:.4f} for episode with weight {weight:.4f}")

            # evaluate the policy step by step
            pred_seq = []
            env_count = 0
            pred_step = torch.zeros(1, act_seq.shape[1])  # Initialize pred_step with zeros

            for i in range(obs_seq.shape[0]):
                if i < seq_window_len:
                    # Not enough context to form full sequence
                    # pred_seq.append(torch.zeros_like(act_seq[i]))
                    continue
                
                if seq_window_len is not None and seq_window_len > 1:
                    obs_step = obs_seq[i - seq_window_len:i]  # shape: (seq_len, obs_dim)
                    obs_step = obs_step.unsqueeze(0)  # (1, seq_len, obs_dim)
                else:
                    obs_step = (obs_seq[i].unsqueeze(0))

                if env_count % update_step == 0:  # only update the output every `update_step` steps
                    pred_step = reload_policy(obs_step)  # (1, act_dim)
                pred_seq.append(pred_step.squeeze(0))
                env_count += 1
            pred_seq = torch.stack(pred_seq)  # shape: (T, act_dim)
            mse = F.mse_loss(pred_seq, act_seq[seq_window_len:]).item()
            print(f"Evaluation MSE: {mse:.4f} for episode with weight {weight:.4f}")

            # === Plot Actions: Ground Truth vs Prediction ===
            T, act_dim = act_seq[seq_window_len:].shape
            fig, axs = plt.subplots(act_dim, 1, figsize=(10, 2 * act_dim), sharex=True)
            if act_dim == 1:
                axs = [axs]

            # plot stertch
            axs[0].plot(range(T), obs_seq[seq_window_len:, 0].cpu().numpy(), label="Obs", color="green")
            axs[0].plot(range(T), act_seq[seq_window_len:, 0].cpu().numpy(), label="Demo_act", color="red")
            axs[0].plot(range(T), pred_seq[:, 0].cpu().numpy(), label="Pred", color="red", linestyle="--")
            if load_fixing_terminal_in_obs:
                # plot the terminal flag
                # Plot terminal flag as background fill
                terminal_flags = obs_seq[seq_window_len:, -1].cpu().numpy()  # Extract terminal flags
                for i in range(len(terminal_flags)):
                    if terminal_flags[i]:  # If terminal flag is True
                        axs[0].axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.2, label="Terminal" if i == 0 else None)
            axs[0].set_ylabel(f"Stretch")
            axs[0].legend()

            axs[1].plot(range(T), obs_seq[seq_window_len:, 3].cpu().numpy(), label="Obs", color="green")
            axs[1].plot(range(T), act_seq[seq_window_len:, 1].cpu().numpy(), label="Demo_act", color="red")
            axs[1].plot(range(T), pred_seq[:, 1].cpu().numpy(), label="Pred", color="red", linestyle="--")
            if load_fixing_terminal_in_obs:
                # plot the terminal flag
                terminal_flags = obs_seq[seq_window_len:, -1].cpu().numpy()
                for i in range(len(terminal_flags)):
                    if terminal_flags[i]:
                        axs[1].axvspan(i - 0.5, i + 0.5, color="gray", alpha=0.2, label="Terminal" if i == 0 else None)
            axs[1].set_ylabel(f"PUsh")
            axs[1].legend()

            axs[-1].set_xlabel("Timestep")
            fig.suptitle(f"Episode {idx} - Return weight: {weight:.2f}, MSE: {mse:.4f}")
            plt.show()

            plt.close()

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
import torch.nn as nn

from build_episode_dataset import load_trajectories
import wandb
import numpy as np
import os
import matplotlib.pyplot as plt


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        dims = [obs_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], act_dim)

    def forward(self, x):  # x: (T, obs_dim)
        return self.head(self.encoder(x))  # (T, act_dim)

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

def collate_variable_episodes(batch):
    observations, actions, weights = zip(*batch)

    lengths = torch.tensor([obs.shape[0] for obs in observations])
    obs_padded = pad_sequence(observations, batch_first=True)  # (B, T_max, obs_dim)
    act_padded = pad_sequence(actions, batch_first=True)       # (B, T_max, act_dim)

    max_len = lengths.max()
    batch_size = len(batch)

    mask = torch.arange(max_len).expand(batch_size, max_len) < lengths.unsqueeze(1)  # shape: (B, T_max)

    return obs_padded, act_padded, torch.tensor(weights), mask, lengths

def compute_weighted_masked_bc_loss(obs_padded, act_padded, weights, mask, policy):
    B, T, obs_dim = obs_padded.shape
    _, _, act_dim = act_padded.shape

    obs_flat = obs_padded.view(B * T, obs_dim)
    act_flat = act_padded.view(B * T, act_dim)
    mask_flat = mask.view(B * T)

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

def train_weighted_bc(dataset, obs_dim, act_dim, num_epochs=50, batch_size=8, lr=1e-3, policy_type="deterministic"):
    run_name = f"Clip_BC_episodewise_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="BC_episodewise", name=run_name, config={
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": num_epochs
    })
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            collate_fn=collate_variable_episodes)
    
    policy = MLPPolicy(obs_dim, act_dim)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    for epoch in range(num_epochs):
        epoch_loss = 0.0

        total_weight = 0.0
        for obs_batch, act_batch, weights, mask, _ in dataloader:
            optimizer.zero_grad()
            weighted_loss, batch_weight = compute_weighted_masked_bc_loss(obs_batch, act_batch, weights, mask, policy)
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            total_weight += batch_weight

        avg_loss = epoch_loss / max(total_weight, 1e-8)
        print(f"Epoch {epoch}: Epoch loss {epoch_loss:.4f}, Weighted Avg Loss = {avg_loss:.4f}")
        wandb.log({"epoch": epoch, "weighted_bc_loss": avg_loss})

    return policy

def evaluate_imitation_policy(policy, dataset):
    policy.eval()
    total_mse = 0.0
    with torch.no_grad():
        for obs, gt_act in zip(dataset.observations, dataset.actions):
            pred_act = policy(torch.tensor(obs).float())
            mse = F.mse_loss(pred_act, torch.tensor(gt_act).float()).item()
            total_mse += mse
    avg_mse = total_mse / len(dataset)
    print(f"Average MSE over dataset: {avg_mse:.4f}")
    return avg_mse


if __name__ == "__main__":
    reload_data=False

    base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/"
    dataset = load_trajectories(base_dir, env_step=5, reload_data=reload_data)

    from torch.utils.data import random_split
    # 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    obs_dim = dataset[0][0].shape[1]
    act_dim = dataset[0][1].shape[1]

    print(f"Training using dataset with {dataset.__len__()} episodes, "
          f"obs_dim={obs_dim}, act_dim={act_dim}")

    policy = train_weighted_bc(
                            train_dataset,
                            obs_dim=obs_dim,
                            act_dim=act_dim,
                            num_epochs=50,
                            batch_size=1,
                            lr=1e-3
                        )

    # offline imitation eval
    policy.eval()
    with torch.no_grad():
         for idx, (obs_seq, act_seq, weight) in enumerate(val_dataset):  # obs_seq: (T, obs_dim)
            pred_seq = policy(obs_seq)  # (T, act_dim), works for MLP
            mse = F.mse_loss(pred_seq, act_seq).item()
            print(f"Evaluation MSE: {mse:.4f} for episode with weight {weight:.4f}")

     # === Plot Actions: Ground Truth vs Prediction ===
            T, act_dim = act_seq.shape
            fig, axs = plt.subplots(act_dim, 1, figsize=(10, 2 * act_dim), sharex=True)
            if act_dim == 1:
                axs = [axs]

            # plot stertch
            axs[0].plot(range(T), obs_seq[:, 0].cpu().numpy(), label="Obs", color="green")
            axs[0].plot(range(T), act_seq[:, 0].cpu().numpy(), label="Demo_act", color="red")
            axs[0].plot(range(T), pred_seq[:, 0].cpu().numpy(), label="Pred", color="red", linestyle="--")
            axs[0].set_ylabel(f"Stretch")
            axs[0].legend()

            axs[1].plot(range(T), obs_seq[:, 3].cpu().numpy(), label="Obs", color="green")
            axs[1].plot(range(T), act_seq[:, 1].cpu().numpy(), label="Demo_act", color="red")
            axs[1].plot(range(T), pred_seq[:, 1].cpu().numpy(), label="Pred", color="red", linestyle="--")
            axs[1].set_ylabel(f"Stretch")
            axs[1].legend()

            axs[-1].set_xlabel("Timestep")
            fig.suptitle(f"Episode {idx} - Return weight: {weight:.2f}, MSE: {mse:.4f}")
            plt.show()

            plt.close()

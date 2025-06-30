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
from d3rlpy.dataset import EpisodeWindowDataset


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        dims = [obs_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], act_dim)
    
    def forward(self, x):  # x: (T, obs_dim) or (B, T, obs_dim)
        if x.dim() == 3:
            B, T, D = x.shape
            x = x.view(B * T, D)
            out = self.head(self.encoder(x))
            return out.view(B, T, -1)  # restore batch + seq
        return self.head(self.encoder(x))

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

def export_to_onnx(policy, obs_dim, export_path="mlp_policy.onnx"):
    policy.eval()
    dummy_input = torch.randn(1, 1, obs_dim)  # 3D input: (batch_size, seq_len, obs_dim) to match the expected input shape in cpp
    torch.onnx.export(
        policy, 
        dummy_input, 
        export_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # support variable batch size
            "output": {0: "batch_size"}
        }
    )
    print(f"ONNX model exported to {export_path}")


if __name__ == "__main__":
    train = True # Set to False to skip training and only evaluate
    update_step = 5  # update the policy output every 5 robot control loops, i.e. 200Hz for the 1000Hz robot control frequency

    '''prepare the dataset'''
    reload_data=False

    dataset_base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/"
    dataset = load_trajectories(dataset_base_dir, env_step=update_step, reload_data=reload_data)
    seq_dataset =  EpisodeWindowDataset(dataset, seq_len=5)  # seq_len=5 for 5-step window

    from torch.utils.data import random_split
    # 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    obs_dim = dataset[0][0].shape[1]
    act_dim = dataset[0][1].shape[1]

    print(f"Training using dataset with {dataset.__len__()} episodes, "
          f"obs_dim={obs_dim}, act_dim={act_dim}")

    '''train the policy'''
    if train:
        policy = train_weighted_bc(
                                train_dataset,
                                obs_dim=obs_dim,
                                act_dim=act_dim,
                                num_epochs=50,
                                batch_size=1,
                                lr=1e-3
                            )
    
    log_base_dir = "/home/tp2/Documents/kejia/d3rlpy/d3rlpy_logs/Clip_Weighted_BC/"
    if train:
        log_dir = os.path.join(log_base_dir, time.strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
    else:
        log_dir = os.path.join(log_base_dir, "20250629_170553")

    policy_path = os.path.join(log_dir, "trained_BC_policy.pth")

    if train:
        # save the policy
        torch.save(policy.state_dict(), policy_path)
        export_to_onnx(policy, obs_dim, export_path=os.path.join(log_dir, "trained_BC_policy.onnx"))

    '''Evaluate the policy'''
    # load the policy
    reload_policy = MLPPolicy(obs_dim, act_dim)
    reload_policy.load_state_dict(torch.load(policy_path))
    reload_policy.eval()
    # offline imitation evaluation
    with torch.no_grad():
         for idx, (obs_seq, act_seq, weight) in enumerate(val_dataset):  # obs_seq: (T, obs_dim)
            # # evaluate the policy episode-wise
            # pred_seq = reload_policy(obs_seq)  # (T, act_dim), works for MLP
            # mse = F.mse_loss(pred_seq, act_seq).item()
            # print(f"Evaluation MSE: {mse:.4f} for episode with weight {weight:.4f}")

            # evaluate the policy step by step
            pred_seq = []
            env_count = 0
            pred_step = torch.zeros(1, act_seq.shape[1])  # Initialize pred_step with zeros

            for i in range(obs_seq.shape[0]):
                obs_step = (obs_seq[i].unsqueeze(0)).unsqueeze(0)
                if env_count % update_step == 0:  # only update the output every `update_step` steps
                    pred_step = reload_policy(obs_step)  # (1, act_dim)
                pred_seq.append(pred_step.squeeze(0))
                env_count += 1
            pred_seq = torch.stack(pred_seq)  # shape: (T, act_dim)
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
            axs[0].plot(range(T), pred_seq[:, 0, 0].cpu().numpy(), label="Pred", color="red", linestyle="--")
            axs[0].set_ylabel(f"Stretch")
            axs[0].legend()

            axs[1].plot(range(T), obs_seq[:, 3].cpu().numpy(), label="Obs", color="green")
            axs[1].plot(range(T), act_seq[:, 1].cpu().numpy(), label="Demo_act", color="red")
            axs[1].plot(range(T), pred_seq[:, 0, 1].cpu().numpy(), label="Pred", color="red", linestyle="--")
            axs[1].set_ylabel(f"PUsh")
            axs[1].legend()

            axs[-1].set_xlabel("Timestep")
            fig.suptitle(f"Episode {idx} - Return weight: {weight:.2f}, MSE: {mse:.4f}")
            plt.show()

            plt.close()

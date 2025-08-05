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
from policies import MLPPolicy, MLPSeqPolicy, StochasticMLPPolicy, StochasticMLPSeqPolicy, StochasticRNNPolicy, export_policy_to_onnx
from dynamics_encoder_decoder import DynamicsRNNEncoder, masked_mse, export_encoder_to_onnx
from return_functions import *

from build_episode_dataset import load_trajectories
import wandb
import numpy as np
import os
import matplotlib.pyplot as plt
from d3rlpy.dataset import EpisodeWindowDataset
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_variable_episodes(batch, include_terminal=False):
    """
    Args:
        batch: List of (obs: (T, obs_dim), act: (T, act_dim), weight: float)

    Returns:
        obs_padded: (B, T_max, obs_dim)
        act_padded: (B, T_max, act_dim)
        weights: (B,)
        mask: (B, T_max) — bool
        lengths: (B,) — original episode lengths
    """
    obs_list, act_list, weights, lengths = [], [], [], []

    for data in batch:
        if include_terminal:
            obs, act, weight, terminals = data
            terminals = terminals.view(-1, 1)
            obs = torch.cat([obs, terminals], dim=-1)
            terminal_idx = (terminals.squeeze() == 1).nonzero(as_tuple=True)[0]
            ep_len = terminal_idx[0].item() + 1 if len(terminal_idx) > 0 else obs.shape[0]
        else:
            obs, act, weight = data
            ep_len = obs.shape[0]

        obs_list.append(obs)
        act_list.append(act)
        weights.append(weight)
        lengths.append(ep_len)  

    lengths = torch.tensor(lengths)
    obs_padded = pad_sequence(obs_list, batch_first=True)
    act_padded = pad_sequence(act_list, batch_first=True)
    mask = torch.arange(obs_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T_max)

    return obs_padded, act_padded, torch.tensor(weights), mask, lengths

def compute_weighted_masked_bc_loss(obs_padded, act_padded, weights, lengths, mask, policy, encoder, policy_type="deterministic"):
    """
    Args:
        obs_seq:      (B, T, S, obs_dim)
        act_padded:   (B, T, act_dim)
        weights:      (B,)
        mask:         (B, T)
        policy:       Sequence-based policy taking (B*T, S, obs_dim)
    """
    B, T, obs_dim = obs_padded.shape

    z_dyn = encoder(obs_padded[:, :, :6], lengths) 
    pred_dist = policy.get_dist(obs_padded, z_dyn)  # (B*T, act_dim)
    log_probs = pred_dist.log_prob(act_padded).sum(dim=-1)  # (B*T,)
    loss_per_step = -log_probs  # (B*T,)

    # loss_per_step: (B, T)
    loss_masked = loss_per_step[mask]  # (num_valid_steps,)

    # Expand weights from (B,) to (B, T)
    weight_per_step = weights.unsqueeze(1).expand(-1, T)  # (B, T)
    weight_masked = weight_per_step[mask]  # (num_valid_steps,)    

    eps = 1e-8
    weighted_loss = (loss_masked * weight_masked).sum() / (weight_masked.sum() +eps)
    return weighted_loss, weight_masked.sum().item()

def train_weighted_bc(dataset, obs_dim, act_dim, num_epochs=50, batch_size=8, lr=1e-3, policy_type="deterministic", log_dir=None, save_epoch=50, include_terminal_in_obs=False):
    run_name = f"Clip_BC_episodewise_{time.strftime('%Y%m%d_%H%M%S')}"
    wandb.init(project="BC_episodewise", name=run_name, config={
        "batch_size": batch_size,
        "learning_rate": lr,
        "epochs": num_epochs
    })

    collate_function = partial(collate_variable_episodes, include_terminal=include_terminal_in_obs)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
    
    encoder = DynamicsRNNEncoder(obs_dim-1).to(device)
    encoder.load_state_dict(torch.load("d3rlpy_logs/Dynamics_Encoder/best_encoder_2_force_only_encoder_decoder.pth"))
    encoder.train()  # Encoder is used for feature extraction only
    for param in encoder.parameters(): # freeze encoder parameters
        param.requires_grad = False

    if policy_type == "gru_with_encoder":
        policy = StochasticRNNPolicy(obs_dim, act_dim).to(device)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Use 'stochastic' or 'deterministic'.")
    policy.train()
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Early stopping variables
    best_loss = float('inf')  # Initialize best loss to infinity
    best_eval_loss = float('inf')
    patience_counter = 0      # Counter for early stopping

    for epoch in range(num_epochs):
        policy.train()  # Set policy to training mode

        epoch_loss = 0.0

        total_weight = 0.0
        for obs_batch, act_batch, weights, mask, lengths in dataloader:
            obs_batch, act_batch, weights, mask, lengths = obs_batch.to(device), act_batch.to(device), weights.to(device), mask.to(device), lengths.to(device)
            optimizer.zero_grad()
            weighted_loss, batch_weight = compute_weighted_masked_bc_loss(obs_batch, act_batch, weights, lengths, mask, policy, encoder, policy_type)
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            total_weight += batch_weight

        avg_loss = epoch_loss / max(total_weight, 1e-8)
        print(f"Epoch {epoch}: Epoch loss {epoch_loss:.4f}, Weighted Avg Loss = {avg_loss:.4f}")
        wandb.log({"epoch": epoch, "weighted_bc_loss": avg_loss})

        if epoch != 0 and epoch % save_epoch == 0:
            if log_dir:
                policy_path = os.path.join(log_dir, f"policy_epoch_{epoch}.pth")
                torch.save(policy.state_dict(), policy_path)
                export_policy_to_onnx(policy, obs_dim, 0, z_dim=32, policy_type=policy_type, export_path=os.path.join(log_dir, f"trained_BC_policy_{epoch}.onnx"))
                print(f"Saved policy at {policy_path}")

                encoder_path = os.path.join(log_dir, f"encoder_epoch_{epoch}.pth")
                torch.save(encoder.state_dict(), encoder_path)
                encoder.set_export(True)  # Enable ONNX export
                export_encoder_to_onnx(encoder, obs_dim, z_dim=32, export_path=os.path.join(log_dir, f"trained_BC_encoder_{epoch}.onnx"))   
                print(f"Saved encoder at {encoder_path}")
                encoder.set_export(False)  # Disable ONNX export after saving

        # Early stopping logic
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0  # Reset patience counter
            # Save the best model
            if log_dir:
                best_model_path = os.path.join(log_dir, "best_policy.pth")
                torch.save(policy.state_dict(), best_model_path)
                print(f"Best model saved at epoch {epoch} with loss {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best loss: {best_loss:.4f})")

        # Check if patience is exceeded
        if patience_counter >= 10:
            print(f"Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.4f}")
            break

    return policy, encoder

def evaluate_policy(policy, encoder, val_loader, loss_fn, policy_type="gru_with_encoder", if_plot=False):
    policy.eval()
    encoder.eval()
    total_loss = 0.0
    total_weight = 0.0

    with torch.no_grad():
        for obs_batch, act_batch, weights, mask, lengths in val_loader:
            obs_batch = obs_batch.to(device)       # (B, T, obs_dim)
            act_batch = act_batch.to(device)       # (B, T, act_dim)
            weights = weights.to(device)           # (B,)
            mask = mask.to(device)                 # (B, T)
            lengths = lengths.to(device)           # (B,)

            if policy_type == "gru_with_encoder":
                # Assuming the encoder input is (B, T, 6)
                encoder_input = obs_batch[:, :, :6]  # or adapt if different
                z_dyn = encoder(encoder_input, lengths)  # (B, z_dim)
                pred = policy(obs_batch, z_dyn)  # (B, T, act_dim)
            else:
                pred = policy(obs_batch)

            # Compute masked loss (can reuse your masked MSE function)
            loss = loss_fn(pred, act_batch, mask)
            total_loss += (loss.item() * weights.sum().item())
            total_weight += weights.sum().item()

            # Optional visualization
            if if_plot:
                for i in range(obs_batch.size(0)):
                    T = obs_batch[i].size(0)
                    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                    axs[0].plot(range(T), obs_batch[i, :, 0].cpu().numpy(), label="Observation", color="green")
                    axs[0].plot(range(T), act_batch[i, :, 0].cpu().numpy(), label="Ground Truth Action", color="red")
                    axs[0].plot(range(T), pred[i, :, 0].cpu().numpy(), label="Predicted Action", color="blue", linestyle="--")
                    axs[0].set_ylabel("Stretch")
                    axs[0].legend()

                    axs[1].plot(range(T), obs_batch[i, :, 3].cpu().numpy(), label="Observation", color="green")
                    axs[1].plot(range(T), act_batch[i, :, 1].cpu().numpy(), label="Ground Truth Action", color="red")
                    axs[1].plot(range(T), pred[i, :, 1].cpu().numpy(), label="Predicted Action", color="blue", linestyle="--")
                    axs[1].set_ylabel("Push")
                    axs[1].legend()

                    plt.xlabel("Time Step")
                    plt.title(f"Episode {i + 1} - Loss: {loss.item():.4f}")
                    plt.show()
                    plt.close(fig)

    return total_loss / max(total_weight, 1e-8)

if __name__ == "__main__":
    train = False # Set to False to skip training and only evaluate
    reload_data= False # Set to True to reload the dataset from raw txt files, False to use the cached npz dataset stored from previous runs
    load_fixing_terminal_in_obs = True # Set to True to load the fixing terminal as an extra input dimension in the observation, False to ignore it
    update_step = 5  # update the policy output every 5 robot control loops, i.e. 200Hz for the 1000Hz robot control frequency
    policy_type = "gru_with_encoder"  # "stochastic" for stochastic policy, "deterministic" for MLPPolicy 
    train_epochs = 200
    learning_rate = 1e-3
    batch_size = 8
    seq_window_len = 0 #  30  # sequence length for the episode window dataset. Corresponding number of control loops: seq_window_len*update_step

    '''prepare the dataset'''

    dataset_base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/off_policy_4/"
    return_function = effort_and_energy_based_return_function  # effort_and_energy_based_return_function or effort_based_return_function
    dataset = load_trajectories(dataset_base_dir, return_function, env_step=update_step, reload_data=reload_data, load_terminal_in_obs=load_fixing_terminal_in_obs) # observation (B, D)

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
            f.write(f"dataset_base_dir: {dataset_base_dir}\n")
            f.write(f"return_function: {return_function.__name__}\n")

        policy, encoder = train_weighted_bc(
                                train_dataset,
                                obs_dim=obs_dim,
                                act_dim=act_dim,
                                num_epochs=train_epochs,
                                batch_size=batch_size,
                                lr=learning_rate,
                                policy_type=policy_type,
                                log_dir=log_dir,
                                save_epoch=10,
                                include_terminal_in_obs=load_fixing_terminal_in_obs
                            )
    else:
        log_stored_folder = "20250805_184513" # "20250805_174938" #20250629_170553 #20250702_160901 20250707_215329 #20250709_141454 # 20250725_162934 #20250727_182152
        log_dir = os.path.join(log_base_dir, log_stored_folder)

    policy_path = os.path.join(log_dir, "trained_BC_policy.pth")
    encoder_path = os.path.join(log_dir, "dynamics_encoder.pth")

    if train:
        # save the policy
        torch.save(policy.state_dict(), policy_path)
        export_policy_to_onnx(policy, obs_dim, seq_window_len, z_dim=32, policy_type=policy_type, export_path=os.path.join(log_dir, "trained_BC_policy.onnx"))

        # save the encoder
        torch.save(encoder.state_dict(), encoder_path)
        encoder.set_export(True)  # Enable ONNX export
        export_encoder_to_onnx(encoder, export_path=os.path.join(log_dir, "dynamics_encoder.onnx"))
        encoder.set_export(False)  # Disable ONNX export after saving

    '''Evaluate the policy'''
    # load the policy
    encoder = DynamicsRNNEncoder(obs_dim-1).to(device)
    encoder.load_state_dict(torch.load("d3rlpy_logs/Dynamics_Encoder/best_encoder_2_force_only_encoder_decoder.pth"))
    encoder.eval()  # Encoder is used for feature extraction only
    # encoder.set_export(True)

    if policy_type == "gru_with_encoder":
        reload_policy = StochasticRNNPolicy(obs_dim, act_dim).to(device)
        # reload_policy = MLPPolicy(obs_dim, act_dim)  # For testing with MLP policy
    else:
        raise ValueError(f"Unknown policy type: {policy_type}. Use 'stochastic' or 'deterministic'.")
    reload_policy.load_state_dict(torch.load(policy_path))
    reload_policy.eval()

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=partial(collate_variable_episodes, include_terminal=load_fixing_terminal_in_obs))

    eval_avg_loss = evaluate_policy(
        reload_policy,
        encoder,
        val_dataloader,
        loss_fn=masked_mse,
        policy_type=policy_type,
        if_plot=True
    )
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
from policies import *
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
    policy_mask = torch.ones(obs_padded.shape[0], dtype=torch.bool)
    encoder_mask = torch.arange(obs_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T_max)

    return obs_padded, act_padded, torch.tensor(weights), policy_mask,encoder_mask, lengths

def compute_weighted_masked_bc_loss_stepwise(obs_padded, act_padded, weights, encoder_window_length, policy_mask, encoder_mask, policy, encoder, policy_type="gru", use_encoder=True, train=True):
    """
    Args:
        obs_seq:      (B, T, S, obs_dim)
        act_padded:   (B, T, act_dim)
        weights:      (B,)
        mask:         (B, T)
        policy:       Sequence-based policy taking (B*T, S, obs_dim)
    """
    B, T, obs_dim = obs_padded.shape
    terminal_index = encoder_mask.sum(dim=1)

    if policy_type == "stochastic_gru":
        policy_h = policy.init_hidden(batch_size=B, device=obs_padded.device)  # Initialize hidden state
    if use_encoder:
        encoder_h = encoder.init_hidden(batch_size=B, device=obs_padded.device)  # Initialize encoder hidden state
    log_prob_seq = []
    pred_seq = []

    if train:
        for t in range(T):
            # 1. One-step observation
            obs_step = obs_padded[:, t, :]  # (B, 1, obs_dim)
            act_step = act_padded[:, t, :]  # (B, 1, act_dim)

            # 2. Encoder windowed input
            if use_encoder:
                if t >= terminal_index[0].item():
                    z_dyn = torch.zeros_like(z_dyn)
                if encoder_window_length > 1:
                    windowed_input = get_obs_padded_window(obs_padded, t, encoder_window_length, window_obs_dim=obs_dim-1)  # (B, window_len, obs_dim-1)
                    z_dyn, encoder_h = encoder(windowed_input, encoder_h)
                else:
                    single_input = obs_step[:, :-1].unsqueeze(1)  # (B, 1, obs_dim-1)
                    z_dyn, encoder_h = encoder(single_input, encoder_h)
            else:
                z_dyn = None

            # TODO: temp solution
            z_dyn = z_dyn.squeeze(1)

            # 3. Policy step-by-step inference
            if policy_type == "stochastic_gru":
                pred_dist_t, policy_h = policy.get_dist_step(obs_step, z_dyn, policy_h)  # (B, act_dim)
            elif policy_type == "stochastic_mlp":
                pred_dist_t = policy.get_dist(obs_step, z_dyn)
            log_prob_t = pred_dist_t.log_prob(act_step).sum(-1) # (B,)
            log_prob_seq.append(log_prob_t)

        log_probs_tensor = torch.stack(log_prob_seq, dim=1)   # (B, T)
        forward_output = log_prob_seq
        loss_per_step = -log_probs_tensor                  # (B, T)
        loss_masked = loss_per_step[policy_mask]  # (num_valid_steps,)
    else:
        for t in range(T):
            # 1. One-step observation
            obs_step = obs_padded[:, t, :]  # (B, 1, obs_dim)

            # 2. Encoder windowed input
            if t >= terminal_index[0].item():
                z_dyn = torch.zeros_like(z_dyn) if use_encoder else None
            windowed_input = get_obs_padded_window(obs_padded, t, encoder_window_length, window_obs_dim=obs_dim-1)  # (B, window_len, obs_dim-1)
            # z_dyn = encoder(windowed_input, lengths=torch.tensor([windowed_input.size(1)]).to(obs_batch.device))  # (B, z_dim)
            z_dyn = encoder(windowed_input) if use_encoder else None

            # 3. Policy single step input
            if policy_type == "stochastic_gru":
                pred_step, h = policy(obs_step, z_dyn, h)
            elif policy_type == "stochastic_mlp":
                pred_step = policy(obs_step, z_dyn)
            pred_seq.append(pred_step)

        pred = torch.stack(pred_seq, dim=1)  # (B, T, act_dim)
        forward_output = pred
        loss_masked = masked_mse(pred, act_padded, policy_mask)

    # Expand weights from (B,) to (B, T)
    weight_per_step = weights.unsqueeze(1).expand(-1, T)  # (B, T)
    weight_masked = weight_per_step[policy_mask]  # (num_valid_steps,)

    eps = 1e-8
    weighted_loss = (loss_masked * weight_masked).sum() / (weight_masked.sum() + eps)
    return forward_output, weighted_loss, weight_masked.sum().item()

def train_weighted_bc(policy, encoder, dataloader, obs_dim, act_dim, num_epochs=50, batch_size=8, lr=1e-3, policy_type="deterministic", use_encoder=True, log_dir=None, save_epoch=50):
    encoder.train()
    policy.train()
    
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # Early stopping variables
    best_loss = float('inf')  # Initialize best loss to infinity
    best_eval_loss = float('inf')
    patience_counter = 0      # Counter for early stopping

    for epoch in range(num_epochs):
        encoder.train()
        policy.train()  # Set policy to training mode

        epoch_loss = 0.0

        total_weight = 0.0
        for obs_batch, act_batch, weights, policy_mask, encoder_mask, lengths in dataloader:
            obs_batch, act_batch, weights, policy_mask, encoder_mask, lengths = obs_batch.to(device), act_batch.to(device), weights.to(device), policy_mask.to(device), encoder_mask.to(device), lengths.to(device)
            optimizer.zero_grad()
            # weighted_loss, batch_weight = compute_weighted_masked_bc_loss(obs_batch, act_batch, weights, lengths, mask, policy, encoder, policy_type)
            policy_pred, weighted_loss, batch_weight = compute_weighted_masked_bc_loss_stepwise(obs_batch, act_batch, weights, encoder_window_length, 
                                                                                   policy_mask, encoder_mask, policy, encoder, policy_type, use_encoder, train=True)
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            total_weight += batch_weight

        avg_loss = epoch_loss / max(total_weight, 1e-8)
        print(f"Epoch {epoch}: Epoch loss {epoch_loss:.4f}, Weighted Avg Loss = {avg_loss:.8f}")
        wandb.log({"epoch": epoch, "weighted_bc_loss": avg_loss})

        if epoch % save_epoch == 0:
            if log_dir:
                policy_path = os.path.join(log_dir, f"policy_epoch_{epoch}.pth")
                torch.save(policy.state_dict(), policy_path)
                export_policy_to_onnx(policy, obs_dim, 0, policy_type=policy_type, export_path=os.path.join(log_dir, f"trained_BC_policy_{epoch}.onnx"), step_wise=True)
                print(f"Saved policy at {policy_path}")

                encoder_path = os.path.join(log_dir, f"encoder_epoch_{epoch}.pth")
                torch.save(encoder.state_dict(), encoder_path)
                # encoder.set_export(True)  # Enable ONNX export
                export_encoder_to_onnx(encoder, export_path=os.path.join(log_dir, f"trained_BC_encoder_{epoch}.onnx"))   
                print(f"Saved encoder at {encoder_path}")
                # encoder.set_export(False)  # Disable ONNX export after saving

        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0  # Reset patience counter
            # Save the best model
            if log_dir:
                best_model_path = os.path.join(log_dir, "best_policy.pth")
                torch.save(policy.state_dict(), best_model_path)
                print(f"Best model saved at epoch {epoch} with avg loss {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best loss: {best_loss:.4f})")

        # Check if patience is exceeded
        if patience_counter >= 10:
            print(f"Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.4f}")
            break

    return policy, encoder

def get_obs_padded_window(obs_batch, t, window_size, window_obs_dim=6):
    B, _, D = obs_batch.shape
    start = max(0, t - window_size + 1)
    pad_len = window_size - (t - start + 1)  # how much to pad

    # Extract actual window
    window = obs_batch[:, start:t+1, :window_obs_dim]  # shape: (B, actual_len, window_obs_dim)

    if pad_len > 0:
        # Pad with the last valid observation
        pad = window[:, -1:, :].expand(B, pad_len, window_obs_dim)  # repeat last valid obs
        window = torch.cat([pad, window], dim=1)      # shape: (B, window_size, window_obs_dim)

    return window

def evaluate_weighted_bc(policy, encoder, val_loader, loss_fn, encoder_window_length=30, policy_type="gru", use_encoder=True, if_plot=False):
    # batch size = 1 for evaluation
    policy.eval()
    encoder.eval()
    total_loss = 0.0
    total_weight = 0.0

    with torch.no_grad():
        for obs_batch, act_batch, weights, policy_mask, encoder_mask, lengths in val_loader:
            obs_batch = obs_batch.to(device)       # (1, T, obs_dim)
            act_batch = act_batch.to(device)       # (1, T, act_dim)
            weights = weights.to(device)           # (1,)
            policy_mask = policy_mask.to(device)   # (1, T)
            encoder_mask = encoder_mask.to(device) # (1, T)
            lengths = lengths.to(device)           # (1,)
            
            pred, masked_weighted_loss, masked_weight = compute_weighted_masked_bc_loss_stepwise(obs_batch, act_batch, weights, encoder_window_length,
                                                                                                policy_mask, encoder_mask, policy, encoder, policy_type, use_encoder, train=False)
            total_loss += masked_weighted_loss.item()
            total_weight += masked_weight

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
                    plt.title(f"Episode {i + 1} - Loss: {masked_weighted_loss.item():.4f}")
                    plt.show()
                    plt.close(fig)

    return total_loss / max(total_weight, 1e-8)

if __name__ == "__main__":
    train = True # Set to False to skip training and only evaluate
    reload_data= False # Set to True to reload the dataset from raw txt files, False to use the cached npz dataset stored from previous runs
    load_fixing_terminal_in_obs = True # Set to True to load the fixing terminal as an extra input dimension in the observation, False to ignore it
    update_step = 5  # update the policy output every 5 robot control loops, i.e. 200Hz for the 1000Hz robot control frequency
    policy_type = "stochastic_mlp"  # stochastic_gru or stochastic_mlp
    use_encoder = True
    train_encoder = True
    if not use_encoder:
        train_encoder = False  # if no encoder is used, no need to train it
    train_epochs = 200
    learning_rate = 1e-3
    batch_size = 8
    seq_window_len = 0 #  30  # sequence length for the episode window dataset. Corresponding number of control loops: seq_window_len*update_step
    encoder_window_length = 0  # length of the sliding window for the encoder, -1 to use full input sequence
    encoder_z_dim = 16
    encoder_hidden_dim = 64
    policy_z_dim = encoder_z_dim if use_encoder else 0  # z_dim for the policy, 0 if no encoder is used
    policy_hidden_dim = 128
    encoder_filename = "predict_f_x_v/best_encoder_2_encoding_True_input_False.pth"
    
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
    if not train:
        log_stored_folder = "20250807_220600" # "20250805_174938" #20250629_170553 #20250702_160901 20250707_215329 #20250709_141454 # 20250725_162934 #20250727_182152
        log_dir = os.path.join(log_base_dir, log_stored_folder)
    else:
        log_dir = os.path.join(log_base_dir, time.strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # save the training config to log_dir
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump({
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "seq_window_len": seq_window_len,
                "num_epochs": train_epochs,
                "batch_size": batch_size,
                "lr": learning_rate,
                "policy_type": policy_type,
                "use_encoder": use_encoder,
                "train_encoder": train_encoder,
                "dataset_base_dir": dataset_base_dir,
                "return_function": return_function.__name__,
                "encoder_z_dim": encoder_z_dim,
                "encoder_hidden_dim": encoder_hidden_dim,
                "policy_hidden_dim": policy_hidden_dim
            }, f, indent=4)

        run_name = f"Clip_BC_episodewise_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb.init(project="BC_episodewise", name=run_name, config={
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": train_epochs
        })

        collate_function = partial(collate_variable_episodes, include_terminal=load_fixing_terminal_in_obs)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_function)
        
        encoder = DynamicsRNNEncoder(input_dim=obs_dim-1, z_dim=encoder_z_dim, hidden_dim=encoder_hidden_dim).to(device)
        encoder.load_state_dict(torch.load(f"d3rlpy_logs/Dynamics_Encoder/{encoder_filename}"))
        if not train_encoder:
            for param in encoder.parameters(): # freeze encoder parameters
                param.requires_grad = False

        if policy_type == "stochastic_gru":
            policy = StochasticRNNPolicyStepwise(obs_dim, act_dim, z_dim=policy_z_dim, hidden_dim=policy_hidden_dim).to(device)
        elif policy_type == "stochastic_mlp":
            policy = StochasticMLPPolicy(obs_dim, act_dim, z_dim=policy_z_dim).to(device)

        policy, encoder = train_weighted_bc(
                                policy,
                                encoder,
                                train_dataloader,
                                obs_dim=obs_dim,
                                act_dim=act_dim,
                                num_epochs=train_epochs,
                                batch_size=batch_size,
                                lr=learning_rate,
                                policy_type=policy_type,
                                use_encoder=use_encoder,
                                log_dir=log_dir,
                                save_epoch=10,
                            )

    policy_path = os.path.join(log_dir, "trained_BC_policy.pth")
    encoder_path = os.path.join(log_dir, "dynamics_encoder.pth")

    if train:
        # save the policy
        torch.save(policy.state_dict(), policy_path)
        export_policy_to_onnx(policy, obs_dim, seq_window_len, policy_type=policy_type, export_path=os.path.join(log_dir, "trained_BC_policy.onnx"), step_wise=True)

        # save the encoder
        torch.save(encoder.state_dict(), encoder_path)
        encoder.set_export(True)  # Enable ONNX export
        export_encoder_to_onnx(encoder, export_path=os.path.join(log_dir, "dynamics_encoder.onnx"))
        encoder.set_export(False)  # Disable ONNX export after saving

    '''Evaluate the policy'''
    # load config
    with open(os.path.join(log_dir, "config.json"), "r") as f:
        reload_config = json.load(f)
    reload_policy_type = reload_config.get("policy_type", "gru")
    print(f"Reloading policy type: {reload_policy_type}")
    reload_use_encoder = reload_config.get("use_encoder", False)
    reload_encoder_z_dim = reload_config.get("encoder_z_dim", encoder_z_dim)
    reload_encoder_hidden_dim = reload_config.get("encoder_hidden_dim", encoder_hidden_dim)
    reload_policy_hidden_dim = reload_config.get("policy_hidden_dim", policy_hidden_dim)

    # load the policy
    encoder = DynamicsRNNEncoder(input_dim=obs_dim-1, z_dim=reload_encoder_z_dim, hidden_dim=reload_encoder_hidden_dim).to(device)
    encoder.load_state_dict(torch.load(f"d3rlpy_logs/Dynamics_Encoder/{encoder_filename}"))
    encoder.eval()  # Encoder is used for feature extraction only
    # encoder.set_export(True)

    # # validae encoder onnx model
    # import onnx
    # encoder_onnx_path = os.path.join(log_dir, "dynamics_encoder.onnx")
    # encoder_onnx_model = onnx.load(encoder_onnx_path)
    # print([i.name for i in encoder_onnx_model.graph.input])
    # print([i.name for i in encoder_onnx_model.graph.output])

    reload_policy_z_dim = reload_encoder_z_dim if reload_use_encoder else 0  # z_dim for the policy, 0 if no encoder is used
    if reload_policy_type == "stochastic_gru":
        reload_policy = StochasticRNNPolicyStepwise(obs_dim, act_dim, z_dim=reload_policy_z_dim, hidden_dim=reload_policy_hidden_dim).to(device)  # With dynamics encoding
    elif reload_policy_type == "stochastic_mlp":
        reload_policy = StochasticMLPPolicy(obs_dim, act_dim, z_dim=reload_policy_z_dim).to(device)
    else:
        raise ValueError(f"Unsupported policy type: {reload_policy_type}")
    reload_policy.load_state_dict(torch.load(policy_path))
    reload_policy.eval()

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=partial(collate_variable_episodes, include_terminal=load_fixing_terminal_in_obs))

    eval_avg_loss = evaluate_weighted_bc(
        reload_policy,
        encoder,
        val_dataloader,
        encoder_window_length=encoder_window_length, # -1 to use full input sequence, otherwise use a sliding window of length encoder_window_length
        loss_fn=masked_mse,
        policy_type=reload_policy_type,
        use_encoder=reload_use_encoder,
        if_plot=True
    )

    print(f"Evaluation completed. Average loss: {eval_avg_loss:.4f}")
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
from dynamics_encoder_decoder import DynamicsRNNEncoder, masked_mse, export_dynamic_encoder_to_onnx
from return_functions import *

from build_episode_dataset import load_trajectories, normalize_obs, unnormalize_obs, normalize_acts, unnormalize_acts
import wandb
import numpy as np
import os
import matplotlib.pyplot as plt
from d3rlpy.dataset import EpisodeWindowDataset
from functools import partial

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_LABELS = 12

def load_slope_distribution(slope_file):
    with open(slope_file, 'r') as f:
        slope_raw_dict = json.load(f)

    slope_distribution = {}

    for i in range(NUM_LABELS):
        label = i+1
        slope_distribution[label] = {
            "mean": 0.0,
            "sigma": 1e-10
        }
        for key, value in slope_raw_dict.items():
            if label in value["config_labels"]:
                slope_distribution[label]["mean"] = value["slope_mean_list"][-1]
                slope_distribution[label]["sigma"] = value["slope_var_list"][-1]
                break
    return slope_distribution

SLOPE_DISTRIBUTION = load_slope_distribution("/home/tp2/Documents/kejia/clip_fixing_dataset/off_policy_4/slope_distribution.json")
# Precompute arrays for all possible labels (1-based indexing)
SLOPE_MEANS = np.array([SLOPE_DISTRIBUTION[i+1]["mean"] for i in range(NUM_LABELS)])  # shape: [NUM_LABELS]
SLOPE_SIGMAS = np.array([SLOPE_DISTRIBUTION[i+1]["sigma"] for i in range(NUM_LABELS)])  # shape: [NUM_LABELS]

def slope_z_wrapper(labels):
    """
    Convert labels to z values based on the slope distribution.
    Args:
        labels: (B, ) tensor of labels
        slope_distribution: dict containing mean and sigma for each label
    Returns:
        z: (B, 3) tensor of z values
    """
    z = []
    labels_np = labels.cpu().numpy()-1
    mu = torch.from_numpy(SLOPE_MEANS[labels_np]).to(labels.device)      # (B,)
    sigma = torch.from_numpy(SLOPE_SIGMAS[labels_np]).to(labels.device)  # (B,)
    c = torch.from_numpy(np.random.uniform(0.5, 1.0, size=labels_np.shape)).to(labels.device)  # (B,)
    z = torch.stack([mu - c * sigma, mu, mu + c * sigma], dim=1).float()   # (B, 3)
    return z

    # for label in labels:
    #     label_int = int(label.item())
    #     mu = slope_distribution[label_int]["mean"]
    #     sigma = slope_distribution[label_int]["sigma"]
    #     z.append(sigma_slope_z(mu, sigma))
    # return torch.stack(z, dim=0).to(device)  # (B, 3)

def sigma_slope_z(mu, sigma):
    # sample c from [0.5, 1]
    c = np.random.uniform(0.5, 1.0)
    return torch.Tensor([mu-c*sigma, mu, mu+c*sigma])

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
    obs_list, act_list, weights, lengths, labels = [], [], [], [], []

    for data in batch:
        if include_terminal:
            obs, act, weight, terminals, label = data
            terminals = terminals.view(-1, 1)
            obs = normalize_obs(obs)
            obs = torch.cat([obs, terminals], dim=-1)
            act = normalize_acts(act)
            terminal_idx = (terminals.squeeze() == 1).nonzero(as_tuple=True)[0]
            ep_len = terminal_idx[0].item() + 1 if len(terminal_idx) > 0 else obs.shape[0]
        else:
            obs, act, weight, label = data
            ep_len = obs.shape[0]

        obs_list.append(obs)
        act_list.append(act)
        weights.append(weight)
        lengths.append(ep_len)
        labels.append(label)

    lengths = torch.tensor(lengths)
    labels = torch.tensor(labels, dtype=torch.int64)
    obs_padded = pad_sequence(obs_list, batch_first=True)
    act_padded = pad_sequence(act_list, batch_first=True)
    policy_mask = torch.ones(obs_padded.shape[0], obs_padded.shape[1], dtype=torch.bool)
    encoder_mask = torch.arange(obs_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)  # (B, T_max)

    return obs_padded, act_padded, torch.tensor(weights), policy_mask, encoder_mask, lengths, labels


def compute_weighted_masked_bc_loss_stepwise(obs_padded, act_padded, weights, policy_window_length, encoder_window_length, 
                                             policy_mask, encoder_mask, policy, encoder, labels,
                                             policy_type="gru", use_encoder="gru", train=True):
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
    if use_encoder == "gru":
        encoder_h = encoder.init_hidden(batch_size=B, device=obs_padded.device)  # Initialize encoder hidden state
    log_prob_seq = []
    pred_seq = []
    entropy_seq = []

    for t in range(T):
        # 1. One-step observation
        obs_step = obs_padded[:, t, :]  # (B, 1, obs_dim)
        act_step = act_padded[:, t, :]  # (B, 1, act_dim)

        # 2. Encoder windowed input
        if use_encoder == "gru":
            if encoder_window_length >= 1:
                windowed_input = get_obs_padded_window(obs_padded, t, encoder_window_length, window_obs_dim=obs_dim-1)  # (B, window_len, obs_dim-1)
                z_dyn, encoder_h = encoder(windowed_input, encoder_h)
            # else:
            #     single_input = obs_step[:, :-1].unsqueeze(1)  # (B, 1, obs_dim-1)
            #     z_dyn, encoder_h = encoder(single_input, encoder_h)
            if t >= terminal_index[0].item():
                z_dyn = torch.zeros_like(z_dyn)
            
            # TODO: temp solution
            # z_dyn = z_dyn.squeeze(1)
        elif use_encoder == "slope":
            z_dyn = slope_z_wrapper(labels)
            z_dyn = z_dyn.unsqueeze(1)  # (B, 1, z_dim)
        else: # null
            z_dyn = None
            encoder_h = None

        # 3. Policy step-by-step inference
        if policy_type == "stochastic_gru":
            if train:
                pred_dist_t, policy_h = policy.get_dist_step(obs_step, z_dyn, policy_h)  # (B, act_dim)
            else:
                pred_step, policy_h = policy(obs_step, z_dyn, policy_h)  # (B, act_dim)
        elif policy_type == "stochastic_mlp":
            terminal_t = obs_step[:, -1].unsqueeze(1)  # (B, 1)
            terminal_t = terminal_t.unsqueeze(1)  # (B, 1, 1)
            if policy_window_length >= 1:
                windowed_input = get_obs_padded_window(obs_padded, t, policy_window_length, window_obs_dim=obs_dim-1)
                if train:
                    pred_dist_t = policy.get_dist(windowed_input, terminal_t, z_dyn)
                else:
                    pred_step = policy(windowed_input, terminal_t, z_dyn)
            # else:
            #     single_input = obs_step.unsqueeze(1)  # (B, 1, obs_dim)
            #     if train:
            #         pred_dist_t = policy.get_dist(single_input, z_dyn)
            #     else:
            #         pred_step = policy(single_input, z_dyn)

        if train:
            # single_act = act_step.unsqueeze(1)  # (B, act_dim)
            log_prob_t = pred_dist_t.log_prob(act_step).sum(-1) # (B,)
            log_prob_seq.append(log_prob_t)
            entropy = pred_dist_t.entropy().sum(-1)          # per-step, sum over dims
            entropy_seq.append(entropy)
        else:
            pred_seq.append(pred_step)
    
    # Expand weights from (B,) to (B, T)
    weight_per_step = weights.unsqueeze(1).expand(-1, T)  # (B, T)
    weight_masked = weight_per_step[policy_mask]  # (num_valid_steps,)

    if train: 
        log_probs_tensor = torch.stack(log_prob_seq, dim=1)   # (B, T)
        forward_output = log_prob_seq
        loss_per_step = -log_probs_tensor                  # (B, T)
        loss_masked = loss_per_step[policy_mask]  # (num_valid_steps,)
        entropy_tensor = torch.stack(entropy_seq, dim=1)  # (B, T)
        entropy_masked = entropy_tensor[policy_mask]      # (num_valid_steps,)

        eps = 1e-8
        weighted_loss = (loss_masked * weight_masked).sum() / (weight_masked.sum() + eps)
        entropy_coef = 0.005
        entropy_reg = entropy_masked.mean()
        final_loss = weighted_loss - entropy_coef * entropy_reg
    else:
        # pred = torch.cat(pred_seq, dim=1)  # (B, T, act_dim)
        pred = torch.stack(pred_seq, dim=1)  # (B, T, act_dim)
        forward_output = pred
        # denormalize actions in evaluation for more realistic metrics
        unnormalized_pred = unnormalize_acts(pred)
        unnormalized_act = unnormalize_acts(act_padded)
        weighted_loss = masked_mse(unnormalized_pred, unnormalized_act, policy_mask, masked_weight=weight_masked)  # (B, T)
        final_loss = weighted_loss
    return forward_output, final_loss, weight_masked.sum().item()

def train_weighted_bc(policy, encoder, dataloader, obs_dim, act_dim, policy_window_len, encoder_window_len, num_epochs=50, batch_size=8, lr=1e-3, policy_type="deterministic", use_encoder="gru", log_dir=None, save_epoch=50):
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
        for obs_batch, act_batch, weights, policy_mask, encoder_mask, lengths, labels in dataloader:
            obs_batch, act_batch, weights, policy_mask, encoder_mask, lengths, labels = obs_batch.to(device), act_batch.to(device), weights.to(device), policy_mask.to(device), encoder_mask.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            # weighted_loss, batch_weight = compute_weighted_masked_bc_loss(obs_batch, act_batch, weights, lengths, mask, policy, encoder, policy_type)
            policy_pred, weighted_loss, batch_weight = compute_weighted_masked_bc_loss_stepwise(obs_batch, act_batch, weights, policy_window_len, encoder_window_len, 
                                                                                   policy_mask, encoder_mask, policy, encoder, labels, policy_type, use_encoder, train=True)
            weighted_loss.backward()
            optimizer.step()

            epoch_loss += weighted_loss.item()
            total_weight += batch_weight

        avg_loss = epoch_loss / max(total_weight, 1e-8)

        val_loss = evaluate_weighted_bc(policy, encoder, dataloader, masked_mse, policy_window_length=policy_window_len, encoder_window_length=encoder_window_len, policy_type=policy_type, use_encoder=use_encoder, if_plot=False)
        print(f"Epoch {epoch}: Epoch loss {epoch_loss:.4f}, Weighted Avg Loss = {avg_loss:.8f}, Val Loss = {val_loss:.8f}")
        wandb.log({"epoch": epoch, "weighted_bc_loss": avg_loss, "val_loss": val_loss})

        if epoch % save_epoch == 0:
            if log_dir:
                policy_path = os.path.join(log_dir, f"policy_epoch_{epoch}.pth")
                torch.save(policy.state_dict(), policy_path)
                export_policy_to_onnx(policy, obs_dim, seq_window_len, policy_type=policy_type, export_path=os.path.join(log_dir, f"trained_BC_policy_{epoch}.onnx"), step_wise=True)
                print(f"Saved policy at {policy_path}")

                encoder_path = os.path.join(log_dir, f"encoder_epoch_{epoch}.pth")
                torch.save(encoder.state_dict(), encoder_path)
                # encoder.set_export(True)  # Enable ONNX export
                export_dynamic_encoder_to_onnx(encoder, export_path=os.path.join(log_dir, f"trained_BC_encoder_{epoch}.onnx"))   
                print(f"Saved encoder at {encoder_path}")
                # encoder.set_export(False)  # Disable ONNX export after saving

        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0  # Reset patience counter
            # Save the best model
            if log_dir:
                best_model_path = os.path.join(log_dir, "best_policy.pth")
                torch.save(policy.state_dict(), best_model_path)
                print(f"Best model saved at epoch {epoch} with validation loss {best_loss:.8f}")

                best_encoder_path = os.path.join(log_dir, "best_encoder.pth")
                torch.save(encoder.state_dict(), best_encoder_path)
                print(f"Best encoder saved at epoch {epoch} with validation loss {best_loss:.8f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epochs (best loss: {best_loss:.8f})")

        # Check if patience is exceeded
        if patience_counter >= 5:
            print(f"Early stopping triggered at epoch {epoch}. Best loss: {best_loss:.8f}")
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

def evaluate_weighted_bc(policy, encoder, val_loader, loss_fn, policy_window_length=30, encoder_window_length=30, policy_type="gru", use_encoder="gru", if_plot=False):
    # batch size = 1 for evaluation
    policy.eval()
    encoder.eval()
    total_loss = 0.0
    total_weight = 0.0

    with torch.no_grad():
        for obs_batch, act_batch, weights, policy_mask, encoder_mask, lengths, labels in val_loader:
            obs_batch = obs_batch.to(device)       # (1, T, obs_dim)
            act_batch = act_batch.to(device)       # (1, T, act_dim)
            weights = weights.to(device)           # (1,)
            policy_mask = policy_mask.to(device)   # (1, T)
            encoder_mask = encoder_mask.to(device) # (1, T)
            lengths = lengths.to(device)           # (1,)
            labels = labels.to(device)             # (1,)

            pred, masked_weighted_loss, masked_weight = compute_weighted_masked_bc_loss_stepwise(obs_batch, act_batch, weights, policy_window_length, encoder_window_length,
                                                                                                policy_mask, encoder_mask, policy, encoder, labels, policy_type, use_encoder, train=False)
            total_loss += masked_weighted_loss.item()
            total_weight += masked_weight

            # Optional visualization
            if if_plot:
                for b in range(obs_batch.size(0)):     #
                    unnormalized_obs_ep = unnormalize_obs(obs_batch[b, :, :])
                    unnormalized_act_ep = unnormalize_acts(act_batch[b, :, :])
                    unnormalize_prediction = unnormalize_acts(pred[b, :, :])

                    T = unnormalized_obs_ep.size(0)
                    fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
                    axs[0].plot(range(T), unnormalized_obs_ep[:, 0].cpu().numpy(), label="Observation", color="green")
                    axs[0].plot(range(T), unnormalized_act_ep[:, 0].cpu().numpy(), label="Ground Truth Action", color="red")
                    axs[0].plot(range(T), unnormalize_prediction[:, 0].cpu().numpy(), label="Predicted Action", color="blue", linestyle="--")
                    axs[0].set_ylabel("Stretch")
                    axs[0].legend()

                    axs[1].plot(range(T), unnormalized_obs_ep[:, 3].cpu().numpy(), label="Observation", color="green")
                    axs[1].plot(range(T), unnormalized_act_ep[:, 1].cpu().numpy(), label="Ground Truth Action", color="red")
                    axs[1].plot(range(T), unnormalize_prediction[:, 1].cpu().numpy(), label="Predicted Action", color="blue", linestyle="--")
                    axs[1].set_ylabel("Push")
                    axs[1].legend()

                    plt.xlabel("Time Step")
                    plt.title(f"Episode {b + 1} - Loss: {masked_weighted_loss.item():.4f}")
                    plt.show()
                    plt.close(fig)

    return total_loss / max(total_weight, 1e-8)

if __name__ == "__main__":
    train = False # Set to False to skip training and only evaluate
    reload_data= False # Set to True to reload the dataset from raw txt files, False to use the cached npz dataset stored from previous runs
    load_fixing_terminal_in_obs = True # Set to True to load the fixing terminal as an extra input dimension in the observation, False to ignore it
    update_step = 5  # update the policy output every 5 robot control loops, i.e. 200Hz for the 1000Hz robot control frequency
    policy_type = "stochastic_mlp"  # stochastic_gru or stochastic_mlp
    use_encoder = "slope" # gru or slope or null
    train_encoder = True
    if use_encoder == "null":
        train_encoder = False  # if no encoder is used, no need to train it
    train_epochs = 200
    learning_rate = 1e-3
    batch_size = 8
    seq_window_len = 30 #  30  # sequence length for the episode window dataset. Corresponding number of control loops: seq_window_len*update_step
    encoder_window_length = 0  # length of the sliding window for the encoder, -1 to use full input sequence
    encoder_z_dim = 16
    encoder_hidden_dim = 64
    policy_z_dim = 0
    if use_encoder == "slope":
        policy_z_dim = 3
    elif use_encoder == "gru":
        policy_z_dim = encoder_z_dim
    policy_hidden_dim = 128
    encoder_filename = "predict_f_x_v/best_encoder_2_encoding_True_input_False.pth"
    
    '''---------------------------------------------prepare the dataset---------------------------------------'''

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

    '''--------------------------------------------train the policy-------------------------------------------'''
    log_base_dir = "/home/tp2/Documents/kejia/d3rlpy/d3rlpy_logs/Clip_Weighted_BC/"
    if train:
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
            policy = StochasticMLPPolicy(obs_dim-1, act_dim, seq_len=seq_window_len, z_dim=policy_z_dim).to(device) # obs_dim excludes the terminal dimension

        policy, encoder = train_weighted_bc(
                                policy,
                                encoder,
                                train_dataloader,
                                obs_dim=obs_dim,
                                act_dim=act_dim,
                                policy_window_len=seq_window_len,
                                encoder_window_len=encoder_window_length,
                                num_epochs=train_epochs,
                                batch_size=batch_size,
                                lr=learning_rate,
                                policy_type=policy_type,
                                use_encoder=use_encoder,
                                log_dir=log_dir,
                                save_epoch=10,
                            )

        policy_path = os.path.join(log_dir, "best_policy.pth")
        encoder_path = os.path.join(log_dir, "encoder_epoch_30.pth")

        # save the policy
        torch.save(policy.state_dict(), policy_path)
        export_policy_to_onnx(policy, obs_dim, seq_window_len, policy_type=policy_type, export_path=os.path.join(log_dir, "trained_BC_policy.onnx"), step_wise=True)

        # save the encoder
        torch.save(encoder.state_dict(), encoder_path)
        export_dynamic_encoder_to_onnx(encoder, export_path=os.path.join(log_dir, "dynamics_encoder.onnx"))

    '''---------------------------------------------Evaluate the policy-------------------------------------'''
    # paths
    if not train:
        log_stored_folder = "20250811_110235" # "20250805_174938" #20250629_170553 #20250702_160901 20250707_215329 #20250709_141454 # 20250725_162934 #20250727_182152
        log_dir = os.path.join(log_base_dir, log_stored_folder)
    
    policy_path = os.path.join(log_dir, "best_policy.pth")
    encoder_path = os.path.join(log_dir, "encoder_epoch_30.pth")

    import onnx
    # load and validat the inputs
    policy_onnx_path = os.path.join(log_dir, "trained_BC_policy_0.onnx")
    policy_onnx_model = onnx.load(policy_onnx_path)
    print([i.name for i in policy_onnx_model.graph.input])
    print([i.name for i in policy_onnx_model.graph.output])


    # load config
    with open(os.path.join(log_dir, "config.json"), "r") as f:
        reload_config = json.load(f)
    reload_policy_type = reload_config.get("policy_type", "gru")
    print(f"Reloading policy type: {reload_policy_type}")
    reload_use_encoder = reload_config.get("use_encoder", "null")
    reload_encoder_z_dim = reload_config.get("encoder_z_dim", encoder_z_dim)
    reload_encoder_hidden_dim = reload_config.get("encoder_hidden_dim", encoder_hidden_dim)
    reload_policy_hidden_dim = reload_config.get("policy_hidden_dim", policy_hidden_dim)
    reload_seq_window_len = reload_config.get("seq_window_len", seq_window_len)

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

    reload_policy_z_dim = 0
    if reload_use_encoder == "slope":
        reload_policy_z_dim = 3
    elif reload_use_encoder == "gru":
        reload_policy_z_dim = reload_encoder_z_dim
    if reload_policy_type == "stochastic_gru":
        reload_policy = StochasticRNNPolicyStepwise(obs_dim, act_dim, z_dim=reload_policy_z_dim, hidden_dim=reload_policy_hidden_dim).to(device)  # With dynamics encoding
    elif reload_policy_type == "stochastic_mlp":
        reload_policy = StochasticMLPPolicy(obs_dim-1, act_dim, seq_len=seq_window_len, z_dim=reload_policy_z_dim).to(device)
    else:
        raise ValueError(f"Unsupported policy type: {reload_policy_type}")
    reload_policy.load_state_dict(torch.load(policy_path))
    reload_policy.eval()    

    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=partial(collate_variable_episodes, include_terminal=load_fixing_terminal_in_obs))

    eval_avg_loss = evaluate_weighted_bc(
        reload_policy,
        encoder,
        val_dataloader,
        policy_window_length=reload_seq_window_len,  # -1 to use full input sequence, otherwise use a sliding window of length policy_window_length
        encoder_window_length=encoder_window_length, # -1 to use full input sequence, otherwise use a sliding window of length encoder_window_length
        loss_fn=masked_mse,
        policy_type=reload_policy_type,
        use_encoder=reload_use_encoder,
        if_plot=True
    )

    print(f"Evaluation completed. Average loss: {eval_avg_loss:.4f}")
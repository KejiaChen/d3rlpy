import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from build_episode_dataset import load_trajectories
from return_functions import *
from dynamics_encoder_decoder import *
from train_ciip_fixing_episodewise_rnn import get_obs_padded_window

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Collate function with padding ---
def normalize_obs(obs):
    obs = obs.clone()
    # Force: [0, 30] → scale to [0, 1]
    obs[:, 0] /= 30.0  # stretch force
    obs[:, 3] /= 30.0  # push force

    # Distance: normalize using global mean/std or empirical range
    # Estimate these from your dataset (preprocess)
    distance_scale = 0.05  # or use empirical max (e.g., 0.015 or 0.025)
    obs[:, 1] /= distance_scale  # stretch displacement
    obs[:, 4] /= distance_scale  # push displacement

    # # Velocity: estimate from data (e.g., max ~0.05 m/s)
    velocity_scale = 0.1
    obs[:, 2] /= velocity_scale
    obs[:, 5] /= velocity_scale
    return obs

def collate_dyn_embedding_2(batch):
    input_seqs = []
    target_seqs = []
    target_initials = [] 
    lengths = []

    for data in batch:
        obs, act, weight, terminals = data
        terminals = np.array(terminals)
        if not np.any(terminals):
            continue

        terminal_index = np.where(terminals)[0][0]
        obs = obs[:terminal_index]
        obs = normalize_obs(obs)

        # input = (f, x, v)
        input_seq = torch.cat([obs[:-1, [0]], obs[:-1, [3]]], dim=-1)  # (T-1, 2)
        # target = (x_i+1, v_i+1)
        target_seq = torch.cat([obs[1:, [1,2]], obs[1:, [4,5]]], dim=-1)       # (T-1, 4)
        target_initial = torch.cat([obs[0, [1,2]], obs[0, [4,5]]], dim=-1)  # shape: (4,)

        input_seqs.append(input_seq)
        target_seqs.append(target_seq)
        target_initials.append(target_initial)
        lengths.append(input_seq.shape[0])

    x_padded = pad_sequence(input_seqs, batch_first=True)
    y_padded = pad_sequence(target_seqs, batch_first=True)
    y_initials = torch.stack(target_initials)  # (B, 4)
    lengths = torch.tensor(lengths)
    mask = torch.arange(x_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

    return x_padded, y_padded, mask, lengths, y_initials

def collate_dyn_embedding(batch):
    full_input_seqs = []
    force_only_input_seqs = []
    target_seqs = []
    target_initials = [] 
    lengths = []

    for data in batch:
        obs, act, weight, terminals = data
        terminals = np.array(terminals)
        if not np.any(terminals):
            continue

        terminal_index = np.where(terminals)[0][0]
        obs = obs[:terminal_index] # only learn dynamics before fixing
        obs = normalize_obs(obs)

        # input = (f, x, v)
        full_input_seq = torch.cat([obs[:-1, [0,1,2]], obs[:-1, [3,4,5]]], dim=-1)  # (T-1, 6)
        # force only = (f)
        force_only_input_seq = torch.cat([obs[:-1, [0]], obs[:-1, [3]]], dim=-1)  # (T-1, 2)
        # target = (x_i+1, v_i+1)
        target_seq = torch.cat([obs[1:, [0,1,2]], obs[1:, [3,4,5]]], dim=-1)       # (T-1, 6)
        # initial (x_0, v_0)
        target_initial = torch.cat([obs[0, [0, 1,2]], obs[0, [3, 4,5]]], dim=-1)  # shape: (6,)

        full_input_seqs.append(full_input_seq)
        force_only_input_seqs.append(force_only_input_seq)
        target_seqs.append(target_seq)
        target_initials.append(target_initial)
        lengths.append(full_input_seq.shape[0])

    full_x_padded = pad_sequence(full_input_seqs, batch_first=True)
    force_only_x_padded = pad_sequence(force_only_input_seqs, batch_first=True)
    y_padded = pad_sequence(target_seqs, batch_first=True)
    y_initials = torch.stack(target_initials)  # (B, 4)
    lengths = torch.tensor(lengths)
    mask = torch.arange(full_x_padded.size(1)).unsqueeze(0) < lengths.unsqueeze(1)

    return full_x_padded, force_only_x_padded, y_padded, mask, lengths, y_initials


def plot_trajectory(y_batch, y_pred, lengths, traj_id, plot_dir="plots"):
    import matplotlib.pyplot as plt
    # Create subplots
    valid_T = lengths.item()  # get actual sequence length for plotting
    time = np.arange(valid_T)
    fig, axs = plt.subplots(3, 1, figsize=(10, 10))

    # Plot distance vs time
    axs[0].plot(time, y_batch[:valid_T, 0].cpu().numpy(), label='Stretch Distance', color='orange')
    axs[0].plot(time, y_batch[:valid_T, 2].cpu().numpy(), label='Push Distance', color='blue')
    axs[0].plot(time, y_pred[:valid_T, 0].cpu().numpy(), label='Reconstructed Stretch Distance', color='orange', linestyle='--')
    axs[0].plot(time, y_pred[:valid_T, 2].cpu().numpy(), label='Reconstructed Push Distance', color='blue', linestyle='--')
    axs[0].set_xlabel('Time (ms)')
    axs[0].set_ylabel('Distance from Start (m)')
    axs[0].set_title(f'Distance vs Time')
    axs[0].legend()
    axs[0].grid()

    # # Plot velocity vs time
    axs[1].plot(time, y_batch[:valid_T, 1].cpu().numpy(), label='Stretch Velocity', color='orange')
    axs[1].plot(time, y_batch[:valid_T, 3].cpu().numpy(), label='Push Velocity', color='blue')
    axs[1].plot(time, y_pred[:valid_T, 1].cpu().numpy(), label='Reconstructed Stretch Velocity', color='orange', linestyle='--')
    axs[1].plot(time, y_pred[:valid_T, 3].cpu().numpy(), label='Reconstructed Push Velocity', color='blue', linestyle='--')
    axs[1].set_xlabel('Time (ms)')
    axs[1].set_ylabel('Velocity (m/s)')
    axs[1].set_title(f'Velocity vs Time')
    axs[1].legend()
    axs[1].grid()

    # Plot force vs time
    axs[2].plot(time, 30*y_batch[:valid_T, 0].cpu().numpy(), label='Ext Stretch Force', color='orange')
    axs[2].plot(time, 30*y_batch[:valid_T, 2].cpu().numpy(), label='Ext Push Force', color='blue')
    axs[2].plot(time, 30*y_pred[:valid_T, 0].cpu().numpy(), label='Reconstructed Ext Stretch Force', color='orange', linestyle='--')
    axs[2].plot(time, 30*y_pred[:valid_T, 2].cpu().numpy(), label='Reconstructed Ext Push Force', color='blue', linestyle='--')
    axs[2].set_xlabel('Time (ms)')
    axs[2].set_ylabel('Force (N)')
    axs[2].set_title(f'Force vs Time')
    axs[2].legend()
    axs[2].grid()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()
    # plt_file = os.path.join(os.path.join(base_dir, traj_dir), f"{traj_dir}_force_plot.png")
    # plt.savefig(plt_file)
    plt.close(fig)


def evaluate_dynamics_encoder(encoder, decoder, encoder_window_length, val_loader, loss_fn, plot=False, decode_using_encoder=True, decode_using_input=True):
    encoder.eval()
    # encoder.set_export(True)  # Set export mode for ONNX export if needed
    decoder.eval()
    total_loss = 0.0
    with torch.no_grad():
        for x_batch, force_only_batch, y_batch, mask, lengths, y_init in val_loader:
            y_pred, y_batch, mask = forward_pass(encoder, decoder, encoder_window_length, x_batch, force_only_batch, y_batch, mask, lengths, y_init, 
                                                decode_using_encoder, decode_using_input)

            loss = loss_fn(y_pred, y_batch, mask)
            total_loss += loss.item()

            if plot:
                for i in range(len(x_batch)):
                    plot_trajectory(y_batch[i], y_pred[i], lengths[i], i, plot_dir="plots")

    encoder.train()
    decoder.train()
    return total_loss / len(val_loader)


def forward_pass(encoder, decoder, encoder_window_length, x_batch, force_only_batch, y_batch, mask, lengths, y_init, decode_using_encoder=True, decode_using_input=True):
    x_batch = x_batch.to(device)
    force_only_batch = force_only_batch.to(device)
    y_batch = y_batch.to(device)
    mask = mask.to(device)
    lengths = lengths.to(device)
    y_init = y_init.to(device)  # (B, 4)

    B, T, obs_dim = x_batch.size()

    y_pred_list = []
    h_encoder = encoder.init_hidden(B, device=device)  # Initialize hidden state
    h_decoder = decoder.init_hidden(B, device=device)  # Initialize decoder hidden state
    y_prev = y_init
    for t in range(T):
        if decode_using_encoder:
            if encoder_window_length > 1:
                # encoding with sliding window
                windowed_x = get_obs_padded_window(x_batch, t, encoder_window_length)  # (B, window_len, obs_dim)
                z_dyn, h_encoder = encoder(windowed_x, h_encoder)
            # z_dyn = torch.zeros(B, 16).to(device)  # Placeholder for z_dyn
            else:
                x_t = x_batch[:, t, :]  # (B, obs_dim)
                x_t = x_t.unsqueeze(1)  # (B, 1, obs_dim)
                z_dyn, h_encoder = encoder(x_t, h_encoder)
        else:
            z_dyn = None

        if decode_using_input:
            input_t = force_only_batch[:, t, :]  # (B, 2)
        else:
            input_t = None

        # current_force_only_batch = current_force_only_batch.unsqueeze(1).expand(-1, 1, -1)  # (B, window_len, 2)
        y, h_decoder = decoder(input_t, z_dyn, h_decoder, y_prev=None)  # force-only decoder with z_dyn
        y_pred_list.append(y)
        # y_prev = y_current  # update y_prev for next step

    y_pred = torch.cat(y_pred_list, dim=1)  # (B, T, 4)

    return y_pred, y_batch, mask

# --- Training loop ---
if __name__ == "__main__":
    train = False  # Set to False to skip training and only evaluate
    reload_data= False
    load_fixing_terminal_in_obs = True
    update_step = 5
    encoder_input_dim = 6
    decoder_input_dim = 2
    output_dim = 6
    encoder_window_length = 0  # 0 for stepwise, >1 for windowed
    z_dim = 16
    encoder_hidden_dim = 64
    decoder_state_dim = output_dim  # No state dimension if not using encoder

    decode_using_encoder = True # Set to False to skip encoder training
    if not decode_using_encoder:
        z_dim = 0  # No z_dim if not using encoder
    decode_using_input = False  # Set to True to use input as decoder
    if not decode_using_input:
        decoder_input_dim = 0
    decode_maintain_state = False
    if not decode_maintain_state:
        decoder_state_dim = 0

    dataset_base_dir = "/home/tp2/Documents/kejia/clip_fixing_dataset/off_policy_4/"
    return_function = effort_and_energy_based_return_function  # effort_and_energy_based_return_function or effort_based_return_function
    dataset = load_trajectories(dataset_base_dir, return_function, env_step=update_step, reload_data=reload_data, load_terminal_in_obs=load_fixing_terminal_in_obs) # observation (B, D)
    
    encoding_save_dir = "/home/tp2/Documents/kejia/d3rlpy/d3rlpy_logs/Dynamics_Encoder/predict_f_x_v"
    encoder_save_path = os.path.join(encoding_save_dir, f"best_encoder_2_encoding_{decode_using_encoder}_input_{decode_using_input}.pth")
    decoder_save_path = os.path.join(encoding_save_dir, f"best_decoder_2_encoding_{decode_using_encoder}_input_{decode_using_input}.pth")

    from torch.utils.data import random_split   
    # 80% train, 20% validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_dyn_embedding)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_dyn_embedding)

    loss_fn = masked_mse

    '''Training'''
    if train:
        encoder = DynamicsRNNEncoder(input_dim=encoder_input_dim, z_dim=z_dim, hidden_dim=encoder_hidden_dim).to(device)
        # dyn_decoder = DynamicsDecoder(z_dim=z_dim, output_dim=output_dim, max_len=2000).to(device)  # <-- increased max_len
        # auto_dyn_decoder = AutoregressiveDynamicsDecoder(input_dim=decoder_input_dim, z_dim=z_dim, output_dim=output_dim).to(device)
        # auto_decoder = AutoregressiveVanillaDecoder(input_dim=decoder_input_dim, output_dim=output_dim).to(device)
        # step_decoder = StepwiseVanillaDecoder(input_dim=decoder_input_dim, output_dim=output_dim).to(device)
        # force_only_decoder_fc = ForceOnlyVanillaDecoderFC(input_dim=decoder_input_dim, state_dim=output_dim, output_dim=output_dim).to(device)
        # force_only_decoder_rnn = ForceOnlyVanillaDecoderRNN(input_dim=decoder_input_dim, state_dim=output_dim, output_dim=output_dim).to(device)
        # force_only_dyn_decoder_fc = ForceOnlyDynamicsDecoderFc(input_dim=decoder_input_dim, z_dim=z_dim, state_dim=output_dim,  output_dim=output_dim).to(device)
        # force_only_dyn_decoder_rnn = ForceOnlyDynamicsDecoderRNN(input_dim=decoder_input_dim, z_dim=z_dim, state_dim=output_dim, output_dim=output_dim).to(device)
        flexible_decoder = FlexibleDynamicsDecoderRNN(obs_dim=decoder_input_dim, z_dim=z_dim, state_dim=decoder_state_dim, output_dim=output_dim).to(device)
        decoder = flexible_decoder

        encoder.train()
        decoder.train()

        optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

        best_loss = float('inf')
        patience_counter = 0
        for epoch in range(200):
            total_loss = 0
            for x_batch, force_only_batch, y_batch, mask, lengths, y_init in train_dataloader:

                y_pred, y_batch, mask = forward_pass(encoder, decoder, encoder_window_length, x_batch, force_only_batch, y_batch, mask, lengths, y_init, decode_using_encoder, decode_using_input)

                loss = loss_fn(y_pred, y_batch, mask)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
            
            val_loss = evaluate_dynamics_encoder(encoder, decoder, encoder_window_length, val_dataloader, loss_fn, plot=False, decode_using_encoder=decode_using_encoder, decode_using_input=decode_using_input)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(encoder.state_dict(), encoder_save_path)
                torch.save(decoder.state_dict(), decoder_save_path)
            else:
                patience_counter += 1

            if patience_counter >= 5:
                print(f"Early stopping at epoch {epoch+1}, no improvement for 10 epochs.")
                break

            print(f"Epoch {epoch+1:02d} | Train Loss: {total_loss / len(train_dataloader):.4f} | Val Loss: {val_loss:.4f}")

    '''Evaluation'''
    eval_encoder = DynamicsRNNEncoder(input_dim=encoder_input_dim, z_dim=z_dim, hidden_dim=encoder_hidden_dim).to(device)
    # eval_dyn_decoder = DynamicsDecoder(z_dim=z_dim, output_dim=output_dim, max_len=2000).to(device)  # <-- increased max_len
    # eval_auto_dyn_decoder = AutoregressiveDynamicsDecoder(input_dim=decoder_input_dim, z_dim=z_dim, output_dim=output_dim).to(device)
    # eval_auto_decoder = AutoregressiveVanillaDecoder(input_dim=decoder_input_dim, output_dim=output_dim).to(device)
    # eval_step_encoder = StepwiseVanillaDecoder(input_dim=decoder_input_dim, output_dim=output_dim).to(device)
    # eval_force_only_decoder_fc = ForceOnlyVanillaDecoderFC(input_dim=decoder_input_dim, state_dim=output_dim, output_dim=output_dim).to(device)
    # eval_force_only_decoder_rnn = ForceOnlyVanillaDecoderRNN(input_dim=decoder_input_dim, state_dim=output_dim, output_dim=output_dim).to(device)
    # eval_force_only_dyn_decoder_fc = ForceOnlyDynamicsDecoderFc(input_dim=decoder_input_dim, z_dim=z_dim, state_dim=output_dim, output_dim=output_dim).to(device)
    # eval_force_only_dyn_decoder_rnn = ForceOnlyDynamicsDecoderRNN(input_dim=decoder_input_dim, z_dim=z_dim, state_dim=output_dim, output_dim=output_dim).to(device)
    eval_flexible_decoder = FlexibleDynamicsDecoderRNN(obs_dim=decoder_input_dim, z_dim=z_dim, state_dim=decoder_state_dim,output_dim=output_dim).to(device)
    eval_decoder = eval_flexible_decoder  # Use the flexible decoder for evaluation

    # Reload best model
    eval_encoder.load_state_dict(torch.load(encoder_save_path))
    eval_decoder.load_state_dict(torch.load(decoder_save_path))

    # Plotting evaluation results
    evaluate_dynamics_encoder(eval_encoder, eval_decoder, encoder_window_length, val_dataloader, loss_fn, plot=True, decode_using_encoder=decode_using_encoder, decode_using_input=decode_using_input)
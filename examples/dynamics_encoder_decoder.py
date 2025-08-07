import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from build_episode_dataset import load_trajectories
from return_functions import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def export_encoder_to_onnx(encoder, export_path="dynamics_encoder.onnx"):
    encoder.eval()
    B, T, D = 1, 100, 6  # batch size 1, time steps 100, input dim 6
    x_dummy = torch.randn(B, T, D)
    lengths_dummy = torch.tensor([T])  # full length sequence
    x_dummy, lengths_dummy = x_dummy.to(device), lengths_dummy.to(device)
    
    torch.onnx.export(
        encoder,
        (x_dummy, ),
        export_path,
        export_params=True,
        opset_version=11,
        input_names=["input"],
        output_names=["z_dyn"],
        # Remove dynamic_axes for batch_size
        dynamic_axes={
            "input": {1: "time_steps"},
            "z_dyn": {0: "batch_size"},
        }
    )
    print(f"ONNX model exported to {export_path}")

def masked_mse(pred, target, mask):
    mse = (pred - target) ** 2
    mse = mse.sum(dim=-1)
    mse = mse * mask
    return mse.sum() / mask.sum().clamp(min=1)

# --- GRU Encoder for z_dyn ---
# (f_i, x_i, v_i)-> z_dyn
# class DynamicsRNNEncoder(nn.Module):
#     def __init__(self, input_dim=6, hidden_dim=64, z_dim=32):
#         super().__init__()
#         self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
#         self.fc = nn.Linear(hidden_dim, z_dim)
#         self.exporting = False  # Flag to control ONNX export

#     def set_export(self, export: bool):
#         self.exporting = export

#     def forward(self, x, lengths):  # x: (B, T, input_dim)
#         if self.exporting:
#             # ONNX can't handle pack_padded_sequence; fallback to regular GRU
#             output, _ = self.gru(x)
#             h_n = output[:, -1, :]  # take last hidden state manually
#         else:
#             packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
#             _, h_n = self.gru(packed)
#         return self.fc(h_n.squeeze(0))  # (B, z_dim)

class DynamicsRNNEncoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=64, z_dim=32, use_mask=False):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, z_dim)
        self.use_mask = use_mask  # optional if you want to add masking later
        self.exporting = False  # NEW FLAG

    def set_export(self, export: bool):
        self.exporting = export

    def forward(self, x, mask=None):  # x: (B, T, input_dim)
        """
        If mask is None: just use final time step (assumes padding is zeroed or irrelevant).
        If mask is provided: use the last valid time step per sample.
        """
        output, _ = self.gru(x)  # output: (B, T, hidden_dim)

        if self.exporting:
            h_n = output[:, -1, :]  # ONNX-safe: use last token
        elif self.use_mask and mask is not None:
            # Get the last valid step using mask (B, T)
            lengths = mask.sum(dim=1)  # (B,)
            idx = (lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1)
            idx = idx.expand(-1, 1, output.size(2))  # (B, 1, H)
            h_n = output.gather(dim=1, index=idx).squeeze(1)  # (B, H)
        else:
            h_n = output[:, -1, :]  # assume last token is valid (i.e., padding at the end is fine)

        return self.fc(h_n)  # (B, z_dim)


class FlexibleDynamicsDecoderRNN(nn.Module):
    def __init__(self, input_dim=2, z_dim=32, state_dim=4, hidden_dim=128, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(input_dim + state_dim + z_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def step(self, obs_t, y_prev, z_dyn, h_prev):
        """
        Single step forward pass.
        obs_t: (B, N) or None
        y_prev: (B, state_dim)
        z_dyn: (B, z_dim) or None
        h_prev: (1, B, hidden_dim)
        """
        B = y_prev.size(0)
        if obs_t is None:
            obs_t = torch.zeros(B, self.input_dim, device=y_prev.device)
        input_parts = [obs_t, y_prev]
        if z_dyn is not None:
            input_parts.append(z_dyn)
        input_t = torch.cat(input_parts, dim=-1).unsqueeze(1)  # (B, 1, D)
        out_t, h_next = self.gru(input_t, h_prev)  # out_t: (B, 1, H)
        y_next = self.fc_out(out_t.squeeze(1))     # (B, 4)
        return y_next, h_next

    def forward(self, init_y, obs_seq=None, z_dyn=None, rollout_steps=None):
        """
        obs_seq: (B, T, N) or None
        init_y: (B, state_dim)
        z_dyn: (B, z_dim) or None
        rollout_steps: int, used only when obs_seq is None
        """
        B = init_y.size(0)
        y_prev = init_y
        h = torch.zeros(1, B, self.hidden_dim, device=init_y.device)
        outputs = []

        if obs_seq is not None:
            T = obs_seq.size(1)
            for t in range(T):
                obs_t = obs_seq[:, t, :]  # (B, 2)
                y, h = self.step(obs_t, y_prev, z_dyn, h)
                outputs.append(y)
                y_prev = y
        else:
            assert rollout_steps is not None, "rollout_steps is required when obs_seq is None"
            for _ in range(rollout_steps):
                y, h = self.step(None, y_prev, z_dyn, h)
                outputs.append(y)
                y_prev = y

        return torch.stack(outputs, dim=1)  # (B, T, 4)
    
    def init_hidden(self, batch_size=1, device='cpu'):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)

    def get_z_dim(self):
        return self.z_dim

    def get_hidden_dim(self):
        return self.hidden_dim


# --- Decoder: reconstruct full sequence ---
''' Z_dyn only Decoder'''
# z_dyn -> (x_i+1, v_i+1)
class DynamicsDecoder(nn.Module):
    def __init__(self, z_dim=32, hidden_dim=64, output_dim=4, max_len=2000):
        super().__init__()
        self.max_len = max_len
        self.fc_init = nn.Linear(z_dim, hidden_dim)
        self.time_embed = nn.Parameter(torch.randn(1, max_len, hidden_dim))
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, z_dyn, T):
        B = z_dyn.size(0)
        h_0 = self.fc_init(z_dyn).unsqueeze(0)  # (1, B, H)
        inp = self.time_embed[:, :T, :].expand(B, -1, -1)  # (B, T, H)
        out, _ = self.gru(inp, h_0)
        return self.fc_out(out)  # (B, T, output_dim)

'''Full Observation Decoders'''
# (f_i, x_i, v_i, z_dyn) -> (x_i+1, v_i+1)
class AutoregressiveDynamicsDecoder(nn.Module):
    def __init__(self, input_dim=6, z_dim=32, hidden_dim=128, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim + z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # (x_{i+1}, v_{i+1})
        )

    def forward(self, input_seq, z_dyn):
        """
        Args:
            input_seq: (B, T, input_dim)  where input_dim = 6 (f, x, v)
            z_dyn: (B, z_dim)
        Returns:
            pred_seq: (B, T, output_dim)  where output_dim = 4 (x_next, v_next)
        """
        B, T, _ = input_seq.shape

        # Repeat z_dyn along T
        z_dyn_expanded = z_dyn.unsqueeze(1).expand(-1, T, -1)  # (B, T, z_dim)

        # Concatenate input + z_dyn
        decoder_input = torch.cat([input_seq, z_dyn_expanded], dim=-1)  # (B, T, input_dim + z_dim)

        # Flatten for MLP processing
        decoder_input = decoder_input.view(B * T, -1)
        pred = self.fc(decoder_input)  # (B*T, output_dim)
        pred = pred.view(B, T, -1)
        return pred

# (f_i, x_i, v_i) -> (x_i+1, v_i+1)
class AutoregressiveVanillaDecoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # (x_{i+1}, v_{i+1})
        )

    def forward(self, input_seq):
        """
        Args:
            input_seq: (B, T, input_dim)  where input_dim = 6 (f, x, v)
        Returns:
            pred_seq: (B, T, output_dim)  where output_dim = 4 (x_next, v_next)
        """
        B, T, _ = input_seq.shape

        # Concatenate input
        decoder_input = torch.cat([input_seq], dim=-1)  # (B, T, input_dim)

        # Flatten for MLP processing
        decoder_input = decoder_input.view(B * T, -1)
        pred = self.fc(decoder_input)  # (B*T, output_dim)
        pred = pred.view(B, T, -1)
        return pred

class StepwiseVanillaDecoder(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=128, output_dim=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input_seq):
        """
        Args:
            input_seq: (B, T, input_dim)
        Returns:
            pred_seq: (B, T, output_dim)
        """
        B, T, D = input_seq.shape
        preds = []

        for t in range(T):
            inp_t = input_seq[:, t, :]  # (B, input_dim)
            pred_t = self.fc(inp_t)     # (B, output_dim)
            preds.append(pred_t)

        return torch.stack(preds, dim=1)  # (B, T, output_dim)

''' Force-only Decoders '''
class ForceOnlyVanillaDecoderRNN(nn.Module):
    def __init__(self, input_dim=2, state_dim=4, hidden_dim=128, output_dim=4):
        # state_dim: dimension of y_0 = (x_0, v_0)
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        self.gru_cell = nn.GRUCell(input_dim + state_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, f_seq, init_y):
        """
        f_seq: (B, T, 2)
        init_y: (B, 4)  — initial (x, v)
        """
        B, T, _ = f_seq.shape
        h = torch.zeros(B, self.fc_out.in_features, device=f_seq.device)
        y_prev = init_y  # (B, 4)
        outputs = []

        for t in range(T):
            input_t = torch.cat([f_seq[:, t], y_prev], dim=-1)  # (B, 6)
            h = self.gru_cell(input_t, h)                        # (B, H)
            y = self.fc_out(h)                                   # (B, 4)
            outputs.append(y)
            y_prev = y  # autoregressive

        return torch.stack(outputs, dim=1)  # (B, T, 4)

class ForceOnlyVanillaDecoderFC(nn.Module):
    def __init__(self, input_dim=2, state_dim=4, hidden_dim=128, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.state_dim = state_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim + state_dim, hidden_dim),  # f_t + y_0
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, f_seq, init_y):
        """
        Args:
            f_seq: (B, T, 2)       — force input at each step
            init_y: (B, 4)         — initial (x, v) for stretch/push
        Returns:
            pred_seq: (B, T, 4)    — predicted (x, v) over time
        """
        B, T, _ = f_seq.shape
        init_y_expanded = init_y.unsqueeze(1).expand(B, T, self.state_dim)  # (B, T, 4)
        input_seq = torch.cat([f_seq, init_y_expanded], dim=-1)             # (B, T, 6)

        # Reshape for FC processing
        input_flat = input_seq.view(B * T, -1)
        output_flat = self.fc(input_flat)
        return output_flat.view(B, T, -1)
    
class ForceOnlyDynamicsDecoderRNN(nn.Module):
    def __init__(self, input_dim=2, z_dim=32, state_dim=4, hidden_dim=128, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.gru_cell = nn.GRUCell(input_dim + z_dim + state_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def step(self, force_t, y_prev, z_dyn, h_prev):
        # force_t: (B, 2)
        # y_prev: (B, 4)
        # z_dyn: (B, z_dim)
        # h_prev: (B, hidden_dim)
        input_t = torch.cat([force_t, y_prev, z_dyn], dim=-1)
        h = self.gru_cell(input_t, h_prev)  # properly propagate h
        y_next = self.fc_out(h)
        return y_next, h

    def forward(self, f_seq, init_y, z_dyn):
        """
        f_seq: (B, T, 2)
        init_y: (B, 4)  — initial (x, v)
        z_dyn: (B, z_dim)
        """
        B, T, _ = f_seq.shape
        h = torch.zeros(B, self.hidden_dim, device=f_seq.device)
        y_prev = init_y  # (B, 4)
        outputs = []

        for t in range(T):
            # input_t = torch.cat([f_seq[:, t], y_prev, z_dyn], dim=-1)  # (B, 6)
            # h = self.gru_cell(input_t, h)                        # (B, H)
            # y = self.fc_out(h)                                   # (B, 4)
            y, h = self.step(f_seq[:, t], y_prev, z_dyn, h)  # autoregressive step
            outputs.append(y)
            y_prev = y  # autoregressive

        return torch.stack(outputs, dim=1)  # (B, T, 4)
    
    def get_z_dim(self):
        return self.z_dim
    
    def get_hidden_dim(self):
        return self.hidden_dim
    
class ForceOnlyDynamicsDecoderFc(nn.Module):
    def __init__(self, input_dim=2, z_dim=32, state_dim=4, hidden_dim=128, output_dim=4):
        super().__init__()
        self.input_dim = input_dim
        self.z_dim = z_dim
        self.state_dim = state_dim

        self.fc = nn.Sequential(
            nn.Linear(input_dim + z_dim + state_dim, hidden_dim),  # f_t + z + y_0
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, f_seq, init_y, z_dyn):
        """
        Args:
            f_seq: (B, T, 2)       — force input at each step
            init_y: (B, 4)         — initial (x, v) for stretch/push
        Returns:
            pred_seq: (B, T, 4)    — predicted (x, v) over time
        """
        B, T, _ = f_seq.shape
        
        # Repeat z_dyn along T
        z_dyn_expanded = z_dyn.unsqueeze(1).expand(-1, T, self.z_dim)  # (B, T, z_dim)

        init_y_expanded = init_y.unsqueeze(1).expand(B, T, self.state_dim)  # (B, T, 4)
        input_seq = torch.cat([f_seq, init_y_expanded, z_dyn_expanded], dim=-1)             # (B, T, 6)

        # Reshape for FC processing
        input_flat = input_seq.view(B * T, -1)
        output_flat = self.fc(input_flat)
        return output_flat.view(B, T, -1)

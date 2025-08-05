import torch
import torch.nn as nn
import torch.distributions as D


def export_policy_to_onnx(policy, obs_dim, seq_len, z_dim, policy_type, export_path="mlp_policy.onnx"):
    policy.eval()
    if seq_len is not None and seq_len > 1:
        dummy_input = torch.randn(1, 1, seq_len, obs_dim)
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
    elif policy_type == "gru_with_encoder":
        dummy_input = torch.randn(1, 1, obs_dim)
        dummy_z_dyn = torch.randn(1, z_dim)               # (B, z_dim)

        dummy_input, dummy_z_dyn = dummy_input.to(next(policy.parameters()).device), dummy_z_dyn.to(next(policy.parameters()).device)

        torch.onnx.export(
        policy,
        (dummy_input, dummy_z_dyn),  # Tuple of inputs
        export_path,
        export_params=True,
        opset_version=11,
        input_names=["obs_seq", "z_dyn"],
        output_names=["output"],
        dynamic_axes={
            "obs_seq": {0: "batch_size", 1: "time_steps"},
            "z_dyn": {0: "batch_size"},
            "output": {0: "batch_size", 1: "time_steps"},
        }
    )
    else:
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
    
class MLPSeqPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, seq_len: int, hidden_dims=(128, 128)):
        super().__init__()
        input_dim = obs_dim * seq_len
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], act_dim)

    def forward(self, obs_seq):  
        # obs_seq: (B, T, seq_len, obs_dim)
        if obs_seq.dim() == 4:
            B, T, S, D = obs_seq.shape
            obs_seq = obs_seq.view(B * T, S * D)
            out = self.head(self.encoder(obs_seq))
            return out.view(B, T, -1)  # restore batch + seq
        elif obs_seq.dim() == 3:
            B, S, D = obs_seq.shape
            obs_seq = obs_seq.view(B, S * D)
            out = self.head(self.encoder(obs_seq))
            return out  # returns (B, act_dim)
        else:
            raise ValueError(f"Unsupported obs_seq shape: {obs_seq.shape}")
    
class StochasticMLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(128, 128)):
        super().__init__()
        layers = []
        dims = [obs_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

        # Output mean action
        self.mean_head = nn.Linear(dims[-1], act_dim)

        # Log std is a learnable parameter, shared across states
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # or init to small value

    def forward(self, x):
        latent = self.encoder(x)
        mean = self.mean_head(latent)
        return mean
    
    def get_dist(self, x):
        latent = self.encoder(x)
        mean = self.mean_head(latent)
        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)

class StochasticMLPSeqPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, seq_len: int, hidden_dims=(128, 128)):
        super().__init__()
        input_dim = obs_dim * seq_len
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

        self.mean_head = nn.Linear(dims[-1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs_seq):  
        # Supports input: (B, T, S, D) or (B, S, D)
        if obs_seq.dim() == 4:
            B, T, S, D = obs_seq.shape
            obs_seq = obs_seq.view(B * T, S * D)
            out = self.mean_head(self.encoder(obs_seq))
            out = torch.clamp(out, 0.0, 30.0)
            return out.view(B, T, -1)
        elif obs_seq.dim() == 3:
            B, S, D = obs_seq.shape
            obs_seq = obs_seq.view(B, S * D)
            out = self.mean_head(self.encoder(obs_seq))
            return torch.clamp(out, 0.0, 30.0)  # returns (B, act_dim)
        else:
            raise ValueError(f"Unsupported input shape: {obs_seq.shape}")

    def get_dist(self, obs_seq):
        # Same logic as forward()
        if obs_seq.dim() == 4:
            B, T, S, D = obs_seq.shape
            obs_seq = obs_seq.view(B * T, S * D)
            mean = self.mean_head(self.encoder(obs_seq)).view(B, T, -1)
            mean = torch.clamp(mean, 0.0, 30.0)
        elif obs_seq.dim() == 3:
            B, S, D = obs_seq.shape
            obs_seq = obs_seq.view(B, S * D)
            mean = self.mean_head(self.encoder(obs_seq))
            mean = torch.clamp(mean, 0.0, 30.0)
        else:
            raise ValueError(f"Unsupported input shape: {obs_seq.shape}")

        std = torch.exp(self.log_std)
        return torch.distributions.Normal(mean, std)

class StochasticRNNPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, z_dim=32, hidden_dim=128):
        super().__init__()
        self.input_dim = obs_dim - 1 + z_dim
        self.gru = nn.GRU(self.input_dim, hidden_dim, batch_first=True)
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs_seq, z_dyn):
        """
        obs_seq: (B, T, obs_dim), with terminal in the last dimension
        z_dyn:   (B, z_dim)
        """
        B, T, obs_dim = obs_seq.shape
        h = torch.zeros(1, B, self.gru.hidden_size, device=obs_seq.device)

        outputs = []
        for t in range(T):
            obs_t = obs_seq[:, t, :]  # (B, obs_dim)
            terminal_mask = obs_t[:, -1] > 0.5  # assuming last dim is terminal

            input_t = obs_t
            if z_dyn is not None:
                z_dyn_exp = z_dyn  # (B, z_dim)
                input_t = torch.cat([obs_t[:, :-1], z_dyn_exp], dim=-1)  # exclude terminal flag from input

            input_t = input_t.unsqueeze(1)  # (B, 1, input_dim)

            # Only update GRU if not terminated
            out_t, h = self.gru(input_t, h)
            h[:, terminal_mask, :] = h[:, terminal_mask, :].detach() * 0  # reset hidden state where terminal = 1

            mean_t = self.mean_head(out_t.squeeze(1))
            mean_t = torch.clamp(mean_t, 0.0, 30.0)
            outputs.append(mean_t)

        return torch.stack(outputs, dim=1)  # (B, T, act_dim)

    def get_dist(self, obs_seq, z_dyn=None):
        mean = self.forward(obs_seq, z_dyn)  # (B, T, act_dim)
        std = torch.exp(self.log_std)        # (act_dim,)
        std = std.expand_as(mean)            # (B, T, act_dim)
        return torch.distributions.Normal(mean, std)

# class ValueMLP(nn.Module):
#     def __init__(self, obs_dim, seq_len, hidden_dims=(128, 128)):
#         super().__init__()
#         input_dim = obs_dim * seq_len
#         layers = []
#         dims = [input_dim] + list(hidden_dims)
#         for in_dim, out_dim in zip(dims[:-1], dims[1:]):
#             layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
#         self.encoder = nn.Sequential(*layers)
#         self.value_head = nn.Linear(dims[-1], 1)

#     def forward(self, obs_seq):
#         B, S, D = obs_seq.shape
#         obs_seq = obs_seq.view(B, S * D)
#         return self.value_head(self.encoder(obs_seq)).squeeze(-1)  # (B,)

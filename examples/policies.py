import torch
import torch.nn as nn
import torch.distributions as D


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

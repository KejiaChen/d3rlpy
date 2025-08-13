from matplotlib.pylab import mean
import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import SigmoidTransform


def export_policy_to_onnx(policy, obs_dim, seq_len, policy_type="stochastic_mlp", export_path="mlp_policy.onnx", step_wise=False):
    policy.eval()
    if policy_type == "stochastic_mlp" or policy_type == "stochastic_two_head_mlp" or policy_type == "stochastic_two_head_transform_mlp" or policy_type == "stochastic_two_head_film_mlp":
        # if seq_len is not None and seq_len > 1:
        #     dummy_input = torch.randn(1, 1, seq_len, obs_dim)
        #     dummy_z_dyn = torch.randn(1, 1, policy.get_z_dim())  # (B, 1, z_dim)

        # #     dummy_input, dummy_z_dyn = dummy_input.to(next(policy.parameters()).device), dummy_z_dyn.to(next(policy.parameters()).device)

        #     torch.onnx.export(
        #         policy, 
        #         (dummy_input, dummy_z_dyn),
        #         export_path,
        #         export_params=True,
        #         opset_version=11,
        #         input_names=["input", "z_dyn"],
        #         output_names=["output"],
        #         dynamic_axes={
        #             "input": {0: "batch_size"},  # support variable batch size
        #             "output": {0: "batch_size"}
        #         }
        #     )
        # else:
        dummy_input = torch.randn(1, seq_len, obs_dim-1)  # 3D input: (batch_size, seq_len, obs_dim) to match the expected input shape in cpp
        dummy_terminal = torch.zeros(1, 1, 1)  # (B, 1, 1) terminal flag
        dummy_z_dyn = torch.randn(1, 1, policy.get_z_dim())  # (B, 1, z_dim)

        dummy_input, dummy_terminal, dummy_z_dyn = dummy_input.to(next(policy.parameters()).device), dummy_terminal.to(next(policy.parameters()).device), dummy_z_dyn.to(next(policy.parameters()).device)

        torch.onnx.export(
            policy, 
            (dummy_input, dummy_terminal, dummy_z_dyn),  # Tuple of inputs
            export_path,
            export_params=True,
            opset_version=11,
            input_names=["input", "terminal", "z_dyn"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},  # support variable batch size
                "terminal": {0: "batch_size"},  # support variable batch size
                "z_dyn": {0: "batch_size"},  # support variable batch size
                "output": {0: "batch_size"}
            }
        )
    elif policy_type == "gru":
        if step_wise:
            dummy_input = torch.randn(1, obs_dim)
            dummy_z_dyn = torch.randn(1, policy.get_z_dim())    # (B, z_dim)
            dummy_h = torch.zeros(1, 1, policy.get_hidden_dim())

            dummy_input, dummy_z_dyn, dummy_h = dummy_input.to(next(policy.parameters()).device), dummy_z_dyn.to(next(policy.parameters()).device), dummy_h.to(next(policy.parameters()).device)

            torch.onnx.export(
                policy,
                (dummy_input, dummy_z_dyn, dummy_h),  # Tuple of inputs
                export_path,
                export_params=True,
                opset_version=11,
                input_names=["obs_seq", "z_dyn", "h_in"],
                output_names=["output", "h_out"],
                dynamic_axes={
                    "obs_seq": {0: "batch_size"},
                    "z_dyn": {0: "batch_size"},
                    "h_in": {0: "batch_size"},
                    "output": {0: "batch_size"},
                    "h_out": {0: "batch_size"},
                },
            )
        else:
            dummy_input = torch.randn(1, 1, obs_dim)
            dummy_z_dyn = torch.randn(1, 1, policy.get_z_dim())               # (B, 1, z_dim)
            dummy_h   = torch.zeros(1, 1, policy.get_hidden_dim())

            dummy_input, dummy_z_dyn, dummy_h = dummy_input.to(next(policy.parameters()).device), dummy_z_dyn.to(next(policy.parameters()).device), dummy_h.to(next(policy.parameters()).device)

            torch.onnx.export(
                policy,
                (dummy_input, dummy_z_dyn, dummy_h),  # Tuple of inputs
                export_path,
                export_params=True,
                opset_version=11,
                input_names=["obs_seq", "z_dyn", "h_in"],
                output_names=["output", "h_out"],
                dynamic_axes={
                    "obs_seq": {0: "batch_size"},
                    "z_dyn": {0: "batch_size"},
                    "h_in": {1: "batch_size"},
                    "output": {0: "batch_size"},
                    "h_out": {1: "batch_size"},
                }
            )
    print(f"ONNX model exported to {export_path}")

# class CensoredSquashedNormal:
#     """
#     Distribution-like wrapper that:
#       - pre-terminal: y = sigmoid(x),  x ~ Normal(mean_pre, std)
#       - post-terminal: left-censored at y <= eps_censor, because sigmoid(x) is bounded in (0, 1) and excludes 0 and 1
#     Provides log_prob(y) and entropy() shaped like (B, act_dim).
#     """
#     def __init__(self, mean_pre, std, terminal, eps_censor=1e-4, eps_obs=1e-6):
#         """
#         mean_pre : (B, A)   pre-sigmoid mean
#         std      : (B, A)   std for latent Normal
#         terminal : (B, 1) or (B, 1, 1) with {0,1}
#         """
#         self.mean_pre = mean_pre
#         self.std = std
#         self.base = Normal(loc=mean_pre, scale=std)
#         self.squashed = TransformedDistribution(self.base, [SigmoidTransform()])
#         # Masks broadcast to action dims
#         if terminal.dim() == 3:
#             terminal = terminal.squeeze(-1)  # (B,1)
#         self.pre_mask  = (terminal == 0).float()            # (B,1)
#         self.post_mask = 1.0 - self.pre_mask                # (B,1)
#         self.eps_censor = eps_censor
#         self.eps_obs = eps_obs

#     def _expand_mask(self, y):
#         # Expand (B,1) -> (B, A)
#         return self.pre_mask.expand_as(y), self.post_mask.expand_as(y)

#     def log_prob(self, y):
#         """
#         Returns per-dimension log-prob, shape (B, A).
#         Pre-terminal: usual logistic-normal log_prob.
#         Post-terminal: log P(Y <= eps_censor) via latent Normal CDF (left-censored), to compensate for 0 that is excluded in sigmoid.
#         """
#         # Clip observations very slightly to avoid logit/0 issues
#         y_clip = y.clamp(self.eps_obs, 1.0 - self.eps_obs)

#         pre_m, post_m = self._expand_mask(y_clip)

#         # A) pre-terminal: standard squashed Gaussian likelihood
#         lp_pre = self.squashed.log_prob(y_clip)

#         # B) post-terminal: left-censored at eps_censor  (doesn't depend on y)
#         x_thr = torch.logit(torch.full_like(y_clip, self.eps_censor))
#         z = (x_thr - self.mean_pre) / self.std
#         std_norm = Normal(torch.zeros_like(z), torch.ones_like(z))
#         # clamp CDF to avoid log(0)
#         lp_post = torch.log(torch.clamp(std_norm.cdf(z), min=1e-12))

#         return pre_m * lp_pre + post_m * lp_post

#     def entropy(self):
#         """
#         Use latent Normal entropy for pre-terminal; zero post-terminal.
#         Returns (B, A).
#         """
#         ent_latent = self.base.entropy()  # (B, A)
#         pre_m, post_m = self._expand_mask(ent_latent)
#         return pre_m * ent_latent + post_m * 0.0

#     # Optional convenience for sampling at runtime
#     def rsample(self):
#         x = self.base.rsample()
#         y = torch.sigmoid(x)
#         # Hard gate to zero after terminal for execution-time safety
#         return y * self.pre_mask  # pre_mask==1 where terminal==0

#     def sample(self):
#         with torch.no_grad():
#             return self.rsample()


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
    def __init__(self, obs_dim=6, act_dim=2, seq_len=1, z_dim=32, hidden_dims=(128, 128)):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.terminal_dim = 1  # terminal flag dimension
        self.seq_len = seq_len
        input_dim = obs_dim*self.seq_len + self.terminal_dim + z_dim  # Combined input
        
        layers = []
        dims = [input_dim] + list(hidden_dims)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

        self.mean_head = nn.Linear(dims[-1], act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.clamp_min = 0
        self.clamp_max = 1
        
    def _pack(self, obs, if_terminal, z_dyn):
        """
        obs: (B, T, obs_dim)
        terminal: (B, 1*1) 
        z_dyn: (B, 1, z_dim) or None
        returns flattened input (B, seq_len*obs_dim [+ z_dim])
        """
        B, T, D = obs.shape
        assert D == self.obs_dim, f"Expected obs_dim={self.obs_dim}, got {D}"

        if T != self.seq_len:
            raise ValueError(f"Input sequence length {T} does not match expected seq_len {self.seq_len}.")

        x = obs.reshape(obs.size(0), -1)          # (B, T*D)
        x = torch.cat([x, if_terminal], dim=-1)  # (B, T*D + 1)

        if z_dyn is None: # z_dyn is only None when z_dim = 0
            x = x
        else:
            z_dyn = z_dyn.view(B, -1)  # (B, z_dim)
            x = torch.cat([x, z_dyn], dim=-1)
        return x

    def forward(self, obs, terminal, z_dyn=None, return_pre_squash=False):
        # obs: (B, T, obs_dim)
        # terminal: (B, 1, 1)
        # z_dyn: (B, 1, z_dim)
        B = obs.size(0)
        if_terminal = (terminal.view(B, -1) > 0.5).float()            # (B, 1)

        x = self._pack(obs, if_terminal, z_dyn)
        latent = self.encoder(x)
        pre_mean = self.mean_head(latent)

        if return_pre_squash:
            return pre_mean

        # force monotonicity
        # mean = F.softplus(mean)
        # mean = torch.clamp(mean, min=self.clamp_min, max=self.clamp_max)  # normalized action is in [0, 1]
        mean = torch.sigmoid(pre_mean)  # normalized action is in [0, 1]

        # Hard gate: set to zero after terminal (per your demos)
        mean = (1.0 - if_terminal) * mean

        return mean

    # def get_dist(self, obs, terminal, z_dyn=None, eps_censor=1e-4, eps_obs=1e-6):
    #     # obs: (B, T, obs_dim)
    #     # terminal: (B, 1, 1)
    #     # z_dyn: (B, 1, z_dim)
    #     mean_pre = self.forward(obs, terminal, z_dyn, return_pre_squash=False)

    #     log_std = torch.clamp(self.log_std, min=-0.92, max=2.0)  # σ in [~0.007, ~7.4]
    #     std = torch.exp(log_std).expand_as(mean_pre)

    #     # base = torch.distributions.Normal(loc=mean_pre, scale=std)
    #     # dist = torch.distributions.TransformedDistribution(base, [torch.distributions.transforms.SigmoidTransform()])  # support (0,1)
    #     return CensoredSquashedNormal(mean_pre, std, terminal, eps_censor=eps_censor, eps_obs=eps_obs)

    def get_dist(self, obs, terminal, z_dyn=None):
        mean = self.forward(obs, terminal, z_dyn, return_pre_squash=False) # TODO@KejiaChen try later with true

        log_std = torch.clamp(self.log_std, min=-0.92, max=2.0)  # σ in [~0.007, ~7.4]
        std = torch.exp(log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)
    
    def get_z_dim(self):
        return self.z_dim

class StochasticTwoHeadMLPPolicy(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, seq_len=1, z_dim=32, obs_hidden_dims=(128, 128), z_hidden_dims=(32, 32)):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.terminal_dim = 1  # terminal flag dimension
        self.seq_len = seq_len
        self.alpha_lo, self.alpha_hi = [0.7, 1.4]
        self.beta_max = 0.1

        # ---- Obs encoder (separate head) ----
        obs_input_dim = obs_dim * seq_len        
        obs_layers = []
        obs_head_dims = [obs_input_dim] + list(obs_hidden_dims)
        for in_dim, out_dim in zip(obs_head_dims[:-1], obs_head_dims[1:]):
            obs_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.obs_head = nn.Sequential(*obs_layers)

        # ---- z_dyn head (separate) ----
        z_head_dims = [z_dim] + list(z_hidden_dims)
        z_mlp = []
        for in_dim, out_dim in zip(z_head_dims[:-1], z_head_dims[1:]):
            z_mlp += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.z_head = nn.Sequential(*z_mlp)

        # ---- fuse obs and terminal and z_dyn ----
        self.fuse_hidden_dim = 128
        self.base_fuse = nn.Sequential(
            nn.Linear(obs_hidden_dims[1] + 1 + z_hidden_dims[1], self.fuse_hidden_dim), nn.ReLU(),
            nn.Linear(self.fuse_hidden_dim, act_dim)
        )

        # log std
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, obs, terminal, z_dyn=None, return_pre_squash=False):
        # obs: (B, T, obs_dim)
        # terminal: (B, 1, 1)
        # z_dyn: (B, 1, z_dim)
        B = obs.size(0)
        terminal = terminal.view(B, -1)  # (B, 1)
        z_dyn = z_dyn.view(B, -1) if z_dyn is not None else None  # (B, z_dim)
        if_terminal = (terminal > 0.5).float()            # (B, 1)

        x = obs.reshape(obs.size(0), -1)  # (B, T*obs_dim)
        h_obs = self.obs_head(x)              # (B, hidden[1])
        h_z = self.z_head(z_dyn)              # (B, 64)

        h_fuse = torch.cat([h_obs, terminal.float(), h_z], dim=-1)  # (B, 128+1+64)
        y_pre = self.base_fuse(h_fuse)

        if return_pre_squash:
            return y_pre

        # y = torch.sigmoid(y_pre)  # sigmoid seems to be a bad choice here because of easy saturation
        y = torch.clamp(y_pre, min=self.clamp_min, max=self.clamp_max)  # normalized action is in [0, 1]
        mean = y * (1.0 - terminal.float())  # broadcast over act_dim

        return mean

    def get_dist(self, obs, terminal, z_dyn=None):
        mean = self.forward(obs, terminal, z_dyn, return_pre_squash=True)

        log_std = torch.clamp(self.log_std, min=-0.92, max=2.0)  # σ in [~0.007, ~7.4]
        std = torch.exp(log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)
    
    def get_z_dim(self):
        return self.z_dim

class StochasticTwoHeadTransformMLPPolicy(nn.Module):
    def __init__(self, obs_dim=6, act_dim=2, seq_len=1, z_dim=32, obs_hidden_dims=(128, 128), z_hidden_dims=(32, 32)):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.terminal_dim = 1  # terminal flag dimension
        self.seq_len = seq_len
        self.alpha_lo, self.alpha_hi = [0.7, 1.4]
        self.beta_max = 0.1

        # ---- Obs encoder (separate head) ----
        obs_input_dim = obs_dim * seq_len        
        obs_layers = []
        obs_head_dims = [obs_input_dim] + list(obs_hidden_dims)
        for in_dim, out_dim in zip(obs_head_dims[:-1], obs_head_dims[1:]):
            obs_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.obs_head = nn.Sequential(*obs_layers)

        # ---- z_dyn head (separate) ----
        z_head_dims = [z_dim] + list(z_hidden_dims)
        z_mlp = []
        for in_dim, out_dim in zip(z_head_dims[:-1], z_head_dims[1:]):
            z_mlp += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.z_head = nn.Sequential(*z_mlp)

        # ---- fuse obs and terminal ----
        self.fuse_hidden_dim = 96
        self.base_fuse = nn.Sequential(
            nn.Linear(obs_hidden_dims[1] + 1, self.fuse_hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.fuse_hidden_dim, act_dim)
        )

        # ---Time-varying adaptation from current base features
        self.adapt_hidden_dim = 32
        self.adapt = nn.Sequential(
            nn.Linear(self.fuse_hidden_dim + z_hidden_dims[1], self.adapt_hidden_dim), 
            nn.ReLU(),
            nn.Linear(self.adapt_hidden_dim, 2 * act_dim)  # split into [alpha_raw, beta_raw]
        )

        # log std
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        self.clamp_min = 0
        self.clamp_max = 1

    def forward(self, obs, terminal, z_dyn=None, return_pre_squash=False):
        # obs: (B, T, obs_dim)
        # terminal: (B, 1, 1)
        # z_dyn: (B, 1, z_dim)
        B = obs.size(0)
        terminal = terminal.view(B, -1)  # (B, 1)
        z_dyn = z_dyn.view(B, -1) if z_dyn is not None else None  # (B, z_dim)
        if_terminal = (terminal > 0.5).float()            # (B, 1)

        x = obs.reshape(obs.size(0), -1)  # (B, T*obs_dim)
        h_obs = self.obs_head(x)              # (B, hidden[1])
        h_z = self.z_head(z_dyn)              # (B, 64)

        # Fuse terminal directly (scalar) into base path
        h_base_in = torch.cat([h_obs, terminal.float()], dim=-1)  # (B, hidden[1]+1)
        h_base = self.base_fuse[0](h_base_in)                    # Linear
        h_base = F.relu(h_base)                                   # ReLU
        base_logits = self.base_fuse[2](h_base)                  # Linear -> (B, act_dim)
        # base = F.softplus(base_logits)                            # ≥ 0

        # Time-varying adaptation α_t, β_t from current features + z
        h_adapt_in = torch.cat([h_base, h_z], dim=-1)             # (B, 128+64)
        ab = self.adapt(h_adapt_in)                               # (B, 2*act_dim)
        a_raw, b_raw = torch.chunk(ab, 2, dim=-1)

        # Bound α and β for stability
        alpha = self.alpha_lo + (self.alpha_hi - self.alpha_lo) * torch.sigmoid(a_raw)  # (B, act_dim)
        beta  = self.beta_max * torch.tanh(b_raw)                                       # (B, act_dim)

        # print(f"alpha: {alpha}, beta: {beta}")

        # Combine, clamp to [0,1], then hard gate by terminal
        y_pre = alpha * base_logits + beta
        if return_pre_squash:
            return y_pre
        # y = torch.sigmoid(y_pre)  
        y = torch.clamp(y_pre, min=self.clamp_min, max=self.clamp_max)  # normalized action is in [0, 1]
        mean = y * (1.0 - terminal.float())  # broadcast over act_dim

        return mean

    def get_dist(self, obs, terminal, z_dyn=None):
        mean = self.forward(obs, terminal, z_dyn, return_pre_squash=True)

        log_std = torch.clamp(self.log_std, min=-0.92, max=2.0)  # σ in [~0.007, ~7.4]
        std = torch.exp(log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std)
    
    def get_z_dim(self):
        return self.z_dim
    
class StochasticTwoHeadFiLMMLPPolicy(nn.Module):
    def __init__(self,
                 obs_dim=6,
                 act_dim=2,
                 seq_len=1,
                 z_dim=32,
                 obs_hidden_dims=(128, 128),
                 z_hidden_dims=(32, 32),
                 fuse_hidden_dim=128,
                 gamma_scale=0.5,   # bounds FiLM scale; 0.5 → (1 ± 0.5)
                 beta_scale=0.2     # bounds FiLM shift
                 ):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.seq_len = seq_len

        self.fuse_hidden_dim = fuse_hidden_dim
        self.gamma_scale = gamma_scale
        self.beta_scale = beta_scale

        # ---- Obs encoder ----
        obs_input_dim = obs_dim * seq_len
        obs_layers = []
        obs_head_dims = [obs_input_dim] + list(obs_hidden_dims)
        for in_dim, out_dim in zip(obs_head_dims[:-1], obs_head_dims[1:]):
            obs_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.obs_head = nn.Sequential(*obs_layers)

        # ---- z_dyn head ----
        z_head_dims = [z_dim] + list(z_hidden_dims)
        z_layers = []
        for in_dim, out_dim in zip(z_head_dims[:-1], z_head_dims[1:]):
            z_layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
        self.z_head = nn.Sequential(*z_layers)
        self.z_out_dim = z_hidden_dims[-1]

        # ---- Base trunk: obs+terminal → features (no z here) ----
        self.base_trunk = nn.Sequential(
            nn.Linear(obs_hidden_dims[-1] + 1, fuse_hidden_dim),
            nn.ReLU(),
        )

        # ---- FiLM generator from z: produces per-feature (gamma, beta) ----
        self.film = nn.Linear(self.z_out_dim, 2 * fuse_hidden_dim)

        # ---- Final action head: modulated features → y_pre (unbounded) ----
        self.action_head = nn.Linear(fuse_hidden_dim, act_dim)

        # ---- Stochasticity (latent Normal over y_pre) ----
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # Inference clamp (if you choose to clamp at runtime)
        self.clamp_min = 0.0
        self.clamp_max = 1.0

        # (Optional) mild init to avoid early saturation
        nn.init.uniform_(self.action_head.weight, -0.02, 0.02)
        nn.init.constant_(self.action_head.bias, 0.0)

    def forward(self, obs, terminal, z_dyn=None, return_pre_squash=False):
        """
        obs:      (B, T, obs_dim)
        terminal: (B, 1, 1) or (B, 1)
        z_dyn:    (B, 1, z_dim) or (B, z_dim)
        """
        B = obs.size(0)
        terminal = terminal.view(B, -1).float()  # (B,1)

        # Encoders
        x = obs.reshape(B, -1)                   # (B, T*obs_dim)
        h_obs = self.obs_head(x)                 # (B, obs_hidden_dims[-1])

        if z_dyn is not None:
            z = z_dyn.view(B, -1)
            h_z = self.z_head(z)                 # (B, z_out_dim)
        else:
            h_z = torch.zeros(B, self.z_out_dim, device=obs.device, dtype=obs.dtype)

        # Base features from obs + terminal
        h_base_in = torch.cat([h_obs, terminal], dim=-1)          # (B, obs_hidden+1)
        h_base = self.base_trunk(h_base_in)                        # (B, H)

        # FiLM from z: per-feature scale & shift
        gb = self.film(h_z)                                        # (B, 2H)
        gamma, beta = torch.chunk(gb, 2, dim=-1)                   # (B, H) each
        # Bound the modulation for stability
        gamma = self.gamma_scale * torch.tanh(gamma)               # in (-γs, γs)
        beta  = self.beta_scale  * torch.tanh(beta)                # in (-βs, βs)

        # Modulate features, then linear to y_pre (unbounded)
        h_tilde = (1.0 + gamma) * h_base + beta                    # (B, H)
        y_pre = self.action_head(h_tilde)                          # (B, act_dim)

        if return_pre_squash:
            return y_pre

        # Inference: map to [0,1] only here (training should use y_pre)
        y = torch.clamp(y_pre, min=self.clamp_min, max=self.clamp_max)
        mean = y * (1.0 - terminal)                                # hard gate after terminal
        return mean

    def get_dist(self, obs, terminal, z_dyn=None):
        """
        Returns an unbounded Normal over the pre-squash output y_pre.
        Train your NLL in this latent space; clamp/sigmoid only for runtime actions.
        """
        mean_pre = self.forward(obs, terminal, z_dyn, return_pre_squash=True)
        log_std = torch.clamp(self.log_std, min=-0.92, max=2.0)    # keep your empirically good floor
        std = torch.exp(log_std).expand_as(mean_pre)
        return torch.distributions.Normal(mean_pre, std)

    def get_z_dim(self):
        return self.z_dim


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
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.input_dim = self.obs_dim - 1 + self.z_dim
        self.gru = nn.GRU(self.input_dim, hidden_dim, batch_first=True)
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))
        self.hidden_dim = hidden_dim

    def forward(self, obs_seq, z_dyn, h=None):
        """
        obs_seq: (B, T, obs_dim) where obs_dim includes terminal flag
        z_dyn:   (B, z_dim)
        """
        B, T, obs_dim = obs_seq.shape

        terminal = obs_seq[..., -1] > 0.5  # (B, T) boolean mask
        obs_input = obs_seq[..., :-1]     # remove terminal flag: (B, T, obs_dim - 1)

        # Expand z_dyn to match time dimension
        if z_dyn is not None:
            rnn_input = torch.cat([obs_input, z_dyn], dim=-1)  # (B, T, input_dim)
        else:
            rnn_input = obs_input  # (B, T, input_dim)

        if T == 1:
            # Step-by-step inference: provide and return hidden state
            rnn_output, new_h = self.gru(rnn_input, h)  # (B, 1, H), h: (1, B, H)
        else:
            # Training or full sequence inference
            rnn_output, new_h = self.gru(rnn_input)  # (B, T, H)

        mean = self.mean_head(rnn_output)    # (B, T, act_dim)
        # Mask outputs after terminal to 0
        mean = mean * (~terminal).unsqueeze(-1).float()  # (B, T, act_dim)
        mean = torch.clamp(mean, 0.0, 30.0)

        if T == 1:
            return mean.squeeze(1), new_h  # return (B, act_dim), hidden
        else:
            return mean  # (B, T, act_dim)

    def get_dist(self, obs_seq, z_dyn=None):
        mean = self.forward(obs_seq, z_dyn)  # (B, T, act_dim)
        std = torch.exp(self.log_std)        # (act_dim,)
        std = std.expand_as(mean)            # (B, T, act_dim)
        return torch.distributions.Normal(mean, std)
    
    def init_hidden(self, batch_size=1, device='cpu'):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)
    
    def get_z_dim(self):
        return self.z_dim
    
    def get_hidden_dim(self):
        return self.hidden_dim

class StochasticRNNPolicyStepwise(nn.Module):
    def __init__(self, obs_dim, act_dim, z_dim=32, hidden_dim=128):
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.input_dim = self.obs_dim - 1 + self.z_dim
        self.hidden_dim = hidden_dim

        self.gru = nn.GRU(self.input_dim, hidden_dim, batch_first=True)
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs_t, z_dyn_t, h_t):
        """
        obs_t: (B, 1, N) or None
        y_prev: (B, 1, state_dim)
        z_dyn: (B, 1, z_dim) or None
        h_prev: (1, B, hidden_dim)
        """
        terminal = obs_t[:, -1] > 0.5  # (B, T) boolean mask
        obs_input = obs_t[:, :-1]     # remove terminal flag: (B, T, obs_dim - 1)

        if z_dyn_t is not None:
            # Concatenate obs_input and z_dyn_t
            x = torch.cat([obs_input, z_dyn_t], dim=-1)
        else:
            x = obs_input
        x = x.unsqueeze(1) # (B, 1, input_dim)

        # GRU step
        output, h_t_next = self.gru(x, h_t)          # output: (B, 1, H), h_t_next: (1, B, H)
        last_hidden = output[:, 0, :]                # (B, hidden_dim)

        # Predict mean
        mean = self.mean_head(last_hidden)           # (B, act_dim)

        return mean, h_t_next

    def get_dist_step(self, obs_t, z_dyn_t, h_t):
        mean, h_t_next = self.forward(obs_t, z_dyn_t, h_t)
        std = torch.exp(self.log_std).expand_as(mean)
        return torch.distributions.Normal(mean, std), h_t_next
    
    def init_hidden(self, batch_size=1, device='cpu'):
        return torch.zeros(1, batch_size, self.hidden_dim, device=device)
    
    def get_z_dim(self):
        return self.z_dim
    
    def get_hidden_dim(self):
        return self.hidden_dim
    

# class StochasticMLPPolicy(nn.Module):
#     def __init__(self, obs_dim, act_dim, seq_len=1, z_dim=32, hidden_dims=(128, 128)):
#         super().__init__()
#         self.obs_dim = obs_dim
#         self.z_dim = z_dim
#         self.seq_len = seq_len
#         if self.seq_len == 0:
#             self.seq_len = 1
#         # input_dim = obs_dim + z_dim  # Combined input
#         input_dim = self.seq_len * obs_dim + z_dim

#         layers = []
#         dims = [input_dim] + list(hidden_dims)
#         for in_dim, out_dim in zip(dims[:-1], dims[1:]):
#             layers += [nn.Linear(in_dim, out_dim), nn.ReLU()]
#         self.backbone = nn.Sequential(*layers)

#         self.mean_head = nn.Linear(dims[-1], act_dim)
#         self.log_std = nn.Parameter(torch.zeros(act_dim))
    
#     def _ensure_window(self, obs):
#         """
#         obs: (B, T, obs_dim)
#         returns: (B, self.seq_len, obs_dim)
#         Pads by repeating the earliest frame if too short; takes the last seq_len if too long.
#         """
#         B, T, D = obs.shape
#         assert D == self.obs_dim, f"Expected obs_dim={self.obs_dim}, got {D}"

#         if T != self.seq_len:
#             raise ValueError(f"Input sequence length {T} does not match expected seq_len {self.seq_len}.")

#         # if T == self.seq_len:
#         #     return obs
#         # elif T > self.seq_len:
#         #     return obs[:, -self.seq_len:, :]  # keep most recent seq_len
#         # else:
#         #     # pad at the front with the earliest available frame
#         #     pad_needed = self.seq_len - T
#         #     pad = obs[:, :1, :].expand(B, pad_needed, D)
#         #     return torch.cat([pad, obs], dim=1)
#         return obs
        
#     def _pack(self, obs, z_dyn):
#         """
#         obs: (B, T, obs_dim)
#         z_dyn: (B, z_dim) or None
#         returns flattened input (B, seq_len*obs_dim [+ z_dim])
#         """
#         obs_win = self._ensure_window(obs)                # (B, S, D)
#         x = obs_win.reshape(obs_win.size(0), -1)          # (B, S*D)
#         # if self.z_dim > 0 and z_dyn is not None:
#         #     x = torch.cat([x, z_dyn], dim=-1)             # (B, S*D + z_dim)

#         if z_dyn is None:
#             z_dyn = obs.new_zeros(obs.size(0), self.export_z_dim)

#         x = torch.cat([x, z_dyn], dim=-1)
#         return x
    
#     def forward(self, obs, z_dyn=None):
#         '''
#         obs: (B, T, obs_dim)
#         z_dyn: (B, z_dim)
#         mean: (B, act_dim)
#         '''
#         x = self._pack(obs, z_dyn)  # (B, seq_len * obs_dim [+ z_dim])
#         latent = self.backbone(x)
#         mean = self.mean_head(latent)
#         return mean
     
#     def get_dist(self, obs, z_dyn=None):
#         mean = self.forward(obs, z_dyn)  # (B, act_dim)
#         # std = torch.exp(self.log_std)
#         log_std = torch.clamp(self.log_std, min=-0.92, max=2.0)  # σ in [~0.007, ~7.4]
#         std = torch.exp(log_std).expand_as(mean)
#         return torch.distributions.Normal(mean, std)
    
#     def get_z_dim(self):
#         return self.z_dim
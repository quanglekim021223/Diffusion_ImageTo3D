import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os
import requests
from tqdm import tqdm
import hashlib
import math
from einops import rearrange, repeat


class SinusoidalPositionEmbeddings(nn.Module):
    """Sinusoidal position embeddings for diffusion timesteps"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(
            half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class CrossAttention(nn.Module):
    """Cross attention module for conditioning on view angles"""

    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = context_dim if context_dim is not None else query_dim

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context):
        h = self.heads

        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class SelfAttention(nn.Module):
    """Self attention module for processing features"""

    def __init__(self, in_channels, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == in_channels)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(in_channels, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, in_channels),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        attn = sim.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        out = rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
        return out


class Residual(nn.Module):
    """Residual block with group normalization"""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ResnetBlock(nn.Module):
    """Resnet block with group normalization"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out)
        ) if time_emb_dim is not None else None

        self.block1 = nn.Sequential(
            nn.GroupNorm(groups, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(groups, dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1)
        )
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)
        if time_emb is not None and self.mlp is not None:
            time_emb = self.mlp(time_emb)
            h = rearrange(time_emb, 'b c -> b c 1 1') + h
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    """U-Net architecture for diffusion model"""

    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        time_emb_dim=None,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = init_dim if init_dim is not None else dim
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = time_emb_dim if time_emb_dim is not None else dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim,
                            groups=resnet_block_groups),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim,
                            groups=resnet_block_groups),
                Residual(SelfAttention(dim_out)),
                nn.Conv2d(dim_out, dim_out, 4, 2,
                          1) if not is_last else nn.Identity()
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.mid_attn = SelfAttention(mid_dim)
        self.mid_block2 = ResnetBlock(
            mid_dim, mid_dim, time_emb_dim=time_dim, groups=resnet_block_groups)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= len(in_out) - 1

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in,
                            time_emb_dim=time_dim, groups=resnet_block_groups),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim,
                            groups=resnet_block_groups),
                Residual(SelfAttention(dim_in)),
                nn.ConvTranspose2d(dim_in, dim_in, 4, 2,
                                   1) if not is_last else nn.Identity()
            ]))

        self.final_res_block = ResnetBlock(
            dim * 2, dim, time_emb_dim=time_dim, groups=resnet_block_groups)
        self.final_conv = nn.Conv2d(
            dim, out_dim if out_dim is not None else channels, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = x_self_cond if x_self_cond is not None else torch.zeros_like(
                x)
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class NoiseScheduler:
    """Noise scheduler for diffusion model"""

    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1. / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * \
            (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * \
            torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def add_noise(self, x, t):
        """Add noise to the input according to the noise schedule"""
        noise = torch.randn_like(x)
        return self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise, noise

    def step(self, model_output, t, x):
        """Perform one step of the reverse diffusion process"""
        alpha = self.alphas[t]
        alpha_prev = self.alphas_cumprod_prev[t]
        beta = self.betas[t]

        # Compute the mean of the posterior
        mean = (1 / torch.sqrt(alpha)) * (x -
                                          (beta / self.sqrt_one_minus_alphas_cumprod[t]) * model_output)

        # Compute the variance of the posterior
        variance = self.posterior_variance[t]

        # Sample from the posterior
        if t > 0:
            noise = torch.randn_like(x)
            return mean + torch.sqrt(variance) * noise
        else:
            return mean


class ViewConditioner(nn.Module):
    """View conditioner for conditioning on elevation and azimuth angles"""

    def __init__(self, dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

    def forward(self, elevation, azimuth):
        x = torch.cat([elevation, azimuth], dim=1)
        return self.net(x)


class Zero123PlusPlus(nn.Module):
    def __init__(self, model_path=None, num_timesteps=1000):
        super().__init__()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # Define model architecture
        self.unet = UNet(
            dim=128,
            channels=3,
            dim_mults=(1, 2, 4, 8),
            time_emb_dim=128,
            self_condition=False
        )

        # Define view conditioner
        self.view_conditioner = ViewConditioner(dim=128)

        # Define noise scheduler
        self.noise_scheduler = NoiseScheduler(num_timesteps=num_timesteps)

        # Load pre-trained weights if provided
        if model_path and os.path.exists(model_path):
            self.load_state_dict(torch.load(
                model_path, map_location=self.device))
        else:
            # Download pre-trained weights if not provided
            model_path = self._download_weights()
            if model_path:
                self.load_state_dict(torch.load(
                    model_path, map_location=self.device))

        self.to(self.device)
        self.eval()

    def _download_weights(self):
        """Download pre-trained weights from Hugging Face"""
        # This is a placeholder for the actual download logic
        # In a real implementation, this would download from Hugging Face or other source
        print("Pre-trained weights not found. Please download them manually.")
        print("Visit: https://huggingface.co/sudo-ai/zero123plusplus")
        return None

    def forward(self, x, elevation, azimuth, timestep=None):
        """
        Forward pass of the model
        Args:
            x: Input image tensor
            elevation: Elevation angle
            azimuth: Azimuth angle
            timestep: Diffusion timestep
        Returns:
            Generated novel view tensor or noise prediction
        """
        # Get time embeddings
        if timestep is not None:
            t = self.noise_scheduler.time_mlp(timestep)
        else:
            t = torch.zeros(x.shape[0], device=x.device)

        # Get view condition
        view_condition = self.view_conditioner(elevation, azimuth)

        # Apply cross-attention to condition on view angles
        # In a real implementation, this would use a proper cross-attention mechanism
        # For simplicity, we'll just add the view condition to the time embedding
        t = t + view_condition

        # Forward through U-Net
        output = self.unet(x, t)

        return output

    def generate_novel_views(self, image_tensor, num_views=8, num_inference_steps=20):
        """
        Generate multiple novel views of the input image using diffusion
        Args:
            image_tensor: Preprocessed input image tensor
            num_views: Number of novel views to generate
            num_inference_steps: Number of diffusion steps
        Returns:
            List of generated novel view tensors
        """
        novel_views = []

        # Generate views at different angles
        for i in range(num_views):
            elevation = torch.tensor(
                [30.0], device=self.device)  # Fixed elevation
            azimuth = torch.tensor(
                [i * (360.0 / num_views)], device=self.device)

            with torch.no_grad():
                # Initialize with random noise
                x = torch.randn_like(image_tensor)

                # Denoise step by step
                for t in range(num_inference_steps - 1, -1, -1):
                    timestep = torch.tensor([t], device=self.device)

                    # Predict noise
                    noise_pred = self.forward(x, elevation, azimuth, timestep)

                    # Update sample
                    x = self.noise_scheduler.step(noise_pred, t, x)

                novel_views.append(x)

        return novel_views


def load_model(model_path=None):
    """
    Load the Zero123++ model
    Args:
        model_path: Path to pre-trained weights
    Returns:
        Loaded model
    """
    model = Zero123PlusPlus(model_path)
    return model

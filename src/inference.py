import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import logging
import os
from tqdm import tqdm
import math
from einops import rearrange, repeat

# Cấu hình logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Các lớp cơ bản cho mô hình diffusion


class SinusoidalPositionEmbeddings(nn.Module):
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
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
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

    def forward(self, x, context=None):
        h = self.heads

        q = self.to_q(x)
        context = x if context is None else context
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
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        b, n, _ = x.shape
        h = self.heads

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        sim = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn = sim.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, use_scale_shift_norm=False):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if time_emb_dim is not None else None

        self.block1 = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(1, dim_out),
            nn.SiLU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1),
            nn.GroupNorm(1, dim_out),
            nn.SiLU(),
        )

        self.block2 = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            h = h * (time_emb[..., None, None] + 1)

        return h + self.block2(x)

# Thêm VAE cho Latent Diffusion


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512], latent_dim=4):
        super().__init__()

        modules = []
        for h_dim in hidden_dims:
            modules.extend([
                nn.Conv2d(in_channels, h_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            ])
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.final_mu = nn.Conv2d(hidden_dims[-1], latent_dim, 1)
        self.final_var = nn.Conv2d(hidden_dims[-1], latent_dim, 1)

    def forward(self, x):
        x = self.encoder(x)
        mu = self.final_mu(x)
        log_var = self.final_var(x)
        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, latent_dim=4, hidden_dims=[512, 256, 128, 64], out_channels=3):
        super().__init__()

        hidden_dims = hidden_dims[::-1]  # Reverse to match encoder

        modules = []
        for i in range(len(hidden_dims) - 1):
            modules.extend([
                nn.ConvTranspose2d(
                    hidden_dims[i], hidden_dims[i + 1], 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dims[i + 1]),
                nn.LeakyReLU()
            ])

        modules.extend([
            nn.ConvTranspose2d(
                hidden_dims[-1], out_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        ])

        self.decoder = nn.Sequential(*modules)

    def forward(self, z):
        return self.decoder(z)


class VAE(nn.Module):
    def __init__(self, in_channels=3, hidden_dims=[64, 128, 256, 512], latent_dim=4):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, in_channels)

    def encode(self, x):
        mu, log_var = self.encoder(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

# Cập nhật UNet để hoạt động trong không gian latent


class UNet(nn.Module):
    def __init__(self, dim, init_dim=None, dim_mults=(1, 2, 4, 8), channels=4, condition_dim=None):
        super().__init__()

        # Điều chỉnh kích thước đầu vào cho không gian latent
        self.channels = channels

        # Time embedding
        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # Condition embedding (for view angles)
        self.condition_dim = condition_dim
        if condition_dim is not None:
            self.condition_mlp = nn.Sequential(
                nn.Linear(condition_dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

        # Layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(dim_mults)

        init_dim = init_dim if init_dim is not None else dim
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        # Downsampling
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1

            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_out, time_emb_dim=time_dim),
                ResnetBlock(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(CrossAttention(
                    dim_out, condition_dim if is_last else None)),
                Residual(CrossAttention(
                    dim_out, condition_dim if is_last else None)),
                nn.Conv2d(dim_out, dim_out, 4, 2,
                          1) if not is_last else nn.Identity()
            ]))

        # Middle
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(SelfAttention(mid_dim))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        # Upsampling
        in_out = list(zip(dims[1:][::-1], dims[:-1][::-1]))

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= len(in_out) - 1

            self.ups.append(nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_in, time_emb_dim=time_dim),
                ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(CrossAttention(
                    dim_in, condition_dim if is_last else None)),
                Residual(CrossAttention(
                    dim_in, condition_dim if is_last else None)),
                nn.ConvTranspose2d(dim_in, dim_in, 4, 2,
                                   1) if not is_last else nn.Identity()
            ]))

        # Final
        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, channels, 1)

    def forward(self, x, time, condition=None):
        # Time embedding
        t = self.time_mlp(time)

        # Condition embedding
        if condition is not None and self.condition_dim is not None:
            c = self.condition_mlp(condition)
            t = t + c

        # Initial convolution
        h = self.init_conv(x)
        h_first = h.clone()

        # Downsampling
        h_stack = []
        for block1, block2, attn1, attn2, downsample in self.downs:
            h = block1(h, t)
            h = block2(h, t)
            h = attn1(h, condition)
            h = attn2(h, condition)
            h_stack.append(h)
            h = downsample(h)

        # Middle
        h = self.mid_block1(h, t)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t)

        # Upsampling
        for block1, block2, attn1, attn2, upsample in self.ups:
            h = torch.cat((h, h_stack.pop()), dim=1)
            h = block1(h, t)
            h = block2(h, t)
            h = attn1(h, condition)
            h = attn2(h, condition)
            h = upsample(h)

        # Final
        h = torch.cat((h, h_first), dim=1)
        h = self.final_res_block(h, t)
        return self.final_conv(h)


class NoiseScheduler:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(
            1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(
            1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(
            1.0 / self.alphas_cumprod - 1)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * \
            (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = self.betas * \
            torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (
            1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def add_noise(self, x, t):
        noise = torch.randn_like(x)
        return self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x + self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1) * noise, noise

    def get_alpha(self, t):
        return self.alphas[t]

    def get_alpha_bar(self, t):
        return self.alphas_cumprod[t]

    def get_beta(self, t):
        return self.betas[t]

    def get_sqrt_alpha_bar(self, t):
        return self.sqrt_alphas_cumprod[t]

    def get_sqrt_one_minus_alpha_bar(self, t):
        return self.sqrt_one_minus_alphas_cumprod[t]

    def get_posterior_mean(self, x_0, x_t, t):
        return self.posterior_mean_coef1[t].view(-1, 1, 1, 1) * x_0 + self.posterior_mean_coef2[t].view(-1, 1, 1, 1) * x_t

    def get_posterior_variance(self, t):
        return self.posterior_variance[t]

    def get_posterior_log_variance_clipped(self, t):
        return self.posterior_log_variance_clipped[t]


class ViewConditioner:
    def __init__(self, elevation_range=(-20, 40), azimuth_range=(0, 360)):
        self.elevation_range = elevation_range
        self.azimuth_range = azimuth_range

    def normalize_elevation(self, elevation):
        min_elev, max_elev = self.elevation_range
        return (elevation - min_elev) / (max_elev - min_elev)

    def normalize_azimuth(self, azimuth):
        min_azim, max_azim = self.azimuth_range
        return (azimuth - min_azim) / (max_azim - min_azim)

    def encode_view(self, elevation, azimuth):
        norm_elev = self.normalize_elevation(elevation)
        norm_azim = self.normalize_azimuth(azimuth)

        # Convert to radians for sin/cos encoding
        elev_rad = norm_elev * 2 * np.pi
        azim_rad = norm_azim * 2 * np.pi

        # Sin/cos encoding
        elev_sin = np.sin(elev_rad)
        elev_cos = np.cos(elev_rad)
        azim_sin = np.sin(azim_rad)
        azim_cos = np.cos(azim_rad)

        return torch.tensor([elev_sin, elev_cos, azim_sin, azim_cos], dtype=torch.float32)


class Zero123PlusPlus(nn.Module):
    def __init__(self, image_size=256, channels=3, dim=128, condition_dim=4, latent_dim=4):
        super().__init__()

        # VAE for latent diffusion
        self.vae = VAE(in_channels=channels, latent_dim=latent_dim)

        # UNet for diffusion in latent space
        self.unet = UNet(
            dim=dim,
            channels=latent_dim,  # Changed from channels to latent_dim
            condition_dim=condition_dim
        )

        # Noise scheduler
        self.noise_scheduler = NoiseScheduler()

        # View conditioner
        self.view_conditioner = ViewConditioner()

        # Image size
        self.image_size = image_size

        # Transform for input images
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

        logger.info(
            f"Initialized Zero123PlusPlus model with latent diffusion on {self.device}")

    def encode_image(self, image):
        """Encode image to latent space using VAE"""
        with torch.no_grad():
            mu, log_var = self.vae.encode(image)
            z = self.vae.reparameterize(mu, log_var)
        return z

    def decode_latent(self, z):
        """Decode latent representation to image using VAE"""
        with torch.no_grad():
            image = self.vae.decode(z)
        return image

    def generate_novel_views(self, input_image, num_views=8, elevation_range=(-20, 40), azimuth_range=(0, 360), num_inference_steps=50):
        """
        Generate novel views of the input image using latent diffusion

        Args:
            input_image: PIL Image or tensor
            num_views: Number of novel views to generate
            elevation_range: Range of elevation angles
            azimuth_range: Range of azimuth angles
            num_inference_steps: Number of denoising steps

        Returns:
            List of PIL Images
        """
        logger.info(
            f"Generating {num_views} novel views with latent diffusion")

        # Process input image
        if isinstance(input_image, Image.Image):
            input_image = self.transform(
                input_image).unsqueeze(0).to(self.device)
        else:
            input_image = input_image.to(self.device)

        # Encode input image to latent space
        with torch.no_grad():
            input_latent = self.encode_image(input_image)
            logger.info(
                f"Encoded input image to latent space with shape {input_latent.shape}")

            # Clear memory after encoding
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        # Generate novel views
        novel_views = []

        # Calculate view angles
        elevations = np.linspace(
            elevation_range[0], elevation_range[1], num_views)
        azimuths = np.linspace(azimuth_range[0], azimuth_range[1], num_views)

        for i in range(num_views):
            elevation = elevations[i]
            azimuth = azimuths[i]

            # Encode view angles
            view_encoding = self.view_conditioner.encode_view(
                elevation, azimuth).to(self.device)

            # Initialize latent with noise
            latent = torch.randn_like(input_latent)

            # Denoising loop
            for t in tqdm(range(num_inference_steps - 1, -1, -1), desc=f"Generating view {i+1}/{num_views}"):
                # Get timestep
                timestep = torch.tensor([t], device=self.device)

                # Predict noise
                noise_pred = self.unet(latent, timestep, view_encoding)

                # Update latent
                alpha = self.noise_scheduler.get_alpha(t)
                alpha_bar = self.noise_scheduler.get_alpha_bar(t)
                beta = self.noise_scheduler.get_beta(t)

                if t > 0:
                    noise = torch.randn_like(latent)
                else:
                    noise = 0

                latent = (1 / torch.sqrt(alpha)) * (latent - (beta /
                                                              torch.sqrt(1 - alpha_bar)) * noise_pred) + torch.sqrt(beta) * noise

                # Clear memory periodically
                if t % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Decode latent to image
            with torch.no_grad():
                novel_view = self.decode_latent(latent)

            # Convert to PIL Image and move to CPU to free GPU memory
            novel_view = novel_view.squeeze(0).cpu()
            novel_view = (novel_view + 1) / 2  # Denormalize
            novel_view = novel_view.clamp(0, 1)
            novel_view = transforms.ToPILImage()(novel_view)

            novel_views.append(novel_view)
            logger.info(
                f"Generated novel view {i+1}/{num_views} with elevation={elevation:.1f}, azimuth={azimuth:.1f}")

            # Clear memory after each view
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()

        return novel_views

    def save_model(self, path):
        """Save model to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.state_dict(),
            'vae_state_dict': self.vae.state_dict(),
            'unet_state_dict': self.unet.state_dict(),
            'noise_scheduler': self.noise_scheduler,
            'view_conditioner': self.view_conditioner,
            'image_size': self.image_size
        }, path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        """Load model from disk"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.vae.load_state_dict(checkpoint['vae_state_dict'])
            self.unet.load_state_dict(checkpoint['unet_state_dict'])
            self.noise_scheduler = checkpoint['noise_scheduler']
            self.view_conditioner = checkpoint['view_conditioner']
            self.image_size = checkpoint['image_size']
            logger.info(f"Model loaded from {path}")
        else:
            logger.warning(f"Model checkpoint not found at {path}")

    def optimize_for_inference(self):
        """
        Optimize the model for inference by:
        1. Converting to half precision
        2. Enabling gradient checkpointing
        3. Setting to evaluation mode
        """
        logger.info("Optimizing model for inference...")

        # Set to evaluation mode
        self.eval()

        # Convert to half precision if on CUDA
        if self.device.type == 'cuda':
            self.half()
            logger.info("Model converted to half precision")

            # Enable gradient checkpointing for UNet
            if hasattr(self.unet, 'gradient_checkpointing_enable'):
                self.unet.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled for UNet")

            # Clear cache
            torch.cuda.empty_cache()
            logger.info("GPU cache cleared")

        return self

"""
TabDDPM-inspired tabular classifier for machine unlearning experiments.

This implementation keeps the repo's binary classification interface intact:
  - forward(x_num, x_cat) -> logits
  - attach_lora(...)
  - get_lora_params()

During training, it samples diffusion timesteps and injects Gaussian noise into
tabular feature tokens. At evaluation time, it runs the deterministic clean
path (t = 0), which keeps the rest of the pipeline unchanged.
"""

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lora import LoRALinear, freeze_non_lora


class NumericEmbedding(nn.Module):
    """Embeds each numeric feature into a token."""

    def __init__(self, num_features: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_features, d_model))
        self.bias = nn.Parameter(torch.empty(num_features, d_model))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


class CategoricalEmbedding(nn.Module):
    """One embedding table per categorical feature."""

    def __init__(self, cat_dims: List[int], d_model: int):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(dim, d_model) for dim in cat_dims])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack([emb(x[:, i]) for i, emb in enumerate(self.embeddings)], dim=1)


class CLSToken(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.token = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.trunc_normal_(self.token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cls = self.token.expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard DDPM sinusoidal timestep embedding."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        device = timesteps.device
        half_dim = self.d_model // 2
        exponent = -math.log(10000.0) * torch.arange(half_dim, device=device) / max(half_dim - 1, 1)
        freqs = torch.exp(exponent)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.d_model % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.proj(emb)


class DiffusionTransformerBlock(nn.Module):
    """Self-attention block with FiLM-style timestep conditioning."""

    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.time_attn = nn.Linear(d_model, d_model * 2)
        self.time_ffn = nn.Linear(d_model, d_model * 2)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        head_dim = d_model // self.n_heads

        residual = x
        x_norm = self.norm1(x)
        shift_attn, scale_attn = self.time_attn(t_emb).chunk(2, dim=-1)
        x_norm = self._modulate(x_norm, shift_attn, scale_attn)

        q = self.q_proj(x_norm).view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        k = self.k_proj(x_norm).view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)
        v = self.v_proj(x_norm).view(batch_size, seq_len, self.n_heads, head_dim).transpose(1, 2)

        scale = math.sqrt(head_dim)
        attn = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        x = residual + self.dropout(self.out_proj(out))

        residual = x
        x_norm = self.norm2(x)
        shift_ffn, scale_ffn = self.time_ffn(t_emb).chunk(2, dim=-1)
        x_norm = self._modulate(x_norm, shift_ffn, scale_ffn)
        x = residual + self.ffn(x_norm)
        return x


class TabDDPM(nn.Module):
    """
    Diffusion-conditioned tabular classifier inspired by TabDDPM.

    It tokenizes numeric and categorical features, injects DDPM noise during
    training, then predicts a binary logit from a CLS representation.
    """

    def __init__(
        self,
        num_num_features: int = 0,
        cat_dims: Optional[List[int]] = None,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_factor: int = 4,
        dropout: float = 0.1,
        num_diffusion_steps: int = 128,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        cat_dims = cat_dims or []
        if num_num_features == 0 and len(cat_dims) == 0:
            raise ValueError("TabDDPM requires at least one numeric or categorical feature.")
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")

        self.num_num_features = num_num_features
        self.num_cat_features = len(cat_dims)
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_diffusion_steps = num_diffusion_steps

        self.num_embed = NumericEmbedding(num_num_features, d_model) if num_num_features > 0 else None
        self.cat_embed = CategoricalEmbedding(cat_dims, d_model) if cat_dims else None
        self.cls_token = CLSToken(d_model)
        self.time_embed = SinusoidalTimeEmbedding(d_model)

        ffn_dim = d_model * ffn_factor
        self.blocks = nn.ModuleList(
            [DiffusionTransformerBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alpha_bars)

    def tokenize(self, x_num: Optional[torch.Tensor], x_cat: Optional[torch.Tensor]) -> torch.Tensor:
        tokens = []
        if x_num is not None and self.num_embed is not None:
            tokens.append(self.num_embed(x_num))
        if x_cat is not None and self.cat_embed is not None:
            tokens.append(self.cat_embed(x_cat))
        if not tokens:
            raise ValueError("TabDDPM received no usable features.")
        return torch.cat(tokens, dim=1)

    def _prepare_timesteps(self, timesteps: Optional[torch.Tensor], batch_size: int, device) -> torch.Tensor:
        if timesteps is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor(timesteps, dtype=torch.long, device=device)
        timesteps = timesteps.to(device=device, dtype=torch.long)
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(batch_size)
        return timesteps

    def _apply_diffusion_noise(self, tokens: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alphas_cumprod[timesteps].view(-1, 1, 1)
        noise = torch.randn_like(tokens)
        return torch.sqrt(alpha_bar) * tokens + torch.sqrt(1.0 - alpha_bar) * noise

    def _encode(self, tokens: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embed(timesteps)
        tokens = self.cls_token(tokens)
        for block in self.blocks:
            tokens = block(tokens, t_emb)
        tokens = self.norm(tokens)
        return tokens[:, 0]

    def forward(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        add_noise: bool = False,
    ) -> torch.Tensor:
        tokens = self.tokenize(x_num, x_cat)
        timesteps = self._prepare_timesteps(timesteps, tokens.size(0), tokens.device)
        if add_noise:
            tokens = self._apply_diffusion_noise(tokens, timesteps)
        cls_repr = self._encode(tokens, timesteps)
        return self.head(cls_repr).squeeze(-1)

    def compute_training_loss(
        self,
        x_num: Optional[torch.Tensor],
        x_cat: Optional[torch.Tensor],
        y: torch.Tensor,
        criterion: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        if criterion is None:
            criterion = nn.BCEWithLogitsLoss()

        batch_size = y.size(0)
        device = y.device
        timesteps = torch.randint(
            low=1,
            high=self.num_diffusion_steps,
            size=(batch_size,),
            device=device,
        )

        noisy_logits = self.forward(x_num, x_cat, timesteps=timesteps, add_noise=True)
        clean_logits = self.forward(x_num, x_cat)
        noisy_loss = criterion(noisy_logits, y)
        clean_loss = criterion(clean_logits, y)
        return 0.5 * (clean_loss + noisy_loss)

    def attach_lora(self, r: int = 8, lora_alpha: float = 16.0, lora_dropout: float = 0.0):
        lora_layers = {}
        for i, block in enumerate(self.blocks):
            block.q_proj = LoRALinear(
                block.q_proj,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            block.v_proj = LoRALinear(
                block.v_proj,
                r=r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            lora_layers[f"block.{i}.q_proj"] = block.q_proj
            lora_layers[f"block.{i}.v_proj"] = block.v_proj
        freeze_non_lora(self)
        return lora_layers

    def get_lora_params(self):
        params = []
        for block in self.blocks:
            for proj in (block.q_proj, block.v_proj):
                if isinstance(proj, LoRALinear):
                    params.extend(proj.trainable_parameters())
        return params

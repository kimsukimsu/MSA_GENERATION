"""
MSAFlow Latent Flow Matching encoder.

Maps a single protein sequence's ESM2 embedding to a synthetic MSA embedding,
enabling zero-shot MSA generation for orphan / low-homology proteins.

Architecture (from paper, Section 3.2 & Table 12):
  • Conditional rectified flow (straight-line paths in R^(L×128))
  • Input projection:      FC(128 → 768)      [MSA embedding dim]
  • Conditioning proj:     FC(1280 → 768)     [ESM2-650M dim]
  • 12 × DiTAdaLN blocks, hidden 768, 12 heads
  • Output projection:     FC(768 → 128)
  • Loss: E[‖v_θ(z_t, e, t) − (z_0 − z_1)‖²]  (rectified flow, Eq. 6)

Inference SDE (Eq. 7, 8, 9): Euler-Maruyama with temperature parameter T.
    z_{t−Δt} = z_t − v_θ(z_t,e,t)Δt − T·g_t·√Δt·ε
where g_t = sqrt(2t/(1−t)) is the diffusion coefficient.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from msaflow.models.sfm_decoder import SinusoidalTimeEmbedding, PosWiseAdaLNBlock


# ──────────────────────────────────────────────────────────────────────────────
# Latent FM velocity network
# ──────────────────────────────────────────────────────────────────────────────

class LatentFMEncoder(nn.Module):
    """
    Conditional rectified flow model that maps ESM2 embedding to MSA embedding.

    Given a noisy latent z_t = (1-t)·z_1 + t·z_0 (where z_1 is the MSA
    embedding and z_0 ~ N(0,I) is Gaussian noise), predicts the velocity
    v_θ(z_t, e, t) that should equal (z_0 − z_1) under the rectified flow
    objective.

    Args:
        msa_dim:     Dimension of MSA embedding (128).
        esm_dim:     Dimension of ESM2 embedding (1280 for ESM2-650M).
        hidden_size: Transformer hidden dim (768).
        depth:       Number of transformer blocks (12).
        num_heads:   Attention heads (12).
        mlp_ratio:   FFN expansion ratio (4.0).
        max_seq_len: Maximum sequence length.
    """

    def __init__(
        self,
        msa_dim: int = 128,
        esm_dim: int = 1280,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.msa_dim = msa_dim
        self.hidden_size = hidden_size

        # ── Input projection (noisy MSA latent) ──────────────────────────────
        self.input_proj = nn.Linear(msa_dim, hidden_size)

        # ── Positional encoding ───────────────────────────────────────────────
        self.register_buffer(
            "pos_emb",
            self._build_sincos_pos_emb(max_seq_len, hidden_size),
            persistent=False,
        )

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_emb = SinusoidalTimeEmbedding(hidden_size)

        # ── ESM2 conditioning projection ──────────────────────────────────────
        self.cond_proj = nn.Linear(esm_dim, hidden_size)

        # ── Transformer blocks (position-wise AdaLN) ──────────────────────────
        self.blocks = nn.ModuleList([
            PosWiseAdaLNBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # ── Output projection ─────────────────────────────────────────────────
        self.norm_out = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, msa_dim)

        self._init_weights()

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.normal_(self.time_emb.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_emb.mlp[2].weight, std=0.02)

    @staticmethod
    def _build_sincos_pos_emb(max_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1).float()
        i = torch.arange(dim // 2).float()
        angle = pos / (10000 ** (2 * i / dim))
        emb = torch.cat([torch.sin(angle), torch.cos(angle)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros(max_len, 1)], dim=-1)
        return emb.unsqueeze(0)  # (1, max_len, dim)

    def forward(
        self,
        z_t: torch.Tensor,
        esm_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_t:     (B, L, msa_dim) — noisy MSA embedding at time t.
            esm_emb: (B, L, esm_dim) — ESM2 representation of query sequence.
            t:       (B,)            — timesteps in [0, 1].

        Returns:
            v: (B, L, msa_dim) — predicted velocity.
        """
        B, L, _ = z_t.shape

        h = self.input_proj(z_t) + self.pos_emb[:, :L, :]      # (B, L, H)

        t_emb = self.time_emb(t).unsqueeze(1).expand(B, L, -1) # (B, L, H)
        cond = t_emb + self.cond_proj(esm_emb)                  # (B, L, H)

        for block in self.blocks:
            h = block(h, cond)

        return self.output_proj(self.norm_out(h))               # (B, L, msa_dim)


# ──────────────────────────────────────────────────────────────────────────────
# Rectified flow training loss
# ──────────────────────────────────────────────────────────────────────────────

def rectified_flow_loss(
    model: LatentFMEncoder,
    z1: torch.Tensor,
    esm_emb: torch.Tensor,
) -> torch.Tensor:
    """
    Compute rectified flow loss (Eq. 6 in paper).

    L_RFM = E_{t, z_0~N(0,I), z_1}[‖v_θ(z_t, e, t) − (z_0 − z_1)‖²]

    The straight-line path is z_t = (1-t)·z_1 + t·z_0, so the reference
    velocity is the constant field u*(z_t; z_0, z_1) = z_0 − z_1.

    Args:
        model:   LatentFMEncoder.
        z1:      (B, L, msa_dim) ground-truth MSA embedding.
        esm_emb: (B, L, esm_dim) ESM2 embedding of query sequence.

    Returns:
        loss: scalar.
    """
    B = z1.shape[0]
    device = z1.device

    z0 = torch.randn_like(z1)                                   # (B, L, msa_dim)
    t = torch.rand(B, device=device)                            # (B,)
    t_bcast = t.view(B, 1, 1)                                   # broadcast over L, msa_dim

    z_t = (1.0 - t_bcast) * z1 + t_bcast * z0                  # linear interpolation

    target = z0 - z1                                            # constant reference velocity
    v_pred = model(z_t, esm_emb, t)

    return F.mse_loss(v_pred, target)


# ──────────────────────────────────────────────────────────────────────────────
# Inference sampling
# ──────────────────────────────────────────────────────────────────────────────

def _diffusion_coeff(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """g_t = sqrt(2t / (1−t)) — diffusion coefficient for SDE (Eq. 8)."""
    return torch.sqrt((2.0 * t) / (1.0 - t).clamp(min=eps))


def _score_from_velocity(
    v: torch.Tensor,
    z_t: torch.Tensor,
    t: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Convert velocity prediction to score s_θ = ∇ log p(z_t|e, t).

    For the rectified flow z_t = (1−t)·z_1 + t·z_0, the score is:
        s_θ(z_t, t) = −(v_θ·(1−t) + z_t) / t

    This follows from the relationship between the score and the
    velocity field in the probability-flow ODE.
    """
    denom = t.clamp(min=eps)
    return -(v * (1.0 - t).view(-1, 1, 1) + z_t) / denom.view(-1, 1, 1)


@torch.no_grad()
def sample_msa_embeddings(
    model: LatentFMEncoder,
    esm_emb: torch.Tensor,
    n_steps: int = 100,
    temperature: float = 0.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Generate synthetic MSA embeddings from ESM2 embeddings (zero-shot mode).

    Implements Euler-Maruyama integration of the reverse-time SDE (Eq. 9):
        z_{t−Δt} = z_t − [v_θ − ½·g_t²·s_θ]·Δt − T·g_t·√Δt·ε

    When temperature=0 (default), reduces to the deterministic probability-flow ODE.

    Args:
        model:       LatentFMEncoder.
        esm_emb:     (B, L, esm_dim) ESM2 embeddings.
        n_steps:     Number of Euler steps (default: 100).
        temperature: SDE noise temperature T ∈ [0, 1].
                     0 = deterministic ODE, 1 = full SDE (training-equivalent).
        eps:         numerical stability constant.

    Returns:
        z1: (B, L, msa_dim) generated MSA embedding.
    """
    device = esm_emb.device
    B, L, _ = esm_emb.shape

    # Start from Gaussian noise (t=1 in the forward process)
    z_t = torch.randn(B, L, model.msa_dim, device=device, dtype=esm_emb.dtype)

    dt = 1.0 / n_steps
    # Integrate from t=1 (noise) down to t=0 (data)
    t_vals = torch.linspace(1.0, dt, n_steps, device=device)

    for t_val in t_vals:
        t = t_val.expand(B)                              # (B,)
        v = model(z_t, esm_emb, t)                      # (B, L, msa_dim)

        if temperature > 0.0:
            g_t = _diffusion_coeff(t_val, eps=eps)       # scalar
            score = _score_from_velocity(v, z_t, t_val, eps=eps)
            drift = v - 0.5 * (g_t ** 2) * score
            noise = temperature * g_t * math.sqrt(dt) * torch.randn_like(z_t)
            z_t = z_t - drift * dt - noise
        else:
            # Deterministic ODE
            z_t = z_t - v * dt

    return z_t

"""
MSAFlow SFM Decoder — Statistical Flow Matching decoder for MSA sequence generation.

Architecture (from paper, Section 3.1.2):
  • 12 × DiT blocks, hidden dim 768, 12 attention heads  (~130M params)
  • Input projection:      FC(22 → 768)          [22 = 20 AA + gap + X]
  • Conditioning proj:     FC(128 → 768)          [per-residue MSA embedding]
  • Time embedding:        sinusoidal → MLP → 768
  • Position-wise AdaLN:   scale/shift computed per residue from (t_emb + cond)
  • Output projection:     FC(768 → 22)

The key innovation over vanilla DiT is **position-wise AdaLN**: each residue
gets its own (shift, scale, gate) derived from the compressed MSA embedding
at that position, enabling fine-grained evolutionary conditioning.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding → MLP, output dim = hidden_size."""

    def __init__(self, hidden_size: int, freq_dim: int = 256):
        super().__init__()
        self.freq_dim = freq_dim
        self.mlp = nn.Sequential(
            nn.Linear(freq_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def _sinusoidal(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)   # (B, half)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (B,) → (B, hidden_size)."""
        t_freq = self._sinusoidal(t, self.freq_dim)
        return self.mlp(t_freq)


# ──────────────────────────────────────────────────────────────────────────────
# Position-wise AdaLN modulation
# ──────────────────────────────────────────────────────────────────────────────

def modulate_poswise(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Apply position-wise adaptive layer norm modulation.

    x:     (B, L, H)
    shift: (B, L, H)
    scale: (B, L, H)
    """
    return x * (1.0 + scale) + shift


class PosWiseAdaLNBlock(nn.Module):
    """DiT block with position-wise AdaLN conditioning.

    Unlike standard DiT where conditioning is global (one scale/shift for all
    tokens), here each residue position is modulated by its own scale/shift
    derived from the per-position conditioning vector c ∈ R^(B×L×H).
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_hidden, hidden_size),
        )

        # Position-wise modulation: c (B,L,H) → 6 × (B,L,H)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size),
        )
        # Zero-init so at the start of training, blocks are identity
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, H)  — noisy sequence representation
        c: (B, L, H)  — per-position conditioning (t_emb broadcast + cond_proj)
        """
        mod = self.adaLN_modulation(c)          # (B, L, 6*H)
        shift_sa, scale_sa, gate_sa, shift_ff, scale_ff, gate_ff = mod.chunk(6, dim=-1)

        # Self-attention with position-wise AdaLN pre-norm
        h = modulate_poswise(self.norm1(x), shift_sa, scale_sa)
        attn_out, _ = self.attn(h, h, h)
        x = x + gate_sa * attn_out

        # Feed-forward with position-wise AdaLN pre-norm
        h = modulate_poswise(self.norm2(x), shift_ff, scale_ff)
        x = x + gate_ff * self.mlp(h)

        return x


# ──────────────────────────────────────────────────────────────────────────────
# Final projection layer
# ──────────────────────────────────────────────────────────────────────────────

class FinalLayer(nn.Module):
    """Final position-wise AdaLN + linear output layer.

    Uses 3-way modulation (shift, scale, gate) to match the trained checkpoint.
    """

    def __init__(self, hidden_size: int, out_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 3 * hidden_size),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """x, c: (B, L, H) → (B, L, out_dim)."""
        mod = self.adaLN_modulation(c)          # (B, L, 3*H)
        shift, scale, gate = mod.chunk(3, dim=-1)
        x = gate * modulate_poswise(self.norm(x), shift, scale)
        return self.linear(x)


# ──────────────────────────────────────────────────────────────────────────────
# Main SFM Decoder
# ──────────────────────────────────────────────────────────────────────────────

class SFMDecoder(nn.Module):
    """
    Statistical Flow Matching decoder for MSA sequence generation.

    Given a compressed MSA embedding m_seq ∈ R^(B×L×msa_dim) and a noisy
    sequence x_t ∈ R^(B×L×vocab_size) on the unit sphere, predict the
    velocity field v_θ(x_t | m_seq, t) ∈ R^(B×L×vocab_size).

    Args:
        vocab_size:  V = 22 (20 AA + gap + X).
        msa_dim:     Dimension of compressed MSA embedding (128 from AF3).
        hidden_size: Transformer hidden dim (768).
        depth:       Number of transformer blocks (12).
        num_heads:   Attention heads (12).
        mlp_ratio:   FFN expansion ratio (4.0).
        max_seq_len: Maximum sequence length for sinusoidal positional encoding.
    """

    def __init__(
        self,
        vocab_size: int = 22,
        msa_dim: int = 128,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        max_seq_len: int = 2048,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # ── Input projection ──────────────────────────────────────────────────
        self.input_proj = nn.Linear(vocab_size, hidden_size)

        # ── Positional encoding (fixed sin-cos) ───────────────────────────────
        self.register_buffer(
            "pos_emb",
            self._build_sincos_pos_emb(max_seq_len, hidden_size),
            persistent=False,
        )

        # ── Time embedding ────────────────────────────────────────────────────
        self.time_emb = SinusoidalTimeEmbedding(hidden_size)

        # ── Conditioning projection (per-residue MSA emb → hidden) ────────────
        # LayerNorm normalises AF3 MSA embeddings (std >> 1) before projection.
        # bias=False to match trained checkpoint.
        self.msa_norm = nn.LayerNorm(msa_dim, bias=False)
        self.cond_proj = nn.Linear(msa_dim, hidden_size)

        # ── Transformer blocks ────────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            PosWiseAdaLNBlock(hidden_size, num_heads, mlp_ratio)
            for _ in range(depth)
        ])

        # Per-block conditioning projections (bias=False, checkpoint convention).
        # Input is hidden_size (after cond_proj), not msa_dim.
        # Each block applies its own linear to the shared MSA representation.
        self.block_cond_projs = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size, bias=False)
            for _ in range(depth)
        ])

        # ── Output ────────────────────────────────────────────────────────────
        self.final_layer = FinalLayer(hidden_size, vocab_size)

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self):
        def _basic(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.apply(_basic)
        # Time MLP
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

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(
        self,
        x_t: torch.Tensor,
        m_seq: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x_t:   (B, L, vocab_size) — noisy sequence on the unit sphere.
            m_seq: (B, L, msa_dim)    — compressed MSA embedding per residue.
            t:     (B,)               — timesteps in [0, 1].

        Returns:
            v: (B, L, vocab_size) — predicted velocity in tangent space.
        """
        B, L, _ = x_t.shape

        # Project input sequence to hidden space + positional encoding
        h = self.input_proj(x_t) + self.pos_emb[:, :L, :]       # (B, L, H)

        # Build per-position conditioning
        t_emb = self.time_emb(t).unsqueeze(1).expand(B, L, -1)  # (B, L, H)
        m_proj = self.cond_proj(self.msa_norm(m_seq))            # (B, L, H)

        # Each block uses its own linear transform of the shared MSA projection
        for block, block_cond_proj in zip(self.blocks, self.block_cond_projs):
            cond = t_emb + block_cond_proj(m_proj)               # (B, L, H)
            h = block(h, cond)

        # Final projection back to vocab space
        v = self.final_layer(h, t_emb + m_proj)                  # (B, L, vocab_size)

        # Project onto tangent space at x_t: remove radial component
        # v_tangent = v - <v, x_t> * x_t
        x_t_f32 = x_t.float()
        v_tangent = v.float() - (v.float() * x_t_f32).sum(dim=-1, keepdim=True) * x_t_f32
        return v_tangent.to(v.dtype)                             # (B, L, vocab_size)


# ──────────────────────────────────────────────────────────────────────────────
# SFM training loss
# ──────────────────────────────────────────────────────────────────────────────

def sfm_loss(
    model: SFMDecoder,
    tokens: torch.Tensor,
    m_seq: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    t_max_eps: float = 0.01,
) -> torch.Tensor:
    """
    Compute SFM training loss for one batch (Eq. 3 in the paper).

    L_SFM(θ) = E_{t~U[0,1], s_i, µ_0}[‖v_θ(x_t|m,t) − u_t(x_t|x_0,x_1)‖²]

    Numerical stability fixes (original code had grad_norm=1305 due to bf16 ops):

      Fix 1 — float32 geometry: x0, x1, x_t and u_t computed in float32
        independent of model autocast dtype.  At t=0.99 in bfloat16, the
        log_map direction (y − dot·x) has SNR≈0.02; in float32 SNR≈750.

      Fix 2 — float32 MSE: v_pred upcast before the squared error.

      Fix 3 — time truncation: t ∈ [0, 1-t_max_eps] avoids the t=1 endpoint
        where log_map involves near-identical points.  With t_max_eps=0.01
        the remaining angle is 1% of the geodesic, giving SNR>100 in float32.

    NOTE: (1-t)² time-weighting was previously applied but has been removed.
    The paper loss (Eq. 3) uses uniform weighting over t.  The (1-t)² variant
    created a severe train–inference mismatch: gradient signal at t>0.7 was
    <9% of the t=0 signal, so the model never learned the high-t regime.
    Inference integrates t=0→1 uniformly, causing garbage output in that range.

    Args:
        model:      SFMDecoder.
        tokens:     (B, L) integer token ids in [0, vocab_size).
        m_seq:      (B, L, msa_dim) compressed MSA embedding.
        weights:    (B,) optional per-sample loss weights (Neff reweighting).
        padding_mask: (B, L) bool tensor — True at real positions, False at padding.
                      Excludes padded positions from the loss (relevant when batch_size>1
                      and MSAs of different lengths are packed into the same batch).
        eps:        numerical stability constant for sphere ops.
        t_max_eps:  upper time truncation margin (default 0.01 → t ∈ [0, 0.99]).

    Returns:
        loss: scalar.
    """
    from msaflow.utils.spherical import (
        onehot_to_sphere,
        sample_sphere_noise,
        geodesic_interpolate,
        target_velocity,
    )

    B, L = tokens.shape
    device = tokens.device

    # Fix 1: x1, x0, x_t and u_t all in float32 for stable sphere geometry
    x1 = onehot_to_sphere(tokens, model.vocab_size, eps=eps)   # (B, L, V) float32
    x0 = sample_sphere_noise((B, L, model.vocab_size), device=device, dtype=torch.float32)

    # Fix 3: truncate t away from the t=1 singularity
    t = torch.rand(B, device=device) * (1.0 - t_max_eps)      # (B,)  ∈ [0, 1-t_max_eps]
    t_bcast = t.view(B, 1, 1)                                  # (B, 1, 1)

    x_t = geodesic_interpolate(x0, x1, t_bcast, eps=eps)      # (B, L, V) float32
    u_t = target_velocity(x_t, x1, t_bcast, eps=eps)          # (B, L, V) float32

    # Predicted velocity (may be bfloat16 under Accelerate autocast)
    v_pred = model(x_t, m_seq, t)                              # (B, L, V)

    # Fix 2: upcast to float32 before MSE
    loss = ((v_pred.float() - u_t) ** 2).sum(dim=-1)           # (B, L)

    if padding_mask is not None:
        # padding_mask: (B, L) bool — zero out padded positions so they don't
        # contribute to the gradient (avoids training the model to predict
        # token 0 / Alanine at padding positions).
        loss = loss * padding_mask.float()                     # (B, L)

    if weights is not None:
        # weights: (B,) — down-weight redundant sequences via Neff reweighting
        loss = loss * weights.unsqueeze(1)                     # (B, L)

    # Normalise by the number of real (non-padded) positions so the loss scale
    # stays invariant to sequence length and batch composition.
    # When padding_mask is None every position is real, so this equals loss.mean().
    if padding_mask is not None:
        n_real = padding_mask.float().sum().clamp(min=1.0)
    else:
        n_real = loss.numel()
    return loss.sum() / n_real

"""
MSAFlow inference pipeline.

Three generation modes (Section 3.3 of the paper):

1. reconstruct(msa_seqs)
   MSA Compression & Reconstruction — encode a deep MSA with Protenix and
   decode 32 sequences with the SFM decoder.

2. augment_shallow(shallow_seqs)
   Few-shot augmentation — encode a shallow MSA, then ALSO generate synthetic
   embeddings from ESM2 and combine both tracks (Syn+Rec, Section 6.5).

3. generate_zeroshot(query_seq)
   Zero-shot augmentation — generate MSA embeddings purely from ESM2
   representation of the query sequence, then decode.

All modes return a list of generated amino acid sequences (no gaps) that can
be written as a FASTA / A3M file.
"""

import sys
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np

from msaflow.models.sfm_decoder import SFMDecoder
from msaflow.models.latent_fm import LatentFMEncoder, sample_msa_embeddings
from msaflow.utils.spherical import sample_sphere_noise, euler_step_sphere, decode_sequences
from msaflow.data.preprocessing import (
    AA_LIST,
    VOCAB_SIZE,
    parse_a3m,
    filter_msa,
    extract_msa_embedding_protenix,
    extract_esm_embedding,
)

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Model loading helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_sfm_decoder(checkpoint_path: str, device: torch.device) -> SFMDecoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    # Infer vocab_size and msa_dim from checkpoint weights
    vocab_size = state["input_proj.weight"].shape[1]
    msa_dim = state["cond_proj.weight"].shape[1]
    hidden_size = state["input_proj.weight"].shape[0]
    depth = sum(1 for k in state if k.startswith("blocks.") and k.endswith(".mlp.0.weight"))
    model = SFMDecoder(
        vocab_size=vocab_size,
        msa_dim=msa_dim,
        hidden_size=hidden_size,
        depth=depth,
    )
    model.load_state_dict(state)
    return model.eval().to(device)


def load_latent_fm(checkpoint_path: str, device: torch.device) -> LatentFMEncoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model", ckpt)
    msa_dim = state["output_proj.weight"].shape[0]
    esm_dim = state["cond_proj.weight"].shape[1]
    hidden_size = state["input_proj.weight"].shape[0]
    depth = sum(1 for k in state if k.startswith("blocks.") and k.endswith(".mlp.0.weight"))
    model = LatentFMEncoder(msa_dim=msa_dim, esm_dim=esm_dim, hidden_size=hidden_size, depth=depth)
    model.load_state_dict(state)
    return model.eval().to(device)


def load_esm2(device: torch.device):
    sys.path.insert(0, str(Path(__file__).parents[3] / "esm"))
    import esm as esm_lib
    model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    return model.eval().to(device), alphabet


def load_protenix(checkpoint_path: str, device: torch.device):
    sys.path.insert(0, str(Path(__file__).parents[3] / "Protenix"))
    from protenix.model.protenix import Protenix
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(Path(checkpoint_path).parent / "config.yaml")
    model = Protenix(cfg).eval().to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state["model"], strict=False)
    return model


# ──────────────────────────────────────────────────────────────────────────────
# SFM decoding (shared across all modes)
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def decode_from_embedding(
    decoder: SFMDecoder,
    m_seq: torch.Tensor,
    n_seqs: int = 32,
    n_steps: int = 100,
    temperature: float = 1.0,
    device: torch.device = None,
) -> list[str]:
    """
    Sample `n_seqs` protein sequences from a single MSA embedding m_seq.

    Implements Euler integration of the SFM ODE (Eq. 5 in the paper):
        x_{t+Δt} = exp_{x_t}(v_θ(x_t|m, t) · Δt)

    Args:
        decoder:     Trained SFMDecoder.
        m_seq:       (L, msa_dim) MSA embedding.
        n_seqs:      Number of sequences to generate.
        n_steps:     ODE integration steps (paper uses Δt = 0.01, i.e. 100 steps).
        temperature: Decoding temperature (≤1 = sharper, default argmax at 1).
        device:      Torch device.

    Returns:
        seqs: List of amino acid strings (gaps removed for ungapped output).
    """
    if device is None:
        device = next(decoder.parameters()).device

    L, msa_dim = m_seq.shape
    logger.info("decode_from_embedding  ----- n_seqs=%d  n_steps=%d  L=%d", n_seqs, n_steps, L)
    m_seq_batch = m_seq.unsqueeze(0).expand(n_seqs, -1, -1).to(device)  # (n, L, msa_dim)

    # Start from noise on the sphere — float32 for numerical stability in sphere ops.
    # Using m_seq.dtype (bf16 when latent FM output) would degrade ODE integration precision.
    x_t = sample_sphere_noise((n_seqs, L, decoder.vocab_size), device=device, dtype=torch.float32)

    dt = 1.0 / n_steps

    # Integrate from t=0 (noise) to t=1 (data)
    for step in range(n_steps):
        t_val = step * dt
        t = torch.full((n_seqs,), t_val, device=device, dtype=m_seq.dtype)
        v = decoder(x_t, m_seq_batch, t)               # (n, L, V)
        x_t = euler_step_sphere(x_t, v, dt)

    # Decode: x_1 → token ids
    token_ids = decode_sequences(x_t, temperature=temperature)   # (n, L)

    seqs = []
    for row in token_ids:
        seq = "".join(AA_LIST[i] for i in row.tolist())
        seqs.append(seq)
    logger.info("decode_from_embedding complete  ----- generated %d sequences", len(seqs))
    return seqs


# ──────────────────────────────────────────────────────────────────────────────
# Mode 1: Reconstruction from deep MSA
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct(
    msa_seqs: list[str],
    decoder: SFMDecoder,
    protenix_model,
    n_seqs: int = 32,
    n_steps: int = 100,
    device: torch.device = None,
) -> list[str]:
    """
    MSA Compression & Reconstruction.

    Encode the MSA with Protenix → decode with SFM decoder.

    Args:
        msa_seqs:        List of aligned sequences (all same length).
        decoder:         Trained SFMDecoder.
        protenix_model:  Loaded Protenix model.
        n_seqs:          Number of sequences to generate.
        n_steps:         ODE steps.
        device:          Compute device.

    Returns:
        generated sequences (gaps preserved from the MSA alignment length).
    """
    if device is None:
        device = next(decoder.parameters()).device

    logger.info("Reconstruction mode  ----- MSA depth=%d  L=%d", len(msa_seqs), len(msa_seqs[0]))
    m_seq = extract_msa_embedding_protenix(msa_seqs, protenix_model, device)  # (L, 128)
    return decode_from_embedding(decoder, m_seq, n_seqs=n_seqs, n_steps=n_steps, device=device)


# ──────────────────────────────────────────────────────────────────────────────
# Mode 2: Few-shot augmentation (Syn + Rec combined)
# ──────────────────────────────────────────────────────────────────────────────

def augment_shallow(
    shallow_seqs: list[str],
    decoder: SFMDecoder,
    latent_fm: LatentFMEncoder,
    protenix_model,
    esm_model,
    alphabet,
    n_syn_seeds: int = 5,
    n_seqs_per_seed: int = 32,
    n_rec_seqs: int = 64,
    n_diverse: int = 16,
    n_steps: int = 100,
    temperature: float = 0.5,
    device: torch.device = None,
) -> list[str]:
    """
    Few-shot MSA augmentation (Syn+Rec strategy from Section 6.5).

    Two complementary tracks:
      - Syn: generate synthetic embeddings from ESM2 (5 seeds × 32 seqs = 160 synthetic)
      - Rec: decode 64 sequences from the shallow MSA embedding

    Then select the 16 most diverse sequences from the combined pool and
    concatenate with the original shallow MSA.

    Args:
        shallow_seqs:     List of aligned sequences (shallow MSA, can be as few as 1).
        decoder:          Trained SFMDecoder.
        latent_fm:        Trained LatentFMEncoder.
        protenix_model:   Loaded Protenix.
        esm_model:        Loaded ESM2.
        alphabet:         ESM2 Alphabet.
        n_syn_seeds:      Number of synthetic embedding seeds (paper: 5).
        n_seqs_per_seed:  Sequences per synthetic seed (paper: 32).
        n_rec_seqs:       Sequences from reconstruction (paper: 64).
        n_diverse:        Most diverse sequences to keep (paper: 16).
        n_steps:          ODE integration steps.
        temperature:      SDE temperature for latent FM (paper: 0.5).
        device:           Compute device.

    Returns:
        combined: original shallow_seqs + n_diverse diverse generated sequences.
    """
    if device is None:
        device = next(decoder.parameters()).device

    logger.info("augment_shallow  ----- shallow MSA depth=%d  L=%d", len(shallow_seqs), len(shallow_seqs[0]))
    query_seq = shallow_seqs[0].replace("-", "")
    L_aligned = len(shallow_seqs[0])

    # ── ESM2 embedding of query ───────────────────────────────────────────────
    esm_emb = extract_esm_embedding(query_seq, esm_model, alphabet, device)   # (L_q, 1280)
    # Pad/crop to aligned length
    L_q = esm_emb.shape[0]
    if L_q < L_aligned:
        pad = torch.zeros(L_aligned - L_q, esm_emb.shape[1])
        esm_emb = torch.cat([esm_emb, pad], dim=0)
    esm_emb = esm_emb[:L_aligned].unsqueeze(0).to(device)                    # (1, L, 1280)

    generated_seqs = []

    # ── Synthetic track (Syn): latent FM seeds ────────────────────────────────
    logger.info("Syn track  ----- %d seeds x %d seqs/seed", n_syn_seeds, n_seqs_per_seed)
    for _ in range(n_syn_seeds):
        z_syn = sample_msa_embeddings(
            latent_fm, esm_emb, n_steps=n_steps, temperature=temperature
        )  # (1, L, 128)
        syn_seqs = decode_from_embedding(
            decoder, z_syn[0].cpu(), n_seqs=n_seqs_per_seed, n_steps=n_steps,
            temperature=temperature, device=device
        )
        generated_seqs.extend(syn_seqs)

    # ── Reconstruction track (Rec): from shallow MSA embedding ───────────────
    logger.info("Rec track  ----- %d seqs from shallow MSA embedding", n_rec_seqs)
    m_seq = extract_msa_embedding_protenix(shallow_seqs, protenix_model, device)   # (L, 128)
    rec_seqs = decode_from_embedding(
        decoder, m_seq, n_seqs=n_rec_seqs, n_steps=n_steps,
        temperature=temperature, device=device
    )
    generated_seqs.extend(rec_seqs)

    # ── Select n_diverse most diverse sequences ───────────────────────────────
    logger.info("Diversity selection  ----- picking %d from %d candidates", n_diverse, len(generated_seqs))
    diverse = _select_diverse(generated_seqs, n_diverse)

    return list(shallow_seqs) + diverse


# ──────────────────────────────────────────────────────────────────────────────
# Mode 3: Zero-shot augmentation
# ──────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_zeroshot(
    query_seq: str,
    decoder: SFMDecoder,
    latent_fm: LatentFMEncoder,
    esm_model,
    alphabet,
    n_seeds: int = 10,
    n_seqs_per_seed: int = 32,
    n_steps: int = 100,
    temperature: float = 0.5,
    device: torch.device = None,
) -> list[str]:
    """
    Zero-shot MSA augmentation from a single query sequence (Section 4.2).

    Generates n_seeds synthetic MSA embeddings via the latent FM model
    and decodes n_seqs_per_seed sequences from each, returning the best
    scoring seed's sequences (by total diversity).

    Args:
        query_seq:       Amino acid sequence (no gaps).
        decoder:         Trained SFMDecoder.
        latent_fm:       Trained LatentFMEncoder.
        esm_model:       Loaded ESM2 model.
        alphabet:        ESM2 Alphabet.
        n_seeds:         Number of latent FM seeds (paper: 10).
        n_seqs_per_seed: Sequences per seed (paper: 32).
        n_steps:         ODE integration steps.
        temperature:     SDE temperature.
        device:          Compute device.

    Returns:
        best_seqs: n_seqs_per_seed generated sequences for the best seed.
    """
    if device is None:
        device = next(decoder.parameters()).device

    logger.info("generate_zeroshot  ----- query L=%d  n_seeds=%d  n_seqs/seed=%d",
                len(query_seq), n_seeds, n_seqs_per_seed)
    esm_emb = extract_esm_embedding(query_seq, esm_model, alphabet, device)   # (L, 1280)
    esm_emb_batch = esm_emb.unsqueeze(0).to(device)                           # (1, L, 1280)

    best_seqs, best_diversity = None, -1.0

    for seed in range(n_seeds):
        logger.info("Seed %d / %d  -----", seed + 1, n_seeds)
        torch.manual_seed(seed)
        z_syn = sample_msa_embeddings(
            latent_fm, esm_emb_batch, n_steps=n_steps, temperature=temperature
        )  # (1, L, 128)

        seqs = decode_from_embedding(
            decoder, z_syn[0].cpu(), n_seqs=n_seqs_per_seed, n_steps=n_steps,
            temperature=temperature, device=device
        )
        div = _mean_pairwise_diversity(seqs)
        logger.info("Seed %d diversity=%.4f  %s", seed + 1, div,
                     "(new best)" if div > best_diversity else "")
        if div > best_diversity:
            best_diversity = div
            best_seqs = seqs

    logger.info("generate_zeroshot complete  ----- best diversity=%.4f", best_diversity)
    return best_seqs


# ──────────────────────────────────────────────────────────────────────────────
# Diversity helpers
# ──────────────────────────────────────────────────────────────────────────────

def _seq_to_arr(seq: str) -> np.ndarray:
    from msaflow.data.preprocessing import AA_TO_IDX
    return np.array([AA_TO_IDX.get(aa.upper(), 21) for aa in seq], dtype=np.int32)


def _hamming_dist(a: np.ndarray, b: np.ndarray) -> float:
    min_len = min(len(a), len(b))
    return np.mean(a[:min_len] != b[:min_len])


def _mean_pairwise_diversity(seqs: list[str]) -> float:
    arrs = [_seq_to_arr(s) for s in seqs]
    n = len(arrs)
    if n < 2:
        return 0.0
    total = sum(_hamming_dist(arrs[i], arrs[j]) for i in range(n) for j in range(i + 1, n))
    return total / (n * (n - 1) / 2)


def _select_diverse(seqs: list[str], k: int) -> list[str]:
    """Greedy diversity selection: pick k sequences maximising pairwise distance."""
    if len(seqs) <= k:
        return seqs
    arrs = [_seq_to_arr(s) for s in seqs]
    selected = [0]
    for _ in range(k - 1):
        best_idx, best_score = -1, -1.0
        for i in range(len(arrs)):
            if i in selected:
                continue
            score = min(_hamming_dist(arrs[i], arrs[j]) for j in selected)
            if score > best_score:
                best_score = score
                best_idx = i
        if best_idx >= 0:
            selected.append(best_idx)
    return [seqs[i] for i in selected]


# ──────────────────────────────────────────────────────────────────────────────
# FASTA / A3M output helpers
# ──────────────────────────────────────────────────────────────────────────────

def write_fasta(seqs: list[str], path: str, prefix: str = "gen"):
    with open(path, "w") as fh:
        for i, seq in enumerate(seqs):
            fh.write(f">{prefix}_{i}\n{seq}\n")


def write_a3m(query_seq: str, seqs: list[str], path: str, prefix: str = "gen"):
    with open(path, "w") as fh:
        fh.write(f">query\n{query_seq}\n")
        for i, seq in enumerate(seqs):
            fh.write(f">{prefix}_{i}\n{seq}\n")


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )
    parser = argparse.ArgumentParser(description="MSAFlow generation")
    parser.add_argument("--mode",           required=True, choices=["reconstruct", "augment", "zeroshot"])
    parser.add_argument("--query_seq",      default=None,  help="Single query sequence (for zeroshot/augment)")
    parser.add_argument("--input_a3m",      default=None,  help="Input A3M file (for reconstruct/augment)")
    parser.add_argument("--decoder_ckpt",   required=True, help="Path to SFM decoder checkpoint")
    parser.add_argument("--latent_fm_ckpt", default=None,  help="Path to latent FM checkpoint")
    parser.add_argument("--protenix_ckpt",  default=None,  help="Path to Protenix checkpoint")
    parser.add_argument("--output",         required=True, help="Output FASTA/A3M path")
    parser.add_argument("--n_seqs",         type=int, default=32)
    parser.add_argument("--n_steps",        type=int, default=100)
    parser.add_argument("--temperature",    type=float, default=0.5)
    parser.add_argument("--device",         default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    decoder = load_sfm_decoder(args.decoder_ckpt, device)

    generated = []

    if args.mode == "reconstruct":
        assert args.input_a3m and args.protenix_ckpt, "--input_a3m and --protenix_ckpt required"
        protenix = load_protenix(args.protenix_ckpt, device)
        _, seqs = parse_a3m(args.input_a3m)
        seqs = filter_msa(seqs)
        generated = reconstruct(seqs, decoder, protenix, n_seqs=args.n_seqs, device=device)
        write_a3m(seqs[0].replace("-", ""), generated, args.output)

    elif args.mode == "augment":
        assert args.input_a3m and args.latent_fm_ckpt and args.protenix_ckpt
        latent_fm = load_latent_fm(args.latent_fm_ckpt, device)
        protenix = load_protenix(args.protenix_ckpt, device)
        esm_model, alphabet = load_esm2(device)
        _, seqs = parse_a3m(args.input_a3m)
        seqs = filter_msa(seqs)
        generated = augment_shallow(seqs, decoder, latent_fm, protenix, esm_model, alphabet, device=device)
        write_a3m(seqs[0].replace("-", ""), generated, args.output)

    elif args.mode == "zeroshot":
        assert args.query_seq and args.latent_fm_ckpt
        latent_fm = load_latent_fm(args.latent_fm_ckpt, device)
        esm_model, alphabet = load_esm2(device)
        generated = generate_zeroshot(
            args.query_seq, decoder, latent_fm, esm_model, alphabet,
            n_seqs_per_seed=args.n_seqs, n_steps=args.n_steps,
            temperature=args.temperature, device=device,
        )
        # Write as A3M (query header + generated seqs) so Protenix can use it
        write_a3m(args.query_seq, generated, args.output)

    logger.info("Wrote %d sequences to %s", len(generated), args.output)


if __name__ == "__main__":
    main()

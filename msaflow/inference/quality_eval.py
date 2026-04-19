"""
MSAFlow MSA Quality Evaluation — no structure prediction required.

Two modes to isolate which model is responsible for generation quality:

  --mode reconstruct  (SFM Decoder only)
      Uses msa_emb directly from LMDB → SFM Decoder.
      Tests: can the SFM Decoder produce good sequences from ground-truth embeddings?

  --mode zeroshot  (Latent FM + SFM Decoder)
      ESM2 emb (from LMDB) → Latent FM → z_syn → SFM Decoder.
      Tests: does the full zero-shot pipeline produce good sequences?

Diagnosis logic:
  reconstruct=good, zeroshot=bad  →  Latent FM is the weak link
  reconstruct=bad,  zeroshot=bad  →  SFM Decoder itself is the problem

Metrics:
  x_frac          fraction of X (unknown) tokens  [target: < 0.02]
  gap_frac         fraction of gap tokens
  mean_diversity   mean pairwise Hamming distance  [target: 0.3–0.7]
  seq_recovery     mean best-match identity to reference  [random baseline: 0.05]
  aa_kl_div        KL(generated AA dist || reference AA dist)
  emb_cosine       cosine similarity z_syn vs real msa_emb  (zeroshot + msa_emb present)

Usage:
    # Step 1 — test SFM decoder with ground-truth msa_emb:
    python msaflow/inference/quality_eval.py \\
        --lmdb_path /gpfs/.../msaflow_merged.lmdb \\
        --decoder_ckpt  runs/decoder/decoder_ema_final.pt \\
        --latent_fm_ckpt runs/latent_fm/latent_fm_ema_final.pt \\
        --mode reconstruct --n_proteins 20 --verbose

    # Step 2 — test full zero-shot pipeline:
    python msaflow/inference/quality_eval.py ... --mode zeroshot
"""

import argparse
import csv
import logging
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

GAP_IDX    = 20    # AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["-", "X"]
X_IDX      = 21
VOCAB_SIZE = 22


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_neff(tokens: np.ndarray, threshold: float = 0.2) -> float:
    N = tokens.shape[0]
    if N == 0:
        return 0.0
    if N == 1:
        return 1.0
    diff    = (tokens[:, None, :] != tokens[None, :, :])
    hamming = diff.mean(axis=-1)
    counts  = (hamming < threshold).sum(axis=1).astype(np.float32)
    return float((1.0 / counts.clip(min=1.0)).sum())


def gap_fraction(tokens: np.ndarray) -> float:
    return float((tokens == GAP_IDX).mean())


def x_fraction(tokens: np.ndarray) -> float:
    return float((tokens == X_IDX).mean())


def mean_pairwise_diversity(tokens: np.ndarray) -> float:
    N = tokens.shape[0]
    if N < 2:
        return 0.0
    diff    = (tokens[:, None, :] != tokens[None, :, :])
    hamming = diff.mean(axis=-1)
    mask    = np.triu(np.ones((N, N), dtype=bool), k=1)
    return float(hamming[mask].mean())


def seq_recovery(gen_tokens: np.ndarray, ref_tokens: np.ndarray) -> float:
    if ref_tokens.shape[0] == 0 or gen_tokens.shape[0] == 0:
        return float("nan")
    L        = min(gen_tokens.shape[1], ref_tokens.shape[1])
    identity = (gen_tokens[:, None, :L] == ref_tokens[None, :, :L]).mean(axis=-1)
    return float(identity.max(axis=1).mean())


def aa_kl_divergence(gen_tokens: np.ndarray, ref_tokens: np.ndarray,
                      vocab: int = VOCAB_SIZE, eps: float = 1e-8) -> float:
    def marginal(toks):
        counts = np.bincount(toks.flatten(), minlength=vocab).astype(float)
        return (counts + eps) / (counts.sum() + eps * vocab)
    p, q = marginal(gen_tokens), marginal(ref_tokens)
    return float((p * np.log(p / q)).sum())


def embedding_cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-position cosine similarity between two (L, D) arrays."""
    L    = min(a.shape[0], b.shape[0])
    a_n  = a[:L] / (np.linalg.norm(a[:L], axis=-1, keepdims=True) + 1e-8)
    b_n  = b[:L] / (np.linalg.norm(b[:L], axis=-1, keepdims=True) + 1e-8)
    return float((a_n * b_n).sum(axis=-1).mean())


# ─────────────────────────────────────────────────────────────────────────────
# Token helpers
# ─────────────────────────────────────────────────────────────────────────────

def seqs_to_tokens(seqs: list[str]) -> np.ndarray:
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from msaflow.data.preprocessing import AA_TO_IDX
    if not seqs:
        return np.zeros((0, 0), dtype=np.int32)
    L   = max(len(s) for s in seqs)
    arr = np.full((len(seqs), L), X_IDX, dtype=np.int32)
    for i, s in enumerate(seqs):
        for j, aa in enumerate(s):
            arr[i, j] = AA_TO_IDX.get(aa.upper(), X_IDX)
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# LMDB sampling
# ─────────────────────────────────────────────────────────────────────────────

def sample_lmdb_entries(
    lmdb_path: str,
    n: int,
    seed: int = 42,
    require_esm: bool = False,
    require_msa_emb: bool = False,
) -> list[dict]:
    import lmdb as lmdb_lib
    env = lmdb_lib.open(lmdb_path, readonly=True, lock=False,
                        readahead=False, meminit=False, subdir=False)
    with env.begin() as txn:
        all_keys = [k.decode() for k in txn.cursor().iternext(keys=True, values=False)]

    logger.info("LMDB: %d entries total, sampling up to %d ...", len(all_keys), n)
    rng          = random.Random(seed)
    sampled_keys = rng.sample(all_keys, min(n * 5, len(all_keys)))

    entries = []
    with env.begin() as txn:
        for key in sampled_keys:
            if len(entries) >= n:
                break
            raw = txn.get(key.encode())
            if raw is None:
                continue
            entry = pickle.loads(raw)
            if require_esm     and entry.get("esm_emb")  is None:
                continue
            if require_msa_emb and entry.get("msa_emb")  is None:
                continue
            entry["_key"] = key
            entries.append(entry)

    env.close()
    logger.info("Loaded %d entries", len(entries))
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def run_quality_eval(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )

    device         = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir     = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    is_reconstruct = (args.mode == "reconstruct")

    sys.path.insert(0, str(Path(__file__).parents[2]))
    from msaflow.inference.generate import load_sfm_decoder, load_latent_fm
    from msaflow.models.latent_fm import sample_msa_embeddings
    from msaflow.inference.generate import decode_from_embedding

    logger.info("=" * 55)
    logger.info("Mode: %s", args.mode)
    logger.info("Decoder  : %s", args.decoder_ckpt)
    if not is_reconstruct:
        logger.info("Latent FM: %s", args.latent_fm_ckpt)
    logger.info("=" * 55)

    decoder    = load_sfm_decoder(args.decoder_ckpt, device)
    latent_fm  = None
    if not is_reconstruct:
        latent_fm = load_latent_fm(args.latent_fm_ckpt, device)
    logger.info("Models loaded.")

    entries = sample_lmdb_entries(
        lmdb_path=args.lmdb_path,
        n=args.n_proteins,
        seed=args.seed,
        require_esm=    (not is_reconstruct),
        require_msa_emb=is_reconstruct,
    )

    if not entries:
        logger.error("No entries found — check lmdb_path / require_msa_emb setting.")
        return

    results = []

    for idx, entry in enumerate(entries):
        key = entry["_key"]
        L   = entry.get("seq_len", 0)

        logger.info("─" * 55)
        logger.info("[%d/%d] %s  L=%d", idx + 1, len(entries), key, L)

        # ── Reference MSA tokens ───────────────────────────────────────────────
        ref_tokens = entry.get("msa_tokens")
        if ref_tokens is None:
            ref_tokens = np.zeros((0, max(L, 1)), dtype=np.int32)
        else:
            ref_tokens = ref_tokens[:, :L].astype(np.int32)

        ref_neff  = compute_neff(ref_tokens) if ref_tokens.shape[0] > 0 else 0.0
        ref_x     = x_fraction(ref_tokens)   if ref_tokens.shape[0] > 0 else float("nan")
        ref_gap   = gap_fraction(ref_tokens)  if ref_tokens.shape[0] > 0 else float("nan")
        logger.info("  Reference : N=%d  Neff=%.1f  gap=%.3f  x_frac=%.4f",
                    ref_tokens.shape[0], ref_neff, ref_gap, ref_x)

        # ── Get MSA conditioning embedding ─────────────────────────────────────
        emb_cosine = float("nan")

        if is_reconstruct:
            # Mode A: use real msa_emb from LMDB
            msa_emb_np = entry["msa_emb"].astype(np.float32)[:L]  # (L, 128)
            m_seq      = torch.from_numpy(msa_emb_np)              # (L, 128)
            logger.info("  msa_emb   : mean=%.4f  std=%.4f",
                        float(m_seq.mean()), float(m_seq.std()))

        else:
            # Mode B: ESM2 emb → Latent FM → z_syn
            esm_np  = entry["esm_emb"].astype(np.float32)[:L]      # (L, 1280)
            esm_emb = torch.from_numpy(esm_np).unsqueeze(0).to(device)
            with torch.no_grad():
                z_syn = sample_msa_embeddings(
                    latent_fm, esm_emb,
                    n_steps=args.n_steps,
                    temperature=args.temperature,
                )  # (1, L, 128)
            m_seq = z_syn[0].cpu()   # (L, 128)

            # Compare z_syn stats to real msa_emb if available
            msa_emb_ref = entry.get("msa_emb")
            if msa_emb_ref is not None:
                msa_emb_np = msa_emb_ref.astype(np.float32)[:L]
                emb_cosine = embedding_cosine_sim(m_seq.numpy(), msa_emb_np)
                logger.info("  z_syn     : mean=%.4f  std=%.4f  "
                            "| msa_emb: mean=%.4f  std=%.4f  "
                            "| cosine=%.4f",
                            float(m_seq.mean()), float(m_seq.std()),
                            float(msa_emb_np.mean()), float(msa_emb_np.std()),
                            emb_cosine)
            else:
                logger.info("  z_syn     : mean=%.4f  std=%.4f  (no msa_emb in LMDB to compare)",
                            float(m_seq.mean()), float(m_seq.std()))

        # ── Decode sequences ───────────────────────────────────────────────────
        gen_seqs   = decode_from_embedding(
            decoder, m_seq,
            n_seqs=args.n_seqs,
            n_steps=args.n_steps,
            temperature=args.temperature,
            device=device,
        )
        gen_tokens = seqs_to_tokens(gen_seqs)   # (n_seqs, L)

        # ── Metrics ────────────────────────────────────────────────────────────
        gen_neff   = compute_neff(gen_tokens)
        g_frac     = gap_fraction(gen_tokens)
        x_frac_gen = x_fraction(gen_tokens)
        diversity  = mean_pairwise_diversity(gen_tokens)
        recovery   = seq_recovery(gen_tokens, ref_tokens)
        kl         = aa_kl_divergence(gen_tokens, ref_tokens) if ref_tokens.shape[0] > 0 else float("nan")

        logger.info("  Generated : N=%d  Neff=%.1f  gap=%.3f  x_frac=%.4f  "
                    "diversity=%.3f  recovery=%.3f  aa_kl=%.4f",
                    len(gen_seqs), gen_neff, g_frac, x_frac_gen, diversity, recovery, kl)

        if args.verbose:
            for i, s in enumerate(gen_seqs[:3]):
                logger.info("    [%d] %s", i, s[:80] + ("..." if len(s) > 80 else ""))

        results.append({
            "key":            key,
            "mode":           args.mode,
            "seq_len":        L,
            "ref_n_seqs":     ref_tokens.shape[0],
            "ref_neff":       ref_neff,
            "ref_x_frac":     ref_x,
            "gen_neff":       gen_neff,
            "gap_frac":       g_frac,
            "x_frac":         x_frac_gen,
            "mean_diversity": diversity,
            "seq_recovery":   recovery,
            "aa_kl_div":      kl,
            "emb_cosine":     emb_cosine,
        })

    # ── CSV ────────────────────────────────────────────────────────────────────
    csv_path = output_dir / f"quality_eval_{args.mode}.csv"
    fields   = ["key", "mode", "seq_len", "ref_n_seqs", "ref_neff", "ref_x_frac",
                "gen_neff", "gap_frac", "x_frac", "mean_diversity",
                "seq_recovery", "aa_kl_div", "emb_cosine"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results → %s", csv_path)

    # ── Aggregate summary ──────────────────────────────────────────────────────
    def _stats(key_name):
        vals = [r[key_name] for r in results if not np.isnan(r[key_name])]
        if not vals:
            return "N/A"
        return f"mean={np.mean(vals):.4f}  median={np.median(vals):.4f}  std={np.std(vals):.4f}"

    print("\n" + "=" * 70)
    print(f"MSA Quality Eval  mode={args.mode}  n={len(results)}")
    print("=" * 70)
    print(f"  X token frac    {_stats('x_frac')}  (ref: {_stats('ref_x_frac')})")
    print(f"  Gap fraction    {_stats('gap_frac')}")
    print(f"  Diversity       {_stats('mean_diversity')}  (target: 0.3–0.7)")
    print(f"  Seq recovery    {_stats('seq_recovery')}  (random: ~0.05)")
    print(f"  AA KL div       {_stats('aa_kl_div')}")
    if not is_reconstruct:
        print(f"  Emb cosine      {_stats('emb_cosine')}  (z_syn vs msa_emb)")
    print("─" * 70)

    # Diagnosis
    x_mean   = np.nanmean([r["x_frac"]          for r in results])
    div_mean = np.nanmean([r["mean_diversity"]   for r in results])
    rec_mean = np.nanmean([r["seq_recovery"]     for r in results])

    issues = []
    if x_mean > 0.05:
        issues.append(f"X token frac {x_mean:.3f} > 0.05 — decoder uncertain / undertrained")
    if div_mean > 0.80:
        issues.append(f"Diversity {div_mean:.3f} > 0.80 — conditioning not being used (target 0.3–0.7)")
    if rec_mean < 0.10:
        issues.append(f"Seq recovery {rec_mean:.3f} ≈ random — not generating valid homologs")

    if issues:
        print("Diagnosis:")
        for issue in issues:
            print(f"  ⚠  {issue}")
    else:
        print("  ✓  Metrics look healthy")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MSAFlow MSA quality evaluation")
    parser.add_argument("--lmdb_path",       required=True)
    parser.add_argument("--decoder_ckpt",    required=True)
    parser.add_argument("--latent_fm_ckpt",  required=True)
    parser.add_argument("--mode",            default="reconstruct",
                        choices=["reconstruct", "zeroshot"],
                        help="reconstruct=real msa_emb→decoder  zeroshot=latentFM→decoder")
    parser.add_argument("--output_dir",      default="runs/quality_eval")
    parser.add_argument("--n_proteins",      type=int, default=20)
    parser.add_argument("--n_seqs",          type=int, default=32)
    parser.add_argument("--n_steps",         type=int, default=100)
    parser.add_argument("--temperature",     type=float, default=0.0)
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--seed",            type=int, default=42)
    parser.add_argument("--verbose",         action="store_true")
    args = parser.parse_args()

    run_quality_eval(args)


if __name__ == "__main__":
    main()

"""
Compute Neff for each FoldBench target's ColabFold reference MSA and
classify into three groups based on the training data Neff thresholds.

Uses the same sequence-weight formula as preprocessing.py:
    w_i = 1 / |{j : hamming(i, j) < theta}|   (theta=0.2)
    Neff = sum(w)

Groups (matching training LMDB Neff distribution breakpoints):
  orphan  : Neff ≤  10.0   → compare zero-shot vs no-MSA
  shallow : 10.0 < Neff ≤ 67.1  → few-shot vs ColabFold(shallow)
  full    : Neff >  67.1   → few-shot vs ColabFold(full)

Outputs (written to --output_dir):
  neff_scores.csv            — all targets: name, neff, depth, group
  foldbench_orphan.fasta     — filtered FASTA
  foldbench_shallow.fasta
  foldbench_full.fasta
  neff_values.npy            — raw Neff array (same format as neff_distribution.py)

Usage:
    python scripts/compute_foldbench_neff.py \\
        --fasta        data/foldbench_monomer.fasta \\
        --ref_msa_dir  /store/deepfold3/results/msa/a3m \\
        --output_dir   data/foldbench_groups \\
        [--theta 0.2]  [--max_seqs 2048]
"""

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import numpy as np


# ── FASTA parsing ─────────────────────────────────────────────────────────────

def parse_fasta(path: str) -> list[tuple[str, str]]:
    proteins, name, buf = [], None, []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if name:
                    proteins.append((name, "".join(buf)))
                name = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line)
    if name:
        proteins.append((name, "".join(buf)))
    return proteins


# ── A3M parsing ───────────────────────────────────────────────────────────────

def parse_a3m(path: str) -> list[str]:
    """Return aligned sequences with lowercase insertions stripped."""
    seqs, buf = [], []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                buf = []
            elif line:
                buf.append(re.sub(r"[a-z]", "", line))
    if buf:
        seqs.append("".join(buf))
    return seqs


# ── Ref MSA lookup (handles ColabFold assembly1__chain naming) ────────────────

def find_ref_a3m(ref_dir: Path, prot_name: str) -> Path | None:
    pdb_id = prot_name.split("-")[0]
    for name in [prot_name, pdb_id]:
        p = ref_dir / f"{name}.a3m"
        if p.exists():
            return p
        p = ref_dir / f"{name}-assembly1__A.a3m"
        if p.exists():
            return p
        hits = sorted(ref_dir.glob(f"{name}*__A.a3m"))
        if hits:
            return hits[0]
        hits = sorted(ref_dir.glob(f"{name}*.a3m"))
        if hits:
            return hits[0]
    return None


# ── Neff computation (same formula as preprocessing.py) ──────────────────────

def compute_neff(seqs: list[str], theta: float = 0.2, max_seqs: int = 2048) -> float:
    """
    w_i = 1 / |{j : hamming(i, j) < theta}|,  Neff = sum(w).

    Randomly subsamples to max_seqs if the MSA is deeper (maintains query).
    """
    if not seqs:
        return 0.0
    if len(seqs) == 1:
        return 1.0

    if len(seqs) > max_seqs:
        rng = np.random.default_rng(42)
        idx = [0] + (rng.choice(len(seqs) - 1, max_seqs - 1, replace=False) + 1).tolist()
        seqs = [seqs[i] for i in sorted(idx)]

    L = len(seqs[0])
    tokens = np.frombuffer(
        b"".join(s[:L].upper().encode("ascii") for s in seqs), dtype=np.uint8
    ).reshape(len(seqs), L)

    # chunk to avoid OOM for large N
    N = len(tokens)
    chunk = 512
    counts = np.zeros(N, dtype=np.float32)
    for i in range(0, N, chunk):
        a = tokens[i : i + chunk]                              # (c, L)
        hamming = (tokens[:, None, :] != a[None, :, :]).mean(-1)  # (N, c)
        counts[i : i + chunk] = (hamming < theta).sum(0).astype(np.float32)
    return float((1.0 / counts.clip(min=1.0)).sum())


# ── Histogram (same style as neff_distribution.py) ───────────────────────────

def ascii_histogram(values: np.ndarray, n_bins: int = 30, bar_width: int = 50) -> str:
    log_vals = np.log10(values.clip(min=1e-3))
    lo, hi = log_vals.min(), log_vals.max()
    bins = np.linspace(lo, hi, n_bins + 1)
    counts, _ = np.histogram(log_vals, bins=bins)
    max_count = counts.max() if counts.max() > 0 else 1
    lines = [
        "  x-axis: log10(Neff)  [actual Neff shown in parentheses]",
        f"  {'─' * (bar_width + 36)}",
    ]
    for i in range(n_bins):
        log_lo, log_hi = bins[i], bins[i + 1]
        bar = "█" * int(bar_width * counts[i] / max_count)
        lines.append(
            f"  [{log_lo:5.2f} – {log_hi:5.2f})  {bar:<{bar_width}}  {counts[i]:>5}"
            f"  (Neff {10**log_lo:6.1f}–{10**log_hi:.1f})"
        )
    lines.append(f"  {'─' * (bar_width + 36)}")
    return "\n".join(lines)


# ── Group thresholds ──────────────────────────────────────────────────────────

THRESHOLDS = [
    ("orphan",  0.0,  10.0),
    ("shallow", 10.0, 67.1),
    ("full",    67.1, float("inf")),
]

def classify(neff: float) -> str:
    for name, lo, hi in THRESHOLDS:
        if lo < neff <= hi or (lo == 0.0 and neff <= hi):
            return name
    return "full"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta",       required=True, help="FoldBench FASTA")
    parser.add_argument("--ref_msa_dir", required=True, help="ColabFold A3M directory")
    parser.add_argument("--output_dir",  required=True, help="Output directory")
    parser.add_argument("--theta",       type=float, default=0.2,
                        help="Hamming threshold for sequence clustering (default 0.2)")
    parser.add_argument("--max_seqs",    type=int, default=2048,
                        help="Max sequences used for Neff computation (default 2048)")
    args = parser.parse_args()

    ref_dir = Path(args.ref_msa_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    proteins = parse_fasta(args.fasta)
    print(f"FoldBench proteins: {len(proteins)}")
    print(f"theta={args.theta}  max_seqs={args.max_seqs}\n")

    rows = []
    groups: dict[str, list[tuple[str, str]]] = {g: [] for g, *_ in THRESHOLDS}
    neff_vals = []

    t0 = time.time()
    for idx, (prot_name, query_seq) in enumerate(proteins):
        ref_path = find_ref_a3m(ref_dir, prot_name)
        if ref_path is None:
            print(f"  [{idx+1:>3}/{len(proteins)}] {prot_name:<32}  MISSING ref A3M")
            rows.append({"name": prot_name, "neff": "nan", "depth": 0, "group": "missing"})
            continue

        seqs = parse_a3m(str(ref_path))
        depth = len(seqs) - 1
        neff = compute_neff(seqs, theta=args.theta, max_seqs=args.max_seqs)
        group = classify(neff)
        neff_vals.append(neff)

        print(f"  [{idx+1:>3}/{len(proteins)}] {prot_name:<32}  depth={depth:>5}  "
              f"Neff={neff:>8.2f}  → {group}")
        rows.append({"name": prot_name, "neff": f"{neff:.4f}", "depth": depth, "group": group})
        groups[group].append((prot_name, query_seq))

    elapsed = time.time() - t0

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = out_dir / "neff_scores.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["name", "neff", "depth", "group"])
        writer.writeheader()
        writer.writerows(rows)

    # ── Per-group FASTAs ──────────────────────────────────────────────────────
    for group, *_ in THRESHOLDS:
        fasta_path = out_dir / f"foldbench_{group}.fasta"
        with open(fasta_path, "w") as fh:
            for name, seq in groups[group]:
                fh.write(f">{name}\n{seq}\n")

    # ── Statistics + histogram ────────────────────────────────────────────────
    neff_arr = np.array(neff_vals, dtype=np.float32)
    np.save(out_dir / "neff_values.npy", neff_arr)

    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    pct_vals = np.percentile(neff_arr, pcts) if len(neff_arr) else [float("nan")] * len(pcts)

    print(f"\n{'='*72}")
    print(f"  FoldBench Neff Distribution  —  {args.ref_msa_dir}")
    print(f"{'='*72}")
    print(f"  Targets computed : {len(neff_arr)}  (elapsed {elapsed:.1f}s)")
    print(f"  Mean   : {neff_arr.mean():>10.2f}")
    print(f"  Std    : {neff_arr.std():>10.2f}")
    print(f"  Min    : {neff_arr.min():>10.2f}")
    print(f"  Max    : {neff_arr.max():>10.2f}")
    print()
    print("  Percentiles:")
    for p, v in zip(pcts, pct_vals):
        print(f"    p{p:02d}  = {v:>10.2f}")
    print()
    print("  Log10(Neff) histogram:")
    print(ascii_histogram(neff_arr))

    print(f"\n  Group summary  (theta={args.theta}):")
    total = len(neff_arr)
    for group, lo, hi in THRESHOLDS:
        hi_str = f"{hi:.1f}" if hi != float("inf") else "∞"
        n = len(groups[group])
        print(f"    {group:<8} (Neff {lo:.1f}–{hi_str:>6}): {n:>3} / {total}")

    print(f"\n  Outputs → {out_dir}/")
    print(f"    neff_scores.csv, neff_values.npy")
    for group, *_ in THRESHOLDS:
        print(f"    foldbench_{group}.fasta  ({len(groups[group])} proteins)")
    print(f"{'='*72}")


if __name__ == "__main__":
    main()

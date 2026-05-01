"""
Compare generated MSAs vs ColabFold reference MSAs for FoldBench targets.

Metrics (per protein):
  recovery_gen  — mean identity of generated seqs to query
  recovery_ref  — mean identity of reference seqs to query
  diversity_gen — mean pairwise identity within generated set (lower = more diverse)
  diversity_ref — mean pairwise identity within reference set
  cross_sim     — mean identity between generated and reference sets
  x_frac_gen    — fraction of X tokens in generated seqs

Usage:
    # 1. Extract reference MSAs (once)
    mkdir -p /tmp/foldbench_msas
    zstd -d /store/deepfold3/results/msa/foldbench.tar.zst -o /tmp/foldbench.tar
    tar -xf /tmp/foldbench.tar -C /tmp/foldbench_msas

    # 2. Run comparison
    python scripts/compare_msa_quality.py \\
        --gen_msa_dir  runs/fold_benchmark_yjlee4_latest/msas \\
        --ref_msa_dir  /tmp/foldbench_msas \\
        --targets 8ikx 8ptd 9fym 9fce 8g11 8fjf 8hxs 8hel 8ey3 9f9u \\
        --n_seqs 32
"""

import argparse
import re
import random
import statistics
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# A3M parsing
# ──────────────────────────────────────────────────────────────────────────────

def parse_a3m(path: str) -> list[str]:
    """Read sequences from an A3M file. Lowercase insertions are stripped."""
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


def sample_seqs(seqs: list[str], n: int, rng: random.Random) -> list[str]:
    """Sample up to n homolog sequences (skipping query at index 0)."""
    homologs = seqs[1:]
    if len(homologs) <= n:
        return homologs
    return rng.sample(homologs, n)


# ──────────────────────────────────────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────────────────────────────────────

def seq_identity(seq1: str, seq2: str) -> float:
    """Identity anchored on seq1 (non-gap positions of seq1)."""
    matches = total = 0
    for a, b in zip(seq1, seq2):
        if a in "-. ":
            continue
        total += 1
        if a.upper() == b.upper():
            matches += 1
    return matches / total if total else 0.0


def mean_identity_to_query(query: str, seqs: list[str]) -> float:
    if not seqs:
        return float("nan")
    return statistics.mean(seq_identity(query, s) for s in seqs)


def mean_pairwise_identity(seqs: list[str]) -> float:
    if len(seqs) < 2:
        return float("nan")
    total = n = 0.0
    for i in range(len(seqs)):
        for j in range(i + 1, len(seqs)):
            total += seq_identity(seqs[i], seqs[j])
            n += 1
    return total / n if n else float("nan")


def mean_cross_identity(seqs_a: list[str], seqs_b: list[str]) -> float:
    if not seqs_a or not seqs_b:
        return float("nan")
    total = n = 0.0
    for a in seqs_a:
        for b in seqs_b:
            total += seq_identity(a, b)
            n += 1
    return total / n if n else float("nan")


def x_fraction(seqs: list[str]) -> float:
    """Fraction of non-gap residues that are X."""
    x_count = total = 0
    for s in seqs:
        for c in s:
            if c in "-. ":
                continue
            total += 1
            if c.upper() == "X":
                x_count += 1
    return x_count / total if total else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# File lookup
# ──────────────────────────────────────────────────────────────────────────────

def find_gen_a3m(gen_dir: Path, target: str) -> str | None:
    """Generated: <gen_dir>/<target>/seed_<n>.a3m  — pick first available seed."""
    prot_dir = gen_dir / target
    for seed in range(10):
        p = prot_dir / f"seed_{seed}.a3m"
        if p.exists():
            return str(p)
    matches = sorted(prot_dir.glob("*.a3m")) if prot_dir.exists() else []
    return str(matches[0]) if matches else None


def find_ref_a3m(ref_dir: Path, target: str) -> str | None:
    """Reference MSA lookup.

    Handles naming conventions:
      - <target>.a3m
      - <pdbid>-assembly1__A.a3m  (ColabFold/FoldBench style)
    Prefers chain A; falls back to first alphabetical match.
    """
    pdb_id = target.split("-")[0]
    for name in [target, pdb_id]:
        p = ref_dir / f"{name}.a3m"
        if p.exists():
            return str(p)
    # ColabFold style: 8hel-assembly1__A.a3m — prefer chain A
    for name in [target, pdb_id]:
        chain_a = ref_dir / f"{name}-assembly1__A.a3m"
        if chain_a.exists():
            return str(chain_a)
        matches = sorted(ref_dir.glob(f"{name}*__A.a3m"))
        if matches:
            return str(matches[0])
        matches = sorted(ref_dir.glob(f"{name}*.a3m"))
        if matches:
            return str(matches[0])
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_msa_dir", required=True)
    parser.add_argument("--ref_msa_dir", required=True)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--n_seqs", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    gen_dir = Path(args.gen_msa_dir)
    ref_dir = Path(args.ref_msa_dir)

    cols = f"{'Target':<22} {'rec_gen':>8} {'rec_ref':>8} {'div_gen':>8} {'div_ref':>8} {'cross':>8} {'x_frac':>8} {'ngen':>5} {'nref':>5}"
    print(cols)
    print("-" * len(cols))

    agg = {k: [] for k in ["rec_gen", "rec_ref", "div_gen", "div_ref", "cross", "x_frac"]}

    for target in args.targets:
        gen_path = find_gen_a3m(gen_dir, target)
        ref_path = find_ref_a3m(ref_dir, target)

        if gen_path is None:
            print(f"{target:<22}  GEN A3M NOT FOUND"); continue
        if ref_path is None:
            print(f"{target:<22}  REF A3M NOT FOUND"); continue

        gen_seqs = parse_a3m(gen_path)
        ref_seqs = parse_a3m(ref_path)
        if not gen_seqs:
            print(f"{target:<22}  GEN EMPTY"); continue

        query       = gen_seqs[0]
        gen_samp    = sample_seqs(gen_seqs, args.n_seqs, rng)
        ref_samp    = sample_seqs(ref_seqs, args.n_seqs, rng)

        rec_gen  = mean_identity_to_query(query, gen_samp)
        rec_ref  = mean_identity_to_query(query, ref_samp)
        div_gen  = mean_pairwise_identity(gen_samp)
        div_ref  = mean_pairwise_identity(ref_samp)
        cross    = mean_cross_identity(gen_samp, ref_samp)
        x_frac   = x_fraction(gen_samp)

        print(f"{target:<22} {rec_gen:>8.4f} {rec_ref:>8.4f} {div_gen:>8.4f} {div_ref:>8.4f} {cross:>8.4f} {x_frac:>8.4f} {len(gen_samp):>5} {len(ref_samp):>5}")

        for k, v in [("rec_gen", rec_gen), ("rec_ref", rec_ref), ("div_gen", div_gen),
                     ("div_ref", div_ref), ("cross", cross), ("x_frac", x_frac)]:
            if v == v:
                agg[k].append(v)

    if agg["rec_gen"]:
        n = len(agg["rec_gen"])
        print("-" * len(cols))
        print(f"\n{'Mean over ' + str(n) + ' targets':<22}", end="")
        for k in ["rec_gen", "rec_ref", "div_gen", "div_ref", "cross", "x_frac"]:
            vals = agg[k]
            print(f" {statistics.mean(vals):>8.4f}", end="")
        print()

    print("""
Columns:
  rec_gen : mean identity of generated seqs to query
  rec_ref : mean identity of reference seqs to query
  div_gen : mean pairwise identity within generated set (lower = more diverse)
  div_ref : mean pairwise identity within reference set
  cross   : mean identity between generated and reference sets
  x_frac  : fraction of X tokens in generated seqs
""")


if __name__ == "__main__":
    main()

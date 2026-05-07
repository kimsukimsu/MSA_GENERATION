"""
Pre-trim A3M files to match FoldBench FASTA sequences.

For each protein in the FASTA:
  - If A3M query length == FASTA length  → copy as-is
  - If FASTA is exact substring          → trim (exact match)
  - If best sliding window identity ≥ threshold → trim (fuzzy match)
  - Otherwise                            → skip (will fold without MSA)

Output directory mirrors the original naming convention so
fold_benchmark.py's _find_msa_file() can locate files unchanged.

Usage:
  python scripts/pretrim_a3m.py \
      --fasta     runs/benchmark_all/foldbench_shallow_full.fasta \
      --msa_dir   /store/deepfold3/results/msa/a3m \
      --out_dir   runs/benchmark_all/trimmed_a3m \
      [--max_mismatch_frac 0.05]
"""
import argparse
import re
import shutil
from pathlib import Path


def parse_fasta(path):
    results, name, buf = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    results.append((name, "".join(buf)))
                name = line[1:].split()[0]
                buf = []
            elif line:
                buf.append(line.upper())
    if name is not None:
        results.append((name, "".join(buf)))
    return results


def find_a3m(msa_dir: Path, prot_name: str):
    pdb_id = prot_name.split("-")[0]
    for name in [prot_name, pdb_id]:
        for pattern in [f"{name}.a3m", f"{name}-assembly1__A.a3m"]:
            p = msa_dir / pattern
            if p.exists():
                return p
        for glob in [f"{name}*__A.a3m", f"{name}*.a3m"]:
            hits = sorted(msa_dir.glob(glob))
            if hits:
                return hits[0]
    return None


def parse_a3m_records(content: str):
    records, header, seq_parts = [], None, []
    for line in content.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(seq_parts)))
            header, seq_parts = line, []
        else:
            seq_parts.append(line)
    if header is not None:
        records.append((header, "".join(seq_parts)))
    return records


def _best_window(a3m_upper: str, fasta_seq: str, max_mismatch_frac: float):
    """Slide a window of len(fasta_seq) over a3m_upper; return (start, n_mismatches)."""
    n = len(fasta_seq)
    if len(a3m_upper) < n:
        return -1, n
    max_mm = max(1, int(n * max_mismatch_frac))
    best_start, best_mm = -1, n + 1
    for i in range(len(a3m_upper) - n + 1):
        mm = sum(a != b for a, b in zip(a3m_upper[i:i + n], fasta_seq))
        if mm < best_mm:
            best_mm, best_start = mm, i
        if mm == 0:
            break  # exact match found, no need to continue
    if best_mm <= max_mm:
        return best_start, best_mm
    return -1, best_mm


def trim_a3m(content: str, query_seq: str, max_mismatch_frac: float = 0.05):
    """
    Return trimmed A3M string, or None if no suitable window found.

    Tries exact substring first; falls back to best sliding-window match
    within max_mismatch_frac tolerance.
    """
    records = parse_a3m_records(content)
    if not records:
        return None

    a3m_upper = re.sub(r"[a-z]", "", records[0][1])
    n = len(query_seq)

    # Exact match first
    start = a3m_upper.find(query_seq)
    n_mm = 0
    if start == -1:
        start, n_mm = _best_window(a3m_upper, query_seq, max_mismatch_frac)
    if start == -1:
        return None
    end = start + n

    def _extract(seq: str) -> str:
        out, col, in_range = [], 0, False
        for ch in seq:
            if ch.isupper() or ch == "-":
                in_range = start <= col < end
                if in_range:
                    out.append(ch)
                col += 1
            else:
                if in_range:
                    out.append(ch)
        return "".join(out)

    lines = []
    for hdr, seq in records:
        lines.append(hdr)
        lines.append(_extract(seq))
    return "\n".join(lines) + "\n", n_mm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta",             required=True)
    parser.add_argument("--msa_dir",           required=True)
    parser.add_argument("--out_dir",           required=True)
    parser.add_argument("--max_mismatch_frac", type=float, default=0.05,
                        help="Max fraction of mismatches allowed in fuzzy trim (default 0.05 = 5%%)")
    args = parser.parse_args()

    proteins = parse_fasta(args.fasta)
    msa_dir  = Path(args.msa_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_exact, n_trimmed_exact, n_trimmed_fuzzy, n_failed, n_missing = 0, 0, 0, 0, 0

    for name, seq in proteins:
        a3m = find_a3m(msa_dir, name)
        if a3m is None:
            print(f"[MISSING] {name}")
            n_missing += 1
            continue

        content = a3m.read_text()
        a3m_upper = re.sub(r"[a-z]", "", parse_a3m_records(content)[0][1])
        out_path = out_dir / a3m.name

        if len(a3m_upper) == len(seq):
            shutil.copy2(a3m, out_path)
            n_exact += 1
            continue

        result = trim_a3m(content, seq, args.max_mismatch_frac)
        if result is not None:
            trimmed, n_mm = result
            out_path.write_text(trimmed)
            if n_mm == 0:
                n_trimmed_exact += 1
            else:
                pct = 100 * n_mm / len(seq)
                print(f"[FUZZY  ] {name}  {len(a3m_upper)}→{len(seq)} aa  mismatches={n_mm} ({pct:.1f}%)")
                n_trimmed_fuzzy += 1
        else:
            a3m_len = len(a3m_upper)
            max_mm = max(1, int(len(seq) * args.max_mismatch_frac))
            print(f"[FAILED ] {name}  {a3m_len}→{len(seq)} aa  best window exceeds {max_mm} mismatches — no MSA")
            n_failed += 1

    print(f"\n{'='*55}")
    print(f"Total           : {len(proteins)}")
    print(f"Exact (no trim) : {n_exact}")
    print(f"Trimmed (exact) : {n_trimmed_exact}")
    print(f"Trimmed (fuzzy) : {n_trimmed_fuzzy}")
    print(f"Failed (no MSA) : {n_failed}")
    print(f"Missing A3M     : {n_missing}")
    print(f"Output          : {out_dir}")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    main()

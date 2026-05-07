"""
Pre-trim A3M files to match FoldBench FASTA sequences.

For each protein in the FASTA:
  - If A3M query length == FASTA length  → copy as-is
  - If FASTA is a substring of A3M query → trim and save
  - Otherwise                            → skip (will fold without MSA)

Output directory mirrors the original naming convention so
fold_benchmark.py's _find_msa_file() can locate files unchanged.

Usage:
  python scripts/pretrim_a3m.py \
      --fasta    runs/benchmark_all/foldbench_shallow_full.fasta \
      --msa_dir  /store/deepfold3/results/msa/a3m \
      --out_dir  runs/benchmark_all/trimmed_a3m
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
        for pattern in [
            f"{name}.a3m",
            f"{name}-assembly1__A.a3m",
        ]:
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


def trim_a3m(content: str, query_seq: str):
    """Return trimmed A3M string, or None if FASTA not a substring."""
    records = parse_a3m_records(content)
    if not records:
        return None

    a3m_query_upper = re.sub(r"[a-z]", "", records[0][1])
    start = a3m_query_upper.find(query_seq)
    if start == -1:
        return None
    end = start + len(query_seq)

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
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta",   required=True)
    parser.add_argument("--msa_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    proteins = parse_fasta(args.fasta)
    msa_dir  = Path(args.msa_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_exact, n_trimmed, n_failed, n_missing = 0, 0, 0, 0

    for name, seq in proteins:
        a3m = find_a3m(msa_dir, name)
        if a3m is None:
            print(f"[MISSING ] {name}")
            n_missing += 1
            continue

        content = a3m.read_text()
        a3m_query_upper = re.sub(r"[a-z]", "", parse_a3m_records(content)[0][1])
        out_path = out_dir / a3m.name

        if len(a3m_query_upper) == len(seq):
            shutil.copy2(a3m, out_path)
            n_exact += 1
        else:
            trimmed = trim_a3m(content, seq)
            if trimmed is not None:
                out_path.write_text(trimmed)
                print(f"[TRIMMED ] {name}  {len(a3m_query_upper)}→{len(seq)} aa")
                n_trimmed += 1
            else:
                print(f"[NO_TRIM ] {name}  {len(a3m_query_upper)} aa (A3M) vs {len(seq)} aa (FASTA) — no MSA")
                n_failed += 1

    print(f"\n{'='*50}")
    print(f"Total    : {len(proteins)}")
    print(f"Exact    : {n_exact}")
    print(f"Trimmed  : {n_trimmed}")
    print(f"No trim  : {n_failed}  ← will fold without MSA")
    print(f"Missing  : {n_missing}")
    print(f"Output   : {out_dir}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()

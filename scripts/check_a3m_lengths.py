"""
Pre-check: compare A3M query lengths vs FASTA sequence lengths.

Usage:
  python scripts/check_a3m_lengths.py \
      --fasta  runs/benchmark_all/foldbench_shallow_full.fasta \
      --msa_dir /store/deepfold3/results/msa/a3m
"""
import argparse
import re
from pathlib import Path


def parse_fasta(path):
    results = []
    name, buf = None, []
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
        p = msa_dir / f"{name}.a3m"
        if p.exists():
            return p
        p = msa_dir / f"{name}-assembly1__A.a3m"
        if p.exists():
            return p
        hits = sorted(msa_dir.glob(f"{name}*__A.a3m"))
        if hits:
            return hits[0]
        hits = sorted(msa_dir.glob(f"{name}*.a3m"))
        if hits:
            return hits[0]
    return None


def a3m_query_upper(a3m_path: Path) -> str:
    parts, in_seq = [], False
    try:
        with open(a3m_path) as fh:
            for line in fh:
                line = line.rstrip()
                if line.startswith(">"):
                    if in_seq:
                        break
                    in_seq = True
                elif in_seq:
                    parts.append(re.sub(r"[a-z]", "", line))
    except Exception:
        pass
    return "".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fasta",   required=True)
    parser.add_argument("--msa_dir", required=True)
    args = parser.parse_args()

    proteins = parse_fasta(args.fasta)
    msa_dir  = Path(args.msa_dir)

    no_a3m    = []
    ok        = []
    trimmable = []
    no_trim   = []

    for name, seq in proteins:
        a3m = find_a3m(msa_dir, name)
        if a3m is None:
            no_a3m.append(name)
            continue
        query = a3m_query_upper(a3m)
        if len(query) == len(seq):
            ok.append(name)
        elif seq in query:
            trimmable.append((name, len(seq), len(query), a3m.name))
        else:
            no_trim.append((name, len(seq), len(query), a3m.name))

    total = len(proteins)
    print(f"\nTotal proteins   : {total}")
    print(f"A3M OK (exact)   : {len(ok)}")
    print(f"Trimmable        : {len(trimmable)}")
    print(f"Not trimmable    : {len(no_trim)}  ← will fold without MSA")
    print(f"A3M not found    : {len(no_a3m)}")

    if no_trim:
        print("\n── Not trimmable (FASTA not substring of A3M query) ──")
        print(f"  {'protein':<30} {'seq':>6} {'a3m':>6}  file")
        for name, slen, ql, fname in no_trim:
            print(f"  {name:<30} {slen:>6} {ql:>6}  {fname}")

    if no_a3m:
        print("\n── No A3M file ──")
        for n in no_a3m:
            print(f"  {n}")

    print()


if __name__ == "__main__":
    main()

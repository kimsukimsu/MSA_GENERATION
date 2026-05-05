"""
Grouped comparison of fold benchmark results across 4 modes.

Reads:
  --base_dir  DIR   base directory containing nomsa/ colabfold/ zeroshot/ fewshot/
                    subdirs, each with benchmark_results.csv
  --neff_csv  CSV   neff_scores.csv from compute_foldbench_neff.py
                    columns: name, neff, depth, group

Prints a table:  group × mode → mean TM-score, mean pLDDT, count
"""

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

MODES   = ["nomsa", "colabfold", "zeroshot", "fewshot"]
GROUPS  = ["orphan", "shallow", "full", "all"]
SEP     = "─"


def load_neff_groups(csv_path: Path) -> dict[str, str]:
    """Return {protein_name: group} from neff_scores.csv."""
    groups = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            groups[row["name"]] = row["group"]   # orphan / shallow / full / missing
    return groups


def load_results(csv_path: Path) -> list[dict]:
    """Return list of rows from benchmark_results.csv; skip empty TM-scores."""
    rows = []
    if not csv_path.exists():
        return rows
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def safe_float(v) -> float | None:
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def stats(values: list[float]) -> tuple[float, float, int]:
    """Return (mean, std, n)."""
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(values) / n
    var  = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0.0
    return mean, math.sqrt(var), n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--neff_csv", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    neff_groups = load_neff_groups(Path(args.neff_csv))

    # ── Load all results ──────────────────────────────────────────────────────
    # data[mode][group] = {tm: [], plddt: []}
    data: dict[str, dict[str, dict[str, list[float]]]] = {
        mode: {g: {"tm": [], "plddt": []} for g in GROUPS}
        for mode in MODES
    }

    for mode in MODES:
        csv_path = base_dir / mode / "benchmark_results.csv"
        rows = load_results(csv_path)
        if not rows:
            # try shard CSVs if benchmark_results.csv not yet merged
            import glob
            shard_files = sorted(glob.glob(str(base_dir / mode / "shard_*.csv")))
            for sf in shard_files:
                rows.extend(load_results(Path(sf)))

        for row in rows:
            name  = row.get("name", row.get("protein", ""))
            group = neff_groups.get(name, "missing")
            if group == "missing":
                continue

            tm    = safe_float(row.get("tm_score", ""))
            plddt = safe_float(row.get("plddt",    ""))

            for g in [group, "all"]:
                if tm    is not None:
                    data[mode][g]["tm"].append(tm)
                if plddt is not None:
                    data[mode][g]["plddt"].append(plddt)

    # ── Print table ───────────────────────────────────────────────────────────
    col_w = 22
    hdr_w = 10

    def fmt(mean, std, n):
        if n == 0:
            return "  n/a".ljust(col_w)
        return f"  {mean:.4f} ± {std:.4f}  (n={n})".ljust(col_w)

    # TM-score table
    print()
    print("=" * 80)
    print("  TM-score comparison  (mean ± std)")
    print("=" * 80)
    header = f"  {'Group':<{hdr_w}}" + "".join(f"  {m:<{col_w-2}}" for m in MODES)
    print(header)
    print("  " + SEP * (hdr_w + len(MODES) * col_w))
    for g in GROUPS:
        row_str = f"  {g:<{hdr_w}}"
        for mode in MODES:
            tm_vals = data[mode][g]["tm"]
            m, s, n = stats(tm_vals)
            row_str += fmt(m, s, n)
        print(row_str)
    print()

    # pLDDT table
    print("=" * 80)
    print("  pLDDT comparison  (mean ± std)")
    print("=" * 80)
    print(header)
    print("  " + SEP * (hdr_w + len(MODES) * col_w))
    for g in GROUPS:
        row_str = f"  {g:<{hdr_w}}"
        for mode in MODES:
            plddt_vals = data[mode][g]["plddt"]
            m, s, n = stats(plddt_vals)
            row_str += fmt(m, s, n)
        print(row_str)
    print()

    # ── Per-group delta vs nomsa ───────────────────────────────────────────────
    print("=" * 80)
    print("  ΔTM-score vs nomsa baseline  (mean)")
    print("=" * 80)
    delta_modes = [m for m in MODES if m != "nomsa"]
    header2 = f"  {'Group':<{hdr_w}}" + "".join(f"  {m:<{col_w-2}}" for m in delta_modes)
    print(header2)
    print("  " + SEP * (hdr_w + len(delta_modes) * col_w))
    for g in GROUPS:
        row_str = f"  {g:<{hdr_w}}"
        base_tm  = stats(data["nomsa"][g]["tm"])[0]
        for mode in delta_modes:
            cmp_tm = stats(data[mode][g]["tm"])[0]
            n      = len(data[mode][g]["tm"])
            if n == 0 or math.isnan(base_tm) or math.isnan(cmp_tm):
                row_str += "  n/a".ljust(col_w)
            else:
                delta = cmp_tm - base_tm
                sign  = "+" if delta >= 0 else ""
                row_str += f"  {sign}{delta:.4f}".ljust(col_w)
        print(row_str)
    print()

    # ── Coverage report ───────────────────────────────────────────────────────
    print("=" * 80)
    print("  Coverage (proteins with TM-score)")
    print("=" * 80)
    print(f"  {'Group':<{hdr_w}}" + "".join(f"  {m:<{col_w-2}}" for m in MODES))
    print("  " + SEP * (hdr_w + len(MODES) * col_w))
    for g in GROUPS:
        row_str = f"  {g:<{hdr_w}}"
        for mode in MODES:
            n = len(data[mode][g]["tm"])
            row_str += f"  {n}".ljust(col_w)
        print(row_str)
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()

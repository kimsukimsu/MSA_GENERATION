"""
Grouped comparison of fold benchmark results.

Comparison design:
  orphan  (Neff ≤ 10)         : nomsa  vs  zeroshot
  shallow (10 < Neff ≤ 67.1)  : colabfold  vs  fewshot
  full    (Neff > 67.1)        : colabfold  vs  fewshot

Usage:
  python scripts/analyze_benchmark.py --base_dir runs/benchmark_all --neff_csv data/foldbench_groups/neff_scores.csv
"""

import argparse
import csv
import glob
import math
from pathlib import Path

SEP = "─"

GROUP_MODES = {
    "orphan":  ["nomsa", "zeroshot"],
    "shallow": ["colabfold", "fewshot"],
    "full":    ["colabfold", "fewshot"],
}
ALL_MODES = ["nomsa", "zeroshot", "colabfold", "fewshot"]


def load_neff_groups(csv_path: Path) -> dict[str, str]:
    groups = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            groups[row["name"]] = row["group"]
    return groups


def load_results(mode_dir: Path) -> list[dict]:
    csv_path = mode_dir / "benchmark_results.csv"
    if csv_path.exists():
        with open(csv_path) as f:
            return list(csv.DictReader(f))
    rows = []
    for sf in sorted(glob.glob(str(mode_dir / "shard_*.csv"))):
        with open(sf) as f:
            rows.extend(csv.DictReader(f))
    return rows


def safe_float(v) -> float | None:
    try:
        x = float(v)
        return None if math.isnan(x) else x
    except (TypeError, ValueError):
        return None


def stats(values: list[float]) -> tuple[float, float, int]:
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), 0
    mean = sum(values) / n
    var = sum((x - mean) ** 2 for x in values) / n if n > 1 else 0.0
    return mean, math.sqrt(var), n


def fmt_cell(mean, std, n, width=26):
    if n == 0:
        return "  —".ljust(width)
    return f"  {mean:.4f} ± {std:.4f}  (n={n})".ljust(width)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--neff_csv", required=True)
    args = parser.parse_args()

    base_dir = Path(args.base_dir)
    neff_groups = load_neff_groups(Path(args.neff_csv))

    # data[mode][group] = {tm: [], plddt: []}
    data: dict[str, dict[str, dict]] = {
        mode: {g: {"tm": [], "plddt": []} for g in ["orphan", "shallow", "full"]}
        for mode in ALL_MODES
    }

    for mode in ALL_MODES:
        for row in load_results(base_dir / mode):
            name = row.get("name", row.get("protein", ""))
            group = neff_groups.get(name, "missing")
            if group not in data[mode]:
                continue
            tm    = safe_float(row.get("tm_score", ""))
            plddt = safe_float(row.get("plddt", ""))
            if tm    is not None: data[mode][group]["tm"].append(tm)
            if plddt is not None: data[mode][group]["plddt"].append(plddt)

    W = 28  # column width

    def section(title, metric_key, label):
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

        # orphan block
        modes_o = GROUP_MODES["orphan"]
        header_o = f"  {'Group':<10}" + "".join(f"  {m:<{W-2}}" for m in modes_o)
        print(header_o)
        print("  " + SEP * (10 + len(modes_o) * W))
        row_str = f"  {'orphan':<10}"
        base_val = stats(data[modes_o[0]]["orphan"][metric_key])[0]
        for i, mode in enumerate(modes_o):
            m, s, n = stats(data[mode]["orphan"][metric_key])
            cell = fmt_cell(m, s, n, W)
            if i > 0 and not math.isnan(base_val) and not math.isnan(m):
                delta = m - base_val
                sign = "+" if delta >= 0 else ""
                cell = cell.rstrip() + f"  (Δ{sign}{delta:.4f})"
            row_str += cell
        print(row_str)

        # shallow + full block
        print()
        modes_sf = GROUP_MODES["shallow"]
        header_sf = f"  {'Group':<10}" + "".join(f"  {m:<{W-2}}" for m in modes_sf)
        print(header_sf)
        print("  " + SEP * (10 + len(modes_sf) * W))
        for group in ["shallow", "full"]:
            row_str = f"  {group:<10}"
            base_val = stats(data[modes_sf[0]][group][metric_key])[0]
            for i, mode in enumerate(modes_sf):
                m, s, n = stats(data[mode][group][metric_key])
                cell = fmt_cell(m, s, n, W)
                if i > 0 and not math.isnan(base_val) and not math.isnan(m):
                    delta = m - base_val
                    sign = "+" if delta >= 0 else ""
                    cell = cell.rstrip() + f"  (Δ{sign}{delta:.4f})"
                row_str += cell
            print(row_str)

    section("TM-score", "tm", "TM-score")
    section("pLDDT", "plddt", "pLDDT")

    # Coverage
    print(f"\n{'='*80}")
    print("  Coverage (proteins with TM-score)")
    print(f"{'='*80}")
    for group in ["orphan", "shallow", "full"]:
        modes = GROUP_MODES[group]
        counts = "  ".join(f"{m}={len(data[m][group]['tm'])}" for m in modes)
        print(f"  {group:<10}  {counts}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

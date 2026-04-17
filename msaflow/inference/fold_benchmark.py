"""
MSAFlow Fold Benchmark — pLDDT / TM-score evaluation.

Pipeline (mirrors paper Section 4.1 / Table 1 evaluation):
  1. Read CAMEO test proteins (FASTA, one sequence per line).
  2. Generate 32 MSA sequences per protein in zero-shot mode
     (Latent FM → SFM Decoder) and write them as A3M files.
  3. Build Protenix JSON inputs pointing to the generated A3M.
  4. Run `protenix pred` for each protein.
  5. Extract pLDDT from the output CIF/PDB.
  6. Optionally compute TM-score vs reference PDB using TMscore binary.
  7. Write summary CSV and print aggregate statistics.

Usage:
    python msaflow/inference/fold_benchmark.py \\
        --fasta        data/cameo_test.fasta \\
        --decoder_ckpt runs/decoder/decoder_ema_final.pt \\
        --latent_fm_ckpt runs/latent_fm/latent_fm_ema_final.pt \\
        --output_dir   runs/fold_benchmark \\
        [--ref_pdb_dir data/cameo_reference_pdbs/] \\
        [--device cuda] \\
        [--n_seqs 32] \\
        [--n_steps 100] \\
        [--protenix_model protenix_base_default_v1.0.0]
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FASTA parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_fasta(path: str) -> list[tuple[str, str]]:
    """Return list of (name, sequence) pairs."""
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
                buf.append(line.upper().replace(" ", ""))
    if name is not None:
        results.append((name, "".join(buf)))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Protenix input JSON builder
# ─────────────────────────────────────────────────────────────────────────────

def build_protenix_json(
    name: str,
    query_seq: str,
    msa_a3m_path: Optional[str] = None,
) -> dict:
    """
    Build a single Protenix prediction task dict.

    Uses `unpairedMsaPath` for the generated A3M (new Protenix format).
    If no MSA path is provided, Protenix will either run its own MSA search
    or fold with no MSA (depending on server setup).
    """
    protein_chain: dict = {"sequence": query_seq, "count": 1}
    if msa_a3m_path is not None:
        protein_chain["unpairedMsaPath"] = str(msa_a3m_path)
        protein_chain["pairedMsaPath"] = str(msa_a3m_path)

    return {
        "name": name,
        "sequences": [{"proteinChain": protein_chain}],
        "modelSeeds": [42],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run Protenix
# ─────────────────────────────────────────────────────────────────────────────

def run_protenix(
    input_json: str,
    output_dir: str,
    model_name: str = "protenix_base_default_v1.0.0",
    protenix_dir: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """Invoke `protenix pred` as a subprocess."""
    cmd = [
        "protenix", "pred",
        "-i", input_json,
        "-o", output_dir,
        "-n", model_name,
    ]
    env = None
    if protenix_dir:
        import os
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{protenix_dir}:{env.get('PYTHONPATH', '')}"

    logger.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if result.returncode != 0:
        logger.warning("protenix stderr:\n%s", result.stderr[-2000:])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# pLDDT extraction from CIF output
# ─────────────────────────────────────────────────────────────────────────────

def extract_plddt_from_cif(cif_path: str) -> float:
    """
    Parse mean pLDDT from a Protenix output CIF file.

    Protenix writes per-residue B-factor as pLDDT (scaled 0–100).
    We average across all protein residues.
    """
    plddt_values = []
    with open(cif_path) as fh:
        in_atom_site = False
        col_idx = {}
        for line in fh:
            line = line.rstrip()
            if line.startswith("_atom_site."):
                col_name = line.split(".")[1]
                col_idx[col_name] = len(col_idx)
                in_atom_site = True
                continue
            if in_atom_site and line.startswith("_"):
                # New block: stop parsing atom_site
                in_atom_site = False
                col_idx = {}
                continue
            if in_atom_site and line and not line.startswith("#"):
                parts = line.split()
                if len(parts) <= max(col_idx.values(), default=-1):
                    continue
                # Extract B_iso_or_equiv (= pLDDT) and group_PDB (ATOM/HETATM)
                group = parts[col_idx["group_PDB"]] if "group_PDB" in col_idx else "ATOM"
                atom_name = parts[col_idx["label_atom_id"]] if "label_atom_id" in col_idx else "CA"
                b_val = parts[col_idx["B_iso_or_equiv"]] if "B_iso_or_equiv" in col_idx else None
                if group == "ATOM" and atom_name == "CA" and b_val is not None:
                    try:
                        plddt_values.append(float(b_val))
                    except ValueError:
                        pass

    if not plddt_values:
        logger.warning("No CA atoms found in %s", cif_path)
        return float("nan")
    return float(np.mean(plddt_values))


def find_protenix_output_cif(output_dir: str, name: str) -> Optional[str]:
    """Find the Protenix output CIF/PDB file for a given prediction name."""
    base = Path(output_dir)
    # Protenix writes to: {output_dir}/{name}/{name}_*.cif (or .pdb)
    for ext in ("*.cif", "*.pdb"):
        matches = list(base.rglob(f"{name}*{ext[1:]}"))
        if matches:
            return str(matches[0])
    # Fallback: any CIF in the output directory
    matches = sorted(base.rglob("*.cif"))
    return str(matches[0]) if matches else None


# ─────────────────────────────────────────────────────────────────────────────
# TM-score computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_tmscore(
    pred_pdb: str,
    ref_pdb: str,
    tmscore_bin: str = "TMscore",
) -> tuple[float, float]:
    """
    Run TMscore binary and return (TM-score, RMSD).

    Returns (nan, nan) if binary not found or parse fails.
    """
    try:
        result = subprocess.run(
            [tmscore_bin, pred_pdb, ref_pdb],
            capture_output=True, text=True, timeout=60,
        )
        tm_score, rmsd = float("nan"), float("nan")
        for line in result.stdout.splitlines():
            if line.startswith("TM-score="):
                try:
                    tm_score = float(line.split("=")[1].split()[0])
                except (IndexError, ValueError):
                    pass
            if "RMSD of" in line and "aligned" in line:
                try:
                    rmsd = float(line.split("RMSD=")[1].split(",")[0].strip())
                except (IndexError, ValueError):
                    pass
        return tm_score, rmsd
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        logger.warning("TMscore failed: %s", exc)
        return float("nan"), float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msa_dir = output_dir / "msas"
    fold_dir = output_dir / "folds"
    msa_dir.mkdir(exist_ok=True)
    fold_dir.mkdir(exist_ok=True)

    # ── Load models ────────────────────────────────────────────────────────────
    logger.info("Loading SFM decoder from %s", args.decoder_ckpt)
    sys.path.insert(0, str(Path(__file__).parents[2]))
    from msaflow.inference.generate import (
        load_sfm_decoder, load_latent_fm, load_esm2,
        generate_zeroshot, write_a3m,
    )

    decoder = load_sfm_decoder(args.decoder_ckpt, device)
    latent_fm = load_latent_fm(args.latent_fm_ckpt, device)
    logger.info("Loading ESM2-650M ...")
    esm_model, alphabet = load_esm2(device)
    logger.info("All models loaded.")

    # ── Read test proteins ─────────────────────────────────────────────────────
    proteins = parse_fasta(args.fasta)
    logger.info("Loaded %d proteins from %s", len(proteins), args.fasta)

    results = []

    for prot_idx, (prot_name, query_seq) in enumerate(proteins):
        logger.info("=" * 60)
        logger.info("[%d/%d] %s  L=%d", prot_idx + 1, len(proteins), prot_name, len(query_seq))

        a3m_path = msa_dir / f"{prot_name}.a3m"
        prot_fold_dir = fold_dir / prot_name
        prot_fold_dir.mkdir(exist_ok=True)

        # ── Step 1: Generate MSA ───────────────────────────────────────────────
        if a3m_path.exists() and not args.regenerate:
            logger.info("  MSA already exists, skipping generation: %s", a3m_path)
        else:
            logger.info("  Generating MSA (zero-shot, %d seeds × %d seqs/seed) ...",
                        args.n_seeds, args.n_seqs)
            try:
                gen_seqs = generate_zeroshot(
                    query_seq=query_seq,
                    decoder=decoder,
                    latent_fm=latent_fm,
                    esm_model=esm_model,
                    alphabet=alphabet,
                    n_seeds=args.n_seeds,
                    n_seqs_per_seed=args.n_seqs,
                    n_steps=args.n_steps,
                    temperature=args.temperature,
                    device=device,
                )
                write_a3m(query_seq, gen_seqs, str(a3m_path), prefix=prot_name)
                logger.info("  Wrote %d sequences to %s", len(gen_seqs), a3m_path)
            except Exception as exc:
                logger.error("  MSA generation failed: %s", exc, exc_info=True)
                results.append({
                    "name": prot_name, "seq_len": len(query_seq),
                    "plddt": float("nan"), "tm_score": float("nan"), "rmsd": float("nan"),
                    "status": f"msa_error:{exc}",
                })
                continue

        # ── Step 2: Build Protenix JSON ────────────────────────────────────────
        task = build_protenix_json(
            name=prot_name,
            query_seq=query_seq,
            msa_a3m_path=str(a3m_path) if a3m_path.exists() else None,
        )
        json_path = prot_fold_dir / f"{prot_name}_input.json"
        with open(json_path, "w") as fh:
            json.dump([task], fh, indent=2)

        # ── Step 3: Run Protenix ───────────────────────────────────────────────
        protenix_out_dir = prot_fold_dir / "protenix_out"
        protenix_out_dir.mkdir(exist_ok=True)

        pred_cif = find_protenix_output_cif(str(protenix_out_dir), prot_name)
        if pred_cif and not args.refold:
            logger.info("  Protenix output already exists, skipping: %s", pred_cif)
        else:
            protenix_result = run_protenix(
                input_json=str(json_path),
                output_dir=str(protenix_out_dir),
                model_name=args.protenix_model,
            )
            if protenix_result.returncode != 0:
                logger.error("  Protenix failed (exit %d)", protenix_result.returncode)
                results.append({
                    "name": prot_name, "seq_len": len(query_seq),
                    "plddt": float("nan"), "tm_score": float("nan"), "rmsd": float("nan"),
                    "status": "protenix_failed",
                })
                continue
            pred_cif = find_protenix_output_cif(str(protenix_out_dir), prot_name)

        if pred_cif is None:
            logger.error("  Could not find Protenix output CIF for %s", prot_name)
            results.append({
                "name": prot_name, "seq_len": len(query_seq),
                "plddt": float("nan"), "tm_score": float("nan"), "rmsd": float("nan"),
                "status": "cif_not_found",
            })
            continue

        # ── Step 4: Extract pLDDT ──────────────────────────────────────────────
        plddt = extract_plddt_from_cif(pred_cif)
        logger.info("  pLDDT = %.2f  (from %s)", plddt, Path(pred_cif).name)

        # ── Step 5: TM-score (optional) ────────────────────────────────────────
        tm_score, rmsd = float("nan"), float("nan")
        if args.ref_pdb_dir:
            ref_pdb = Path(args.ref_pdb_dir) / f"{prot_name}.pdb"
            if not ref_pdb.exists():
                # Try .cif
                ref_pdb = Path(args.ref_pdb_dir) / f"{prot_name}.cif"
            if ref_pdb.exists():
                tm_score, rmsd = compute_tmscore(pred_cif, str(ref_pdb), args.tmscore_bin)
                logger.info("  TM-score = %.4f  RMSD = %.2f Å", tm_score, rmsd)
            else:
                logger.warning("  Reference structure not found: %s", ref_pdb)

        results.append({
            "name": prot_name,
            "seq_len": len(query_seq),
            "plddt": plddt,
            "tm_score": tm_score,
            "rmsd": rmsd,
            "status": "ok",
        })

    # ── Write CSV summary ──────────────────────────────────────────────────────
    csv_path = output_dir / "benchmark_results.csv"
    fieldnames = ["name", "seq_len", "plddt", "tm_score", "rmsd", "status"]
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Wrote results to %s", csv_path)

    # ── Aggregate statistics ───────────────────────────────────────────────────
    ok = [r for r in results if r["status"] == "ok"]
    plddts = [r["plddt"] for r in ok if not np.isnan(r["plddt"])]
    tms = [r["tm_score"] for r in ok if not np.isnan(r["tm_score"])]

    print("\n" + "=" * 60)
    print(f"Fold Benchmark Results  ({len(ok)}/{len(results)} succeeded)")
    print("=" * 60)
    if plddts:
        print(f"  pLDDT  mean={np.mean(plddts):.2f}  "
              f"median={np.median(plddts):.2f}  "
              f"min={np.min(plddts):.2f}  max={np.max(plddts):.2f}")
    if tms:
        print(f"  TM-score  mean={np.mean(tms):.4f}  "
              f"median={np.median(tms):.4f}  "
              f"min={np.min(tms):.4f}  max={np.max(tms):.4f}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MSAFlow fold benchmark")
    parser.add_argument("--fasta",           required=True,
                        help="FASTA file with test protein sequences")
    parser.add_argument("--decoder_ckpt",    required=True,
                        help="Path to SFM decoder EMA checkpoint")
    parser.add_argument("--latent_fm_ckpt",  required=True,
                        help="Path to latent FM EMA checkpoint")
    parser.add_argument("--output_dir",      required=True,
                        help="Root directory for MSA files, fold outputs, and CSV")
    parser.add_argument("--ref_pdb_dir",     default=None,
                        help="Directory with reference PDB files named {name}.pdb")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--n_seqs",          type=int, default=32,
                        help="Sequences per seed for zero-shot generation")
    parser.add_argument("--n_seeds",         type=int, default=10,
                        help="Number of latent FM seeds")
    parser.add_argument("--n_steps",         type=int, default=100,
                        help="ODE integration steps")
    parser.add_argument("--temperature",     type=float, default=0.0,
                        help="SDE temperature (0 = deterministic ODE)")
    parser.add_argument("--protenix_model",  default="protenix_base_default_v1.0.0",
                        help="Protenix model name")
    parser.add_argument("--tmscore_bin",     default="TMscore",
                        help="Path or name of TMscore binary")
    parser.add_argument("--regenerate",      action="store_true",
                        help="Re-generate MSAs even if A3M files already exist")
    parser.add_argument("--refold",          action="store_true",
                        help="Re-run Protenix even if CIF output already exists")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()

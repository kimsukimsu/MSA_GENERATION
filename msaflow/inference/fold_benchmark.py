"""
MSAFlow Fold Benchmark — pLDDT / TM-score evaluation.

Pipeline (mirrors paper Section 4.1 / Table 2 evaluation):
  1. Read test proteins (FASTA, one sequence per entry).
  2. Generate MSA sequences via one of three modes:

       zeroshot  — 10 seeds × 32 seqs each (Section 4.2 protocol):
                   fold each seed's MSA separately with Protenix, report best pLDDT.

       fewshot   — Syn+Rec augmentation of a shallow MSA (Section 4.2):
                   requires --shallow_msa_dir and optionally --protenix_ckpt.
                   Folds the combined augmented MSA once.

       nomsa     — Baseline: skip MSA generation, fold with sequence only.

  3. Build Protenix JSON inputs pointing to the generated A3M.
  4. Run `protenix pred` for each protein (/ seed).
  5. Extract pLDDT from the output CIF/PDB.
  6. Optionally compute TM-score vs reference PDB using TMscore binary.
  7. Write summary CSV and print aggregate statistics.

Usage (zero-shot):
    python msaflow/inference/fold_benchmark.py \\
        --fasta          data/foldbench_monomer.fasta \\
        --decoder_ckpt   runs/decoder/decoder_ema_final.pt \\
        --latent_fm_ckpt runs/latent_fm/latent_fm_ema_final.pt \\
        --output_dir     runs/fold_benchmark \\
        --mode           zeroshot \\
        [--ref_pdb_dir   data/reference_pdbs/] \\
        [--device cuda]  [--n_seeds 10] [--n_seqs 32] [--n_steps 100]

Usage (no-MSA baseline):
    python msaflow/inference/fold_benchmark.py \\
        --fasta          data/foldbench_monomer.fasta \\
        --decoder_ckpt   dummy  --latent_fm_ckpt dummy \\
        --output_dir     runs/fold_benchmark_nomsa \\
        --mode           nomsa

Usage (few-shot):
    python msaflow/inference/fold_benchmark.py \\
        --fasta          data/foldbench_monomer.fasta \\
        --decoder_ckpt   runs/decoder/decoder_ema_final.pt \\
        --latent_fm_ckpt runs/latent_fm/latent_fm_ema_final.pt \\
        --output_dir     runs/fold_benchmark_fewshot \\
        --mode           fewshot \\
        --shallow_msa_dir data/shallow_msas/ \\
        [--protenix_ckpt  runs/protenix/protenix_base.pt]
"""

import argparse
import csv
import json
import logging
import os
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


def parse_a3m_seqs(path: str) -> list[str]:
    """Read sequences from an A3M file (first entry = query, rest = hits)."""
    seqs = []
    buf = []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                buf = []
            elif line:
                buf.append(line.upper())
    if buf:
        seqs.append("".join(buf))
    return seqs


# ─────────────────────────────────────────────────────────────────────────────
# MSA file lookup (handles both plain and ColabFold assembly1__chain naming)
# ─────────────────────────────────────────────────────────────────────────────

def _find_msa_file(msa_dir: str, prot_name: str) -> Optional[str]:
    """Find an A3M file for prot_name in msa_dir.

    Tries in order:
      1. {prot_name}.a3m
      2. {pdbid}-assembly1__A.a3m   (ColabFold naming)
      3. glob {prot_name}*__A.a3m
      4. glob {prot_name}*.a3m
      (and repeats 1-4 for bare pdb_id)
    """
    d = Path(msa_dir)
    pdb_id = prot_name.split("-")[0]
    for name in [prot_name, pdb_id]:
        p = d / f"{name}.a3m"
        if p.exists():
            return str(p)
        p = d / f"{name}-assembly1__A.a3m"
        if p.exists():
            return str(p)
        hits = sorted(d.glob(f"{name}*__A.a3m"))
        if hits:
            return str(hits[0])
        hits = sorted(d.glob(f"{name}*.a3m"))
        if hits:
            return str(hits[0])
    return None


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
    If no MSA path is provided, Protenix will fold with sequence only.
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
    use_msa: bool = True,
) -> subprocess.CompletedProcess:
    """Invoke `protenix pred` as a subprocess.

    Builds a clean PYTHONPATH that strips the MSA_FLOW repo root so that the
    local esm/ submodule does not shadow the installed fair-esm package and
    cause "AttributeError: module 'esm' has no attribute 'data'".
    """
    cmd = [
        "protenix", "pred",
        "-i", input_json,
        "-o", output_dir,
        "-n", model_name,
        # Disable all custom CUDA kernels — fall back to pure PyTorch.
        # Required when cuequivariance / fast_layer_norm are not compiled
        # for the current GPU architecture.
        "--enable_fusion",  "False",
        "--trimul_kernel",  "torch",
        "--triatt_kernel",  "torch",
        "--use_msa",        str(use_msa),
    ]

    # Repo root = two levels above this file (msaflow/inference/fold_benchmark.py)
    repo_root = str(Path(__file__).parents[2].resolve())
    protenix_root = str((Path(__file__).parents[2] / "Protenix").resolve())

    # Strip $REPO_DIR from PYTHONPATH — it exposes the local esm/ submodule
    # which shadows the installed fair-esm package inside Protenix subprocess.
    old_path = os.environ.get("PYTHONPATH", "")
    clean_paths = [
        p for p in old_path.split(os.pathsep)
        if p and p.rstrip("/") != repo_root.rstrip("/")
    ]
    if protenix_root not in clean_paths:
        clean_paths.insert(0, protenix_root)
    if protenix_dir and protenix_dir not in clean_paths:
        clean_paths.insert(0, protenix_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(clean_paths)

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
    We average across all protein CA atoms.
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
                in_atom_site = False
                col_idx = {}
                continue
            if in_atom_site and line and not line.startswith("#"):
                parts = line.split()
                if len(parts) <= max(col_idx.values(), default=-1):
                    continue
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
    """Find the Protenix output CIF file for a given prediction name."""
    base = Path(output_dir)
    for ext in ("*.cif", "*.pdb"):
        matches = list(base.rglob(f"{name}*{ext[1:]}"))
        if matches:
            return str(matches[0])
    matches = sorted(base.rglob("*.cif"))
    return str(matches[0]) if matches else None


# ─────────────────────────────────────────────────────────────────────────────
# TM-score computation via USalign (CIF-native, replaces old TMscore+PDB path)
# ─────────────────────────────────────────────────────────────────────────────

def _find_ref_cif(ref_cif_dir: str, prot_name: str) -> Optional[str]:
    """Find ground-truth CIF. Tries prot_name and bare pdb_id, with/without -assembly1."""
    d = Path(ref_cif_dir)
    pdb_id = prot_name.split("-")[0]
    for name in [prot_name, pdb_id]:
        for ext in [".cif", ".cif.gz"]:
            p = d / f"{name}{ext}"
            if p.exists():
                return str(p)
        hits = list(d.glob(f"{name}*.[Cc][Ii][Ff]*"))
        if hits:
            return str(hits[0])
    return None


def _run_usalign(pred_cif: str, ref_cif: str, usalign_bin: str) -> tuple[float, float]:
    """Run USalign -outfmt 2 and return (TM-score normalised by ref len, RMSD)."""
    try:
        result = subprocess.run(
            [usalign_bin, pred_cif, ref_cif, "-outfmt", "2"],
            capture_output=True, text=True, timeout=120,
        )
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 5 and not line.startswith("#"):
                try:
                    return float(parts[3]), float(parts[4])
                except ValueError:
                    continue
        if result.returncode != 0:
            logger.warning("USalign exit %d: %s", result.returncode, result.stderr[:200])
    except FileNotFoundError:
        logger.error("USalign binary not found: %s", usalign_bin)
    except subprocess.TimeoutExpired:
        logger.warning("USalign timed out for %s", pred_cif)
    return float("nan"), float("nan")


def compute_tmscore(pred_cif: str, ref_cif: str, usalign_bin: str = "USalign") -> tuple[float, float]:
    """Thin wrapper kept for call-site compatibility."""
    return _run_usalign(pred_cif, ref_cif, usalign_bin)


# ─────────────────────────────────────────────────────────────────────────────
# Per-protein folding helpers
# ─────────────────────────────────────────────────────────────────────────────

def fold_once(
    prot_name: str,
    query_seq: str,
    a3m_path: Optional[Path],
    fold_dir: Path,
    args,
    use_msa: bool = True,
    tag: str = "",
) -> tuple[float, Optional[str]]:
    """
    Build Protenix input, run Protenix, extract pLDDT.

    Returns (plddt, cif_path_or_None).
    """
    run_name = f"{prot_name}{tag}"
    run_dir = fold_dir / run_name
    run_dir.mkdir(exist_ok=True)

    task = build_protenix_json(
        name=run_name,
        query_seq=query_seq,
        msa_a3m_path=str(a3m_path) if (a3m_path is not None and a3m_path.exists()) else None,
    )
    json_path = run_dir / f"{run_name}_input.json"
    with open(json_path, "w") as fh:
        json.dump([task], fh, indent=2)

    protenix_out_dir = run_dir / "protenix_out"
    protenix_out_dir.mkdir(exist_ok=True)

    pred_cif = find_protenix_output_cif(str(protenix_out_dir), run_name)
    if pred_cif and not args.refold:
        logger.info("  Protenix output exists, skipping: %s", pred_cif)
        return extract_plddt_from_cif(pred_cif), pred_cif

    result = run_protenix(
        input_json=str(json_path),
        output_dir=str(protenix_out_dir),
        model_name=args.protenix_model,
        use_msa=use_msa,
    )
    if result.returncode != 0:
        logger.error("  Protenix failed (exit %d) for %s", result.returncode, run_name)
        return float("nan"), None

    pred_cif = find_protenix_output_cif(str(protenix_out_dir), run_name)
    if pred_cif is None:
        logger.error("  CIF not found for %s", run_name)
        return float("nan"), None

    return extract_plddt_from_cif(pred_cif), pred_cif


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  [ %(name)s ]  %(message)s",
        datefmt="%Y-%m-%d %H:%M",
    )

    # Resolve mode (--no_msa is a deprecated alias for --mode nomsa)
    mode = args.mode
    if args.no_msa:
        mode = "nomsa"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    msa_dir = output_dir / "msas"
    fold_dir = output_dir / "folds"
    msa_dir.mkdir(exist_ok=True)
    fold_dir.mkdir(exist_ok=True)

    # ── Load models (skip in nomsa / colabfold baseline modes) ────────────────
    if mode not in ("nomsa", "colabfold"):
        logger.info("Loading SFM decoder from %s", args.decoder_ckpt)
        sys.path.insert(0, str(Path(__file__).parents[2]))
        from msaflow.inference.generate import (
            load_sfm_decoder, load_latent_fm, load_esm2, load_protenix,
            generate_zeroshot_seeds, augment_shallow, write_a3m,
        )
        decoder = load_sfm_decoder(args.decoder_ckpt, device)
        latent_fm = load_latent_fm(args.latent_fm_ckpt, device)
        logger.info("Loading ESM2-650M ...")
        esm_model, alphabet = load_esm2(device)

        protenix_model = None
        if mode == "fewshot" and args.protenix_ckpt:
            logger.info("Loading Protenix for Rec track from %s", args.protenix_ckpt)
            protenix_model = load_protenix(args.protenix_ckpt, device)
            logger.info("Protenix loaded for Rec track.")
        elif mode == "fewshot":
            logger.warning(
                "--protenix_ckpt not provided; few-shot will run Syn track only "
                "(no Rec track from shallow MSA embedding)."
            )
    elif mode == "colabfold":
        logger.info("ColabFold direct fold mode: skipping MSA model loading.")

        logger.info("All models loaded.")
    else:
        logger.info("No-MSA baseline mode: skipping MSA model loading.")

    # ── Read test proteins ─────────────────────────────────────────────────────
    proteins = parse_fasta(args.fasta)
    logger.info("Loaded %d proteins from %s", len(proteins), args.fasta)

    # ── Shard the protein list across parallel workers ─────────────────────────
    if args.num_shards > 1:
        proteins = proteins[args.shard_id :: args.num_shards]
        logger.info("Shard %d/%d: processing %d proteins",
                    args.shard_id, args.num_shards, len(proteins))

    results = []

    for prot_idx, (prot_name, query_seq) in enumerate(proteins):
        logger.info("=" * 60)
        logger.info("[%d/%d] %s  L=%d  mode=%s",
                    prot_idx + 1, len(proteins), prot_name, len(query_seq), mode)

        # ── Zero-shot: fold each seed separately, report best pLDDT ───────────
        if mode == "zeroshot":
            prot_msa_dir = msa_dir / prot_name
            prot_msa_dir.mkdir(exist_ok=True)

            best_plddt = float("nan")
            best_seed = -1
            best_cif = None

            for seed, seqs in generate_zeroshot_seeds(
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
            ):
                seed_a3m = prot_msa_dir / f"seed_{seed}.a3m"
                if not seed_a3m.exists() or args.regenerate:
                    write_a3m(query_seq, seqs, str(seed_a3m), prefix=prot_name)
                    logger.info("  Seed %d: wrote %d seqs to %s", seed, len(seqs), seed_a3m)

                plddt, cif = fold_once(
                    prot_name=prot_name,
                    query_seq=query_seq,
                    a3m_path=seed_a3m,
                    fold_dir=fold_dir,
                    args=args,
                    use_msa=True,
                    tag=f"_seed{seed}",
                )
                logger.info("  Seed %d pLDDT=%.2f  (best so far=%.2f)",
                            seed, plddt, best_plddt if not np.isnan(best_plddt) else -1)

                if np.isnan(best_plddt) or plddt > best_plddt:
                    best_plddt = plddt
                    best_seed = seed
                    best_cif = cif

            logger.info("  Best seed=%d  pLDDT=%.2f", best_seed, best_plddt)

            tm_score, rmsd = float("nan"), float("nan")
            if args.ref_cif_dir and best_cif:
                ref_cif = _find_ref_cif(args.ref_cif_dir, prot_name)
                if ref_cif:
                    tm_score, rmsd = _run_usalign(best_cif, ref_cif, args.usalign_bin)
                    logger.info("  TM-score=%.4f  RMSD=%.2f Å", tm_score, rmsd)

            results.append({
                "name": prot_name, "seq_len": len(query_seq),
                "plddt": best_plddt, "best_seed": best_seed,
                "tm_score": tm_score, "rmsd": rmsd,
                "status": "ok" if not np.isnan(best_plddt) else "failed",
            })

        # ── ColabFold direct fold (no generation) ─────────────────────────────
        elif mode == "colabfold":
            ref_a3m = _find_msa_file(args.shallow_msa_dir, prot_name) if args.shallow_msa_dir else None
            if ref_a3m is None:
                logger.warning("  ColabFold A3M not found for %s — skipping", prot_name)
                results.append({
                    "name": prot_name, "seq_len": len(query_seq),
                    "plddt": float("nan"), "best_seed": -1,
                    "tm_score": float("nan"), "rmsd": float("nan"),
                    "status": "ref_a3m_missing",
                })
                continue
            logger.info("  ColabFold MSA: %s", ref_a3m)
            plddt, cif = fold_once(
                prot_name=prot_name,
                query_seq=query_seq,
                a3m_path=Path(ref_a3m),
                fold_dir=fold_dir,
                args=args,
                use_msa=True,
                tag="_colabfold",
            )
            logger.info("  pLDDT=%.2f", plddt)
            tm_score, rmsd = float("nan"), float("nan")
            if args.ref_cif_dir and cif:
                ref_cif = _find_ref_cif(args.ref_cif_dir, prot_name)
                if ref_cif:
                    tm_score, rmsd = _run_usalign(cif, ref_cif, args.usalign_bin)
                    logger.info("  TM-score=%.4f  RMSD=%.2f Å", tm_score, rmsd)
            results.append({
                "name": prot_name, "seq_len": len(query_seq),
                "plddt": plddt, "best_seed": -1,
                "tm_score": tm_score, "rmsd": rmsd,
                "status": "ok" if not np.isnan(plddt) else "failed",
            })

        # ── Few-shot: augment shallow MSA, fold once ───────────────────────────
        elif mode == "fewshot":
            # Load shallow MSA for this target
            shallow_seqs = None
            if args.shallow_msa_dir:
                shallow_a3m = _find_msa_file(args.shallow_msa_dir, prot_name)
                if shallow_a3m:
                    shallow_seqs = parse_a3m_seqs(shallow_a3m)
                    logger.info("  Loaded shallow MSA: %d seqs from %s",
                                len(shallow_seqs), shallow_a3m)
                else:
                    logger.warning("  Shallow A3M not found for %s; using depth-1 (query only)",
                                   prot_name)

            if not shallow_seqs:
                # Depth-1 MSA = query sequence only (ungapped, aligned to itself)
                shallow_seqs = [query_seq]
                logger.info("  Using depth-1 shallow MSA (query only)")

            prot_msa_dir = msa_dir / prot_name
            prot_msa_dir.mkdir(exist_ok=True)
            aug_a3m = prot_msa_dir / "augmented.a3m"

            if not aug_a3m.exists() or args.regenerate:
                try:
                    aug_seqs = augment_shallow(
                        shallow_seqs=shallow_seqs,
                        decoder=decoder,
                        latent_fm=latent_fm,
                        protenix_model=protenix_model,   # None → Syn track only
                        esm_model=esm_model,
                        alphabet=alphabet,
                        n_syn_seeds=args.n_seeds,
                        n_seqs_per_seed=args.n_seqs,
                        n_rec_seqs=64 if protenix_model else 0,
                        n_diverse=16,
                        n_steps=args.n_steps,
                        temperature=args.temperature,
                        max_rec_depth=args.max_rec_depth,
                        device=device,
                    )
                    write_a3m(query_seq, aug_seqs, str(aug_a3m), prefix=prot_name)
                    logger.info("  Wrote %d augmented seqs to %s", len(aug_seqs), aug_a3m)
                except Exception as exc:
                    logger.error("  Augmentation failed: %s", exc, exc_info=True)
                    results.append({
                        "name": prot_name, "seq_len": len(query_seq),
                        "plddt": float("nan"), "best_seed": -1,
                        "tm_score": float("nan"), "rmsd": float("nan"),
                        "status": f"aug_error:{exc}",
                    })
                    continue

            plddt, cif = fold_once(
                prot_name=prot_name,
                query_seq=query_seq,
                a3m_path=aug_a3m,
                fold_dir=fold_dir,
                args=args,
                use_msa=True,
                tag="_fewshot",
            )
            logger.info("  pLDDT=%.2f", plddt)

            tm_score, rmsd = float("nan"), float("nan")
            if args.ref_cif_dir and cif:
                ref_cif = _find_ref_cif(args.ref_cif_dir, prot_name)
                if ref_cif:
                    tm_score, rmsd = _run_usalign(cif, ref_cif, args.usalign_bin)
                    logger.info("  TM-score=%.4f  RMSD=%.2f Å", tm_score, rmsd)

            results.append({
                "name": prot_name, "seq_len": len(query_seq),
                "plddt": plddt, "best_seed": -1,
                "tm_score": tm_score, "rmsd": rmsd,
                "status": "ok" if not np.isnan(plddt) else "failed",
            })

        # ── No-MSA baseline ────────────────────────────────────────────────────
        elif mode == "nomsa":
            plddt, cif = fold_once(
                prot_name=prot_name,
                query_seq=query_seq,
                a3m_path=None,
                fold_dir=fold_dir,
                args=args,
                use_msa=False,
                tag="",
            )
            logger.info("  pLDDT=%.2f", plddt)

            tm_score, rmsd = float("nan"), float("nan")
            if args.ref_cif_dir and cif:
                ref_cif = _find_ref_cif(args.ref_cif_dir, prot_name)
                if ref_cif:
                    tm_score, rmsd = _run_usalign(cif, ref_cif, args.usalign_bin)
                    logger.info("  TM-score=%.4f  RMSD=%.2f Å", tm_score, rmsd)

            results.append({
                "name": prot_name, "seq_len": len(query_seq),
                "plddt": plddt, "best_seed": -1,
                "tm_score": tm_score, "rmsd": rmsd,
                "status": "ok" if not np.isnan(plddt) else "failed",
            })

    # ── Write CSV summary ──────────────────────────────────────────────────────
    shard_suffix = f"_shard{args.shard_id}" if args.num_shards > 1 else ""
    csv_path = output_dir / f"shard{shard_suffix}.csv"
    fieldnames = ["name", "seq_len", "plddt", "best_seed", "tm_score", "rmsd", "status"]
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
    print(f"Fold Benchmark Results  mode={mode}  ({len(ok)}/{len(results)} succeeded)")
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
    parser.add_argument("--mode",            default="zeroshot",
                        choices=["zeroshot", "fewshot", "nomsa", "colabfold"],
                        help="Generation mode: zeroshot | fewshot | nomsa | colabfold (default: zeroshot)")
    parser.add_argument("--ref_cif_dir",     default=None,
                        help="Directory with ground-truth CIF files (USalign TM-score inline)")
    parser.add_argument("--device",          default="cuda")
    parser.add_argument("--n_seqs",          type=int, default=32,
                        help="Sequences per seed")
    parser.add_argument("--n_seeds",         type=int, default=10,
                        help="Number of latent FM seeds (zeroshot: folds each, reports best)")
    parser.add_argument("--n_steps",         type=int, default=100,
                        help="ODE integration steps")
    parser.add_argument("--temperature",     type=float, default=0.5,
                        help="SDE temperature (paper: 0.5)")
    parser.add_argument("--protenix_model",  default="protenix_base_default_v1.0.0",
                        help="Protenix model name for structure prediction")
    parser.add_argument("--usalign_bin",     default="USalign",
                        help="Path to USalign binary (used for inline TM-score computation)")
    # Few-shot / colabfold specific
    parser.add_argument("--shallow_msa_dir", default=None,
                        help="[fewshot|colabfold] Directory with A3M files. "
                             "Tries {target}.a3m then {pdbid}-assembly1__A.a3m (ColabFold naming).")
    parser.add_argument("--protenix_ckpt",   default=None,
                        help="[fewshot] Protenix .pt checkpoint for Rec track MSA embedding")
    parser.add_argument("--max_rec_depth",   type=int, default=128,
                        help="[fewshot] Max sequences fed to Protenix MSA encoder "
                             "(subsampled randomly if exceeded; prevents OOM on full MSAs)")
    # Parallelism
    parser.add_argument("--shard_id",        type=int, default=0,
                        help="0-based shard index for parallel runs (default: 0)")
    parser.add_argument("--num_shards",      type=int, default=1,
                        help="Total number of shards (default: 1 = no sharding)")
    # Re-run flags
    parser.add_argument("--regenerate",      action="store_true",
                        help="Re-generate MSAs even if A3M files already exist")
    parser.add_argument("--refold",          action="store_true",
                        help="Re-run Protenix even if CIF output already exists")
    # Deprecated alias
    parser.add_argument("--no_msa",          action="store_true",
                        help="Deprecated: use --mode nomsa instead")
    args = parser.parse_args()

    run_benchmark(args)


if __name__ == "__main__":
    main()

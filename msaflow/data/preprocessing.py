"""
MSAFlow data preprocessing pipeline.

Two preprocessing stages:

1. extract_msa_embeddings()
   Runs the Protenix MSAModule over OpenFold MSA files and dumps the
   compressed pair representation (L×128) to an LMDB database.
   This corresponds to the AF3-encoder step described in Section 3.1 and 6.8.1.

2. extract_esm_embeddings()
   Computes ESM2-650M representations (L×1280) for query sequences and stores
   them alongside the MSA embeddings.

Both functions write to the same LMDB, keyed by protein/MSA ID.

LMDB schema per entry (pickled dict):
  {
    "msa_emb":     torch.Tensor  [L, 128]   – compressed MSA embedding
    "esm_emb":     torch.Tensor  [L, 1280]  – ESM2 query embedding
    "msa_tokens":  torch.Tensor  [N, L]     – integer token ids (0..21)
    "weights":     torch.Tensor  [N]        – Neff reweighting
    "query_seq":   str                      – query sequence (no gaps)
    "seq_len":     int                      – L
  }
"""

import os
import sys
import pickle
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import lmdb
from tqdm import tqdm

# ── Alphabet shared between MSA tokenisation and SFM decoder ─────────────────
# 20 canonical AAs (ACDEFGHIKLMNPQRSTVWY), gap '-', unknown 'X'
AA_LIST = list("ACDEFGHIKLMNPQRSTVWY") + ["-", "X"]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
VOCAB_SIZE = len(AA_LIST)  # 22

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# MSA I/O helpers
# ──────────────────────────────────────────────────────────────────────────────

def parse_a3m(path: str) -> tuple[list[str], list[str]]:
    """Parse A3M file; remove insertion columns (lowercase letters)."""
    names, seqs = [], []
    name, buf = None, []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs.append("".join(buf))
                name = line[1:]
                names.append(name)
                buf = []
            else:
                buf.append("".join(c for c in line if c.isupper() or c == "-"))
        if name is not None:
            seqs.append("".join(buf))
    return names, seqs


def filter_msa(seqs: list[str], max_gap_frac: float = 0.1, min_seqs: int = 10) -> list[str]:
    """Keep sequences where at most max_gap_frac of columns are gaps."""
    if not seqs:
        return seqs
    L = len(seqs[0])
    filtered = [s for s in seqs if s.count("-") / max(L, 1) <= max_gap_frac]
    return filtered if len(filtered) >= min_seqs else seqs[:min_seqs]


def tokenise_msa(seqs: list[str]) -> np.ndarray:
    """Convert list of aligned sequences to integer array (N, L)."""
    N = len(seqs)
    L = len(seqs[0])
    arr = np.zeros((N, L), dtype=np.int32)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            arr[i, j] = AA_TO_IDX.get(aa.upper(), AA_TO_IDX["X"])
    return arr


def compute_sequence_weights(tokens: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """
    Neff reweighting: w_i = 1 / |{j : hamming(i,j) < threshold}|

    This is the scheme used in the paper (Section 6.8.2):
        w_i = (1 + Σ_{j≠i} 1{d_hamming(x_i, x_j) < 0.2})^{-1}
    """
    N, L = tokens.shape
    if N == 1:
        return np.ones(1, dtype=np.float32)
    weights = np.zeros(N, dtype=np.float32)
    for i in range(N):
        n_sim = 1  # count self
        for j in range(N):
            if j != i:
                hamming = np.mean(tokens[i] != tokens[j])
                if hamming < threshold:
                    n_sim += 1
        weights[i] = 1.0 / n_sim
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# MSA embedding extraction via Protenix
# ──────────────────────────────────────────────────────────────────────────────

def _build_protenix_msa_input(seqs: list[str], device: torch.device) -> dict:
    """
    Build minimal input_feature_dict for the Protenix MSAModule.

    The MSAModule (Algorithm 8 in AF3) expects:
      - msa:             (1, N_msa, L) int64  — amino acid indices (0..31 = AF3 vocab)
      - has_deletion:    (1, N_msa, L) float
      - deletion_value:  (1, N_msa, L) float

    We map our 22-token alphabet to AF3's 32-token residue encoding.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parents[3] / "Protenix"))
    from protenix.data.constants import STD_RESIDUES_WITH_GAP

    # Build AF3 residue → index mapping
    af3_res_to_idx = {r: i for i, r in enumerate(STD_RESIDUES_WITH_GAP)}
    # Map our tokens to AF3 indices (unknown → last index 31)
    unk_idx = len(STD_RESIDUES_WITH_GAP) - 1

    N = len(seqs)
    L = len(seqs[0])
    msa_arr = np.zeros((N, L), dtype=np.int64)
    for i, seq in enumerate(seqs):
        for j, aa in enumerate(seq):
            msa_arr[i, j] = af3_res_to_idx.get(aa.upper(), unk_idx)

    msa_t = torch.from_numpy(msa_arr).unsqueeze(0).to(device)   # (1, N, L)
    zeros = torch.zeros(1, N, L, device=device)

    return {
        "msa": msa_t,
        "has_deletion": zeros,
        "deletion_value": zeros,
    }


def extract_msa_embedding_protenix(
    seqs: list[str],
    protenix_model,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract compressed MSA embedding using Protenix's MSAModule.

    Steps (mirrors Section 3.1 & 6.8.1 of the paper):
      1. Build AF3-format input features from MSA sequences.
      2. Run the MSAModule to obtain pair representation P ∈ R^(L×L×128).
      3. Mean-pool along the second spatial dimension → m_seq ∈ R^(L×128).

    Args:
        seqs:            List of aligned sequences (all same length L).
        protenix_model:  Loaded Protenix model (nn.Module) with .msa_module attribute.
        device:          Compute device.

    Returns:
        m_seq: (L, 128) compressed MSA embedding.
    """
    L = len(seqs[0])

    # Build dummy pair embedding z and single embedding s_inputs
    # (MSAModule updates z; we start from zeros for embedding extraction)
    c_z = 128
    c_s = 449
    z = torch.zeros(1, L, L, c_z, device=device)
    s_inputs = torch.zeros(1, L, c_s, device=device)
    pair_mask = torch.ones(1, L, L, device=device, dtype=torch.bool)

    feat = _build_protenix_msa_input(seqs, device)

    with torch.no_grad():
        z_updated = protenix_model.msa_module(
            input_feature_dict=feat,
            z=z,
            s_inputs=s_inputs,
            pair_mask=pair_mask,
        )  # (1, L, L, 128)

    # Mean-pool along second spatial dimension (Eq. 4 in paper)
    m_seq = z_updated[0].mean(dim=1)   # (L, 128)
    return m_seq.cpu()


# ──────────────────────────────────────────────────────────────────────────────
# ESM2 embedding extraction
# ──────────────────────────────────────────────────────────────────────────────

def extract_esm_embedding(
    query_seq: str,
    esm_model,
    alphabet,
    device: torch.device,
    layer: int = 33,
) -> torch.Tensor:
    """
    Extract ESM2-650M representation for a single query sequence.

    Args:
        query_seq: Amino acid sequence (no gaps), length L.
        esm_model: Loaded ESM2 model.
        alphabet:  ESM2 Alphabet object.
        device:    Compute device.
        layer:     Representation layer (33 for ESM2-650M).

    Returns:
        emb: (L, 1280) ESM2 representation (BOS/EOS tokens removed).
    """
    converter = alphabet.get_batch_converter()
    _, _, tokens = converter([("query", query_seq)])
    tokens = tokens.to(device)

    with torch.no_grad():
        out = esm_model(tokens, repr_layers=[layer], return_contacts=False)

    emb = out["representations"][layer][0, 1:-1, :]  # remove BOS/EOS → (L, 1280)
    return emb.cpu()


# ──────────────────────────────────────────────────────────────────────────────
# Main LMDB building routine
# ──────────────────────────────────────────────────────────────────────────────

def build_lmdb(
    a3m_dir: str,
    output_path: str,
    protenix_checkpoint: Optional[str] = None,
    max_msa_seqs: int = 512,
    max_seq_len: int = 1024,
    device: str = "cuda",
    map_size_gb: int = 500,
):
    """
    Process all A3M files in a3m_dir and write LMDB database.

    Args:
        a3m_dir:              Directory containing *.a3m files.
        output_path:          Path for the output LMDB.
        protenix_checkpoint:  Path to Protenix checkpoint. If None, skip MSA emb.
        max_msa_seqs:         Maximum MSA depth to use.
        max_seq_len:          Maximum sequence length.
        device:               Compute device string.
        map_size_gb:          LMDB map size in GB.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")

    # Load Protenix if checkpoint provided
    protenix_model = None
    if protenix_checkpoint is not None:
        sys.path.insert(0, str(Path(__file__).parents[3] / "Protenix"))
        from protenix.model.protenix import Protenix
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(Path(protenix_checkpoint).parent / "config.yaml")
        protenix_model = Protenix(cfg).eval().to(dev)
        state = torch.load(protenix_checkpoint, map_location=dev)
        protenix_model.load_state_dict(state["model"], strict=False)
        logger.info("Loaded Protenix from %s", protenix_checkpoint)

    # Load ESM2-650M
    sys.path.insert(0, str(Path(__file__).parents[3] / "esm"))
    import esm as esm_lib
    esm_model, alphabet = esm_lib.pretrained.esm2_t33_650M_UR50D()
    esm_model = esm_model.eval().to(dev)
    logger.info("Loaded ESM2-650M")

    a3m_files = list(Path(a3m_dir).glob("**/*.a3m"))
    logger.info("Found %d A3M files", len(a3m_files))

    env = lmdb.open(
        output_path,
        map_size=map_size_gb * (1024 ** 3),
        subdir=False,
        meminit=False,
        map_async=True,
    )

    n_written = 0
    for a3m_path in tqdm(a3m_files, desc="Building LMDB"):
        key = a3m_path.stem
        try:
            names, seqs = parse_a3m(str(a3m_path))
            if not seqs:
                continue
            seqs = filter_msa(seqs)
            seqs = [s[:max_seq_len] for s in seqs]
            L = len(seqs[0])
            if L == 0 or L > max_seq_len:
                continue

            query_seq = seqs[0].replace("-", "")
            tokens = tokenise_msa(seqs[:max_msa_seqs])   # (N, L)
            weights = compute_sequence_weights(tokens)    # (N,)

            entry = {
                "msa_tokens": tokens,
                "weights": weights,
                "query_seq": query_seq,
                "seq_len": L,
                "msa_emb": None,
                "esm_emb": None,
            }

            # ESM2 embedding for query
            entry["esm_emb"] = extract_esm_embedding(
                query_seq, esm_model, alphabet, dev
            ).half().numpy()

            # Protenix MSA embedding
            if protenix_model is not None:
                entry["msa_emb"] = extract_msa_embedding_protenix(
                    seqs[:max_msa_seqs], protenix_model, dev
                ).half().numpy()

            with env.begin(write=True) as txn:
                txn.put(key.encode(), pickle.dumps(entry))
            n_written += 1

        except Exception as exc:
            logger.warning("Skipped %s: %s", a3m_path.name, exc)

    env.close()
    logger.info("Wrote %d entries to %s", n_written, output_path)


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Build MSAFlow LMDB dataset")
    parser.add_argument("--a3m_dir",              required=True)
    parser.add_argument("--output",               required=True)
    parser.add_argument("--protenix_checkpoint",  default=None)
    parser.add_argument("--max_msa_seqs",         type=int, default=512)
    parser.add_argument("--max_seq_len",          type=int, default=1024)
    parser.add_argument("--device",               default="cuda")
    parser.add_argument("--map_size_gb",          type=int, default=500)
    args = parser.parse_args()

    build_lmdb(
        a3m_dir=args.a3m_dir,
        output_path=args.output,
        protenix_checkpoint=args.protenix_checkpoint,
        max_msa_seqs=args.max_msa_seqs,
        max_seq_len=args.max_seq_len,
        device=args.device,
        map_size_gb=args.map_size_gb,
    )

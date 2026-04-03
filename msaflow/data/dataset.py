"""
MSAFlow PyTorch Datasets.

Two datasets are provided:

  MSADecoderDataset — for training the SFM decoder.
    Each item: (msa_emb, tokens_32, weights_32)
    Samples 32 sequences from the MSA with Neff reweighting.

  LatentFMDataset — for training the latent flow matching encoder.
    Each item: (msa_emb, esm_emb)
    One-to-one: query embedding → MSA embedding.
"""

import pickle
import random
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from msaflow.data.preprocessing import VOCAB_SIZE


class MSADecoderDataset(Dataset):
    """
    Dataset for training the SFM decoder.

    Reads from the LMDB written by preprocessing.build_lmdb().
    Each __getitem__ samples `n_seqs_per_msa` sequences from the MSA
    (weighted by Neff weights), returning:
      - msa_emb:   (L, 128) compressed MSA embedding  [float16 → float32]
      - tokens:    (n_seqs, L) integer token ids
      - weights:   (n_seqs,) Neff weights for the sampled sequences

    Args:
        lmdb_path:       Path to the LMDB database.
        n_seqs_per_msa:  Number of sequences to sample per MSA (paper uses 32).
        max_seq_len:     Crop/pad to this length.
        require_msa_emb: If True, skip entries without a precomputed MSA embedding.
    """

    def __init__(
        self,
        lmdb_path: str,
        n_seqs_per_msa: int = 32,
        max_seq_len: int = 1024,
        require_msa_emb: bool = True,
    ):
        import lmdb as lmdb_lib
        self.env = lmdb_lib.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.n_seqs = n_seqs_per_msa
        self.max_len = max_seq_len
        self.require_msa_emb = require_msa_emb

        with self.env.begin() as txn:
            all_keys = [k.decode() for k in txn.cursor().iternext(keys=True, values=False)]

        if require_msa_emb:
            # Filter out entries without MSA embeddings
            valid = []
            with self.env.begin() as txn:
                for k in all_keys:
                    val = txn.get(k.encode())
                    entry = pickle.loads(val)
                    if entry.get("msa_emb") is not None:
                        valid.append(k)
            self.keys = valid
        else:
            self.keys = all_keys

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict:
        with self.env.begin() as txn:
            entry = pickle.loads(txn.get(self.keys[idx].encode()))

        tokens_all = torch.from_numpy(entry["msa_tokens"].astype(np.int64))   # (N, L)
        weights_all = torch.from_numpy(entry["weights"].astype(np.float32))   # (N,)
        msa_emb = torch.from_numpy(entry["msa_emb"].astype(np.float32))       # (L, 128)

        N, L = tokens_all.shape
        L = min(L, self.max_len)
        tokens_all = tokens_all[:, :L]
        msa_emb = msa_emb[:L]

        # Sample n_seqs sequences with Neff weighting
        if N <= self.n_seqs:
            idx_sampled = torch.arange(N)
        else:
            probs = weights_all / weights_all.sum()
            idx_sampled = torch.multinomial(probs, self.n_seqs, replacement=False)

        tokens = tokens_all[idx_sampled]          # (n_seqs, L)
        weights = weights_all[idx_sampled]        # (n_seqs,)

        return {
            "msa_emb": msa_emb,           # (L, 128)
            "tokens": tokens,             # (n_seqs, L)
            "weights": weights,           # (n_seqs,)
        }


class LatentFMDataset(Dataset):
    """
    Dataset for training the latent flow matching encoder.

    Each item is a (msa_emb, esm_emb) pair for the query sequence.

    Args:
        lmdb_path:   Path to LMDB.
        max_seq_len: Crop to this length.
    """

    def __init__(self, lmdb_path: str, max_seq_len: int = 1024):
        import lmdb as lmdb_lib
        self.env = lmdb_lib.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        self.max_len = max_seq_len

        with self.env.begin() as txn:
            all_keys = [k.decode() for k in txn.cursor().iternext(keys=True, values=False)]

        # Require both embeddings
        valid = []
        with self.env.begin() as txn:
            for k in all_keys:
                entry = pickle.loads(txn.get(k.encode()))
                if entry.get("msa_emb") is not None and entry.get("esm_emb") is not None:
                    valid.append(k)
        self.keys = valid

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict:
        with self.env.begin() as txn:
            entry = pickle.loads(txn.get(self.keys[idx].encode()))

        L = min(entry["seq_len"], self.max_len)
        msa_emb = torch.from_numpy(entry["msa_emb"].astype(np.float32))[:L]   # (L, 128)
        esm_emb = torch.from_numpy(entry["esm_emb"].astype(np.float32))[:L]   # (L, 1280)

        return {
            "msa_emb": msa_emb,    # (L, 128)
            "esm_emb": esm_emb,    # (L, 1280)
        }


# ──────────────────────────────────────────────────────────────────────────────
# Collation helpers (variable-length sequences)
# ──────────────────────────────────────────────────────────────────────────────

def _pad2d(t: torch.Tensor, L: int, pad_val: float = 0.0) -> torch.Tensor:
    """Pad tensor to length L along dim 0 (or dim 1 for 2D)."""
    if t.dim() == 1:
        return torch.nn.functional.pad(t, (0, L - t.shape[0]), value=pad_val)
    return torch.nn.functional.pad(t, (0, 0, 0, L - t.shape[0]), value=pad_val)


def decoder_collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for MSADecoderDataset.

    Pads all sequences in the batch to the same length.
    The SFM decoder processes each sequence independently, so we flatten the
    (n_seqs, L) token arrays into the batch dimension:
        tokens: (B * n_seqs, L_max)
        msa_emb: (B * n_seqs, L_max, 128)   — repeated for each seq in the MSA
        weights: (B * n_seqs,)
    """
    L_max = max(item["msa_emb"].shape[0] for item in batch)

    msa_embs, tokens_list, weights_list = [], [], []
    for item in batch:
        L = item["msa_emb"].shape[0]
        n_seqs = item["tokens"].shape[0]

        msa_emb_pad = _pad2d(item["msa_emb"], L_max)           # (L_max, 128)
        tokens_pad = torch.nn.functional.pad(                   # (n_seqs, L_max)
            item["tokens"], (0, L_max - L), value=0
        )

        # Repeat MSA embedding for every sequence in this MSA
        msa_embs.append(msa_emb_pad.unsqueeze(0).expand(n_seqs, -1, -1))
        tokens_list.append(tokens_pad)
        weights_list.append(item["weights"])

    return {
        "msa_emb": torch.cat(msa_embs, dim=0),        # (B*n_seqs, L_max, 128)
        "tokens":  torch.cat(tokens_list, dim=0),      # (B*n_seqs, L_max)
        "weights": torch.cat(weights_list, dim=0),     # (B*n_seqs,)
    }


def latent_collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for LatentFMDataset.
    Pads all sequences to the same length.
    """
    L_max = max(item["msa_emb"].shape[0] for item in batch)

    msa_embs, esm_embs = [], []
    for item in batch:
        msa_embs.append(_pad2d(item["msa_emb"], L_max))
        esm_embs.append(_pad2d(item["esm_emb"], L_max))

    return {
        "msa_emb": torch.stack(msa_embs),    # (B, L_max, 128)
        "esm_emb": torch.stack(esm_embs),    # (B, L_max, 1280)
    }

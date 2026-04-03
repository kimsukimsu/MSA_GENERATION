# MSAFlow

**MSAFlow: A Unified Approach for MSA Representation, Augmentation, and Family-based Protein Design**

> Reproduction of the MSAFlow paper (ICLR 2026 under review).  
> Statistical Flow Matching on the Fisher-Rao manifold for generative MSA modeling.

---

## Overview

Multiple Sequence Alignments (MSAs) encode evolutionary information critical for protein structure prediction. MSAFlow addresses two core challenges:

1. **Orphan proteins** — insufficient homologs found by HHBlits/MMSeqs2
2. **Storage efficiency** — compressing thousands of sequences into a compact latent

The framework combines:
- A **generative autoencoder**: AF3 MSAModule encodes an MSA → compressed embedding; SFM decoder reconstructs sequences
- A **latent flow matching model**: maps a single ESM2 embedding → synthetic MSA embedding (zero-shot)

### Three generation modes

| Mode | Input | Use case |
|------|-------|----------|
| **Reconstruct** | Deep MSA | Compress & reconstruct (6.5% storage, pLDDT 89.0) |
| **Few-shot augment** | Shallow MSA | Syn+Rec combined tracks |
| **Zero-shot** | Single sequence | Orphan / de novo proteins |

---

## Architecture

```
Single sequence ──► ESM2-650M ──► Latent FM (130M) ──┐
                                                       ▼
Shallow/Deep MSA ──► AF3 MSAModule ──► Compress ──► SFM Decoder (130M) ──► Generated MSA
```

### SFM Decoder
- 12 × DiT blocks, hidden dim 768, 12 attention heads (~130M params)
- **Position-wise AdaLN**: each residue position gets its own (shift, scale, gate) modulated by the per-residue MSA embedding — enabling fine-grained evolutionary conditioning
- Operates on the **Fisher-Rao statistical manifold** via the mapping π: µ → √µ

### Latent FM Encoder
- 12 × DiT blocks, hidden dim 768, 12 attention heads (~130M params)
- Conditional rectified flow: ESM2 embedding (L×1280) → MSA embedding (L×128)
- SDE inference with temperature control for diversity/fidelity tradeoff

---

## Repository Structure

```
msaflow/
├── configs/
│   ├── decoder.yaml           # SFM decoder training config
│   ├── latent_fm.yaml         # Latent FM training config
│   └── accelerate_4gpu.yaml   # Distributed training config
├── data/
│   ├── dataset.py             # MSADecoderDataset, LatentFMDataset
│   └── preprocessing.py       # LMDB builder (Protenix + ESM2 embeddings)
├── models/
│   ├── sfm_decoder.py         # SFM decoder (DiT + position-wise AdaLN)
│   └── latent_fm.py           # Latent FM encoder + SDE sampler
├── training/
│   ├── train_decoder.py       # Decoder training loop (Accelerate)
│   └── train_latent_fm.py     # Latent FM training loop (Accelerate)
├── inference/
│   └── generate.py            # Three-mode generation + CLI
└── utils/
    └── spherical.py           # Spherical exp/log maps, geodesic interpolation

esm/                           # Meta ESM2 (submodule)
Protenix/                      # ByteDance Protenix / AF3 (submodule)
LFM/                           # Latent Flow Matching reference (submodule)
```

---

## Installation

```bash
git clone https://github.com/kimsukimsu/MSA_GENERATION.git
cd MSA_GENERATION

# Core dependencies
pip install torch torchvision torchaudio
pip install accelerate omegaconf lmdb tqdm
pip install fair-esm                     # ESM2

# Protenix (AF3 MSAModule)
cd Protenix && pip install -e . && cd ..
```

---

## Quick Start

### 1. Build the LMDB dataset

Preprocesses OpenFold A3M files: extracts Protenix MSA embeddings (L×128) and ESM2 query embeddings (L×1280).

```bash
python msaflow/data/preprocessing.py \
  --a3m_dir      /data/openfold_a3m \
  --output       /data/msaflow.lmdb \
  --protenix_checkpoint /models/protenix.pt \
  --max_msa_seqs 512 \
  --max_seq_len  1024 \
  --device       cuda
```

### 2. Train the SFM decoder

```bash
accelerate launch \
  --config_file msaflow/configs/accelerate_4gpu.yaml \
  msaflow/training/train_decoder.py \
  --config      msaflow/configs/decoder.yaml \
  --lmdb_path   /data/msaflow.lmdb \
  --output_dir  /runs/decoder
```

Training settings (paper Section 6.8.2):
- 7 epochs · LR 1e-5 · warmup 5000 steps · weight decay 0.1
- BF16 mixed precision · EMA decay 0.9999

### 3. Train the latent FM encoder

```bash
accelerate launch \
  --config_file msaflow/configs/accelerate_4gpu.yaml \
  msaflow/training/train_latent_fm.py \
  --config      msaflow/configs/latent_fm.yaml \
  --lmdb_path   /data/msaflow.lmdb \
  --output_dir  /runs/latent_fm
```

Training settings:
- 15 epochs · LR 2.6e-4 · warmup 3000 steps · weight decay 0.1

### 4. Inference

**Zero-shot** (single sequence → synthetic MSA):
```bash
python msaflow/inference/generate.py \
  --mode          zeroshot \
  --query_seq     MKTAYIAKQRQISFVKSHFSRQ... \
  --decoder_ckpt  /runs/decoder/decoder_ema_final.pt \
  --latent_fm_ckpt /runs/latent_fm/latent_fm_ema_final.pt \
  --n_seqs        32 \
  --output        generated.fasta
```

**Reconstruction** (deep MSA → compressed → reconstructed):
```bash
python msaflow/inference/generate.py \
  --mode           reconstruct \
  --input_a3m      query.a3m \
  --decoder_ckpt   /runs/decoder/decoder_ema_final.pt \
  --protenix_ckpt  /models/protenix.pt \
  --n_seqs         32 \
  --output         reconstructed.a3m
```

**Few-shot augmentation** (shallow MSA → augmented):
```bash
python msaflow/inference/generate.py \
  --mode            augment \
  --input_a3m       shallow.a3m \
  --decoder_ckpt    /runs/decoder/decoder_ema_final.pt \
  --latent_fm_ckpt  /runs/latent_fm/latent_fm_ema_final.pt \
  --protenix_ckpt   /models/protenix.pt \
  --output          augmented.a3m
```

---

## Key Results (from paper)

### MSA Autoencoding (CAMEO benchmark, 50 proteins)

| Method | pLDDT | TM-score | Storage |
|--------|-------|----------|---------|
| No MSA | ~47   | 0.33     | —       |
| Ground Truth (7k seqs) | **91.6** | **0.89** | 100% |
| **MSAFlow Recon (32 seqs)** | **89.0** | **0.86** | **6.5%** |
| MSAFlow Zero-shot | 75–80 | 0.62–0.70 | — |

### MSA Augmentation (zero-shot, 200 proteins from CAMEO/CASP/PDB)

| Method | pLDDT | TM-score |
|--------|-------|----------|
| No/Shallow MSA | 73.1 | 0.55 |
| EvoDiff (650M) | 67.7 | 0.49 |
| MSAGPT (3B) | 71.6 | 0.53 |
| **MSAFlow (130M)** | **75.2** | **0.62** |

### Enzyme Design (EC classes, <20 training sequences)

| Method | Acc×Uniqueness (Fixed) | Acc×Uniqueness (Variable) |
|--------|------------------------|---------------------------|
| ProfileBFN | 42–100% | N/A |
| MSAGPT | — | 15–38% |
| **MSAFlow** | **83–100%** | **51–92%** |

---

## Mathematical Background

### Statistical Flow Matching (SFM)

MSAFlow operates on the **Fisher-Rao statistical manifold** via the isometric mapping:

```
π : µ → x = √µ    (probability simplex → unit sphere)
π⁻¹: x → µ = x²  (unit sphere → probability simplex)
```

**Training** interpolates along spherical geodesics:
```
x_t = exp_{x₀}(t · log_{x₀}(x₁))     t ∈ [0,1]
```

**Target velocity** (points from xₜ toward x₁):
```
u_t(xₜ | x₁) = log_{xₜ}(x₁) / (1 − t)
```

**Loss**:
```
L_SFM(θ) = E_{t, sᵢ, µ₀}[ ‖v_θ(xₜ | m, t) − uₜ‖² ]
```

### Latent Rectified Flow

Straight-line interpolation in R^(L×128):
```
z_t = (1−t)·z₁ + t·z₀,   z₀ ~ N(0, I)
```

```
L_RFM(θ) = E_{t, z₀, z₁}[ ‖v_θ(zₜ, e, t) − (z₀ − z₁)‖² ]
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| PyTorch | ≥2.1 | Core deep learning |
| Accelerate | ≥0.26 | Distributed training |
| OmegaConf | ≥2.3 | Configuration |
| lmdb | ≥1.4 | Dataset storage |
| fair-esm | — | ESM2-650M embeddings |
| Protenix | — | AF3 MSAModule encoder |

---

## Citation

```bibtex
@article{msaflow2026,
  title   = {MSAFlow: A Unified Approach for MSA Representation, Augmentation, and Family-based Protein Design},
  author  = {Anonymous},
  journal = {ICLR 2026 (under review)},
  year    = {2026}
}
```

---

## Acknowledgements

- [Protenix](https://github.com/bytedance/protenix) — AF3 MSAModule implementation
- [ESM](https://github.com/facebookresearch/esm) — ESM2 protein language model
- [LFM](https://github.com/FieteLab/LFM) — Latent Flow Matching reference implementation
- [Statistical Flow Matching (SFM)](https://arxiv.org/abs/2405.16441) — Cheng et al., 2024

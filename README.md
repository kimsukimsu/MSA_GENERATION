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
│   ├── decoder.yaml             # SFM decoder training config
│   ├── latent_fm.yaml           # Latent FM training config
│   ├── accelerate_2gpu.yaml     # 2-GPU distributed training config
│   └── accelerate_4gpu.yaml     # 4-GPU distributed training config
├── data/
│   ├── dataset.py               # MSADecoderDataset, LatentFMDataset
│   └── preprocessing.py         # LMDB builder (Protenix + ESM2 embeddings)
├── models/
│   ├── sfm_decoder.py           # SFM decoder (DiT + position-wise AdaLN)
│   └── latent_fm.py             # Latent FM encoder + SDE sampler
├── training/
│   ├── train_decoder.py         # Decoder training loop (Accelerate)
│   └── train_latent_fm.py       # Latent FM training loop (Accelerate)
├── inference/
│   └── generate.py              # Three-mode generation + CLI
└── utils/
    └── spherical.py             # Spherical exp/log maps, geodesic interpolation

scripts/
├── preprocess.sh                # SLURM: LMDB 전처리 (GPU 1개)
├── train_decoder.sh             # SLURM: SFM decoder 학습 (GPU 2개)
└── train_latent_fm.sh           # SLURM: Latent FM 학습 (GPU 2개)

esm/                             # Meta ESM2 (submodule)
Protenix/                        # ByteDance Protenix / AF3 (submodule)
LFM/                             # Latent Flow Matching reference (submodule)
```

---

## Installation

```bash
git clone https://github.com/DeepFoldProtein/MSA_FLOW.git
cd MSA_FLOW

# uv로 가상환경 생성 및 의존성 설치
uv venv
uv sync

# pre-commit 훅 설정
pre-commit install

# Protenix (AF3 MSAModule)
cd Protenix && pip install -e . && cd ..
```

---

## Training Pipeline

전처리부터 학습까지 3단계로 구성됩니다.

```
A3M 파일 (uniclust30)
    ↓  Step 1. preprocess.sh
msaflow.lmdb
    ↓  Step 2. train_decoder.sh
decoder_ema_final.pt
    ↓  Step 3. train_latent_fm.sh
latent_fm_ema_final.pt
```

### Step 1. 전처리 — LMDB 빌드

OpenFold A3M 파일을 읽어 Protenix MSA embedding (L×128)과 ESM2 query embedding (L×1280)을 LMDB로 저장합니다.

```bash
# 경로를 환경변수로 지정
A3M_DIR=/store/database/openfold/uniclust30 \
OUTPUT_LMDB=/store/msaflow.lmdb \
PROTENIX_CKPT=/path/to/protenix.ckpt \
sbatch scripts/preprocess.sh
```

> `PROTENIX_CKPT` 생략 시 ESM2 embedding만 저장됩니다 (decoder 학습에는 Protenix embedding이 필요합니다).

### Step 2. SFM Decoder 학습

```bash
LMDB_PATH=/store/msaflow.lmdb \
OUTPUT_DIR=/store/runs/decoder \
sbatch scripts/train_decoder.sh
```

| 설정 | 값 |
|------|-----|
| epochs | 7 |
| LR | 1e-5 |
| warmup steps | 5000 |
| weight decay | 0.1 |
| precision | BF16 |
| EMA decay | 0.9999 |

### Step 3. Latent FM 학습

```bash
LMDB_PATH=/store/msaflow.lmdb \
OUTPUT_DIR=/store/runs/latent_fm \
sbatch scripts/train_latent_fm.sh
```

| 설정 | 값 |
|------|-----|
| epochs | 15 |
| LR | 2.6e-4 |
| warmup steps | 3000 |
| weight decay | 0.1 |
| precision | BF16 |
| EMA decay | 0.9999 |

### SLURM 의존성 체인 (자동 순서 실행)

```bash
JID1=$(sbatch --parsable scripts/preprocess.sh)
JID2=$(sbatch --parsable --dependency=afterok:$JID1 scripts/train_decoder.sh)
sbatch --dependency=afterok:$JID2 scripts/train_latent_fm.sh
```

---

## Quick Start (SLURM 없이 직접 실행)

### 1. Build the LMDB dataset

```bash
python -m msaflow.data.preprocessing \
  --a3m_dir      /store/database/openfold/uniclust30 \
  --output       /store/msaflow.lmdb \
  --protenix_checkpoint /models/protenix.ckpt \
  --max_msa_seqs 512 \
  --max_seq_len  1024 \
  --device       cuda
```

### 2. Train the SFM decoder

```bash
accelerate launch \
  --config_file msaflow/configs/accelerate_2gpu.yaml \
  msaflow/training/train_decoder.py \
  --config      msaflow/configs/decoder.yaml \
  --lmdb_path   /store/msaflow.lmdb \
  --output_dir  /runs/decoder
```

### 3. Train the latent FM encoder

```bash
accelerate launch \
  --config_file msaflow/configs/accelerate_2gpu.yaml \
  msaflow/training/train_latent_fm.py \
  --config      msaflow/configs/latent_fm.yaml \
  --lmdb_path   /store/msaflow.lmdb \
  --output_dir  /runs/latent_fm
```

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

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Git & Commit Guidelines (DeepFold Team Standard)

### Commit Message Format

```
type: Subject

body

footer
```

**Type** (required, one of):
- `feat` — 새로운 기능
- `fix` — 오류 수정
- `dep` — 의존성 수정 (pyproject.toml, uv.lock 등)
- `ci` — CI/CD 관련
- `refactor` — 리팩토링
- `style` — 코드 스타일 수정 (로직 변경 없음)
- `docs` — 문서 추가/수정/삭제
- `test` — 테스트 코드
- `chore` — 기타 변경사항 (빌드 스크립트 등)

**Subject:** 50자 이하, 영문 첫 글자 대문자

**Body (optional):** 무엇을, 왜 변경했는지. 각 줄 72자 이하

**Footer (optional):** `Closes #123` 형식의 Issue tracker ID

### Development Environment

```bash
uv venv           # 가상 환경 생성
uv sync           # 의존성 설치
pre-commit install  # pre-commit 훅 설정
```

### Large Files

100MB 이상 파일(데이터셋, 모델 가중치)은 Git에 커밋하지 않는다.
DVC 또는 외부 클라우드 스토리지(S3, GDrive 등) 사용.

### Remotes

- `origin` — `kimsukimsu/MSA_GENERATION` (개인 포크)
- `deepfold` — `DeepFoldProtein/MSA_FLOW` (팀 공식 저장소)

---

## Repository Overview

This workspace contains 4 independent research/production projects for protein structure prediction and generative modeling. They are **not** a monorepo — each subdirectory is a standalone project with its own dependencies.

| Project | Purpose |
|---------|---------|
| `esm/` | Meta's protein language models (ESM-2, ESMFold, MSA Transformer, inverse folding) |
| `LFM/` | Flow Matching in Latent Space — generative diffusion models for image synthesis |
| `Protenix/` | ByteDance's AlphaFold-3-inspired biomolecular structure prediction system |
| `claude-code-infrastructure-showcase/` | Reference library for Claude Code automation (agents, skills, hooks) |

---

## ESM

**Install:**
```bash
cd esm && python setup.py install
```

**Entry points:**
```bash
esm-extract    # Extract protein embeddings from FASTA
esm-fold       # Run ESMFold structure prediction
```

**Linting:** `.flake8` configured (280 char line limit), `pyproject.toml` uses Black at 99 chars.

**Architecture:** Protein sequences are tokenized via `esm/data.py` (`Alphabet`, `BatchConverter`), then passed through `ESM2` (33 transformer layers, 1280 embed dim) defined in `esm/model/esm2.py`. ESMFold (`esm/esmfold/`) wraps ESM2 with an OmegaFold-style structure module. Inverse folding (`esm/inverse_folding/`) uses a Geometric Vector Perceptron encoder (`gvp_encoder.py`) to design sequences from 3D structure. PyTorch Hub integration is in `hubconf.py`.

---

## LFM (Latent Flow Matching)

**Install:**
```bash
cd LFM && pip install lmdb diffusers torchdiffeq ml_collections omegaconf timm
```

**Train:**
```bash
bash bash_scripts/run.sh
```

**Test/sample:**
```bash
bash bash_scripts/run_test.sh <path_to_arg_file>
bash bash_scripts/run_test_cls.sh <path_to_arg_file>    # classifier-guided
```

**Architecture:** Images pass through a VAE encoder (`models/encoder.py`) into latent space. A Diffusion Transformer (`models/DiT.py`) or EDM (`models/EDM.py`) is trained with flow matching objectives. At inference, ODE solvers from `torchdiffeq` (`sampler/`) integrate the learned velocity field back to clean latents, decoded by the VAE. Conditioning (labels, masks) is handled via `models/x_transformer.py` and `datasets_prep/conditional_builder.py`. Training uses Accelerate + EMA (`EMA.py`).

---

## Protenix

**Install:**
```bash
cd Protenix
pip install protenix
pip install -r requirements.txt
```

**Inference (CLI):**
```bash
protenix pred -i examples/input.json -o ./output -n protenix_base_default_v1.0.0
```

**Inference (Python):**
```bash
python runner/inference.py --model_name protenix_base_default_v1.0.0 --input_json_path examples/input.json
```

See `inference_demo.sh` for 8 complete examples covering all model variants, RNA MSA, atom-level constraints, and TF32 acceleration.

**Training / fine-tuning:**
```bash
bash train_demo.sh
bash finetune_demo.sh
```

**Linting:**
```bash
flake8 .    # configured via .flake8
```

**Model variants:**
- `protenix_base_default_v1.0.0` — recommended for benchmarking (368M params)
- `protenix_base_20250630_v1.0.0` — production, updated data cutoff
- `protenix_mini_*` / `protenix_tiny_*` — lighter variants for fast iteration

**Architecture:** Input JSON (sequences, optional MSA/template/constraints) → feature extraction (`protenix/data/`) including ESM2 embeddings (`protenix/data/esm/`), HMMer/Kalign MSA (`protenix/data/msa/`), and template features (`protenix/data/template/`) → Pairformer with triangular multiplicative updates (`protenix/model/triangular/`) + triangle attention (`protenix/model/tri_attention/`, optionally using cuequivariance CUDA kernels) over N cycles → diffusion refinement → PDB output. Custom CUDA layer norm lives in `protenix/model/layer_norm/`. Config is OmegaConf-based (`protenix/config/`). The web service API is in `protenix/web_service/`. Evaluation metrics are in `protenix/metrics/` (PXMeter).

Protenix depends on ESM — `fair-esm` is in its `requirements.txt`.

---

## Claude Code Infrastructure Showcase

Reference implementation for Claude Code automation patterns. See `claude-code-infrastructure-showcase/README.md` and `CLAUDE_INTEGRATION_GUIDE.md` for full documentation.

Key components:
- `.claude/agents/` — 10 specialized subagents (code review, refactoring, docs, etc.)
- `.claude/skills/` — 5 modular skills (backend, frontend, route-tester, etc.)
- `.claude/hooks/` — 6 hooks including `skill-activation-prompt.sh` (UserPromptSubmit) and `post-tool-use-tracker.sh` (PostToolUse)
- `.claude/settings.json` — hook registration

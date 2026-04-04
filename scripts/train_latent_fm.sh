#!/bin/bash
#SBATCH --job-name=msaflow-latent-fm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu          # 클러스터에 맞게 수정
#SBATCH --output=logs/latent_fm_%j.out
#SBATCH --error=logs/latent_fm_%j.err

# ── 경로 설정 (환경에 맞게 수정) ──────────────────────────────────────────────
REPO_DIR=${REPO_DIR:-$HOME/MSA_Generation}
LMDB_PATH=${LMDB_PATH:-/data/msaflow.lmdb}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/runs/latent_fm}
CONFIG=$REPO_DIR/msaflow/configs/latent_fm.yaml
ACCEL_CONFIG=$REPO_DIR/msaflow/configs/accelerate_2gpu.yaml

# ── 환경 활성화 ────────────────────────────────────────────────────────────────
source $REPO_DIR/.venv/bin/activate

# ── NCCL 튜닝 ─────────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=8

# ── 로그 디렉터리 생성 ─────────────────────────────────────────────────────────
mkdir -p $OUTPUT_DIR logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Output dir : $OUTPUT_DIR"
echo "LMDB       : $LMDB_PATH"
date

# ── 학습 실행 ──────────────────────────────────────────────────────────────────
accelerate launch \
    --config_file $ACCEL_CONFIG \
    $REPO_DIR/msaflow/training/train_latent_fm.py \
    --config $CONFIG \
    --lmdb_path $LMDB_PATH \
    --output_dir $OUTPUT_DIR

echo "Done: $(date)"

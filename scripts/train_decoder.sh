#!/bin/bash
#SBATCH --job-name=msaflow-decoder
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=gpu          # 클러스터에 맞게 수정
#SBATCH --output=logs/decoder_%j.out
#SBATCH --error=logs/decoder_%j.err

# ── Python 모듈 로드 ───────────────────────────────────────────────────────────
module load python/3.11.14

# ── uv PATH 설정 ──────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"

# ── 경로 설정 (환경에 맞게 수정) ──────────────────────────────────────────────
REPO_DIR=${REPO_DIR:-$HOME/projects/MSA_FLOW}
LMDB_PATH=${LMDB_PATH:-/store/msaflow.lmdb}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/runs/decoder}
CONFIG=$REPO_DIR/msaflow/configs/decoder.yaml
ACCEL_CONFIG=$REPO_DIR/msaflow/configs/accelerate_2gpu.yaml

# ── 환경 활성화 ────────────────────────────────────────────────────────────────
source $REPO_DIR/.venv/bin/activate

# ── NCCL 튜닝 ─────────────────────────────────────────────────────────────────
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export OMP_NUM_THREADS=8

# ── 로그 디렉터리 생성 ─────────────────────────────────────────────────────────
mkdir -p $OUTPUT_DIR $REPO_DIR/logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Python     : $(python --version)"
echo "GPUs       : $CUDA_VISIBLE_DEVICES"
echo "Output dir : $OUTPUT_DIR"
echo "LMDB       : $LMDB_PATH"
date

# ── 학습 실행 ──────────────────────────────────────────────────────────────────
accelerate launch \
    --config_file $ACCEL_CONFIG \
    $REPO_DIR/msaflow/training/train_decoder.py \
    --config $CONFIG \
    --lmdb_path $LMDB_PATH \
    --output_dir $OUTPUT_DIR

echo "Done: $(date)"

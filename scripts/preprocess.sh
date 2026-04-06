#!/bin/bash
#SBATCH --job-name=msaflow-preprocess
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --partition=gpu          # 클러스터에 맞게 수정
#SBATCH --output=logs/preprocess_%j.out
#SBATCH --error=logs/preprocess_%j.err

# ── Python 모듈 로드 ───────────────────────────────────────────────────────────
module load python/3.11.14

# ── uv PATH 설정 ──────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"

# ── 경로 설정 (환경에 맞게 수정) ──────────────────────────────────────────────
REPO_DIR=${REPO_DIR:-$HOME/projects/MSA_FLOW}
A3M_DIR=${A3M_DIR:-/store/database/openfold/uniclust30}
OUTPUT_LMDB=${OUTPUT_LMDB:-/store/msaflow.lmdb}
PROTENIX_CKPT=${PROTENIX_CKPT:-""}   # 없으면 ESM2만 추출

# ── 환경 활성화 ────────────────────────────────────────────────────────────────
source $REPO_DIR/.venv/bin/activate

# ── 로그 디렉터리 생성 ─────────────────────────────────────────────────────────
mkdir -p $REPO_DIR/logs

echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Python     : $(python --version)"
echo "A3M dir    : $A3M_DIR"
echo "Output     : $OUTPUT_LMDB"
echo "Protenix   : ${PROTENIX_CKPT:-'(skipped)'}"
date

# ── 전처리 실행 ────────────────────────────────────────────────────────────────
CKPT_ARG=""
if [ -n "$PROTENIX_CKPT" ]; then
    CKPT_ARG="--protenix_checkpoint $PROTENIX_CKPT"
fi

python -m msaflow.data.preprocessing \
    --a3m_dir $A3M_DIR \
    --output $OUTPUT_LMDB \
    $CKPT_ARG \
    --max_msa_seqs 512 \
    --max_seq_len 1024 \
    --device cuda \
    --map_size_gb 500

echo "Done: $(date)"

#!/bin/bash
#SBATCH --job-name=msaflow-all
#SBATCH --nodes=1
#SBATCH --nodelist=ada-001
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/all_%j.out
#SBATCH --error=logs/all_%j.err

# ── 4-GPU 동시 실행 ────────────────────────────────────────────────────────────
# GPU 0: nomsa      (생성 없음, 가장 빠름)
# GPU 1: colabfold  (생성 없음, ColabFold MSA 그대로 fold)
# GPU 2: zeroshot   (MSAFlow zero-shot 생성 + fold)
# GPU 3: fewshot    (ColabFold MSA + MSAFlow Rec/Syn 보강 + fold)

module load python/3.11.14
module load cuda/13.0.2

export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.local/bin:$PATH"

REPO_DIR=${REPO_DIR:-/home/paul3875/projects/MSA_FLOW}
DECODER_CKPT=${DECODER_CKPT:-/gpfs/deepfold/users/yjlee4/decoder/latest.pt}
LATENT_FM_CKPT=${LATENT_FM_CKPT:-$REPO_DIR/runs/latent_fm/latent_fm_ema_final.pt}
PROTENIX_CKPT=${PROTENIX_CKPT:-$REPO_DIR/checkpoint/protenix_base_default_v1.0.0.pt}
FASTA=${FASTA:-$REPO_DIR/data/foldbench_monomer.fasta}
export BASE_DIR=${BASE_DIR:-$REPO_DIR/runs/benchmark_all}
SHALLOW_MSA_DIR=${SHALLOW_MSA_DIR:-/store/deepfold3/results/msa/a3m}
REF_CIF_DIR=${REF_CIF_DIR:-/gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520}
NEFF_CSV=${NEFF_CSV:-$REPO_DIR/data/foldbench_groups/neff_scores.csv}
USALIGN_BIN=${USALIGN_BIN:-USalign}

PROTENIX_MODEL=${PROTENIX_MODEL:-protenix_base_default_v1.0.0}
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-$REPO_DIR}

N_SEQS=${N_SEQS:-32}
N_SEEDS=${N_SEEDS:-5}
N_STEPS=${N_STEPS:-100}
TEMPERATURE=${TEMPERATURE:-0.5}
MAX_REC_DEPTH=${MAX_REC_DEPTH:-128}

source $REPO_DIR/.venv/bin/activate
export PYTHONPATH=$REPO_DIR/Protenix:$PYTHONPATH

mkdir -p $BASE_DIR $REPO_DIR/logs
for MODE in nomsa colabfold zeroshot fewshot; do
    mkdir -p $BASE_DIR/$MODE
done

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Base dir : $BASE_DIR"
echo "FASTA    : $FASTA  ($(grep -c '^>' $FASTA) proteins)"
date

# ── USalign 빌드 (없으면) ──────────────────────────────────────────────────────
if ! command -v $USALIGN_BIN &> /dev/null; then
    echo "Building USalign..."
    wget -q https://zhanggroup.org/US-align/bin/module/USalign.cpp -O /tmp/USalign.cpp
    g++ -O3 -ffast-math -lm -o $HOME/.local/bin/USalign /tmp/USalign.cpp
    USALIGN_BIN=$HOME/.local/bin/USalign
fi

COMMON="
    --fasta          $FASTA
    --protenix_model $PROTENIX_MODEL
    --ref_cif_dir    $REF_CIF_DIR
    --usalign_bin    $USALIGN_BIN
    --device         cuda
    --num_shards     1
    --shard_id       0
"

# ── GPU 0: nomsa ───────────────────────────────────────────────────────────────
echo "=== [GPU 0] nomsa ==="
CUDA_VISIBLE_DEVICES=0 python $REPO_DIR/msaflow/inference/fold_benchmark.py \
    $COMMON \
    --decoder_ckpt   $LATENT_FM_CKPT \
    --latent_fm_ckpt $LATENT_FM_CKPT \
    --output_dir     $BASE_DIR/nomsa \
    --mode           nomsa \
    > $BASE_DIR/nomsa/run.log 2>&1 &

# ── GPU 1: colabfold ───────────────────────────────────────────────────────────
echo "=== [GPU 1] colabfold ==="
CUDA_VISIBLE_DEVICES=1 python $REPO_DIR/msaflow/inference/fold_benchmark.py \
    $COMMON \
    --decoder_ckpt   $LATENT_FM_CKPT \
    --latent_fm_ckpt $LATENT_FM_CKPT \
    --output_dir     $BASE_DIR/colabfold \
    --mode           colabfold \
    --shallow_msa_dir $SHALLOW_MSA_DIR \
    > $BASE_DIR/colabfold/run.log 2>&1 &

# ── GPU 2: zeroshot ────────────────────────────────────────────────────────────
echo "=== [GPU 2] zeroshot ==="
CUDA_VISIBLE_DEVICES=2 python $REPO_DIR/msaflow/inference/fold_benchmark.py \
    $COMMON \
    --decoder_ckpt   $DECODER_CKPT \
    --latent_fm_ckpt $LATENT_FM_CKPT \
    --output_dir     $BASE_DIR/zeroshot \
    --mode           zeroshot \
    --n_seqs         $N_SEQS \
    --n_seeds        $N_SEEDS \
    --n_steps        $N_STEPS \
    --temperature    $TEMPERATURE \
    > $BASE_DIR/zeroshot/run.log 2>&1 &

# ── GPU 3: fewshot ─────────────────────────────────────────────────────────────
echo "=== [GPU 3] fewshot ==="
CUDA_VISIBLE_DEVICES=3 python $REPO_DIR/msaflow/inference/fold_benchmark.py \
    $COMMON \
    --decoder_ckpt   $DECODER_CKPT \
    --latent_fm_ckpt $LATENT_FM_CKPT \
    --output_dir     $BASE_DIR/fewshot \
    --mode           fewshot \
    --n_seqs         $N_SEQS \
    --n_seeds        $N_SEEDS \
    --n_steps        $N_STEPS \
    --temperature    $TEMPERATURE \
    --shallow_msa_dir $SHALLOW_MSA_DIR \
    --protenix_ckpt  $PROTENIX_CKPT \
    --max_rec_depth  $MAX_REC_DEPTH \
    > $BASE_DIR/fewshot/run.log 2>&1 &

echo "All 4 modes launched (PIDs: $(jobs -p))"
wait
echo "All done: $(date)"

# ── 비교 분석 ──────────────────────────────────────────────────────────────────
echo "=== Running comparison analysis ==="
python $REPO_DIR/scripts/analyze_benchmark.py \
    --base_dir  $BASE_DIR \
    --neff_csv  $NEFF_CSV \
    | tee $BASE_DIR/comparison.txt

echo "Comparison saved → $BASE_DIR/comparison.txt"

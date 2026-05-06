#!/bin/bash
#SBATCH --job-name=msaflow-all
#SBATCH --nodes=1
#SBATCH --nodelist=ada-003
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=72:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/all_%j.out
#SBATCH --error=logs/all_%j.err

# ── Neff 그룹별 현실적 비교 설계 ──────────────────────────────────────────────
#
# orphan  (Neff ≤ 10)   : MSA 자체가 없는 상황
#                          nomsa  vs  zeroshot
#
# shallow (10 < Neff ≤ 67.1) : MSA가 얕게 있는 상황
# full    (Neff > 67.1)       : MSA가 풍부한 상황
#                          colabfold  vs  fewshot
#
# Stage 순서 (각 stage는 NUM_SHARDS GPU 병렬):
#   1. nomsa     — foldbench_orphan.fasta
#   2. zeroshot  — foldbench_orphan.fasta
#   3. colabfold — foldbench_shallow.fasta + foldbench_full.fasta
#   4. fewshot   — foldbench_shallow.fasta + foldbench_full.fasta

module load python/3.11.14
module load cuda/13.0.2

export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.local/bin:$PATH"

REPO_DIR=${REPO_DIR:-/home/paul3875/projects/MSA_FLOW}
DECODER_CKPT=${DECODER_CKPT:-/gpfs/deepfold/users/yjlee4/decoder/latest.pt}
LATENT_FM_CKPT=${LATENT_FM_CKPT:-$REPO_DIR/runs/latent_fm/latent_fm_ema_final.pt}
PROTENIX_CKPT=${PROTENIX_CKPT:-$REPO_DIR/checkpoint/protenix_base_default_v1.0.0.pt}
export BASE_DIR=${BASE_DIR:-$REPO_DIR/runs/benchmark_all}
FASTA_DIR=${FASTA_DIR:-$REPO_DIR/data/foldbench_groups}
SHALLOW_MSA_DIR=${SHALLOW_MSA_DIR:-/store/deepfold3/results/msa/a3m}
REF_CIF_DIR=${REF_CIF_DIR:-/gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520}
NEFF_CSV=${NEFF_CSV:-$FASTA_DIR/neff_scores.csv}
USALIGN_BIN=${USALIGN_BIN:-USalign}

PROTENIX_MODEL=${PROTENIX_MODEL:-protenix_base_default_v1.0.0}
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-$REPO_DIR}

N_SEQS=${N_SEQS:-32}
N_SEEDS=${N_SEEDS:-5}
N_STEPS=${N_STEPS:-100}
TEMPERATURE=${TEMPERATURE:-0.5}
MAX_REC_DEPTH=${MAX_REC_DEPTH:-128}

# 그룹별 FASTA
ORPHAN_FASTA=$FASTA_DIR/foldbench_orphan.fasta
SHALLOW_FASTA=$FASTA_DIR/foldbench_shallow.fasta
FULL_FASTA=$FASTA_DIR/foldbench_full.fasta

source $REPO_DIR/.venv/bin/activate
export PYTHONPATH=$REPO_DIR/Protenix:$PYTHONPATH

mkdir -p $BASE_DIR $REPO_DIR/logs
for MODE in nomsa zeroshot colabfold fewshot; do
    mkdir -p $BASE_DIR/$MODE
done

# shallow + full 합친 FASTA 생성
SHALLOW_FULL_FASTA=$BASE_DIR/foldbench_shallow_full.fasta
cat $SHALLOW_FASTA $FULL_FASTA > $SHALLOW_FULL_FASTA

echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Base dir : $BASE_DIR"
echo "orphan   : $(grep -c '^>' $ORPHAN_FASTA) proteins"
echo "shallow  : $(grep -c '^>' $SHALLOW_FASTA) proteins"
echo "full     : $(grep -c '^>' $FULL_FASTA) proteins"
echo "shallow+full: $(grep -c '^>' $SHALLOW_FULL_FASTA) proteins"
date

# ── USalign 빌드 (없으면) ──────────────────────────────────────────────────────
if ! command -v $USALIGN_BIN &> /dev/null; then
    echo "Building USalign..."
    wget -q https://zhanggroup.org/US-align/bin/module/USalign.cpp -O /tmp/USalign.cpp
    g++ -O3 -ffast-math -lm -o $HOME/.local/bin/USalign /tmp/USalign.cpp
    USALIGN_BIN=$HOME/.local/bin/USalign
fi

# ── shard 병합 함수 ────────────────────────────────────────────────────────────
merge_shards() {
    local mode_dir=$1
    python - << PYEOF
import csv, glob, os, math
output_dir = "$mode_dir"
rows, header = [], None
for shard_csv in sorted(glob.glob(f"{output_dir}/shard_*.csv")):
    with open(shard_csv) as fh:
        reader = csv.DictReader(fh)
        if header is None:
            header = reader.fieldnames
        rows.extend(reader)
if rows:
    out_path = f"{output_dir}/benchmark_results.csv"
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    tm_vals = [float(r["tm_score"]) for r in rows
               if r.get("tm_score") not in ("", "nan", None)
               and not math.isnan(float(r["tm_score"]))]
    print(f"  merged {len(rows)} rows → {out_path}")
    if tm_vals:
        print(f"  TM-score  n={len(tm_vals)}  mean={sum(tm_vals)/len(tm_vals):.4f}")
else:
    print(f"  No shard CSVs found in {output_dir}")
PYEOF
}

# ── SLURM GPU 할당 파싱 + 불량 GPU 제거 ──────────────────────────────────────
IFS=',' read -ra SLURM_GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1}"
echo "SLURM allocated GPUs: ${SLURM_GPUS[*]}"

HEALTHY_GPUS=()
for _GPU in "${SLURM_GPUS[@]}"; do
    if CUDA_VISIBLE_DEVICES=$_GPU python -c \
        "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        HEALTHY_GPUS+=("$_GPU")
        echo "  GPU $_GPU: OK"
    else
        echo "  GPU $_GPU: CUDA init failed — skipped"
    fi
done

if [ ${#HEALTHY_GPUS[@]} -eq 0 ]; then
    echo "ERROR: No healthy GPUs found. Exiting."
    exit 1
fi
NUM_SHARDS=${#HEALTHY_GPUS[@]}
echo "Using GPUs: ${HEALTHY_GPUS[*]}  (NUM_SHARDS=$NUM_SHARDS)"

gpu_for_shard() { echo "${HEALTHY_GPUS[$1]}"; }

is_done() { [ -s "$1/benchmark_results.csv" ]; }

# ── Stage 1: nomsa (orphan only) ───────────────────────────────────────────────
echo ""
echo "=== Stage 1/4: nomsa — orphan ($(date)) ==="
if is_done $BASE_DIR/nomsa; then
    echo "  already done — skipping"
else
    for SHARD_ID in $(seq 0 $((NUM_SHARDS-1))); do
        CUDA_VISIBLE_DEVICES=$(gpu_for_shard $SHARD_ID) \
        python $REPO_DIR/msaflow/inference/fold_benchmark.py \
            --fasta          $ORPHAN_FASTA \
            --decoder_ckpt   $LATENT_FM_CKPT \
            --latent_fm_ckpt $LATENT_FM_CKPT \
            --output_dir     $BASE_DIR/nomsa \
            --mode           nomsa \
            --protenix_model $PROTENIX_MODEL \
            --ref_cif_dir    $REF_CIF_DIR \
            --usalign_bin    $USALIGN_BIN \
            --device         cuda \
            --num_shards     $NUM_SHARDS \
            --shard_id       $SHARD_ID \
            > $BASE_DIR/nomsa/shard_${SHARD_ID}.log 2>&1 &
    done
    wait
    echo "  nomsa done: $(date)"
    merge_shards $BASE_DIR/nomsa
fi

# ── Stage 2: zeroshot (orphan only) ────────────────────────────────────────────
echo ""
echo "=== Stage 2/4: zeroshot — orphan ($(date)) ==="
if is_done $BASE_DIR/zeroshot; then
    echo "  already done — skipping"
else
    for SHARD_ID in $(seq 0 $((NUM_SHARDS-1))); do
        CUDA_VISIBLE_DEVICES=$(gpu_for_shard $SHARD_ID) \
        python $REPO_DIR/msaflow/inference/fold_benchmark.py \
            --fasta          $ORPHAN_FASTA \
            --decoder_ckpt   $DECODER_CKPT \
            --latent_fm_ckpt $LATENT_FM_CKPT \
            --output_dir     $BASE_DIR/zeroshot \
            --mode           zeroshot \
            --protenix_model $PROTENIX_MODEL \
            --ref_cif_dir    $REF_CIF_DIR \
            --usalign_bin    $USALIGN_BIN \
            --device         cuda \
            --num_shards     $NUM_SHARDS \
            --shard_id       $SHARD_ID \
            --n_seqs         $N_SEQS \
            --n_seeds        $N_SEEDS \
            --n_steps        $N_STEPS \
            --temperature    $TEMPERATURE \
            > $BASE_DIR/zeroshot/shard_${SHARD_ID}.log 2>&1 &
    done
    wait
    echo "  zeroshot done: $(date)"
    merge_shards $BASE_DIR/zeroshot
fi

# ── Stage 3: colabfold (shallow + full) ────────────────────────────────────────
echo ""
echo "=== Stage 3/4: colabfold — shallow+full ($(date)) ==="
if is_done $BASE_DIR/colabfold; then
    echo "  already done — skipping"
else
    for SHARD_ID in $(seq 0 $((NUM_SHARDS-1))); do
        CUDA_VISIBLE_DEVICES=$(gpu_for_shard $SHARD_ID) \
        python $REPO_DIR/msaflow/inference/fold_benchmark.py \
            --fasta           $SHALLOW_FULL_FASTA \
            --decoder_ckpt    $LATENT_FM_CKPT \
            --latent_fm_ckpt  $LATENT_FM_CKPT \
            --output_dir      $BASE_DIR/colabfold \
            --mode            colabfold \
            --protenix_model  $PROTENIX_MODEL \
            --shallow_msa_dir $SHALLOW_MSA_DIR \
            --ref_cif_dir     $REF_CIF_DIR \
            --usalign_bin     $USALIGN_BIN \
            --device          cuda \
            --num_shards      $NUM_SHARDS \
            --shard_id        $SHARD_ID \
            > $BASE_DIR/colabfold/shard_${SHARD_ID}.log 2>&1 &
    done
    wait
    echo "  colabfold done: $(date)"
    merge_shards $BASE_DIR/colabfold
fi

# ── Stage 4: fewshot (shallow + full) ──────────────────────────────────────────
echo ""
echo "=== Stage 4/4: fewshot — shallow+full ($(date)) ==="
if is_done $BASE_DIR/fewshot; then
    echo "  already done — skipping"
else
    for SHARD_ID in $(seq 0 $((NUM_SHARDS-1))); do
        CUDA_VISIBLE_DEVICES=$(gpu_for_shard $SHARD_ID) \
        python $REPO_DIR/msaflow/inference/fold_benchmark.py \
            --fasta           $SHALLOW_FULL_FASTA \
            --decoder_ckpt    $DECODER_CKPT \
            --latent_fm_ckpt  $LATENT_FM_CKPT \
            --output_dir      $BASE_DIR/fewshot \
            --mode            fewshot \
            --protenix_model  $PROTENIX_MODEL \
            --shallow_msa_dir $SHALLOW_MSA_DIR \
            --protenix_ckpt   $PROTENIX_CKPT \
            --max_rec_depth   $MAX_REC_DEPTH \
            --ref_cif_dir     $REF_CIF_DIR \
            --usalign_bin     $USALIGN_BIN \
            --device          cuda \
            --num_shards      $NUM_SHARDS \
            --shard_id        $SHARD_ID \
            --n_seqs          $N_SEQS \
            --n_seeds         $N_SEEDS \
            --n_steps         $N_STEPS \
            --temperature     $TEMPERATURE \
            > $BASE_DIR/fewshot/shard_${SHARD_ID}.log 2>&1 &
    done
    wait
    echo "  fewshot done: $(date)"
    merge_shards $BASE_DIR/fewshot
fi

# ── 비교 분석 ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Running comparison analysis ==="
python $REPO_DIR/scripts/analyze_benchmark.py \
    --base_dir  $BASE_DIR \
    --neff_csv  $NEFF_CSV \
    | tee $BASE_DIR/comparison.txt

echo "Comparison saved → $BASE_DIR/comparison.txt"
echo "All done: $(date)"

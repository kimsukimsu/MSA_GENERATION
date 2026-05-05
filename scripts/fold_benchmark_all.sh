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

# ── 4-GPU 순차 실행 ────────────────────────────────────────────────────────────
# 각 모드를 4-GPU shard로 병렬 실행한 뒤 다음 모드로 진행.
# GPU가 항상 100% 활용됨 (빠른 모드가 끝나도 놀지 않음).
#
# Stage 1: nomsa     (fold only, ~빠름)
# Stage 2: colabfold (fold only, ~빠름)
# Stage 3: zeroshot  (Syn 생성 + fold, ~중간)
# Stage 4: fewshot   (Rec+Syn 생성 + fold, ~느림)

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
NUM_SHARDS=4

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

# ── 이미 완료된 stage 체크 (resume용) ─────────────────────────────────────────
# benchmark_results.csv 가 존재하면 해당 stage 스킵.
# 중간에 실패했을 경우: 해당 모드 디렉토리의 shard_*.csv 와
# benchmark_results.csv 를 삭제한 뒤 재실행.
is_done() {
    local mode_dir=$1
    [ -s "$mode_dir/benchmark_results.csv" ]
}

# ── Stage 1: nomsa ─────────────────────────────────────────────────────────────
echo ""
echo "=== Stage 1/4: nomsa ($(date)) ==="
if is_done $BASE_DIR/nomsa; then
    echo "  already done — skipping (delete $BASE_DIR/nomsa/benchmark_results.csv to rerun)"
else
    for SHARD_ID in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD_ID \
        python $REPO_DIR/msaflow/inference/fold_benchmark.py \
            --fasta          $FASTA \
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

# ── Stage 2: colabfold ─────────────────────────────────────────────────────────
echo ""
echo "=== Stage 2/4: colabfold ($(date)) ==="
if is_done $BASE_DIR/colabfold; then
    echo "  already done — skipping (delete $BASE_DIR/colabfold/benchmark_results.csv to rerun)"
else
    for SHARD_ID in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD_ID \
        python $REPO_DIR/msaflow/inference/fold_benchmark.py \
            --fasta           $FASTA \
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

# ── Stage 3: zeroshot ──────────────────────────────────────────────────────────
echo ""
echo "=== Stage 3/4: zeroshot ($(date)) ==="
if is_done $BASE_DIR/zeroshot; then
    echo "  already done — skipping (delete $BASE_DIR/zeroshot/benchmark_results.csv to rerun)"
else
    for SHARD_ID in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD_ID \
        python $REPO_DIR/msaflow/inference/fold_benchmark.py \
            --fasta          $FASTA \
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

# ── Stage 4: fewshot ───────────────────────────────────────────────────────────
echo ""
echo "=== Stage 4/4: fewshot ($(date)) ==="
if is_done $BASE_DIR/fewshot; then
    echo "  already done — skipping (delete $BASE_DIR/fewshot/benchmark_results.csv to rerun)"
else
    for SHARD_ID in 0 1 2 3; do
        CUDA_VISIBLE_DEVICES=$SHARD_ID \
        python $REPO_DIR/msaflow/inference/fold_benchmark.py \
            --fasta           $FASTA \
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

#!/bin/bash
#SBATCH --job-name=msaflow-yjlee4-latest
#SBATCH --nodes=1
#SBATCH --nodelist=ada-001
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=28
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/yjlee4_latest_%j.out
#SBATCH --error=logs/yjlee4_latest_%j.err

module load python/3.11.14
module load cuda/13.0.2

export CUDA_HOME=${CUDA_HOME:-$(dirname $(dirname $(which nvcc)))}
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
export PATH="$HOME/.local/bin:$PATH"

REPO_DIR=${REPO_DIR:-/home/paul3875/projects/MSA_FLOW}
DECODER_CKPT=/gpfs/deepfold/users/yjlee4/decoder/latest.pt
LATENT_FM_CKPT=${LATENT_FM_CKPT:-$REPO_DIR/runs/latent_fm/latent_fm_ema_final.pt}
FASTA=${FASTA:-$REPO_DIR/data/foldbench_monomer.fasta}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/runs/fold_benchmark_yjlee4_latest}
REF_CIF_DIR=/gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520
USALIGN_BIN=${USALIGN_BIN:-USalign}

N_SEQS=${N_SEQS:-32}
N_SEEDS=${N_SEEDS:-5}
N_STEPS=${N_STEPS:-100}
TEMPERATURE=${TEMPERATURE:-0.5}

PROTENIX_MODEL=${PROTENIX_MODEL:-protenix_base_default_v1.0.0}
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-$REPO_DIR}

source $REPO_DIR/.venv/bin/activate
export PYTHONPATH=$REPO_DIR/Protenix:$PYTHONPATH

mkdir -p $OUTPUT_DIR $REPO_DIR/logs

echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Python        : $(python --version)"
echo "Decoder ckpt  : $DECODER_CKPT"
echo "Latent FM ckpt: $LATENT_FM_CKPT"
echo "FASTA         : $FASTA"
echo "Output dir    : $OUTPUT_DIR"
echo "Ref CIF dir   : $REF_CIF_DIR"
echo "n_seqs=$N_SEQS  n_seeds=$N_SEEDS  n_steps=$N_STEPS  temperature=$TEMPERATURE"
date

NUM_SHARDS=5

# ── MSAFlow zero-shot (5 GPU 병렬, GPU 1–5 할당) ─────────────────────────────
echo "=== Launching MSAFlow zero-shot shards ==="
for SHARD_ID in 0 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=$SHARD_ID \
    python $REPO_DIR/msaflow/inference/fold_benchmark.py \
        --fasta          $FASTA \
        --decoder_ckpt   $DECODER_CKPT \
        --latent_fm_ckpt $LATENT_FM_CKPT \
        --output_dir     $OUTPUT_DIR \
        --mode           zeroshot \
        --device         cuda \
        --n_seqs         $N_SEQS \
        --n_seeds        $N_SEEDS \
        --n_steps        $N_STEPS \
        --temperature    $TEMPERATURE \
        --protenix_model $PROTENIX_MODEL \
        --shard_id       $SHARD_ID \
        --num_shards     $NUM_SHARDS \
        > $OUTPUT_DIR/shard_${SHARD_ID}.log 2>&1 &
done

echo "Launched shards (PIDs: $(jobs -p))"
wait
echo "All shards done: $(date)"

# ── 결과 병합 ──────────────────────────────────────────────────────────────────
python - << 'PYEOF'
import csv, glob, os

output_dir = os.environ["OUTPUT_DIR"]
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
    print(f"Merged {len(rows)} rows → {out_path}")
else:
    print(f"No shard CSVs found in {output_dir}")
PYEOF

# ── USalign 없으면 빌드 ────────────────────────────────────────────────────────
if ! command -v $USALIGN_BIN &> /dev/null; then
    echo "USalign not found — building from source..."
    wget -q https://zhanggroup.org/US-align/bin/module/USalign.cpp -O /tmp/USalign.cpp
    g++ -static -O3 -ffast-math -lm -o $HOME/.local/bin/USalign /tmp/USalign.cpp
    USALIGN_BIN=$HOME/.local/bin/USalign
    echo "Built USalign at $USALIGN_BIN"
fi

# ── TM-score 계산 ──────────────────────────────────────────────────────────────
RESULTS_CSV=$OUTPUT_DIR/benchmark_results.csv
if [ -f "$RESULTS_CSV" ]; then
    echo "=== Computing TM-scores ==="
    python $REPO_DIR/scripts/compute_tmscore.py \
        --results_csv  $RESULTS_CSV \
        --fold_dir     $OUTPUT_DIR/folds \
        --ref_cif_dir  $REF_CIF_DIR \
        --usalign_bin  $USALIGN_BIN \
        --mode         zeroshot
else
    echo "ERROR: $RESULTS_CSV not found — fold benchmark may have failed"
    exit 1
fi

echo "All done: $(date)"

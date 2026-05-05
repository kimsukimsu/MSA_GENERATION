#!/bin/bash
#SBATCH --job-name=msaflow-fewshot
#SBATCH --nodes=1
#SBATCH --nodelist=ada-001
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/fewshot_%j.out
#SBATCH --error=logs/fewshot_%j.err

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
export OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/runs/fold_benchmark_fewshot}
SHALLOW_MSA_DIR=${SHALLOW_MSA_DIR:-/store/deepfold3/results/msa/a3m}
REF_CIF_DIR=${REF_CIF_DIR:-/gpfs/deepfold/users/paul3875/foldbench_ground_truths/ground_truth_20250520}
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

mkdir -p $OUTPUT_DIR $REPO_DIR/logs

echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Python        : $(python --version)"
echo "Decoder ckpt  : $DECODER_CKPT"
echo "Latent FM ckpt: $LATENT_FM_CKPT"
echo "Protenix ckpt : $PROTENIX_CKPT"
echo "FASTA         : $FASTA"
echo "Shallow MSA   : $SHALLOW_MSA_DIR"
echo "Output dir    : $OUTPUT_DIR"
echo "Ref CIF dir   : $REF_CIF_DIR"
echo "max_rec_depth : $MAX_REC_DEPTH"
echo "n_seqs=$N_SEQS  n_seeds=$N_SEEDS  n_steps=$N_STEPS  temperature=$TEMPERATURE"
date

# ── USalign 빌드 (없으면) ──────────────────────────────────────────────────────
if ! command -v $USALIGN_BIN &> /dev/null; then
    echo "USalign not found — building from source..."
    wget -q https://zhanggroup.org/US-align/bin/module/USalign.cpp -O /tmp/USalign.cpp
    g++ -O3 -ffast-math -lm -o $HOME/.local/bin/USalign /tmp/USalign.cpp
    USALIGN_BIN=$HOME/.local/bin/USalign
    echo "Built USalign at $USALIGN_BIN"
fi

NUM_SHARDS=4

# ── MSAFlow few-shot (4 GPU 병렬) ─────────────────────────────────────────────
# Rec track: ColabFold A3M → Protenix MSA encoder → 새 서열 생성
# Syn track: ESM2 → latent FM → 새 서열 생성
# 최종: original ColabFold MSA + 16개 MSAFlow 서열
echo "=== Launching MSAFlow few-shot shards ==="
for SHARD_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD_ID \
    python $REPO_DIR/msaflow/inference/fold_benchmark.py \
        --fasta          $FASTA \
        --decoder_ckpt   $DECODER_CKPT \
        --latent_fm_ckpt $LATENT_FM_CKPT \
        --output_dir     $OUTPUT_DIR \
        --mode           fewshot \
        --device         cuda \
        --n_seqs         $N_SEQS \
        --n_seeds        $N_SEEDS \
        --n_steps        $N_STEPS \
        --temperature    $TEMPERATURE \
        --protenix_model $PROTENIX_MODEL \
        --shallow_msa_dir $SHALLOW_MSA_DIR \
        --protenix_ckpt  $PROTENIX_CKPT \
        --max_rec_depth  $MAX_REC_DEPTH \
        --ref_cif_dir    $REF_CIF_DIR \
        --usalign_bin    $USALIGN_BIN \
        --shard_id       $SHARD_ID \
        --num_shards     $NUM_SHARDS \
        > $OUTPUT_DIR/shard_${SHARD_ID}.log 2>&1 &
done

echo "Launched shards (PIDs: $(jobs -p))"
wait
echo "All shards done: $(date)"

# ── 결과 병합 ──────────────────────────────────────────────────────────────────
python - << 'PYEOF'
import csv, glob, os, math

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
    tm_vals = [float(r["tm_score"]) for r in rows
               if r.get("tm_score") not in ("", "nan", None)
               and not math.isnan(float(r["tm_score"]))]
    print(f"Merged {len(rows)} rows → {out_path}")
    if tm_vals:
        print(f"TM-score  n={len(tm_vals)}  mean={sum(tm_vals)/len(tm_vals):.4f}")
else:
    print(f"No shard CSVs found in {output_dir}")
PYEOF

echo "All done: $(date)"

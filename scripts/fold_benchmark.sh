#!/bin/bash
#SBATCH --job-name=msaflow-bench
#SBATCH --nodes=1
#SBATCH --nodelist=ada-004
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/bench_%j.out
#SBATCH --error=logs/bench_%j.err

# ── Python 모듈 로드 ───────────────────────────────────────────────────────────
module load python/3.11.14

# ── uv PATH ───────────────────────────────────────────────────────────────────
export PATH="$HOME/.local/bin:$PATH"

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
REPO_DIR=${REPO_DIR:-/home/paul3875/projects/MSA_FLOW}
DECODER_CKPT=${DECODER_CKPT:-$REPO_DIR/runs/decoder/decoder_ema_final.pt}
LATENT_FM_CKPT=${LATENT_FM_CKPT:-$REPO_DIR/runs/latent_fm/latent_fm_ema_final.pt}
FASTA=${FASTA:-$REPO_DIR/data/foldbench_monomer.fasta}
OUTPUT_DIR=${OUTPUT_DIR:-$REPO_DIR/runs/fold_benchmark}
REF_PDB_DIR=${REF_PDB_DIR:-}          # leave empty if no reference PDBs
TMSCORE_BIN=${TMSCORE_BIN:-TMscore}   # set to full path if not in $PATH

N_SEQS=${N_SEQS:-32}       # sequences per seed (paper: 32)
N_SEEDS=${N_SEEDS:-10}     # latent FM seeds
N_STEPS=${N_STEPS:-100}    # ODE steps
TEMPERATURE=${TEMPERATURE:-0.0}

PROTENIX_MODEL=${PROTENIX_MODEL:-protenix_base_default_v1.0.0}
# Protenix looks for checkpoints under $PROTENIX_ROOT_DIR/checkpoint/
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-$REPO_DIR}

# ── 환경 활성화 ────────────────────────────────────────────────────────────────
source $REPO_DIR/.venv/bin/activate

# ── Protenix 경로 추가 ─────────────────────────────────────────────────────────
# NOTE: $REPO_DIR is intentionally omitted here — including it would put the
# local esm/ submodule on PYTHONPATH and shadow the installed fair-esm package,
# causing "AttributeError: module 'esm' has no attribute 'data'" in Protenix.
# fold_benchmark.py adds $REPO_DIR to sys.path at import time for its own use.
export PYTHONPATH=$REPO_DIR/Protenix:$PYTHONPATH

# ── 로그 디렉터리 ──────────────────────────────────────────────────────────────
mkdir -p $OUTPUT_DIR $REPO_DIR/logs

echo "Job ID        : $SLURM_JOB_ID"
echo "Node          : $SLURMD_NODENAME"
echo "Python        : $(python --version)"
echo "GPUs          : $CUDA_VISIBLE_DEVICES"
echo "Decoder ckpt  : $DECODER_CKPT"
echo "Latent FM ckpt: $LATENT_FM_CKPT"
echo "FASTA         : $FASTA"
echo "Output dir    : $OUTPUT_DIR"
echo "Protenix model: $PROTENIX_MODEL"
date

# ── 벤치마크 실행 (4 GPU 병렬) ────────────────────────────────────────────────
REF_ARG=""
if [ -n "$REF_PDB_DIR" ]; then
    REF_ARG="--ref_pdb_dir $REF_PDB_DIR"
fi

NUM_SHARDS=4

for SHARD_ID in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$SHARD_ID \
    python $REPO_DIR/msaflow/inference/fold_benchmark.py \
        --fasta          $FASTA \
        --decoder_ckpt   $DECODER_CKPT \
        --latent_fm_ckpt $LATENT_FM_CKPT \
        --output_dir     $OUTPUT_DIR \
        --device         cuda \
        --n_seqs         $N_SEQS \
        --n_seeds        $N_SEEDS \
        --n_steps        $N_STEPS \
        --temperature    $TEMPERATURE \
        --protenix_model $PROTENIX_MODEL \
        --tmscore_bin    $TMSCORE_BIN \
        --shard_id       $SHARD_ID \
        --num_shards     $NUM_SHARDS \
        $REF_ARG \
        > $OUTPUT_DIR/shard_${SHARD_ID}.log 2>&1 &
done

echo "Launched 4 shards (PIDs: $(jobs -p))"
wait
echo "All shards done: $(date)"

# ── 결과 병합 ──────────────────────────────────────────────────────────────────
python - << 'PYEOF'
import csv, glob, os

output_dir = os.environ.get("OUTPUT_DIR", "runs/fold_benchmark")
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
PYEOF

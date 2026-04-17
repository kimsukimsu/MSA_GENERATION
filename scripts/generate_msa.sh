#!/bin/bash
# Quick single-protein MSA generation (no SLURM).
# Generates 32 sequences in zero-shot mode for a single query sequence.
#
# Usage:
#   bash scripts/generate_msa.sh <query_seq> <output.a3m>
#
# Or override env vars:
#   DECODER_CKPT=... LATENT_FM_CKPT=... bash scripts/generate_msa.sh MKTAY... out.a3m
#
# For reconstruct / augment modes see msaflow/inference/generate.py directly.

set -euo pipefail

QUERY_SEQ=${1:-""}
OUTPUT=${2:-"output.a3m"}

if [ -z "$QUERY_SEQ" ]; then
    echo "Usage: $0 <query_seq> <output.a3m>"
    exit 1
fi

REPO_DIR=${REPO_DIR:-/home/paul3875/projects/MSA_FLOW}
DECODER_CKPT=${DECODER_CKPT:-$REPO_DIR/runs/decoder/decoder_ema_final.pt}
LATENT_FM_CKPT=${LATENT_FM_CKPT:-$REPO_DIR/runs/latent_fm/latent_fm_ema_final.pt}
N_SEQS=${N_SEQS:-32}
N_STEPS=${N_STEPS:-100}
TEMPERATURE=${TEMPERATURE:-0.0}
DEVICE=${DEVICE:-cuda}

export PATH="$HOME/.local/bin:$PATH"
source $REPO_DIR/.venv/bin/activate
export PYTHONPATH=$REPO_DIR/Protenix:$REPO_DIR:$PYTHONPATH

python $REPO_DIR/msaflow/inference/generate.py \
    --mode         zeroshot \
    --query_seq    "$QUERY_SEQ" \
    --decoder_ckpt $DECODER_CKPT \
    --latent_fm_ckpt $LATENT_FM_CKPT \
    --output       $OUTPUT \
    --n_seqs       $N_SEQS \
    --n_steps      $N_STEPS \
    --temperature  $TEMPERATURE \
    --device       $DEVICE

echo "Generated MSA written to $OUTPUT"

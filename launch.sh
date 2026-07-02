#!/bin/bash

# --- 1. Default Values ---
DMODEL=128
HEAD_DIM=128
LR_MODE="clipping"
N_TOKENS="None"


# --- 2. Parse Flags passed to ./launch.sh ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --d_model)
      DMODEL="$2"
      shift 2
      ;;
    --head_dimension)
      HEAD_DIM="$2"
      shift 2
      ;;
    --lr_schedule_mode)
      LR_MODE="$2"
      shift 2
      ;;
    --n_training_tokens)
      N_TOKENS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# determine number of tokens if doing whole training
if [[ "$N_TOKENS" == "whole" ]]; then
    N_TOKENS="5_846_302_720"
elif [[ "$N_TOKENS" == "chinchilla" ]]; then
    N_TOKENS="None"
fi

# --- 3. Dynamically Generate SLURM Names ---
# This automatically formats your names so you don't have to type them out!
JOB_NAME="${DMODEL}_${LR_MODE}"
if [[ "$LR_MODE" == "clipping" ]]; then
    if [[ "$N_TOKENS" == "5_846_302_720" ]]; then
        JOB_NAME="${DMODEL}_clipping"
        OUTPUT_DIR="grid_logs/${DMODEL}_whole_length"
    else
        JOB_NAME="${DMODEL}_whole"
        OUTPUT_DIR="grid_logs/${DMODEL}_chinchilla_length"
    fi
elif [[ "$LR_MODE" == "relative" ]]; then
    JOB_NAME="${DMODEL}_relative"
    OUTPUT_DIR="grid_logs/${DMODEL}_chinchilla_length_relative_mode"
fi

OUTPUT_LOG="${OUTPUT_DIR}/grid_%A_%a.log"

# Create the log directory automatically if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# --- 4. Launch sbatch ---
echo "Submitting SLURM job: $JOB_NAME"
sbatch \
  --job-name="$JOB_NAME" \
  --output="$OUTPUT_LOG" \
  submit_array.sh \
  --d_model "$DMODEL" \
  --head_dimension "$HEAD_DIM" \
  --lr_schedule_mode "$LR_MODE" \
  --n_training_tokens "$N_TOKENS"
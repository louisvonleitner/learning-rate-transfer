#!/bin/bash
#SBATCH --job-name=jax_device_test
#SBATCH -p grete:shared
#SBATCH -G A100:1
#SBATCH -c 4
#SBATCH --output=model_training_test_%j.log
#SBATCH --constraint="inet"
#SBATCH --mem=100G

# Tells Hugging Face NOT to attempt any network requests
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Get internet access route
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"
# export WANDB_API_KEY="your-key"

# Exit immediately if a command exits with a non-zero status
set -e

# --- 1. Initialize Conda for Batch/Cluster Shells ---
# Cluster nodes often don't load your ~/.bashrc automatically. 
# This safely hooks Conda into the current script process.
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    # Fallback if conda is already in the PATH but functions aren't exported
    eval "$($CONDA_EXE shell.bash hook)"
fi

# load modules
module load miniforge3
module load gcc
module load cuda

# --- 2. Activate the Environment ---
echo "Activating conda environment: mu_transformer"
conda activate mu_transformer

cd lingle

# --- 3. Debug Environment Check (Optional but helpful) ---
echo "Using Python from: $(which python)"
if command -v nvidia-smi &> /dev/null; then
    echo "Allocated CUDA Devices:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
fi
echo "--------------------------------------------------"

# --- 4. Execute the Target Script ---
echo "Launching model training..."
srun python mu_transformer/jax_impl/launch.py \
    --config=mu_transformer/configs/Louis_base.py \
    --mode=train \
    --workdir=./run_01 \
    --config.tokens_per_global_batch=131072 \
    --config.sequence_len=1024 \
    --config.n_mesh_rows=1 \
    --config.n_mesh_cols=1 \
    --config.hftr_tokenizer_name=T5TokenizerFast \
    --config.hftr_tokenizer_instance=t5-base \
    --config.hfds_identifier=allenai/c4 \
    --config.hfds_config=en \
    --config.hfds_datacol=text \
    --wb_enabled=False \
    --experiment_group="test"

echo "Job completed successfully."
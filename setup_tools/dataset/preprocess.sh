#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH -p standard96
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --output=model_preprocessing_%j.log
#SBATCH --time=24:00:00

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_NUM_PROC=$SLURM_CPUS_PER_TASK
export HTTP_PROXY="http://www-cache.gwdg.de:3128"
export HTTPS_PROXY="http://www-cache.gwdg.de:3128"
export FTP_PROXY="http://www-cache.gwdg.de:3128"

set -e

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
    eval "$($CONDA_EXE shell.bash hook)"
fi

module load miniforge3

echo "Activating conda environment: cpu_mu_transformer"
conda activate cpu_mu_transformer
cd ../../lingle

echo "Using Python from: $(which python)"
echo "--------------------------------------------------"
export JAX_PLATFORMS=cpu

echo "Launching preprocessing..."
srun --ntasks=1 python mu_transformer/jax_impl/launch.py \
    --config=mu_transformer/configs/Louis_base.py \
    --mode=train \
    --workdir=./run_01 \
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

echo "Bash script done."

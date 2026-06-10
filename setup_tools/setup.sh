#!/bin/bash

# This script initializes everything for training models via Lingle's mu_transformer on the cluster
# Exit immediately if a command exits with a non-zero status
set -e

echo "Setting up project..."
module load miniforge3

# Check if conda environment 'mu_transformer' exists, create it if it doesn't
if ! conda info --envs | grep -q "mu_transformer"; then
    echo "Creating conda environment 'mu_transformer'..."
    conda create -n mu_transformer python=3.9 -y
else
    echo "Conda environment 'mu_transformer' already exists."
fi

# Activate the environment
source activate mu_transformer

cd "$PROJECT"
mkdir -p mutransfer_test && cd mutransfer_test

echo "Loading from Git..."
if [ ! -d "mu_transformer" ]; then
    git clone https://github.com/lucaslingle/mu_transformer.git
fi
cd mu_transformer

module load gcc cuda

# Install pip packages
echo "Installing pip packages..."
# Note: pip install uses -e for editable or no flag for standard. -y is not a valid flag for 'pip install'
pip install . 
pip install --upgrade "jax[cuda12]"

# Verify JAX CUDA compatibility
echo "Verifying jax cuda compatibility..."
srun -p grete:shared -G A100:1 -c 4 python "$PROJECT/mutransfer/setup_tools/jax_env/test_jax_devices.py"

# Download tokenizer
echo "Downloading Tokenizer..."
python "$PROJECT/mutransfer/setup_tools/tokenizer/get_tokenizer.py"

# Verify tokenizer
echo "Verifying Tokenizer..."
python "$PROJECT/mutransfer/setup_tools/tokenizer/verify_tokenizer.py"

# Configure Hugging Face as offline only
echo "Configuring Hugging Face for offline mode..."
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Ask user if they want to download dataset locally
echo "Do you want to download the dataset locally? (y/n)"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Downloading dataset, this may take a while..."
    python "$PROJECT/mutransfer/setup_tools/dataset/download_dataset.py"
    
    echo "Verifying dataset..."
    python "$PROJECT/mutransfer/setup_tools/dataset/verify_dataset.py"
else
    echo "Skipping dataset download."
fi

echo "Finished setup..."
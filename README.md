This project is an implementation of LLM-Transformer Training under muTransfer [1]. 
Specifically, the experiments are conducted to analyse the accuracy and compute scaling of muTransfer for different model sizes.

The implementation is built on an existing repository, which was also set up to research [2] muTransfer on LLM-Transformers by Lucas Lingle: https://github.com/lucaslingle/mu_transformer. It was cloned with an Apache 2.0 license on 7. June 2026.


# Overview
muTransfer comes from muParametrization introduced by Yang and Greg [1], already described in Tensor Programs IV and extended in Tensor Programs VI. It is a way to transfer optimal training hyperparameters from a small (proxy) model to a big (target) model via scaling rules. This is a zero-shot transfer method, working simply based on mathematically derived scaling rules and validated empirically in experiments.

I specifically research the accuracy of the transferred hyperparameters for different proxy and target model sizes and try to give a recommendation on how to chose the proxy model size $n$ given a target model size $N$.

# Getting Started
You can easily launch the trianing of Transformers on the C4 dataset yourself, however doing this on a compute cluster is recommended, as the Dataset is huge as a download (TBs) and tokenization takes a lot of time, even on 64 processes in parallel. Training of course is also expensive, requiring strong GPUs with a good chunk of memory.

## Environment Setup
You can set up a conda environment and download the C4 dataset via HuggingFace's datasets with the setup_tools/setup.sh script. Simply run it as an executable.

Once you downloaded the dataset, you should tokenize it on CPUs before starting training on GPUs. For this, you use the setup_tools/dataset/preprocess.sh script, which will launch a model training, which kicks off the tokenizing process and probably fails before training can start because of JAX dependency errors on CPUs. 
I recommend not to tokenize and save the whole dataset as a binary, because that takes a lot of time and space. For most people, it will be sufficient to tokenize a fraction of it, stronlgy increasing training speed (preloading datset) and data handling in general. 

## Launching your first training
The best way to launch trainings is via analysis/make_run_matrix.py. 
This requires a hyperparameter grid, which can easily be generated via analysis/make_run_matrix.py.
'''
srun python analysis/make_run_matrix.py \
    --config=lingle/mu_transformer/configs/Louis_base.py \
    --mode=train \
    --workdir=lingle/run_01 \
    --config.tokens_per_global_batch=65536 \
    --config.sequence_len=1024 \
    --config.n_mesh_rows=1 \
    --config.n_mesh_cols=1 \
    --config.hftr_tokenizer_name=T5TokenizerFast \
    --config.hftr_tokenizer_instance=t5-base \
    --config.hfds_identifier=allenai/c4 \
    --config.hfds_config=en \
    --config.hfds_datacol=text \
    --wb_enabled=True \
    --experiment_group="test"
'''



# References
[1] Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer; Yang and Greg; 2022; http://arxiv.org/abs/2203.03466
[2] An Empirical Study of $\mu$P Learning Rate Transfer; Lucas Lingle; 2025; http://arxiv.org/abs/2404.05728
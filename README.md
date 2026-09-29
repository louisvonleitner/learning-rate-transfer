# Optimizing Learning Rate Transfer

Code for the Bachelor's thesis **"Optimizing Learning Rate Transfer"** (Louis von Leitner, Georg-August-Universität Göttingen, 2026).

The thesis asks how to choose the proxy model for µTransfer: how wide it should be and how long to train it. This repository contains the pipeline used for all experiments. It trains µP-parameterized decoder-only Transformers (widths 32–4096) on C4, runs grid sweeps over base learning rate and initialization scale, and includes the notebooks that turn the results into the thesis figures.

The model and training loop are Lucas Lingle's [`mu_transformer`](https://github.com/lucaslingle/mu_transformer) (JAX/Flax, Apache 2.0, see `lingle/LICENSE`), cloned on 7 June 2026 into `lingle/` and adapted. Every change to his files is marked in the code with an `Added by Louis` / `Changed by Louis` / `Removed by Louis` comment. Everything outside `lingle/` is new.

> **Scope:** the pipeline was built for, and only runs on, the GWDG GPU cluster (SLURM, one A100 per run). Running it elsewhere requires the changes listed under [Adapting to another cluster](#adapting-to-another-cluster).

---

## Branches

| Branch | Contents | Thesis |
|---|---|---|
| `master` | Pipeline for the 24-layer models (copy of `experiments`, plus this README) | Experiments 1 and 2 (Sections 4.2, 4.3) |
| `experiments` | Development branch of the above | – |
| `small_experiments` | Same pipeline for 2-layer models trained for 100 or 1,000 steps over several random seeds | Experiment 3 (Section 4.4) |
| `result-analysis` | Jupyter notebooks that produce the figures and run statistics | Figures 2–8, Table 4 |

---

## How the pipeline works

```
create_grid.py ──► grid_manifest.csv        one row per (base_lr, base_init_stddev) hyperparameter grid pair
                         │
launch.sh --d_model … ──► sbatch submit_array.sh   SLURM job array, one task per grid row
                         │
                         ▼
run_management.py  (one TrainingRun per array task)
   • counts parameters and sets the training horizon
   • derives the number of steps and warmup steps from the LR schedule mode
   • applies µP scaling to per-layer learning rates and init std
   • overrides Lingle's config and calls his training loop
                         │
                         ▼
run_results.csv (one row per run) + <run_id>/losses.csv + SLURM logs in grid_logs/
```

### Main settings

| Setting | Options | Meaning |
|---|---|---|
| `--d_model` | 32 … 1024 (4096 on `small_experiments`) | Model width *n* |
| `--head_dimension` | 32 or 128 | Attention head size (`d_model` must be divisible by it) |
| `--n_training_tokens` | `whole` | 5,846,302,720 tokens: the Chinchilla horizon of the width-1024 target, i.e. 89,208 steps at 65,536 tokens/step. Used in Experiment 1. |
| | `chinchilla` | 20 tokens per parameter of the model itself. Used in Experiment 2. |
| `--lr_schedule_mode` | `relative` | Linear warmup over 10,000/89,208 ≈ 11.2 % of the steps, then linear decay |
| | `clipping` | 10,000 warmup steps (all steps if the run is shorter), then linear decay |

- **Grid:** `create_grid.py` builds log₂-spaced base learning rates and five base init scales {2⁻², …, 2²}. The analysis keeps, for each learning rate, the best of the five init scales.
- **µP scaling (Adam):** see `get_abs_mup_scaling` in `run_management.py`.
  - The embedding uses the base LR.
  - Attention, FFN-in and unembedding use base LR / `d_model`.
  - FFN-out uses base LR / (4·`d_model`).
  - Init std is base init std · `d_model`^(-1/2).
- **Training-horizon scaling (Experiment 2)** is not applied during training. All widths sweep the same base-LR grid. The notebooks then compute the effective LR = base LR · √(training tokens).

### Horizons the pipeline computes (24 layers, 65,536 tokens per step)

| `d_model` | Chinchilla tokens | Steps | Warmup (`relative`) | Warmup (`clipping`) |
|---:|---:|---:|---:|---:|
| 32 | 45.6 M | 696 | 78 | 696 |
| 64 | 100.0 M | 1,526 | 171 | 1,526 |
| 128 | 235.3 M | 3,591 | 402 | 3,591 |
| 256 | 612.2 M | 9,342 | 1,047 | 9,342 |
| 512 | 1.79 B | 27,324 | 3,062 | 10,000 |
| 1024 (= `whole`) | 5.85 B | 89,208 | 10,000 | 10,000 |

---

## Repository layout (`master`)

```
experiment_pipeline/
  experiment_management/
    create_grid.py            # writes the (base_lr, init_stddev) grid to grid_manifest.csv
    run_management.py         # TrainingRun: horizon, schedule, µP scaling, launch, result saving
  launching/
    launch.sh                 # entry point: one call = one model configuration = one job array
    submit_array.sh           # SLURM batch script (partition grete:shared, 1×A100 per task)
    launch_wrapper_d_head.sh  # all launch.sh calls for Experiments 1 and 2 (head dim 32)
setup_tools/
  setup.sh                    # conda env, JAX/CUDA check, tokenizer download, optional C4 download
  dataset/preprocess.sh       # CPU job that tokenizes C4 into memmaps before GPU training
  tokenizer/, jax_env/        # helpers called by setup.sh
lingle/                       # adapted copy of Lingle's mu_transformer
  mu_transformer/configs/Louis_base.py   # base config for all runs (optimizer, depth, batch size, …)
  mu_transformer/jax_impl/launch.py      # training loop (returns loss history and wall time)
  mu_transformer/data.py                 # C4 download, tokenization and memmap loading
```

`lingle/scripts/` and `lingle/tests/` are upstream files (Lingle's TPU sweeps) and are not used here.

---

## Setup (once on GWDG GPU Node)

**1. Environment.** The setup script expects the repository at `$PROJECT/mutransfer`.

```bash
cd $PROJECT
git clone https://github.com/louisvonleitner/learning-rate-transfer.git mutransfer
cd mutransfer
bash setup_tools/setup.sh
```

`setup.sh` does the following:
- creates the conda env `mu_transformer` (Python 3.9);
- installs the dependencies and `jax[cuda12]`;
- checks on a GPU node that JAX sees the A100;
- saves the T5 tokenizer locally for offline use;
- optionally downloads C4.

The pipeline imports the *adapted* package in `lingle/` (it contains `Louis_base.py`). Make sure that is the `mu_transformer` Python finds, for example with `pip install -e ./lingle` inside the env.

**2. Tokenize C4 on CPUs** (before any GPU training).

```bash
cd setup_tools/dataset && sbatch preprocess.sh
```

This starts Lingle's training entry point with `JAX_PLATFORMS=cpu`, in a CPU-only conda env named `cpu_mu_transformer`. It tokenizes C4 into train/validation/test memmaps inside the working directory, capped at 26.6 B training tokens in `data.py`. 
Be prepared that this takes a a couple of hours.

The job usually crashes once training would start. That is expected: only the tokenized memmaps are needed. Training jobs later read the memmaps from the same working directory.

**3. Weights & Biases.** Runs log to W&B (switched on in `run_management.py`). Either run `wandb login` once, or set `WANDB_API_KEY` in `submit_array.sh`, or use `export WANDB_MODE=offline`.

---

## Quick launch example

This sweeps a width-128 proxy at its Chinchilla horizon with the relative schedule, which gives one curve of Figure 3 (top row).

```bash
cd $PROJECT/mutransfer                      # repository root

# 1. Build the grid: 10 base LRs in [2^-10, 2^-3] × 5 init scales = 50 runs
python experiment_pipeline/experiment_management/create_grid.py   # prints the number of rows (50)

# 2. Size the job array to the grid. In experiment_pipeline/launching/submit_array.sh set
#      #SBATCH --array=0-49%10      # tasks 0 … N-1, at most 10 running at once

# 3. Submit (launch.sh must be called from its own folder)
./launch.sh --d_model 128 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens chinchilla
```

This submits one SLURM array with 50 tasks. Each task trains a 24-layer, width-128 model for 3,591 steps (about 20 minutes on one A100) with one (base LR, init scale) pair.

- Check progress with `squeue --me`.
- Logs go to `grid_logs/32_128_chinchilla_length_relative_mode/`.
- Every finished run appends a row to `run_results.csv`: configuration, parameter count, steps, per-layer µP learning rates, final and best loss, wall time.

The matching full-length reference (Experiment 1) is

```bash
./launch.sh --d_model 1024 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens whole
```

These runs take about 68 h, longer than the 2-day SLURM limit. Checkpoints are saved every 7,500 steps under a name unique to each configuration, so submitting the same command again resumes the unfinished runs. Finished runs return immediately.

`launch_wrapper_d_head.sh` contains every `launch.sh` call used for Experiments 1 and 2. For the head-dimension-128 variant, replace `--head_dimension 32` with `128`.

---

## Experiment 3 (`small_experiments` branch)

This branch uses the same pipeline with the following changes:
- config `Louis_small.py`: 2 layers, 16 × 1,024 tokens per step, no checkpoints;
- a `--random_seed` flag;
- widths up to 4096.

The scripts sit in the repository root, and `run_management.py` is in `analysis/`.

```bash
git checkout small_experiments
# set #SBATCH --array=0-(N-1) in submit_array.sh
./launch.sh --d_model 256 --head_dimension 32 --lr_schedule_mode relative \
            --n_training_tokens 16_384_000 --random_seed 4           # 16,384,000 tokens = 1,000 steps
```

- `launch_wrapper_d_head.sh` launches all eight widths for one seed. Rerun it with a different `RANDOM_SEED` for each seed; the thesis uses six seeds, including the default 42.
- **100-step runs** use 1,638,400 tokens. `run_management.py` only accepts one fixed horizon (the `elif n_training_tokens == …` check), so change that value to match what you pass.

---

## Result analysis (`result-analysis` branch)

The notebooks in `analysis/result_analysis/` read the result CSV from their own folder. The CSVs are not versioned (see `.gitignore`); copy them there from the results folder.

| Notebook | Input | Produces |
|---|---|---|
| `learning_rate_transfer_plots.ipynb` | `run_results.csv` | Figures 2, 3, 6; wall time, FLOPs and token statistics behind Table 4 |
| `learning_rate_results_small_models.ipynb` | `run_results_small_model.csv` | Figures 4, 5, 7, 8 (seed averages, variance fits, quadratic fits) |
| `check_results.ipynb` | `run_results.csv` | Sanity checks: runs per configuration, steps, warmup |

---

## Adapting to another cluster

Paths and cluster settings are hard-coded for the author's GWDG project:

| What | Where |
|---|---|
| Working directory (tokenized data, checkpoints) | `workdir` default in `TrainingRun.__init__`, `run_management.py` |
| Results folder (`run_results.csv`, per-run losses) | `base_folder_path` in `run_management.py` |
| Local tokenizer path | `tokenizer_factory()` in `lingle/mu_transformer/jax_impl/launch.py`; `setup_tools/tokenizer/*.py` |
| Repository location | `setup.sh` expects `$PROJECT/mutransfer` |
| SLURM partition, GPU type, memory, time limit | `#SBATCH` header of `submit_array.sh`, `preprocess.sh` |
| Internet proxy and modules | `www-cache.gwdg.de` exports and `module load miniforge3 gcc cuda` in the same scripts |
| Grid size | `#SBATCH --array=…` in `submit_array.sh` must equal the number of rows in `grid_manifest.csv` |
| Fixed-horizon value | `elif n_training_tokens == …` in `run_management.py` |
| W&B logging | `FLAGS.wb_enabled = True` in `run_management.py` (overrides the command-line flag) |

Other limitations:
- Every run uses a single GPU; Lingle's multi-device sharding is switched off (1×1 mesh).
- Batch size is fixed per branch (65,536 tokens on `master`, 16,384 on `small_experiments`).
- The pipeline was used for the experiments in the thesis and has not been tested as a general-purpose tool.

---

## Acknowledgements

Computing time was provided on the supercomputer Emmy/Grete at NHR-Nord@Göttingen as part of the NHR infrastructure (project `bthesis_louis_vonleitner`).

The project was built on Lingle's github repository with the Apache 2.0 license. Thanks for providing it under this license.

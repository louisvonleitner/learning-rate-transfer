import numpy as np
import pandas as pd
from datetime import datetime
import itertools
from filelock import FileLock

import os
import sys

from absl import flags
import jax

# import Lingle adapted functions
from mu_transformer.jax_impl.launch import main as lingle_main
from mu_transformer.configs.Louis_base import get_config


class TrainingRun:

    def __init__(
        self,
        d_model: int,
        base_lr: float,
        init_stddev: float,
        workdir: str = "/mnt/vast-nhr/projects/bthesis_louis_vonleitner/mutransfer/lingle/run_01",
        n_training_tokens=None,
    ):

        # get base config
        self.cfg = get_config()

        # model parameters
        self.d_model = d_model
        self.model_depth = 24  # same over all experiments
        self.head_dimension = 128  # same over all experiments
        assert self.d_model % self.head_dimension == 0
        self.n_heads = self.d_model / self.head_dimension
        self.d_ffn = d_model * 4

        # model size
        self.vocab_size = 32128  # from T5 Tokenizer
        self.n_params_embedding = self.vocab_size * self.d_model
        self.n_params_encoder = self.n_params_embedding
        self.n_params_decoder = self.n_params_embedding
        self.n_params_mha = self.d_model**2
        self.n_params_rms_norm = self.d_model
        self.n_params_ffn = 2 * (self.d_ffn * self.d_model)
        self.n_params_transformer_block = sum(
            [self.n_params_mha, self.n_params_rms_norm, self.n_params_ffn]
        )
        self.n_parameters = sum(
            [
                self.n_params_embedding,
                self.n_params_decoder,
                self.n_params_transformer_block * self.model_depth,
            ]
        )

        # optimization parameters
        self.base_lr = base_lr
        self.max_lr = self.base_lr
        self.lr_schedule_name = self.cfg.lr_schedule_name
        self.optim_name = self.cfg.optim_name
        self.optim_beta1 = self.cfg.optim_beta1
        self.optim_beta2 = self.cfg.optim_beta2
        self.optim_eps = self.cfg.optim_eps
        self.weight_decay = self.cfg.wd

        # initialization variance parameters
        self.init_stddev = init_stddev
        self.absolute_init_stddev = self.init_stddev * self.d_model**-0.5

        # Chinchilla is used if n_training_tokens == None
        if n_training_tokens == None:
            self.n_training_tokens = (
                self.determine_chinchilla_optimal_n_training_tokens()
            )
        # If n_training_tokens is given
        else:
            self.n_training_tokens = n_training_tokens

        # batch and sequence length
        self.tokens_per_global_batch = self.cfg.tokens_per_global_batch
        self.sequence_len = self.cfg.sequence_len
        assert self.tokens_per_global_batch % self.sequence_len == 0
        self.batch_size = self.tokens_per_global_batch / self.sequence_len
        self.n_pretrain_steps = np.ceil(
            self.n_training_tokens / self.tokens_per_global_batch
        )
        self.n_warmup_steps = self.determine_n_warmup_steps()

        assert self.n_warmup_steps <= self.n_pretrain_steps

        # getting absolute lrs after mup
        self.absolute_lrs = self.get_abs_mup_scaling(
            base_lr=self.base_lr, d_model=self.d_model
        )
        self.embedding_matrix_lr = self.absolute_lrs["embedding_matrix_lr"]
        self.attention_weight_matrix_lr = self.absolute_lrs[
            "attention_weight_matrix_lr"
        ]
        self.unembedding_matrix_lr = self.absolute_lrs["unembedding_matrix_lr"]
        self.attention_bias_lr = self.absolute_lrs["attention_bias_lr"]
        self.w_ffn_in_lr = self.absolute_lrs["w_ffn_in_lr"]
        self.w_ffn_out_lr = self.absolute_lrs["w_ffn_out_lr"]
        self.bias_lr = self.absolute_lrs["bias_lr"]

        # tracking model runs and results
        self.workdir = workdir
        self.run_id = self.generate_run_id()
        self.base_folder_path = os.path.join(
            "/projects/extern/CIDAS/cidas_digitalisierung_lehre/bthesis_louis_vonleitner/dir.project/mutransfer/results"
        )
        self.base_result_df_path = os.path.join(
            self.base_folder_path, "run_results.csv"
        )
        self.run_folder_path = os.path.join(self.base_folder_path, self.run_id)
        self.run_losses_df_path = os.path.join(self.run_folder_path, "losses.csv")
        os.makedirs(self.base_folder_path, exist_ok=True)
        os.makedirs(self.run_folder_path, exist_ok=True)

        # predicting run time
        self.determine_theoretical_flops_and_walltime()

        # results
        self.final_loss = None
        self.best_loss = None
        self.training_wall_time = None
        self.training_loss_time_series = None

        # modify base config
        # ===================================================================
        self.cfg.d_model = self.d_model
        self.cfg.lr_base = self.base_lr
        self.cfg.init_stddev = self.init_stddev
        self.cfg.absolute_init_stddev = self.absolute_init_stddev
        self.cfg.n_layer = self.model_depth
        self.cfg.d_head = self.head_dimension
        self.cfg.n_pretrain_steps = self.n_pretrain_steps
        self.cfg.n_warmup_steps = self.n_warmup_steps

        # 3. Spoof the FLAGS object for the third-party library.
        # ===================================================================
        # You must set every flag that `main()` explicitly calls in its logging/setup block.
        FLAGS.config = self.cfg
        FLAGS.mode = "train"
        FLAGS.workdir = self.workdir

        # Mocking the remaining flags from the third-party main() snippet you provided
        FLAGS.experiment_group = "grid_search"
        FLAGS.rng_seed = 42
        FLAGS.rng_fold = 0
        FLAGS.wb_enabled = True  # Set to True if you want wandb
        FLAGS.wb_run = None
        FLAGS.load_suffix = ""
        FLAGS.save_suffix = ""
        FLAGS.verbosity = 0

    def generate_run_id(self):
        return str(self.d_model) + "_" + str(datetime.now())

    def determine_chinchilla_optimal_n_training_tokens(
        self, chinchilla_multiplier: float = 20
    ):
        # we use 20:1 ratio for training_tokens:parameters, taken from the original paper
        # the accuracy is not totally important, this is a proof of concept

        self.n_training_tokens = chinchilla_multiplier * self.n_parameters
        print(
            f"Using {self.n_training_tokens} training tokens according to chinchilla."
        )
        print(f"Number of Parameters in the model is {self.n_parameters}")
        return self.n_training_tokens

    def determine_n_warmup_steps(self):
        """
        Determines the number of warmup iterations.
        At least 10,000 warmup iterations are recommended for training stability in NLP.
        Therefore, we clip the warmup iterations to 10,000 if there would be less.
        """
        fraction = int(self.n_pretrain_steps / 10)

        if fraction < 10_000:
            if self.n_pretrain_steps >= 10_000:
                self.n_warmup_steps = 10_000
            else:
                self.n_warmup_steps = self.n_pretrain_steps
                print(
                    "All training iterations are warmup iterations, because n_iterations {self.n_pretrain_steps} < 10,000...",
                    flush=True,
                )
        else:
            self.n_warmup_steps = fraction

        return self.n_warmup_steps

    def determine_theoretical_flops_and_walltime(self, GPU="A100", GPU_stats=None):
        """
        Determines the FLOPS and walltime necessary based on model size and training horizon.
        For walltime, we assume specific GPU_stats
        GPU_stats = {
            flops,
            efficiency
            }
        """
        # Determine GPU_stats for compute
        if GPU_stats == None:
            if GPU == "A100":
                GPU_stats = {"flops": 312 * 10**12, "efficiency": 0.4}

        # number of total FLOPs for training (theoretical)
        self.theoretical_training_flops = 6 * self.n_parameters * self.n_training_tokens

        # walltime on GPU
        self.optimal_theoretical_training_walltime = (
            self.theoretical_training_flops / GPU_stats["flops"]
        )
        self.realistic_theoretical_training_walltime = (
            self.optimal_theoretical_training_walltime / GPU_stats["efficiency"]
        )

        return (
            self.theoretical_training_flops,
            self.optimal_theoretical_training_walltime,
            self.realistic_theoretical_training_walltime,
        )

    def get_abs_mup_scaling(self, base_lr, d_model, ffn_factor=4):
        dm = d_model
        dff = d_model * ffn_factor
        return {
            # embeddings
            "embedding_matrix_lr": base_lr,
            # attention
            "attention_weight_matrix_lr": base_lr / dm,
            "attention_bias_lr": base_lr,
            # feed-forward
            "w_ffn_in_lr": base_lr / dm,
            "w_ffn_out_lr": base_lr / dff,
            "bias_lr": base_lr,
            # unembedding
            "unembedding_matrix_lr": base_lr / dm,
        }

    def save_run_results(self, variables_to_save=None):
        """
        Save all run results, including the following:
        - Add stats to global run csv
        - Train loss over time
        """
        # Add stats to global run csv
        # ===================================
        if self.training_loss_time_series is not None:
            # determine what variables to save
            if variables_to_save is None:
                variables_to_save = [
                    "d_model",
                    "model_depth",
                    "head_dimension",
                    "n_heads",
                    "d_ffn",
                    "vocab_size",
                    "n_params_embedding",
                    "n_params_encoder",
                    "n_params_decoder",
                    "n_params_mha",
                    "n_params_rms_norm",
                    "n_params_ffn",
                    "n_params_transformer_block",
                    "n_parameters",
                    "base_lr",
                    "init_stddev",
                    "absolute_init_stddev",
                    "max_lr",
                    "lr_schedule_name",
                    "optim_name",
                    "optim_beta1",
                    "optim_beta2",
                    "optim_eps",
                    "weight_decay",
                    "n_training_tokens",
                    "tokens_per_global_batch",
                    "batch_size",
                    "sequence_len",
                    "n_pretrain_steps",
                    "n_warmup_steps",
                    "embedding_matrix_lr",
                    "attention_weight_matrix_lr",
                    "attention_bias_lr",
                    "w_ffn_in_lr",
                    "w_ffn_out_lr",
                    "bias_lr",
                    "unembedding_matrix_lr",
                    "run_id",
                    "run_folder_path",
                    "run_losses_df_path",
                    "final_loss",
                    "best_loss",
                    "training_wall_time",
                ]
            # variables_to_save is not None
            else:
                assert type(variables_to_save) == list

            # fetch values and save
            print("Writing results of run to global result csv.")
            os.makedirs(self.base_folder_path, exist_ok=True)

            results_dict = {
                variable: getattr(self, variable) for variable in variables_to_save
            }
            results_df = pd.DataFrame([results_dict])

            # handle simultaneous accessing of file by locking it
            lock = FileLock(self.base_result_df_path + ".lock")
            with lock:
                with open(self.base_result_df_path, "a") as f:
                    header_mode = f.tell() == 0  # empty file means no header yet
                    # write with pd.to_csv
                    results_df.to_csv(f, index=False, mode="a", header=header_mode)

            print(
                f"Wrote results of run to global result csv. at {self.base_result_df_path}"
            )

            # Write loss time series to csv
            # ===================================
            assert (
                type(self.training_loss_time_series) == dict
            ) or self.training_loss_time_series is None
            os.makedirs(self.run_folder_path, exist_ok=True)
            ts_df = pd.DataFrame([self.training_loss_time_series])
            ts_df.to_csv(self.run_losses_df_path, index=False)
            print(f"Wrote loss time series to csv. at {self.run_losses_df_path}")

    def launch(self):
        """
        Launch training run via Lingle's implementation.
        """

        print(f"Launching run with d_model={self.cfg.d_model}, lr={self.cfg.lr_base}")

        # Safe initialization check for older JAX versions
        try:
            jax.distributed.initialize()
        except RuntimeError:
            logging.info("JAX distributed framework already initialized. Skipping.")
        except AttributeError:
            # Fallback if the cluster environment's JAX handles initialization uniquely
            pass

        # launching Lingle model training
        run_stats = lingle_main(None)

        # saving run state
        if run_stats is not None:
            self.training_wall_time = run_stats["training_wall_time"]
            self.training_loss_time_series = run_stats["loss_time_series"]
            self.best_loss = run_stats["best_loss"]
            self.final_loss = run_stats["final_loss"]

        # run_stats is None
        else:
            print("Run stats are 'None'. Something did not work!")

        return run_stats


# Parse the CLI arguments if they haven't been parsed yet.
FLAGS = flags.FLAGS
if not FLAGS.is_parsed():
    FLAGS(sys.argv)

# if __name__ == "__main__":
#     # Example: A simple Pythonic loop for a grid search
#     lr = 0.01
#     stddev = 1.0

#     runner = TrainingRun(
#         d_model=1024,
#         base_lr=lr,
#         init_stddev=stddev,
#     )


if __name__ == "__main__":
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    row = pd.read_csv("analysis/grid_manifest.csv").iloc[task_id]

    runner = TrainingRun(
        d_model=128,
        base_lr=row["base_lr"],
        init_stddev=row["base_init_stddev"],
        n_training_tokens=5_846_302_720,
    )

    runner.launch()
    runner.save_run_results()

import numpy as np
import pandas as pd
import os
from datetime import datetime

from mu_transformer.configs.Louis_base import get_config


class TrainingRun:

    def __init__(self, config, d_model, base_lr, n_training_tokens=None):

        # get base config that we can work on
        self.cfg = config

        # model parameters
        self.d_model = d_model
        self.cfg.d_model = d_model
        self.model_depth = 24  # same over all experiments
        self.cfg.n_layer = self.model_depth
        self.head_dimension = 128  # same over all experiments
        self.cfg.d_head = self.head_dimension
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
        self.n_params_FFN = 2 * (self.d_ffn * self.d_model)
        self.n_params_transformer_block = sum(
            [self.n_params_mha, self.n_params_rms_norm, self.n_params_FFN]
        )
        self.n_parameters = sum(
            [
                self.n_params_embedding,
                self.n_params_decoder,
                self.n_params_transformer_block * self.model_depth,
            ]
        )

        # optimization parameters
        # TODO: To be changed and updated
        self.base_lr = base_lr
        self.cfg.lr_base = self.base_lr
        self.max_lr = self.base_lr
        self.lr_schedule_name = self.cfg.lr_schedule_name
        self.optim_name = self.cfg.optim_name
        self.optim_beta1 = self.cfg.optim_beta1
        self.optim_beta2 = self.cfg.optim_beta2
        self.optim_eps = self.cfg.optim_eps
        self.weight_decay = self.cfg.wd

        # Chinchilla is used if n_training_tokens == None
        if n_training_tokens == None:
            self.n_training_tokens = (
                self.determine_chinchilla_optimal_n_training_tokens()
            )
        # If n_training_tokens is given
        else:
            self.n_training_tokens = n_training_tokens

        self.tokens_per_global_batch = self.cfg.tokens_per_global_batch
        self.batch_size = 256
        self.sequence_len = self.cfg.sequence_len
        assert self.batch_size * self.sequence_len == self.tokens_per_global_batch
        self.n_pretrain_steps = np.ceil(
            self.n_training_tokens / self.tokens_per_global_batch
        )
        self.cfg.n_pretrain_steps = self.n_pretrain_steps
        self.n_warmup_steps = self.determine_n_warmup_steps()
        self.cfg.n_warmup_steps = self.n_warmup_steps

        assert self.n_warmup_steps <= self.n_pretrain_steps

        self.absolute_lrs = absolute_lrs  # <-- get this form Lingle script

        # tracking model runs and results
        self.run_id = self.generate_run_id()
        self.base_folder_path = os.path.join("mutransfer/results")
        self.base_result_df_path = os.path.join(
            self.base_folder_path, "run_results.csv"
        )
        self.run_folder_path = os.path.join(self.base_folder_path, self.run_id)
        self.run_losses_df_path = os.path.join(self.run_folder_path, "losses.csv")

        # predicting run time
        self.determine_theoretical_flops_and_walltime()

        # results
        # TODO: set up this extraction
        self.final_loss = None
        self.best_loss = None
        self.training_wall_time = None
        self.training_loss_time_series = None

    def generate_run_id(self):
        return str(self.d_model) + "_" + str(datetime.now())

    def determine_chinchilla_optimal_n_training_tokens(
        self, chinchilla_multiplier: float = 20
    ):
        # we use 20:1 ratio for training_tokens:parameters, taken from the original paper
        # the accuracy is not totally important, this is a proof of concept

        self.n_training_tokens = chinchilla_multiplier * self.n_parameters
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

    def save_run_results(self, variables_to_save=None):
        """
        Save all run results, including the following:
        - Add stats to global run csv
        - Train loss over time
        """
        # Add stats to global run csv
        # ===================================
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
                "max_lr",
                "warmup_iterations",
                "n_pretrain_steps",
                "n_training_tokens",
                "absolute_lrs",
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
        with open(self.base_result_df_path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            header_mode = f.tell() == 0  # empty file means no header yet
            # write with pd.to_csv
            results_df.to_csv(f, index=False, mode="a", header=header_mode)
            fcntl.flock(f, fcntl.LOCK_UN)

        print("Wrote results of run to global result csv.")

        # Write loss time series to csv
        # ===================================
        assert type(self.training_loss_time_series) == dict
        os.makedirs(self.run_folder_path, exist_ok=True)
        ts_df = pd.DataFrame([self.training_loss_time_series])
        ts_df.to_csv(self.run_losses_df_path, index=False)

    def launch_run(self):
        """
        Launch training run via Lingle's implementation.
        Parameters need to be set before!
        """
        pass

    def launch_run(self, mode, train_loop_fn, eval_loop_fn, sampling_loop_fn):
        """Dispatches execution to execution loops based on current global mode flags."""
        start_time = datetime.now()

        if mode == "train":
            if self.cfg.is_sweep:
                # Add your sweep handling code here
                train_loop_fn()
            else:
                train_loop_fn()
        elif mode in {"validation", "test"}:
            eval_metrics = eval_loop_fn(params=None, n_eval_step=None, mode=mode)
            self.final_loss = eval_metrics.get("loss_avg")
        elif mode == "sample":
            sampling_loop_fn()
        else:
            raise NotImplementedError

        self.training_wall_time = (datetime.now() - start_time).total_seconds()
        self.save_run_results()

import numpy as np
import pandas as pd
import os


class TrainingRun:

    def __init__(self, d_model, base_lr, n_training_tokens):

        # model parameters
        self.d_model = d_model
        self.model_depth = 32  # same over all experiments
        self.head_dimension = 128  # same over all experiments
        assert self.model_width % self.head_dimension == 0
        self.n_heads = self.model_width / self.head_dimension
        self.d_ffn = d_model * 4

        # model size
        self.vocab_size = 32128  # from T5 Tokenizer
        self.n_params_embedding = self.vocab_size * self.d_model
        self.n_params_encoder = self.d_embedding
        self.n_params_decoder = self.d_encoding
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
        self.max_lr = self.base_lr
        self.warmup_iterations = 10000
        self.training_iterations = 100000
        self.n_training_tokens = (
            TrainingRun.determine_chinchilla_optimal_n_training_tokens()
        )
        self.lrs = lrs  # <-- get this form Lingle script

        # tracking
        # TODO: set up run_id generation mechanism
        self.run_id = None

        # results
        # TODO: set up this extraction
        self.final_loss = None
        self.best_loss = None
        self.training_wall_time = None
        self.training_loss_list = None

        @classmethod
        def determine_chinchilla_optimal_n_training_tokens(
            self, chinchilla_multiplier: float = 20
        ):
            # we use 20:1 ratio for training_tokens:parameters, taken from the original paper
            # the accuracy is not totally important, this is a proof of concept

            self.n_training_tokens = chinchilla_multiplier * self.n_parameters
            return self.n_training_tokens

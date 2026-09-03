#!/bin/bash

N_TOKENS=16_384_000
# default random seed was 42
RANDOM_SEED=4
./launch.sh --d_model 32 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens $N_TOKENS --random_seed $RANDOM_SEED
./launch.sh --d_model 64 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens $N_TOKENS --random_seed $RANDOM_SEED
./launch.sh --d_model 128 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens $N_TOKENS --random_seed $RANDOM_SEED
./launch.sh --d_model 256 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens $N_TOKENS --random_seed $RANDOM_SEED
./launch.sh --d_model 512 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens $N_TOKENS --random_seed $RANDOM_SEED
./launch.sh --d_model 1024 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens $N_TOKENS --random_seed $RANDOM_SEED
./launch.sh --d_model 2048 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens $N_TOKENS --random_seed $RANDOM_SEED
./launch.sh --d_model 4096 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens $N_TOKENS --random_seed $RANDOM_SEED


echo "Launched all scripts!"

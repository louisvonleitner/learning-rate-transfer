#!/bin/bash
./launch.sh --d_model 32 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens chinchilla
./launch.sh --d_model 32 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens chinchilla
./launch.sh --d_model 32 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens whole

./launch.sh --d_model 64 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens chinchilla
./launch.sh --d_model 64 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens chinchilla
./launch.sh --d_model 64 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens whole

./launch.sh --d_model 128 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens chinchilla
./launch.sh --d_model 128 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens chinchilla
./launch.sh --d_model 128 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens whole

./launch.sh --d_model 256 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens chinchilla
./launch.sh --d_model 256 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens chinchilla
./launch.sh --d_model 256 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens whole

./launch.sh --d_model 512 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens chinchilla
./launch.sh --d_model 512 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens chinchilla
./launch.sh --d_model 512 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens whole

./launch.sh --d_model 1024 --head_dimension 32 --lr_schedule_mode relative --n_training_tokens chinchilla
./launch.sh --d_model 1024 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens chinchilla
./launch.sh --d_model 1024 --head_dimension 32 --lr_schedule_mode clipping --n_training_tokens whole

echo "Launched all scripts!"

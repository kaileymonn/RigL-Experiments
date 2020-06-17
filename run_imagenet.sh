#!/bin/bash

# source env/Scripts/activate

### Configure your environment to satisfy the requirements below ###
# absl-py>=0.6.0
# numpy>=1.15.4
# six>=1.12.0
# tensorflow-gpu>=1.12.0,<2.0
# tensorflow-datasets

# 1281167 training images
# 50 training epochs, batch_size 50 => 1281167 training steps

#### DEFAULTS ####
# num_train_images = 1281167
# num_eval_images = 50000
# train_batch_size = 1024
# eval_batch_size = 1000

# train_steps = (num_train_images/train_batch_size) * num_epochs
# => 90 epochs @ 1024 train_batch_size: 112590 steps
# => 50 epochs @ 128 train_batch_size: 500400 steps

# steps_per_eval: (112590 train_steps, 90 epochs) train_steps/num_epochs = 1251
# => (500400 train_steps, 50 epochs) 10008 steps

# iterations_per_loop: (112590 train_steps, 90 epochs) 1251
# => (500400 train_steps, 50 epochs) 10008 iters

# maskupdate_begin_step: 5000 steps @ 1024 train_batch_size, 90 total epochs
# => 20000 steps @ 128 train_batch_size, 50 total epochs
# maskupdate_end_step: 20000 steps @ 1024 train_batch_size, 90 total epochs
# => 80000 steps @ 128 train_batch_size, 50 total epochs
# maskupdate_frequency: 1000 steps @ 1024 train_batch_size, 90 total epochs
# => 1000 steps @ 128 train_batch_size, 50 total epochs => ~4x more frequent 

export PYTHONPATH=$PYTHONPATH:$PWD

# ###################### MobileNet_V1 ######################
# # Eval
# python3 rigl/imagenet_resnet/imagenet_train_eval.py \
#     --output_dir experiments/imagenet/mobilenet_v1/train \
#     --initial_value_checkpoint experiments/imagenet/mobilenet_v1/train/rigl/0.9/5000/20000/1000/0.3/0.1/0.0001/model.ckpt-343000 \
#     --model_architecture mobilenet_v1 \
#     --training_method rigl \
#     --mask_init_method erdos_renyi \
#     --data_format channels_last \
#     --mode eval \
#     --steps_per_checkpoint 2000 \
#     --train_steps 1281167 \
#     --train_batch_size 50 \
#     --eval_batch_size 50 \
#     --num_parallel_calls 6 \
#     --num_cores 4

# # Train and Eval (end_sparsity: 0.8)
# python3 rigl/imagenet_resnet/imagenet_train_eval.py \
#     --output_dir experiments/imagenet/mobilenet_v1/train_and_eval \
#     --data_directory /cv1/sequence/data_set/imagenet/train-images/imagenet/tf_records \
#     --model_architecture mobilenet_v1 \
#     --end_sparsity 0.8 \
#     --training_method rigl \
#     --mask_init_method erdos_renyi \
#     --data_format channels_last \
#     --mode train_and_eval \
#     --steps_per_checkpoint 2000 \
#     --train_steps 1200960 \
#     --steps_per_eval 10008 \
#     --iterations_per_loop 10008 \
#     --train_batch_size 128 \
#     --eval_batch_size 100 \
#     --maskupdate_begin_step 20000 \
#     --maskupdate_end_step 80000 \
#     --maskupdate_frequency 1000 \
#     --num_parallel_calls 16 \
#     --num_cores 8 \
#     --keep_checkpoint_max 10 \
#     # --initial_value_checkpoint experiments/imagenet/mobilenet_v1/train_and_eval/rigl/0.8/20000/80000/1000/0.3/0.1/0.0001/model.ckpt-1170928

# Eval
python3 rigl/imagenet_resnet/imagenet_train_eval.py \
    --initial_value_checkpoint experiments/imagenet/mobilenet_v1/train_and_eval/rigl/0.8/20000/80000/1000/0.3/0.1/0.0001/model.ckpt-1200960 \
    --output_dir experiments/imagenet/mobilenet_v1/train_and_eval \
    --data_directory /cv1/sequence/data_set/imagenet/train-images/imagenet/tf_records \
    --model_architecture mobilenet_v1 \
    --end_sparsity 0.8 \
    --training_method rigl \
    --mask_init_method erdos_renyi \
    --data_format channels_last \
    --mode eval \
    --steps_per_checkpoint 2000 \
    --train_steps 1200960 \
    --steps_per_eval 10008 \
    --iterations_per_loop 10008 \
    --train_batch_size 128 \
    --eval_batch_size 50 \
    --maskupdate_begin_step 20000 \
    --maskupdate_end_step 80000 \
    --maskupdate_frequency 1000 \
    --num_parallel_calls 16 \
    --num_cores 8 \
    --keep_checkpoint_max 10 \
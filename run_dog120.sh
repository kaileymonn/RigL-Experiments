#!/bin/bash

### Configure your environment to satisfy the requirements below ###
# absl-py>=0.6.0
# numpy>=1.15.4
# six>=1.12.0
# tensorflow-gpu>=1.12.0,<2.0
# tensorflow-datasets

export PYTHONPATH=$PYTHONPATH:$PWD

# 16437 training images
# 50 epochs at batch size 50 => 16437 training steps 
python rigl/imagenet_resnet/imagenet_train_eval.py 
    --output_dir dog120_mobilenetv1_saves \
    --model_architecture mobilenet_v1 \
    --training_method rigl \
    --mask_init_method erdos_renyi \
    --data_format channels_last \
    --mode train_and_eval \
    --data_directory stanford_dog_120 \
    --steps_per_checkpoint 1000 \
    --train_steps 16437 \
    --train_batch_size 50 \
    --eval_batch_size 50 \
    --num_train_images 16437 \
    --num_eval_images 4100 \
    --num_label_classes 120 \
    --num_parallel_calls 6 \
    --num_cores 4 
     
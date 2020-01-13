#!/bin/bash

# source env/Scripts/activate

### Configure your environment to satisfy the requirements below ###
# absl-py>=0.6.0
# numpy>=1.15.4
# six>=1.12.0
# tensorflow-gpu>=1.12.0,<2.0
# tensorflow-datasets

# 50 training epochs, batch_size 50 => 1281167 training steps
export PYTHONPATH=$PYTHONPATH:$PWD

python rigl/imagenet_resnet/imagenet_train_eval.py --output_dir imagenet_mobilenetv1_saves \
    --model_architecture mobilenet_v1 \
    --training_method rigl \
    --mask_init_method erdos_renyi \
    --train_steps 1281167 \
    --train_batch_size 50 \
    --eval_batch_size 50 \
    --num_parallel_calls 6 \
    --num_cores 4 
    
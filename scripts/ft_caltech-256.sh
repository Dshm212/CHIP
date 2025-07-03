#!/bin/bash

# Define commands to run in a list
commands=(
    # **********ResNet**********

    # ==========PAN==========
#    "CUDA_VISIBLE_DEVICES=1 python train_v23.py --exp-id 2_bn_image --dataset caltech-256 \
#    --passport-config passport_configs/alexnet_passport_ori.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type bn --tag last3 --train-private \
#    -tf --tl-dataset caltech-101 --epoch 100 \
#    --pretrained-path pretrained/alexnet.pth"
#
#    "CUDA_VISIBLE_DEVICES=1 python train_v23.py --exp-id 2_gn_image --dataset caltech-256 \
#    --passport-config passport_configs/alexnet_passport_ori.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private \
#    -tf --tl-dataset caltech-101 --epoch 100 \
#    --pretrained-path pretrained/alexnet.pth"

    # ==========TdN==========
#    "CUDA_VISIBLE_DEVICES=1 python train_v23.py --exp-id 3_bn_image --dataset caltech-256 \
#    --passport-config passport_configs/alexnet_passport_ori.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type bn --tag last3 --train-private --hash \
#    -tf --tl-dataset caltech-101 --epoch 100 \
#    --pretrained-path pretrained/alexnet.pth"
#
#    "CUDA_VISIBLE_DEVICES=1 python train_v23.py --exp-id 3_gn_image --dataset caltech-256 \
#    --passport-config passport_configs/alexnet_passport_ori.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private --hash \
#    -tf --tl-dataset caltech-101 --epoch 100 \
#    --pretrained-path pretrained/alexnet.pth"

    # ==========CHIP==========
      "CUDA_VISIBLE_DEVICES=1 python train_v23.py --exp-id bn_image --dataset caltech-256 \
      --passport-config passport_configs/resnet_passport_l4.json --arch resnet --epoch 200 \
      --key-type image --norm-type bn --tag l4 --train-private --hash --chameleon \
      -tf --tl-dataset caltech-101 --epoch 100 \
      --pretrained-path pretrained/resnet.pth"
)

# Iterate through each command
for cmd in "${commands[@]}"; do
    echo "Running: $cmd"
    eval "$cmd"

    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error encountered. Skipping to the next command."
    fi
done

echo "All commands processed."

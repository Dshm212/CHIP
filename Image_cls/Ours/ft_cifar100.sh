#!/bin/bash

# Define commands to run in a list
commands=(
    # **********ResNet**********

    # ==========PAN==========
#    "CUDA_VISIBLE_DEVICES=1 python train_v23.py --exp-id 2_bn_image --dataset cifar100 \
#    --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type bn --tag last3 --train-private \
#    -tf --tl-dataset cifar10 --epoch 100 \
#    --pretrained-path pretrained/alexnet.pth"
#
#    "CUDA_VISIBLE_DEVICES=1 python train_v23.py --exp-id 2_gn_image --dataset cifar100 \
#    --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private \
#    -tf --tl-dataset cifar10 --epoch 100 \
#    --pretrained-path pretrained/alexnet.pth"

    # ==========TdN==========
#    "CUDA_VISIBLE_DEVICES=1 python train_v23.py --exp-id 3_bn_image --dataset cifar100 \
#    --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type bn --tag last3 --train-private --hash \
#    -tf --tl-dataset cifar10 --epoch 100 \
#    --pretrained-path pretrained/alexnet.pth"
#
#    "CUDA_VISIBLE_DEVICES=1 python train_v23.py --exp-id 3_gn_image --dataset cifar100 \
#    --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private --hash \
#    -tf --tl-dataset cifar10 --epoch 100 \
#    --pretrained-path pretrained/alexnet.pth"

    # ==========CHIP==========
      "CUDA_VISIBLE_DEVICES=1 python train_v23.py --exp-id 4_bn_image --dataset cifar100 \
      --passport-config passport_configs/alexnet_passport_ori.json --arch alexnet --epoch 200 \
      --key-type image --norm-type bn --tag last3 --train-private --hash --chameleon \
      -tf --tl-dataset cifar10 --epoch 100 \
      --pretrained-path pretrained/alexnet.pth"

#      "CUDA_VISIBLE_DEVICES=1 python train_v23.py --exp-id 4_gn_image --dataset cifar100 \
#      --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
#      --key-type image --norm-type gn --tag last3 --train-private --hash --chameleon \
#      -tf --tl-dataset cifar10 --epoch 100 \
#      --pretrained-path pretrained/alexnet.pth"
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

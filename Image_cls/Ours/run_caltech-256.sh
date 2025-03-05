#!/bin/bash

# Define commands to run in a list
commands=(
    # **********AlexNet**********

    # ==========PAN==========
#    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#    --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private \
#    --pretrained-path pretrained/alexnet.pth"
#
#    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#    --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private \
#    --pretrained-path pretrained/alexnet.pth"

    # ==========TdN==========
#    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#    --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private --hash \
#    --pretrained-path pretrained/alexnet.pth"
#
#    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#    --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private --hash \
#    --pretrained-path pretrained/alexnet.pth"

    # ==========CHIP==========
#      "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#      --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
#      --key-type image --norm-type gn --tag last3 --train-private --hash --chameleon \
#      --pretrained-path pretrained/alexnet.pth"
#      "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#      --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
#      --key-type image --norm-type gn --tag last3 --train-private --hash --chameleon \
#      --pretrained-path pretrained/alexnet.pth"

    # **********ResNet**********

    # ==========PAN==========
#    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#    --passport-config passport_configs/resnet_passport_l4.json --arch resnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private \
#    --pretrained-path pretrained/resnet.pth"
#
#    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#    --passport-config passport_configs/resnet_passport_l4.json --arch resnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private \
#    --pretrained-path pretrained/resnet.pth"

    # ==========TdN==========
#    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#    --passport-config passport_configs/resnet_passport_l4.json --arch resnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private --hash \
#    --pretrained-path pretrained/resnet.pth"
#
#    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#    --passport-config passport_configs/resnet_passport_l4.json --arch resnet --epoch 200 \
#    --key-type image --norm-type gn --tag last3 --train-private --hash \
#    --pretrained-path pretrained/resnet.pth"

    # ==========CHIP==========

    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
    --passport-config passport_configs/alexnet_passport_1.json --arch alexnet --epoch 200 \
    --key-type image --norm-type gn --tag last1 --train-private --hash --chameleon \
    --pretrained-path pretrained/alexnet.pth"

    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
    --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
    --key-type image --norm-type gn --tag last3 --train-private --hash --chameleon \
    --pretrained-path pretrained/alexnet.pth"

    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
    --passport-config passport_configs/alexnet_passport_5.json --arch alexnet --epoch 200 \
    --key-type image --norm-type gn --tag last5 --train-private --hash --chameleon \
    --pretrained-path pretrained/alexnet.pth"

    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
    --passport-config passport_configs/resnet_passport_l4.json --arch resnet --epoch 200 \
    --key-type image --norm-type gn --tag l4 --train-private --hash --chameleon \
    --pretrained-path pretrained/resnet.pth"

    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
    --passport-config passport_configs/resnet_passport_l34.json --arch resnet --epoch 200 \
    --key-type image --norm-type gn --tag l34 --train-private --hash --chameleon \
    --pretrained-path pretrained/resnet.pth"

    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
    --passport-config passport_configs/resnet_passport_l234.json --arch resnet --epoch 200 \
    --key-type image --norm-type gn --tag l234 --train-private --hash --chameleon \
    --pretrained-path pretrained/resnet.pth"






    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id bn_image --dataset caltech-256 \
    --passport-config passport_configs/alexnet_passport_1.json --arch alexnet --epoch 200 \
    --key-type image --norm-type bn --tag last1 --train-private --hash --chameleon \
    --pretrained-path pretrained/alexnet.pth"

    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id bn_image --dataset caltech-256 \
    --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --epoch 200 \
    --key-type image --norm-type bn --tag last3 --train-private --hash --chameleon \
    --pretrained-path pretrained/alexnet.pth"

    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id bn_image --dataset caltech-256 \
    --passport-config passport_configs/alexnet_passport_5.json --arch alexnet --epoch 200 \
    --key-type image --norm-type bn --tag last5 --train-private --hash --chameleon \
    --pretrained-path pretrained/alexnet.pth"

    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id bn_image --dataset caltech-256 \
    --passport-config passport_configs/resnet_passport_l4.json --arch resnet --epoch 200 \
    --key-type image --norm-type bn --tag l4 --train-private --hash --chameleon \
    --pretrained-path pretrained/resnet.pth"

    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id bn_image --dataset caltech-256 \
    --passport-config passport_configs/resnet_passport_l34.json --arch resnet --epoch 200 \
    --key-type image --norm-type bn --tag l34 --train-private --hash --chameleon \
    --pretrained-path pretrained/resnet.pth"

    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id bn_image --dataset caltech-256 \
    --passport-config passport_configs/resnet_passport_l234.json --arch resnet --epoch 200 \
    --key-type image --norm-type bn --tag l234 --train-private --hash --chameleon \
    --pretrained-path pretrained/resnet.pth"

    # ==========baseline==========

#    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#    --passport-config passport_configs/alexnet_passport_1.json --arch alexnet --epoch 200 \
#    --key-type image --norm-type gn --tag baseline \
#    --pretrained-path pretrained/alexnet.pth"
#
#    "CUDA_VISIBLE_DEVICES=2 python train_v23.py --exp-id gn_image --dataset caltech-256 \
#    --passport-config passport_configs/resnet_passport_l1.json --arch resnet --epoch 200 \
#    --key-type image --norm-type gn --tag baseline \
#    --pretrained-path pretrained/resnet.pth"
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

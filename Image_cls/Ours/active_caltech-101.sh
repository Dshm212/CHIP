#!/bin/bash

# Define commands to run in a list
commands=(
  "CUDA_VISIBLE_DEVICES=1 python amb_attack_v3.py  --type random --tag last2 \
  --passport-config passport_configs/resnet18_passport_l34.json --arch resnet \
  --dataset caltech-101 --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/resnet.pth \
  --train-private --hash --chameleon --run-id 1"

  "CUDA_VISIBLE_DEVICES=1 python amb_attack_v3.py  --type random --tag last2 \
  --passport-config passport_configs/resnet18_passport_l34.json --arch resnet \
  --dataset caltech-101 --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/resnet.pth \
  --train-private --hash --chameleon --run-id 2"

  "CUDA_VISIBLE_DEVICES=1 python amb_attack_v3.py  --type random --tag last2 \
  --passport-config passport_configs/resnet18_passport_l34.json --arch resnet \
  --dataset caltech-101 --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/resnet.pth \
  --train-private --hash --chameleon --run-id 3"

  "CUDA_VISIBLE_DEVICES=1 python amb_attack_v3.py  --type random --tag last2 \
  --passport-config passport_configs/resnet18_passport_l34.json --arch resnet \
  --dataset caltech-101 --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/resnet.pth \
  --train-private --hash --chameleon --run-id 4"

  "CUDA_VISIBLE_DEVICES=1 python amb_attack_v3.py  --type random --tag last2 \
  --passport-config passport_configs/resnet18_passport_l34.json --arch resnet \
  --dataset caltech-101 --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/resnet.pth \
  --train-private --hash --chameleon --run-id 5"
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

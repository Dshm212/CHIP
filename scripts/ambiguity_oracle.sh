#!/bin/bash

# Define commands to run in a list
commands=(
  "CUDA_VISIBLE_DEVICES=3 python amb_attack_v3.py  --type oracle --tag last3  --pretrained-path pretrained/alexnet.pth \
  --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --dataset caltech-101 \
  --norm-type bn --exp-id bn_image --pretrained-path pretrained/alexnet.pth --train-private --hash --chameleon"
  
  "CUDA_VISIBLE_DEVICES=3 python amb_attack_v3.py  --type oracle --tag l4  --pretrained-path pretrained/resnet.pth \
  --passport-config passport_configs/resnet_passport_l4.json --arch resnet --dataset caltech-101 \
  --norm-type bn --exp-id bn_image --pretrained-path pretrained/resnet.pth --train-private --hash --chameleon"
  
  "CUDA_VISIBLE_DEVICES=3 python amb_attack_v3.py  --type oracle --tag last3  --pretrained-path pretrained/alexnet.pth \
  --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --dataset caltech-256 \
  --norm-type bn --exp-id bn_image --pretrained-path pretrained/alexnet.pth --train-private --hash --chameleon"
  
  "CUDA_VISIBLE_DEVICES=3 python amb_attack_v3.py  --type oracle --tag l4  --pretrained-path pretrained/resnet.pth \
  --passport-config passport_configs/resnet_passport_l4.json --arch resnet --dataset caltech-256 \
  --norm-type bn --exp-id bn_image --pretrained-path pretrained/resnet.pth --train-private --hash --chameleon"
  
  "CUDA_VISIBLE_DEVICES=3 python amb_attack_v3.py  --type oracle --tag last3  --pretrained-path pretrained/alexnet.pth \
  --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --dataset cifar10 \
  --norm-type bn --exp-id bn_image --pretrained-path pretrained/alexnet.pth --train-private --hash --chameleon"
  
  "CUDA_VISIBLE_DEVICES=3 python amb_attack_v3.py  --type oracle --tag l4  --pretrained-path pretrained/resnet.pth \
  --passport-config passport_configs/resnet_passport_l4.json --arch resnet --dataset cifar10 \
  --norm-type bn --exp-id bn_image --pretrained-path pretrained/resnet.pth --train-private --hash --chameleon"
  
  "CUDA_VISIBLE_DEVICES=3 python amb_attack_v3.py  --type oracle --tag last3  --pretrained-path pretrained/alexnet.pth \
  --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --dataset cifar100 \
  --norm-type bn --exp-id bn_image --pretrained-path pretrained/alexnet.pth --train-private --hash --chameleon"
  
  "CUDA_VISIBLE_DEVICES=3 python amb_attack_v3.py  --type oracle --tag l4  --pretrained-path pretrained/resnet.pth \
  --passport-config passport_configs/resnet_passport_l4.json --arch resnet --dataset cifar100 \
  --norm-type bn --exp-id bn_image --pretrained-path pretrained/resnet.pth --train-private --hash --chameleon"
  
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

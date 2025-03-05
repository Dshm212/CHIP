#!/bin/bash

# Define commands to run in a list
commands=(

#  "CUDA_VISIBLE_DEVICES=0 python amb_attack_v3.py  --type random --tag last1  --pretrained-path pretrained/alexnet.pth \
#  --passport-config passport_configs/alexnet_passport_1.json --arch alexnet --dataset caltech-101 \
#  --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/alexnet.pth --train-private --hash --chameleon"
#
#  "CUDA_VISIBLE_DEVICES=0 python amb_attack_v3.py  --type random --tag last3  --pretrained-path pretrained/alexnet.pth \
#  --passport-config passport_configs/alexnet_passport_3.json --arch alexnet --dataset caltech-101 \
#  --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/alexnet.pth --train-private --hash --chameleon"
#
#  "CUDA_VISIBLE_DEVICES=0 python amb_attack_v3.py  --type random --tag last5  --pretrained-path pretrained/alexnet.pth \
#  --passport-config passport_configs/alexnet_passport_5.json --arch alexnet --dataset caltech-101 \
#  --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/alexnet.pth --train-private --hash --chameleon"

  "CUDA_VISIBLE_DEVICES=0 python amb_attack_v3.py  --type random --tag resleaky  --pretrained-path pretrained/resnet.pth \
  --passport-config passport_configs/resnet18_passport_1.json --arch resnet --dataset caltech-101 \
  --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/resnet.pth --train-private --hash --chameleon"

#  "CUDA_VISIBLE_DEVICES=0 python amb_attack_v3.py  --type random --tag last2  --pretrained-path pretrained/resnet.pth \
#  --passport-config passport_configs/resnet18_passport_l3.json --arch resnet --dataset caltech-101 \
#  --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/resnet.pth --train-private --hash --chameleon"
#
#  "CUDA_VISIBLE_DEVICES=0 python amb_attack_v3.py  --type random --tag last3  --pretrained-path pretrained/resnet.pth \
#  --passport-config passport_configs/resnet18_passport_l1.json --arch resnet --dataset caltech-101 \
#  --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/resnet.pth --train-private --hash --chameleon"
#
#  "CUDA_VISIBLE_DEVICES=0 python amb_attack_v3.py  --type random --tag last4  --pretrained-path pretrained/resnet.pth \
#  --passport-config passport_configs/resnet18_passport_l2.json --arch resnet --dataset caltech-101 \
#  --norm-type bn --exp-id 4_bn_image --pretrained-path pretrained/resnet.pth --train-private --hash --chameleon"
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

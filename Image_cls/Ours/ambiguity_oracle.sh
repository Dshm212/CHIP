#!/bin/bash

# Define commands to run in a list
commands=(
  "CUDA_VISIBLE_DEVICES=1 python amb_attack_v3.py  --type oracle --tag last5  --pretrained-path pretrained/alexnet.pth \
  --passport-config passport_configs/alexnet_passport_5.json --arch alexnet --dataset caltech-101 \
  --norm-type gn --exp-id gn_image --pretrained-path pretrained/alexnet.pth --train-private --hash --chameleon"
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

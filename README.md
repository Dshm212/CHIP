# CHIP
This repository contains the official implementation of our paper "[CHIP: Chameleon Hash-based Irreversible Passport for Robust Deep Model Ownership Verification and Active Usage Control](https://arxiv.org/abs/2505.24536)". **Authors**: Chaohui Xu, Qi Cui, and Chip-Hong Chang.

## Introduction
The pervasion of large-scale Deep Neural Networks (DNNs) and their enormous training costs make their intellectual property (IP) protection of paramount importance. Recently introduced passport-based methods attempt to steer DNN watermarking towards strengthening ownership verification against ambiguity attacks by modulating the affine parameters of normalization layers. Unfortunately, neither watermarking nor passport-based methods provide a holistic protection with robust ownership proof, high fidelity, active usage authorization and user traceability for offline access distributed models and multi-user Machine-Learning as a Service (MLaaS) cloud model. In this paper, we propose a Chameleon Hash-based Irreversible Passport (CHIP) protection framework that utilizes the cryptographic chameleon hash function to achieve all these goals. The collision-resistant property of chameleon hash allows for strong model ownership claim upon IP infringement and liable user traceability, while the trapdoor-collision property enables hashing of multiple user passports and licensee certificates to the same immutable signature to realize active usage control. Using the owner passport as an oracle, multiple user-specific triplets, each contains a passport-aware user model, a user passport, and a licensee certificate can be created for secure offline distribution. The watermarked master model can also be deployed for MLaaS with usage permission verifiable by the provision of any trapdoor-colliding user passports. CHIP is extensively evaluated on four datasets and two architectures to demonstrate its protection versatility and robustness.
<div align="center">
  <img width="476" height="285" alt="compressed_CHIP" src="https://github.com/user-attachments/assets/6ca830d6-b1f2-47d9-8d6f-12ef6b3ee03b" />
</div>

## Installation
Our code is built on Python 3.8.16, torch 1.11.0, and cuda 11.3
### Create the environment
```bash
conda create -n CHIP python=3.8.16
conda activate CHIP
```
### Clone the repository
```bash
git clone https://github.com/Dshm212/CHIP.git
cd CHIP
```
### Install dependencies
```bash
pip install -r requirements.txt
```

## Run

### Train the master model
The following script is an example to train a master model embedded with the **immutable** signature.
```bash
CUDA_VISIBLE_DEVICES=0 python train_v23.py \
  --exp-id bn_image \
  --dataset cifar10 \
  --passport-config passport_configs/alexnet_passport_3.json \
  --arch alexnet \
  --epoch 200 \
  --key-type image \
  --norm-type bn \
  --tag last3 \
  --train-private \
  --hash \
  --chameleon
```

The default copyright text is "Copyright to Alice". To customize it, modify line 151 in `./CHIP/models/layers/passportconv2d_private.py`.

### User triplets generation
Once the master model is trained, the model owner can conduct trapdoor collision to generate multiple user triplets, each contains a protected user model and its uniquely bound key, with the following script.
```bash
CUDA_VISIBLE_DEVICES=0 python amb_attack_v3.py \
  --type random \
  --tag last3 \
  --passport-config passport_configs/alexnet_passport_3.json \
  --arch alexnet \
  --dataset cifar10 \
  --norm-type bn \
  --exp-id bn_image \
  --train-private \
  --hash \
  --chameleon \
  --run-id 1
```

The default number of users is set in line 96 in `./CHIP/amb_attack_v3.py`.

### Robustness evaluation
We provide scripts to evaluate the robustness of CHIP against ambiguity attacks (random / oracle) and fine-tuning attacks.
#### Ambiguity attacks with random passports
```bash
CUDA_VISIBLE_DEVICES=0 python amb_attack_v3.py \
  --type random \
  --tag last3 \
  --passport-config passport_configs/alexnet_passport_3.json \
  --arch alexnet \
  --dataset cifar10 \
  --norm-type bn \
  --exp-id bn_image \
  --pretrained-path pretrained/alexnet.pth \
  --train-private \
  --hash \
  --chameleon
```
#### Ambiguity attacks with oracle passports
```bash
CUDA_VISIBLE_DEVICES=0 python amb_attack_v3.py \
  --type oracle \
  --tag last3 \
  --passport-config passport_configs/alexnet_passport_3.json \
  --arch alexnet \
  --dataset cifar10 \
  --norm-type bn \
  --exp-id bn_image \
  --pretrained-path pretrained/alexnet.pth \
  --train-private \
  --hash \
  --chameleon
```
#### Fine-tuning attacks
```bash
CUDA_VISIBLE_DEVICES=0 python train_v23.py \
  --exp-id bn_image \
  --dataset cifar10 \
  --passport-config passport_configs/alexnet_passport_3.json \
  --arch alexnet \
  --epoch 200 \
  --key-type image \
  --norm-type bn \
  --tag last3 \
  --train-private \
  --hash \
  --chameleon \
  -tf \
  --tl-dataset \
  cifar100 \
  --epoch 100
```

## Acknowledgement

This research is supported by the National Research Foundation, Singapore, and Cyber Security Agency of Singapore under its National Cybersecurity Research & Development Programme (Development of Secured Components & Systems in Emerging Technologies through Hardware & Software Evaluation NRF-NCR25-DeSNTU-0001). Any opinions, findings and conclusions or recommendations expressed in this material are those of the author(s) and do not reflect the view of National Research Foundation, Singapore and Cyber Security Agency of Singapore.

Our code is developed based on previous repositories ([DeepIPR](https://github.com/kamwoh/DeepIPR), [PAN](https://github.com/ZJZAC/Passport-aware-Normalization), [Steganographic Passport](https://github.com/TracyCuiq/Steganographic-Passport)). We appreciate their outstanding works.

## Citation
If you use CHIP in your research or wish to refer to the results published here, please cite our paper as follows.
```bibtex
@article{xu2025chip,
  title={CHIP: Chameleon Hash-based Irreversible Passport for Robust Deep Model Ownership Verification and Active Usage Control},
  author={Xu, Chaohui and Cui, Qi and Chang, Chip-Hong},
  journal={arXiv preprint arXiv:2505.24536},
  year={2025}
}
```

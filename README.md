# CHIP
This repository contains the official implementation of our paper "[CHIP: Chameleon Hash-based Irreversible Passport for Robust Deep Model Ownership Verification and Active Usage Control](https://arxiv.org/abs/2505.24536)". Authors: Chaohui Xu, Qi Cui, and Chip-Hong Chang.

## Introduction
The pervasion of large-scale Deep Neural Networks (DNNs) and their enormous training costs make their intellectual property (IP) protection of paramount importance. Recently introduced passport-based methods attempt to steer DNN watermarking towards strengthening ownership verification against ambiguity attacks by modulating the affine parameters of normalization layers. Unfortunately, neither watermarking nor passport-based methods provide a holistic protection with robust ownership proof, high fidelity, active usage authorization and user traceability for offline access distributed models and multi-user Machine-Learning as a Service (MLaaS) cloud model. In this paper, we propose a Chameleon Hash-based Irreversible Passport (CHIP) protection framework that utilizes the cryptographic chameleon hash function to achieve all these goals. The collision-resistant property of chameleon hash allows for strong model ownership claim upon IP infringement and liable user traceability, while the trapdoor-collision property enables hashing of multiple user passports and licensee certificates to the same immutable signature to realize active usage control. Using the owner passport as an oracle, multiple user-specific triplets, each contains a passport-aware user model, a user passport, and a licensee certificate can be created for secure offline distribution. The watermarked master model can also be deployed for MLaaS with usage permission verifiable by the provision of any trapdoor-colliding user passports. CHIP is extensively evaluated on four datasets and two architectures to demonstrate its protection versatility and robustness.
<div align="center">
  <img width="476" height="285" alt="compressed_CHIP" src="https://github.com/user-attachments/assets/6ca830d6-b1f2-47d9-8d6f-12ef6b3ee03b" />
</div>

## Installation
Our code is bulit on Python 3.8.16, torch 1.11.0, and cuda 11.3
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

## Data

# SRGAN-Image_Superresolution
This repository contains a PyTorch implementation of SRGAN based on CVPR 2017 paper [Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) by Christian Ledig et al. SRGAN is a deep learning-based model designed to perform super-resolution tasks, generating high-resolution (HR) images from low-resolution (LR) inputs.

## Introduction
SRGAN utilizes a Generative Adversarial Network (GAN) framework, where:
* The generator learns to create visually realistic HR images from LR inputs.
* The discriminator ensures the HR images are indistinguishable from real HR images by evaluating perceptual quality.

The main contributions of the SRGAN model:
* Perceptual loss, combining content loss and adversarial loss.
* Enhanced capability to recover finer details, surpassing traditional SR approaches like bicubic interpolation.

## Features
* Implementation of SRGAN using PyTorch.
* Support for training and testing on custom datasets.
* Visual comparison between LR, bicubic upsampling, and SRGAN results.
* PSNR and SSIM metrics for performance evaluation.

## Prerequisites
### Hardware
* Minimum 8GB RAM for efficient training.
* Nvidia GPU with CUDA support(if you want to train the model on GPU)

### Software
* Python3+
* PyTorch
* OpenCV
* VS Code

## Installation
### 1.Clone the git repository:
git clone https://github.com/kodeking-081/SRGAN-Image_Superresolution.git
cd SRGAN-Image_Superresolution

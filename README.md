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

### 2.Install dependencies
pip install -r requirements.txt

## Usage
### Dataset Preparation
#### 1.Collect the High Resolution Images:
Using two separate datasets - DIV2K and CelebHqA, i have trained two different models. The DIV2K dataset consists of 800 high resolution images for training and 200 HR images for validation. Meanwhile, Frpm CelebHq dataset , I sampled 7000 images for training and 3000 images for validation.
You can get both the datasets at [kaggle](https://www.kaggle.com/). Download the dataset and extract it into your dataset folder.

#### 2.Generate low resolution images by downsampling:
Using the [downsample.py](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/downsample.py), generate Low Resolution(LR) image set from HR image set.



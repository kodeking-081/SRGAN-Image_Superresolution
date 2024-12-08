# SRGAN-Image_Superresolution
<p align="center">
  <img src="https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/images/monarch.png" alt="Image 1" width="45%">
  <img src="https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/images/out_srf_14_monarch.png" alt="Image 2" width="45%">
</p>

This repository contains a PyTorch implementation of SRGAN based on CVPR 2017 paper [Photo Realistic Single Image Super-Resolution Using a Generative Adversarial Network](https://arxiv.org/abs/1609.04802) by Christian Ledig et al. SRGAN is a deep learning-based model designed to perform super-resolution tasks, generating high-resolution (HR) images from low-resolution (LR) inputs.

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

### Training
Run the train.py to train the SRGAN model. I have trained the model for 100 epochs with batch size=64.

### Testing
For testing, batch size is set to 1.
* Run testimage.py to test single image.
* Run testvideo.py to test the SRGAN model on video input.
* Run testbenchmark.py to test the SRGAN model on benchmark dataset such as Set5, Set14 etc
* Use testImgSet.py to first convert HR benchmark dataset to LR.*

## Model Architechture
![image](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/Architechture(SRGAN).jpeg)

### Generator
* Based on ResNet with residual blocks.
* Upsampling via sub-pixel convolution layers.

### Discriminator
* Deep CNN with LeakyReLU activations.
* Binary classification output for real/fake.

### Loss Functions
Go through [loss.py](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/loss.py) file.
* Content Loss: MSE or VGG feature loss.
* Adversarial Loss: Binary cross-entropy loss for GAN training.
Get detailed insights on different [losses associated with SRGAN](https://paperswithcode.com/method/srgan)

You can see the detailed document on [SRGAN Architechture](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/archSRGAN.txt).
Also, Go through [model.py](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/model.py) to understand the code structure for architechture.

## Training
The output val results obtained after training are stored at [/training_results/SRF_4](https://github.com/kodeking-081/SRGAN-Image_Superresolution/tree/main/training_results)
### Training Results[(Images)](https://github.com/kodeking-081/SRGAN-Image_Superresolution/tree/main/training_results/SRF_4):
## Epoch 1:
![image](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/training_results/SRF_4/epoch_1_index_1.png)

## Epoch 50 :
![image](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/training_results/SRF_4/epoch_75_index_19.png)

## Epoch 100:
![image](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/training_results/SRF_4/epoch_100_index_20.png)

### Training Results(Graphs):
* DIV2K
![image](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/images/Div2kTrain.svg)

* CELEBhq
![image](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/images/celebtraingraph.svg)

## Testing
Upscale_factor = 4
### Test on a Single Image:
  * DIV2K  Model:
<p align="center">
  <img src="https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/images/comic.png" alt="Image 1" width="45%">
  <img src="https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/images/out_srf_14_comic.png" alt="Image 2" width="45%">
</p>

 * CELEBhq Model:
<p align="center">
  <img src="https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/images/monarch.png" alt="Image 1" width="45%">
  <img src="https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/images/out_srf_14_monarch.png" alt="Image 2" width="45%">
</p>

### Test on Benchmark Dataset:
  SET5
  ### Using [Div2k model](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/epochs/netG_epoch_4_300.pth):
  
  * PSNR= 26.4870 & SSIM=0.7661

  ![image](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/images/Set5benchresult300/baby_psnr_26.4870_ssim_0.7661.png)

  ### Using [Celeb model](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/epochs/netGc_epoch_4_84.pth):

  * PSNR=21.0035 & SSIM=0.7198

  ![image](https://github.com/kodeking-081/SRGAN-Image_Superresolution/blob/main/images/Set5resultG100/woman_psnr_21.0035_ssim_0.7198.png)







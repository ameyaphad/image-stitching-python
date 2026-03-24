# Homography Estimation using Supervised and Unsupervised Deep Learning

This repository contains code which performs homography estimation for image patches/pairs which are either **low resolution** or have **less reliable features**, for whom the traditional method of finding homography using feature extraction and matching would fail. The code is inspired from the following paper: [Link](https://arxiv.org/pdf/1606.03798).

## Supervised Learning Approach

To train a Convolutional Neural Network (CNN) to estimate homography between a pair of images (this network is called the HomographyNet), data or pairs of images are required with the known homogaraphy between them. This is in-general hard to obtain as the 3D movement between the pair of images would be required to obtain the homography between them. 
An easier option is to generate synthetic pairs of images to train a network. Hence, the a small subset of images from the MSCOCO dataset is used to obtain image patches to train the network on.

### Dataset Generation


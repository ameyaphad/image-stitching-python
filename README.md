# Homography Estimation using Supervised and Unsupervised Deep Learning

This repository contains code which performs homography estimation for image patches/pairs which are either **low resolution** or have **less reliable features**, for whom the traditional method of finding homography using feature extraction and matching would fail. The code is inspired from the following paper: [Link](https://arxiv.org/pdf/1606.03798).

## Supervised Learning Approach

To train a Convolutional Neural Network (CNN) to estimate homography between a pair of images (this network is called the HomographyNet), data or pairs of images are required with the known homogaraphy between them. This is in-general hard to obtain as the 3D movement between the pair of images would be required to obtain the homography between them. 
An easier option is to generate synthetic pairs of images to train a network. Hence, the a small subset of images from the MSCOCO dataset is used to obtain image patches to train the network on.

### Dataset Generation
In this, patches are generated from the MSCOCO images with known homographies. While stakcing these patches, augmentation like **Motion Blur** and **Occlusions** are added to the Patch B, so that the process can be generalized to different images and becomes more robust to any noise addition or translation. Motion Blur is an effect which occurs when a camera or object moves during exposure, causing streaks in the image. It is typically simulated by convolving the image with a blur kernel that mimics linear or radial motion. Occlusions, on the other hand, happen when parts of an object are blocked by another object in the scene. Occlusions can be artificially added by overlaying random shapes, noise patches, or real-world objects onto the image to challenge the model’s robustness in recognizing partially visible structures.

The pipeline is shown in the image:
<p align='center'>
    <img src="images/image.png" alt="drawing" width="800"/>
</p>

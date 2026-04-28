## BTS-SAM3D
a lightweight 3D semi-automatic segmentation framework tailored for brain tumor segmentation in MR


## Overview
We propose a lightweight and efficient 3D semi-automatic interactive segmentation framework (BTS-SAM) for accurately segmenting brain tumors from multi-modal magnetic resonance imaging (MRI) by combining a 3D hierarchical encoder based on Variable Shape Mixed Transformer with a feature-augmented 3D mask decoder equipped with multilayer feature fusion (MLFF-3D). We implemented this method based on the open-source machine learning framework PyTorch.

## Demo
![Example Image1](https://github.com/appiek/BTS-SAM3D/blob/main/2020-model_comparison_bar_chart.png)
![Example Image1](https://github.com/appiek/BTS-SAM3D/blob/main/2021-model_comparison_bar_chart.png)

## Dependencies
* Python 3.8 or higher version
* Pytorch 2.4.0
* MONAI 1.5.1
* Nibabel 5.3.2
* TorchIO 0.21.0
* SimpleITK 2.5.2
* Timm 1.0.20
* Einops 0.8.1
* Albumentations 2.0.8
* opencv-python-headless 4.12.0
* Numpy
* Scipy

## Composition of code
1. train.py: Brain Tumor Segmentation Training Based on BTS-SAM and Progressive Prompt Training Strategy
2. /segment_anything: model construction
3. 

## Quick Start

# CVPR18-SFTGAN

### [[PyTorch(Under Construction)]](https://github.com/xinntao/CVPR18-SFTGAN)   [[project page]](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/)   [[paper]](https://arxiv.org/abs/1804.02815)

Torch implementation for [Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform](https://arxiv.org/abs/1804.02815)

Training code is coming soon...

### Table of Contents
1. [Introduction](#introduction)
1. [Citation](#citation)
1. [Requirements and Dependencies](#requirements-and-dependencies)

### Introduction
We have explored the use of semantic segmentation maps as categorical prior for SR.

A novel Spatial Feature Transform (SFT) layer has been proposed to efficiently incorporate the categorical conditions into a CNN network.

For more details, please check out our [project webpage](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/) and [paper](https://arxiv.org/abs/1804.02815).

<img src='imgs/network_structure.png' align="center">

### Citation

If you find the code and datasets useful in your research, please cite:

    @inproceedings{wang2018sftgan,
        author = {Xintao Wang, Ke Yu, Chao Dong and Chen Change Loy},
        title = {Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2018}

### Requirements and Dependencies
- Torch
- cuda & cudnn
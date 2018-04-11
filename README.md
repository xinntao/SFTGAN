# CVPR18-SFTGAN

### [PyTorch(Under Construction)]   [[project page]](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/)   [[paper]](https://arxiv.org/abs/1804.02815)

Torch implementation for [Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform](https://arxiv.org/abs/1804.02815)

Training code is coming soon...

### Table of Contents
1. [Introduction](#introduction)
1. [Requirements and Dependencies](#requirements-and-dependencies)
1. [Test](#test)
1. [Citation](#citation)

### Introduction
We have explored the use of semantic segmentation maps as categorical prior for SR.

A novel Spatial Feature Transform (SFT) layer has been proposed to efficiently incorporate the categorical conditions into a CNN network.

For more details, please check out our [project webpage](http://mmlab.ie.cuhk.edu.hk/projects/SFTGAN/) and [paper](https://arxiv.org/abs/1804.02815).

<img src='imgs/network_structure.png' align="center">


### Requirements and Dependencies
- Torch
- cuda & cudnn
- other torch dependencies, e.g. nngraph/paths/image (install them by `luarocks install ...`)

### Test
We test our model with Titan X/Xp GPU.

1. Download segmentation model (OutdoorSceneSeg_bic_iter_30000.t7) and SFT-GAN model (SFT-GAN.t7) from [google drive](https://drive.google.com/drive/folders/1kFxjStgGxrKCdNzaa0Cwje5gR3OR-q1r?usp=sharing). Put them in the `model` folder.
1. There are 2 sample images putted in `data/samples`  folder.
1. Run `th test_seg.t7`. The segmentation results are then generated in `data/` with `_segprob/_colorimg/_byteimg` suffix.
1. Run `th test_SFT-GAN.lua`. The results are then in `data/` with prefix `rlt_`.

### Citation
If you find the code and datasets useful in your research, please cite:

    @inproceedings{wang2018sftgan,
        author = {Xintao Wang, Ke Yu, Chao Dong and Chen Change Loy},
        title = {Recovering Realistic Texture in Image Super-resolution by Deep Spatial Feature Transform},
        booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        year = {2018}

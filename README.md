# Swin-iPASSR
This is the source code of the 7th place solution for stereo image super resolution task in 2022 CVPR NTIRE challenge.

## Network Architecture:

<p align="center">
<img src="./figs/network.png" alt="drawing" width="100%" height="700%"/>
    <h4 align="center">An overview of our Swin-iPASSR network.</h4>
</p>

## Codes and Models:
### Requirement:
* **PyTorch 1.3.0, torchvision 0.4.1. The code is tested with python=3.7, cuda=9.0.**
* **Matlab (For training/test data generation and performance evaluation)**

### Train:
* **Download the training sets from [Baidu Drive](https://pan.baidu.com/s/173UGmmN0rtOUghIT40oy8w) (Key: NUDT) and unzip them to `./data/train/`.** 
* **Run `./data/train/GenerateTrainingPatches.m` to generate training patches.**
* **Run `train.py` to perform training. Checkpoint will be saved to  `./log/`.**

### Test:
* **Download the test sets and unzip them to `./data`. Here, we provide the full test sets used in our paper on [Google Drive](https://drive.google.com/file/d/1LQDUclNtNZWTT41NndISLGvjvuBbxeUs/view?usp=sharing) and [Baidu Drive](https://pan.baidu.com/s/1SIYGcMBEDDZ0wYrkxL9bnQ) (Key: NUDT).** 
* **Run `test.py` to perform a demo inference. Results (`.png` files) will be saved to `./results`.**
* **Run `evaluation.m` to calculate PSNR and SSIM scores.**

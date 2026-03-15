# Denoising Diffusion Probabilistic Models

## Background

在本文发表前，深度生成模型（Deep generative models）（如VAE，GAN等）已经实现了大量高质量的样本生成，包括图像/文本/音频等。此外，能量基模型（Energy-Based Models, EBMs）和得分匹配（Score Matching），它们在图像生成方面取得了显著进展，产生的图像质量与生成对抗网络（Generative Adversarial Networks, GANs）相媲美。

### Score Matching

传统的匹配概率密度函数的挑战在于难以直接计算概率密度函数或是包含不容易处理的配分函数（归一化常数）。得分匹配的好处在于相比于直接匹配概率密度函数，匹配得分能够允许模型训练绕过配分函数的直接计算，因为得分（概率密度的梯度）不直接依赖于它。这使得得分匹配特别适用于能量基模型（EBMs）和其他未归一化模型。

此外，得分匹配关注数据分布的局部结构（通过梯度捕获），这有助于更精细地学习和理解数据特征，从而提高模型的表现力。

得分匹配因此被用于 Diffusion 相关的工作中并取得了出色的成绩。

[得分匹配](https://zhuanlan.zhihu.com/p/556175230)

## Introduction

本文主要展现在扩散概率模型（diffusion probabilistic models）方面取得的进步。

## DDPM code implementation

https://zhuanlan.zhihu.com/p/617895786

https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-

https://learnopencv.com/denoising-diffusion-probabilistic-models/

## Generative Modeling by Estimating Gradients of the Data Distribution

https://yang-song.net/blog/2021/score/
# 生成模型

生成模型的核心目标：学习数据分布 $p(x)$，从中采样生成新数据。

## 知识点索引

### 基础范式
- [VAE 详解](VAE详解.md) — AE→VAE（ELBO、重参数化、KL 解析解、模糊原因）→ CVAE → VQ-VAE
- [Diffusion 详解](Diffusion详解.md) — DDPM（前向加噪、反向去噪、预测噪声）、Score Matching、SDE 统一框架

### 采样与条件控制
- [采样加速与条件生成](采样加速与条件生成.md) — DDIM、Classifier Guidance、CFG、Consistency Models、DPM-Solver

### 现代架构与范式
- [Stable Diffusion 详解](Stable_Diffusion详解.md) — LDM 架构（VAE+U-Net/DiT+Text Encoder）、ControlNet、IP-Adapter、各版本对比
- [Flow Matching 详解](Flow_Matching详解.md) — 条件 Flow Matching、Rectified Flow、SD3/FLUX/Sora 的训练范式

## 范式对比

| | VAE | GAN | Diffusion (DDPM) | Flow Matching |
|---|---|---|---|---|
| **训练目标** | 最大化 ELBO | 极小极大博弈 | 预测噪声 | 预测速度场 |
| **生成质量** | 模糊 | 锐利但多样性差 | 高质量 | **最高质量** |
| **采样速度** | 快（一步） | 快（一步） | 慢（多步） | 较快（少步 ODE） |
| **代表模型** | VQ-VAE | StyleGAN | DDPM, SD 1/2 | SD3, FLUX, Sora |

## 发展时间线

```
2013  VAE (Kingma & Welling)
2014  GAN (Goodfellow et al.)
2017  VQ-VAE (van den Oord et al.)
2020  DDPM (Ho et al.) ← 扩散模型真正起飞
2021  DDIM、Classifier Guidance、Score SDE 统一框架
2022  Stable Diffusion / DALL-E 2 / Imagen ← 文生图爆发
      Flow Matching (Lipman et al.)、Rectified Flow (Liu et al.)
2023  SDXL、Consistency Models、DALL-E 3、ControlNet
2024  SD3 (DiT + Flow Matching)、FLUX ← Flow Matching 成为主流
2025  视频生成 (Sora)、3D 生成
```

## 面试高频问题

> **Q: VAE 和 AE 的本质区别？**
> A: AE 只优化重建，$z$ 空间无结构，不能生成。VAE 通过 KL 正则强制 $z$ 服从 $\mathcal{N}(0,I)$，使得任意采样的 $z$ 都能解码出有意义的样本。

> **Q: 重参数化技巧解决了什么问题？**
> A: "从分布中采样"不可导。把 $z = \mu + \sigma \odot \epsilon$ 重写后，随机性移到外部输入 $\epsilon$ 上，梯度可以正常反传。

> **Q: VAE 为什么生成模糊？**
> A: z 空间维度低导致编码分布重叠 + 高斯似然只有单峰。重叠区的 z 对应多种 x，Decoder 只能输出折中均值。

> **Q: DDPM 的训练目标是什么？**
> A: $L = \mathbb{E}\|\epsilon - \epsilon_\theta(x_t, t)\|^2$。随机选时间步加噪，让网络预测加了什么噪声。

> **Q: Diffusion 和 Score Matching 是什么关系？**
> A: 预测噪声等价于预测 score 的缩放版。Yang Song 用 SDE 统一了 DDPM 和 NCSN。

> **Q: Stable Diffusion 为什么要用 VAE？**
> A: 在像素空间做扩散太贵（786K 维）。VAE 压缩到 64×64×4 的潜在空间后再做扩散。

> **Q: Flow Matching 和 Diffusion 的区别？**
> A: DDPM 需要噪声调度系数，预测噪声。Flow Matching 直接线性插值，预测速度。公式更简洁，SD3/FLUX/Sora 都已转向 Flow Matching。

---
[返回视觉](../README.md) | [返回总目录](../../README.md)

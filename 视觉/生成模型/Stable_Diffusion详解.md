# Stable Diffusion 详解：Latent Diffusion Model

> 参考：[Rombach et al., 2022 - High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

---

## 1. 动机

直接在像素空间做扩散计算量巨大。一张 $512 \times 512 \times 3$ 的图片有 786K 维，每一步去噪都要在这么高的维度上做前向推理。

```
像素空间扩散:  在 512×512×3 上做 T 步去噪   → 慢，贵
潜在空间扩散:  在 64×64×4 上做 T 步去噪     → 快 ~48 倍，质量几乎不变
```

**核心思路**：先用 VAE 把图片压缩到低维潜在空间，然后在潜在空间做扩散。这就是 **Latent Diffusion Model（LDM）**。

---

## 2. 架构总览

```
训练:
  图片 x (512×512×3)
    → VAE Encoder → z₀ (64×64×4)           ← 压缩到潜在空间（空间 8x，通道 3→4）
    → 前向扩散加噪 → z_t
    → U-Net 预测噪声 ε_θ(z_t, t, c)        ← 在潜在空间去噪
    → 简化 loss: ||ε - ε_θ||²

推理:
  z_T ~ N(0, I)  (64×64×4)
    → U-Net 逐步去噪（DDIM 加速）→ z₀
    → VAE Decoder → 图片 (512×512×3)        ← 解码回像素空间
```

三个独立组件：
1. **VAE**：图片 ↔ 潜在表示（预训练后冻结）
2. **U-Net / DiT**：在潜在空间做去噪（核心训练对象）
3. **Text Encoder**：文本 → 嵌入向量（预训练后冻结，如 CLIP / T5）

---

## 3. VAE 组件

### 作用

- **编码**：将 512×512×3 图片压缩为 64×64×4 的潜在表示
- **解码**：将去噪后的潜在表示还原为高分辨率图片

### 技术细节

Stable Diffusion 的 VAE 是 **KL-regularized autoencoder**（介于标准 VAE 和 VQ-VAE 之间）：

```
与标准 VAE 的区别:
  - 使用很小的 KL 权重（~10⁻⁶）→ 几乎是纯 AE，重建质量优先
  - 空间下采样 8 倍（3 次 stride-2 卷积）
  - 通道数 3 → 4（略微过完备，保留更多信息）

与 VQ-VAE 的区别:
  - 不做离散量化 → 潜在空间是连续的 → 适合做扩散
```

### 为什么 VAE 单独预训练？

1. VAE 的训练目标（重建质量）和扩散的训练目标（预测噪声）不同
2. 分开训练可以各自优化，更稳定
3. 扩散训练时冻结 VAE，只训 U-Net → 计算量大幅减少

---

## 4. 条件注入

### Cross-Attention（SD 1/2 的方式）

U-Net 通过 **Cross-Attention** 注入文本条件：

```
文本 "a cat sitting on a chair"
  → CLIP Text Encoder → 文本嵌入 (77×768)
  → 作为 Cross-Attention 的 K, V

U-Net 中间层特征 → 作为 Q
  → Cross-Attention(Q, K, V) → 融合文本信息的特征
```

### Joint Attention（SD3 / FLUX 的方式）

SD3 使用 **MM-DiT**，文本 token 和图像 token 在同一个 Transformer 中做联合注意力：

```
SD 1/2:  图像特征(Q) × 文本特征(K,V)  → Cross-Attention（文本只能被动影响图像）
SD 3:    [图像 tokens | 文本 tokens]   → Joint Self-Attention（双向交互）
```

### 条件注入方式汇总

| 方法 | 模型 | 原理 |
|------|------|------|
| Cross-Attention | SD 1/2, SDXL | 文本作为 K,V，图像作为 Q |
| Joint Attention | SD3, FLUX | 文本和图像 token 拼接后做 Self-Attention |
| AdaLN (Adaptive LayerNorm) | DiT | 时间步和类别通过 LayerNorm 的 scale/shift 注入 |
| Concatenation | ControlNet | 条件图（边缘/深度）拼接到输入通道 |

---

## 5. Stable Diffusion 各版本对比

| 版本 | 年份 | 骨干 | 训练范式 | Text Encoder | 分辨率 |
|------|------|------|---------|-------------|--------|
| **SD 1.5** | 2022 | U-Net | DDPM | CLIP ViT-L/14 | 512×512 |
| **SD 2.1** | 2022 | U-Net | DDPM | OpenCLIP ViT-H/14 | 768×768 |
| **SDXL** | 2023 | U-Net (大) | DDPM | CLIP + OpenCLIP 双编码器 | 1024×1024 |
| **SD3** | 2024 | MM-DiT | Flow Matching | CLIP + T5-XXL | 1024×1024 |
| **FLUX** | 2024 | DiT | Flow Matching | CLIP + T5-XXL | 1024×1024 |

### 关键演进

```
骨干网络:   U-Net (SD1/2/XL) → DiT / MM-DiT (SD3, FLUX)
训练范式:   DDPM (预测噪声)  → Flow Matching (预测速度)
文本编码:   CLIP only        → CLIP + T5 双编码器
分辨率:     512              → 1024+
```

---

## 6. ControlNet：精细控制

ControlNet（Zhang et al., 2023）在不修改原模型的前提下，增加空间条件控制：

```
原始 U-Net:     z_t → U-Net → ε_θ
                      ↑
ControlNet:     条件图（边缘/深度/姿态）→ ControlNet 副本 → 注入 U-Net

做法:
  1. 复制 U-Net 的 Encoder 部分作为 ControlNet
  2. ControlNet 输入条件图，输出特征
  3. 通过 zero convolution 将特征加到原始 U-Net 的对应层
  4. 原始 U-Net 权重冻结，只训练 ControlNet 部分
```

支持的条件类型：Canny 边缘、深度图、人体姿态（OpenPose）、语义分割图、法线图等。

---

## 7. IP-Adapter：图像条件

IP-Adapter（Ye et al., 2023）用图像作为条件（而非文本），实现"以图生图"：

```
参考图片 → CLIP Image Encoder → 图像嵌入
         → 可学习的投影层
         → 作为额外的 Cross-Attention 的 K, V（与文本 Cross-Attention 并行）
```

---

## 8. 推理优化

| 方法 | 效果 |
|------|------|
| DDIM（10-50 步） | 基础加速 |
| DPM-Solver（10-20 步） | 更高效的 ODE 求解 |
| LCM（2-4 步） | 蒸馏后极速生成 |
| Torch.compile | 计算图优化，~30% 加速 |
| xFormers / Flash Attention | 注意力计算加速 |
| 半精度 (FP16/BF16) | 显存和速度双优化 |

---

**相关文档**：
- [VAE 详解](VAE详解.md) — LDM 中 VAE 组件的理论基础
- [Diffusion 详解](Diffusion详解.md) — DDPM 训练目标
- [采样加速与条件生成](采样加速与条件生成.md) — DDIM、CFG 详解
- [Flow Matching 详解](Flow_Matching详解.md) — SD3/FLUX 使用的训练范式
- [对比学习与CLIP详解](../表示学习/对比学习与CLIP详解.md) — Text Encoder 的基础

[返回上级](README.md) | [返回视觉](../README.md) | [返回总目录](../../README.md)

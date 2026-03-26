# VLM 三大组件详解

深入解析 VLM 的三大核心组件：Vision Encoder、投影模块、LLM 骨干。

## 一、Vision Encoder（视觉编码器）

### 1.1 作用

将图像编码为一组特征向量（视觉 token），供后续模块处理。

```
图像 (H×W×3) → Vision Encoder → N 个 patch 特征 (N×D)
```

具体来说，ViT 类编码器会将图像切成固定大小的 patch（如 14×14 像素），每个 patch 经过线性投影后加上位置编码，通过多层 Transformer 编码。

### 1.2 主流 Vision Encoder 对比

| 编码器 | 预训练方式 | 参数量 | 特点 | 使用的 VLM |
|--------|-----------|--------|------|-----------|
| **CLIP ViT-L/14** | 对比学习（图文对） | 304M | 语义对齐好，最主流 | LLaVA、MiniGPT-4 |
| **CLIP ViT-bigG** | 对比学习 | 1.8B | 更大更强 | EVA-CLIP → 多模型使用 |
| **SigLIP ViT-SO400M** | Sigmoid 对比学习 | 400M | 比 CLIP 更稳定，无需全局归一化 | PaliGemma、InternVL2、LLaVA-OneVision |
| **DINOv2 ViT-L** | 自监督（自蒸馏） | 304M | 局部特征突出，密集预测好 | 常与 CLIP 互补使用 |
| **InternViT-6B** | 对比学习 + 渐进训练 | 6B | 最大的开源 ViT | InternVL 系列 |
| **EVA-02 ViT** | MIM + CLIP 蒸馏 | 304M~1B | 融合重建和对比学习优势 | EVA-CLIP 系列 |

### 1.3 CLIP ViT vs SigLIP：核心区别

CLIP 使用 softmax 对比损失（InfoNCE），需要全 batch 负样本；SigLIP 改用 sigmoid 损失，对每个图文对独立计算：

```
CLIP 损失 (InfoNCE):
  L = -log( exp(sim(I_i, T_i)/τ) / Σ_j exp(sim(I_i, T_j)/τ) )
  → 需要 batch 内所有负样本，计算量 O(N²)

SigLIP 损失 (Sigmoid):
  L = -log σ(z_ij · sim(I_i, T_j)/τ)     z_ij = +1 if matched, -1 if not
  → 每对独立计算，可用更大 batch、更易分布式训练
```

**实际影响**：SigLIP 训练更稳定，在 VLM 中效果通常不弱于甚至优于 CLIP，已成为新一代 VLM 的主流选择。

### 1.4 CLIP 特征 vs DINO 特征

| 特性 | CLIP | DINOv2 |
|------|------|--------|
| 预训练信号 | 图文对比（语义级） | 自蒸馏（像素/patch 级） |
| 语义理解 | 强（天然与语言对齐） | 较弱（没见过文本） |
| 局部特征 | 较弱（倾向全局语义） | 强（patch 级特征丰富） |
| 适合任务 | 图像描述、VQA | 分割、定位、密集预测 |
| 在 VLM 中的角色 | 主编码器 | 辅助/互补编码器 |

部分研究（如 Cambrian-1）探索了**多编码器融合**，同时使用 CLIP + DINOv2 + 其他编码器，取各家之长。

### 1.5 输出维度示例

以 CLIP ViT-L/14 处理 336×336 图像为例：

```
图像 336×336
  → 切成 14×14 的 patch → 24×24 = 576 个 patch
  → 每个 patch 编码为 1024 维向量
  → 输出: 576 × 1024 的特征矩阵

加上 [CLS] token → 577 × 1024（但多数 VLM 只用 patch token，不用 CLS）
```

---

## 二、投影模块（Vision-Language Connector）

### 2.1 为什么需要投影模块？

Vision Encoder 和 LLM 是独立预训练的，它们的特征空间不同：

```
Vision Encoder 输出: 576 × 1024 (ViT 的维度)
LLM 输入需要:       N × 4096  (如 LLaMA-7B 的维度)

投影模块的任务: 1024 维 → 4096 维，同时进行语义对齐
```

### 2.2 主流投影方法详解

#### 方法一：线性投影 / MLP（LLaVA 方案）

最简单直接的方法：

```python
# LLaVA-1.0: 单层线性
projector = nn.Linear(1024, 4096)

# LLaVA-1.5: 两层 MLP + GELU
projector = nn.Sequential(
    nn.Linear(1024, 4096),
    nn.GELU(),
    nn.Linear(4096, 4096)
)
```

**特点**：
- 不改变 token 数量（576 个进 → 576 个出）
- 结构简单，训练稳定
- LLaVA 证明这就够用了，Q-Former 不是必须的

#### 方法二：Q-Former（BLIP-2 方案）

用一组**可学习的 query token** 通过交叉注意力从视觉特征中"提取"信息：

```
可学习 Query (32 个, 每个 768 维)
    │
    ├── Self-Attention (query 之间交互)
    │
    └── Cross-Attention (query 注意力到 ViT 输出)
            │
            ↓
    压缩后的视觉特征 (32 × 768)
            │
    Linear → LLM 输入维度 (32 × LLM_dim)
```

**特点**：
- 将数百个视觉 token **压缩到 32 个**，大幅减少计算量
- 但可能丢失细粒度信息（OCR、小物体识别变差）
- 结构复杂，训练需要多阶段

#### 方法三：Perceiver Resampler（Flamingo 方案）

与 Q-Former 类似的注意力池化，但结构更简洁：

```
可学习 Query (64 个)
    │
    └── Cross-Attention × L 层 (query 注意力到 ViT 输出)
            │
            ↓
    压缩后的视觉特征 (64 × LLM_dim)
```

**与 Q-Former 的区别**：没有 Q-Former 的多阶段预训练目标（ITC/ITM/ITG），直接端到端训练。

#### 方法四：Pixel Shuffle 下采样（InternVL2 方案）

不用注意力机制，而是通过**空间下采样**减少 token：

```
输入: H×W×C 的特征图 (如 24×24×1024)
Pixel Shuffle (2×2):
  将相邻 2×2 的 patch 合并
  24×24×1024 → 12×12×4096
  token 数从 576 减少到 144，但每个 token 信息更丰富

然后再接一个 MLP 调整维度
```

**特点**：
- 计算简单高效
- 保留了空间结构信息
- token 数减少 4 倍，在推理效率和信息保留之间取得平衡

### 2.3 投影方法对比总结

| 方法 | 输出 token 数 | 信息保留 | 计算开销 | 复杂度 |
|------|-------------|---------|---------|--------|
| MLP | 不变（如 576） | 最好 | 最低 | 最简单 |
| Q-Former | 固定（如 32） | 一般 | 中等 | 复杂 |
| Perceiver | 固定（如 64） | 一般 | 中等 | 中等 |
| Pixel Shuffle | 1/4（如 144） | 较好 | 低 | 简单 |

**趋势**：早期 VLM 偏好 Q-Former 压缩（省计算），后来发现保留更多视觉 token 对细粒度任务很重要，MLP + 下采样成为主流。

---

## 三、LLM 骨干（语言模型）

### 3.1 LLM 在 VLM 中的角色

LLM 是 VLM 的"大脑"，负责：
1. **融合理解**：将视觉 token 和文本 token 在同一个 Transformer 中联合处理
2. **推理能力**：继承 LLM 预训练获得的逻辑推理、世界知识
3. **语言生成**：生成自然语言回答

### 3.2 主流 LLM 选择

| VLM | LLM 骨干 | 参数量 | 说明 |
|-----|---------|--------|------|
| LLaVA-1.5 | Vicuna (LLaMA 微调) | 7B / 13B | 最早的经典选择 |
| LLaVA-OneVision | Qwen2 | 0.5B / 7B / 72B | 新一代多尺寸 |
| InternVL2 | InternLM2 | 1B ~ 76B | 全系列覆盖 |
| Qwen-VL2 | Qwen2 | 2B / 7B / 72B | 阿里自研 |
| DeepSeek-VL2 | DeepSeek-MoE | 16B (激活 2.8B) | MoE 高效推理 |
| Phi-3-Vision | Phi-3 | 4.2B | 微软小模型 |
| PaliGemma | Gemma | 2B / 9B | Google 开源 |

### 3.3 LLM 在 VLM 训练中的状态

| 训练阶段 | LLM 状态 | 原因 |
|---------|---------|------|
| Stage 1 (对齐) | **冻结** | 只学投影，避免破坏 LLM 已有能力 |
| Stage 2 (多任务) | **解冻/部分解冻** | LLM 需要学习处理视觉信息 |
| Stage 3 (指令微调) | **全量微调或 LoRA** | 精调回答质量 |

### 3.4 LLM 如何处理视觉 token

视觉 token 和文本 token 在 LLM 内部的处理方式完全相同——都是参与 Self-Attention 的 token 序列：

```
LLM 输入序列:
[视觉token_1] [视觉token_2] ... [视觉token_576] [User:] [文本token_1] [文本token_2] ...

在 Self-Attention 中:
- 每个文本 token 可以 attend 到所有视觉 token（"看到"图像信息）
- 每个视觉 token 也可以 attend 到文本 token（理解问题上下文）
- 这就是为什么 VLM 能做到"看图回答问题"
```

**关键洞察**：VLM 并没有为视觉设计特殊的注意力机制，而是复用了 LLM 原有的 Self-Attention，让视觉和语言在同一个空间里自然交互。

---

## 组件选型的发展趋势

```
2023 年初 (BLIP-2):
  CLIP ViT-G (冻结) + Q-Former (可训) + 冻结 LLM
  → 只训 Q-Former，参数高效

2023 年中 (LLaVA):
  CLIP ViT-L (冻结) + 简单 MLP (可训) + LLM (可训)
  → 证明简单架构就够用

2024 年 (InternVL2, Qwen-VL2):
  大 ViT (可训) + Pixel Shuffle/MLP + 大 LLM (可训)
  → 端到端训练，全部解冻，追求极限性能

趋势: 更大的 ViT + 更简单的投影 + 更大的 LLM + 全量训练
```

---

**相关文档**：
- [VLM 概述](VLM概述.md) — VLM 总览与入门
- [VLM 主流架构详解](VLM主流架构详解.md) — 具体模型的架构设计
- [对比学习与CLIP详解](../../视觉/表示学习/对比学习与CLIP详解.md) — CLIP Vision Encoder 原理
- [DINO详解](../../视觉/表示学习/DINO详解.md) — DINOv2 特征详解

[返回上级](README.md) | [返回总目录](../../README.md)

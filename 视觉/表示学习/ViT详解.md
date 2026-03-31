# ViT 详解

Vision Transformer — 将 Transformer 从 NLP 引入视觉，开启视觉基础模型时代。

> **核心论文**：
> - ViT: *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale* (Dosovitskiy et al., ICLR 2021, Google)
> - DeiT: *Training data-efficient image transformers & distillation through attention* (Touvron et al., ICML 2021, Meta/FAIR)
> - Swin Transformer: *Swin Transformer: Hierarchical Vision Transformer using Shifted Windows* (Liu et al., ICCV 2021 Best Paper, Microsoft)
> - Swin V2: *Swin Transformer V2: Scaling Up Capacity and Resolution* (Liu et al., CVPR 2022, Microsoft)

---

## 一、ViT (2020, Google)

### 1.1 背景与动机

2020 年之前，计算机视觉的主流架构是 CNN（ResNet、EfficientNet 等）。CNN 具有两个强归纳偏置（inductive bias）：

| 归纳偏置 | 含义 | 优势 | 局限 |
|----------|------|------|------|
| **局部性（Locality）** | 卷积核只看局部邻域 | 参数少、高效 | 感受野有限，需要堆叠层数才能看到全局 |
| **平移等变性（Translation Equivariance）** | 卷积核权重共享 | 对平移鲁棒 | 对旋转、缩放等其他变换无天然不变性 |

与此同时，Transformer 在 NLP 领域已经证明了其强大的建模能力（BERT、GPT）。Transformer 的核心优势是**全局自注意力**——每个 token 可以直接关注序列中的所有其他 token，不受局部性限制。

ViT 的核心问题：**能否把 Transformer 直接用于图像，完全不用卷积？**

### 1.2 核心做法

ViT 的思路极其简洁：**把图像当成一句话，patch 当成 word**。

```
输入图像 (H × W × C)，例如 224 × 224 × 3
  │
  ▼
切成 N 个不重叠 patch，每个 P × P
  N = (H × W) / P² = (224 × 224) / 16² = 196 个 patch
  │
  ▼
每个 patch 展平为向量 (P² × C = 16 × 16 × 3 = 768 维)
  │
  ▼
线性投影 (Linear Projection): ℝ^(P²C) → ℝ^D
  │
  ▼
前面拼一个可学习的 [CLS] token，总共 N+1 = 197 个 token
  │
  ▼
加上可学习的位置编码 E_pos ∈ ℝ^{(N+1) × D}
  │
  ▼
┌──────────────────────────────────┐
│  标准 Transformer Encoder        │
│  L 层 (MSA + FFN + LayerNorm)    │
└──────────────────────────────────┘
  │
  ▼
取 [CLS] token 的输出 → MLP Head → 分类
```

数学表述：

$$\mathbf{z}_0 = [\mathbf{x}_\text{cls};\; \mathbf{x}_1^p \mathbf{E};\; \mathbf{x}_2^p \mathbf{E};\; \cdots;\; \mathbf{x}_N^p \mathbf{E}] + \mathbf{E}_\text{pos}$$

$$\mathbf{z}_\ell' = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}, \quad \ell = 1, \ldots, L$$

$$\mathbf{z}_\ell = \text{FFN}(\text{LN}(\mathbf{z}_\ell')) + \mathbf{z}_\ell', \quad \ell = 1, \ldots, L$$

$$\mathbf{y} = \text{LN}(\mathbf{z}_L^0)$$

其中 $\mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 是 patch embedding 矩阵，$\mathbf{E}_\text{pos} \in \mathbb{R}^{(N+1) \times D}$ 是位置编码。

### 1.3 Patch Embedding 详解

Patch Embedding 是 ViT 将图像转为序列的关键步骤：

```python
# 实现方式 1：展平 + 线性层
patches = rearrange(image, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=16, p2=16)
patch_embed = nn.Linear(16 * 16 * 3, D)  # 768 → D
tokens = patch_embed(patches)  # (B, 196, D)

# 实现方式 2：等价的卷积实现（实际代码常用）
patch_embed = nn.Conv2d(3, D, kernel_size=16, stride=16)
tokens = patch_embed(image).flatten(2).transpose(1, 2)  # (B, 196, D)
```

两种实现数学上完全等价，但 Conv2d 实现更高效（利用 GPU 的卷积优化）。

### 1.4 位置编码

ViT 中 patch 被展平为 1D 序列后，空间位置信息丢失，必须通过位置编码补回来。

| 类型 | 做法 | 效果 |
|------|------|------|
| **1D Learnable**（ViT 默认） | 可学习的 $\mathbf{E}_\text{pos} \in \mathbb{R}^{(N+1) \times D}$，每个位置一个向量 | 实验中表现最好 |
| **2D Learnable** | 行列分开编码，$\mathbf{E}_\text{row} + \mathbf{E}_\text{col}$ | 与 1D 效果几乎无差别 |
| **Sinusoidal（固定）** | 类似原始 Transformer 的正弦余弦编码 | 略差于 learnable |
| **无位置编码** | 不加任何位置信息 | 性能明显下降（~3%） |

ViT 论文的一个有趣发现：**学到的 1D 位置编码自动捕获了 2D 空间结构**——相邻位置的编码向量余弦相似度高，同行/同列的编码也呈现出明显的模式。

### 1.5 [CLS] Token 的作用

[CLS] token 借鉴自 BERT：

- 它是一个**可学习的向量**，拼在 patch token 序列最前面
- 经过 L 层 Transformer 后，[CLS] token 通过自注意力聚合了所有 patch 的信息
- 最终用 [CLS] 的输出向量做分类

替代方案：也可以不用 [CLS]，而是对所有 patch token 的输出做 **Global Average Pooling (GAP)**，效果相当。DeiT 和后续工作中 GAP 逐渐成为主流。

### 1.6 模型配置

| 模型 | 层数 L | 隐藏维度 D | 头数 h | 参数量 |
|------|--------|-----------|--------|--------|
| ViT-Base/16 | 12 | 768 | 12 | 86M |
| ViT-Large/16 | 24 | 1024 | 16 | 307M |
| ViT-Huge/14 | 32 | 1280 | 16 | 632M |

命名规则：`ViT-{Size}/{Patch Size}`，如 ViT-B/16 表示 Base 模型、patch 大小 16。

### 1.7 自注意力复杂度分析

对于输入序列长度 $N$（patch 数量），自注意力的计算复杂度为：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

- $\mathbf{Q}\mathbf{K}^\top$ 的计算：$O(N^2 d)$
- Softmax 后乘以 $\mathbf{V}$：$O(N^2 d)$
- 总复杂度：$O(N^2 d)$

其中 $N = \frac{H \times W}{P^2}$。对于 224×224 图像、patch size 16：$N = 196$，计算量可接受。但若提高分辨率到 448×448：$N = 784$，计算量增加 16 倍（$N^2$ 关系）。**这是 ViT 的核心瓶颈——分辨率扩展性差。**

### 1.8 预训练策略与数据需求

这是 ViT 最重要的发现之一：

| 预训练数据 | 数据量 | ViT-L Top-1 | ResNet152x4 Top-1 |
|-----------|--------|-------------|-------------------|
| ImageNet-1K | 1.3M | ~76% | ~81% |
| ImageNet-21K | 14M | ~83% | ~83% |
| **JFT-300M** | 300M | **~88%** | ~87% |

**关键结论**：

- **小数据集上 ViT 不如 CNN**：因为 Transformer 缺少局部性和平移等变性等归纳偏置，需要从数据中学到这些模式
- **大数据集上 ViT 超越 CNN**：当数据足够多时，归纳偏置反而成为限制，Transformer 的灵活性成为优势
- 数据量的"拐点"大约在 **10M-100M** 级别

---

## 二、DeiT (2021, Meta/FAIR)

### 2.1 动机

ViT 的最大问题：**需要 JFT-300M（Google 私有数据集）才能训练好**，普通研究者只有 ImageNet-1K（1.3M 图像），直接训练 ViT 效果很差。

DeiT 的目标：**仅用 ImageNet-1K + 知识蒸馏 + 强数据增强，让 ViT 达到 SOTA 性能。**

### 2.2 Distillation Token 机制

DeiT 的核心创新是引入一个专门用于蒸馏的 **distillation token**：

```
[CLS] token    Patch 1    Patch 2  ...  Patch N    [DIST] token
   ↓              ↓          ↓              ↓           ↓
   ↓         Transformer Encoder (L 层)               ↓
   ↓              ↓          ↓              ↓           ↓
分类头 → CE Loss                              蒸馏头 → 蒸馏 Loss
 (与真实标签)                                   (与教师预测)
```

$$\mathcal{L} = (1 - \lambda) \cdot \mathcal{L}_\text{CE}(\psi(\mathbf{z}_\text{cls}),\; y) + \lambda \cdot \mathcal{L}_\text{distill}(\psi(\mathbf{z}_\text{dist}),\; y_t)$$

其中：
- $\mathbf{z}_\text{cls}$ 是 [CLS] token 的输出，$\mathbf{z}_\text{dist}$ 是 distillation token 的输出
- $y$ 是真实标签，$y_t$ 是教师模型的预测
- 教师模型通常是 **RegNetY-16GF**（CNN），不是 Transformer

**两种蒸馏方式**：

| 蒸馏方式 | Loss | 效果 |
|----------|------|------|
| **Soft distillation** | KL 散度：$\text{KL}(\sigma(\mathbf{z}_t / \tau) \| \sigma(\mathbf{z}_\text{dist} / \tau))$ | 略差 |
| **Hard distillation** | 交叉熵：$\text{CE}(\psi(\mathbf{z}_\text{dist}), \arg\max(y_t))$ | **更好**（DeiT 默认） |

有趣发现：[CLS] 和 [DIST] token 学到的表示是不同的——[CLS] 更像 Transformer，[DIST] 更像 CNN 教师。

### 2.3 数据增强策略

DeiT 能在 ImageNet-1K 上成功训练 ViT，强数据增强功不可没：

| 增强方法 | 作用 |
|----------|------|
| **RandAugment** | 随机组合多种图像变换（旋转、色彩调整等） |
| **Mixup** | 将两张图像线性混合：$\tilde{x} = \lambda x_i + (1-\lambda) x_j$ |
| **CutMix** | 将一张图的部分区域替换为另一张图 |
| **Random Erasing** | 随机遮挡图像的一部分区域 |
| **Repeated Augmentation** | 同一张图像在一个 batch 中出现多次（不同增强） |
| **Label Smoothing** | $y_\text{smooth} = (1 - \epsilon) \cdot y + \epsilon / K$ |
| **Stochastic Depth** | 随机跳过某些 Transformer 层 |

### 2.4 性能对比

| 模型 | 参数量 | 预训练数据 | ImageNet Top-1 |
|------|--------|-----------|----------------|
| ViT-B/16 | 86M | ImageNet-1K | 77.9% |
| ViT-B/16 | 86M | JFT-300M | 84.0% |
| **DeiT-B** | 86M | **ImageNet-1K** | **81.8%** |
| **DeiT-B ↑384** | 86M | **ImageNet-1K** | **83.1%** |
| DeiT-B (hard distill) | 86M | ImageNet-1K | **83.4%** |

DeiT 证明了：**ViT 不是必须依赖海量数据，合理的训练策略同样有效。**

---

## 三、Swin Transformer (2021, Microsoft)

### 3.1 动机

ViT 有两个关键问题限制了它作为通用视觉 backbone：

1. **全局自注意力计算量太大**：$O(N^2)$ 复杂度，高分辨率图像（如检测任务的 800×1200）token 数可达数千甚至上万
2. **单尺度特征图**：ViT 全程保持相同分辨率（如 14×14），无法像 CNN（ResNet）那样产生多尺度特征金字塔（如 1/4、1/8、1/16、1/32），而检测和分割任务强烈依赖多尺度特征

Swin Transformer 的目标：**设计一个具有层级结构和线性计算复杂度的视觉 Transformer，能作为通用 backbone。**

### 3.2 整体架构

```
输入图像 (H × W × 3)
  │
  ▼
Patch Partition: 4×4 patch → (H/4 × W/4) 个 token，dim = 48
  │
  ▼
┌─────────────────────────────────────┐
│ Stage 1: Linear Embedding → dim C   │
│ Swin Transformer Block × 2         │  输出: (H/4 × W/4 × C)
└─────────────────────────────────────┘
  │  Patch Merging (2×2 → 1, dim: 2C)
  ▼
┌─────────────────────────────────────┐
│ Stage 2:                            │
│ Swin Transformer Block × 2         │  输出: (H/8 × W/8 × 2C)
└─────────────────────────────────────┘
  │  Patch Merging (2×2 → 1, dim: 4C)
  ▼
┌─────────────────────────────────────┐
│ Stage 3:                            │
│ Swin Transformer Block × 6         │  输出: (H/16 × W/16 × 4C)
└─────────────────────────────────────┘
  │  Patch Merging (2×2 → 1, dim: 8C)
  ▼
┌─────────────────────────────────────┐
│ Stage 4:                            │
│ Swin Transformer Block × 2         │  输出: (H/32 × W/32 × 8C)
└─────────────────────────────────────┘
```

与 ResNet 类似的四阶段结构，每个阶段分辨率减半、通道翻倍，天然产生 FPN 所需的多尺度特征。

### 3.3 窗口注意力 (Window-based MSA, W-MSA)

核心思想：**不在全局做注意力，只在局部窗口内做注意力。**

将特征图划分为 $M \times M$ 的不重叠窗口（默认 $M = 7$），每个窗口内有 $M^2 = 49$ 个 token，只在窗口内部做自注意力。

**复杂度对比**（设特征图大小为 $h \times w$，通道数 $C$）：

全局 MSA：
$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$$

窗口 MSA：
$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2 hwC$$

关键差异在第二项：
- 全局 MSA 第二项是 $(hw)^2$，即 $O(N^2)$，与图像面积的平方成正比
- 窗口 MSA 第二项是 $M^2 \cdot hw$，即 $O(N)$，与图像面积**线性**关系（$M$ 是固定常数 7）

以 $h = w = 56$（Stage 1 的特征图大小）为例：$(hw)^2 = 56^4 \approx 10^7$，而 $M^2 \cdot hw = 49 \times 3136 \approx 1.5 \times 10^5$，**计算量降低近两个数量级**。

### 3.4 移位窗口 (Shifted Window, SW-MSA)

窗口注意力的问题：**窗口之间完全隔离，没有信息交互。**

Swin 的解决方案：在连续两层中交替使用**常规窗口分割**和**移位窗口分割**：

```
Layer l (W-MSA):                Layer l+1 (SW-MSA):
┌───┬───┬───┬───┐              ┌──┬────┬────┬──┐
│   │   │   │   │              │  │    │    │  │
│ 1 │ 2 │ 3 │ 4 │              ├──┼────┼────┼──┤
│   │   │   │   │      移位     │  │    │    │  │
├───┼───┼───┼───┤   ────→     │  │ A  │ B  │  │
│   │   │   │   │   (M/2,M/2)  │  │    │    │  │
│ 5 │ 6 │ 7 │ 8 │              ├──┼────┼────┼──┤
│   │   │   │   │              │  │    │    │  │
└───┴───┴───┴───┘              └──┴────┴────┴──┘

窗口边界位于不同位置，原本跨窗口的 token 现在可以交互
```

**Cyclic Shift 高效实现**：

移位后窗口边界处会产生不完整的小窗口。朴素做法需要 padding + 更多窗口计算。Swin 的巧妙做法是**循环移位 (cyclic shift)**：

1. 将特征图向左上方循环移位 $(M/2, M/2)$ 个 pixel
2. 按照常规方式划分窗口（窗口数量不变）
3. 对不应交互的区域用 **attention mask** 遮蔽
4. 计算完成后再循环移回去

这样窗口数量不变，计算量完全相同，只是多了一个 mask。

### 3.5 Patch Merging

Patch Merging 是 Swin 实现多尺度特征的关键，类似 CNN 中的下采样：

```
输入: (H × W × C)

在空间维度上，每 2×2 个相邻 patch 拼接：
  ┌───┬───┐
  │ a │ b │  → 拼接 [a; b; c; d]，得到 4C 维向量
  │ c │ d │
  └───┴───┘

拼接后: (H/2 × W/2 × 4C)
  │
  ▼
Linear(4C → 2C)  ← 降维
  │
  ▼
输出: (H/2 × W/2 × 2C)
```

效果：分辨率减半，通道数翻倍，与 CNN 的 stride-2 卷积下采样类比。

### 3.6 相对位置偏置 (Relative Position Bias)

Swin 不使用绝对位置编码，而是在注意力计算中加入**相对位置偏置**：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d}} + \mathbf{B}\right)\mathbf{V}$$

其中 $\mathbf{B} \in \mathbb{R}^{M^2 \times M^2}$ 是相对位置偏置矩阵。

对于 $M \times M$ 窗口，相对坐标范围为 $[-(M-1), M-1]$，因此偏置参数表的大小为 $(2M-1) \times (2M-1)$，当 $M=7$ 时只有 $13 \times 13 = 169$ 个参数。

**相对位置偏置 vs 绝对位置编码的优势**：

- 可以自然泛化到不同分辨率（绝对位置编码需要插值）
- 在目标检测和分割任务中效果更好
- 消融实验：+1.2% Top-1 相比无位置编码，+0.5% 相比绝对位置编码

### 3.7 模型配置

| 模型 | C | 各 Stage 层数 | 参数量 | FLOPs | ImageNet Top-1 |
|------|---|--------------|--------|-------|----------------|
| Swin-T | 96 | [2, 2, 6, 2] | 29M | 4.5G | 81.3% |
| Swin-S | 96 | [2, 2, 18, 2] | 50M | 8.7G | 83.0% |
| Swin-B | 128 | [2, 2, 18, 2] | 88M | 15.4G | 83.5% |
| Swin-L | 192 | [2, 2, 18, 2] | 197M | 34.5G | 86.3% (ImageNet-22K pretrain) |

### 3.8 作为通用 Backbone

Swin 的层级结构使其可以无缝替代 CNN backbone，接入各种下游框架：

| 任务 | 框架 | Swin-L 性能 |
|------|------|------------|
| 目标检测 | Cascade Mask R-CNN | 58.7 box AP (COCO) |
| 实例分割 | Cascade Mask R-CNN | 51.1 mask AP (COCO) |
| 语义分割 | UPerNet | 53.5 mIoU (ADE20K) |

---

## 四、后续发展

### 4.1 Swin V2 (2022)

Swin V2 解决了把 Swin 扩展到更大模型（3B 参数）和更大分辨率（1536×1536）时遇到的训练不稳定问题：

| 改进 | 问题 | 解决方案 |
|------|------|---------|
| **Residual-post-norm** | Pre-norm 下深层激活值爆炸 | 改为 post-norm + 残差后归一化 |
| **Cosine Attention** | 大模型中注意力 logit 数值过大 | $\text{sim}(\mathbf{q}, \mathbf{k}) = \cos(\mathbf{q}, \mathbf{k}) / \tau$，用余弦相似度替代点积 |
| **Log-spaced CPB** | 相对位置偏置不能跨分辨率迁移 | 连续相对位置偏置 (Continuous Position Bias)，用小 MLP 预测偏置值 |
| **渐进式训练** | 高分辨率直接训练不稳定 | 从 192→256→384→512 逐步增大分辨率 |

### 4.2 ViT 变体总结对比

| 模型 | 年份 | 注意力机制 | 多尺度 | 位置编码 | 核心特点 |
|------|------|-----------|--------|---------|---------|
| **ViT** | 2020 | 全局 $O(N^2)$ | 无 | 1D Learnable | 开山之作，需大数据 |
| **DeiT** | 2021 | 全局 $O(N^2)$ | 无 | 1D Learnable | 知识蒸馏 + 数据增强 |
| **Swin** | 2021 | 窗口 $O(N)$ | 有 | 相对位置偏置 | 层级结构 + 移位窗口 |
| **PVT** | 2021 | SRA（空间降采样） $O(N \cdot N/R^2)$ | 有 | 绝对位置编码 | Spatial Reduction Attention |
| **Twins** | 2021 | 局部 + 全局交替 | 有 | 条件位置编码 | 局部-全局注意力交替 |
| **CSWin** | 2022 | 十字形窗口 | 有 | 条件位置编码 | 水平/垂直条形窗口 |
| **MaxViT** | 2022 | 块 + 网格注意力 | 有 | 相对位置编码 | 同时兼顾局部和全局 |

### 4.3 ViT 在下游任务的应用

| 应用方向 | 代表方法 | 简述 |
|----------|---------|------|
| **目标检测** | ViTDet (Li et al., 2022) | 直接用 ViT（无层级结构）做检测，靠简单 FPN 弥补多尺度问题 |
| **语义分割** | SegViT, Segmenter | 用 ViT 作为编码器，搭配 mask transformer 解码器 |
| **视频理解** | ViViT, TimeSformer | 将时间维度引入 ViT，时空注意力 |
| **自监督预训练** | MAE, DINO, BEiT | ViT 成为自监督预训练的标准 backbone |
| **多模态** | CLIP, LiT | ViT 作为视觉编码器，与文本 Transformer 联合训练 |

---

## 五、面试高频问题

### Q1：ViT 为什么需要大数据？

**核心原因：ViT 缺少 CNN 的归纳偏置。**

CNN 通过卷积核天然具有局部性（只看邻域）和平移等变性（权重共享），这些先验知识让 CNN 在小数据集上也能快速学到有用模式。

ViT 的自注意力是**全连接**的——每个 patch 可以关注任意位置，这意味着：
- 模型不知道"相邻 patch 更可能相关"（缺少局部性先验）
- 模型不知道"同样的模式在不同位置应被同样处理"（缺少平移等变性先验）

这些模式必须从数据中学习，因此需要更多数据。当数据量足够大（>10M），这些先验可以被学到，且 Transformer 不受先验限制的灵活性反而成为优势。

### Q2：Swin 和 ViT 的本质区别？

| 对比维度 | ViT | Swin |
|----------|-----|------|
| **注意力范围** | 全局（所有 patch 互相关注） | 局部窗口内 + 移位窗口跨窗口 |
| **计算复杂度** | $O(N^2)$，与图像面积平方成正比 | $O(N)$，与图像面积线性 |
| **特征图分辨率** | 单尺度（如始终 14×14） | 多尺度（1/4, 1/8, 1/16, 1/32） |
| **位置编码** | 绝对位置编码 | 相对位置偏置 |
| **设计哲学** | 尽量少改 Transformer | 借鉴 CNN 分层设计的优点 |
| **适用范围** | 分类为主（需额外适配做检测/分割） | 通用 backbone（分类/检测/分割通吃） |

本质上，Swin 是**CNN 设计原则与 Transformer 注意力机制的融合**——用 Transformer 替代了 CNN 中的卷积操作，但保留了 CNN 的分层架构、局部计算和逐步下采样的设计范式。

### Q3：为什么 Swin 能替代 CNN Backbone？

1. **多尺度特征金字塔**：Patch Merging 产生的四级特征图与 ResNet 的 C2-C5 完全对应，可以直接接入 FPN、UPerNet 等模块
2. **线性计算复杂度**：窗口注意力使计算量与图像面积线性相关，能处理高分辨率输入（如检测任务的 800×1200）
3. **灵活的感受野**：虽然单层只看 7×7 窗口，但通过移位窗口和层级堆叠，信息可以跨窗口、跨尺度传播，有效感受野远大于实际窗口
4. **即插即用**：输出格式与 CNN backbone 一致，现有检测/分割框架（Faster R-CNN、Mask R-CNN、DeepLab 等）几乎不需要修改

### Q4：ViT 的位置编码为什么用 1D 而不是 2D？

ViT 论文实验表明 1D learnable 和 2D learnable 效果几乎一样。原因是：

- Transformer 的自注意力足够强大，即使给 1D 编码，也能从数据中学到 2D 空间关系
- 作者可视化了学到的 1D 位置编码的相似度矩阵，发现它自动呈现出 2D 网格结构
- 1D 编码更简单，不需要人为引入关于空间结构的假设

### Q5：DeiT 为什么用 CNN 做教师而不用 Transformer？

DeiT 的实验发现，用 CNN（RegNet）做教师比用 Transformer 做教师效果更好。原因是：

- CNN 和 ViT 的归纳偏置不同，CNN 教师提供了 ViT 缺少的局部性先验
- 蒸馏本质上是让学生学习教师的"暗知识"（soft label），CNN 教师的输出分布包含了局部特征的信息
- 这也解释了为什么 [CLS] token 和 [DIST] token 学到的表示不同——它们分别从标签和 CNN 教师学到了不同的特征模式

---

[返回表示学习](README.md) | [返回视觉](../README.md)

# 对比学习与 CLIP 详解

从自监督对比学习到视觉-语言对齐。

## 一、对比学习 (Contrastive Learning) 基础

### 核心思想

不需要人工标注，通过**拉近相似样本、推远不相似样本**来学习表示。

```
               特征空间
      ●  正样本对 → 拉近
     ╱
    ● anchor
     ╲
      ○  负样本 → 推远
      ○
      ○
```

### InfoNCE 损失（对比学习的核心损失函数）

$$L = -\log \frac{\exp(\text{sim}(z_i, z_j^+) / \tau)}{\sum_{k=1}^{N} \exp(\text{sim}(z_i, z_k) / \tau)}$$

- $z_i$：anchor 的特征
- $z_j^+$：正样本的特征
- $z_k$：所有样本的特征（1 个正样本 + N-1 个负样本）
- $\tau$：温度系数（通常 0.07~0.5）
- $\text{sim}$：通常用余弦相似度：$\text{sim}(z_i, z_j) = \frac{z_i \cdot z_j}{\|z_i\| \|z_j\|}$，即向量点积除以模长之积，结果范围 [-1, 1]。实践中先做 L2 归一化使 $\|z\|=1$，之后点积即等于余弦相似度，一次矩阵乘法就能算出整个 batch 的 N×N 相似度矩阵

> 本质上是一个 **N 分类的交叉熵**：在 N 个样本中识别出正样本。

### 温度系数 $\tau$ 的作用

| $\tau$ 值 | 效果 |
|-----------|------|
| 很小 (如 0.01) | 分布很尖锐，只关注最难的负样本，训练不稳定 |
| 适中 (如 0.07) | 平衡难/易负样本 |
| 很大 (如 1.0) | 分布很平坦，所有负样本权重接近，学不到细粒度区分 |

### 正样本怎么构造？

这是对比学习的关键设计选择：

| 方法 | 正样本来源 |
|------|-----------|
| **图像自监督** | 同一张图的不同数据增强（裁剪、翻转、颜色变换） |
| **CLIP** | 同一个(图片, 文本)配对 |
| **语音/文本** | 同一段内容的不同表示 |

---

## 二、经典对比学习方法

### 2.1 SimCLR (2020, Google)

最简洁的对比学习框架：

```
原始图像 x
  ├── 数据增强 t₁ → x₁ → Encoder → Projection Head → z₁
  └── 数据增强 t₂ → x₂ → Encoder → Projection Head → z₂
                                                        │
                                          InfoNCE Loss (z₁, z₂ 为正样本对)
```

**关键发现：**
1. **数据增强非常重要**：随机裁剪 + 颜色抖动是最有效的组合
2. **Projection Head 很重要**：在投影头之后的特征空间做对比，但最终使用投影头之前的特征做下游任务
3. **大 batch size 很重要**：更多负样本 → 更好的对比学习（论文用了 batch=4096）

**局限**：需要非常大的 batch size（大量 GPU），否则负样本不够。

### 2.2 MoCo (Momentum Contrast, 2020, Meta/FAIR)

用**动量队列**解决 SimCLR 需要大 batch 的问题：

```
Query 分支:  x → Encoder_q (梯度更新) → q
Key 分支:    x → Encoder_k (动量更新) → k → 入队列

对比: q 与队列中所有 k 做 InfoNCE
      (队列中保存了最近 K 步的 key，K=65536)
```

**动量更新**（核心创新）：

$$\theta_k \leftarrow m \cdot \theta_k + (1 - m) \cdot \theta_q$$

- $m$ = 0.999（非常接近 1，Key 编码器变化极慢）
- 保证队列中不同 batch 产生的 key 特征是**一致的**
- 不需要大 batch，因为负样本来自队列

> MoCo 的思想后来被 DINO 等方法广泛采用。

### 2.3 SimCLR vs MoCo 对比

| 维度 | SimCLR | MoCo |
|------|--------|------|
| 负样本来源 | 同 batch 内其他样本 | 动量队列 |
| 需要大 batch | 是（4096+） | 否（256 就够） |
| 额外结构 | Projection Head | 动量编码器 + 队列 |
| GPU 需求 | 高 | 中等 |
| 效果 | 相当 | 相当 |

---

## 三、CLIP (Contrastive Language-Image Pre-training)

**论文**：Learning Transferable Visual Models From Natural Language Supervision (2021, OpenAI)

### 核心思想

用**自然语言监督**学习视觉表示——将图像和文本映射到同一个特征空间，通过对比学习对齐。

### 架构

![CLIP 架构: 图像编码器 + 文本编码器 + 对比学习 (Radford et al., 2021)](../assets/clip_architecture.png)

> 图源: *Learning Transferable Visual Models From Natural Language Supervision (CLIP)*, Figure 1. 图像和文本分别经过各自的 Encoder 投影到同一特征空间，通过对比学习对齐配对的图文特征。

### 训练目标

一个 batch 有 N 个 (图像, 文本) 对，构建 N×N 的相似度矩阵：

```
           文本₁  文本₂  文本₃  ...  文本ₙ
图像₁  [   ✓     ✗     ✗    ...   ✗  ]
图像₂  [   ✗     ✓     ✗    ...   ✗  ]
图像₃  [   ✗     ✗     ✓    ...   ✗  ]
 ...
图像ₙ  [   ✗     ✗     ✗    ...   ✓  ]

✓ = 正样本对（对角线），✗ = 负样本对
```

对称 InfoNCE 损失：

$$L = \frac{1}{2}(L_{i2t} + L_{t2i})$$

- $L_{i2t}$：图像找文本（每行做 N 分类）
- $L_{t2i}$：文本找图像（每列做 N 分类）

### 训练数据

- **4 亿**个 (图像, 文本) 对，从互联网爬取（WebImageText, WIT）
- 不需要人工标注，自然语言本身就是监督信号

### CLIP 的 Zero-shot 分类

不需要微调，直接做分类：

```
1. 构造文本 prompt: "a photo of a {类别名}"
   → "a photo of a cat"
   → "a photo of a dog"
   → ...

2. 将所有 prompt 编码为文本特征

3. 将测试图像编码为图像特征

4. 计算图像特征与所有文本特征的余弦相似度

5. 最高相似度对应的类别就是预测结果
```

### CLIP 的关键特性

| 特性 | 说明 |
|------|------|
| **Zero-shot** | 不需要微调就能做分类、检索 |
| **开放词汇** | 不限于固定类别集，任意文本都可以 |
| **分布鲁棒** | 在分布外数据上表现好（比 ImageNet 训练的模型好） |
| **多模态对齐** | 图文特征在同一空间，可以跨模态检索 |

### CLIP 的局限

| 局限 | 说明 |
|------|------|
| 空间理解弱 | 分不清 "A on top of B" 和 "B on top of A" |
| 计数能力差 | 不擅长理解数量 |
| 细粒度识别 | 难以区分相似的子类 |
| 组合理解 | "red car and blue house" vs "blue car and red house" |

> 这些局限源于对比学习只做**全局匹配**，不关注局部细节。

### CLIP 的变体与后续

| 模型 | 改进 |
|------|------|
| **OpenCLIP** | 开源复现，多种规模 |
| **SigLIP** | 用 sigmoid 损失替代 softmax，不需要全局 batch 通信，更适合分布式训练 |
| **EVA-CLIP** | 更大规模训练 + 蒸馏，刷新 CLIP 性能 |
| **MetaCLIP** | 用元数据策划训练数据，提升数据质量 |
| **BLIP-2** | CLIP 特征 + Q-Former + LLM |
| **SigLIP2** | Google 2025 年最新，结合 CLIP 对比学习与自监督目标 |

### SigLIP 详解

**论文**：Sigmoid Loss for Language Image Pre-Training (2023, Google)

CLIP 用 softmax 做对比损失，SigLIP 改用 **sigmoid**——看起来只是换个损失函数，但影响深远。

#### CLIP 的问题：softmax 需要全局通信

CLIP 的 InfoNCE 损失对 N×N 相似度矩阵做 softmax，分母需要**整个 batch 所有样本的相似度之和**。在多 GPU 分布式训练时，这意味着每个 GPU 必须收集所有其他 GPU 上的特征才能算分母——通信开销巨大。

```
CLIP (softmax):
  GPU₁ 有样本 1~1000，GPU₂ 有样本 1001~2000，...
  计算 softmax 分母时：每个 GPU 都需要所有 GPU 的特征
  → all-gather 通信，随 GPU 数量线性增长

SigLIP (sigmoid):
  每个 (图像, 文本) 对独立算 sigmoid
  → 不需要全局通信，天然适合分布式
```

#### 损失函数对比

**CLIP（softmax / InfoNCE）**：

$$L_{i2t} = -\frac{1}{N}\sum_{i=1}^{N} \log \frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{N} \exp(s_{ij}/\tau)}$$

这是一个 **N 分类问题**：在 N 个文本中找到匹配的那一个。分母耦合了所有样本。

**SigLIP（sigmoid）**：

$$L = -\frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N} \log \frac{1}{1 + \exp(z_{ij} \cdot (-t \cdot s_{ij} + b))}$$

其中 $z_{ij} = \begin{cases} 1 & \text{if } i=j \text{ (正样本对)} \\ -1 & \text{if } i \neq j \text{ (负样本对)} \end{cases}$，$t$ 和 $b$ 是可学习的温度和偏置。

这是 $N^2$ 个**独立的二分类问题**：每个 (图像, 文本) 对独立判断"是否匹配"。

```
CLIP:   "这张图和哪个文本最匹配？"   → N 选 1（softmax）
SigLIP: "这张图和这个文本匹配吗？"   → 是/否（sigmoid）× N² 次
```

#### 为什么 sigmoid 能 work？

直觉上 softmax 更好——它有明确的"在 N 个中选最好的"语义。sigmoid 把问题拆成独立的二分类，丢失了负样本之间的相对关系。

但实际上 SigLIP 效果和 CLIP **持平甚至更好**，原因：

1. **去掉全局通信后可以用更大的 batch**：通信瓶颈消除，实际训练 batch size 可以开更大，负样本更多，弥补了 sigmoid 的信息损失
2. **更稳定的梯度**：softmax 中一个特别难的负样本会主导梯度；sigmoid 中每对独立贡献，训练更平稳
3. **chunked loss 计算**：不需要 materialize 完整的 N×N 矩阵，可以分块计算，显存更友好

#### SigLIP 在 VLM 中的地位

SigLIP 的视觉编码器（SigLIP-SO400M/14）已经成为当前 VLM 最主流的视觉骨干，取代了原版 CLIP ViT：

| VLM | 视觉编码器 |
|-----|-----------|
| LLaVA-1.5 | CLIP ViT-L/14 |
| LLaVA-NeXT / LLaVA-OneVision | **SigLIP-SO400M/14** |
| InternVL 2.5 | InternViT（SigLIP 训练方式） |
| Qwen2-VL | 自训练 ViT（SigLIP 风格损失） |
| PaliGemma | **SigLIP-SO400M/14** |

> SO400M = Shape-Optimized 400M 参数，Google 通过 NAS 搜索出的 ViT 架构变体。

---

## 四、CLIP 在 VLM 中的角色

CLIP 的视觉编码器是当前几乎所有 VLM（视觉语言模型）的**视觉骨干**：

```
LLaVA / Qwen-VL / InternVL 等 VLM 架构:

图像 → CLIP ViT (视觉编码器，通常冻结或微调) → 视觉 token
                                                    ↓
                                               投影层 / Q-Former
                                                    ↓
                                    与文本 token 一起送入 LLM
```

为什么用 CLIP 而不是普通 ViT？因为 CLIP 的特征**天然与语言对齐**，更容易被 LLM 理解。

> VLM 的详细内容参见 [VLM详解](../大模型/多模态/VLM详解.md)

---

**相关文档**：
- [DINO详解](DINO详解.md)
- [VLM详解](../大模型/多模态/VLM详解.md)

[返回上级](README.md) | [返回总目录](../../README.md)

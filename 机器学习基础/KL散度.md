# KL 散度 (Kullback-Leibler Divergence)

> 更新日期: 2026-03-17

KL 散度是连接交叉熵、ELBO、知识蒸馏、RLHF 的核心纽带。本文从"为什么需要度量分布差异"出发，一步步推出 KL 散度，再展开到它在大模型中的各种应用。

## 从信息量到熵

要理解 KL 散度，先回到信息论的起点。

**信息量 (Information)**：一个事件越不可能发生，它发生时携带的信息量越大。

$$I(x) = -\log P(x)$$

- 太阳从东边升起（$P \approx 1$）：信息量 ≈ 0，"废话"
- 中彩票（$P \approx 0.000001$）：信息量很大，"大新闻"

**熵 (Entropy)**：一个分布的平均信息量，衡量"不确定性"。

$$H(P) = -\sum_x P(x) \log P(x) = \mathbb{E}_{x \sim P}[I(x)]$$

- 均匀分布 → 熵最大（最不确定）
- 确定性分布 → 熵为 0（完全确定）

## 交叉熵：用错误分布编码的代价

如果真实分布是 $P$，但我们用 $Q$ 来编码，平均每个样本需要多少信息量？

$$H(P, Q) = -\sum_x P(x) \log Q(x) = \mathbb{E}_{x \sim P}[-\log Q(x)]$$

直觉：$Q$ 和 $P$ 越接近，交叉熵越小；如果 $Q = P$，交叉熵就等于熵本身（最优编码）。

**这就是 LLM next-token prediction 的损失函数**——$P$ 是真实的下一个 token 分布（one-hot），$Q$ 是模型预测的概率分布，最小化交叉熵就是让模型预测尽量接近真实。

### CE Loss：交叉熵损失在实践中的样子

交叉熵是理论公式，CE Loss 是它在分类任务中的具体形态。

**多分类 CE Loss**：真实标签是 one-hot（只有一个类别概率为 1），交叉熵简化为：

$$L_{CE} = -\log Q(y_{true})$$

- $y_{true}$：正确类别
- $Q(y_{true})$：模型对正确类别的预测概率（经过 softmax）

直觉：模型对正确答案越有信心（概率越高），loss 越小。$Q(y_{true}) = 1$ 时 loss = 0，$Q(y_{true}) \to 0$ 时 loss → ∞。

**PyTorch 实现**：

```python
import torch.nn.functional as F

# logits: 模型原始输出 [batch, num_classes]，未经 softmax
# labels: 正确类别索引 [batch]
loss = F.cross_entropy(logits, labels)
# 内部做了: softmax → 取 log → 取正确类别 → 取负 → 求平均
```

> 注意：`F.cross_entropy` 接收的是 **logits（未经 softmax 的原始分数）**，不是概率。它内部会先做 log_softmax 再取负，数值上更稳定。

**LLM 中的 CE Loss** 就是对每个位置的 next-token prediction 求 CE，再对序列求平均：

$$L = -\frac{1}{T}\sum_{t=1}^{T} \log Q(y_t | y_{<t})$$

这和 SFT 的 loss 完全一样——逐 token 交叉熵。

### NCE Loss：当类别太多，softmax 算不动时

CE Loss 有一个致命问题：**softmax 需要对所有类别求和**。

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}$$

当词表大小 $V$ 是几万甚至几十万时（Word2Vec 时代），每算一次 loss 都要遍历整个词表，计算量爆炸。

**NCE (Noise Contrastive Estimation)** 的核心思路：**把多分类问题转化为二分类问题**——不再问"这个词是词表中哪个词？"，而是问"这个词是真实的还是噪声？"

具体做法：
1. **正样本**：真实的上下文-目标词对 $(w, c)$
2. **负样本**：从噪声分布 $P_n$（通常是词频分布）中随机抽 $k$ 个"假的"目标词
3. 训练一个二分类器来区分真假

$$L_{NCE} = -\log \sigma(s(w, c)) - \sum_{i=1}^{k} \mathbb{E}_{w_i \sim P_n}\left[\log \sigma(-s(w_i, c))\right]$$

- $s(w, c)$：模型给 $(w, c)$ 打的分（比如向量内积）
- $\sigma$：sigmoid 函数
- $k$：负样本数量

直觉：让模型学会给真实的词对打高分，给随机凑的假词对打低分。不需要遍历整个词表，只需要和 $k$ 个负样本比较。

### InfoNCE：对比学习的基石

NCE 的思想被进一步发展为 **InfoNCE**（用于 CPC、SimCLR、CLIP 等对比学习方法）：

$$L_{InfoNCE} = -\log \frac{\exp(s(x, x^+) / \tau)}{\exp(s(x, x^+) / \tau) + \sum_{i=1}^{k} \exp(s(x, x_i^-) / \tau)}$$

- $x, x^+$：正样本对（同一图片的两个增强、图文匹配对等）
- $x_i^-$：负样本（batch 内的其他样本）
- $\tau$：温度参数，控制分布的锐利程度

**和 CE Loss 的关系**：InfoNCE 本质上就是一个 **(k+1) 分类的 CE Loss**——在 1 个正样本和 k 个负样本中，"选出正确的那个"。Softmax + 交叉熵的形式完全一样，只是分母不再是遍历整个词表，而是只看 batch 内的样本。

### 三者的关系

| | CE Loss | NCE Loss | InfoNCE |
|---|---|---|---|
| **本质** | 多分类交叉熵 | 二分类（真/假） | (k+1) 分类交叉熵 |
| **分母** | 遍历所有类别 | 无需求和 | 只看 batch 内样本 |
| **计算量** | $O(V)$，类别多就崩 | $O(k)$，与负样本数成正比 | $O(k)$，同左 |
| **典型应用** | 分类、LLM 训练 | Word2Vec | CLIP、SimCLR、MoCo |

> **延伸阅读**：CLIP 中 InfoNCE 的具体使用见 [对比学习与CLIP](../视觉/对比学习与CLIP.md)

## KL 散度：交叉熵减去熵

既然用 $Q$ 编码比用 $P$ 编码多花了一些代价，那**多花了多少**？

$$D_{KL}(P \| Q) = H(P, Q) - H(P) = -\sum_x P(x) \log Q(x) - \left(-\sum_x P(x) \log P(x)\right)$$

整理一下：

$$\boxed{D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]}$$

连续形式：

$$D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

**KL 散度 = 用错误分布编码所多付出的额外代价**。$Q$ 越接近 $P$，额外代价越小。

> 因为 $H(P)$ 是常数（真实分布固定），所以 **最小化交叉熵 = 最小化 KL 散度**。这就是为什么 LLM 训练用交叉熵损失，本质就是在做 Forward KL 最小化。

## 核心性质

| 性质 | 说明 |
|------|------|
| 非负性 | $D_{KL}(P\|Q) \geq 0$，当且仅当 $P=Q$ 时取等 (Gibbs 不等式) |
| 不对称 | $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$，所以不是"距离" |
| 不满足三角不等式 | 不是度量 (metric) |

## Forward KL vs Reverse KL

这是面试高频考点，必须理清方向。

### Forward KL: $D_{KL}(P \| Q)$ — 用 Q 去拟合 P

- P 是目标分布（真实/教师），Q 是我们要学的分布（模型/学生）
- **Mean-seeking (均值追求)**：Q 会尽量覆盖 P 的所有模式
- 当 $P(x) > 0$ 时，要求 $Q(x) > 0$（否则 KL 趋向无穷），所以 Q 不敢在 P 有概率的地方给 0
- 结果：Q 倾向"更宽"，可能把多个模式模糊地混在一起

### Reverse KL: $D_{KL}(Q \| P)$ — 用 P 去评价 Q

- **Mode-seeking (模式追求)**：Q 会集中在 P 的某一个模式上
- 当 $Q(x) > 0$ 时，要求 $P(x) > 0$，所以 Q 不敢在 P 概率为 0 的地方分配概率
- 结果：Q 倾向"更窄"，精确匹配某一个峰，但可能丢失其他模式

### 直觉记忆

```
Forward KL: "我（Q）不能漏掉你（P）的任何东西" → 宽泛覆盖
Reverse KL: "我（Q）不能编造你（P）没有的东西" → 精准集中
```

### 在大模型中的应用方向

| 场景 | 使用方向 | 原因 |
|------|---------|------|
| 知识蒸馏 | Forward KL $D_{KL}(P_{teacher}\|Q_{student})$ | 学生需要覆盖教师的所有知识 |
| RLHF/PPO | Reverse KL $D_{KL}(\pi_\theta\|\pi_{ref})$ | 策略不要偏离参考模型太远 |
| DPO | 隐式 Reverse KL 约束 | 同上，通过 $\beta$ 参数控制 |
| VAE | Reverse KL $D_{KL}(q(z\|x)\|p(z))$ | 后验逼近先验 |

## 从 KL 散度到 ELBO

KL 散度不仅连接交叉熵，还能推出 VAE 的核心——ELBO（证据下界）。

**目标**：我们想让生成模型最大化数据的对数似然 $\log p(x)$，但直接算 $p(x) = \int p(x|z)p(z)dz$ 是 intractable 的。

**做法**：引入一个近似后验 $q(z|x)$ 来逼近真实后验 $p(z|x)$，用 KL 散度衡量它们的差距：

$$D_{KL}(q(z|x) \| p(z|x)) = \mathbb{E}_{q}\left[\log \frac{q(z|x)}{p(z|x)}\right] \geq 0$$

把 $p(z|x) = \frac{p(x|z)p(z)}{p(x)}$ 代入，展开后移项：

$$\log p(x) = D_{KL}(q(z|x) \| p(z|x)) + \underbrace{\mathbb{E}_{q}[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))}_{\text{ELBO}}$$

因为 $D_{KL} \geq 0$，所以：

$$\boxed{\log p(x) \geq \text{ELBO} = \underbrace{\mathbb{E}_{q}[\log p(x|z)]}_{\text{重建项}} - \underbrace{D_{KL}(q(z|x) \| p(z))}_{\text{正则项}}}$$

**ELBO 两项的直觉**：

- **重建项** $\mathbb{E}_{q}[\log p(x|z)]$：从潜变量 z 能不能还原出 x？（解码器要好）
- **正则项** $D_{KL}(q(z|x) \| p(z))$：后验别离先验太远（编码器不要太离谱）

注意正则项用的是 **Reverse KL**——这对应前面表格里 VAE 那一行，后验"不能编造先验没有的东西"。

> **延伸阅读**：完整的 VAE 推导和 Diffusion 模型的 ELBO 见 [生成模型：VAE与Diffusion](../论文阅读/生成模型/生成模型_VAE与Diffusion.md)

## 全景图：KL 散度如何串联 ML 各领域

```
信息量 → 熵 → 交叉熵 ──拆开──→ 熵 + KL散度
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
               Forward KL      Reverse KL      KL ≥ 0
                    │               │               │
             最小化交叉熵      KL 惩罚项       ELBO 推导
                    │               │               │
            ┌───────┤          ┌────┤          VAE / Diffusion
            │       │          │    │
     LLM 预训练  知识蒸馏   RLHF/PPO  DPO
   (next-token)            (约束策略偏移)
```

每条线都用到了 KL 散度的某个性质：
- **LLM 训练**：最小化交叉熵 = 最小化 Forward KL
- **知识蒸馏**：Forward KL，学生覆盖教师的全部知识
- **RLHF/DPO**：Reverse KL 惩罚，策略不偏离参考模型
- **VAE**：Reverse KL 正则 + ELBO 下界

## 与其他散度/距离的对比

| 度量 | 公式 | 特点 |
|------|------|------|
| KL 散度 | $\sum P \log(P/Q)$ | 不对称，信息论基础 |
| JS 散度 | $\frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$, $M=\frac{P+Q}{2}$ | 对称，有界 $[0, \log 2]$，GAN 原始损失 |
| Wasserstein 距离 | 最优传输距离 | WGAN 使用，即使分布不重叠也有梯度 |

---

相关手撕代码: [KL Loss 实现](../面试手撕/大模型手撕/手撕代码合集.md#9-kl-散度损失)
在大模型训练中的应用: [知识蒸馏与KL约束](../大模型/训练与微调/知识蒸馏与KL约束.md)
RLHF 中的 KL 惩罚: [RLHF与PPO详解](../大模型/训练与微调/RLHF与PPO详解.md)

---
[返回上级](README.md) | [返回总目录](../README.md)

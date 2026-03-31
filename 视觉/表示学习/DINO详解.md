# DINO 详解

Self-DIstillation with NO labels — 通过自蒸馏学习视觉特征，无需任何标注。

> **论文**：
> - DINOv1: *Emerging Properties in Self-Supervised Vision Transformers* (Caron et al., ICCV 2021)
> - DINOv2: *DINOv2: Learning Robust Visual Features without Supervision* (Oquab et al., TMLR 2024)

---

## 一、DINOv1 (2021, Meta/FAIR)

### 1.1 背景与动机

2020-2021 年自监督学习的主流范式是**对比学习**（SimCLR、MoCo），核心思路是拉近正样本对、推远负样本对。但对比学习存在几个问题：

- **需要大量负样本**：SimCLR 需要大 batch（4096+），MoCo 需要维护队列
- **负样本可能有假阴性**：两张语义相同的图被当作负样本推开
- **与知识蒸馏的关系不清晰**：自监督学习本质上在学什么？

同期 BYOL（Bootstrap Your Own Latent, 2020）已经证明**不需要负样本也能学到好特征**，但 BYOL 的成功原因一直存在争议（BatchNorm 的隐式信息泄露？还是架构本身？）。

DINO 的动机：

1. 用更简洁的**自蒸馏**框架统一理解自监督学习
2. 探索 ViT 在自监督下的特殊涌现能力（CNN 上看不到的现象）
3. 不依赖 BatchNorm 等隐式机制，用显式的 centering + sharpening 防止坍缩

### 1.2 核心思想

不用对比学习的负样本，也不用标签。用**自蒸馏**——让学生网络去模仿教师网络的输出分布。

```
同一张图像 x
  ├── 全局裁剪 (224×224) ──→ Teacher 编码 → tₛ (停止梯度)
  ├── 全局裁剪 (224×224) ──→ Teacher 编码 → tₛ
  ├── 局部裁剪 (96×96)  ──→ Student 编码 → sₛ
  ├── 局部裁剪 (96×96)  ──→ Student 编码 → sₛ
  └── ...

Loss: 让 Student 输出分布 → 匹配 Teacher 输出分布
```

### 1.3 师生架构

![DINO 自蒸馏架构 (Caron et al., 2021)](../assets/dino_architecture.png)

> 图源: *Emerging Properties in Self-Supervised Vision Transformers (DINO)*, Figure 2. Teacher 通过动量更新 (EMA)，Student 通过梯度更新。两者对同一图像的不同增强视图分别计算输出分布，通过交叉熵损失对齐。

#### 网络结构详解

Teacher 和 Student 具有**完全相同的架构**，由三部分组成：

```
输入图像 x
    │
    ▼
┌──────────────────┐
│  Backbone (ViT)   │  → 提取特征 [CLS] token
│  ViT-S/16 或 B/8  │
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Projection Head  │  → MLP: Linear(d, 2048) → GELU → Linear(2048, 2048) → GELU → Linear(2048, 256)
│  3 层 MLP + L2    │     最后一层做 L2 归一化
└──────────────────┘
    │
    ▼
┌──────────────────┐
│  Prototype Layer  │  → Linear(256, K)，K=65536（不带 bias）
│  (权重归一化)      │     权重做 L2 归一化 → 相当于余弦相似度
└──────────────────┘
    │
    ▼
  softmax(·/τ) → 概率分布 p ∈ ℝ^K
```

**关键细节**：
- Projection Head 在**评估时丢弃**，只用 backbone 的 [CLS] token 作为特征
- Prototype Layer 的 K=65536 维输出不是类别预测，而是在一组**可学习原型向量**上的分布
- 最终输出经过温度缩放的 softmax，得到概率分布

#### 动量更新（EMA）

**与 MoCo 的联系**：Teacher 的动量更新机制直接借鉴了 MoCo：

$$\theta_t \leftarrow m \cdot \theta_t + (1 - m) \cdot \theta_s, \quad m = 0.996 \to 1.0$$

- 训练初期 $m=0.996$，Teacher 跟随 Student 较快，便于早期对齐
- 训练过程中 $m$ 按余弦调度递增到 $1.0$，Teacher 越来越稳定
- 调度公式：$m = 1 - (1 - m_0) \cdot (\cos(\pi k / K) + 1) / 2$，其中 $k$ 是当前 step

> **为什么不直接复制 Student？** 如果 Teacher = Student（即 $m=0$），模型立刻坍缩。动量更新让 Teacher 成为 Student 历史版本的滑动平均，提供更稳定的目标。

### 1.4 损失函数详解

对于一张图像，生成一组视图 $V = \{x_1^g, x_2^g, x_1^l, \ldots, x_n^l\}$（2 个全局 + $n$ 个局部）。

Teacher 只处理全局视图，Student 处理所有视图。损失为：

$$\mathcal{L} = \sum_{x \in \{x_1^g, x_2^g\}} \sum_{\substack{x' \in V \\ x' \neq x}} H(P_t(x), P_s(x'))$$

其中 $H(a, b) = -\sum_k a_k \log b_k$ 是交叉熵，$P_t$ 和 $P_s$ 分别是 Teacher 和 Student 输出的概率分布：

$$P_s(x)^{(i)} = \frac{\exp(g_{\theta_s}(x)^{(i)} / \tau_s)}{\sum_k \exp(g_{\theta_s}(x)^{(k)} / \tau_s)}$$

$$P_t(x)^{(i)} = \frac{\exp((g_{\theta_t}(x)^{(i)} - c) / \tau_t)}{\sum_k \exp((g_{\theta_t}(x)^{(k)} - c) / \tau_t)}$$

注意 Teacher 的输出先**减去中心 $c$**（centering），且使用更低的温度 $\tau_t < \tau_s$（sharpening）。

**梯度只通过 Student 回传**，Teacher 的参数通过 EMA 更新。

### 1.5 Multi-crop 策略

| 裁剪类型 | 分辨率 | 数量 | 覆盖范围 | 送入 |
|---------|--------|------|---------|------|
| 全局裁剪 | 224×224 | 2 | >50% 图像面积 | Teacher + Student |
| 局部裁剪 | 96×96 | 6~8 | <50% 图像面积 | 仅 Student |

核心思想：**局部到全局的对应** —— Student 看到局部小图，要能预测出 Teacher 看到全局大图时的输出。迫使模型学习语义一致的特征。

**Multi-crop 的计算效率技巧**：
- 局部裁剪分辨率小（96 vs 224），计算量远低于全局裁剪
- Teacher 只处理 2 个全局视图（计算量固定）
- Student 虽然处理 2+8=10 个视图，但 8 个局部视图很轻量
- 总计算量约 2× 全局视图 + 8× (96/224)² ≈ 2 + 1.47 ≈ 3.5× 单视图，比 10× 单视图高效很多

**数据增强细节**：

```
全局裁剪增强:
  RandomResizedCrop(224, scale=(0.4, 1.0))
  → RandomHorizontalFlip
  → ColorJitter(0.4, 0.4, 0.2, 0.1), p=0.8
  → RandomGrayscale(p=0.2)
  → GaussianBlur(σ ∈ [0.1, 2.0]), p=1.0 (第1个) / p=0.1 (第2个)
  → Solarization(threshold=128), p=0.0 (第1个) / p=0.2 (第2个)
  → Normalize

局部裁剪增强:
  RandomResizedCrop(96, scale=(0.05, 0.4))
  → RandomHorizontalFlip
  → ColorJitter(同上)
  → GaussianBlur(p=0.5)
  → Normalize
```

### 1.6 Centering 和 Sharpening（防止坍缩）

对比学习用负样本防止所有特征坍缩到同一点。DINO 没有负样本，用两个技巧：

#### Centering

Teacher 输出减去全局均值中心（EMA 更新），防止某个维度主导：

$$g_t(x) \leftarrow g_t(x) - c$$

$$c \leftarrow m_c \cdot c + (1-m_c) \cdot \frac{1}{B} \sum_{i=1}^{B} g_t(x_i)$$

其中 $m_c = 0.9$，$B$ 是 batch size。

**直觉**：如果没有 centering，Teacher 可能输出一个固定的向量（常数坍缩），此时 Student 也只需输出这个常数就能最小化损失。减去均值后，Teacher 的输出被迫"围绕零点波动"，不同输入必须产生不同的偏移方向。

#### Sharpening

Teacher 用较低温度 $\tau_t$（如 0.04），Student 用较高温度 $\tau_s$（如 0.1）：

- 低温度 → 分布更尖锐 → 接近 one-hot → 更确定的"伪标签"
- 高温度 → 分布更平滑 → Student 的预测带有更多不确定性

**温度调度**：$\tau_t$ 在训练前 30 个 epoch 从 0.04 线性 warmup 到 0.04（即 $\tau_t$ 固定），$\tau_s = 0.1$ 固定不变。部分实现中 $\tau_t$ 会从 0.04 warmup 到 0.07。

### 1.7 完整算法伪代码

```python
# DINO 训练伪代码
# gs, gt: student 和 teacher 网络
# C: center (初始化为 0)
# tps, tpt: student/teacher 温度
# m: EMA 动量系数

for x in dataloader:
    # 1. 生成多视图
    x1g, x2g = global_crop(x), global_crop(x)    # 2 个全局裁剪
    x1l, ..., xnl = [local_crop(x) for _ in range(n_local)]  # n 个局部裁剪

    # 2. Teacher 前向 (仅全局视图, 停止梯度)
    with torch.no_grad():
        t1 = softmax((gt(x1g) - C) / tpt)
        t2 = softmax((gt(x2g) - C) / tpt)

    # 3. Student 前向 (所有视图)
    s1 = softmax(gs(x1g) / tps)
    s2 = softmax(gs(x2g) / tps)
    s_locals = [softmax(gs(xl) / tps) for xl in [x1l, ..., xnl]]

    # 4. 计算交叉熵损失 (避免同一视图自匹配)
    loss = 0
    loss += H(t1, s2) + H(t2, s1)                    # 全局-全局
    loss += sum(H(t1, sl) for sl in s_locals)         # 全局1-局部
    loss += sum(H(t2, sl) for sl in s_locals)         # 全局2-局部
    loss /= (2 * n_local + 2)                         # 归一化

    # 5. Student 梯度更新
    loss.backward()
    # 梯度裁剪: clip_grad_norm_(gs.parameters(), 3.0)  # ViT 需要
    optimizer.step()

    # 6. Teacher EMA 更新
    with torch.no_grad():
        for pt, ps in zip(gt.parameters(), gs.parameters()):
            pt.data = m * pt.data + (1 - m) * ps.data

    # 7. 更新 center
    with torch.no_grad():
        C = 0.9 * C + 0.1 * torch.cat([gt(x1g), gt(x2g)]).mean(0)
```

### 1.8 关键训练超参数

| 超参数 | DINOv1 值 | 说明 |
|--------|----------|------|
| Backbone | ViT-S/16, ViT-S/8, ViT-B/16, ViT-B/8 | /16 表示 patch size=16 |
| Optimizer | AdamW | 与 ViT 标配一致 |
| 学习率 | 0.0005 × batchsize/256 | 线性缩放规则 |
| LR 调度 | 10 epoch warmup + cosine decay | - |
| 权重衰减 | 0.04 → 0.4（cosine 调度） | 从小到大 |
| Batch size | 1024 | 16 × V100 GPU |
| 训练 epoch | 300（ViT-S）/ 400（ViT-B） | ImageNet |
| EMA 动量 $m$ | 0.996 → 1.0（cosine） | - |
| Teacher 温度 $\tau_t$ | 0.04（warmup 30 epoch 从 0.04 开始） | - |
| Student 温度 $\tau_s$ | 0.1 | 固定 |
| 输出维度 K | 65536 | Prototype 数量 |
| Projection Head | 3 层 MLP，隐藏维度 2048，bottleneck 256 | - |
| 梯度裁剪 | max_norm = 3.0 | ViT 训练稳定性需要 |

### 1.9 DINO 的涌现特性

DINOv1 最令人惊讶的发现：**ViT 的注意力图自动学会了语义分割**。

```
原始图像: [一只狗在草地上]

ViT [CLS] token 对其他 token 的注意力:
→ 自动聚焦在狗的轮廓上
→ 没有任何分割标注！纯自监督学到的
```

这说明 DINO 学到的特征具有很强的**局部语义感知**能力。

#### 不同注意力头的专业化

DINO-ViT 的不同注意力头会**自动分工**：

```
Head 1: 关注前景物体的整体轮廓
Head 2: 关注物体的内部纹理
Head 3: 关注背景区域
Head 4: 关注边缘和边界
...
```

这种多头注意力的自发分工在监督学习训练的 ViT 中**不明显**，是自蒸馏训练的特有现象。

#### k-NN 分类性能

DINO 的另一个惊喜：冻结 backbone，用简单的 **k-NN（k=20）** 就能在 ImageNet 上达到很好的分类准确率，不需要训练任何线性层：

| 模型 | k-NN Top-1 | 线性探头 Top-1 |
|------|-----------|--------------|
| DINO ViT-S/16 | 74.5% | 77.0% |
| DINO ViT-S/8 | 76.1% | 78.3% |
| DINO ViT-B/16 | 76.1% | 78.2% |
| DINO ViT-B/8 | 77.4% | 80.1% |

k-NN 性能接近线性探头，说明 DINO 特征空间**天然具有良好的聚类结构**。

#### ViT vs CNN 的差异

DINO 在 ViT 和 ResNet 上都能训练，但涌现特性主要出现在 ViT 上：

| 特性 | DINO + ViT | DINO + ResNet50 |
|------|-----------|----------------|
| 注意力图语义分割 | ✅ 非常清晰 | ❌ 无注意力图 |
| k-NN 分类 | 非常好 | 好但不如 ViT |
| 线性探头分类 | 好 | 好（ResNet 在线性探头上差距小） |
| 局部特征质量 | 极好 | 一般 |

### 1.10 消融实验关键结论

论文中的消融研究揭示了每个组件的重要性（ViT-S/16, ImageNet, k-NN 评估）：

| 消融项 | k-NN Acc. | 说明 |
|--------|----------|------|
| 完整 DINO | 74.5% | 基线 |
| 去掉 centering | 坍缩 | 输出变成常数 |
| 去掉 sharpening（$\tau_t = \tau_s = 0.1$） | 坍缩 | 分布太平滑 |
| 去掉 multi-crop（只用 2 个全局视图） | 72.8% | 下降 ~2% |
| 去掉 EMA（$m=0$，Teacher=Student） | 坍缩 | 即刻坍缩 |
| 去掉 projection head | ~70% | 下降明显 |
| 用 BatchNorm 替代 centering | 74.2% | 可以，但不如 centering 清晰 |

---

## 二、DINOv2 (2023, Meta/FAIR)

### 2.1 相比 v1 的改进

DINOv2 不是方法的革新，而是**工程和数据的全面升级**。

| 维度 | DINOv1 | DINOv2 |
|------|--------|--------|
| 数据 | ImageNet (1.2M) | LVD-142M（自动策划的 142M 图像） |
| 模型 | ViT-S/B | ViT-S/B/L/g（最大 1.1B 参数） |
| 训练目标 | 纯自蒸馏 | 自蒸馏 + iBOT (mask image modeling) |
| 蒸馏 | 无 | 大模型蒸馏到小模型 |
| 正则化 | 基础 | KoLeo 正则化 + 改进的 centering |
| 效果 | 好 | 接近/超越 OpenCLIP（不需要文本！） |

### 2.2 训练目标：DINO Loss + iBOT Loss + KoLeo

DINOv2 的总损失是三部分的加权和：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DINO}} + \lambda_1 \mathcal{L}_{\text{iBOT}} + \lambda_2 \mathcal{L}_{\text{KoLeo}}$$

#### (1) DINO Loss（图像级）

与 DINOv1 基本相同——[CLS] token 的 Teacher-Student 交叉熵匹配。

#### (2) iBOT Loss（Patch 级 Masked Prediction）

```
DINOv2 = DINO 自蒸馏 (图像级) + iBOT (patch级 masked prediction)

图像级: [CLS] token 的 Teacher-Student 匹配（同 DINOv1）
Patch级: 随机 mask 一些 patch，Student 预测被 mask patch 的 Teacher 特征
```

详细流程：

```
Student 侧:
  输入: [CLS] [p1] [MASK] [p3] [MASK] [p5] ...    ← 随机 mask 掉部分 patch
  输出: [CLS] [h1] [h2*] [h3] [h4*] [h5] ...      ← 对 MASK 位置也产生输出

Teacher 侧:
  输入: [CLS] [p1] [p2] [p3] [p4] [p5] ...         ← 完整输入（无 mask）
  输出: [CLS] [t1] [t2] [t3] [t4] [t5] ...

iBOT Loss:
  仅在被 mask 的位置计算:
  L_iBOT = H(softmax(t2/τ_t), softmax(h2*/τ_s)) + H(softmax(t4/τ_t), softmax(h4*/τ_s)) + ...
```

Mask 策略使用 **BEiT 式的块状 mask**（block masking），mask 比例约 50%。

两个目标互补：
- 图像级目标（DINO Loss）→ 全局语义理解
- Patch 级目标（iBOT Loss）→ 局部细节感知、空间关系建模

#### (3) KoLeo 正则化（特征均匀性）

KoLeo（Kozachenko-Leonenko）正则化鼓励特征在超球面上均匀分布：

$$\mathcal{L}_{\text{KoLeo}} = -\frac{1}{n} \sum_{i=1}^{n} \log(d_{nn}(z_i))$$

其中 $d_{nn}(z_i)$ 是样本 $z_i$ 到其**最近邻**的距离。最小化此损失 = 最大化最近邻距离 = 让特征尽量分散。

**作用**：进一步防止特征坍缩，鼓励特征空间的充分利用。

### 2.3 改进的 Centering：Sinkhorn-Knopp

DINOv2 不再使用 DINOv1 的简单均值 centering，而是采用 **Sinkhorn-Knopp（SK）批归一化**：

```
DINOv1 centering:  输出减去 EMA 均值（简单但粗糙）

DINOv2 Sinkhorn-Knopp:
  1. 将 Teacher 输出组成矩阵 Q ∈ ℝ^{B×K}
  2. 用 SK 算法迭代归一化行和列，使 Q 成为"软"双随机矩阵
  3. 即每个样本的输出在 prototype 上的分布更均匀，
     每个 prototype 被所有样本"均匀使用"
```

SK 的优势：
- 不仅防止输出坍缩到常数（centering 能做到）
- 还防止**模式坍缩**——即只使用少数几个 prototype 维度
- 确保所有 prototype 维度都被充分利用

### 2.4 数据策划 (LVD-142M)

DINOv2 的数据策划管道是论文的重要贡献之一：

```
第一步: 数据源收集
  ├── 公开数据集（ImageNet-22k, Google Landmarks 等）
  ├── 网络爬取的未标注图像池（~1.2B 候选）
  └── 无版权限制的图像

第二步: 去重
  ├── 基于 copy detection 模型的近似去重
  └── 去掉与下游评估集（ImageNet val 等）重复的图像

第三步: 检索式策划 (Retrieval-based curation)
  ├── 用预训练模型（自监督 ViT）提取所有图像的 embedding
  ├── 以 ImageNet-22k 为"参考分布"
  ├── 对参考集中的每张图，检索 embedding 最近邻的未标注图像
  └── 得到与 ImageNet 分布相似但规模大 100 倍的数据集

第四步: 自蒸馏式精炼
  ├── 在初版数据上训练一个小模型
  ├── 用该模型重新做 embedding → 重新检索
  └── 迭代 2~3 轮，数据质量逐步提升

最终: LVD-142M（142 million 图像）
```

**关键设计选择**：
- 不需要任何人工标注
- 以 ImageNet 的概念分布为参考（确保覆盖常见语义）
- 去重确保数据多样性
- 迭代精炼提升质量

### 2.5 模型规模与训练设施

| 模型变体 | 参数量 | Patch Size | Embedding Dim | Heads | Layers |
|---------|--------|-----------|--------------|-------|--------|
| ViT-S/14 | 21M | 14 | 384 | 6 | 12 |
| ViT-B/14 | 86M | 14 | 768 | 12 | 12 |
| ViT-L/14 | 300M | 14 | 1024 | 16 | 24 |
| ViT-g/14 | 1.1B | 14 | 1536 | 24 | 40 |

**注意 Patch Size = 14**（而非常见的 16），这是 DINOv2 的选择——14 带来更多 token（$224/14 = 16 \times 16 = 256$ tokens vs $224/16 = 14 \times 14 = 196$ tokens），特征图分辨率更高。

**训练规模**：
- ViT-g/14：在 **16 个 A100 节点**（128 GPUs）上训练
- 训练约 **625k 迭代**，batch size = 3072
- 总训练时间约 **12 天**
- 使用 **FSDP**（Fully Sharded Data Parallelism）进行分布式训练

### 2.6 模型蒸馏

DINOv2 训练好 ViT-g/14（1.1B 参数）后，将其蒸馏到更小的模型：

```
ViT-g/14 (Teacher, 1.1B, 冻结)
    │
    ▼  蒸馏
ViT-S/14, ViT-B/14, ViT-L/14 (Student)
```

蒸馏方式非常简单：
- Teacher 冻结，Student 随机初始化
- Student 直接模仿 Teacher 的 [CLS] token 和 patch token 输出
- 不需要原始训练数据的 multi-crop 等复杂流程

这样得到的小模型（如 DINOv2-ViT-S/14, 21M 参数）性能远超从头训练的同规模模型。

### 2.7 DINOv2 的特征质量

DINOv2 的特征被广泛用于下游任务（通常冻结 backbone，只训线性探头或轻量 head）：

| 下游任务 | 表现 | 对比 |
|---------|------|------|
| ImageNet 线性探头 | **81.1%**（ViT-g） | 接近监督 ViT（84.4%） |
| ImageNet k-NN | **79.0%**（ViT-g） | 远超其他自监督方法 |
| ADE20k 语义分割 | **49.0 mIoU**（线性探头） | 超越 OpenCLIP |
| NYUd 深度估计 | 非常强 | Depth Anything 就用 DINOv2 |
| Oxford/Paris 检索 | 刷新 SOTA | 特征匹配质量极高 |
| 特征匹配 | 跨图像的 patch 特征高度语义一致 | - |

**跨数据集迁移**：DINOv2 的最大优势是**通用性**——同一个冻结 backbone 在十几个不同任务上都表现优异，不需要针对每个任务微调。

### 2.8 DINOv2 关键消融

| 消融项 | ImageNet 线性 | ADE20k 分割 |
|--------|-------------|------------|
| 完整 DINOv2（DINO + iBOT + KoLeo） | 81.1 | 49.0 |
| 去掉 iBOT（只有 DINO） | 80.2 | 45.3 |
| 去掉 KoLeo | 80.6 | 47.8 |
| 使用 ImageNet-1k 数据（1.2M） | 78.1 | 44.2 |
| 使用 ImageNet-22k 数据（14M） | 79.8 | 47.1 |
| 使用 LVD-142M | 81.1 | 49.0 |

**结论**：iBOT 对密集预测任务（分割）帮助巨大；数据规模和质量是关键提升因素。

---

## 三、DINO vs CLIP

| 维度 | DINO | CLIP |
|------|------|------|
| **监督信号** | 纯视觉自监督（无文本） | 图文对比（需要文本） |
| **训练数据** | 纯图像 | (图像, 文本) 对 |
| **全局特征** | 好 | 非常好 |
| **局部特征** | 非常好（注意力图有语义） | 较弱 |
| **语言对齐** | 无（特征空间与语言无关） | 有（天然对齐） |
| **Zero-shot 分类** | 不能直接做 | 强项 |
| **密集预测（分割/深度）** | 非常强 | 较弱 |
| **VLM 中的角色** | 提供局部视觉特征 | 提供语言对齐的视觉特征 |
| **数据获取难度** | 低（只需图像） | 高（需要图文配对） |
| **特征可解释性** | 注意力图直观 | 需要文本做探针 |

> **互补关系**：CLIP 擅长全局语义和语言对齐，DINO 擅长局部细节和密集预测。一些 VLM（如 InternVL）会同时利用两者。

**融合方案示例**：

```
InternVL 的做法:
  1. InternViT = CLIP 目标 + DINO 目标联合训练
  2. 既有语言对齐能力，又保留了 DINO 的局部特征质量

Grounding DINO 的做法:
  1. 用 DINO 特征做目标检测的 backbone
  2. 用 BERT 编码文本查询
  3. 多模态融合 → 开放词汇目标检测
```

---

## 四、自蒸馏为什么不会坍缩？

这是 DINO 系列最深层的问题——没有负样本，为什么不会所有特征都一样？

### 坍缩的两种形式

1. **常数坍缩（Complete Collapse）**：所有输入都映射到同一个向量
2. **模式坍缩（Mode Collapse）**：输出只利用特征空间的少数几个方向/维度

### 三重防线

答案是三个机制的组合：

| 机制 | 防止的坍缩类型 | 原理 |
|------|-------------|------|
| **动量更新** | 常数坍缩 | Teacher 变化慢，提供稳定的"伪标签"，避免两个网络相互强化同一错误方向 |
| **Centering** | 常数坍缩 | 减去均值，迫使输出围绕零点波动，不同输入必须产生不同方向的偏移 |
| **Sharpening** | 模式坍缩 | Teacher 用低温度，输出接近 one-hot，Student 被迫学习有区分度的特征 |

如果去掉任何一个，训练都会坍缩。

### 更深层的理论分析

DINO 的自蒸馏可以从**信息论**角度理解：

- **Centering** 约束了 Teacher 输出的**边际分布**（marginal distribution）趋向均匀 → 最大化熵 $H(P_t)$
- **Sharpening** 约束了 Teacher 对每个输入的**条件分布**趋向 one-hot → 最小化条件熵 $H(P_t | X)$
- 两者结合 = 最大化**互信息** $I(P_t; X) = H(P_t) - H(P_t | X)$

这与对比学习的 InfoNCE 目标本质相同——都在最大化表示与输入之间的互信息。DINO 只是用不同的机制实现了相同的优化目标。

---

## 五、DINO 在下游应用中的影响

### 5.1 Grounding DINO（2023）

开放集目标检测：文本描述 → 检测对应物体。

```
架构:
  Image Backbone: Swin Transformer (也可替换为 DINO 预训练的 ViT)
  Text Encoder: BERT
  Feature Enhancer: 双向跨模态注意力
  Language-Guided Query Selection: 文本指导的查询生成
  Cross-Modality Decoder: 融合视觉和文本特征

注意: Grounding DINO 的 "DINO" 指的是 DETR-like DINO 检测器，
      与自监督 DINO 同名但不同。不过两者可以结合使用。
```

### 5.2 Depth Anything（2024）

单目深度估计的 SOTA 模型，直接使用 DINOv2 作为 backbone：

```
Depth Anything = DINOv2 (冻结 backbone) + DPT Head (轻量解码器)

为什么选 DINOv2 而非 CLIP？
  → 深度估计需要精确的空间/局部特征
  → DINOv2 的 patch 特征在空间上语义一致
  → CLIP 的特征偏向全局语义，空间细节较弱
```

### 5.3 Segment Anything (SAM)

虽然 SAM 没有直接使用 DINO，但 SAM 的成功验证了与 DINO 相同的理念：**大规模数据 + 自监督/半监督 → 强大的视觉基础模型**。后续 SAM 2 等工作中，DINOv2 特征常被用于辅助。

### 5.4 多模态大模型中的 DINO

多个 VLM 架构利用 DINO 特征：

| 模型 | DINO 的角色 |
|------|----------|
| InternVL | 联合训练 CLIP + DINO 目标 |
| LLaVA-OneVision | 探索 DINOv2 作为视觉编码器 |
| Cambrian-1 | 融合 CLIP + DINOv2 + SigLIP 多种视觉特征 |
| DeepSeek-VL | 使用 SigLIP + DINOv2 双编码器 |

---

## 六、实践指南：如何使用 DINOv2

### 6.1 加载预训练模型

```python
import torch

# 方式1: 通过 torch.hub
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vitb14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
dinov2_vitg14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14')

# 方式2: 通过 transformers 库
from transformers import AutoModel
model = AutoModel.from_pretrained('facebook/dinov2-base')
```

### 6.2 提取特征

```python
from torchvision import transforms

# DINOv2 标准预处理
transform = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 提取特征
model.eval()
with torch.no_grad():
    img_tensor = transform(img).unsqueeze(0)  # [1, 3, 224, 224]

    # 全局特征 (CLS token)
    cls_token = model(img_tensor)  # [1, 768] for ViT-B

    # Patch 级特征 (用于密集预测)
    features = model.forward_features(img_tensor)
    patch_tokens = features['x_norm_patchtokens']  # [1, 256, 768]
    # 256 = (224/14)^2 = 16×16 个 patch

    # reshape 为空间特征图
    B, N, D = patch_tokens.shape
    h = w = int(N ** 0.5)  # 16
    feature_map = patch_tokens.reshape(B, h, w, D).permute(0, 3, 1, 2)  # [1, 768, 16, 16]
```

### 6.3 常见下游任务用法

```
图像分类:
  DINOv2 backbone (冻结) → [CLS] token → Linear(768, num_classes)

语义分割:
  DINOv2 backbone (冻结) → patch tokens [B, 256, 768]
    → reshape [B, 768, 16, 16] → Linear Head 或 DPT Head → 上采样到原图大小

深度估计:
  DINOv2 backbone (冻结) → 多层 patch tokens → DPT Decoder → 深度图

图像检索:
  DINOv2 backbone (冻结) → [CLS] token → L2 归一化 → 余弦相似度检索

语义对应 (Semantic Correspondence):
  图像 A 的 patch tokens ↔ 图像 B 的 patch tokens → 余弦相似度匹配
```

### 6.4 与 DINOv2 Registers

DINOv2 发布后，研究者发现 ViT 的注意力图中存在**伪影**（artifacts）——某些背景 patch 的注意力异常高。

论文 *Vision Transformers Need Registers* (2024, Meta) 提出在输入序列中添加额外的**可学习 register token**：

```
标准输入: [CLS] [p1] [p2] ... [pN]
加 register: [CLS] [REG1] [REG2] [REG3] [REG4] [p1] [p2] ... [pN]

Register token 充当"垃圾桶"，吸收全局信息的聚合需求，
让 patch token 的注意力更专注于局部语义 → 注意力图更干净。
```

带 register 的 DINOv2 模型：`dinov2_vits14_reg`, `dinov2_vitb14_reg`, `dinov2_vitl14_reg`, `dinov2_vitg14_reg`

---

## 七、DINO 系列发展时间线

```
2020.06  BYOL         ── 证明不需要负样本也能自监督学习
2020.11  SwAV         ── 聚类 + 多视图对比
2021.04  DINO (v1)    ── 自蒸馏 + ViT 注意力涌现
2021.10  iBOT         ── 在线分词器 + mask image modeling
2021.11  MAE          ── 另一条路线：masked autoencoder
2023.04  DINOv2       ── 大规模数据 + DINO + iBOT + 蒸馏
2023.10  DINOv2 + Registers ── 解决注意力伪影
2024.01  Depth Anything    ── DINOv2 backbone → 深度估计 SOTA
2024.06  Depth Anything v2 ── 延续 DINOv2 backbone
```

---

**相关文档**：
- [对比学习与CLIP详解](对比学习与CLIP详解.md)
- [VLM详解](../大模型/多模态/VLM详解.md)

[返回上级](README.md) | [返回总目录](../../README.md)

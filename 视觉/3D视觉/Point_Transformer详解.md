# Point Transformer 详解

Point Transformer 系列 -- 将 Transformer 的注意力机制引入点云处理，实现点云理解的 SOTA。

> **核心论文**
> - **Point Transformer V1** (ICCV 2021): *Point Transformer*, Zhao et al.
> - **Point Transformer V2** (NeurIPS 2022): *Point Transformer V2: Grouped Vector Attention and Partition-based Pooling*, Wu et al.
> - **Point Transformer V3** (CVPR 2024): *Point Transformer V3: Simpler, Faster, Stronger*, Wu et al.

---

## 一、背景：为什么在点云上用 Transformer？

### 1. 点云处理的核心挑战

| 挑战 | 说明 |
|:-----|:-----|
| **无序性** | 同一组点的任意排列都表示同一个场景，模型必须满足 Permutation Invariance |
| **稀疏性** | 3D 空间中绝大多数位置是空的，数据分布极不均匀 |
| **不规则性** | 点的分布无固定拓扑，不像图像有规则网格，无法直接套用 2D 卷积 |
| **密度不均** | 近处密、远处疏，同一个语义物体在不同距离的采样密度差异巨大 |

### 2. 经典方法回顾

**PointNet (2017, Qi et al.)**
- 逐点 MLP + 对称函数（Max Pooling）实现排列不变性
- 优点：简洁优雅，首次直接处理原始点云
- 缺点：全局池化丢失所有局部几何结构

```
PointNet:
  Input Points (N×3) → Shared MLP → Point Features (N×1024) → Max Pool → Global Feature (1024)
                                                                              ↓
                                                             Classification / Segmentation
```

**PointNet++ (2017, Qi et al.)**
- 层级结构：FPS (最远点采样) + Ball Query (球查询) + PointNet (局部特征提取)
- 引入 Set Abstraction 层，逐层扩大感受野
- 仍然依赖 MLP，缺乏点与点之间的显式交互

**3D 稀疏卷积 (MinkowskiNet, spconv)**
- 将点云体素化后在稀疏体素上做卷积
- 高效但引入**量化损失**：连续坐标 → 离散网格
- 详见 [3D稀疏卷积](3D稀疏卷积.md)

### 3. Transformer 的天然契合

点云本质上是一个**无序集合 (Set)**，而 Transformer 最初就是为集合/序列设计的：
- Self-Attention 天然满足 **Permutation Equivariance**（加上位置编码后）
- 注意力机制能**显式建模点与点之间的关系**，而非像 MLP 那样独立处理每个点
- 通过位置编码引入几何信息，避免体素化的量化损失

> **关键洞察**：PointNet 用 MLP 独立处理每个点，再用 Max Pool 聚合；Transformer 用 Attention 让每个点主动"看到"邻居，信息交互更充分。

---

## 二、Point Transformer V1 (ICCV 2021, Zhao et al.)

### 1. 核心创新：Vector Attention（向量注意力）

标准 Transformer 使用 **Scalar Attention**，即每个 key-value 对只产生一个标量权重：

$$\alpha_{ij} = \text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right), \quad y_i = \sum_j \alpha_{ij} \cdot v_j$$

问题在于：同一个标量 $\alpha_{ij}$ 被广播到 value 的所有通道，不同通道无法获得差异化的权重。

Point Transformer V1 提出 **Vector Attention**：为 value 的每个通道分配独立的权重向量：

$$y_i = \sum_{j \in \mathcal{N}(i)} \rho\Big(\gamma(\varphi(x_i) - \psi(x_j) + \delta_{ij})\Big) \odot (\alpha(x_j) + \delta_{ij})$$

展开各符号：

| 符号 | 含义 |
|:-----|:-----|
| $\varphi, \psi, \alpha$ | 分别对应 Query、Key、Value 的线性映射 |
| $\delta_{ij}$ | **相对位置编码**，通过 MLP 编码 $p_i - p_j$ |
| $\gamma$ | MLP 映射（将注意力 logit 映射到权重向量） |
| $\rho$ | Softmax 归一化 |
| $\odot$ | **逐通道 (element-wise) 相乘**，而非标量乘 |

**关键区别**：$\rho(\cdot)$ 的输出是一个 $d$ 维向量而非标量，因此每个通道获得独立的注意力权重。

```
Vector Attention 计算流程:

  x_i ──→ Linear(Q) ──→ q_i ─────┐
                                   ├──→ q_i - k_j + δ_ij ──→ MLP(γ) ──→ softmax ──→ w_ij (d-dim vector)
  x_j ──→ Linear(K) ──→ k_j ─────┘                                                       │
                                                                                           ⊙ (element-wise)
  x_j ──→ Linear(V) ──→ v_j + δ_ij ─────────────────────────────────────────────────────→ Σ ──→ y_i

  p_i - p_j ──→ MLP ──→ δ_ij (位置编码，加到 Q-K 差和 V 上)
```

### 2. 相对位置编码

位置编码是点云 Transformer 的关键，因为几何结构信息完全靠坐标传递：

$$\delta_{ij} = \theta(p_i - p_j)$$

其中 $\theta$ 是一个两层 MLP。注意 $\delta_{ij}$ 被加到了 **两个地方**：
1. 注意力权重计算：$q_i - k_j + \delta_{ij}$（影响"看哪里"）
2. Value 加权：$v_j + \delta_{ij}$（影响"传什么"）

这种设计让模型同时在 **注意力分配** 和 **特征聚合** 两个环节都感知几何关系。

### 3. 局部注意力 (Local Attention)

全局注意力复杂度 $O(N^2)$，对于数万甚至数十万的点云不可行。V1 采用 **KNN 局部注意力**：

- 对每个点 $i$，通过 K 近邻搜索找到邻域 $\mathcal{N}(i)$（通常 $K=16$）
- 注意力仅在邻域内计算

$$y_i = \sum_{j \in \text{KNN}(i, K)} w_{ij} \odot (v_j + \delta_{ij})$$

### 4. 网络架构：U-Net 式编码器-解码器

```
输入点云 (N points)
    │
    ▼
┌──────────────────────┐
│ Stage 1: Attention ×n│  N points,   C dims
└──────────┬───────────┘
           │ Transition Down (FPS + MLP)
           ▼
┌──────────────────────┐
│ Stage 2: Attention ×n│  N/4 points, 2C dims
└──────────┬───────────┘
           │ Transition Down
           ▼
┌──────────────────────┐
│ Stage 3: Attention ×n│  N/16 points, 4C dims
└──────────┬───────────┘
           │ Transition Down
           ▼
┌──────────────────────┐
│ Stage 4: Attention ×n│  N/64 points, 8C dims
└──────────┬───────────┘
           │ Transition Down
           ▼
┌──────────────────────┐
│ Stage 5: Attention ×n│  N/256 points, 16C dims
└──────────┬───────────┘
           │ Transition Up (插值 + skip connection)
           ▼
      ... 逐层上采样回 N points ...
           │
           ▼
    语义分割 / 分类头
```

**Transition Down（下采样）**：
1. **FPS (Farthest Point Sampling)**：从 $N$ 个点中选取 $N/4$ 个最远点作为中心
2. 对每个中心点，用 KNN 找到邻居
3. 用 MLP + Max Pool 聚合邻居特征

**Transition Up（上采样）**：
1. 三线性插值将低分辨率特征插值到高分辨率点位置
2. 与 Encoder 对应层的 Skip Connection 相加

### 5. 实验结果

| 任务 | 数据集 | 指标 | PTv1 成绩 |
|:-----|:------|:-----|:----------|
| 语义分割 | S3DIS Area 5 | mIoU | 70.4% |
| 语义分割 | ScanNet | mIoU | 70.6% |
| 分类 | ModelNet40 | Acc | 93.7% |

---

## 三、Point Transformer V2 (NeurIPS 2022, Wu et al.)

V2 针对 V1 的三个瓶颈做出改进：计算效率、下采样策略、位置编码。

### 1. 分组向量注意力 (Grouped Vector Attention, GVA)

V1 的向量注意力在每个通道都独立计算权重，参数量和计算量较大。V2 借鉴 **Multi-Head Attention 的分组思想**：

- 将 $C$ 个通道分为 $G$ 组，每组 $C/G$ 个通道
- **组内共享同一个标量注意力权重**，组间独立

$$y_i = \sum_{j \in \mathcal{N}(i)} \text{softmax}\Big(\text{GroupReduce}\big(\gamma(q_i - k_j + \delta_{ij})\big)\Big) \odot (v_j + \delta_{ij})$$

其中 GroupReduce 将每组的 $C/G$ 维 logit 聚合为一个标量（如求和或均值），再广播回该组的所有通道。

| | V1 Vector Attention | V2 GVA |
|:--|:---|:---|
| 权重粒度 | 每通道独立 ($C$ 个权重) | 每组共享 ($G$ 个权重) |
| 等价关系 | $G = C$ 时退化为 V1 | $G = 1$ 时退化为标准 Scalar Attention |
| 优势 | 最大表达力 | 在表达力与效率之间取得更好平衡 |

### 2. 分区池化 (Partition-based Pooling / Grid Pooling)

V1 的下采样使用 **FPS（最远点采样）**，复杂度 $O(N^2)$，是整个管线的瓶颈。

V2 改用 **Grid Pooling**：
1. 将空间划分为均匀的 3D 网格（类似体素化）
2. 每个网格内的点通过平均池化合并为一个点
3. 新点的坐标 = 网格内点的坐标均值，特征 = 网格内点的特征均值

$$p'_g = \frac{1}{|S_g|}\sum_{i \in S_g} p_i, \quad f'_g = \frac{1}{|S_g|}\sum_{i \in S_g} f_i$$

| | FPS (V1) | Grid Pooling (V2) |
|:--|:---|:---|
| 复杂度 | $O(N^2)$ 或 $O(N \log N)$ | $O(N)$ |
| 采样均匀性 | 保证几何均匀 | 依赖网格大小 |
| 速度 | 慢（大规模点云瓶颈） | 快（适合大规模场景） |

### 3. 改进的位置编码

V2 引入 **条件位置编码 (Conditional Positional Encoding)**：位置编码不仅依赖坐标差 $p_i - p_j$，还依赖特征：

$$\delta_{ij} = \theta(p_i - p_j) + \phi(f_i - f_j)$$

其中 $\phi$ 是额外的 MLP，编码特征差异。这使位置编码能适应不同语义上下文。

### 4. 实验结果

| 任务 | 数据集 | PTv1 | PTv2 |
|:-----|:------|:-----|:-----|
| 语义分割 | ScanNet val | 70.6 | **75.4** |
| 语义分割 | S3DIS Area 5 | 70.4 | **71.6** |
| 语义分割 | nuScenes val | - | **80.2** |

V2 在速度上也有显著提升，归功于 Grid Pooling 替代 FPS。

---

## 四、Point Transformer V3 (CVPR 2024, Wu et al.)

V3 是一次**范式级别**的改变：完全放弃 KNN 注意力，转向序列化注意力。

### 1. KNN 注意力的瓶颈

| 问题 | 说明 |
|:-----|:-----|
| **KNN 搜索慢** | 每次 Attention 都需要 KNN 查询，$O(N \log N)$，且 GPU 不友好 |
| **不规则内存访问** | KNN 邻居在内存中不连续，导致严重的 Cache Miss |
| **无法利用 FlashAttention** | FlashAttention 要求连续的窗口结构，KNN 邻域无法满足 |
| **扩展性差** | 点云规模增大时，KNN 成为性能瓶颈 |

> **V3 的核心哲学**：与其在不规则邻域上做精确注意力，不如在序列化的规则窗口上做高效注意力。**精度与规模的权衡中，规模更重要。**

### 2. 空间填充曲线 (Space-Filling Curves)

V3 使用空间填充曲线将 3D 点云映射为 1D 序列，使得空间相邻的点在序列中也相邻。

**Z-order 曲线 (Morton Code)**：
- 将 $(x, y, z)$ 坐标的二进制位交错排列
- 例如：$x = 101_2, y = 011_2, z = 110_2 \Rightarrow \text{Morton} = 101\ 011\ 110_2$（按位交错）
- 计算简单，速度极快

**Hilbert 曲线**：
- 比 Z-order 保持更好的局部性（避免"跳跃"）
- 计算稍复杂，但局部连续性更强

```
Z-order 曲线示意 (2D):          Hilbert 曲线示意 (2D):

  0 ── 1    4 ── 5              0 ── 1    14──15
       │    │    │              │         │
  3 ── 2    7 ── 6              3 ── 2    13──12
                                     │    │
  8 ── 9   12──13               4 ── 5    10──11
       │    │    │                   │    │
 11──10   15──14                7 ── 6    9 ── 8
```

### 3. 序列化注意力 (Serialized Attention)

将 3D 点映射为 1D 序列后，就可以使用**窗口注意力**（类似 Swin Transformer）：

1. **序列化**：用空间填充曲线对所有点排序，得到 1D 序列
2. **窗口划分**：将序列切分为固定大小的窗口（如 window_size = 1024）
3. **窗口内注意力**：在每个窗口内做标准 Self-Attention
4. **窗口移位**：类似 Swin 的 shifted window，交替使用不同的窗口偏移

$$y_i = \text{Attention}(Q_i, K_{\mathcal{W}(i)}, V_{\mathcal{W}(i)})$$

其中 $\mathcal{W}(i)$ 是点 $i$ 所在的窗口。

**关键优势**：窗口内的点在内存中是**连续的**，可以直接使用 **FlashAttention** 加速！

### 4. 稀疏卷积作为位置编码 (Sparse Conv CPE)

V1/V2 使用显式的相对位置编码 $\delta_{ij} = \text{MLP}(p_i - p_j)$，但 V3 采用了完全不同的策略：**用 3D 稀疏卷积隐式提供位置信息**。

#### 为什么不再用显式相对位置编码？

V3 的窗口注意力是在**序列化后的 1D 窗口**上做的，不再有明确的"邻居对 $(i, j)$"的概念（窗口内所有点两两交互）。如果仍然计算每对点的 $p_i - p_j$，注意力矩阵每个元素都需要一个位置编码向量，计算量和内存开销很大，也无法享受 FlashAttention 的加速（FlashAttention 不支持逐元素的自定义 bias）。

#### Conditional Positional Encoding (CPE) 的做法

V3 借鉴了 2D 视觉 Transformer 中 CPE 的思想（来自 CPVT, Chu et al., 2021），用一个**轻量级的 depth-wise 3D 稀疏卷积**在 Attention Block 内注入位置信息：

$$X = X + \text{SpConv}_{\text{dw}}(X)$$

具体来说：

| 细节 | 说明 |
|:-----|:-----|
| **卷积类型** | 3D 稀疏深度可分离卷积 (Depth-wise Sparse Convolution) |
| **卷积核** | $3 \times 3 \times 3$，仅在体素化后的稀疏位置上操作 |
| **位置** | 放在每个 Attention Block 内部，**在 Self-Attention 之前**执行 |
| **作用** | 通过局部卷积的感受野，将 3D 空间邻域信息编码进特征中 |
| **参数量** | 极少（depth-wise，每个通道独立的 $3^3=27$ 个参数） |

```
Serialized Attention Block (含 Sparse Conv CPE):

输入: 点特征 X (M×C)
    │
    ▼ ① Sparse Conv CPE:
    │    X = X + DepthWise_SpConv3x3x3(X)
    │    → 在原始 3D 坐标上做稀疏卷积，注入局部几何信息
    │
    ▼ ② 多序列化窗口注意力:
    │    对 S=4 种序列化分别做 Window Self-Attention → 融合
    │
    ▼ ③ Residual + LayerNorm
    ▼ ④ FFN (MLP + GELU)
    ▼ ⑤ Residual + LayerNorm
    │
输出: X' (M×C)
```

#### 为什么稀疏卷积能替代显式位置编码？

1. **局部感受野 = 隐式位置感知**：$3 \times 3 \times 3$ 的稀疏卷积让每个点能感知其 3D 空间邻域的特征分布，相当于隐式编码了"我在哪里、周围有什么"
2. **条件性 (Conditional)**：编码结果依赖于输入特征 $X$（而非仅依赖坐标），能自适应不同语义上下文
3. **与序列化解耦**：稀疏卷积在**原始 3D 坐标空间**上操作，不受序列化顺序的影响——先在 3D 中注入位置信息，再在 1D 序列上做注意力
4. **极低开销**：depth-wise 卷积参数少、计算快，相比 V1 的逐对 MLP 位置编码高效得多

> **与 V1/V2 的对比**：
> - V1/V2：显式计算每对邻居的相对位置 $\delta_{ij} = \text{MLP}(p_i - p_j)$，嵌入到注意力权重和 Value 中
> - V3：通过 Sparse Conv 在 3D 空间中预先将位置信息"烤入"特征，Attention 本身不再需要位置编码
> 
> 这也是 V3 能使用**标准 Multi-Head Attention**（而非 Vector Attention）的前提——位置信息已经在特征中了，Attention 只需做标准的 $QK^TV$ 即可。

### 5. 多序列化 Ensemble

单一序列化顺序不可避免地会将某些空间邻居分到不同窗口。V3 的解决方案：

- 同时使用**多种序列化顺序**（如 4 种不同的空间填充曲线 / 不同的坐标轴排列）
- 每种顺序产生不同的窗口划分
- 多种划分的注意力结果**融合**

$$y_i = \frac{1}{S}\sum_{s=1}^{S} \text{WindowAttn}_s(x_i)$$

这种 ensemble 策略确保：即使在某种序列化中两个邻居被分隔，在另一种序列化中它们大概率在同一个窗口内。

```
多种序列化顺序示意:

  序列化 1 (Z-order, xyz):     [A B C D | E F G H | ...]
  序列化 2 (Z-order, zyx):     [A C E G | B D F H | ...]
  序列化 3 (Hilbert, xyz):     [A B D C | E F H G | ...]
  序列化 4 (Hilbert, zyx):     [A D E H | B C F G | ...]

  → 每种产生不同窗口，ensemble 后邻域覆盖更完整
```

### 6. Patch Attention

为进一步降低计算量，V3 引入 **Patch Attention**：

1. 将窗口内的点进一步划分为 Patch（如 patch_size = 64）
2. 每个 Patch 内的特征先通过聚合（如均值）得到一个 Patch Token
3. 在 Patch Token 层面做注意力（相当于在更粗粒度上做全局注意力）
4. 将结果分配回 Patch 内的各点

这类似于一种**两级注意力**：窗口内细粒度 + Patch 间粗粒度。

### 7. 整体架构

V3 保持了 U-Net 式编码器-解码器结构，但核心模块替换为序列化注意力：

```
输入点云
    │
    ├──→ 体素化 (Grid Sampling)
    │
    ├──→ 多种空间填充曲线序列化
    │
    ▼
┌────────────────────────────────────┐
│ Stage 1: Serialized Window Attn ×n │  N points
│         + Shifted Window           │
└──────────────┬─────────────────────┘
               │ Grid Pooling (2× 下采样)
               ▼
┌────────────────────────────────────┐
│ Stage 2: Serialized Window Attn ×n │  N/8 points
└──────────────┬─────────────────────┘
               │ Grid Pooling
               ▼
          ... (更多 Stage) ...
               │
               ▼
          Decoder (上采样 + Skip Connection)
               │
               ▼
          任务头 (分割 / 检测)
```

### 8. 前向过程详解：点云如何被 Encoder 压缩、被 Decoder 还原

下面按照 PTV3 前向传播的实际数据流，逐步追踪一个点云从输入到输出的完整路径。

#### Phase 0：输入预处理

**原始输入**：点云 $P = \{(p_i, f_i)\}_{i=1}^{N}$，其中 $p_i \in \mathbb{R}^3$ 是坐标，$f_i$ 是原始特征（如颜色、法向量、强度等）。

**Step 1 — 体素化 (Grid Sampling)**：

```
原始点云 (N 个点，如 ~100K)
    │
    ▼ 用固定大小的 3D 网格量化坐标
    │  同一个体素内的多个点 → 平均坐标 + 平均特征 → 合并为 1 个点
    ▼
体素化后的点云 (N' 个点，如 ~40K)
```

- 不可学习，纯数据预处理；目的是去除重叠点、统一密度、减少点数
- 体素大小（如 2cm）是超参数，决定了初始分辨率

**Step 2 — 特征嵌入 (Feature Embedding)**：

```
体素化的点 (N'×3 坐标 + N'×d_raw 原始特征)
    │
    ▼ Linear / MLP
    ▼
嵌入后的点特征 (N'×C)，如 C=64
```

**Step 3 — 多序列化排序**：

```
N' 个点的 3D 坐标
    │
    ├──→ Z-order 曲线 (xyz 轴序)  → 排列 σ₁
    ├──→ Z-order 曲线 (zyx 轴序)  → 排列 σ₂
    ├──→ Hilbert 曲线 (xyz 轴序)  → 排列 σ₃
    └──→ Hilbert 曲线 (zyx 轴序)  → 排列 σ₄
```

- 每种空间填充曲线将 3D 坐标转为 1D 排序索引，得到 $S=4$ 种不同的点排列
- 这些排列在每个 Stage 开头只需算一次（下采样后坐标改变时重新计算）

#### Phase 1：Encoder（逐层压缩）

Encoder 由多个 Stage 组成（典型 5 个），每个 Stage = **若干 Serialized Attention Block** + **一次 Grid Pooling 下采样**。

**Serialized Attention Block 内部流程**：

```
输入: 点特征 X (M×C)
    │
    ▼ ① Sparse Conv CPE (在原始 3D 坐标上):
    │    X = X + DepthWise_SpConv3x3x3(X)
    │    → 注入 3D 空间位置信息
    │
    │ ② 对 S=4 种序列化分别做窗口注意力:
    │
    ├──→ 按 σ₁ 重排 → 切窗口(w=1024) → Window Self-Attention → 结果 y₁
    ├──→ 按 σ₂ 重排 → 切窗口(w=1024) → Window Self-Attention → 结果 y₂
    ├──→ 按 σ₃ 重排 → 切窗口(w=1024) → Window Self-Attention → 结果 y₃
    └──→ 按 σ₄ 重排 → 切窗口(w=1024) → Window Self-Attention → 结果 y₄
    │
    ▼ ③ 融合: y = (y₁ + y₂ + y₃ + y₄) / 4
    ▼ + Residual Connection + LayerNorm
    ▼ ④ FFN (两层 MLP + GELU)
    ▼ + Residual Connection + LayerNorm
    ▼
输出: 更新后的点特征 X' (M×C)
```

- **Sparse Conv CPE** 在 Attention 之前执行，在原始 3D 空间中注入位置信息，使后续标准 Attention 无需显式位置编码
- 窗口内做**标准 Multi-Head Self-Attention**（不再是 V1/V2 的 Vector Attention），因为位置信息已通过 SpConv 编码进特征
- 窗口内数据在内存中连续，直接调用 **FlashAttention** 加速
- 交替层使用 **Shifted Window**（窗口偏移 $w/2$），让窗口边界的点也能交互

**Stage 间的 Grid Pooling（下采样）**：

```
Stage k 输出: M 个点, C_k 维
    │
    ▼ 体素大小放大 2×（如 2cm → 4cm → 8cm → 16cm → 32cm）
    │  每个新体素内:  坐标 → 均值，特征 → 均值
    ▼ Linear: C_k → C_{k+1}（通道扩大）
    ▼
Stage k+1 输入: ~M/8 个点, C_{k+1} 维
```

> 体素边长 2×，体积 $2^3=8$ 倍，点数约变为 1/8。

**Encoder 整体数据流**：

```
输入: ~40K 个点, C=64
    │
    ▼ [Stage 1] Attention Block × n₁
    │  40K × 64                          ──→ 保存 skip₁
    ▼ Grid Pooling (体素 2×)
    │
    ▼ [Stage 2] Attention Block × n₂
    │  ~5K × 128                         ──→ 保存 skip₂
    ▼ Grid Pooling (体素 2×)
    │
    ▼ [Stage 3] Attention Block × n₃
    │  ~600 × 256                        ──→ 保存 skip₃
    ▼ Grid Pooling (体素 2×)
    │
    ▼ [Stage 4] Attention Block × n₄
    │  ~80 × 512                         ──→ 保存 skip₄
    ▼ Grid Pooling (体素 2×)
    │
    ▼ [Stage 5] Attention Block × n₅
    │  ~10 × 512                         ← bottleneck，最压缩的表示
```

**Encoder 压缩了什么？**
- **空间分辨率**：从 ~40K 点压缩到 ~10 个点，每个点代表一大片区域
- **语义抽象**：低层（边缘、曲率） → 高层（物体类别、场景语义）
- **通道补偿**：64 → 128 → 256 → 512，用更多通道编码更丰富的语义信息

#### Phase 2：Decoder（逐层还原）

Decoder 将底层粗粒度的语义信息**逐层上采样**回原始分辨率，用于逐点预测。

**每一步上采样**：

```
低分辨率特征: M_low 个点, C_high 维
    │
    ▼ ① 插值上采样:
    │    对高分辨率的每个点 p_i，找低分辨率中最近的 k 个点
    │    用距离倒数加权插值:
    │    f_i^{up} = Σ_j (w_j · f_j^{low}) / Σ_j w_j，其中 w_j = 1/||p_i - p_j||²
    │
    ▼ ② Skip Connection:
    │    与 Encoder 对应 Stage 保存的特征相加（或 concat 后过 Linear）
    │    f_i^{combined} = f_i^{up} + skip_k[i]
    │
    ▼ ③ Linear / MLP: 调整通道维度 C_high → C_low
    │
    ▼ ④ (可选) Attention Block 精炼特征
    │
    ▼
高分辨率特征: M_high 个点, C_low 维
```

**Decoder 整体数据流**：

```
Encoder Stage 5 输出: ~10 × 512
    │
    ▼ 插值上采样 + skip₄  →  ~80 × 512
    ▼ 插值上采样 + skip₃  →  ~600 × 256
    ▼ 插值上采样 + skip₂  →  ~5K × 128
    ▼ 插值上采样 + skip₁  →  ~40K × 64
    │
    ▼
Decoder 输出: ~40K × 64（每个点一个 64 维特征向量）
```

#### Phase 3：任务头

```
Decoder 输出: N' × 64
    │
    ▼ Linear(64 → num_classes)
    ▼ softmax → 每个点的类别预测
    ▼ Loss: CrossEntropy(pred, label)
```

#### 完整前向流程一览

```
原始点云 (~100K 点, xyz+rgb)
    │
    ▼ 体素化 → ~40K 点
    ▼ 特征嵌入 → 40K × 64
    ▼ 空间填充曲线排序 (4种)
    │
    ╔═══════════ ENCODER ═══════════╗
    ║                               ║
    ║  Stage1: 40K×64   ──→ skip₁   ║
    ║     ↓ Grid Pool (÷8)         ║
    ║  Stage2: 5K×128   ──→ skip₂   ║
    ║     ↓ Grid Pool (÷8)         ║
    ║  Stage3: 600×256  ──→ skip₃   ║
    ║     ↓ Grid Pool (÷8)         ║
    ║  Stage4: 80×512   ──→ skip₄   ║
    ║     ↓ Grid Pool (÷8)         ║
    ║  Stage5: 10×512   (bottleneck)║
    ║                               ║
    ╚═══════════════════════════════╝
    │
    ╔═══════════ DECODER ═══════════╗
    ║                               ║
    ║  ↑ 插值 + skip₄ → 80×512     ║
    ║  ↑ 插值 + skip₃ → 600×256    ║
    ║  ↑ 插值 + skip₂ → 5K×128     ║
    ║  ↑ 插值 + skip₁ → 40K×64     ║
    ║                               ║
    ╚═══════════════════════════════╝
    │
    ▼ Linear → 40K × num_classes
    ▼ 逐点语义分割预测
```

> **设计哲学**：Encoder 通过 Grid Pooling 不断增大体素尺寸，减少点数、提升通道数，在越来越粗的粒度上做序列化窗口注意力，逐步提取高级语义。Decoder 通过插值 + Skip Connection 将语义信息还原到每个点，Skip Connection 补回下采样丢失的几何细节。V3 的关键突破在于 Attention 不再依赖 KNN 搜索邻居，而是通过空间填充曲线排序后直接在连续内存窗口上做标准 Attention，可调用 FlashAttention，速度提升 3 倍以上。

### 9. 实验结果

| 任务 | 数据集 | PTv2 | PTv3 | 速度对比 |
|:-----|:------|:-----|:-----|:---------|
| 语义分割 | ScanNet val | 75.4 | **77.5** | 3.3x faster |
| 语义分割 | nuScenes val | 80.2 | **81.2** | - |
| 实例分割 | ScanNet | 75.5 | **77.7** | - |
| 语义分割 | S3DIS 6-fold | 76.8 | **78.6** | - |

**核心结论**：V3 不仅更快（3x+ 加速），精度也更高，证明了**放弃精确邻域、拥抱高效序列化**的正确性。

---

## 五、系列对比总览

| 版本 | 年份 | 注意力类型 | 邻域定义 | 位置编码 | 下采样 | 核心亮点 |
|:-----|:-----|:----------|:--------|:---------|:-------|:---------|
| **V1** | 2021 | Vector Attention | KNN ($K=16$) | 显式相对 PE: MLP($p_i - p_j$) | FPS | 首次将 Transformer 成功应用于点云，向量注意力 |
| **V2** | 2022 | Grouped Vector Attention | KNN | 条件相对 PE: MLP($p_i - p_j$) + MLP($f_i - f_j$) | Grid Pooling | 分组降低计算量，Grid Pooling 替代 FPS |
| **V3** | 2024 | Serialized Window Attention | 序列化窗口 | Sparse Conv CPE: DW-SpConv $3^3$ | Grid Pooling | 放弃 KNN，空间填充曲线 + FlashAttention，3x 提速 |

**演进主线**：
1. V1 → V2：**优化效率**（分组注意力 + Grid Pooling）
2. V2 → V3：**范式转换**（KNN 邻域 → 序列化窗口，拥抱硬件友好的计算模式）

---

## 六、核心公式速查

### Vector Attention (V1)

$$y_i = \sum_{j \in \mathcal{N}(i)} \underbrace{\rho\Big(\gamma(\varphi(x_i) - \psi(x_j) + \delta_{ij})\Big)}_{\text{d-dim 权重向量}} \odot \underbrace{(\alpha(x_j) + \delta_{ij})}_{\text{value + 位置编码}}$$

### Grouped Vector Attention (V2)

$$w_{ij}^{(g)} = \text{softmax}\left(\frac{1}{C/G}\sum_{c \in \text{Group}(g)} [\gamma(q_i - k_j + \delta_{ij})]_c\right)$$

$$y_i = \sum_{j \in \mathcal{N}(i)} \text{Broadcast}(w_{ij}^{(1:G)}) \odot (v_j + \delta_{ij})$$

### Serialized Window Attention (V3)

$$y_i = \frac{1}{S}\sum_{s=1}^{S} \sum_{j \in \mathcal{W}_s(i)} \text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right) v_j$$

其中 $\mathcal{W}_s(i)$ 是第 $s$ 种序列化下点 $i$ 所在的窗口。

---

## 七、与其他方法的对比

| 方法 | 优势 | 劣势 |
|:-----|:-----|:-----|
| **PointNet/PointNet++** | 简单，轻量 | MLP 独立处理，局部交互弱 |
| **3D 稀疏卷积** | 高效，成熟工程生态 | 体素化量化损失，感受野固定 |
| **Point Transformer** | 自适应感受野，动态权重 | V1/V2 有 KNN 瓶颈 |
| **PTv3** | 速度快，可扩展到百万点 | 序列化近似可能丢失部分邻域信息 |
| **MinkowskiNet** | 与 2D ResNet 类似的设计，易迁移 | 量化损失，卷积核固定 |

---

## 八、面试高频问题

### Q1: Vector Attention 和标准 Scaled Dot-Product Attention 的区别？

**答**：
- **标准注意力**：$q \cdot k$ 产生一个**标量**权重，所有通道共享同一权重
- **Vector Attention**：$\gamma(q - k + \delta)$ 产生一个**向量**权重，每个通道有独立权重
- 另外，标准注意力使用**点积**度量相似性，而 Vector Attention 使用**减法 + MLP**，表达力更强
- Vector Attention 的位置编码同时作用于注意力权重和 value，融入更充分

### Q2: 点云处理为什么天然适合 Transformer？

**答**：
- 点云是**无序集合**，Transformer 的 Self-Attention 天然对集合操作（排列等变）
- Attention 的**动态权重**可以自适应不同几何结构（vs. 卷积的固定权重）
- 位置编码机制可以灵活引入 3D 坐标信息
- 点云密度不均时，Attention 可以自适应调整关注范围

### Q3: PTv3 为什么放弃 KNN？空间填充曲线的作用？

**答**：
- KNN 搜索本身 $O(N \log N)$，且产生**不规则内存访问模式**，无法利用 FlashAttention
- 空间填充曲线（Z-order / Hilbert）将 3D 点映射为 1D 序列，**保持空间局部性**
- 映射后可以使用**连续内存的窗口注意力**，直接调用 FlashAttention
- 多种序列化 ensemble 弥补单一曲线的邻域损失
- 结果：3x+ 加速，精度反而更高（因为可以处理更大的点云 / 更大的模型）

### Q4: PTv3 不用显式位置编码了，怎么感知 3D 几何？

**答**：
- V1/V2 用显式相对位置编码 $\text{MLP}(p_i - p_j)$，嵌入到每对注意力权重中
- V3 放弃了这种方式，因为：(1) 窗口注意力没有固定的"邻居对"概念；(2) 逐对计算 PE 无法利用 FlashAttention
- V3 改用 **Sparse Conv CPE**：在每个 Attention Block 前，用一个 **depth-wise 3D 稀疏卷积** ($3 \times 3 \times 3$) 将 3D 空间邻域信息编码进特征
- 公式：$X = X + \text{DW-SpConv}_{3 \times 3 \times 3}(X)$
- 稀疏卷积在**原始 3D 坐标空间**上操作，与序列化解耦。先在 3D 中注入位置信息，再在 1D 序列上做标准 Attention
- 这也是 V3 能用**标准 Multi-Head Attention**（不再需要 Vector Attention）的关键前提

### Q5: Point Transformer vs 3D 稀疏卷积各自优势？

**答**：
| 维度 | Point Transformer | 3D 稀疏卷积 |
|:-----|:-----------------|:------------|
| 权重 | 动态（数据依赖） | 固定（学习后不变） |
| 感受野 | 自适应（Attention 范围） | 固定（卷积核大小） |
| 精度 | 一般更高 | 略低但差距在缩小 |
| 速度 | V3 已非常快 | 工程优化更成熟 (spconv) |
| 量化损失 | 无（直接操作点坐标） | 有（体素化） |
| 工程生态 | 较新 | 成熟（spconv, TorchSparse） |

### Q6: PTv3 的序列化会丢失邻域信息吗？如何缓解？

**答**：
- 单一序列化确实可能将空间邻居分到不同窗口
- 缓解策略：(1) **多种序列化 ensemble**（4种不同顺序）；(2) **Shifted Window**（窗口偏移）；(3) **多层堆叠**（信息逐层传播）
- 实验证明这种"近似"不仅没有损害精度，反而因为可以扩大模型规模和数据量而提升了性能

---

## 九、PTV3 前向传播过程详解

下面按照 PTV3 前向传播的实际数据流，一步一步讲清楚点云从输入到 Encoder 压缩、再到 Decoder 还原的全过程。

### 📥 Phase 0：输入预处理

原始输入：点云 $P = \{(p_i, f_i)\}_{i=1}^{N}$，其中 $p_i \in \mathbb{R}^3$ 是坐标，$f_i$ 是原始特征（如颜色、法向量、强度等）。

#### Step 1：体素化 (Grid Sampling)

```
原始点云 (N 个点，如 ~100K)
    │
    ▼ 用固定大小的 3D 网格量化坐标
    │  同一个体素内的多个点 → 平均坐标 + 平均特征 → 合并为 1 个点
    ▼
体素化后的点云 (N' 个点，如 ~40K)
```

- 这一步**不是可学习的**，纯粹是数据预处理
- 目的：去除重叠点、统一密度、减少点数
- 体素大小（如 2cm）是超参数，决定了初始分辨率

#### Step 2：特征嵌入 (Feature Embedding)

```
体素化的点 (N'×3 坐标 + N'×d_raw 原始特征)
    │
    ▼ Linear / MLP
    │
    ▼
嵌入后的点特征 (N'×C)，如 C=64
```

将原始特征映射到统一的 $C$ 维隐空间。

#### Step 3：多序列化排序

```
N' 个点的 3D 坐标
    │
    ├──→ Z-order 曲线 (xyz 轴序)  → 排列 σ₁
    ├──→ Z-order 曲线 (zyx 轴序)  → 排列 σ₂
    ├──→ Hilbert 曲线 (xyz 轴序)  → 排列 σ₃
    └──→ Hilbert 曲线 (zyx 轴序)  → 排列 σ₄
```

- 每种空间填充曲线将 3D 坐标转为 1D 排序索引
- 得到 $S=4$ 种不同的点排列方式
- 这些排列在每个 Stage 开头只需算一次（坐标不变时）

---

### 🔽 Phase 1：Encoder（逐层压缩）

Encoder 由多个 Stage 组成（典型配置是 5 个 Stage），每个 Stage 包含**若干 Serialized Attention Block + 一次 Grid Pooling 下采样**。

#### 每个 Serialized Attention Block 内部

```
输入: 点特征 X (M×C)
    │
    ▼ ① Sparse Conv CPE (在原始 3D 坐标上):
    │    X = X + DepthWise_SpConv3x3x3(X)
    │    → 通过 3D 稀疏卷积注入空间位置信息
    │
    │ ② 对 S=4 种序列化分别做窗口注意力:
    │
    ├──→ 按 σ₁ 重排 → 切窗口(1024) → Window Self-Attention → 结果 y₁
    ├──→ 按 σ₂ 重排 → 切窗口(1024) → Window Self-Attention → 结果 y₂
    ├──→ 按 σ₃ 重排 → 切窗口(1024) → Window Self-Attention → 结果 y₃
    └──→ 按 σ₄ 重排 → 切窗口(1024) → Window Self-Attention → 结果 y₄
    │
    ▼ ③ 融合: y = (y₁ + y₂ + y₃ + y₄) / 4
    │
    ▼ + Residual Connection + LayerNorm
    │
    ▼ ④ FFN (两层 MLP + GELU)
    │
    ▼ + Residual Connection + LayerNorm
    │
    ▼
输出: 更新后的点特征 X' (M×C)
```

**Sparse Conv CPE 细节**：

- 在 Attention 之前，先在**原始 3D 坐标空间**上执行一次 depth-wise 3D 稀疏卷积
- $3 \times 3 \times 3$ 卷积核让每个点感知其 3D 空间邻域，将几何位置信息编码进特征
- 这使后续的 Window Self-Attention **不需要显式位置编码**，可以直接用标准 $QK^TV$
- 稀疏卷积与序列化解耦：SpConv 在 3D 中操作，Attention 在 1D 序列中操作

**Window Self-Attention 细节**：

- 序列被切成不重叠的窗口，每个窗口 $w = 1024$ 个点
- 窗口内做标准的 **Multi-Head Self-Attention**（不是 V1/V2 的 Vector Attention！位置信息已由 SpConv CPE 注入）
- 因为窗口内数据在内存中连续，可以直接调用 **FlashAttention**
- 交替层使用 **Shifted Window**（窗口偏移 $w/2$），让相邻窗口的边界点也能交互

$$y_i = \frac{1}{S} \sum_{s=1}^{S} \sum_{j \in \mathcal{W}_s(i)} \text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d}}\right) v_j$$

#### Stage 之间的 Grid Pooling（下采样）

```
Stage k 的输出: M 个点，特征维度 C_k
    │
    ▼ 将体素大小放大 2×（如从 2cm → 4cm → 8cm → 16cm → 32cm）
    │  每个新体素内的点:
    │    - 坐标 → 取均值（新的点坐标）
    │    - 特征 → 取均值（新的点特征）
    │
    ▼ Linear: C_k → C_{k+1}（通道扩大，如 64→128→256→512→512）
    │
    ▼
Stage k+1 的输入: M/8 个点，特征维度 C_{k+1}
```

> **为什么是 M/8？** 体素边长扩大 2×，体积变 $2^3 = 8$ 倍，所以点数大约变为 1/8。

#### Encoder 整体流程

```
输入: N' 个点, C=64
    │
    ▼ [Stage 1] Attention Block × n₁
    │  N' 个点, C=64                      ──→ 保存特征 F₁ (skip connection)
    │
    ▼ Grid Pooling (体素 2×)
    │
    ▼ [Stage 2] Attention Block × n₂
    │  ~N'/8 个点, C=128                  ──→ 保存特征 F₂
    │
    ▼ Grid Pooling (体素 2×)
    │
    ▼ [Stage 3] Attention Block × n₃
    │  ~N'/64 个点, C=256                 ──→ 保存特征 F₃
    │
    ▼ Grid Pooling (体素 2×)
    │
    ▼ [Stage 4] Attention Block × n₄
    │  ~N'/512 个点, C=512                ──→ 保存特征 F₄
    │
    ▼ Grid Pooling (体素 2×)
    │
    ▼ [Stage 5] Attention Block × n₅
    │  ~N'/4096 个点, C=512               ──→ 最底层（最压缩的表示）
    │
    ▼
Encoder 输出: 极少的点（如 ~10 个），每个点携带 512 维的高级语义特征
```

**Encoder 压缩了什么？**

- **空间分辨率**：从 ~40K 点压缩到 ~10 个点（每个代表一大片区域）
- **语义抽象**：低层特征（边缘、曲率） → 高层特征（物体类别、场景语义）
- **通道维度增加**：64 → 128 → 256 → 512，用更多通道编码更丰富的语义信息

---

### 🔼 Phase 2：Decoder（逐层还原）

Decoder 的任务是把底层粗粒度的语义信息逐层上采样回原始分辨率，用于逐点预测（如语义分割）。

#### 每一步上采样

```
低分辨率特征: M_low 个点, C_high 维
    │
    ▼ ① 插值上采样:
    │    对高分辨率的每个点 p_i，找低分辨率中最近的 k 个点
    │    用距离倒数加权插值:
    │
    │    f_i^{up} = Σ_j (w_j · f_j^{low}) / Σ_j w_j
    │    其中 w_j = 1 / ||p_i - p_j||²
    │
    ▼ ② Skip Connection:
    │    与 Encoder 对应 Stage 保存的特征 F_k 拼接或相加
    │    f_i^{combined} = f_i^{up} + F_k[i]   (或 concat 后过 Linear)
    │
    ▼ ③ Linear / MLP:
    │    调整通道维度 C_high → C_low
    │
    ▼ ④ (可选) 再过一个 Attention Block 精炼特征
    │
    ▼
高分辨率特征: M_high 个点, C_low 维
```

#### Decoder 整体流程

```
Encoder Stage 5 输出: ~10 个点, 512 维
    │
    ▼ 上采样 + Skip(F₄)
    │  ~N'/512 个点, 512 维
    │
    ▼ 上采样 + Skip(F₃)
    │  ~N'/64 个点, 256 维
    │
    ▼ 上采样 + Skip(F₂)
    │  ~N'/8 个点, 128 维
    │
    ▼ 上采样 + Skip(F₁)
    │  N' 个点, 64 维
    │
    ▼
Decoder 输出: N' 个点，每个点一个 64 维特征向量
```

---

### 📤 Phase 3：任务头

```
Decoder 输出: N' × 64
    │
    ▼ Linear(64 → num_classes)
    │
    ▼ 语义分割: 每个点的类别 logits → softmax → 类别预测
    │
    ▼ Loss: CrossEntropy(pred, label)
```

---

### 🎯 总结：完整前向流程图

```
原始点云 (~100K 点, xyz+rgb)
    │
    ▼ 体素化 → ~40K 点
    ▼ 特征嵌入 → 40K × 64
    ▼ 空间填充曲线排序 (4种)
    │
    ╔═══════════ ENCODER ═══════════╗
    ║                               ║
    ║  Stage1: 40K×64   ──→ skip₁   ║
    ║     ↓ Grid Pool (÷8)         ║
    ║  Stage2: 5K×128   ──→ skip₂   ║
    ║     ↓ Grid Pool (÷8)         ║
    ║  Stage3: 600×256  ──→ skip₃   ║
    ║     ↓ Grid Pool (÷8)         ║
    ║  Stage4: 80×512   ──→ skip₄   ║
    ║     ↓ Grid Pool (÷8)         ║
    ║  Stage5: 10×512   (bottleneck)║
    ║                               ║
    ╚═══════════════════════════════╝
    │
    ╔═══════════ DECODER ═══════════╗
    ║                               ║
    ║  ↑ 插值 + skip₄ → 80×512     ║
    ║  ↑ 插值 + skip₃ → 600×256    ║
    ║  ↑ 插值 + skip₂ → 5K×128     ║
    ║  ↑ 插值 + skip₁ → 40K×64     ║
    ║                               ║
    ╚═══════════════════════════════╝
    │
    ▼ Linear → 40K × num_classes
    ▼ 逐点语义分割预测
```

**核心设计哲学**：

- **Encoder**：通过 Grid Pooling 不断增大体素尺寸，减少点数、提升通道数，在越来越粗的粒度上做序列化窗口注意力，逐步提取高级语义
- **Decoder**：通过插值 + Skip Connection 将语义信息还原到每一个点上，Skip Connection 补回下采样丢失的几何细节
- **V3 的关键突破**：Attention 不再依赖 KNN 搜索邻居，而是通过空间填充曲线排序后直接在连续内存窗口上做标准 Attention，可以调用 FlashAttention，速度提升 3 倍以上

---

[返回3D视觉](README.md) | [返回视觉](../README.md)

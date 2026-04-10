# MLA 详解 (Multi-head Latent Attention)

Multi-head Latent Attention（多头潜在注意力，简称 MLA）是 DeepSeek-V2/V3 引入的注意力机制创新，通过**低秩压缩 KV Cache** 大幅降低推理时的显存占用，同时保持与标准多头注意力相当甚至更优的模型性能。

本文从零开始讲解 MLA，适合有 Transformer 基础但没有看过 DeepSeek-V2 论文的读者。

---

## 一、为什么需要 MLA？

### 1.1 KV Cache 的显存瓶颈

在 Transformer 推理时，为了避免重复计算，会把每个 token 的 Key 和 Value 缓存起来，这就是 **KV Cache**。

自回归生成的过程如下：

```
生成第 1 个 token:
  输入: [prompt]
  → 计算所有 token 的 K, V → 缓存到 KV Cache
  → 用最后一个位置的 Q 与 K 做注意力 → 输出第 1 个 token

生成第 2 个 token:
  输入: [prompt + token_1]
  → 只需计算 token_1 的 K, V → 追加到 KV Cache
  → 用最新位置的 Q 与缓存中所有 K 做注意力 → 输出第 2 个 token

生成第 n 个 token:
  → 只计算最新 token 的 K, V → 追加到 KV Cache
  → 所有历史 K, V 直接从 Cache 读取（无需重算）
```

**KV Cache 的代价**：每一层、每一个头、每一个历史 token，都要存一份 K 和 V。显存消耗随序列长度线性增长。

对于标准 MHA（Multi-Head Attention），每层 KV Cache 的大小为：

```
每层 KV Cache = 2 × n_heads × d_head × seq_len × bytes_per_element

以 LLaMA-3 70B 为例：
  n_heads = 64, d_head = 128, seq_len = 8192, FP16 (2 bytes)
  每层 KV Cache = 2 × 64 × 128 × 8192 × 2 ≈ 268 MB
  全模型 80 层 = 80 × 268 MB ≈ 21 GB
```

**21 GB 仅仅用于 KV Cache**，这还不包括模型权重本身（70B 参数 ≈ 140 GB BF16）。

在实际批量推理场景下，如果同时处理 batch_size=32 的请求，KV Cache 就需要 21 × 32 ≈ 672 GB，这对硬件要求极高。

### 1.2 已有方案的不足

为了缓解 KV Cache 的显存压力，学界提出了多种方案：

**MQA（Multi-Query Attention，2019）**：所有 Q 头共用同一份 K/V。显存缩减效果显著，但共享程度过高，模型质量有所下降。

**GQA（Grouped-Query Attention，2023）**：将 Q 头分组，每组共用一份 K/V。LLaMA-2/3、Qwen 等模型使用此方案，是 MQA 和 MHA 之间的折中。

```
MHA:  Q1 K1V1 | Q2 K2V2 | Q3 K3V3 | Q4 K4V4   ← 4 个 Q 各用独立 KV
GQA:  Q1 Q2 → K1V1 | Q3 Q4 → K2V2              ← 每 2 个 Q 共用一份 KV
MQA:  Q1 Q2 Q3 Q4 → K1V1                        ← 所有 Q 共用同一份 KV
```

GQA 减小了 KV Cache，但其本质仍然是"减少 KV 头的数量"，每个 KV 头的 d_head 维度没有变化，**压缩空间有限**。

**核心矛盾**：MQA/GQA 通过减少头数来减小显存，但头数减少意味着注意力的表达能力下降；要保住表达能力，就得保住头数，KV Cache 就无法大幅压缩。

MLA 的思路完全不同——它不减少头数，而是**对 KV 做低秩压缩**，把所有头的 K/V 信息一起压缩到一个低维的潜在向量里，推理时只缓存这个向量，需要时再解压。

---

## 二、MLA 的核心思想

### 2.1 类比：压缩与解压

可以用一个简单的类比来理解 MLA 的核心思想：

> 标准 KV Cache 就像把一本书的每一页都完整保存。一本 500 页的书，你存了 500 页。
>
> MLA 的做法像是给这本书做了一个精华摘要（潜在向量 c），需要某一页的内容时，从摘要里恢复出来。摘要只有 50 页，存储空间减少到 1/10。

关键点：**缓存的不是 K 和 V，而是生成 K 和 V 之前的"中间压缩表示"**，即潜在向量 c_KV。

### 2.2 低秩分解的直觉

从矩阵的角度理解：

```
标准 MHA：
  K = x · W_K    ← x: (seq, d_model), W_K: (d_model, n_heads × d_head)
  V = x · W_V

  KV Cache 存: K 和 V，大小都是 (seq, n_heads × d_head)

MLA 的做法：
  c_KV = x · W_dkv    ← 压缩：d_model → d_c，d_c << n_heads × d_head
  K = c_KV · W_uk     ← 解压：d_c → n_heads × d_head
  V = c_KV · W_uv

  KV Cache 只存: c_KV，大小是 (seq, d_c)
```

这里 W_dkv 是"下投影"矩阵（d: down，降维），W_uk 和 W_uv 是"上投影"矩阵（u: up，升维）。整体上，W_K ≈ W_dkv · W_uk，即用两个小矩阵的乘积近似原来的大矩阵，这正是**低秩分解**的思想。

**为什么低秩假设合理？** 语言模型中，不同位置、不同层的 K/V 往往存在大量冗余，实际有效信息量（即矩阵秩）远小于理论维度上限。这一点已被大量关于注意力头冗余性的研究所证实。

---

## 三、数学推导

### 3.1 标准 MHA 回顾

给定输入序列 $X \in \mathbb{R}^{L \times d}$（$L$ 为序列长度，$d$ 为模型维度），MHA 的计算为：

$$Q_i = X W_Q^i, \quad K_i = X W_K^i, \quad V_i = X W_V^i$$

$$\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^\top}{\sqrt{d_h}}\right) V_i$$

$$\text{MHA}(X) = \text{concat}(\text{head}_1, \ldots, \text{head}_{n_h}) W_O$$

其中 $n_h$ 为头数，$d_h$ 为每头维度（满足 $n_h \times d_h = d$）。

**推理时 KV Cache 大小（每层）**：

$$\text{Size}_{\text{KV}} = 2 \times n_h \times d_h \times L = 2 \times d \times L$$

### 3.2 GQA 的改进

GQA 将 $n_h$ 个 Q 头分为 $g$ 组，每组共用一对 K/V 头（$n_h / g$ 个 Q 头共用一组）：

$$\text{Size}_{\text{KV}}^{\text{GQA}} = 2 \times g \times d_h \times L$$

当 $g = n_h$ 时退化为 MHA，当 $g = 1$ 时退化为 MQA。LLaMA-3 70B 使用 $g = 8$（64 个 Q 头，8 个 KV 头），相比 MHA 节省了 8 倍 KV Cache。

### 3.3 MLA 的压缩方案

MLA 引入一个**联合压缩向量** $c_{KV}$，同时压缩 K 和 V：

**压缩（Down-projection）**：

$$c_{KV} = X W_{DKV}$$

其中 $W_{DKV} \in \mathbb{R}^{d \times d_c}$，$d_c \ll n_h \times d_h$。每个 token 的 $c_{KV} \in \mathbb{R}^{d_c}$。

**解压（Up-projection）**，在实际注意力计算时执行：

$$K = c_{KV} W_{UK}, \quad W_{UK} \in \mathbb{R}^{d_c \times (n_h \times d_h)}$$
$$V = c_{KV} W_{UV}, \quad W_{UV} \in \mathbb{R}^{d_c \times (n_h \times d_h)}$$

**KV Cache 只缓存 $c_{KV}$**：

$$\text{Size}_{\text{KV}}^{\text{MLA}} = d_c \times L$$

注意这里不再有系数 2，因为 K 和 V 从同一个 $c_{KV}$ 解压出来，只需缓存一份 $c_{KV}$。

### 3.4 Q 的压缩（同样适用）

MLA 对 Q 也做了类似的低秩压缩：

$$c_Q = X W_{DQ}, \quad W_{DQ} \in \mathbb{R}^{d \times d_c'}$$
$$Q = c_Q W_{UQ}, \quad W_{UQ} \in \mathbb{R}^{d_c' \times (n_h \times d_h)}$$

**Q 的压缩不影响 KV Cache 大小**（因为 Q 在解码时只对当前位置计算，不需要缓存历史值），但可以减少矩阵乘法的中间计算量，从而加速训练。

### 3.5 DeepSeek-V2 的具体数值

```
DeepSeek-V2 配置：
  模型维度:         d = 5120
  注意力头数:       n_h = 128
  每头维度:         d_h = 128
  MHA KV 维度:      n_h × d_h = 128 × 128 = 16384

  MLA KV 压缩维度:  d_c = 512         ← 只有 MHA 的 1/32
  MLA Q 压缩维度:   d_c' = 1536

每层每 token 的 KV Cache 大小（FP16）：
  MHA:  2 × 16384 × 2 bytes = 65536 bytes = 64 KB
  GQA:  以 g=8 组为例, 2 × 8 × 128 × 2 = 4096 bytes = 4 KB
  MLA:  512 × 2 bytes = 1024 bytes = 1 KB        ← 仅为 MHA 的 1/64
```

从上面的数字可以看出：**MLA 将 KV Cache 压缩到了 MHA 的 1/64，甚至比 GQA（g=8）还小 4 倍**，同时由于保留了完整的 128 个头，注意力的表达能力没有损失。

---

## 四、与 RoPE 的兼容问题

### 4.1 问题的来源

RoPE（Rotary Position Embedding）是当前主流 LLM 使用的位置编码方式，它的工作机制是：

```
在注意力计算之前，对 Q 和 K 施加旋转变换：
  Q_rope = RoPE(Q, pos)
  K_rope = RoPE(K, pos)

旋转后，Q_i · K_j 的内积中自然包含了相对位置 (i - j) 的信息。
```

RoPE 的关键特性：**旋转变换必须作用在完整的、维度对齐的 Q/K 向量上**，且需要在缓存之前完成（因为每个 token 的位置是固定的，缓存之后就不再改变）。

MLA 的 KV Cache 缓存的是**解压之前的 $c_{KV}$**，而不是已经计算好的 K。这带来了矛盾：

```
标准流程: x → K → RoPE(K) → 缓存
MLA 流程: x → c_KV → 缓存 → (推理时) c_KV → K → RoPE(K)
```

如果每次推理都要从 $c_{KV}$ 解压出完整的 K 再做 RoPE，那么推理时的计算量和显存访问量都会变大，MLA 节省显存的优势部分被抵消。

### 4.2 MLA 的解法：解耦 RoPE

DeepSeek-V2 引入了**解耦 RoPE（Decoupled RoPE）** 方案：将位置信息从主 K/V 压缩流中分离出来，单独处理。

具体做法：

```
对于每个 Q，将其分成两部分：
  Q = concat(Q_nope, Q_rope)
              ↑         ↑
        不含位置信息   含 RoPE 位置编码

对于每个 K，同样分成两部分：
  K = concat(K_nope, K_rope)

其中：
  K_nope 从 c_KV 解压得到 → 不做 RoPE，缓存 c_KV 即可
  K_rope 独立计算，单独加 RoPE → 需要单独缓存
```

KV Cache 的实际内容变为：

$$\text{KV Cache} = [c_{KV},\ K_{\text{rope}}]$$

其中 $K_{\text{rope}}$ 的维度很小（DeepSeek-V2 中仅 64 维，远小于完整 K 的 16384 维），额外开销可以忽略不计。

### 4.3 解耦 RoPE 的完整注意力计算

```
推理时（对新 token t，历史长度为 L）：

1. 计算新 token 的 c_KV(t) 和 K_rope(t)，追加到 Cache
2. 从 Cache 读取所有历史的 c_KV(1:t) 和 K_rope(1:t)
3. 解压 K_nope(1:t) = c_KV(1:t) · W_UK_nope  ← 解压主 KV
4. 拼接 K(1:t) = concat(K_nope(1:t), K_rope(1:t))
5. 类似地得到 V(1:t) = c_KV(1:t) · W_UV
6. 计算注意力：Attention(Q(t), K(1:t), V(1:t))
```

这样的设计使得：
- 缓存量 ≈ $d_c + d_{\text{rope}}$ 远小于原始 $n_h \times d_h$
- 解压只需要当前推理步骤的一次矩阵乘，不额外引入历史计算

---

## 五、KV Cache 横向对比

以 DeepSeek-V2 的模型配置为基准进行数值对比：

```
基准配置：
  d_model = 5120, n_heads = 128, d_head = 128
  GQA 组数 g = 8（对标 LLaMA-3 70B 规格）
  MLA: d_c = 512, d_rope = 64
  序列长度 = 4096, 精度 FP16 (2 bytes)
```

| 方案 | 每层每 token 缓存量 | 全模型 60 层，seq=4096 | 相比 MHA |
|------|-------------------|----------------------|---------|
| MHA  | 2 × 128 × 128 = 32768 维 | 32768 × 2 × 4096 × 60 / 1e9 = **15.7 GB** | 1× |
| MQA  | 2 × 1 × 128 = 256 维   | 256 × 2 × 4096 × 60 / 1e9 ≈ **0.12 GB** | 1/128 |
| GQA (g=8) | 2 × 8 × 128 = 2048 维 | 2048 × 2 × 4096 × 60 / 1e9 ≈ **1.0 GB** | 1/16 |
| MLA  | 512 + 64 = 576 维 | 576 × 2 × 4096 × 60 / 1e9 ≈ **0.28 GB** | 1/56 |

**关键结论**：
- MLA 的 KV Cache 约为 MHA 的 1/56，比 GQA（g=8）还小约 3.6 倍
- MQA 虽然 Cache 更小，但模型质量损失大；MLA 通过低秩压缩保持了 n_heads=128 的全头表达能力
- MLA 将"头数"和"缓存量"解耦——可以拥有很多头（高表达能力），同时 Cache 很小（低显存开销）

---

## 六、训练与推理的流程差异

### 6.1 训练阶段

训练时，不使用 KV Cache（因为做的是 Teacher Forcing 全序列并行计算），整个计算图需要反向传播：

```
训练流程（单层）：
  输入 X (batch, seq, d)
      ↓
  c_KV = X · W_DKV           ← 压缩，(batch, seq, d_c)
      ↓ (分两路)
  K_nope = c_KV · W_UK       ← 解压 K 的主体部分
  V = c_KV · W_UV            ← 解压 V
      ↓
  K_rope = X · W_KR          ← 独立计算带位置的 K 分量
  Q_rope = ...               ← 同理
      ↓
  K = concat(K_nope, K_rope) ← 拼接完整 K
  Q = concat(Q_nope, Q_rope)
      ↓
  Attention(Q, K, V)         ← 标准注意力计算（全序列，有 causal mask）
      ↓
  输出 O = Attention · W_O
```

训练时需要缓存 c_KV（用于反向传播），但因为是前向传播阶段的中间变量，这只是临时显存，不是推理时的持久 KV Cache 问题。

### 6.2 推理阶段

推理时，自回归逐 token 生成，使用 KV Cache 避免重复计算：

```
Prefill 阶段（处理 prompt，长度 L）：
  1. 计算所有 prompt token 的 c_KV 和 K_rope
  2. 将 (c_KV, K_rope) 全部存入 KV Cache
  3. 用全部 K, V 计算 prompt 部分的注意力（并行）

Decode 阶段（逐 token 生成）：
  对于每个新 token t：
  1. 计算 c_KV(t) 和 K_rope(t)，追加到 KV Cache
  2. 从 Cache 读取历史 c_KV(1:t-1), K_rope(1:t-1)
  3. 解压: K_nope = c_KV · W_UK, V = c_KV · W_UV
  4. 组合完整 K, Q，计算注意力，输出下一个 token
```

**训练与推理的核心区别**：
- 训练：并行计算所有位置，c_KV 只是中间变量，内存随 seq 线性增长但可以分块
- 推理：自回归模式，历史 c_KV 需要持久缓存，但每步只新增一个 token 的 (d_c + d_rope) 维向量

### 6.3 关键工程优化：吸收矩阵乘法

推理时，解压步骤 $V = c_{KV} \cdot W_{UV}$ 之后立即与 $W_O$（输出投影）相乘：

$$O = \text{Attention} \cdot V \cdot W_O = \text{Attention} \cdot (c_{KV} \cdot W_{UV}) \cdot W_O$$

可以预先计算 $W_{UV} \cdot W_O$（离线合并），推理时直接用：

$$O = \text{Attention} \cdot c_{KV} \cdot (W_{UV} W_O)$$

这样实际上只需要一次矩阵乘，减少了推理时的计算量，代价是 $W_{UV} W_O$ 的参数量略大（但这是权重不是 Cache，常驻显存即可）。类似的技巧也适用于 K 的解压。

---

## 七、实际效果

### 7.1 DeepSeek-V2 的实验数据

DeepSeek-V2（2024 年 5 月）是首次将 MLA 大规模应用于实际 LLM 的工作，其论文报告了以下关键数据：

**模型规模**：总参数 236B，每 token 激活参数 21B，64 层，128 个注意力头

**KV Cache 压缩效果**（与同等配置的 MHA 对比）：

```
标准 MHA KV Cache（若 DeepSeek-V2 用 MHA）：
  每 token 每层 = 2 × 128 × 128 = 32768 维
  每 token 全模型 = 32768 × 64 = 2,097,152 维 ≈ 4 MB (FP16)

MLA 实际 KV Cache：
  每 token 每层 = 512 + 64 = 576 维
  每 token 全模型 = 576 × 64 = 36,864 维 ≈ 72 KB (FP16)

压缩比：4 MB / 72 KB ≈ 56.9×
```

**结论**：MLA 使得每 token 的 KV Cache 从约 4 MB 降低到 72 KB，**缩小至原来的 1/57**。

**对吞吐量的影响**：在 batch inference 场景下，KV Cache 减少直接转化为更大的可用 batch size：

```
假设 A100 80GB 显卡，分配 40GB 给 KV Cache，seq_len=4096：

使用 MHA KV Cache:  40 GB / (4 MB × 4096) ≈ 2.5 ≈ 支持 batch=2
使用 MLA KV Cache:  40 GB / (72 KB × 4096) ≈ 139 ≈ 支持 batch=139
```

吞吐量提升约 55 倍（理论上界），实际中受计算限制会有所折扣，但量级上的差距是真实存在的。

**模型质量**：DeepSeek-V2 论文显示，在相同训练数据量下，MLA 的性能与 MHA 相当甚至略优，这说明低秩压缩并没有损失有效信息。

### 7.2 DeepSeek-V3 的继承与延伸

DeepSeek-V3（2024 年 12 月）延续了 MLA 设计，参数规模提升到 671B（激活 37B），但由于 MLA 的 KV Cache 极小，单张 H800 80GB 在合理批量下仍可高效运行。

DeepSeek-V3 的 MLA 配置：

```
n_heads = 128, d_head = 128
d_c (KV 压缩维度) = 512
d_c' (Q 压缩维度) = 1536
d_rope = 64
模型层数 = 61
```

与 GQA（g=8，当前 LLaMA-3 等模型广泛使用的方案）对比：

```
GQA (g=8)  KV Cache：2 × 8 × 128 = 2048 维/token/层
MLA        KV Cache：512 + 64 = 576 维/token/层

MLA 是 GQA(g=8) 的 576/2048 ≈ 28%，即节省约 72%
```

即使与 GQA 相比，MLA 仍然节省了约 3/4 的 KV Cache 显存。

---

## 八、总结：MLA 的设计精华

```
问题：KV Cache 显存占用 = 2 × n_heads × d_head × seq_len × 层数

MHA 的困境：表达能力（n_heads）和显存（KV Cache）强绑定，想要更强的模型就需要更多显存

MQA/GQA 的思路：减少 KV 头数 → 降低显存 → 但损失表达能力

MLA 的思路：
  1. 保持 n_heads 不变（不损失表达能力）
  2. 用低秩压缩将 KV 信息编码到 d_c 维的潜在向量 c_KV
  3. 缓存 c_KV 而不是 K/V（显存大幅降低）
  4. 解耦 RoPE，单独处理位置信息（兼容旋转位置编码）
  5. 推理时按需从 c_KV 解压出 K/V（计算量小）

结果：表达能力 ≈ MHA，KV Cache ≈ MHA 的 1/57
```

**MLA 本质上是把"空间-计算"的 tradeoff 引入了注意力机制**：用少量额外的推理计算（解压 c_KV），换取大量的显存节省（不再缓存完整的 K/V）。这在 GPU 显存受限的推理场景中是非常有价值的设计。

---

**相关文档**：
- [Transformer架构详解](Transformer架构详解.md) — Attention 机制基础
- [MoE详解](MoE详解.md) — DeepSeek 的 MoE 创新
- [长上下文详解](长上下文详解.md) — KV Cache 优化的其他方法
- [推理优化详解](../推理优化/推理优化详解.md) — 推理部署中的 KV Cache 管理

[返回上级](README.md) | [返回总目录](../../README.md)

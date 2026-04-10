# Transformer 架构详解

## 一、经典 Transformer (Encoder-Decoder)

原始论文 "Attention Is All You Need" (2017) 的完整架构：

![Transformer Encoder-Decoder 架构 (Vaswani et al., 2017)](assets/transformer_architecture.png)

> 图源: *Attention Is All You Need*, Figure 1. 左侧为 Encoder（双向 Self-Attention + FFN），右侧为 Decoder（Masked Self-Attention + Cross-Attention + FFN）。

但当前主流 LLM **只用 Decoder 部分**（GPT 范式）。下面先讲清楚原始 Encoder-Decoder 的工作方式，再展开 Decoder-Only。

### 原始 Encoder-Decoder 如何工作

以机器翻译 "I love you" → "我爱你" 为例：

```
Encoder 输入:  I    love   you        ← 源语言序列
Decoder 输入:  <BOS>  我    爱         ← 目标序列右移一位 (shifted right)
Decoder 输出:    我    爱    你         ← 预测下一个 token
```

**训练时**：Decoder 输入是**目标句子右移一位**（前面加 `<BOS>`），这叫 Teacher Forcing——直接把正确答案喂给 Decoder，让它学会"给定前面的正确 token，预测下一个"。

**推理时**：自回归生成。一开始只输入 `<BOS>`，预测出"我"，再把"我"拼回输入，预测"爱"，直到输出 `<EOS>`。

### Decoder 的三个子层

```
Decoder Layer
┌──────────────────────────────────────────────┐
│  1. Masked Self-Attention                    │ ← Q, K, V 全来自 Decoder
│     (Decoder tokens 互相看，但只能看左边)      │
│                                              │
│  2. Cross-Attention                          │ ← Q 来自 Decoder, K/V 来自 Encoder
│     (Decoder 去 Encoder 那里 "查资料")        │
│                                              │
│  3. FFN                                      │
└──────────────────────────────────────────────┘
```

**Cross-Attention** 的核心：**Q 来自 Decoder，K 和 V 来自 Encoder 输出**。

```
Encoder 输出: [h_I, h_love, h_you]    ← 提供 K 和 V
Decoder 当前: [h_<BOS>, h_我]          ← 提供 Q

"我" 的 Q × "I" 的 K    → 高权重 (发现 "我" 对应 "I")
"我" 的 Q × "love" 的 K → 中权重
"我" 的 Q × "you" 的 K  → 低权重

→ 加权求和 V，得到融合了源语言信息的表示
```

直觉：Decoder 在生成每个词时，通过 Cross-Attention 去 Encoder 那里查——源句子里哪些词跟当前要生成的词最相关。

### Encoder-Decoder vs Decoder-Only

| | Encoder-Decoder (原始) | Decoder-Only (GPT) |
|---|---|---|
| 输入输出 | 两段不同的序列（源→目标） | 一段序列（prompt + 续写） |
| Decoder 输入 | 目标序列右移 | prompt 本身就在同一个序列里 |
| Cross-Attention | 需要，连接 Encoder 和 Decoder | 不需要，所有信息都在 Self-Attention 里 |
| 典型应用 | 翻译、摘要 (T5, BART) | 通用生成 (GPT, LLaMA) |

Decoder-Only 把 prompt 和生成的内容拼成一个序列，用 causal mask 的 Self-Attention 一起处理，不需要单独的 Encoder 和 Cross-Attention。

---

## 二、GPT 式 Decoder-Only 架构（当前 LLM 标准）

这是 GPT、LLaMA、Qwen、DeepSeek 等所有主流 LLM 使用的架构。

### 完整网络结构图

```
输入文本: "今天天气真好"
         │
         ▼
┌─────────────────────────────────────────────────────┐
│                    Token Embedding                   │
│  词表大小 V × 隐藏维度 d                              │
│  "今" → [0.12, -0.34, ...] (d 维向量)                │
│  "天" → [0.56, 0.78, ...] (d 维向量)                 │
│  ...                                                 │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│              Positional Encoding / RoPE              │
│  为每个 token 注入位置信息                             │
│  (现代 LLM 用 RoPE，在注意力计算时旋转 Q/K)            │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
          ┌─────────────────────────────┐
          │                             │
          │    Transformer Block × L    │ ← 重复 L 层 (如 LLaMA-7B: L=32)
          │    (以下展开单个 Block)       │
          │                             │
          │  ┌────────────────────────┐ │
          │  │     RMSNorm (Pre)      │ │ ← 层归一化 (放在注意力之前)
          │  └───────────┬────────────┘ │
          │              │              │
          │  ┌───────────▼────────────┐ │
          │  │  Masked Self-Attention │ │ ← 因果注意力 (只看前面的 token)
          │  │                        │ │
          │  │  Q = X·W_q             │ │
          │  │  K = X·W_k             │ │
          │  │  V = X·W_v             │ │
          │  │                        │ │
          │  │  Attn = Softmax(QK^T   │ │
          │  │         / √d_k + Mask) │ │
          │  │         · V            │ │
          │  │                        │ │
          │  │  Out = Attn · W_o      │ │
          │  └───────────┬────────────┘ │
          │              │              │
          │         ┌────▼────┐         │
          │         │ 残差连接 │ ← output = input + attention_output
          │         └────┬────┘         │
          │              │              │
          │  ┌───────────▼────────────┐ │
          │  │     RMSNorm (Pre)      │ │ ← 第二个层归一化
          │  └───────────┬────────────┘ │
          │              │              │
          │  ┌───────────▼────────────┐ │
          │  │      FFN (SwiGLU)      │ │ ← 前馈网络
          │  │                        │ │
          │  │  gate = X · W_gate     │ │
          │  │  up   = X · W_up       │ │
          │  │  out  = SiLU(gate)     │ │
          │  │        ⊙ up · W_down   │ │
          │  └───────────┬────────────┘ │
          │              │              │
          │         ┌────▼────┐         │
          │         │ 残差连接 │ ← output = input + ffn_output
          │         └────┬────┘         │
          │              │              │
          └──────────────┼──────────────┘
                         │
              (重复 L 次后)
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                    RMSNorm (Final)                    │
│                    最终层归一化                        │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────┐
│                 LM Head (Linear)                     │
│  隐藏维度 d → 词表大小 V                              │
│  输出每个位置上下一个 token 的概率分布                  │
│  logits = X · W_vocab^T                              │
└───────────────────────┬─────────────────────────────┘
                        │
                        ▼
                 Softmax → 概率
                 → 采样/贪心 → 下一个 token
```

---

## 三、逐组件详解

### 3.1 Token Embedding

词表由 [BPE 等 Subword 算法](/大模型/基础理论/Tokenizer详解.md#二bpe-byte-pair-encoding) 构建，详见 [Tokenizer详解](/大模型/基础理论/Tokenizer详解.md)。

```
词表 V (如 32000~150000 个 token)
                                    Embedding 矩阵
"今" → token_id: 1234  ──→  E[1234] = [0.12, -0.34, ..., 0.56]  (d 维)
"天" → token_id: 5678  ──→  E[5678] = [0.78, 0.23, ..., -0.91] (d 维)
```

- 可学习的查找表，参数量 = V × d
- 通常与 LM Head 共享权重（weight tying），节省参数

### 3.2 位置编码

Transformer 本身对输入顺序完全无感知——打乱 token 顺序，Self-Attention 的输出（在忽略位置编码时）是一样的。因此必须显式注入位置信息。主流方案有三种：

#### 3.2.1 正弦余弦位置编码（Sinusoidal PE）

原始 Transformer (Vaswani et al., 2017) 使用固定的三角函数编码，**直接加到 token embedding 上**：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

- $pos$：token 在序列中的位置（0, 1, 2, ...）
- $i$：维度索引（$0 \le i < d/2$）
- $d$：模型隐藏维度

```
位置 pos=0:  PE = [sin(0/1), cos(0/1), sin(0/100), cos(0/100), ...]
位置 pos=1:  PE = [sin(1/1), cos(1/1), sin(1/100), cos(1/100), ...]

低维度 (i小) → 频率高，变化快 → 区分相邻位置
高维度 (i大) → 频率低，变化慢 → 编码长距离位置
```

**为什么用三角函数？** 关键性质是：位置 $pos+k$ 的编码可以表示为位置 $pos$ 编码的线性变换：

$$PE_{pos+k} = M_k \cdot PE_{pos}$$

其中 $M_k$ 是一个只依赖偏移量 $k$ 的旋转矩阵。这让模型有可能学到相对位置关系。

**特点**：
- 不需要训练，参数量为零
- 理论上可以外推到任意长度（但实际效果一般）
- 现代 LLM 已不再使用，但 BERT 等早期模型广泛使用

#### 3.2.2 可学习位置嵌入（Learnable PE）

**ViT (Dosovitskiy et al., 2020)** 和 **GPT-2** 等使用的方案：为每个位置学习一个可训练的向量，同样**加到 token/patch embedding 上**：

$$x'_i = x_i + E_{pos}[i], \quad E_{pos} \in \mathbb{R}^{L_{max} \times d}$$

- $E_{pos}$ 是一个可学习的参数矩阵，每一行对应一个位置的嵌入
- $L_{max}$ 是预设的最大序列长度（ViT-B/16 中为 196 个 patch + 1 个 CLS token = 197）

```
ViT 中的流程：
  图像 224×224 → 分成 16×16 的 patch → 14×14 = 196 个 patch
  每个 patch 线性投影为 d 维向量
  前面拼一个 [CLS] token → 共 197 个 token
  加上 197 个可学习位置嵌入 → 送入 Transformer

位置嵌入矩阵 E_pos (197 × 768):
  E_pos[0]   = [0.02, -0.15, ...]   ← CLS token 的位置
  E_pos[1]   = [0.11,  0.03, ...]   ← 第 1 个 patch
  E_pos[2]   = [-0.05, 0.22, ...]   ← 第 2 个 patch
  ...
  E_pos[196] = [0.08, -0.11, ...]   ← 最后一个 patch
```

**特点**：
- 参数量 = $L_{max} \times d$（ViT-B: 197 × 768 ≈ 150K，很小）
- 训练后位置嵌入会自动学出有意义的结构（ViT 论文中可视化显示，邻近 patch 的嵌入余弦相似度高，呈现 2D 空间结构）
- **无法外推**：超出 $L_{max}$ 的位置没有对应嵌入。ViT 对不同分辨率输入需要对位置嵌入做插值
- GPT-2 的 context length = 1024 就是因为位置嵌入表只有 1024 行

#### 3.2.3 RoPE（Rotary Position Embedding）

**当前主流 LLM（LLaMA、Qwen、DeepSeek 等）标配**。核心思想：不加在 embedding 上，而是在 Attention 计算时对 Q 和 K 做旋转。

**一句话概括**：RoPE 用分块对角旋转矩阵分别旋转 Q 和 K，使得 Q·K 内积中出现只依赖相对距离的角度差，从而让 attention score 天然编码相对位置关系，V 不参与旋转。

**完整信息流**：

```
Q = xW_Q,  K = xW_K  （普通线性投影，无位置信息）
      ↓           ↓
Q' = R(m)·Q,  K' = R(n)·K  （对位置 m、n 的 token 施加旋转）
      ↓           ↓
score = Q'ᵀK' = Qᵀ R(m)ᵀR(n) K = Qᵀ R(n-m) K
                        ↑
               只依赖相对距离 (n-m)
      ↓
attention weight → 加权求和 V（V 不旋转，位置信息通过 attention weight 间接传递）
```

**基本公式**：将 Q/K 向量的每两个相邻维度视为一个二维平面，在位置 $m$ 处施加旋转：

$$f(q, m) = R_{\Theta,m} \cdot q$$

其中旋转矩阵为分块对角矩阵：

$$R_{\Theta,m} = \begin{pmatrix} \cos m\theta_0 & -\sin m\theta_0 & & \\ \sin m\theta_0 & \cos m\theta_0 & & \\ & & \cos m\theta_1 & -\sin m\theta_1 \\ & & \sin m\theta_1 & \cos m\theta_1 \\ & & & & \ddots \end{pmatrix}$$

每个 2D 子空间的频率：

$$\theta_i = 10000^{-2i/d}, \quad i = 0, 1, \ldots, d/2 - 1$$

展开来看，对第 $i$ 个维度对：

$$\begin{pmatrix} q'_{2i} \\ q'_{2i+1} \end{pmatrix} = \begin{pmatrix} \cos m\theta_i & -\sin m\theta_i \\ \sin m\theta_i & \cos m\theta_i \end{pmatrix} \begin{pmatrix} q_{2i} \\ q_{2i+1} \end{pmatrix}$$

**为什么 RoPE 能编码相对位置？** 关键在于内积的性质：

$$\langle f(q_m, m), f(k_n, n) \rangle = \langle R_{\Theta,m} q_m, R_{\Theta,n} k_n \rangle = q_m^T R_{\Theta,m}^T R_{\Theta,n} k_n = q_m^T R_{\Theta, n-m} k_n$$

旋转矩阵是正交阵，$R_m^T R_n = R_{n-m}$，因此 Q·K 的内积 **只依赖相对距离 $n-m$** ，自动实现了相对位置编码。

**关键概念辨析**：

- **位置 $m$ 是 token 在序列中的索引**（第几个 token），不是向量内部维度的索引
- **维度分组 $i$ 决定的是旋转频率** $\theta_i$：低维度组 → 高频旋转（捕捉近距离关系），高维度组 → 低频旋转（捕捉远距离关系），类似傅里叶变换中不同频率的基
- **旋转角度 = 位置 $m$ × 频率 $\theta_i$** ，每个 token 被旋转多少取决于它在序列中排第几

**为什么 V 不需要旋转？**

V 提供"这些位置的内容"，而 Q·K 决定"关注哪些位置"。位置关系已经通过 attention weight（由 Q·K 计算）传递给了 V 的加权求和，V 本身不需要位置信息。

**高效实现**（不需要构造完整矩阵，用逐元素乘法）：

```python
# 伪代码
def apply_rope(x, freqs_cos, freqs_sin):
    # x: (seq_len, d), freqs: (seq_len, d/2)
    x_even = x[..., 0::2]  # 偶数维度
    x_odd  = x[..., 1::2]  # 奇数维度

    out_even = x_even * freqs_cos - x_odd  * freqs_sin
    out_odd  = x_even * freqs_sin + x_odd  * freqs_cos

    return interleave(out_even, out_odd)  # 交错合并
```

**RoPE 的长度外推**：

RoPE 的基础频率 base=10000 决定了编码的"波长范围"。超出训练长度时注意力分数会退化，常见扩展方法：

| 方法 | 思路 | 代表模型 |
|------|------|----------|
| 位置插值 (PI) | 将位置 $m$ 缩放为 $m \cdot L_{train}/L_{target}$ | Code LLaMA |
| NTK-aware 缩放 | 增大 base（如 10000→160000），等效降低高频分量 | 多个开源模型 |
| YaRN | 对不同频率分量分别处理：低频不动，高频插值，中间过渡 | LLaMA 3, Qwen 2.5 |
| Dynamic NTK | 推理时根据实际序列长度动态调整 base | 早期方案，已被 YaRN 取代 |

#### 位置编码方案对比

| | 正弦余弦 (Sinusoidal) | 可学习 (Learnable) | **RoPE** | ALiBi |
|---|---|---|---|---|
| 注入方式 | 加到 embedding | 加到 embedding | **旋转 Q/K** | 加到 attention score |
| 参数量 | 0 | $L_{max} \times d$ | 0 | 0 |
| 位置类型 | 绝对（但含相对信息） | 绝对 | **相对** | 相对 |
| Q 有位置信息 | 间接有 | 间接有 | **有** | 无 |
| K 有位置信息 | 间接有 | 间接有 | **有** | 无 |
| V 有位置信息 | 间接有 | 间接有 | **无** | 无 |
| 长度外推 | 理论可以，实际差 | 不行，需插值 | **较好（配合 NTK/YaRN）** | 好 |
| 使用模型 | 原始 Transformer, BERT | GPT-2, ViT | **LLaMA, Qwen, DeepSeek** | BLOOM |

> **注意**：Sinusoidal/Learnable PE 加在 embedding 上，Q/K/V 都从带位置的 embedding 投影而来，因此三者都间接包含位置信息（尽管 V 携带位置信息并非必要）。RoPE 只在投影后旋转 Q 和 K，V 保持纯内容向量，设计上更优雅。ALiBi 则完全不修改任何向量，只给 attention score 加一个与距离成正比的偏置 $-m \cdot |i - j|$。

### 3.3 Masked Self-Attention（因果自注意力）

这是 Decoder-Only 与 Encoder 的核心区别：**每个 token 只能看到它前面的 token**。

```
                  Q (Query)                    因果 Mask
              今  天  天  气  真  好         今  天  天  气  真  好
         今 [ q₁                    ]   今 [ 1   0   0   0   0   0 ]
         天 [ q₂                    ]   天 [ 1   1   0   0   0   0 ]
   Q =   天 [ q₃                    ]   天 [ 1   1   1   0   0   0 ]
         气 [ q₄                    ]   气 [ 1   1   1   1   0   0 ]
         真 [ q₅                    ]   真 [ 1   1   1   1   1   0 ]
         好 [ q₆                    ]   好 [ 1   1   1   1   1   1 ]

Attention = Softmax(QK^T / √d_k + CausalMask) · V
            (CausalMask: 0处填-∞, softmax后变成0)
```

#### 为什么要除以 √d_k？

Attention 公式中的 $\frac{QK^T}{\sqrt{d_k}}$ 这个缩放因子看似简单，但至关重要。

**问题根源**：Q 和 K 是 $d_k$ 维向量，它们的点积 $q \cdot k = \sum_{i=1}^{d_k} q_i k_i$ 是 $d_k$ 个乘积项的求和。假设 $q_i$ 和 $k_i$ 各自独立、均值为 0、方差为 1，则：

$$\mathbb{E}[q \cdot k] = 0, \quad \text{Var}[q \cdot k] = d_k$$

即点积的**方差随维度 $d_k$ 线性增长**。$d_k$ 越大，点积的绝对值越大。

**为什么大点积有问题？** 因为 Softmax 对输入值的量级极其敏感：

```
d_k = 64 时:   QK^T 大概在 [-16, +16] 范围 (√64=8 量级)
d_k = 128 时:  QK^T 大概在 [-22, +22] 范围 (√128≈11 量级)

假设 Softmax 输入为 [10, 1, 1, 1]:
  → Softmax → [0.9999, 0.0000, 0.0000, 0.0000]  几乎 one-hot

除以 √d_k 后变为 [1.25, 0.125, 0.125, 0.125] (d_k=64):
  → Softmax → [0.46, 0.18, 0.18, 0.18]  分布更平滑，梯度更健康
```

当点积值很大时，Softmax 输出趋近 one-hot（一个接近 1，其余接近 0），落入**饱和区**，梯度消失。下面从 Softmax 的 Jacobian 推导具体原因：

**Softmax 的梯度公式**：设 $p_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$，其 Jacobian 为：

$$\frac{\partial p_i}{\partial z_j} = \begin{cases} p_i(1 - p_i) & \text{if } i = j \\ -p_i \cdot p_j & \text{if } i \neq j \end{cases}$$

即 $\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$，其中 $\delta_{ij}$ 是 Kronecker delta。

**关键观察**：所有梯度项都包含 $p_i$ 或 $p_j$ 作为因子。当 Softmax 饱和时：

```
饱和状态: p = [0.9999, 0.0001, 0.0001, 0.0001]  (假设 token 0 主导)

对角梯度 (i=j):
  ∂p₀/∂z₀ = p₀(1 - p₀) = 0.9999 × 0.0001 ≈ 0.0001  ← 接近 0！
  ∂p₁/∂z₁ = p₁(1 - p₁) = 0.0001 × 0.9999 ≈ 0.0001  ← 接近 0！

非对角梯度 (i≠j):
  ∂p₀/∂z₁ = -p₀ · p₁ = -0.9999 × 0.0001 ≈ -0.0001  ← 接近 0！
  ∂p₁/∂z₂ = -p₁ · p₂ = -0.0001 × 0.0001 ≈ 0          ← 更接近 0！

对比正常状态: p = [0.4, 0.3, 0.2, 0.1]
  ∂p₀/∂z₀ = 0.4 × 0.6 = 0.24       ← 健康的梯度
  ∂p₁/∂z₁ = 0.3 × 0.7 = 0.21       ← 健康的梯度
  ∂p₀/∂z₁ = -0.4 × 0.3 = -0.12     ← 健康的梯度
```

**本质**：$p_i(1-p_i)$ 的形态就是一个"钟形"——当 $p_i$ 接近 0 或 1 时值都趋近 0，只有在 $p_i = 0.5$ 时取最大值 0.25。这和 Sigmoid 的梯度消失问题是同一个数学原因（Softmax 是 Sigmoid 的多维推广）。

**饱和时梯度消失的后果**：

1. $\frac{\partial p_i}{\partial z_j} \approx 0$ → 梯度无法从 attention weight 传回到 $QK^T$ 的分数 → $W_Q$ 和 $W_K$ 几乎收不到梯度更新 → **注意力模式被"冻住"，无法学习**
2. 注意力"锁死"在某个 token 上，失去灵活分配权重的能力

**除以 $\sqrt{d_k}$ 后**，点积的方差被归一化为 1（与维度无关），Softmax 的输入保持在合理范围，梯度正常流动。

**一句话总结**：$\sqrt{d_k}$ 缩放是为了**抵消高维点积的方差膨胀**，防止 Softmax 进入饱和区导致梯度消失。这是一个与维度无关的归一化，让注意力机制在任意 $d_k$ 下都能稳定训练。

> **补充**：原始论文 (Vaswani et al., 2017) 在 3.2.1 节明确指出："We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_k}}$."

#### Multi-Head Attention (MHA)

将注意力拆分为多个"头"，每个头关注不同的信息：

```
MHA (以 d=4096, n_heads=32 为例):

输入 X (d=4096)
  │
  ├──→ W_q → Q (4096) → 拆成 32 个头 → [q₁(128), q₂(128), ..., q₃₂(128)]
  ├──→ W_k → K (4096) → 拆成 32 个头 → [k₁(128), k₂(128), ..., k₃₂(128)]
  └──→ W_v → V (4096) → 拆成 32 个头 → [v₁(128), v₂(128), ..., v₃₂(128)]
                                              │
                            每个头独立做注意力计算
                                              │
                                   拼接 32 个头的输出
                                              │
                                        W_o → 输出 (4096)
```

#### GQA (Grouped Query Attention)

多个 Q 头共享同一组 K/V，减少 KV Cache 大小。详细原理和 MHA/MQA/GQA/MLA 的完整对比见 [KV Cache 详解](../推理优化/KV_Cache详解.md)。

### 3.4 FFN (前馈网络)

#### FFN 的角色

Transformer 中 Attention 和 FFN 分工明确：

- **Attention**：token 之间的信息交互（"谁和谁相关"）
- **FFN**：每个 token **独立地** 做非线性变换（"提取和组合特征"）

可以理解为：Attention 负责 **"从哪里收集信息"**，FFN 负责 **"对收集到的信息做什么处理"**。FFN 对每个 token 位置施加完全相同的变换（共享参数），不同 token 之间不交互。

从参数量看，FFN 占了 Transformer 每层参数的约 **2/3**（经典 FFN 有 $2 \times d \times 4d = 8d^2$，而 Attention 有 $4d^2$）。有研究认为 FFN 层充当了一种 **"键值记忆"**（Geva et al., 2021）：$W_1$ 的每一行是一个 "key pattern"，$W_2$ 的对应列是存储的 "value"，升维到 4d 意味着有 4d 个记忆槽。

#### 经典 FFN

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 \cdot x + b_1) + b_2$$

```
x:  (d,)
W₁: d → 4d   (升维，扩展表达能力)
W₂: 4d → d   (降维，回到原始维度)
```

**为什么要先升维再降维？** 升维到 4d 是为了在更高维空间做非线性变换——低维空间中线性不可分的特征，在高维空间中可能变得可分。这和 SVM 的核函数思想类似。降维回 d 是为了保持残差连接的维度一致。

#### SwiGLU FFN（当前主流）

LLaMA、Qwen、DeepSeek 等现代 LLM 都使用 SwiGLU 替代经典 FFN：

$$\text{SwiGLU}(x) = (\text{SiLU}(x W_{gate}) \odot x W_{up}) \cdot W_{down}$$

```
x:      (d,)
W_gate: d → d_ff    (门控分支：决定"让多少信息通过")
W_up:   d → d_ff    (值分支：提供实际内容)
W_down: d_ff → d    (降维回 d)

⊙ 是逐元素乘法（Hadamard product）
SiLU(x) = x · σ(x)  (Sigmoid Linear Unit, 又叫 Swish)
```

**SwiGLU 的关键设计——门控机制**：

```
        x
       / \
      /   \
   W_gate  W_up
     |       |
   SiLU     (线性)
     |       |
     └── ⊙ ──┘    ← 门控：SiLU 分支控制 up 分支的信息流
         |
       W_down
         |
       output
```

- `SiLU(x · W_gate)` 输出的值在 $(-0.28, +\infty)$ 范围，接近 0 时相当于 "关门"，较大时 "开门"
- 逐元素乘法让网络学会 **选择性地激活** 不同维度的特征，比单纯 ReLU 更细粒度

**为什么 SwiGLU 更好？**

- 门控机制让网络对特征有更精细的控制（相比 ReLU 的硬截断 0/通过）
- Shazeer (2020) 的实验表明，GLU 变体在同等参数量下一致优于 ReLU/GELU FFN
- SiLU 处处可导（不像 ReLU 在 0 处不可导），有利于优化

> **参数量对齐**：SwiGLU 有 3 个权重矩阵（$3 \times d \times d_{ff}$），经典 FFN 有 2 个（$2 \times d \times 4d$）。为保持总参数量相近，SwiGLU 的 $d_{ff}$ 通常设为 $\frac{8d}{3}$（而非 4d），使得 $3 \times d \times \frac{8d}{3} = 8d^2 \approx 2 \times d \times 4d$。

#### 激活函数演变

| 激活函数 | 公式 | 特点 |
|---|---|---|
| ReLU | $\max(0, x)$ | 简单高效，但有 "死神经元" 问题 |
| GELU | $x \cdot \Phi(x)$ | BERT/GPT-2 使用，平滑版 ReLU |
| SiLU/Swish | $x \cdot \sigma(x)$ | 处处可导，非单调，表达力更强 |
| SwiGLU | $\text{SiLU}(xW_g) \odot xW_{up}$ | 门控 + SiLU，当前 LLM 标配 |

### 3.5 归一化

#### LayerNorm

LayerNorm（Ba et al., 2016）对每个 token 的隐藏向量做两步：先中心化（减均值），再缩放（除标准差）：

$$\text{LayerNorm}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} \cdot \gamma + \beta$$

其中 $\mu = \frac{1}{d}\sum_{i=1}^{d} x_i$，$\sigma^2 = \frac{1}{d}\sum_{i=1}^{d}(x_i - \mu)^2$，$\gamma$（缩放）和 $\beta$（偏移）是可学习参数。

**为什么需要归一化？** 深层网络中，每层输出的分布会随训练不断漂移（internal covariate shift），导致后续层的输入不稳定，训练困难。归一化将每层输出拉回到均值 0、方差 1 附近，稳定训练。

**LayerNorm vs BatchNorm** ：BatchNorm 对 batch 维度求均值/方差，依赖 batch size，在序列长度可变的 NLP 任务中不适用。LayerNorm 对特征维度求均值/方差，与 batch size 和序列长度无关，天然适合 Transformer。

#### RMSNorm（现代 LLM 标配）

RMSNorm（Zhang & Sennrich, 2019）去掉了均值中心化和偏置 $\beta$，只用 RMS（均方根）做缩放：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^{d}x_i^2 + \epsilon}} \cdot \gamma$$

**为什么去掉均值也行？** 原论文的实验发现：LayerNorm 的效果主要来自 **缩放不变性** （re-scaling），而不是中心化（re-centering）。去掉均值后模型性能基本不变，但省了一次 reduce 操作——在 GPU 上 reduce 是显存带宽的瓶颈，省一次就是实打实的加速。

#### LayerNorm vs RMSNorm 对比

| | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | 是 | **否** |
| 除标准差/RMS | 除标准差 | 除 RMS |
| 可学习偏置 $\beta$ | 有 | **无** |
| 计算量 | 2 次 reduce（均值 + 方差） | **1 次 reduce（平方均值）** |
| 效果 | 经典有效 | 实践中几乎无损，速度更快 |
| 使用模型 | BERT, GPT-2, 原始 Transformer | LLaMA, Qwen, DeepSeek 等现代 LLM |

#### Pre-Norm vs Post-Norm

这是归一化 **放在哪里** 的设计选择，与用哪种归一化正交：

```
Post-Norm (原始 Transformer):        Pre-Norm (现代 LLM):
x → Attention → Add → Norm          x → Norm → Attention → Add
                                          ↑ residual 直接连 ↑
```

- **Post-Norm** ：残差连接后再归一化，梯度需要穿过 Norm 层，深层网络容易梯度不稳定，需要 learning rate warmup
- **Pre-Norm** ：先归一化再进子层，残差是"干净的"直连，梯度可以畅通无阻地反传，训练更稳定

现代 LLM 标配是 **Pre-RMSNorm** ：既用 RMSNorm 省计算，又用 Pre-Norm 稳训练。

### 3.6 残差连接

每个子层（Attention、FFN）都有残差连接：

```
output = input + SubLayer(Norm(input))   ← Pre-Norm 残差
```

残差连接的重要性：
- 缓解梯度消失，让深层网络（32~80 层）可训练
- 保证信息可以跳过某些层直接传递

### 3.7 Gradient Clipping（梯度裁剪）

深层 Transformer 训练中，梯度可能在某些 step 突然变得极大（**梯度爆炸**），尤其是：
- 训练早期参数还不稳定时
- 遇到异常数据（如特别长或特别罕见的序列）时
- 学习率设置偏大时

梯度爆炸会导致参数更新幅度过大，模型 loss 突然飙升甚至 NaN。**Gradient Clipping 是防止这种情况的安全网**。

#### 两种常见方式

**1. 按范数裁剪（Clip by Global Norm）—— 最常用**

计算所有参数梯度拼接后的全局 L2 范数，如果超过阈值就等比缩小：

$$\text{if } \|g\|_2 > \text{max\_norm}: \quad g \leftarrow g \cdot \frac{\text{max\_norm}}{\|g\|_2}$$

```python
# PyTorch 实现 —— 几乎所有 LLM 训练都用这行
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

关键特性：
- **保持梯度方向不变**，只缩放大小——不会扭曲优化方向
- 对所有参数的梯度**等比例缩放**，保持各参数间梯度的相对比例
- 正常训练时梯度范数一般在阈值以内，裁剪不会触发，**不影响正常优化**

```
正常 step:  ‖g‖ = 0.8,  max_norm = 1.0  → 不裁剪，原样使用
异常 step:  ‖g‖ = 50.0, max_norm = 1.0  → 缩放为 g × (1.0/50.0)，方向不变
```

**2. 按值裁剪（Clip by Value）—— 较少使用**

将每个梯度元素独立截断到 $[-\text{clip\_value}, +\text{clip\_value}]$ 范围：

$$g_i \leftarrow \text{clamp}(g_i, -c, +c)$$

缺点：会改变梯度方向（不同维度裁剪程度不同），所以在 LLM 训练中几乎不用。

#### 在 LLM 训练中的使用

几乎所有主流 LLM 都使用 **clip by global norm**，阈值通常为 **1.0**：

| 模型 | Gradient Clip | 备注 |
|------|--------------|------|
| GPT-3 | 1.0 | global norm |
| LLaMA | 1.0 | global norm |
| LLaMA-2 | 1.0 | global norm |
| Qwen | 1.0 | global norm |
| DeepSeek-V2/V3 | 1.0 | global norm |

#### 完整训练循环中的位置

```python
for batch in dataloader:
    optimizer.zero_grad()
    
    loss = model(batch)                    # 前向传播
    loss.backward()                        # 反向传播，计算梯度
    
    # ★ 梯度裁剪 —— 在 backward() 之后、optimizer.step() 之前
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()                       # 用（可能被裁剪的）梯度更新参数
    scheduler.step()                       # 更新学习率
```

注意：梯度裁剪必须在 `backward()` 之后（梯度已计算出来）、`step()` 之前（参数还没更新）。

#### 与其他稳定训练手段的关系

Gradient Clipping 是训练稳定性的**最后一道防线**，与其他机制互补：

| 手段 | 解决什么问题 | 作用时机 |
|------|------------|---------|
| RMSNorm / LayerNorm | 每层输出分布漂移 | 前向传播 |
| 残差连接 | 梯度消失 | 前向 + 反向 |
| $\sqrt{d_k}$ 缩放 | Softmax 饱和 → 梯度消失 | 前向传播 |
| Learning Rate Warmup | 训练初期参数不稳定 | 优化器 |
| **Gradient Clipping** | **梯度爆炸（异常大梯度）** | **反向传播后** |

直觉：RMSNorm 和残差连接让梯度"大部分时候"正常流动，Gradient Clipping 处理那些"偶尔出现"的异常大梯度——**常规手段管常态，Clipping 管极端情况**。

---

## 四、主流 LLM 的具体参数

| 模型 | 层数 L | 隐藏维度 d | 注意力头数 | KV 头数 | FFN 维度 | 词表 V | 总参数 |
|------|--------|-----------|----------|---------|---------|--------|--------|
| LLaMA-7B | 32 | 4096 | 32 | 32 (MHA) | 11008 | 32000 | 6.7B |
| LLaMA-2-13B | 40 | 5120 | 40 | 40 (MHA) | 13824 | 32000 | 13B |
| LLaMA-3-8B | 32 | 4096 | 32 | 8 (GQA) | 14336 | 128256 | 8B |
| LLaMA-3-70B | 80 | 8192 | 64 | 8 (GQA) | 28672 | 128256 | 70B |
| Qwen-2.5-7B | 28 | 3584 | 28 | 4 (GQA) | 18944 | 152064 | 7.6B |
| DeepSeek-V3 | 61 | 7168 | 128 | MLA | MoE | 129280 | 671B |

### 参数量粗算

#### 逐组件拆解

对于一个 L 层、隐藏维度 d、FFN 中间维度 $d_{ff}$ 的 Dense 模型（忽略 bias）：

**1. Embedding 层**

```
Token Embedding:  V × d       (词表大小 × 隐藏维度)
```

> 注意：如果 LM Head 与 Embedding 共享权重（weight tying），则不额外计算 LM Head 参数。LLaMA 系列 **不共享**，所以 LM Head 单独算一份 $V \times d$。

**2. 每层 Attention**

```
W_Q: d × d       ┐
W_K: d × d       │  4 个矩阵，共 4d²
W_V: d × d       │
W_O: d × d       ┘
```

如果使用 GQA（K/V head 数为 $n_{kv}$，Q head 数为 $n_h$，每头维度 $d_h = d / n_h$）：

```
W_Q: d × d               = d²
W_K: d × (n_kv × d_h)    = d × n_kv × d_h
W_V: d × (n_kv × d_h)    = d × n_kv × d_h
W_O: d × d               = d²

GQA 每层 Attention = 2d² + 2d × n_kv × d_h
```

标准 MHA（$n_{kv} = n_h$）就退化为 $4d^2$。

**3. 每层 FFN**

经典 FFN（2 个矩阵）：

```
W₁: d × 4d    ┐
W₂: 4d × d    ┘  共 8d²
```

SwiGLU FFN（3 个矩阵，$d_{ff} \approx 8d/3$）：

```
W_gate: d × d_ff    ┐
W_up:   d × d_ff    │  共 3 × d × d_ff ≈ 8d²
W_down: d_ff × d    ┘
```

> 为什么 $d_{ff} = 8d/3$？因为 SwiGLU 有 3 个矩阵，为保持与经典 FFN 的 $8d^2$ 参数量对齐：$3 \times d \times d_{ff} = 8d^2 \Rightarrow d_{ff} = 8d/3$。实际实现中会取最近的 128 的倍数（如 LLaMA-7B 的 11008）。

**4. 每层 RMSNorm**

```
每个 RMSNorm: d 个参数 (只有 γ)
每层 2 个 RMSNorm (Attention 前 + FFN 前): 2d
最终输出还有 1 个 RMSNorm: d
```

归一化参数极少，通常忽略不计。

#### 汇总公式

```
每层参数 ≈ 4d² + 3d·d_ff  (Attention + SwiGLU FFN)
        ≈ 4d² + 8d²       (当 d_ff = 8d/3)
        ≈ 12d²

总参数 ≈ L × 12d² + V × d  (+ 若不共享 LM Head: 再加 V × d)
```

#### 实例验算

**LLaMA-7B**（MHA, L=32, d=4096, $d_{ff}$=11008, V=32000, 不共享 LM Head）：

```
每层 Attention:  4 × 4096²           = 67,108,864
每层 FFN:        3 × 4096 × 11008    = 135,266,304
每层合计:        67M + 135M           ≈ 202M

32 层合计:       32 × 202M            ≈ 6.47B
Embedding:       32000 × 4096         ≈ 0.13B
LM Head:         32000 × 4096         ≈ 0.13B (不共享)
──────────────────────────────────────
总计:            6.47 + 0.13 + 0.13   ≈ 6.7B ✓
```

**LLaMA-3-8B**（GQA-8, L=32, d=4096, $n_h$=32, $n_{kv}$=8, $d_h$=128, $d_{ff}$=14336, V=128256）：

```
每层 Attention:
  W_Q: 4096 × 4096           = 16,777,216
  W_K: 4096 × (8 × 128)      = 4,194,304
  W_V: 4096 × (8 × 128)      = 4,194,304
  W_O: 4096 × 4096           = 16,777,216
  合计:                        ≈ 41.9M  (注意比 MHA 的 67M 少了不少)

每层 FFN:  3 × 4096 × 14336  = 176,160,768 ≈ 176M

32 层合计:  32 × (41.9M + 176M)  ≈ 6.97B
Embedding:  128256 × 4096        ≈ 0.53B
LM Head:    128256 × 4096        ≈ 0.53B
──────────────────────────────────────
总计:       6.97 + 0.53 + 0.53   ≈ 8.0B ✓
```

> **观察**：LLaMA-3-8B 相比 LLaMA-7B，GQA 省下的 Attention 参数被更大的 FFN（$d_{ff}$ 从 11008→14336）和更大的词表（32K→128K）吃掉了，总参数反而更多。

---

## 五、推理过程：Prefill + Decode

LLM 推理分两个阶段：

```
用户输入: "今天天气怎么样？"

阶段 1: Prefill (预填充)
  一次性处理所有输入 token
  [今, 天, 天, 气, 怎, 么, 样, ？] → 并行计算 → 缓存所有层的 K, V

阶段 2: Decode (逐 token 生成)
  Step 1: 基于 KV Cache + 上一个 token → 生成 "今"
  Step 2: 更新 KV Cache → 生成 "天"
  Step 3: 更新 KV Cache → 生成 "的"
  ...
  每步只计算 1 个新 token 的注意力（利用缓存的 KV）
```

> KV Cache 是推理加速的关键：避免重复计算历史 token 的 K 和 V。代价是显存占用随序列长度线性增长。

---

## 六、训练目标：Next Token Prediction

LLM 的训练目标非常简单——**预测下一个 token**。

### 交叉熵损失

给定一个训练序列 $[x_1, x_2, \ldots, x_T]$，模型在每个位置 $t$ 输出一个词表大小的概率分布 $P(x | x_{\lt t})$，训练目标是最大化正确 token 的概率，等价于最小化交叉熵损失：

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_1, x_2, \ldots, x_{t-1})$$

### 具体计算流程

以训练序列 "今天天气真好" 为例：

```
输入:    [<BOS>, 今, 天, 天, 气, 真]     ← 右移一位
目标:    [今,    天, 天, 气, 真, 好]     ← 原始序列

位置 1: 输入 <BOS>        → 模型输出概率分布 → 正确答案是 "今" → -log P(今)
位置 2: 输入 <BOS>,今     → 模型输出概率分布 → 正确答案是 "天" → -log P(天)
位置 3: 输入 <BOS>,今,天  → 模型输出概率分布 → 正确答案是 "天" → -log P(天)
...

Loss = 所有位置的 -log P 取平均
```

关键点：

- **因果 Mask 保证每个位置只能看到前面的 token**，所以虽然整个序列一次性并行输入，但每个位置的预测都是"给定前文→预测下一个"
- **训练时并行，推理时自回归**：训练时所有位置的 loss 同时算（高效），推理时必须逐个生成（因为后面的 token 还不存在）
- 这就是所谓的 **Teacher Forcing**：训练时用真实 token 作为输入，而不是模型自己的预测

### 从 logits 到 loss

```
模型最后一层输出 hidden state h (d 维)
    → LM Head: logits = h · W_vocab^T    (d 维 → V 维, 每个词一个分数)
    → Softmax: P(x) = exp(logit_x) / Σ exp(logit_j)
    → 取正确 token 的概率: P(x_t)
    → Loss = -log P(x_t)
```

实际实现中，Softmax + (-log) 合并为 `CrossEntropyLoss`（数值更稳定）：

```python
import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
# logits: (batch_size, seq_len, vocab_size)
# labels: (batch_size, seq_len) — 每个位置的正确 token id
loss = loss_fn(logits.view(-1, vocab_size), labels.view(-1))
```

### Perplexity（困惑度）

Perplexity 是交叉熵损失的指数形式，是评估语言模型最常用的指标：

$$\text{PPL} = e^{\mathcal{L}} = \exp\left(-\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{\lt t})\right)$$

直觉：**模型在每个位置平均在多少个 token 之间犹豫**。

- PPL = 1：完美预测，每个 token 概率都是 1
- PPL = 100：平均在 100 个 token 之间犹豫
- PPL = V（词表大小）：完全随机猜

PPL 越低越好。当前主流 LLM 在通用语料上 PPL 通常在 5~15 左右。

### 为什么是 Next Token Prediction？

这个看似简单的目标为什么能训练出强大的 LLM？

- **压缩即智能**：要准确预测下一个 token，模型必须理解语法、语义、逻辑、世界知识——所有这些都隐含在"正确预测下一个词"这个目标中
- **数据利用率高**：一个长度为 T 的序列提供 T 个训练样本（每个位置都是一个预测任务），不浪费任何数据
- **Scaling Law**：当模型参数、数据量、计算量按比例增大时，loss 会平滑下降（Kaplan et al., 2020），这种可预测的扩展性是 LLM 成功的基础

---

## 七、Temperature 与采样策略

LM Head 输出的是 logits（未归一化的原始分数），需要转换为概率再选 token。

**Temperature 发生在推理的最后一步**——模型所有 Transformer 层计算完毕、LM Head 输出 logits 之后，选下一个 token 之前：

```
输入 tokens → Transformer 所有层 → 最后一层 hidden state
    → LM Head (线性层) → logits (词表大小的向量，每个词一个分数)
    → 除以 Temperature T
    → Softmax → 概率分布
    → 选一个 token（argmax 或采样）
```

举个具体例子，假设词表只有 4 个词：

```
LM Head 输出 logits:  [猫: 5.0,  狗: 3.0,  鱼: 1.0,  鸟: 0.5]

T=1.0 (原始):  Softmax → [猫: 0.72, 狗: 0.18, 鱼: 0.06, 鸟: 0.04]
T=0.1 (锐化):  logits/0.1 = [50, 30, 10, 5] → Softmax → [猫: ~1.0, 其他: ~0]
T=2.0 (平滑):  logits/2.0 = [2.5, 1.5, 0.5, 0.25] → Softmax → [猫: 0.44, 狗: 0.24, ...]
```

- **T 小** → logits 被放大 → Softmax 后分布更尖 → 几乎确定选最高分的词
- **T 大** → logits 被压缩 → Softmax 后分布更平 → 各词被选中的概率更接近

### Temperature

在 softmax 前将 logits 除以温度参数 $T$：

$$P(i) = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$

| Temperature | 效果 | 适用场景 |
|---|---|---|
| T → 0 | 分布极度尖锐，几乎等于 argmax | 代码生成、数学推理 |
| T = 1.0 | 原始分布 | 默认 |
| T > 1.0 | 分布趋于均匀，更随机 | 创意写作、头脑风暴 |

### argmax vs 采样

**Temperature 只在采样 (sampling) 时有意义**：

```python
logits = model(input_ids)          # 模型输出原始分数
logits = logits / temperature      # 除以温度
probs = softmax(logits)            # 转概率分布

# 贪心 (argmax) —— 永远选概率最大的，temperature 没意义
next_token = argmax(probs)

# 采样 (sample) —— 按概率随机抽，temperature 控制随机性
next_token = sample(probs)
```

| 策略 | T=0.1 | T=1.0 | T=2.0 |
|---|---|---|---|
| argmax | 词A | 词A | 词A（永远一样）|
| sample | 99%抽到词A | 59%词A, 22%词B | 42%词A, 33%词B |

API 里 `temperature=0` 实际就等于关闭采样、退化为 greedy decoding。

### 其他采样控制参数

- **Top-k**：只保留概率最高的 k 个 token，其余概率置零后重新归一化
- **Top-p (nucleus)**：按概率从高到低累加，保留到累积概率 ≥ p 的 token
- **Repetition Penalty**：对已生成的 token 降低 logits，减少重复

### Temperature 在其他场景的应用

- **知识蒸馏** (Hinton 2015)：高温度（T=3~20）产生 soft label，让学生模型学到类间相似度的"暗知识"
- **对比学习** (CLIP/SimCLR)：InfoNCE loss 中的 τ 控制正负样本区分度，τ 越小对困难负样本惩罚越重

---

**相关文档**：
- [MoE详解](MoE详解.md) — FFN 层替换为稀疏专家
- [对比学习与CLIP详解](../../视觉/对比学习与CLIP详解.md) — ViT 作为视觉编码器

[返回上级](README.md) | [返回总目录](../../README.md)

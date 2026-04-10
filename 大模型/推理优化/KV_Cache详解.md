# KV Cache 详解

KV Cache 是大模型推理中最核心的显存组件。本文从原理、显存分析、分页管理到服务系统，全面介绍 KV Cache 相关的知识。

---

## 一、KV Cache 基础

### 1.1 为什么需要 KV Cache

自回归生成时，每生成一个新 token 都要对所有历史 token 做注意力。如果不缓存，第 $t$ 步要重新算前 $t-1$ 个 token 的 K 和 V，复杂度是 $O(t^2)$。

**KV Cache**：把每一层已经算好的 K、V 存起来，新 token 只需算自己的 Q/K/V，然后跟缓存拼接。

```
Step 1: 输入 [A]       → 计算 K₁,V₁ → 缓存 [K₁,V₁]
Step 2: 输入 [B]       → 计算 K₂,V₂ → 缓存 [K₁,V₁, K₂,V₂]，Q₂ attend to [K₁,K₂]
Step 3: 输入 [C]       → 计算 K₃,V₃ → 缓存 [K₁,V₁, K₂,K₃, V₁,V₂,V₃]
...
每步只算 1 个 token 的 QKV，但要读取所有缓存的 KV
```

### 1.2 KV Cache 显存计算（面试高频）

```
KV Cache 显存 = 2 × L × n_kv × d_head × seq_len × bytes_per_param

其中:
  2        = K 和 V 各一份
  L        = 层数
  n_kv     = KV 头数（MHA 等于 n_heads，GQA 则更少）
  d_head   = 每个头的维度（通常 128）
  seq_len  = 序列长度
  bytes    = FP16 = 2 bytes, FP32 = 4 bytes
```

**示例：多个主流模型对比（seq_len=8192，FP16）**

| 模型 | L | n_kv | d_head | 单请求 KV Cache | batch=32 |
|---|---|---|---|---|---|
| LLaMA-2-7B (MHA) | 32 | 32 | 128 | 4 GB | 128 GB |
| LLaMA-3-8B (GQA) | 32 | 8 | 128 | 1 GB | 32 GB |
| LLaMA-3-70B (GQA) | 80 | 8 | 128 | 2.5 GB | 80 GB |
| Qwen-2.5-7B (GQA) | 28 | 4 | 128 | 0.44 GB | 14 GB |

```
以 LLaMA-3-8B 为例:
KV Cache = 2 × 32 × 8 × 128 × 8192 × 2 bytes
         = 1,073,741,824 bytes ≈ 1 GB  (单个请求！)

模型参数本身: 8B × 2 bytes = 16 GB (FP16)
batch=32 时: KV Cache = 32 GB > 模型本身 16 GB
```

**GQA 的价值**：LLaMA-2-7B（MHA, n_kv=32）→ 4 GB/请求；LLaMA-3-8B（GQA, n_kv=8）→ 1 GB/请求。**节省 4 倍 KV Cache 显存**。

### 1.3 Decode 阶段的 Arithmetic Intensity 分析

理解为什么 Decode 是"访存密集"而非"计算密集"：

```
Arithmetic Intensity (算术强度) = FLOPs / Bytes_transferred

Decode 一步的计算 (简化为单层线性层 W ∈ ℝ^{d×d}, 输入 x ∈ ℝ^{1×d}):
  FLOPs:  2d²     (矩阵乘)
  Bytes:  2d² × 2 (读取权重 W, FP16) + 2d (读 x) + 2d (写 y) ≈ 4d²
  AI = 2d² / 4d² = 0.5 FLOPs/Byte

A100 的算力: 312 TFLOPS (BF16)
A100 的带宽: 2 TB/s
A100 的 "屋脊" 转折点: 312T / 2T = 156 FLOPs/Byte

Decode 的 AI = 0.5 << 156 → 严重访存瓶颈！
GPU 算力利用率 ≈ 0.5/156 ≈ 0.3%
```

**这就是为什么 Decode 慢**：每生成一个 token 都要把 16GB 的模型权重从 HBM 读一遍，但只做极少计算。增大 batch_size 可以分摊权重读取的开销。

### 1.4 Prefill vs Decode 的性能特征

| | Prefill（预填充） | Decode（逐 token 生成） |
|---|---|---|
| 输入 | 整个 prompt（N 个 token） | 1 个 token |
| 计算模式 | 大矩阵乘，**计算密集** | 小向量 × 大矩阵，**访存密集** |
| 瓶颈 | GPU 算力 (FLOPS) | 显存带宽 (Memory Bandwidth) |
| 优化方向 | FlashAttention、Tensor 并行 | KV Cache 优化、量化、投机解码 |

> Decode 阶段每生成一个 token 都要读取整个 KV Cache + 模型权重，但计算量很小。这就是为什么 Decode 阶段 GPU 利用率低——带宽被打满，算力闲着。

---

## 二、PagedAttention 与 vLLM

### 2.1 KV Cache 的显存碎片问题

传统 KV Cache 为每个请求**预分配**最大长度的连续显存：

```
请求 A: 实际长度 100，预分配 2048 → 浪费 95% 显存
请求 B: 实际长度 500，预分配 2048 → 浪费 75%
请求 C: 想分配但显存不够了       → 被拒绝

实际显存利用率可能只有 20-30%
```

### 2.2 PagedAttention 的解决方案

借鉴操作系统的**虚拟内存分页**：

```
物理显存被划分为固定大小的 Block（如 16 tokens）

请求 A (100 tokens):
  逻辑 KV: [0..99]
  物理 Block: Block#3 → Block#7 → Block#12 → ... → Block#45
  (不需要连续！通过 Block Table 映射)

请求 B (500 tokens):
  逻辑 KV: [0..499]
  物理 Block: Block#1 → Block#5 → Block#8 → ...

┌─────────────┐
│ Block Table │  请求A: [3, 7, 12, ..., 45]
│  (页表)     │  请求B: [1, 5, 8, ..., 99]
└─────────────┘
```

**优势**：
- 按需分配，几乎无浪费（碎片 < 4%）
- 支持 **Copy-on-Write**：beam search 中多个 beam 共享 KV 前缀
- 动态增长：生成更多 token 时分配新 block

### 2.3 vLLM 系统架构

```
                    ┌──────────────┐
                    │   Scheduler  │ ← 调度：哪些请求可以进 batch
                    └──────┬───────┘
                           │
              ┌────────────▼────────────┐
              │    Continuous Batching   │ ← 不等最慢的请求，完成一个立即加入新的
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    PagedAttention       │ ← 分页 KV Cache 管理
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │    Model Execution      │ ← 实际模型推理 (CUDA kernels)
              └─────────────────────────┘
```

**Continuous Batching（连续批处理）**：

```
Static Batching (传统):
  Batch = [请求A(200 tokens), 请求B(50 tokens), 请求C(150 tokens)]

  iter 1:  A□□□  B□□□  C□□□  ← 三个都在生成
  iter 50: A□□□  B[完成]C□□□  ← B 完成了，但 GPU 必须等 A 和 C
  iter 51: A□□□  [空转] C□□□  ← B 的 GPU 资源浪费！
  ...
  iter 150:A□□□  [空转] C[完成]
  iter 200:A[完成][空转] [空转] ← 最后只有 A 在算，2/3 资源浪费

Continuous Batching:
  iter 1:  A□□□  B□□□  C□□□
  iter 50: A□□□  B[完成]C□□□
  iter 51: A□□□  D□□□  C□□□  ← B 完成立即替换为新请求 D！
  iter 150:A□□□  D□□□  C[完成]
  iter 151:A□□□  D□□□  E□□□  ← C 完成立即替换为 E
  ...
  → GPU 始终满载运行
```

- Static Batching 的吞吐量受限于**最慢的请求**
- Continuous Batching 在 iteration 级别调度，吞吐量提升 **2-10x**
- 进一步优化：**Chunked Prefill** —— 长 prompt 的 prefill 也分块穿插在 decode 中，避免长 prefill 阻塞所有 decode 请求

### 2.4 Prefix Caching

多个请求共享相同的 system prompt → 只算一次 KV Cache，后续请求复用。

```
请求 1: [system prompt] + [用户问题 A]  → 计算完整 KV Cache
请求 2: [system prompt] + [用户问题 B]  → 复用 system prompt 的 KV Cache，只算问题 B 部分
请求 3: [system prompt] + [用户问题 C]  → 同样复用

节省: system prompt 部分的 Prefill 计算 + KV Cache 显存
```

---

## 三、面试快速回答

> **Q: KV Cache 占多少显存？**
> A: `2 × 层数 × KV头数 × 头维度 × 序列长度 × 字节数`。7B 模型 FP16 下单请求 8K 长度约 1GB，batch=32 就是 32GB，可能超过模型本身。GQA 可以减少到 1/4。
>
> **Q: PagedAttention 解决什么问题？**
> A: KV Cache 的显存碎片。传统方式预分配连续显存浪费 70-80%，PagedAttention 借鉴 OS 虚拟内存分页，按需分配非连续 block，碎片降到 <4%，还支持 Copy-on-Write。
>
> **Q: Continuous Batching 比 Static Batching 好在哪？**
> A: Static Batching 整个 batch 等最慢的请求，GPU 资源浪费严重。Continuous Batching 在 iteration 级别调度，完成一个立即替换新请求，GPU 始终满载，吞吐量提升 2-10x。

---

**相关文档**：
- [推理优化详解](推理优化详解.md) — FlashAttention、量化、投机解码等其他优化技术
- [Transformer架构详解](../基础理论/Transformer架构详解.md) — KV Cache 在注意力机制中的角色
- [MLA详解](../基础理论/MLA详解.md) — Multi-head Latent Attention 对 KV Cache 的压缩

[返回上级](../README.md) | [返回总目录](../../README.md)

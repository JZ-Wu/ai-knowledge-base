# InternVL 系列技术解读

InternVL 是上海 AI Lab（书生团队）开发的多模态视觉语言模型系列，从 2024 年起以开源姿态在多个多模态基准上持续挑战闭源旗舰。本文梳理 InternVL 全系列的演进脉络，**重点解读最新的 InternVL 3.5 版本**（2025.8），涵盖其架构创新、训练策略与性能表现。

> 更新时间：2026-04-13

---

## 一、系列演进概览

### 1.1 版本脉络

```
InternVL 1.0 (2023.12)    → 初版，验证大视觉编码器可行性
InternVL 1.5 (2024.4)     → 动态高分辨率策略，OCR/文档能力飞跃
InternVL 2.0 (2024.7)     → 全系列规模化（2B~76B），多图+视频支持
InternVL 2.5 (2024.12)    → 精细化改进，静态图像任务追平 GPT-4o
InternVL 3.0 (2025.4)     → 原生多模态预训练，引入 RL 对齐
InternVL 3.5 (2025.8)     → Cascade RL + 动态压缩，整体追平 GPT-5 ★
```

### 1.2 旗舰模型 Benchmark 演进

| 版本 | 最大参数 | MMMU | MathVista | OCRBench | 对标模型 |
|---|---|---|---|---|---|
| InternVL 1.5 | 26B | 46.8 | 53.5 | 724 | GPT-4V |
| InternVL 2.0 | 76B | 55.2 | 67.3 | 794 | GPT-4V |
| InternVL 2.5 | 78B | 70.1 | 76.6 | 822 | GPT-4o |
| InternVL 3.0 | 72B | ~72 | ~78 | ~860 | GPT-4o |
| **InternVL 3.5** | **241B(A28B)** | **77.7** | **82.7** | **907** | **GPT-5** |

---

## 二、早期版本核心创新回顾（1.5 / 2.0 / 2.5）

> 本节精简呈现早期版本的关键技术贡献，详细技术细节可参阅各版本技术报告原文。

### 2.1 InternVL 1.5：动态高分辨率 + 大视觉编码器

**解决的核心问题**：早期 VLM 将图像缩放到固定 336×336，OCR/文档任务细节大量丢失。

**两大核心创新**：

**① Dynamic Tiling（动态分块）**

```
输入图像 → 根据宽高比选最优分块方案（如 3×2）
         → 切分为若干 448×448 Tiles
         → 每个 Tile 独立过 InternViT 提取特征（256 tokens/Tile）
         → 加入全局缩略图（Thumbnail）保留全局语义
         → 所有 Tile 特征拼接 → 送入 LLM
```

- 每 Tile 448×448，默认最多 12 Tiles，视觉 tokens 最高 3328
- 缩略图保证局部细节 + 全局语义兼顾

**② InternViT-6B（自研大视觉编码器）**

- ~6B 参数的 ViT，远超 CLIP ViT-L（~300M）
- 45 层 Transformer，3200 维特征，Patch Size 14×14
- 通过 Pixel Shuffle（Pixel Unshuffle r=2）将 1024 tokens 零信息损失压缩到 256 tokens
- 消融实验证明：视觉编码器 300M→6B（20×），MMMU 提升 5.6 分，OCRBench 提升 109 分

**③ 三阶段训练**：MLP 对齐预训练 → 全参数训练（解冻 ViT）→ 指令微调（SFT）

### 2.2 InternVL 2.0：全系列规模化

**核心贡献**：

- **Progressive Scaling**：2B/4B/8B/26B/40B/76B 全系列统一架构，小模型先探索超参，大模型继承
- **多图 + 视频输入**：多图 token 顺序拼接 + 特殊分隔符；视频均匀抽帧（8~32 帧），每帧走动态分辨率
- **精细化数据配比**：SFT 约 1200 万条，系统实验了多任务数据平衡点，解决多任务遗忘问题
- **数据质量控制**：GPT-4o 辅助标注约 30%，CoT 数据初步引入

### 2.3 InternVL 2.5：追平 GPT-4o

**核心改进**：

- **InternViT-6B v2.5**：预训练引入高分辨率图像 + ITC-Dense（区域级对比），增强细粒度定位
- **MLP Projector 加入 LayerNorm**：归一化视觉特征尺度，防止破坏 LLM 激活分布
- **渐进式分辨率预热**：Stage 2 中从 1 Tile 逐渐增到 12 Tiles，训练更稳定
- **数据课程学习**：Easy→Medium→Hard（5:3:2），SFT 约 1600 万条，CoT 数据占 15%
- **结果**：在 MMMU(70.1)、MathVista(76.6)、OCRBench(822) 上追平甚至超越 GPT-4o

### 2.4 Tile 间无注意力——早期架构的核心局限

InternVL 1.5~2.5 的一个关键架构特点：**不同 Tile 之间在 ViT 内无交互**，跨 Tile 信息融合完全依赖 LLM。

```
优势: 计算可并行、显存线性增长、灵活性高
劣势: Tile 边界物体被截断、LLM 负担重、位置编码有歧义
```

对比 Qwen2-VL 的 NaViT 方案（全局 self-attention、M-RoPE 3D 位置编码），InternVL 的 Tile 隔离方案在空间推理上有天然劣势，但凭借 6B 视觉编码器的特征质量弥补。

---

## 三、InternVL 3.0：原生多模态预训练的范式转换

> InternVL 3.0（2025.4，arXiv:2504.10479）是从传统"文本 LLM 适配多模态"到"原生多模态预训练"的关键转折。

### 3.1 训练范式变革

**传统范式（1.5~2.5）**：先训练纯文本 LLM → 冻结/解冻 ViT 做多模态对齐 → SFT

**原生多模态预训练（3.0 起）**：文本和多模态数据从一开始就联合训练

```
Stage 1: 原生多模态预训练（CPT）
  数据: ~200B tokens（文本:多模态 ≈ 1:3）
  所有模块同时训练（ViT + MLP + LLM）
  → 视觉和语言能力从底层就协同发展

Stage 2: 指令微调（SFT）
  数据: ~2170 万条
  
Stage 3: 偏好优化（MPO）
  Mixed Preference Optimization
  L_MPO = w_p × L_DPO + w_q × L_NLL(rejected) + w_g × L_NLL(chosen)
```

### 3.2 关键技术引入

- **V2PE（Variable Visual Position Encoding）**：为视觉 tokens 分配更灵活的位置增量，替代固定位置 ID
- **MPO（Mixed Preference Optimization）**：DPO + BCO + LM Loss 的混合偏好优化
- **LLM 底座切换**：从 InternLM 切换到 **Qwen2.5** 系列
- **测试时缩放**：支持深度思考（sequential CoT）和并行思考（Best-of-N + VisualPRM 打分）

---

## 四、InternVL 3.5 技术深度解读 ★

> InternVL 3.5（2025.8，arXiv:2508.18265）是当前最新版本，在 3.0 的基础上引入 Cascade RL、动态视觉压缩、解耦部署等创新，**整体性能追平 GPT-5**。

### 4.1 架构概览

基础架构保持 **ViT-MLP-LLM** 三段式，但各组件全面升级：

```
┌─────────────────────────────────────────────────────────┐
│  InternVL 3.5 架构                                      │
│                                                         │
│  视觉编码器: InternViT-300M (小模型) / InternViT-6B (大模型)│
│       ↓                                                │
│  MLP Projector (含 LayerNorm)                           │
│       ↓                                                │
│  ViR (Visual Resolution Router) ← 3.5 新增              │
│       ↓ 动态压缩视觉 tokens                              │
│  LLM: Qwen3 系列 (0.6B ~ 235B-A22B)                    │
│       ↓                                                │
│  输出（支持思考模式 <think>...</think>）                   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 全系列模型配置

InternVL 3.5 提供 Dense 和 MoE 两种架构，覆盖 1B 到 241B：

| 模型 | 视觉编码器 | LLM 底座 | 总参数 |
|---|---|---|---|
| InternVL3.5-1B | InternViT-300M | Qwen3-0.6B | 1.06B |
| InternVL3.5-2B | InternViT-300M | Qwen3-1.7B | 2.35B |
| InternVL3.5-4B | InternViT-300M | Qwen3-4B | 4.73B |
| InternVL3.5-8B | InternViT-300M | Qwen3-8B | 8.53B |
| InternVL3.5-14B | InternViT-300M | Qwen3-14B | 15.12B |
| InternVL3.5-38B | **InternViT-6B** | Qwen3-32B | 38.40B |
| InternVL3.5-20B-A4B (MoE) | InternViT-300M | Qwen3-30B-A3B | 21.23B |
| InternVL3.5-30B-A3B (MoE) | InternViT-300M | Qwen3-30B-A3B | 30.85B |
| **InternVL3.5-241B-A28B** (MoE) | **InternViT-6B** | Qwen3-235B-A22B | **240.70B** |

**关键变化**：
- LLM 底座从 InternLM → Qwen2.5（3.0）→ **Qwen3**（3.5），追踪最强开源 LLM
- 小模型（≤14B）统一使用 InternViT-300M 而非 6B，平衡效率
- 新增 MoE 版本，旗舰 241B 模型激活参数仅 28B

### 4.3 核心创新一：Cascade RL（级联强化学习）

这是 InternVL 3.5 最重要的训练创新，**推理能力平均提升 16%**。

**为什么需要级联？**

单独使用在线 RL（如 GRPO）容易出现奖励作弊（reward hacking）和训练不稳定；单独使用离线 RL（如 DPO）则探索能力不足。Cascade RL 结合两者优势：

```
Stage 3a: 离线 RL — MPO（Mixed Preference Optimization）
  数据: ~200K 偏好对（MMPR v1.2 数据集）
  损失: L_MPO = w_p × L_DPO + w_q × L_NLL(rejected) + w_g × L_NLL(chosen)
  作用: 稳定基础对齐，防止奖励作弊
  输出: 稳定的 MPO checkpoint
         ↓

Stage 3b: 在线 RL — GSPO（Generalized Self-Play Optimization）
  初始化: 从 MPO checkpoint 开始
  数据: ~70K 查询（MMPR-Tiny）
  特点: 
    - 类似 GRPO 但无参考模型约束
    - 仅选取准确率在 0.2~0.8 之间的查询（太简单或太难的跳过）
    - 同时适用于 Dense 和 MoE 架构
  作用: 在稳定基础上进一步精细化对齐
```

**效果量化**：

| 训练阶段 | 推理 Benchmark 平均分 |
|---|---|
| SFT-only (InternVL3.5-8B) | ~52 |
| + MPO (离线 RL) | ~56 (+4) |
| + GSPO (在线 RL，从 MPO 初始化) | **~60 (+8)** |
| 直接用 GSPO (跳过 MPO) | ~54（不稳定，低于级联方案） |

> 关键发现：离线 MPO 提供稳定起点，在线 GSPO 在此基础上精细优化。级联方案比单独使用任一方法都更优。

### 4.4 核心创新二：ViR（Visual Resolution Router）

**解决的问题**：Dynamic Tiling 生成大量视觉 tokens（12 Tiles × 256 = 3072），但并非所有区域都需要高分辨率表示。

**ViR 的核心思路**：为每个 Tile 动态决定压缩率——语义丰富的区域保留更多 tokens，简单背景区域大幅压缩。

```
标准模式 (无 ViR):
  每 Tile → Pixel Shuffle r=2 → 256 tokens（固定）

Flash 模式 (带 ViR):
  每 Tile → ViR Router 判断语义复杂度
          → 复杂区域: Pixel Shuffle r=2 → 256 tokens（保留）
          → 简单区域: Pixel Shuffle r=4 → 64 tokens（压缩）
  
  平均效果: 减少约 50% 视觉 tokens
```

**Router 的训练方式**：

ViR 的 Router 标签来自 ViCO（Visual Consistency Learning）的损失比：

```
对于每个 Tile i:
  r_i = L_ViCO(y_i | 高压缩率) / L_ViCO(y_i | 低压缩率)

  r_i 大 → 高压缩会严重损失信息 → 保留高分辨率（256 tokens）
  r_i 小 → 高压缩影响不大        → 使用低分辨率（64 tokens）

  阈值 τ 控制压缩比例
```

### 4.5 核心创新三：ViCO（Visual Consistency Learning）

**目的**：让模型在不同压缩率下保持输出一致性，为 ViR 的动态压缩提供基础。

```
训练过程:
  同一张图像，用两种压缩率（r=2 即 1/4 和 r=4 即 1/16）分别前向传播
  
  L_ViCO = KL(P(y | I_1/4) || P(y | I_1/16))
  
  最小化 KL 散度 → 模型学会在低分辨率下也能给出接近高分辨率的回答
  
训练后:
  用 ViCO 损失比训练 Router 标签（见上节）
```

**两阶段流程**：
1. **一致性训练**：用 ViCO 损失训练模型对不同压缩率的鲁棒性
2. **Router 训练**：根据 ViCO 损失比生成标签，训练 ViR Router

### 4.6 核心创新四：DvD（Decoupled Vision-Language Deployment）

**解决的问题**：视觉编码器和 LLM 的计算特性不同（ViT 是固定计算，LLM 是自回归），放在同一 GPU 会造成资源浪费。

```
传统部署:
  [GPU] ViT + MLP + LLM → 所有组件共享 GPU 资源

DvD 部署:
  [GPU 1] ViT + MLP + ViR    → 视觉处理专用
       ↓ 单向 TCP 传输 (BF16 visual features)
  [GPU 2~N] LLM              → 语言生成专用

  异步 3 阶段流水线:
    Stage A: 预填充视觉 tokens (ViT)
    Stage B: 预填充文本 tokens (LLM)
    Stage C: 自回归解码 (LLM)
    
  → A 和 B/C 可异步执行，提高吞吐量
```

### 4.7 完整训练流程

InternVL 3.5 采用 4 阶段训练（Flash 模型为 5 阶段）：

```
Stage 1: 持续预训练（CPT）
  ├── 数据: ~116M 样本，~250B tokens
  ├── 文本:多模态 ≈ 1:2.5
  ├── 损失: NTP (Next Token Prediction)，平方根平均
  ├── 所有模块可训练（ViT + MLP + LLM）
  └── 与 3.0 一致的原生多模态预训练

Stage 2: 指令微调（SFT）
  ├── 数据: ~56M 样本，~130B tokens
  ├── 文本:多模态 ≈ 1:3.5
  ├── 上下文长度: 32K
  ├── 新增数据类型: 思考模式、GUI 交互、具身智能、SVG
  └── 远超 2.5 的 16M 规模

Stage 3: 级联 RL
  ├── Stage 3a: 离线 MPO（~200K 偏好对）
  └── Stage 3b: 在线 GSPO（~70K 查询）

Stage 4: ViCO + ViR（仅 Flash 模型）
  ├── 一致性学习
  └── Router 训练
```

**数据规模对比**：

| 维度 | InternVL 2.5 | InternVL 3.0 | InternVL 3.5 |
|---|---|---|---|
| 预训练 tokens | ~100B | ~200B | ~250B |
| SFT 样本数 | ~16M | ~21.7M | **~56M** |
| RL 数据 | 无 | ~200K (MPO) | ~200K (MPO) + ~70K (GSPO) |
| SFT 新增类型 | CoT | V2PE | 思考模式/GUI/具身/SVG |

### 4.8 新能力

**① 思考模式（Thinking Mode）**

支持 `<think>...</think>` 标签的深度推理，通过系统提示激活：

```
系统提示: "请在回答前先进行深入思考，用 <think> 标签包裹思考过程"

模型输出:
<think>
这道几何题需要...
首先观察到三角形ABC中...
根据勾股定理...
</think>

答案是 ...
```

**② 测试时缩放（Test-Time Scaling）**

- **深度思考**：延长推理链（Sequential CoT），更多思考步骤
- **并行思考**：Best-of-N 采样 + VisualPRM v1.1 打分器选最优

**③ GUI 交互与 Agent 能力**

- ScreenSpot-v2: 92.9（GUI 元素定位）
- OSWorld-G: 53.2（桌面操作系统交互）

**④ 3D 空间理解**

- VSI-Bench: 63.7~69.5（3D 空间推理）

**⑤ SVG 理解与生成**

- SGP-Bench 上表现优异

### 4.9 Benchmark 全面评测

**旗舰模型对比（InternVL3.5-241B-A28B）**：

| Benchmark | InternVL 3.5-241B | GPT-5 | GPT-4o | Qwen2.5-VL-72B |
|---|---|---|---|---|
| **MMMU** | 77.7 | 84.2 | 69.9 | 68.2 |
| **MathVista** | **82.7** | 81.9 | 63.8 | 74.2 |
| **OCRBench** | **90.7** | 80.7 | 73.6 | 88.5 |
| MMBench | 87.4 | 88.6 | 83.4 | — |
| MMStar | **77.9** | 75.7 | — | — |
| AIME24 | 84.7 | 90.0 | — | — |
| MMLU-Pro | 81.3 | 85.6 | — | — |
| Video-MME | 72.9 | 81.8 | 77.2 | — |
| ScreenSpot-v2 | **92.9** | — | — | — |

**综合评分**：InternVL3.5-241B-A28B 整体 74.1 vs GPT-5 的 74.0，**开源首次追平 GPT-5 级别**。

**各尺寸模型表现**：

| 模型 | MMMU | OCRBench | MathVista |
|---|---|---|---|
| InternVL3.5-1B | 44.2 | 795 | — |
| InternVL3.5-2B | 59.0 | 836 | — |
| InternVL3.5-8B | 73.4 | 832 | 78.4 |
| InternVL3.5-30B-A3B (MoE) | 75.6 | 880 | 80.9 |
| InternVL3.5-38B | 76.9 | — | — |
| InternVL3.5-241B-A28B | 77.7 | 907 | 82.7 |

**InternVL 3.5 vs 3.0 推理能力提升**：

| 模型 | InternVL 3.0 | InternVL 3.5 | 提升 |
|---|---|---|---|
| 2B 推理平均分 | 32.4 | 50.7 | **+18.3** |
| 8B 推理平均分 | 44.3 | 60.3 | **+16.0** |

> Cascade RL 是推理能力大幅跃升的主因。

**InternVL 3.5 优于 GPT-5 的领域**：MathVista、OCRBench、MMStar
**GPT-5 仍领先的领域**：MMMU、AIME24、Video-MME、MMLU-Pro

### 4.10 Flash 模式效率分析

ViR + ViCO 带来的推理加速效果：

```
InternVL3.5-8B 标准模式:
  12 Tiles × 256 tokens = 3072 视觉 tokens
  推理延迟: 1.0x

InternVL3.5-8B Flash 模式:
  ViR 动态压缩后: ~1500 视觉 tokens（平均减少 ~50%）
  推理延迟: ~0.25x（4.05x 加速）
  性能损失: <1%（在大多数 Benchmark 上）
```

---

## 五、与其他 VLM 的技术对比（更新版）

### 5.1 架构路线对比

| 维度 | InternVL 3.5 | Qwen2.5-VL | LLaVA-OneVision |
|---|---|---|---|
| 视觉编码器 | InternViT (300M/6B) | ViT-600M (NaViT) | CLIP ViT-L (~300M) |
| 动态分辨率 | Dynamic Tiling + Thumbnail | Naive Dynamic (原生变长) | AnyRes |
| Tile 间交互 | 无（ViT 内隔离） | **全局 self-attention** | 无 |
| 位置编码 | 1D RoPE + V2PE | **M-RoPE (3D)** | 1D RoPE |
| Token 压缩 | Pixel Shuffle + **ViR 动态** | 2×2 merging | 无 |
| LLM 底座 | Qwen3 | Qwen2.5 | LLaMA-3 |
| RL 对齐 | **Cascade RL (MPO+GSPO)** | DPO | 无 |
| 思考模式 | ✅ | ✅ | ❌ |
| MoE 支持 | ✅ (241B-A28B) | ❌ | ❌ |

### 5.2 核心差异分析

**InternVL 3.5 vs Qwen2.5-VL**：

```
InternVL 3.5 优势:
  ✓ Cascade RL 带来更强的推理能力（MathVista 82.7 vs 74.2）
  ✓ 大视觉编码器在细粒度视觉特征上更强
  ✓ ViR 动态压缩提供更灵活的效率-性能权衡
  ✓ MoE 旗舰模型规模更大（241B vs 72B）

Qwen2.5-VL 优势:
  ✓ M-RoPE 提供天然的空间/时序位置编码（空间推理更强）
  ✓ NaViT 全局注意力无 Tile 边界问题
  ✓ 视频理解数据更丰富
  ✓ 视觉编码器更轻量，推理效率更高（不用 6B ViT）

本质差异:
  InternVL: "重视觉 + 重 RL 对齐" → 推理和细粒度理解强
  Qwen-VL: "重位置编码 + 重数据" → 空间理解和视频理解强
```

### 5.3 性能对比总结

| Benchmark | InternVL3.5-241B | Qwen2.5-VL-72B | GPT-5 | 能力维度 |
|---|---|---|---|---|
| MMMU | 77.7 | 68.2 | 84.2 | 综合多学科理解 |
| MathVista | **82.7** | 74.2 | 81.9 | 数学图像推理 |
| OCRBench | **90.7** | 88.5 | 80.7 | OCR 综合 |
| MMStar | **77.9** | — | 75.7 | 多维度视觉问答 |
| Video-MME | 72.9 | ~80 | 81.8 | 视频理解 |

---

## 六、关键技术总结与启示

### 6.1 InternVL 系列的核心技术贡献

```
一级创新（系列性原创）:
  1. 大视觉编码器路线 (InternViT-6B) — 证明视觉编码器规模效益被低估
  2. Dynamic Tiling + Thumbnail — 动态分辨率的工程最优解
  3. Cascade RL (MPO + GSPO) — 级联强化学习，推理能力大幅提升 ★ 3.5 新增
  4. 原生多模态预训练 — 从"适配"到"共训"的范式转换 ★ 3.0 新增

二级创新（精细化工程）:
  5. ViR + ViCO — 动态视觉 token 压缩，4x 加速 <1% 性能损失
  6. V2PE — 可变视觉位置编码
  7. 数据课程学习 + 多任务平衡 — 系统性数据工程
  8. Progressive Scaling — 小模型探索，大模型继承

三级创新（工程实践）:
  9. Pixel Shuffle 压缩 — 零信息损失的空间降采样
  10. DvD 解耦部署 — 视觉/语言模型异步流水线
```

### 6.2 系列演进的核心规律

```
1. LLM 底座追踪最强开源: InternLM → Qwen2.5 → Qwen3
   → 不执着于自研 LLM，务实选择最优底座

2. 训练策略持续深化: 三阶段 SFT → 原生预训练 + MPO → Cascade RL
   → RL 对齐成为 VLM 推理能力的关键

3. 数据规模指数增长: SFT 4M → 12M → 16M → 56M
   → 但质量控制（GPT-4o 标注、CoT 数据）同等重要

4. 效率优化逐步成熟: 固定压缩 → 动态压缩(ViR) → 解耦部署(DvD)
   → 从"能用"到"好用"的工程化进程

5. 视觉编码器策略分化: 大模型用 6B ViT，小模型用 300M ViT
   → 实际部署考虑，不再一刀切
```

### 6.3 当前局限与未来方向

**仍有差距的领域**：
- **视频长序列理解**：Video-MME 仍落后 GPT-5 约 9 分
- **极端推理**：AIME24 落后 GPT-5 约 5 分
- **纯语言能力**：MMLU-Pro 落后 GPT-5 约 4 分

**可预期的演进方向**：
- 更高效的原生多模态预训练（减少对文本 LLM 预训练权重的依赖）
- 视频原生 Transformer（端到端时序建模，而非抽帧）
- 更强的 RL 对齐（结合 RLHF 和过程奖励模型）
- 3D 空间理解增强（与具身智能结合）
- 更轻量的视觉编码器方案（追求 InternViT-6B 级别效果、300M 级别参数）

---

**相关文档**：
- [VLM概述](../多模态/VLM概述.md)
- [VLM主流架构详解](../多模态/VLM主流架构详解.md)
- [主流模型总览](主流模型总览.md)

[返回上级](README.md) | [返回总目录](../../README.md)

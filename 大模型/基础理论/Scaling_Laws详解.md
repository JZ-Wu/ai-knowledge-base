# Scaling Laws 详解

Scaling Laws 描述了模型性能（loss）与**模型大小 N、数据量 D、计算量 C** 之间的幂律关系，是大模型训练的"规划指南"。

---

## 一、Kaplan Scaling Law (OpenAI, 2020)

### 1.1 核心发现

模型的 loss 与 N、D、C 分别呈**幂律**关系：

$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}$$

其中 $\alpha_N \approx 0.076$，$\alpha_D \approx 0.095$，$\alpha_C \approx 0.050$

直观理解：**模型每大 10 倍，loss 下降一个固定的比例**。而且这个关系在很大范围内（6 个数量级）稳定成立。

### 1.2 Kaplan 的结论（后来被修正）

> 给定固定计算预算 C，应该**优先扩大模型**，数据量不需要等比增长。
>
> 最优分配：$N \propto C^{0.73}$，$D \propto C^{0.27}$

这意味着：计算翻 10 倍 → 模型大 5.4 倍，数据只多 1.9 倍。

**这个结论导致了 GPT-3 (175B) 只用 300B tokens 训练** —— 后来被证明数据严重不足。

---

## 二、Chinchilla Scaling Law (DeepMind, 2022)

### 2.1 修正 Kaplan

Chinchilla 论文（"Training Compute-Optimal Large Language Models"）发现 Kaplan 的实验设计有偏差，重新拟合后结论完全不同：

> **模型参数 N 和训练数据 D 应该等比例增长。**
>
> 最优分配：$N \propto C^{0.5}$，$D \propto C^{0.5}$

### 2.2 Chinchilla 最优比例（面试必记）

$$D_{\text{optimal}} \approx 20 \times N$$

**即：训练 token 数应该是模型参数的 20 倍。**

| 模型参数 N | 最优训练数据 D | 典型代表 |
|---|---|---|
| 1B | 20B tokens | — |
| 7B | 140B tokens | — |
| 70B | 1.4T tokens | Chinchilla (70B, 1.4T) |
| 175B | 3.5T tokens | GPT-3 只用了 300B（严重欠训练）|

### 2.3 Chinchilla 的实验验证

DeepMind 训练了 Chinchilla（70B，1.4T tokens），对比 Gopher（280B，300B tokens）：

```
Gopher:     280B params, 300B tokens  → 不符合 Chinchilla 最优
Chinchilla:  70B params, 1.4T tokens  → 符合 20:1 比例

结果: Chinchilla 全面优于 Gopher！
      模型小了 4 倍，但数据多了 4.7 倍
      → 更小的模型 + 更多数据 = 更好的性能 + 更低的推理成本
```

### 2.4 对行业的影响

Chinchilla 之后，所有大模型团队都大幅增加了训练数据量：

| 模型 | 参数 | 训练 tokens | D/N 比例 |
|---|---|---|---|
| GPT-3 | 175B | 300B | 1.7x（严重欠训练）|
| Chinchilla | 70B | 1.4T | 20x（最优）|
| LLaMA-1 | 7/13/33/65B | 1~1.4T | 20-200x |
| LLaMA-2 | 7/13/70B | 2T | 29-286x |
| LLaMA-3 | 8/70B | 15T | 214-1875x |

> 注意 LLaMA 系列明显超过了 20:1。这是因为实际生产中，推理成本比训练成本更重要——**过度训练一个小模型**（训练贵一些），换来推理时用小模型（推理便宜很多），总成本更低。

---

## 三、给你 X 张卡，怎么规划训练？

面试常见问题："给你 100 张 A100，训练一个最好的模型，怎么规划？"

### 3.1 计算量估算

```
训练计算量 C ≈ 6 × N × D  (单位: FLOPs)

其中:
  N = 模型参数
  D = 训练 token 数
  6 = 每个 token 大约 6N 次浮点运算 (前向 2N + 反向 4N)
```

### 3.2 训练时间估算

```
训练时间 T = C / (GPU数 × 单卡算力 × MFU)

A100:  单卡算力 = 312 TFLOPS (BF16)
H100:  单卡算力 = 989 TFLOPS (BF16)
MFU (Model FLOPs Utilization) ≈ 0.3~0.5（实际利用率）
```

### 3.3 示例计算

```
100 张 A100，训练 7B 模型：

最优数据量: D = 20 × 7B = 140B tokens
计算量: C = 6 × 7B × 140B = 5.88 × 10²¹ FLOPs

训练时间 T = 5.88×10²¹ / (100 × 312×10¹² × 0.4)
           = 5.88×10²¹ / (1.248×10¹⁶)
           ≈ 4.7×10⁵ 秒
           ≈ 5.4 天

如果想训更多数据 (2T tokens):
T = 6 × 7B × 2T / (100 × 312T × 0.4) ≈ 78 天
```

---

## 四、Over-training（过度训练）

### 4.1 为什么要 Over-train？

Chinchilla 最优是**训练 compute-optimal**——用最少的计算达到最好的 loss。但实际部署时还要考虑**推理成本**：

```
场景: 你有 1000 GPU × 30 天的计算预算

Chinchilla 最优:
  训练一个 30B 模型，用 600B tokens
  → 训练成本: 全部预算
  → 推理成本: 30B 模型，需要 60GB 显存，A100×1 运行慢

Over-training 方案:
  训练一个 7B 模型，用 2.5T tokens (D/N = 360)
  → 训练成本: 约同样的预算（C ≈ 6×7B×2.5T ≈ 同量级）
  → 推理成本: 7B 模型，14GB 显存，单卡快速推理

  虽然 7B+2.5T 的 loss 略高于 30B+600B (训练不是最优的)
  但 7B 模型的推理成本只有 30B 的 1/4！
  如果模型要服务百万用户 → 推理成本远大于训练成本 → Over-train 小模型更划算
```

### 4.2 Over-training 的比例

| 模型 | 参数 N | 训练 tokens D | D/N | 相对 Chinchilla |
|---|---|---|---|---|
| Chinchilla | 70B | 1.4T | 20x | 最优 |
| LLaMA-1-7B | 7B | 1T | 143x | 7x over-train |
| LLaMA-2-7B | 7B | 2T | 286x | 14x over-train |
| LLaMA-3-8B | 8B | 15T | 1875x | 94x over-train |
| Qwen-2.5-7B | 7.6B | 18T | 2368x | 118x over-train |

LLaMA-3 用了 Chinchilla 最优量 **94 倍** 的数据来训练 8B 模型 → 小模型能力大幅提升。

### 4.3 Over-training 的 diminishing returns

```
Over-training 不是无限有效的:

  D/N = 20x (Chinchilla):  loss 最优
  D/N = 100x:              loss 只比 20x 低 ~5%
  D/N = 1000x:             loss 只比 100x 低 ~2%
  D/N = 10000x:            几乎不再下降

→ 收益递减 (diminishing returns)
→ 到某个点后，继续喂数据不如增大模型

实际的决策取决于:
  训练 1 次 vs 推理 1 万亿次 → 推理成本主导 → Over-train 合理
  训练 1 次 vs 推理 100 次 → 训练成本主导 → 用 Chinchilla 最优
```

---

## 五、涌现能力 (Emergent Abilities)

### 5.1 什么是涌现能力

```
定义 (Wei et al., 2022):
  涌现能力 = 在小模型中不存在、但在大模型中出现的能力
  "An ability is emergent if it is not present in smaller models
   but is present in larger models."

Scaling Laws 预测: loss 随 N 平滑下降（幂律）
涌现现象: 某些能力在模型小的时候完全没有，突然在某个规模出现
```

**典型涌现能力及其临界规模**：

| 能力 | 大致临界规模 | 代表 benchmark |
|---|---|---|
| Chain-of-Thought 推理 | ~60-100B | GSM8K, BBH |
| In-Context Learning (few-shot) | ~10B+ | 各种 NLP 任务 |
| 多步数学推理 | ~100B+ | MATH, GSM8K |
| 代码生成 | ~10B+ | HumanEval |
| 指令遵循 | ~10B+ | IFEval |
| 多步逻辑推理 | ~60B+ | Big-Bench Hard |
| 幽默理解 | ~100B+ | — |

```
示例 — GSM8K (小学数学推理):
  GPT-3 (175B, 无 CoT):    ~20%
  GPT-3 (175B, 有 CoT):    ~60%  ← CoT 在大模型上突然有效
  GPT-2 (1.5B, 有 CoT):    ~5%   ← 小模型用 CoT 也没用

  画成图:
  准确率
  100%|
     |                           * (175B, CoT)
   60%|                         *
     |
   20%|              *          * (175B, no CoT)
     |    *    *
    0%|___*____*________________*________
       1B  3B  10B  30B  60B  175B   模型规模

  → 没有 CoT 时准确率缓慢增长；有 CoT 时在 ~60B 出现"拐点"
```

### 5.2 "涌现是幻觉"的挑战

2023 年 Stanford 的论文 **"Are Emergent Abilities of Large Language Models a Mirage?"** (Schaeffer et al.) 引发了激烈争论：

#### 核心论点：涌现 = 度量假象

```
Schaeffer 的核心论证:

  Step 1: 选择不同的评测指标
    离散指标: 完全匹配 (Exact Match)
      → 模型输出必须 100% 匹配标准答案才算对
      → 接近正确但不完全对 = 0 分

    连续指标: Token-level accuracy / Brier Score / Log-likelihood
      → 衡量模型"接近正确"的程度
      → 部分正确也能得到部分分数

  Step 2: 同样的模型，不同的指标，不同的结论
    用 Exact Match:
      1B: 0%, 10B: 0%, 60B: 2%, 175B: 58%  → 看起来"突然涌现"

    用 Token-level accuracy:
      1B: 10%, 10B: 25%, 60B: 40%, 175B: 75%  → 平滑增长，无突变

  Step 3: 数学解释
    考虑一个 k 选 1 的选择题:
      随机猜对概率 = 1/k
      模型每个 token 的正确概率 = p (随规模平滑增长)
      但 Exact Match 要求所有 token 都对 → p^n (n 个 token)
      → 当 p 从 0.8 增到 0.95:
         单 token: 0.8 → 0.95 (平滑)
         10 token 全对: 0.8^10=0.107 → 0.95^10=0.599 (看起来像突变)

    这就是"幂次的阈值效应": 底层能力平滑增长，
    但复合指标在某个临界点急剧上升
```

#### 反驳与再争论

```
支持涌现是真实的论据:

  1. 定性能力的出现
     小模型完全不会 CoT → 大模型突然会
     这不是"部分正确"的问题，而是能力的质变
     例: 3B 模型的 CoT 输出是乱码，175B 的 CoT 是有逻辑的推理链

  2. Few-shot 行为的非线性
     小模型: 给 few-shot 示例反而降低性能（被示例干扰）
     大模型: 给 few-shot 示例显著提升性能
     → 存在一个从"不理解示例"到"理解示例"的质变

  3. 训练动态中的 Phase Transition
     训练过程中，某些能力不是渐进出现的:
     - Induction heads (Olsson et al., 2022): 在特定训练步骤突然形成
     - Grokking 现象: 过拟合很久后突然泛化
     → 内部表示确实可能经历相变

  4. 神经科学类比
     人脑也有类似的"涌现":
     神经元数量增加 → 产生意识（质变，非量变）
     但这个类比有争议
```

### 5.3 当前共识与开放问题

```
大多数研究者同意的:
  ✓ 预训练 loss 随规模平滑下降（Scaling Laws 可靠）
  ✓ 离散评测指标的"突变"很大程度上是度量问题
  ✓ 用连续指标看，大多数能力是平滑增长的
  ✓ 但确实存在"能力组合"的门槛——单个能力平滑增长，
    但多个能力的组合使用需要足够的模型容量

仍有争议的:
  ? "涌现"应该如何严格定义？是关于度量还是关于能力？
  ? 模型内部表示是否存在真正的 phase transition？
  ? 能否从 Scaling Laws 预测哪些能力会在什么规模出现？

对科研的启示:
  1. 评测方法很重要——选错指标会导致错误结论
  2. 要区分"模型能力的增长"和"benchmark 分数的增长"
  3. 涌现可能不是一个 binary 问题，而是一个"多尺度平滑"问题:
     - 底层 (token-level): 平滑增长
     - 中层 (task-level): 存在 soft transition
     - 顶层 (综合能力): 看起来像突变
```

### 5.4 从 Scaling Laws 到能力预测

```
Scaling Laws 的梦想: 训练前就预测模型能力
Scaling Laws 的现实: 只能预测 loss，不能预测具体能力

为什么?
  loss 是所有 token 的平均表现 → 是一个全局指标
  具体能力 (如数学推理) 是特定子集的表现 → 局部指标
  全局平均平滑 ≠ 局部突破也平滑

部分进展:
  1. Broken Neural Scaling Laws (Caballero et al., 2023)
     提出修正版 scaling law，能拟合非幂律的曲线
     对 benchmark 级别的预测更准确
     但仍需要至少 3-4 个规模点的数据来拟合

  2. Predictable Scaling (Gadre et al., 2024)
     某些下游任务的 loss 也遵循 power law
     → 如果有足够多的小规模实验数据，可以外推大模型表现
     但精度有限，尤其对"涌现式"跳变预测不准

  3. Task-Specific Scaling Laws
     不同任务有不同的 scaling 行为:
     - 知识密集任务 (QA, fact): 随数据量平滑提升
     - 推理密集任务 (math, code): 随模型规模有更强的超线性增长
     - 组合能力 (multi-hop reasoning): 可能存在阈值效应

实际意义:
  - 不能指望 Scaling Laws 告诉你"训练多大模型能解 IMO 题"
  - 但可以用它来规划计算资源分配和训练数据量
  - 对能力的预测仍然需要实验验证
```

---

## 六、Scaling Laws 的局限与前沿

### 6.1 已知局限

| 局限 | 说明 |
|---|---|
| **只预测 loss** | 不能预测下游任务的具体分数 |
| **数据质量未建模** | 同样 1T token，质量不同效果天差地别 |
| **后训练未建模** | SFT/RLHF 的效果没有 Scaling Law |
| **架构无关假设** | 不同架构（Dense vs MoE）有不同的 Scaling Law |
| **多模态未建模** | 图文混合训练的 Scaling Law 尚不明确 |

### 6.2 Data Wall 问题

```
互联网高质量文本数据估计: ~10T tokens (去重后)
当前最大模型已经用了 15T+ tokens (LLaMA-3)

应对策略:
  1. 合成数据: 用强模型生成高质量训练数据
  2. 多模态数据: 利用图片、视频、音频中的信息
  3. 数据质量 > 数量: 用 classifier 筛选高质量子集
  4. 重复使用高质量数据: 高质量数据可以多 epoch 使用（2-4x）
     但不能太多 epoch → 过拟合

未来趋势:
  - "Test-time compute scaling": 不增大训练量，
    而是在推理时花更多计算（如 CoT、搜索）
  - DeepSeek-R1 的成功证明了这个方向的潜力
```

### 6.3 Test-Time Compute Scaling

```
传统 Scaling: 更多训练 → 更好的模型
TTC Scaling:  固定模型 → 推理时花更多计算 → 更好的结果

方式 1: 更长的推理链 (Chain-of-Thought)
  简单问题: "1+1=?" → 直接回答
  复杂问题: "..." → 让模型思考 1000 个 token 再回答

方式 2: 多次采样 + 验证 (Best-of-N)
  生成 N 个答案 → 用 reward model 选最好的

方式 3: 搜索 (MCTS / Beam Search)
  像 AlphaGo 一样搜索推理路径

DeepSeek-R1 的证明:
  让模型自由思考 (可能思考几万个 token)
  → 数学推理能力大幅超越同规模模型
  → 推理时的 compute scaling 也遵循某种 power law
```

---

**面试快速回答模板**：

> **Q: Chinchilla Scaling Law 说了什么？**
> A: 给定计算预算，模型参数和训练数据应等比增长。最优比例约 D = 20N（训练 token 数是参数的 20 倍）。但实际中推理成本主导，所以大家都在 over-train 小模型（如 LLaMA-3-8B 用了 15T tokens，D/N 高达 1875x）。
>
> **Q: 给你 X 卡训模型怎么规划？**
> A: 先算总计算量 C = GPU数 × 算力 × MFU × 训练时间，再根据 C ≈ 6ND 决定 N 和 D 的分配。如果推理场景多，选小 N 大 D（over-train）；如果只用一次，选 Chinchilla 最优。
>
> **Q: 涌现能力是真的吗？**
> A: loss 随规模平滑下降是确定的。"涌现"更多是离散评测指标的阈值效应——换成连续指标后能力增长是平滑的。但确实存在某些能力（如 CoT 推理）需要足够大的模型才能有效使用。

---

[返回上级](README.md) | [返回总目录](../../README.md)

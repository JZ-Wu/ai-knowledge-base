# SFT 详解 (Supervised Fine-Tuning)

有监督微调，是后训练的第一步，将基座模型转变为能遵循指令的助手模型。

## 一、核心思想

用 **(指令, 高质量回答)** 配对数据，以监督学习的方式微调基座模型。

本质上仍然是 **next token prediction**，但只在回答部分计算损失（指令部分的 token 被 mask 掉，不参与梯度更新）。

```
[指令 token ... ] [回答 token ...]
      ↑ 不计算 loss       ↑ 计算 loss
```

## 二、数据格式

### 单轮对话

```json
{
  "instruction": "解释什么是梯度下降",
  "output": "梯度下降是一种优化算法..."
}
```

### 多轮对话

```json
{
  "messages": [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是梯度下降？"},
    {"role": "assistant", "content": "梯度下降是..."},
    {"role": "user", "content": "它有哪些变体？"},
    {"role": "assistant", "content": "主要变体包括..."}
  ]
}
```

> 通常只对 `assistant` 角色的 token 计算 loss。

### Chat Template

每个模型有自己的对话模板，将 messages 格式转化为实际输入的 token 序列。例如 ChatML 格式：

```
<|im_start|>system
你是一个有帮助的助手。<|im_end|>
<|im_start|>user
什么是梯度下降？<|im_end|>
<|im_start|>assistant
梯度下降是...<|im_end|>
```

不同模型模板不同（LLaMA 用 `[INST]`，Qwen 用 ChatML），**用错模板会严重影响效果**。HuggingFace 的 `tokenizer.apply_chat_template()` 可自动处理。

## 三、关键技术细节

### 损失函数

标准交叉熵，但只在回答部分计算（对 response token 数取平均）：

$$L_{SFT} = -\frac{1}{|T_{\text{resp}}|}\sum_{t \in \text{response}} \log P_\theta(x_t | x_{\lt t})$$

其中 $|T_{\text{resp}}|$ 是回答部分的 token 数。不取平均的话，长回答的 loss 天然比短回答大，会导致梯度尺度不一致。

### 学习率

- 比预训练低 1~2 个数量级，通常 1e-5 ~ 5e-5
- 使用 cosine 或 linear decay 调度
- 预热步数较少（几十到几百步）

### 训练轮次

- 通常 1~3 个 epoch
- 过多 epoch 容易过拟合（SFT 数据量远小于预训练数据）
- 观察验证集 loss，一旦开始上升就停止

### Packing（序列打包）

- 将多条短样本拼接到一个序列中，填满 max_length
- 用 attention mask 隔离不同样本，防止跨样本注意力
- 提升 GPU 利用率，避免 padding 浪费
- 注意：需要正确处理 position_ids 和 loss mask

### NEFTune（噪声嵌入微调）

- 在 embedding 层输出上添加均匀分布噪声
- 论文发现可以提升 SFT 效果（类似正则化）
- 实现简单：`embedding_output += noise * alpha / sqrt(seq_len * hidden_dim)`

## 四、SFT 数据质量 > 数量

### 关键发现

- **LIMA 论文 (2023)**：仅用 1000 条高质量数据做 SFT，效果接近 GPT-4 时代的模型
- 核心洞察：SFT 不是"教"模型新知识，而是"激活"预训练中已学到的能力，教它以正确的格式输出
- **Superficial Alignment Hypothesis**：对齐是表面的，模型的知识和能力几乎全部来自预训练

### 数据质量原则

1. **多样性**：覆盖不同任务类型（问答、写作、代码、数学、翻译等）
2. **准确性**：回答必须正确，错误回答会直接污染模型
3. **风格一致性**：保持统一的回答风格和格式
4. **适当长度**：过短缺乏信息，过长引入噪声
5. **去重**：相似的指令太多会导致过拟合某类任务

### 常见数据来源

| 来源 | 说明 | 代表 |
|------|------|------|
| 人工标注 | 质量最高，成本也最高 | InstructGPT 的标注数据 |
| 蒸馏 (Distillation) | 用更强的模型（如 GPT-4）生成回答 | Alpaca、Vicuna |
| Self-Instruct | 让模型自己生成指令和回答，人工筛选 | Self-Instruct 论文 |
| 真实对话 | 从用户实际使用中收集（脱敏） | ShareGPT |
| 开源数据集 | 社区整理的高质量数据 | OpenHermes、Infinity-Instruct |

## 五、多阶段 SFT

实际生产中，SFT 通常不是一步完成，而是分阶段进行：

```
阶段 1: 通用 SFT
├── 大量通用指令数据（10万~100万条）
├── 学会基本的指令遵循和对话格式
└── 学习率较高，训练较多步

阶段 2: 能力增强 SFT
├── 针对性数据（数学、代码、长文本等）
├── 提升特定能力
└── 数据量中等

阶段 3: 安全 & 风格 SFT
├── 少量高质量安全数据 + 风格数据
├── 微调回答风格、拒绝策略
└── 学习率很低，少量步数
```

> 类似预训练中的数据课程 (Curriculum)，先粗后细。

## 六、灾难性遗忘 (Catastrophic Forgetting)

SFT 的一个重要问题：在新数据上微调后，模型**忘记**了预训练/之前学到的能力。

### 表现

- 微调了中文对话后，英文能力下降
- 微调了代码后，通用问答变差
- 微调了安全拒绝后，有用性降低

### 缓解方法

| 方法 | 原理 |
|------|------|
| **数据混合 (Replay)** | 在 SFT 数据中混入一定比例的预训练数据 |
| **低学习率** | 减小参数更新幅度 |
| **LoRA** | 只更新少量参数，基座模型冻结，天然缓解遗忘 |
| **EWC / L2 正则化** | 约束重要参数不要变化太大（见下方详解） |
| **多任务混合训练** | 保证 SFT 数据覆盖所有需要保留的能力 |

### L2 正则化防遗忘

注意这里的 L2 正则化 **不是** 普通的 weight decay（把参数往 0 拉），而是把参数 **往预训练权重拉**：

$$\mathcal{L} = \mathcal{L}_{SFT} + \frac{\lambda}{2} \sum_i (w_i - w_i^{pretrained})^2$$

直觉：惩罚的不是参数的大小，而是参数 **偏离预训练值的程度**。微调时如果某个参数想跑太远，这个惩罚项就会把它拉回来，从而保留预训练学到的能力。

对 $w_i$ 求梯度，惩罚项贡献 $\lambda(w_i - w_i^{pretrained})$，更新规则变成：

$$w_i \leftarrow w_i - \eta\left(\nabla \mathcal{L}_{SFT} + \lambda(w_i - w_i^{pretrained})\right)$$

当 $w_i$ 远离预训练值时，惩罚梯度增大，把它拉回来；当 $w_i$ 接近预训练值时，惩罚几乎为 0，不影响正常学习。

EWC（Elastic Weight Consolidation）是这个思想的进阶版——它对 **每个参数** 给不同的 $\lambda_i$（用 Fisher 信息矩阵衡量参数的重要性），重要参数拉得紧，不重要的参数可以自由变化。

## 七、SFT 的局限

| 局限 | 说明 |
|------|------|
| **暴露偏差 (Exposure Bias)** | 训练时看到的都是正确回答，推理时遇到自身错误不知如何纠正 |
| **无法学习偏好** | 只学"正确答案是什么"，不学"什么样的回答更好" |
| **过度模仿** | 可能学到数据中的套话、冗余表述，而非真正有用的回答方式 |
| **上限受限于数据** | 模型无法超越标注者的水平 |

> 这些局限正是后续需要 RLHF / DPO 等偏好优化的原因。

---

## 八、参数高效微调 (PEFT)

全量 SFT 需要更新所有参数，对于 7B+ 的模型需要多张 GPU。参数高效微调只更新极少量参数，大幅降低成本。

### 总览

| 方法 | 核心思想 | 可训练参数 | 显存 | 效果 |
|------|---------|-----------|------|------|
| **全量微调** | 更新所有参数 | 100% | 极高 | 最好 |
| **LoRA** | 低秩分解旁路 | 0.1%~1% | 低 | 接近全量 |
| **QLoRA** | 量化基座 + LoRA | 0.1%~1% | 很低 | 接近 LoRA |
| **Adapter** | 层间插入小网络 | 1%~5% | 中 | 良好 |
| **Prefix Tuning** | 可学习虚拟前缀 | <1% | 低 | 一般 |
| **P-Tuning v2** | 每层加可学习前缀 | <1% | 低 | 较好 |
| **IA3** | 学习激活缩放因子 | <0.1% | 极低 | 一般 |

---

### 8.1 LoRA (Low-Rank Adaptation)

**论文**：LoRA: Low-Rank Adaptation of Large Language Models (2021, Microsoft)

#### 核心思想

预训练权重矩阵 $W_0$ 冻结不动，旁边加一个**低秩分解**的增量矩阵：

$$h = W_0 x + \Delta W x = W_0 x + B A x$$

其中：
- $W_0 \in \mathbb{R}^{d \times d}$：冻结的原始权重
- $A \in \mathbb{R}^{r \times d}$：降维矩阵（初始化为随机高斯）
- $B \in \mathbb{R}^{d \times r}$：升维矩阵（初始化为零）
- $r$：秩 (rank)，通常 4~64，远小于 $d$

```
输入 x ──→ W₀ (冻结) ──→ ┐
    │                      ├── 相加 → 输出 h
    └──→ A ──→ B ──────→ ┘
         r×d    d×r
     (降维)  (升维)
```

#### 为什么有效？

- **低秩假说**：微调时权重的变化量 $\Delta W$ 是低秩的（信息量不大）
- 预训练已经学到了强大的表示，微调只需要做小幅调整
- 实验表明 r=4~16 就足够（远小于 d=4096）

#### 通常应用在哪些层？

| 目标矩阵 | 说明 |
|----------|------|
| $W_q, W_v$ | 最常见的选择，效果好 |
| $W_q, W_k, W_v, W_o$ | 更多层，略好，参数稍多 |
| 全部线性层（含 FFN） | 参数更多，但效果最好 |

#### 关键超参数

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `r` (rank) | 8~64 | 秩越大表达能力越强，但参数越多 |
| `alpha` | 16~32 | 缩放系数，实际缩放因子为 `alpha/r` |
| `target_modules` | q,v 或全部 | 应用 LoRA 的层 |
| `dropout` | 0~0.1 | LoRA 层的 dropout |

#### 推理时的合并

训练完成后，可以将 LoRA 权重合并回原始权重，**推理时零开销**：

$$W_{merged} = W_0 + B A$$

合并后就是一个普通的 Dense 模型，不需要额外的 LoRA 代码。

#### LoRA 的变体

| 变体 | 改进 |
|------|------|
| **LoRA+** | A 和 B 使用不同学习率 |
| **DoRA** | 分解为方向和大小分别优化 |
| **AdaLoRA** | 自适应分配不同层的秩 |
| **rsLoRA** | 改进缩放因子为 $\alpha / \sqrt{r}$ |

---

### 8.2 QLoRA (Quantized LoRA)

**论文**：QLoRA: Efficient Finetuning of Quantized LLMs (2023, UW)

#### 核心思想

在 LoRA 的基础上，将冻结的基座模型量化到 **4-bit**，进一步压缩显存：

```
基座模型 W₀ ──(4-bit量化)──→ W₀_quant (冻结, 4-bit存储)
                                   │
LoRA 增量 BA ──(保持fp16/bf16)──→ 训练时反量化 W₀ + BA 做前向
```

#### 三项关键技术

1. **NF4 (4-bit NormalFloat)**：一种对正态分布权重最优的 4-bit 量化格式
2. **双重量化 (Double Quantization)**：对量化常数本身再做量化，进一步节省显存
3. **分页优化器 (Paged Optimizer)**：当 GPU 显存不够时，自动将优化器状态卸载到 CPU

#### 显存对比（以 LLaMA-65B 为例）

| 方法 | 显存 |
|------|------|
| 全量微调 (fp16) | ~780 GB（多卡） |
| LoRA (fp16 基座) | ~130 GB |
| QLoRA (4-bit 基座) | ~33 GB（**单张 A100 可训**） |

#### 效果

QLoRA 的效果非常接近全量微调（差距在 1% 以内），是当前**个人/小团队微调大模型的首选方案**。

---

### 8.3 Adapter

**论文**：Parameter-Efficient Transfer Learning for NLP (2019, Google)

这是最早的参数高效微调方法之一。

#### 核心思想

在 Transformer 每一层的内部，**插入小型的瓶颈网络**（Adapter 模块），只训练这些模块：

![Adapter 架构: 原始 Transformer 层 vs 加 Adapter 的层 (Houlsby et al., 2019)](assets/adapter_architecture.png)

> 图源: *Parameter-Efficient Transfer Learning for NLP*, Figure 2. 左图为 Adapter 模块的内部瓶颈结构 (Down-project → 非线性 → Up-project + 残差)；右图为 Adapter 在 Transformer 层中的插入位置 (Attention 之后和 FFN 之后各插一个)。

#### Adapter 模块的内部结构

一个经典的 **瓶颈 (Bottleneck)** 结构: 输入 (d维) → LayerNorm → Down-project (d→r) → 非线性激活 → Up-project (r→d) → 残差连接

- $r$ 是瓶颈维度（bottleneck size），通常 64~256
- 参数量：$2 \times d \times r$（两个线性层）
- 残差连接保证：当 Adapter 输出为 0 时，等价于原始模型

#### Adapter vs LoRA 的本质区别

| 维度 | Adapter | LoRA |
|------|---------|------|
| **位置** | 串联在层之间（sequential） | 并联在权重旁边（parallel） |
| **推理开销** | 有额外计算（过 Adapter 网络） | 可合并，零额外开销 |
| **非线性** | 有非线性激活函数 | 纯线性变换 |
| **参数量** | 略多（1%~5%） | 更少（0.1%~1%） |
| **灵活性** | 可以学更复杂的变换 | 受限于低秩线性变换 |

> **结论**：LoRA 因为可合并、零推理开销的优势，在当前实践中已基本取代了 Adapter。但理解 Adapter 的瓶颈设计思想很重要——它是所有 PEFT 方法的思想源头。

#### Adapter 的变体

| 变体 | 改进 |
|------|------|
| **AdapterFusion** | 同时插入多个 Adapter，学习融合不同任务的知识 |
| **Parallel Adapter** | 将 Adapter 从串联改为并联（类似 LoRA 的位置） |
| **Compacter** | 用 Kronecker 积参数化 Adapter，进一步压缩参数 |

---

### 8.4 Prefix Tuning

**论文**：Prefix-Tuning: Optimizing Continuous Prompts for Generation (2021, Stanford)

#### 核心思想

在每一层 Transformer 的 key 和 value 前面，拼接一组**可学习的虚拟 token**（前缀）：

```
正常输入:    [token1, token2, token3, ...]
                        ↓ 注意力计算
Prefix Tuning: [p1, p2, ..., pm, token1, token2, token3, ...]
                ↑ 可学习前缀     ↑ 原始输入（冻结的模型处理）
```

更准确地说，是在每一层 Transformer 的 K 和 V 矩阵前面拼接可学习的向量：

$$K' = [\underbrace{K_{prefix}}_{可学习}; K_{input}], \quad V' = [\underbrace{V_{prefix}}_{可学习}; V_{input}]$$

#### 与 Prompt Tuning 的区别

| 方法 | 作用位置 | 参数量 |
|------|---------|--------|
| **Prompt Tuning** | 只在输入 embedding 层加前缀 | 极少 |
| **Prefix Tuning** | 每一层都加前缀（K 和 V） | 更多但仍然很少 |
| **P-Tuning v2** | Prefix Tuning 的工程优化版 | 类似 Prefix Tuning |

#### 局限

- 占用序列长度（前缀 token 占了一部分 context window）
- 效果通常不如 LoRA
- 目前使用较少

---

### 8.5 各方法选择指南

```
你要微调一个大模型，该选什么？

                ┌── 有多卡 / 充足显存？
                │     是 → 全量微调（效果最好）
                │     否 ↓
                │
                ├── 单卡 A100 80G 或类似？
                │     是 → LoRA（推荐 r=16~64, 应用到全部线性层）
                │     否 ↓
                │
                ├── 单卡消费级 GPU (24G)?
                │     是 → QLoRA（4-bit 量化 + LoRA）
                │     否 ↓
                │
                └── 极端资源受限？
                      → Prompt Tuning / Prefix Tuning（效果打折扣）
```

---

**相关文档**：
- [预训练与后训练](预训练与后训练.md)
- [RLHF与PPO详解](RLHF与PPO详解.md)
- [DPO详解](DPO详解.md)
- [GRPO详解](GRPO详解.md)

[返回上级](README.md) | [返回总目录](../../README.md)

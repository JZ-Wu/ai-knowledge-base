# Qwen 系列技术报告深度解读

深度剖析阿里 Qwen 系列模型的核心技术：从 Qwen-1 到 Qwen3 的演进脉络，Qwen-2.5 的 18T tokens 训练策略与架构设计，Qwen2.5-Omni 的 Thinker-Talker 全模态架构，以及 Qwen3-VL 的 DeepStack 视觉注入与 Interleaved-MRoPE 创新。

> **更新日期**: 2026-04-13 | **最新覆盖**: Qwen3-VL (2025.9), Qwen2.5-Omni (2025.3)

---

## 一、演进脉络：Qwen 系列全景

### 1.1 时间线

```
Qwen-1 (2023.8)
  → 7B/14B，中文为核心优势
  → 词表 152064（超大，为中文 token 效率设计）

Qwen-1.5 (2024.2)
  → 全系列引入 GQA，规模覆盖 0.5B~110B
  → 对齐方式升级：引入 DPO

Qwen-2 (2024.6)
  → 训练数据 ~7T tokens，上下文 128K（YaRN 外推）
  → Qwen-2-72B：MMLU 84.2，超越 LLaMA-3-70B

Qwen-2.5 (2024.9) ★
  → 训练数据跃升至 18T tokens（~130× Chinchilla 最优）
  → 全系列 7 档规模：0.5B/1.5B/3B/7B/14B/32B/72B
  → 数学/代码专项增强，指令遵循大幅提升

Qwen-2.5-Coder (2024.9)
  → 代码特化，训练语料超 5.5T tokens
  → 32B 旗舰可在单卡 A100 部署

QwQ (2024.11)
  → 推理模型，基于 Qwen-2.5-72B + RL
  → MATH-500 达 90%+，对标 o1

Qwen2-VL (2024.10) → Qwen2.5-VL (2025.2)
  → NaViT 原生动态分辨率 + M-RoPE 3D 位置编码
  → OCRBench 866（开源最强），视频理解显著领先

Qwen2.5-Omni (2025.3) ★★ 新
  → 首个全模态端到端模型：文本/图像/音频/视频 → 文本+语音
  → Thinker-Talker 架构，TMRoPE 时间对齐
  → 7B / 3B 两档

Qwen3 (2025.4) + Qwen3-VL (2025.9) ★★ 新
  → 语言模型进入 MoE 时代：235B-A22B 旗舰
  → Qwen3-VL: DeepStack 多层视觉注入，Interleaved-MRoPE
  → Dense: 2B/4B/8B/32B | MoE: 30B-A3B, 235B-A22B
  → 每个规模均有 Instruct + Thinking 双版本
```

### 1.2 核心演进逻辑

Qwen 系列的演进揭示了阿里的产品策略——**不追求单一"最强模型"，而是覆盖全规模、全场景、全模态**：

1. **规模全覆盖**：从嵌入式 0.5B 到服务器端 235B MoE，每档充分训练
2. **专项特化**：语言、代码、数学、视觉、全模态分开维护，避免"平均主义"退化
3. **数据密度递增**：18T tokens 的 over-training 策略，用训练成本换推理效率
4. **MoE 化演进**：从 Qwen-2.5 的纯 Dense 到 Qwen3 的 Dense+MoE 双轨

---

## 二、Qwen-2.5：奠定基础的关键代际（精要版）

> 详细的 18T 训练策略、架构对比、后训练流程等内容已精简，保留核心要点。

### 2.1 架构要点

| 特性 | Qwen-2.5-7B | LLaMA-3-8B | 差异说明 |
|---|---|---|---|
| 词表大小 | 152,064 | 128,256 | Qwen 更大，中文 token 效率高 50%+ |
| KV 头数 | 4 | 8 | Qwen GQA 更激进，KV Cache 减半 |
| QK Bias | 保留 | 无 | 中文任务上轻微提升（~0.3% CEVAL）|
| Weight Tying | 否 | 是 | 大词表下不共享以避免 embedding 被拉伸 |
| 原生上下文 | 32K → 128K | 8K | YaRN 外推 |

**152K 词表的核心价值**：同一段 100 汉字文本，LLaMA 需要 180-250 tokens，Qwen 仅需 100-120 tokens。相同上下文窗口下处理更多中文内容，推理速度更快。

### 2.2 18T Tokens 训练策略

Qwen-2.5-7B 的训练量是 Chinchilla 最优的 **~130 倍**。核心逻辑：**训练是一次性成本，推理是持续成本**。对部署数百万次的 7B 模型，"多花训练成本换无需更大模型"极其划算。

**数据课程**：
- **阶段 1（~17T）**：通用网页+代码+科学文献，4K 上下文
- **阶段 2（~0.5T）**：长文档+代码仓库，上下文 32K→128K
- **阶段 3**：精选高质量数据，STEM 重点增强

**数据混合**：高质量数据（STEM、代码、教材）重复 5-10 次，通用网页最多 2-3 次，中英文比例约 3:7。合成数据（由 Qwen-2 生成）形成"蒸馏飞轮"。

### 2.3 后训练：SFT + DPO + RL

- **SFT**：100 万+ 指令-响应对，50-60% 为合成数据，用 reward model 过滤低质量样本
- **DPO**：降低有害输出，提升格式一致性和指令遵循
- **在线 RL**：对数学/代码用可验证奖励（答案正确性、测试通过率），是 QwQ 推理能力的雏形

### 2.4 Qwen-2.5-Coder 要点

- 代码训练语料 5.5T tokens，旗舰是 32B（非 72B，因单卡可部署）
- 核心创新：**仓库级上下文训练**——用 AST 分析跨文件引用，检索相关文件拼入 prompt
- **FIM 训练**：30-50% 样本转为 Fill-in-the-Middle 格式，支持 IDE 中间插入补全
- SWE-bench Verified ~43%，BigCodeBench 接近 GPT-4 水平

### 2.5 QwQ 推理模型要点

- 基于 Qwen-2.5-72B-Instruct 后训练，思考过程对用户可见（vs o1 隐藏）
- 训练：推理 SFT（长链 CoT 数据）+ RL 强化可验证奖励（GRPO/PPO 变体）
- 核心洞察：**推理时计算可替代参数量**——同样 72B，允许长链思考后 AIME 从 ~20% 提升到 ~50%

### 2.6 Qwen-MoE 设计哲学

- 配置：**60 专家选 4**，介于 Mixtral（8 选 2）和 DeepSeek（256 选 8）之间
- **共享专家**：1-2 个始终激活，存储通用知识；路由专家专注特化知识
- 激活比例 6.7%（高于 DeepSeek 的 3.1%），"宁可多激活保质量"的保守策略
- 负载均衡：辅助 loss + Expert Choice 路由（训练早期使用）

---

## 三、Qwen2-VL：视觉语言模型架构创新（精要版）

> Qwen2-VL 的 NaViT 和 M-RoPE 是理解 Qwen3-VL 的基础，此处保留架构核心。

### 3.1 核心架构：ViT-600M + Projector + LLM

```
输入图像 (任意分辨率)
  → Patch 14×14，NaViT 编码器（600M 参数，全局 attention）
  → 2×2 Spatial Merging（token 数÷4）
  → Linear 投影 → LLM 隐藏维度
  → M-RoPE 位置编码 (temporal × height × width)
  → Qwen2 LLM (2B / 7B / 72B)
```

### 3.2 NaViT（原生动态分辨率）

- **传统 ViT**：固定输入（如 448×448），非标准图像需 resize 或切 Tile
- **NaViT**：任意 H×W 输入，patches 做全局 self-attention，2D 位置编码按实际行列分配
- **Batch 处理**：Packing 方式将不同分辨率图像的 patches 打包到同一序列，用 attention mask 隔离
- 优势：无 Tile 隔离问题，GPU 利用率高

### 3.3 M-RoPE（3D 多模态位置编码）

将 RoPE 的频率空间三等分为 **temporal + height + width**：

- **文本**：t=h=w=递增序列 → 等价于标准 1D RoPE
- **图像**：t 固定，h/w 按 patch 行列 → 注意力感知空间距离和方向
- **视频**：t=帧号，h/w=帧内位置 → 可追踪"同一位置不同时间的变化"

### 3.4 关键结果

| Benchmark | Qwen2-VL-72B | InternVL 2.5-78B | GPT-4o |
|---|---|---|---|
| OCRBench | **866** | 822 | 736 |
| DocVQA | **96.5** | 95.1 | 92.8 |
| Video-MME | **~80** | 72.1 | 77.2 |
| MMMU | 70.2 | 70.1 | 69.9 |

OCR 领先归因于 M-RoPE 精确位置信息 + 阿里电商 OCR 数据；视频领先归因于 M-RoPE 时间维度 + 优酷等平台数据。

---

## 四、Qwen2.5-Omni：首个全模态端到端模型 ★

### 4.1 定位与意义

Qwen2.5-Omni（2025.3）是阿里推出的**首个全模态端到端模型**，能够处理文本、图像、音频、视频等任意组合的输入，并**同时生成文本和自然语音输出**。这标志着从"多模态理解"到"全模态交互"的跨越。

```
传统 VLM 路线（Qwen2-VL）:
  输入: 图像/视频 + 文本 → 输出: 文本
  → 单向：只能"看"和"读"，不能"听"和"说"

Qwen2.5-Omni 路线:
  输入: 图像 + 视频 + 音频 + 文本 (任意组合)
  → 输出: 文本 + 语音 (同时生成)
  → 全向：能"看"、"听"、"读"、"说"
```

**规格概览**：

| 版本 | Thinker | Talker | 音频编码器 | 视觉编码器 | 总参数 |
|---|---|---|---|---|---|
| 7B | 8.93B | 1.35B | 0.64B (Whisper-large-v3) | 0.68B (ViT) | ~10.73B |
| 3B | ~3B | 较小 | 同上 | 同上 | ~5B |

### 4.2 核心架构：Thinker-Talker

Qwen2.5-Omni 最重要的架构创新是 **Thinker-Talker 双组件设计**，将多模态理解与语音生成解耦。

```
┌────────────────────────────────────────────────────────────┐
│                  Qwen2.5-Omni 架构                          │
│                                                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ 输入层                                                │  │
│  │  视觉编码器 (ViT-675M, 来自 Qwen2.5-VL)              │  │
│  │  音频编码器 (Whisper-large-v3, 640M)                  │  │
│  │  → Block-wise Processing (分块流式处理)               │  │
│  └──────────┬───────────────────────────────┘            │
│             ↓                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Thinker ("大脑")                                      │  │
│  │  基于 Qwen2.5-7B 的 Transformer Decoder               │  │
│  │  → 接收文本 + 视觉特征 + 音频特征                     │  │
│  │  → 输出: 文本 tokens + 高层语义表征 (hidden states)    │  │
│  │  → TMRoPE 编码多模态时空位置                          │  │
│  └──────────┬───────────────┬──────────────┘              │
│             ↓ text tokens   ↓ hidden representations       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Talker ("嘴巴")                                       │  │
│  │  双轨自回归 Transformer Decoder (1.35B)                │  │
│  │  → 输入: Thinker 的 hidden states + 文本 tokens        │  │
│  │  → 输出: 离散语音 tokens (流式)                        │  │
│  │  → 不需要词级别时间戳对齐                              │  │
│  └──────────┬──────────────────────────────┘              │
│             ↓ speech tokens                                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ Token2Wav (语音合成)                                  │  │
│  │  Sliding-Window DiT (Flow-Matching) → mel-spectrogram │  │
│  │  BigVGAN 声码器 → 波形                                │  │
│  │  → 低延迟流式输出                                     │  │
│  └──────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

**为什么要分成 Thinker 和 Talker？**

```
方案 A（单模型直出语音）:
  一个模型同时输出文本 token 和语音 token
  问题: 跨模态干扰——语音的韵律/节奏会影响文本的语义质量
        训练目标冲突——文本 loss 和语音 loss 梯度互相拉扯

方案 B（Thinker-Talker 解耦）: ← Qwen2.5-Omni 选择
  Thinker 专注理解和推理，输出高层语义
  Talker 专注语音生成，消费 Thinker 的表征
  优点:
    1. 文本质量不受语音生成影响
    2. Talker 可以独立优化语音自然度
    3. 不需要语音时可跳过 Talker，节省计算
    4. 语音和文本同步但互不干扰
```

### 4.3 TMRoPE：时间对齐的多模态位置编码

TMRoPE (Time-aligned Multimodal RoPE) 是 M-RoPE 的进化版，核心改进是**音频与视频的精确时间对齐**。

```
M-RoPE (Qwen2-VL) 的局限:
  只处理 视觉+文本，没有音频维度
  视频帧的时间编码粒度较粗

TMRoPE 的改进:
  音频时间粒度: 每个 temporal ID 代表 40ms 间隔
  视频时间粒度: 每帧按 40ms 粒度动态调整
  
  关键: 音频和视频按交错方式排列，时间戳严格同步

  示例（处理一段带音频的视频）:
    视频帧 0 (t=0~40ms):  patches → t=0, h=行, w=列
    音频片段 0 (t=0~40ms): tokens → t=0, h=0, w=0
    视频帧 1 (t=40~80ms): patches → t=1, h=行, w=列
    音频片段 1 (t=40~80ms): tokens → t=1, h=0, w=0
    ...
    
  效果: 模型天然理解"这段语音是在说画面中正在发生的事"
        音频和视觉信息在注意力计算中通过相同的 temporal ID 关联
```

### 4.4 Block-wise Processing：流式处理长序列

处理长视频+音频时，token 序列可能非常长。Qwen2.5-Omni 用分块策略解决：

```
传统方式: 等整个视频/音频处理完再输入 LLM
  → 延迟高，内存占用大

Block-wise Processing:
  1. 视觉编码器和音频编码器将输入分块处理
  2. 每个 block 独立编码，流式送入 Thinker
  3. Thinker 可以在接收部分输入时就开始生成
  4. Talker 进一步流式生成语音 tokens
  
  → 首包延迟大幅降低
  → 支持实时对话场景
```

### 4.5 Sliding-Window DiT：低延迟语音合成

Talker 输出的离散语音 tokens 需要转换为实际波形：

```
语音 tokens → Sliding-Window DiT (Flow-Matching) → mel-spectrogram → BigVGAN → 波形

Sliding-Window DiT 的关键设计:
  注意力感受野: lookback 2 blocks + 当前 block + lookahead 1 block
  → 总共只看 4 个 block，而非全部历史
  → 目的: 控制延迟，不等待全部语音 tokens 生成完毕
  → 代价: 无法利用超长距离的韵律信息（实际影响很小）
```

### 4.6 训练策略

**预训练数据规模**：

| 模态 | 数据量 |
|---|---|
| 文本 | 18T tokens（与 Qwen-2.5 共享） |
| 图像+视频 | 800B tokens |
| 音频 | 300B tokens |
| 带音频的视频 | 100B tokens |

**三阶段预训练**：

```
阶段 1: 编码器对齐
  训练: 视觉/音频编码器 + adapter（LLM 冻结）
  数据: 音频-文本对 + 图像-文本对
  目标: 让多模态特征与语言空间对齐

阶段 2: 全参数联合训练
  训练: 全部参数解冻
  数据: 1.2T tokens 多模态数据，序列长度 8K
  目标: 视觉/音频/语言协同优化

阶段 3: 长序列多模态训练
  训练: 全部参数
  序列长度: 32K
  目标: 长视频、长音频理解
```

**后训练**：
- SFT + DPO
- 语音质量专项优化：用 WER（词错误率）和标点反馈过滤低质量语音输出，减少发音错误和注意力错位

### 4.7 Benchmark 表现

**全模态理解（OmniBench）**——唯一同时处理语音+声音+音乐的端到端模型：

| 模型 | Speech | Sound | Music | 平均 |
|---|---|---|---|---|
| **Qwen2.5-Omni-7B** | **55.25%** | **60.00%** | **52.83%** | **56.13%** |
| Gemini-1.5-Pro | - | - | - | 42.91% |

**与同规模 Qwen2.5-VL-7B 的视觉能力对比**：

| Benchmark | Qwen2.5-Omni-7B | Qwen2.5-VL-7B | 说明 |
|---|---|---|---|
| MMMU-val | 59.2% | 58.6% | 全模态模型未牺牲视觉能力 |
| MMStar | 64.0% | 64.0% | 完全持平 |
| Video-MME (有字幕) | **72.4%** | - | 视频理解 SOTA |

**语音生成质量**：

| 测试集 | WER |
|---|---|
| seed-tts-eval test-zh | 1.42% |
| seed-tts-eval test-en | 2.33% |

语音生成质量优于 MaskGCT 和 CosyVoice 2 等专用 TTS 模型。

**关键洞察**：Qwen2.5-Omni 证明了**全模态端到端模型不一定要牺牲单模态性能**。通过 Thinker-Talker 解耦，视觉理解能力与同规模的 Qwen2.5-VL 基本持平，同时额外获得了音频理解和语音生成能力。

---

## 五、Qwen3-VL：新一代视觉语言模型 ★

### 5.1 定位与规模

Qwen3-VL（2025.9）是 Qwen 视觉语言模型的第三代，也是首次在 VLM 中引入 **MoE 架构**和 **Thinking（推理增强）模式**的版本。

**发布时间线**：
- 2025.9.23: 235B-A22B Instruct/Thinking（旗舰）
- 2025.10.4: 30B-A3B Instruct/Thinking（轻量 MoE）
- 2025.10.15: 4B / 8B Instruct/Thinking
- 2025.10.21: 2B / 32B Instruct/Thinking

**全系列规格**：

| 模型 | 类型 | 总参数 | 激活参数 | 版本 |
|---|---|---|---|---|
| Qwen3-VL-2B | Dense | 2B | 2B | Instruct / Thinking |
| Qwen3-VL-4B | Dense | 4B | 4B | Instruct / Thinking |
| Qwen3-VL-8B | Dense | 8B | 8B | Instruct / Thinking |
| Qwen3-VL-32B | Dense | 32B | 32B | Instruct / Thinking |
| Qwen3-VL-30B-A3B | MoE | 30B | 3B | Instruct / Thinking |
| Qwen3-VL-235B-A22B | MoE | 235B | 22B | Instruct / Thinking |

关键变化：**每个规模都有 Instruct 和 Thinking 双版本**。Thinking 版本经过额外的长链推理 (CoT) 训练，在数学和复杂推理任务上显著更强。

### 5.2 架构演进：从 Qwen2-VL 到 Qwen3-VL

```
Qwen2-VL 架构:
  ViT-600M (patch 14×14) → 2×2 Merging → Linear → M-RoPE → LLM
  视觉特征只在 LLM 输入层注入（单点注入）

Qwen3-VL 架构:
  SigLIP-2 ViT (patch 16×16) → 2×2 Merging → MLP Merger → LLM
  + DeepStack: 多层视觉特征注入 LLM 的前 3 层（多点注入）
  + Interleaved-MRoPE: 频率全维度均匀分配
  + 文本时间戳替代 T-RoPE
```

**架构对比**：

| 维度 | Qwen2-VL | Qwen3-VL | 改进原因 |
|---|---|---|---|
| 视觉编码器 | ViT-600M | SigLIP-2 ViT | SigLIP-2 在细粒度识别上更强 |
| Patch 大小 | 14×14 | 16×16 | 配合 2×2 merging 实现 32× 压缩 |
| 压缩比 | 28× (14×2) | 32× (16×2) | token 更少，LLM 负担更轻 |
| 视觉注入 | 输入层单点 | DeepStack 多层 | 保留低层细粒度 + 高层语义 |
| 位置编码 | M-RoPE | Interleaved-MRoPE | 长视频推理更稳定 |
| 时间定位 | T-RoPE | 文本时间戳 `<3.8s>` | 更直观，效果更好 |
| LLM 架构 | Dense only | Dense + MoE | MoE 旗舰大幅提升能力上限 |
| 推理模式 | 无 | Thinking 模式 | 复杂任务深度推理 |
| 上下文 | 128K | 256K → 1M | 超长视频理解 |
| OCR 语种 | ~10 | 32 | 国际化覆盖 |

### 5.3 DeepStack：多层视觉特征注入

DeepStack 是 Qwen3-VL 最重要的架构创新，解决了传统 VLM "单点注入"的信息瓶颈。

**问题：为什么单点注入不够？**

```
传统 VLM（包括 Qwen2-VL）:
  ViT 输出最终层特征 → 投影 → 作为 LLM 第 0 层的输入 tokens
  
  问题:
  1. ViT 最终层特征是高度抽象的语义表征
     → 丢失了低层的细粒度视觉细节（边缘、纹理、小文字）
  2. LLM 需要在第 0 层就"一次性消化"所有视觉信息
     → 对 LLM 早期层的负担过重
  3. 低层视觉信息经过 LLM 多层传播后衰减严重
     → 在需要精细视觉的任务（OCR、图表）上性能受限
```

**DeepStack 的设计**：

```
┌──────────────────────────────────────────────────────┐
│                  DeepStack 示意图                      │
│                                                      │
│  SigLIP-2 ViT                    Qwen3 LLM           │
│  ┌──────────┐                   ┌──────────────┐     │
│  │ Layer 1-8 │ ─── MLP_1 ───→  │ LLM Layer 0  │     │
│  │ (低层特征) │   (底层细节)     │ + 视觉底层    │     │
│  ├──────────┤                   ├──────────────┤     │
│  │Layer 9-16│ ─── MLP_2 ───→  │ LLM Layer 1  │     │
│  │ (中层特征) │   (结构信息)     │ + 视觉中层    │     │
│  ├──────────┤                   ├──────────────┤     │
│  │Layer 17-N│ ─── MLP_3 ───→  │ LLM Layer 2  │     │
│  │ (高层特征) │   (语义抽象)     │ + 视觉高层    │     │
│  └──────────┘                   ├──────────────┤     │
│                                 │ LLM Layer 3  │     │
│                                 │ ...          │     │
│                                 │ LLM Layer N  │     │
│                                 └──────────────┘     │
└──────────────────────────────────────────────────────┘

具体实现:
  1. 从 ViT 的多个中间层提取特征（而非只用最终层）
  2. 每组中间层特征通过独立的 MLP Merger 投影到 LLM 维度
  3. 投影后的特征注入 LLM 的前 3 层（具体是加法还是拼接取决于实现）
  4. LLM 的后续层正常处理文本+视觉融合特征

效果:
  - LLM Layer 0 获得 ViT 低层特征 → 保留边缘、纹理、小字体信息
  - LLM Layer 1 获得 ViT 中层特征 → 保留物体结构、布局信息
  - LLM Layer 2 获得 ViT 高层特征 → 保留语义抽象信息
  - 额外计算成本极小（只是几个 MLP + 几层的特征加法）
```

**DeepStack vs 竞品的视觉注入方式**：

| 方式 | 代表模型 | 注入层 | 视觉层级 | 缺点 |
|---|---|---|---|---|
| 单点注入 | Qwen2-VL, LLaVA | LLM Layer 0 | 仅最终层 | 丢失细粒度信息 |
| 交叉注意力 | Flamingo | 每隔 N 层 | 仅最终层 | 计算量大，仍只用最终层 |
| **DeepStack** | **Qwen3-VL** | LLM 前 3 层 | **多层级** | 计算成本极低 |

### 5.4 Interleaved-MRoPE：改进的 3D 位置编码

```
M-RoPE (Qwen2-VL):
  频率维度分组: D = D_t | D_h | D_w (三段连续分配)
  例如 D=128: temporal 用 dim 0-41, height 用 dim 42-84, width 用 dim 85-127
  
  问题: 不同维度使用不同频率范围
        temporal 维度用低频（dim 0-41 频率高）
        width 维度用高频（dim 85-127 频率低）
        → 长视频中 temporal 维度的位置区分能力不均匀

Interleaved-MRoPE (Qwen3-VL):
  频率维度交错分配: t, h, w, t, h, w, t, h, w, ...
  配置: mrope_section = [24, 20, 20]
  
  每个维度均匀地使用从高到低的所有频率范围
  → 三个维度的表达能力更均衡
  → 长视频推理时 temporal 位置编码更稳定
  → 空间位置编码精度也有提升
```

### 5.5 文本时间戳替代 T-RoPE

```
Qwen2-VL 的时间定位方式:
  通过 M-RoPE 的 temporal 维度隐式编码帧号
  → 模型需要从 RoPE 的旋转角度"反推"时间信息
  → 在需要精确时间定位时不够直观

Qwen3-VL 的改进:
  在视频帧前插入显式文本时间戳: <3.8 seconds>
  → 模型直接从文本中理解时间信息
  → 可以用自然语言表达时间（"在第 3.8 秒时..."）
  → 时间定位更精确，尤其在长视频中

效果:
  长视频 needle-in-haystack 测试:
    2 小时视频中定位特定事件 → 99.5% 准确率
```

### 5.6 训练流程

**四阶段预训练**：

```
S0 (热启动):
  训练: 仅 Merger 层
  数据: ~67B tokens
  上下文: 8K
  目标: 初步对齐视觉和语言空间

S1 (全参数训练):
  训练: 全部参数解冻
  数据: ~1T tokens
  上下文: 8K
  目标: 深度视觉-语言融合

S2 (长上下文训练):
  训练: 全部参数
  数据: ~1T tokens
  上下文: 32K
  目标: 长文档、多图理解

S3 (超长上下文):
  训练: 全部参数
  数据: ~100B tokens
  上下文: 262K
  目标: 超长视频、大规模文档
```

**数据规模**：
- 网页爬取数据 + ~300 万 PDF 文档 + 6000 万+ STEM 任务
- 训练规模高达 10,000 GPU

**目标函数平衡**：使用 **平方根重加权（Square-root Reweighting）** 平衡纯文本和多模态学习目标，避免多模态训练损害文本能力。

**后训练（3 阶段）**：
1. **SFT**：长链 CoT 数据的监督微调
2. **知识蒸馏**：从更强模型蒸馏
3. **RL**：GRPO 等算法强化推理

Thinking 版本额外接受推理增强训练，走独立的后训练路径。

### 5.7 Benchmark 表现

**Qwen3-VL-235B-A22B Instruct vs Thinking vs 竞品**：

| Benchmark | Instruct | Thinking | GPT-4o | Gemini-2.5-Pro | 说明 |
|---|---|---|---|---|---|
| MMMU | 78.7 | **80.6** | 69.9 | - | 大幅领先 GPT-4o |
| MMMU-Pro | 68.1 | **69.3** | - | - | |
| MathVista | 84.9 | **85.8** | 63.8 | 81.3* | 超越 GPT-5 |
| MathVision | 66.5 | **74.6** | - | 73.3 | Thinking 版追平 Gemini |
| DocVQA | **97.1** | 96.5 | 92.8 | - | 文档理解 SOTA |
| OCRBench | **92.0** | 87.5 | 73.6 | - | 远超 GPT-4o |
| MMBench | 89.9 | **90.6** | 83.4 | - | |
| Video-MME (w/o sub) | **79.2** | 79.0 | 77.2 | - | 视频理解继续领先 |
| ScreenSpot | **95.4** | - | - | - | GUI 自动化 |
| OSWorld | **66.7** | - | - | - | 操作系统级 Agent |
| AIME 2025 | - | **89.7** | - | - | Thinking 版数学推理极强 |

**各规模关键定位**：

```
Qwen3-VL-2B:    端侧部署，移动设备上的视觉助手
Qwen3-VL-4B:    轻量级本地部署，中等视觉任务
Qwen3-VL-8B:    主力轻量级，单卡 24GB 运行
Qwen3-VL-30B-A3B: MoE 轻量旗舰，激活仅 3B 但总容量 30B
                   → 适合追求低推理成本但高能力的场景
Qwen3-VL-32B:   Dense 高质量本地部署，单卡 A100 80GB
Qwen3-VL-235B-A22B: 旗舰，激活 22B 但总容量 235B
                     → 开源 VLM 能力天花板
```

### 5.8 Thinking 模式的意义

Qwen3-VL 的 Thinking 版本将 QwQ 的推理增强思路引入视觉语言模型：

```
Instruct 模式:
  用户: [图片] 这道数学题怎么解？
  模型: 答案是 42。

Thinking 模式:
  用户: [图片] 这道数学题怎么解？
  模型: <think>
    首先识别图中的数学表达式...
    这是一个积分问题，被积函数是...
    我先尝试换元法... 不对，应该用分部积分...
    [长链推理过程]
  </think>
  答案是 42，推导过程如下...

效果对比 (MathVision):
  Instruct: 66.5%
  Thinking: 74.6% (+8.1%)
  → 允许模型"思考"显著提升视觉数学推理
```

---

## 六、路线对比与技术总结

### 6.1 Qwen VLM 三代架构对比

| 维度 | Qwen2-VL | Qwen2.5-Omni | Qwen3-VL |
|---|---|---|---|
| 定位 | 视觉语言理解 | 全模态交互 | 视觉语言理解+推理 |
| 输入 | 图像/视频+文本 | 图像/视频/音频+文本 | 图像/视频+文本 |
| 输出 | 文本 | 文本+语音 | 文本 |
| 视觉编码器 | ViT-600M | ViT-675M | SigLIP-2 ViT |
| 视觉注入 | 单点（LLM 输入层）| 单点 | **DeepStack 多层** |
| 位置编码 | M-RoPE | TMRoPE | Interleaved-MRoPE |
| LLM | Dense only | Dense (Thinker) | Dense + MoE |
| 推理模式 | 无 | 无 | Thinking 模式 |
| 上下文 | 128K | 32K | 256K → 1M |

### 6.2 与 InternVL 路线的最新对比

```
Qwen3-VL 路线 — "高效编码 + DeepStack + MoE LLM":
  视觉编码器: SigLIP-2 (中等规模)
  核心创新: DeepStack 多层注入 + Interleaved-MRoPE
  动态分辨率: NaViT 原生 (全局 attention)
  LLM: MoE 旗舰 (235B-A22B)
  
  优点: 推理效率高 (MoE 稀疏激活), 视频理解强, 位置感知精确
  适合: 视频理解, OCR, GUI Agent, 需要 Thinking 的复杂推理

InternVL 路线 — "大编码器 + 工程化数据":
  视觉编码器: InternViT-6B (重量级)
  核心创新: 细粒度视觉表征
  动态分辨率: Dynamic Tiling
  LLM: Dense
  
  优点: 视觉特征质量高, 细粒度任务强
  适合: 文档理解, 数学图像, 图表分析

共识:
  数据工程 > 架构选择 (约 70% vs 30% 的性能贡献)
  开源 VLM 已全面超越 GPT-4o，正在追赶 GPT-5 和 Gemini 2.5 Pro
```

### 6.3 Qwen 系列对行业的关键贡献

1. **Thinker-Talker 架构**：证明全模态端到端模型可以不牺牲单模态性能，为"AI 助手能说话"提供了工程可行的路径
2. **DeepStack**：以极小的额外成本实现多层级视觉注入，可能成为 VLM 的新标准做法
3. **VLM + Thinking**：首次在视觉语言模型中引入推理增强，MathVision 提升 8+ 个百分点
4. **MoE 进入 VLM**：235B-A22B 证明 MoE 在多模态场景同样有效，稀疏激活控制推理成本
5. **全规模 + 全模态**：从 2B Dense 到 235B MoE，从纯语言到全模态，产品线覆盖最全

---

**相关文档**：
- [InternVL系列技术解读](InternVL系列技术解读.md)
- [MoE详解](../基础理论/MoE详解.md)
- [主流模型总览](主流模型总览.md)

[返回上级](README.md) | [返回总目录](../../README.md)

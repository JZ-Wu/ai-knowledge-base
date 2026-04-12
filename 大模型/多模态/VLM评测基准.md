# VLM 评测基准

理解 VLM Benchmark 的方法论——评什么、怎么评、怎么用，以及主流 benchmark 速查。

> **版本说明**：本文内容截至 2026 年 4 月，benchmark 数据和模型成绩会持续更新。

---

## 一、为什么需要 Benchmark？

在 VLM 领域，模型能力极其多样——识别物体、阅读文字、理解图表、推理空间关系、抵抗幻觉……单靠 "看 demo" 无法系统评估。Benchmark 的核心价值：

1. **标准化对比**：用统一的题目和评分规则，让不同模型在同一把尺子下比较
2. **能力拆解**：把 "多模态理解" 拆成细粒度维度（OCR、空间、推理等），暴露模型短板
3. **追踪进步**：同一个 benchmark 上分数的提升，直观反映技术迭代效果
4. **指导优化**：哪个维度分低，就在对应数据/训练策略上发力

**但也要注意 benchmark 的局限**：
- 刷分 ≠ 真实好用（benchmark 覆盖不了所有真实场景）
- 数据污染问题（训练集可能泄漏 benchmark 题目）
- 评测方式本身有偏差（多选题 vs 开放问答能测的东西不同）

---

## 二、Benchmark 在评什么？（能力维度）

VLM 的能力可以拆成几个层次，所有 benchmark 本质上都在测其中一个或多个：

| 能力层次 | 具体内容 | 典型 Benchmark |
|---------|---------|---------------|
| **底层视觉感知** | 深度、大小、空间关系、视觉匹配——人一眼就能看出来的东西 | BLINK |
| **粗粒度感知** | 场景识别、物体存在性、主体识别 | MME (感知部分) |
| **细粒度感知** | 属性识别、计数、位置关系、OCR/文字阅读 | TextVQA、DocVQA |
| **结构化理解** | 图表解读、文档版面理解、科学示意图 | ChartQA、AI2D |
| **推理** | 逻辑推理、数学推理、常识推理、跨信息关联 | MathVista、MMMU |
| **时序理解** | 动作识别、事件顺序、因果推理（视频专属） | MVBench、Video-MME |
| **可靠性** | 幻觉抑制——不"脑补"图中没有的东西 | POPE、HallusionBench |

**关键洞察**：从下往上，越底层的能力人类越轻松、模型越挣扎（BLINK 人类 >95% vs 模型 ~60%）；越高层的推理能力，模型反而可以靠语言能力弥补。这暴露了当前 VLM 的根本问题：**语言推理强，视觉感知弱**。

---

## 三、Benchmark 怎么评？（设计方法论）

### 3.1 题目形式

| 形式 | 优点 | 缺点 | 代表 |
|------|-----|------|------|
| **是非题 (Yes/No)** | 成本最低、出结果快 | 信息量少，模型易蒙 | MME、POPE |
| **多选题 (A/B/C/D)** | 客观评分、可自动化 | 存在猜对概率，且不考察表达能力 | MMBench、SEED-Bench |
| **开放式问答** | 最接近真实使用 | 评分难，需要 GPT-4 等辅助打分，有偏差 | MM-Vet |

### 3.2 防作弊 / 防刷分设计

这是 benchmark 设计中最关键的"技术含量"所在：

| 策略 | 原理 | 出处 |
|------|------|------|
| **CircularEval** | 同一题打乱选项测多次，全对才算对 → 过滤蒙对 | MMBench |
| **正反配对** | 每题一正一反两个 Yes/No 问题，都对才得分 → 防止全答 Yes | MME |
| **视觉不可或缺筛选** | 用纯文本 LLM 过滤掉"不看图也能答"的题 → 确保真正测视觉 | MMStar |
| **难度递增采样** | 从随机→高频→高共现，逐级增加干扰 → 精确量化幻觉程度 | POPE |
| **多选项扩展** | 4 选 1 变 10 选 1 → 降低猜对概率 | MMMU-Pro |

### 3.3 评价指标

| 指标 | 含义 | 适用场景 |
|------|------|---------|
| **Accuracy** | 直接准确率 | 选择题（MMBench、MMMU 等大多数） |
| **VQA Accuracy** | 允许多个标准答案，取最高匹配 | TextVQA 等开放式 VQA |
| **ANLS** | 容忍 OCR 小错误（基于编辑距离） | DocVQA（文档场景） |
| **Relaxed Accuracy** | 数值答案允许 5% 误差 | ChartQA（图表场景） |
| **GPT-4 评分** | 用 GPT-4 对自由回答打分 (0-1) | MM-Vet 等开放式问答 |
| **F1-Score** | 衡量检索的精确度+召回率 | MMLongBench-Doc（长文档） |
| **Score（分制）** | 每子任务满分 200，总分 2800 | MME（各子任务单独计分） |

### 3.4 综合 vs 专项的设计哲学

- **综合型**（MMBench、MEGA-Bench）：广而浅，给"体检报告"，快速定位弱项
- **专项型**（POPE、MathVista）：窄而深，精准测量单一能力的上限
- **最佳实践**：综合型找到弱项 → 用专项型深入诊断

---

## 四、Benchmark 的数据格式与使用方法

了解方法论之后，还需要知道：benchmark 数据实际长什么样？怎么拿来跑评测？

### 4.1 数据存储格式

大多数 VLM benchmark 遵循类似的数据组织方式：

```
benchmark_name/
├── images/          # 或 videos/，存放原始图片/视频
│   ├── 000001.jpg
│   ├── 000002.png
│   └── ...
├── questions.json   # 或 .jsonl / .tsv / .parquet，题目和标注
└── README.md        # 数据说明
```

**题目标注的典型 JSON 格式**（以多选题为例）：

```json
{
  "question_id": 1,
  "image": "images/000001.jpg",
  "question": "What color is the car in the image?",
  "choices": ["Red", "Blue", "Green", "White"],
  "answer": "A",
  "category": "fine-grained_perception",
  "sub_category": "color_recognition"
}
```

**开放式问答的格式**：

```json
{
  "question_id": 42,
  "image": "images/000042.jpg",
  "question": "Describe the relationship between the two people in the image.",
  "answer": "A teacher is explaining a math problem to a student at the whiteboard.",
  "capability": ["recognition", "spatial_awareness", "language_generation"]
}
```

**视频 benchmark 的格式**（额外包含时间信息）：

```json
{
  "video": "videos/cooking_003.mp4",
  "start_time": 10.0,
  "end_time": 30.0,
  "question": "What did the person do after picking up the knife?",
  "choices": ["Cut vegetables", "Washed hands", "Opened the fridge", "Turned on the stove"],
  "answer": "A"
}
```

> **注**：不同 benchmark 格式不完全统一，以上是最常见的模式。实际使用时需阅读各 benchmark 的 README。

### 4.2 数据获取方式

| 来源 | 说明 |
|------|------|
| **HuggingFace Datasets** | 大多数 benchmark 都上传了 HuggingFace，`datasets.load_dataset("benchmark_name")` 即可加载 |
| **官方 GitHub** | 部分 benchmark 通过 GitHub 发布，附带评测脚本 |
| **OpenCompass 内置** | VLMEvalKit 和 lmms-eval 框架内置了主流 benchmark 的下载和解析逻辑，无需手动处理 |

### 4.3 评测流程（怎么用 Benchmark 跑评测）

一次完整的 VLM 评测流程如下：

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ 1. 准备数据   │ →  │ 2. 模型推理   │ →  │ 3. 答案匹配   │ →  │ 4. 汇总打分   │
│ 下载benchmark │    │ 逐题送入模型  │    │ 提取+对比答案  │    │ 分维度统计    │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
```

**Step 1：准备数据**
- 下载 benchmark 数据（图片 + 题目标注）
- 按 benchmark 要求构造 prompt（如多选题需要把选项拼进 prompt）

**Step 2：模型推理**
- 将 `(image, prompt)` 逐题送入模型，收集模型输出
- 需要注意 batch size、图片分辨率、max_tokens 等配置

**Step 3：答案匹配（Answer Extraction）**

这是评测中最容易出错的环节。模型的回答是自由文本（如 `"The answer is A, because..."`），需要从中提取出答案：

- **多选题**：用正则匹配提取选项字母（A/B/C/D）。如果提取失败，部分框架会用 GPT 辅助提取
- **是非题**：匹配 "Yes"/"No"
- **开放式问答**：直接拿完整回答送去评分（GPT-4 评分 或 指标计算）
- **数值题**：提取数字，与标准答案对比（可能允许误差）

**Step 4：汇总打分**
- 按 benchmark 的计分规则计算分数
- 按能力维度分组统计（如 MMBench 的 20+ 子维度各自出分）
- 输出成绩表 / 雷达图

### 4.4 用评测框架一键跑评（推荐）

手动走上面四步很繁琐，实际工作中推荐用现成的评测框架：

**VLMEvalKit（推荐，国内主流）**：

```bash
# 安装
pip install vlmeval

# 一键评测：模型 + benchmark 名即可
python run.py --model qwen2_5_vl_72b --data MMBench_DEV_EN MMStar POPE

# 输出结果在 outputs/ 目录下，包含分维度成绩
```

**lmms-eval（国际主流）**：

```bash
# 安装
pip install lmms-eval

# 评测
python -m lmms_eval \
  --model llava \
  --model_args pretrained=llava-hf/llava-v1.6-34b-hf \
  --tasks mmbench,mmstar,pope \
  --batch_size 1
```

**这两个框架帮你做了**：
- 自动下载 benchmark 数据
- 自动构造 prompt（按各 benchmark 的官方格式）
- 自动做答案提取和匹配
- 自动按官方规则计分
- 输出标准化的成绩表

### 4.5 评测中的常见坑

| 问题 | 说明 |
|------|------|
| **Prompt 格式不一致** | 不同框架构造的 prompt 略有不同（如选项前加不加序号），可能导致同一模型分数差 2-3 分 |
| **答案提取失败** | 模型输出格式不规范（如回答了一大段但没明确选择），导致提取不到答案，被记为错误 |
| **图片预处理差异** | 不同模型对图片分辨率、裁剪方式的要求不同，用错了影响成绩 |
| **随机性** | 部分模型有 temperature > 0 的采样随机性，多次评测结果可能不同 |
| **版本差异** | 同一 benchmark 可能有不同版本（如 MMBench v1.0 vs v1.1），需确认对齐版本 |

> **建议**：如果要对比多个模型，务必用**同一个评测框架 + 同一版本 benchmark + 同一评测配置**，否则结果不可比。

---

## 五、主流 VLM 评测成绩对比

### 5.1 经典 Benchmark 成绩 (2025-2026)

| 模型 | MMBench | TextVQA | DocVQA | MathVista | POPE | ChartQA |
|------|---------|---------|--------|-----------|------|---------|
| **闭源模型** | | | | | | |
| GPT-4.1 (OpenAI, 2025) | ~86 | ~82 | ~95 | ~72 | ~90 | ~89 |
| o3 (OpenAI, 2025) | ~85 | ~80 | ~94 | ~78 | ~89 | ~88 |
| Claude 4 Sonnet (Anthropic, 2025) | ~85 | ~81 | ~94 | ~71 | ~90 | ~87 |
| Gemini 2.5 Pro (Google, 2025) | ~87 | ~84 | ~96 | ~75 | ~91 | ~90 |
| Gemini 2.5 Flash (Google, 2025) | ~84 | ~81 | ~93 | ~70 | ~89 | ~86 |
| **开源模型** | | | | | | |
| Qwen2.5-VL-72B (阿里, 2025) | ~86 | ~84 | ~96 | ~74 | ~91 | ~90 |
| InternVL3-78B (上海 AI Lab, 2025) | ~86 | ~82 | ~95 | ~73 | ~91 | ~88 |
| LLaVA-NeXT-72B (2025) | ~83 | ~80 | ~93 | ~70 | ~89 | ~85 |
| DeepSeek-VL2 (深度求索, 2025) | ~84 | ~81 | ~94 | ~72 | ~90 | ~87 |

### 5.2 新一代 Benchmark 成绩

| 模型 | MMMU | MMMU-Pro | MMStar | BLINK | MMLongBench-Doc |
|------|------|----------|--------|-------|-----------------|
| **闭源模型** | | | | | |
| GPT-4.1 | ~70 | ~55 | ~67 | ~60 | ~50 |
| o3 | ~74 | ~60 | ~70 | ~58 | ~48 |
| Claude 4 Sonnet | ~71 | ~56 | ~68 | ~59 | ~49 |
| Gemini 2.5 Pro | ~73 | ~58 | ~70 | ~62 | ~55 |
| **开源模型** | | | | | |
| Qwen2.5-VL-72B | ~71 | ~55 | ~68 | ~57 | ~47 |
| InternVL3-78B | ~70 | ~54 | ~67 | ~56 | ~46 |
| DeepSeek-VL2 | ~69 | ~53 | ~66 | ~55 | ~44 |

> **注**：成绩会随版本更新变化，此处为近似值，供参考。各模型可能使用不同的评测配置。o3 为推理模型（Reasoning Model），在数学推理类任务上优势明显。

### 5.3 历史成绩对比 (2024 早期，留作参考)

| 模型 | MMBench | TextVQA | DocVQA | MathVista | POPE | ChartQA |
|------|---------|---------|--------|-----------|------|---------|
| GPT-4o (2024) | ~83 | ~77 | ~92 | ~63 | ~87 | ~85 |
| Claude 3.5 Sonnet (2024) | ~79 | ~74 | ~90 | ~62 | ~86 | ~82 |
| Gemini 1.5 Pro (2024) | ~81 | ~78 | ~93 | ~63 | ~87 | ~84 |
| InternVL2.5-78B (2024) | ~82 | ~77 | ~93 | ~68 | ~89 | ~84 |
| Qwen-VL2-72B (2024) | ~83 | ~79 | ~94 | ~70 | ~88 | ~86 |

### 5.4 从成绩中读出的趋势 (2025-2026)

- **开源全面追平闭源**：Qwen2.5-VL-72B、InternVL3 在多项 benchmark 上达到甚至超过 GPT-4.1 和 Claude 4 Sonnet，开闭源差距基本消失
- **Gemini 2.5 Pro 多项领先**：Google 在多模态方向持续发力，Gemini 2.5 Pro 在 DocVQA、ChartQA、长文档等场景表现突出
- **推理模型崛起**：o3 等推理模型在 MathVista、MMMU-Pro 等需要深度推理的 benchmark 上显著领先传统模型
- **数学推理显著进步**：MathVista 从 2024 年的 ~63 提升到 2026 年的 ~75，但 MMMU-Pro 仍是所有模型的难点（<60%）
- **底层视觉感知仍是短板**：BLINK 上最强模型仍只有 ~62%，远低于人类 >95%，说明"能理解但不能精确感知"的根本问题尚未解决
- **长文档是新前沿**：MMLongBench-Doc 上所有模型均低于 55%，长上下文多模态理解仍有巨大提升空间
- **中小尺寸模型快速进步**：7B-13B 级模型（如 Qwen2.5-VL-7B、InternVL3-8B）在 2026 年已能达到 2024 年 70B 模型的水平

---

## 六、评测榜单与工具

### 6.1 综合排行榜

| 榜单 | 链接 | 特点 |
|------|------|------|
| **OpenCompass（司南）** | https://opencompass.org.cn | 国内最全面的多模态评测平台，支持多个 benchmark 一站式对比 |
| **Open VLM Leaderboard** | HuggingFace 上 | 社区维护，持续更新各模型在主流 benchmark 上的成绩 |
| **LMSYS Vision Arena** | https://chat.lmsys.org | 人类盲测投票排名，最贴近真实使用体验（非固定题目） |
| **WildVision Arena** | https://huggingface.co/spaces/WildVision/vision-arena | 收集真实用户与 VLM 交互的 query，测试模型在"野生"场景下的表现 |
| **Artificial Analysis** | https://artificialanalysis.ai | 综合评测性能、速度、价格，适合选型决策 |

### 6.2 评测工具

| 工具 | 说明 |
|------|------|
| **VLMEvalKit** | 开源评测工具包，支持一键评测 100+ 个 VLM 在 40+ 个 benchmark 上的成绩（2025 年大幅扩充） |
| **lmms-eval** | LMMs-Eval，另一个主流评测框架，类似 lm-evaluation-harness 的多模态版 |
| **OpenCompass MMBench 工具链** | 配合 MMBench 系列 benchmark 的官方评测代码 |
| **MEGA-Bench 工具链** | 支持 500+ 任务的大规模评测，输出细粒度能力雷达图 |

---

## 七、如何选择 Benchmark？

不同场景关注不同 benchmark：

| 你的目标 | 优先看这些 Benchmark |
|---------|---------------------|
| 全面了解模型综合能力 | MMBench、MMStar、MEGA-Bench |
| 评估专业/学术场景 | MMMU、MMMU-Pro |
| 评估文档处理/OCR 场景 | DocVQA、TextVQA、ChartQA、MMLongBench-Doc |
| 评估模型可靠性（幻觉） | POPE、HallusionBench |
| 评估数学/科学推理 | MathVista、AI2D |
| 评估底层视觉感知 | BLINK |
| 评估视频理解 | Video-MME、MVBench、EgoSchema |
| 了解真实用户体验 | LMSYS Vision Arena、WildVision |
| 选型决策（性能+成本） | Artificial Analysis + Arena 排名 |

> **建议**：不要只看一个 benchmark 的分数，要**综合多个 benchmark** 形成全面判断。同时关注 Arena 排名（人类偏好）和固定题目 benchmark（客观分数）的差异。2025 年以来，推荐优先参考 **MMMU-Pro + MMStar + BLINK** 这组新 benchmark——它们在数据质量和区分度上优于早期基准。

---

## 附录：各 Benchmark 详细介绍

> 以下按类别列出各 benchmark 的详细信息，供查阅。

### A. 通用综合评测

#### A.1 MMBench

| 项目 | 内容 |
|------|------|
| **全称** | MMBench: Is Your Multi-modal Model an All-around Player? |
| **出品** | 清华大学 & 上海 AI Lab (2023) |
| **Motivation** | 早期 benchmark（如 VQAv2）题目单一、维度不够细。需要一个能**系统拆解**模型各项能力的综合评测 |
| **任务形式** | 多选题（A/B/C/D），~3000 题 |
| **评价指标** | Accuracy（准确率） |

**核心设计思路**：

- **20+ 能力维度**：将多模态能力细分为 Coarse Perception（粗粒度感知）、Fine-grained Perception（细粒度感知）、Reasoning（推理）三大类，下设属性识别、空间关系、OCR、逻辑推理等 20+ 子维度
- **CircularEval 策略**：同一道题，把选项顺序打乱测多次，只有每次都答对才算对。过滤掉模型"蒙对"的情况，评测更鲁棒
- **为什么重要**：目前最广泛使用的 VLM 综合 benchmark 之一，几乎所有 VLM 论文都会报 MMBench 分数

#### A.2 MME

| 项目 | 内容 |
|------|------|
| **全称** | MME: A Comprehensive Evaluation Benchmark for Multimodal LLMs |
| **出品** | 2023 |
| **Motivation** | 需要一个**简单直接**、覆盖面广的评测，用最基础的是非题快速摸底模型能力 |
| **任务形式** | 是非题（Yes/No），共 14 个子任务 |
| **评价指标** | Score（每个子任务满分 200，总分 2800） |

**核心设计思路**：

- **感知 vs 认知 双轨**：
  - **感知（Perception）**：物体存在性、计数、位置、颜色、OCR、海报理解、名人识别、场景理解、地标识别、艺术品理解（10 个子任务）
  - **认知（Cognition）**：常识推理、数值计算、文本翻译、代码推理（4 个子任务）
- **计分方式**：每个子任务包含多道题，每题有一正一反两个 Yes/No 问题，两个都答对才得分。防止模型"全答 Yes"刷分
- **为什么重要**：评测成本极低（都是是非题），出结果快，适合快速对比大量模型

#### A.3 SEED-Bench

| 项目 | 内容 |
|------|------|
| **全称** | SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension |
| **出品** | Tencent (2023) |
| **Motivation** | 现有 benchmark 只关注静态图像，缺乏**视频理解**评测。需要同时覆盖图像和视频的综合基准 |
| **任务形式** | 多选题，~19K 题（图像 ~14K + 视频 ~5K） |
| **评价指标** | Accuracy |

**核心设计思路**：

- **12 个评测维度**：场景理解、实例身份、实例属性、实例位置、实例计数、空间关系、实例交互、视觉推理、文字理解 + 3 个视频专属维度（动作识别、动作预测、过程理解）
- **题目来源**：基于真实数据集（CC3M、Something-Something 等）+ 人工标注 + GPT 辅助生成
- **为什么重要**：少数同时覆盖图像和视频的综合 benchmark，数据量大，维度全

#### A.4 MM-Vet

| 项目 | 内容 |
|------|------|
| **全称** | MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities |
| **出品** | NUS (2023) |
| **Motivation** | 多选题/是非题太简单，无法反映模型在**开放式场景**下的真实表现。需要评测模型"综合运用多种能力"解决复杂问题的水平 |
| **任务形式** | 开放式问答（自由回答），218 道精选题 |
| **评价指标** | GPT-4 评分（0-1 分） |

**核心设计思路**：

- **6 大核心能力**：识别（Recognition）、知识（Knowledge）、OCR、空间感知（Spatial Awareness）、语言生成（Language Generation）、数学（Math）
- **能力组合评测**：每道题可能同时需要 2-3 种能力的组合（如"读图中的数字 + 做数学计算"），更贴近真实使用场景
- **GPT-4 自动评分**：开放式回答没有标准答案，用 GPT-4 对模型的回答打分
- **为什么重要**：题目质量高、区分度强，但依赖 GPT-4 评分带来一定偏差

#### A.5 MMMU / MMMU-Pro — 大学级多学科多模态理解

| 项目 | 内容 |
|------|------|
| **全称** | MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark |
| **出品** | IN.AI Research & 多所高校 (2024) |
| **Motivation** | 现有 benchmark 大多停留在"常识级"，缺乏**专业领域知识**的评测。大学课程中大量使用图表、公式、电路图等，模型需要具备领域专业知识才能作答 |
| **任务形式** | 多选题 + 开放式问答，~11.5K 题（覆盖 30+ 学科） |
| **评价指标** | Accuracy |

**核心设计思路**：

- **30+ 学科覆盖**：涵盖艺术设计、商科、理科、工科、医学、人文社科六大领域
- **大学考试级难度**：题目来源于真实的大学课程教材、考试题和教科书
- **MMMU-Pro（2025 升级版）**：增加更多干扰选项（4 选 1 变 10 选 1）、引入 Vision-only 输入、增加开放式问答，难度大幅提升
- **为什么重要**：2025-2026 年最具影响力的 VLM benchmark 之一，顶级模型在 MMMU-Pro 上准确率仍低于 70%

#### A.6 MMStar — 精选高质量综合评测

| 项目 | 内容 |
|------|------|
| **全称** | MMStar: Are We on the Right Way for Evaluating Large Vision-Language Models? |
| **出品** | 2024 |
| **Motivation** | 现有 benchmark 存在严重的**数据泄漏**和**视觉无关性**问题——部分题目不看图就能答对 |
| **任务形式** | 多选题，1500 道精选题 |
| **评价指标** | Accuracy |

**核心设计思路**：

- **双重筛选机制**：先用纯文本 LLM 过滤掉"不看图也能答"的题目，再人工核验确保每道题都**视觉不可或缺**
- **6 大核心能力 × 18 细分维度**：粗粒度感知、细粒度感知、实例推理、逻辑推理、科学技术、数学
- **为什么重要**：有效避免数据污染导致的分数虚高，评测结果更可靠

#### A.7 MEGA-Bench — 超大规模能力评测

| 项目 | 内容 |
|------|------|
| **全称** | MEGA-Bench: Scaling Multimodal Evaluation to over 500 Real-World Tasks |
| **出品** | 2025 |
| **Motivation** | 单一 benchmark 覆盖的任务类型有限，需要一个**海量真实任务**的综合评测 |
| **任务形式** | 500+ 种不同类型的真实世界任务 |
| **评价指标** | 任务特定指标（多种指标聚合） |

**核心设计思路**：

- 覆盖**多图、视频、多轮对话、工具使用**等多种交互形式
- 支持**能力雷达图**可视化，一目了然看出模型优劣势分布
- **为什么重要**：截至 2026 年规模最大的多模态评测，适合深度分析模型的能力谱系

### B. 专项能力评测

#### B.1 TextVQA — OCR + 阅读理解

| 项目 | 内容 |
|------|------|
| **全称** | Towards VQA Models That Can Read |
| **出品** | Facebook AI (2019) |
| **数据** | ~45K 问题，基于 Open Images 中包含文字的图片 |
| **评价指标** | VQA Accuracy |

图片中包含各种文字（路牌、广告牌、书籍封面等），回答问题必须先"读"到图中的文字，再结合视觉上下文理解。与纯 OCR 的区别：不是简单提取文字，而是需要理解文字的含义和上下文。

#### B.2 DocVQA — 文档理解

| 项目 | 内容 |
|------|------|
| **全称** | DocVQA: A Dataset for VQA on Document Images |
| **出品** | CVC Barcelona (2021) |
| **数据** | ~50K 问题，基于 ~12K 文档图像 |
| **评价指标** | ANLS（Average Normalized Levenshtein Similarity） |

文档种类丰富（表格、表单、信件、报告、发票等），不仅要识别文字，还要理解**版面结构**。实际工业应用价值极高——IDP（智能文档处理）的核心评测。

#### B.3 ChartQA — 图表理解

| 项目 | 内容 |
|------|------|
| **全称** | ChartQA: A Benchmark for Question Answering about Charts |
| **出品** | 2022 |
| **数据** | ~32K 问题，覆盖多种图表类型 |
| **评价指标** | Relaxed Accuracy（数值答案允许 5% 误差） |

两类问题：**人工标注**（需要复杂推理，如"哪一年增速最快？"）和**机器生成**（较简单）。综合考验 OCR + 视觉理解 + 数值推理。

#### B.4 MathVista — 数学视觉推理

| 项目 | 内容 |
|------|------|
| **全称** | MathVista: Evaluating Mathematical Reasoning in Visual Contexts |
| **出品** | UCLA & Microsoft (2023) |
| **数据** | 6141 题，来自 31 个已有数据集 |
| **评价指标** | Accuracy |

覆盖 5 类数学推理（代数、算术、几何、逻辑、统计）× 7 类视觉上下文（几何图形、函数图、表格等）。难度跨度从小学到大学，暴露了当前 VLM 在精确数学推理上的短板。

#### B.5 RealWorldQA — 真实场景理解

| 项目 | 内容 |
|------|------|
| **出品** | xAI (2024) |
| **数据** | ~700 张真实世界照片 + 多选题 |
| **评价指标** | Accuracy |

图片来自真实拍摄（驾驶场景、日常生活等），问题侧重实用性（如"这个路口可以左转吗？"）。数据量较小，但题目质量高。

#### B.6 AI2D — 科学图表理解

| 项目 | 内容 |
|------|------|
| **全称** | AI2D: A Dataset of Illustrative Diagrams |
| **出品** | Allen AI (AI2) |
| **数据** | ~5000 张科学图 + ~15K 多选题 |
| **评价指标** | Accuracy |

图片都是科学教育领域的示意图/流程图，需要理解箭头指向、标注含义、元素间的因果/层级关系。与 ChartQA 的区别：ChartQA 侧重数据图表，AI2D 侧重概念示意图。

#### B.7 POPE — 物体幻觉评测

| 项目 | 内容 |
|------|------|
| **全称** | Evaluating Object Hallucination in Large Vision-Language Models |
| **出品** | 2023 |
| **任务形式** | 是非题——"图中是否有 [某物体]？" |
| **评价指标** | Accuracy、F1-Score、Yes 比例 |

**三种采样策略**（难度递增）：Random（随机）→ Popular（高频物体）→ Adversarial（高共现物体，如图中有餐桌就问有没有椅子）。Yes 比例远高于 50% 说明模型有严重的 Yes-bias。

#### B.8 HallusionBench — 综合幻觉评测

| 项目 | 内容 |
|------|------|
| **全称** | HallusionBench: An Advanced Diagnostic Suite for Entangled Language Hallucination and Visual Illusion in LVLMs |
| **出品** | 2024 |
| **数据** | 346 张图 + 1129 个问题 |
| **评价指标** | Accuracy |

区分**语言幻觉**（模型被文字误导）和**视觉错觉**（图片本身具有欺骗性），问题涉及数量、颜色、空间位置等多种属性的幻觉。与 POPE 互补。

#### B.9 BLINK — 人易模型难的视觉感知

| 项目 | 内容 |
|------|------|
| **全称** | BLINK: Multimodal Large Language Models Can See but Not Perceive |
| **出品** | 2024 |
| **数据** | 3807 道多选题，14 种视觉感知任务 |
| **评价指标** | Accuracy |

14 种感知任务（相对深度、拼图还原、视觉对应、多视角推理等），人类 >95% 但最强 VLM 仅 ~60%。揭示了 VLM"能理解但不能感知"的根本性短板。

#### B.10 MMLongBench-Doc — 长文档理解

| 项目 | 内容 |
|------|------|
| **全称** | MMLongBench-Doc: Benchmarking Long-context Document Understanding |
| **出品** | 2025 |
| **数据** | 1062 个问题，基于 130 篇长文档（平均 47 页） |
| **评价指标** | F1-Score |

文档长度 10-120 页，需要跨页面检索、关联和推理。测试长上下文多模态理解能力，与百万 token 级长上下文模型的发展趋势高度匹配。

#### B.11 MMT-Bench — 多任务多模态评测

| 项目 | 内容 |
|------|------|
| **全称** | MMT-Bench: A Comprehensive Multimodal Benchmark for Evaluating LMMs on Massive and Diverse Tasks |
| **出品** | 2024 |
| **数据** | ~32K 题，覆盖 162 种子任务 |
| **评价指标** | Accuracy |

涵盖视觉识别、定位、OCR、计数、3D 感知、动作理解等 30+ 能力维度，引入了自动驾驶、机器人、医疗影像等垂直行业场景。

### C. 视频理解评测

#### C.1 MVBench

| 项目 | 内容 |
|------|------|
| **全称** | MVBench: A Comprehensive Multi-modal Video Understanding Benchmark |
| **出品** | 2024 |
| **任务形式** | 多选题，20 个时序理解维度 |
| **评价指标** | Accuracy |

20 个维度包括：动作序列、动作预测、动作反推、动作定位、动作计数、异常检测、物体存在性、角色识别、场景切换、移动方向、运动属性等。每个维度独立评分。

#### C.2 Video-MME

| 项目 | 内容 |
|------|------|
| **全称** | Video-MME: The First-Ever Comprehensive Evaluation Benchmark of Multi-modal LLMs in Video Analysis |
| **出品** | 2024 |
| **数据** | 900 个视频，2700 个问题 |
| **评价指标** | Accuracy |

三档时长：短视频（<2 min）、中视频（4-15 min）、长视频（30-60 min）。有/无字幕两种设置，涵盖 30+ 视频类型。长视频理解是当前 VLM 的前沿难题。

#### C.3 EgoSchema

| 项目 | 内容 |
|------|------|
| **全称** | EgoSchema: A Diagnostic Benchmark for Very Long-form Video Language Understanding |
| **出品** | 2023 |
| **数据** | 5031 道多选题，基于 Ego4D 数据集（第一人称活动视频） |
| **评价指标** | Accuracy |

每个视频长达 3 分钟（传统视频 QA 仅几秒），第一人称视角带来额外难度（视角不稳定、遮挡多）。子集 EgoSchema-500 常用于快速评测。

---

**相关文档**：
- [VLM 概述](VLM概述.md) — VLM 总览与入门
- [VLM 训练与评测](VLM训练与评测.md) — 训练流程、数据构造、高效训练与部署
- [VLM 主流架构详解](VLM主流架构详解.md) — 具体模型架构

[返回上级](README.md) | [返回总目录](../../README.md)

---
name: paper-card
description: Build a comprehensive Chinese paper card under idea-research/ideas/<slug>/wiki/papers/<paper>.md — one card that lets reader fully understand the paper in a single read without reopening the PDF. Covers TL;DR + 背景与任务 + 方法概览 + 方法细节（含公式直觉） + 实现 + 关键结果（表 + 解读）+ 关联工作 + 与本课题的关系. Faithful prose explanation with original symbols preserved. **No critical analysis, no judgment, no gap-finding, no author affiliations, no future plans**. Critical reading happens in AI Assistant chats, not in the card. Embeds PDF reader iframe at top. Triggered by "论文卡片", "paper card", "写一张卡片", "新增论文卡片", or direct invocation.
---

# Paper Card Skill

为 `idea-research/ideas/<slug>/wiki/papers/<paper>.md` 产出**全面理解式中文笔记** —— 读者读完这一张卡片就能完整理解论文的"问题来源 / 方法主干 / 实现选择 / 结果数字"，**不需要再回原文**（除非要核对某个具体公式或表格细节）。

**和 AI Assistant 对话的分工**：本卡片只承担"沉淀理解 + 事实记录"。批判分析（作者声称 vs 真实成立、未明说的局限、对本课题的研究缺口）由用户和 AI Assistant 在卡片基础上对话挖掘，**不写进卡片**。

**产出场景**：私人知识库，自用 + 给学姐 / 导师 / 合作者看。目标是"半年后回来读这一张卡，能完整复盘论文做了什么和为什么"。

---

## 卡片结构（强模板）

每张卡严格按这个顺序，**节标题完全照抄**。某节没有材料就写一句"原文未给出"，不要删节。

### 1. Frontmatter

```yaml
---
title: "<完整英文标题>"
arxiv: <id, 如 2501.11347 or null 若无 arXiv>
ieee: <doc id 若仅 IEEE>
doi: <如有>
venue: "<会议/期刊 + 年 + 卷/期 + 文章号 或 'arXiv YYYY-MM (preprint，截至 YYYY-MM-DD 无正式接收信息)'>"
year: <发表年>
slug: <短名>
tags: [tag1, tag2, ...]
updated: <YYYY-MM-DD>
---
```

**venue 字段**：必须精确到能引用的级别——
- 会议：`MICCAI 2024 (Early Accept)` / `CVPR 2026` / `ICRA 2023`
- 期刊：`Medical Image Analysis 2025, Vol.107CA, Article 103789`
- IEEE：`IEEE J-BHI 2025, Vol.29 Issue 12, pp.9027-9040`
- Preprint：`arXiv 2026-05 (preprint，截至 2026-05-28 无正式接收信息)`
- 含奖项：`IPCAI 2024 Best Paper / IJCARS Vol.19, 2024`

### 2. 标题块 + venue + PDF iframe

**venue 必须在 body 显式出现一次**——前端渲染会隐藏 frontmatter。

```markdown
# <slug>

> 📄 **原始论文**：[arXiv <id>](https://arxiv.org/abs/<id>) · **发表场所**：<同 frontmatter venue> · 代码：[github.com/...](https://github.com/...) 或写"原文声明 'will be released'，目前未释出"
> 📖 **内嵌阅读**：在下方阅读器中**选中段落**，右下角会弹出 "Ask AI" 按钮，可把段落喂给右侧 AI 助手追问（AI 已加载本文全文为上下文）。如需全屏请 <a href="/docs/tools/pdf-reader.html?pdf=idea-research/ideas/<slug>/papers/<paper>.pdf&title=<ShortName>" target="_blank">在新标签打开</a>。

<iframe
  class="pdf-embed"
  src="/docs/tools/pdf-reader.html?pdf=idea-research/ideas/<slug>/papers/<paper>.pdf&title=<ShortName>&embed=1"
  loading="lazy"
  title="<ShortName> PDF 内嵌阅读器"></iframe>

---
```

### 3. TL;DR — 300-500 字

一段 dense 中文，**严格事实，但要让一个陌生读者读这 500 字能立刻明白本文的全貌**：

- 第 1 句：作者立了什么任务（用名词短语带住）
- 第 2-3 句：作者诊断既有方法的哪一类局限（**只复述论文 §1 的诊断，不评价它的对错**）
- 第 4-5 句：本作方法的中心想法（一两个核心机制 + 它和上述局限的对应关系）
- 第 6-7 句：用了什么数据 + 关键 SOTA 数字 + 最相关的 baseline 数字（带 Δ）
- 第 8 句（可选）：代码 / 数据 release 状态、训练规模量级

**不写**：作者动机评判、"这是 surgical 领域第一个 X"（除非原文明确这么声称）、"对本课题非常重要" 类。**不要 bullet**。

### 4. 任务背景与定义 — 300-700 字

这一节让读者明白"为什么作者要做这个任务"，**全部转述论文 §1 / §2 的事实**。结构：

**4.1 任务背景**：一段散文，转述论文对所在领域当前研究状态的描述。如：
- 该领域已经有 A、B、C 等成熟工作
- 但缺乏对 D 维度的处理（用论文原话 / 原文引用号）
- 现有方法在 E 场景下失效（论文 §1 / §2 给的具体场景）

> 注意：这一节**只搬运论文自己的诊断**，不补自己的判断。如果论文的诊断有问题，那是 AI Assistant 对话时挖的事，不写进卡。

**4.2 任务定义**：

```markdown
- **输入**：...
- **输出**：...
- 任务谱系：是哪类既有任务的扩展 / 限制
```

若任务有多个范式 / 子任务，**用表格列出**所有变体（参考 endochat.md 的 5 范式 7 子任务表）。

**4.3 典型用例**（可选，原文若给出）：1-2 句话描述一个具体的 input-output 例子，让读者抓到任务的形态。

### 5. 方法概览 — 散文 200-400 字 + ASCII pipeline

让读者立刻看到论文的**结构骨架**，再读详细模块时心里有图。结构：

**5.1 整体 pipeline**：ASCII / fenced code 画一次。例：

```
Image + Question
       ↓
[Module A: 视觉特征提取]  ← 论文 §3.2
       ↓
[Module B: 跨模态融合]    ← 论文 §3.3
       ↓
[Module C: 输出头]        ← 论文 §3.4
       ↓
(bbox, answer)
```

**5.2 设计立意**：一段 150-300 字散文，说清楚：
- 论文的整体设计走的是哪条技术路线（如 "two-stage detection + LLM fine-tune" / "end-to-end MLLM with grounding head"）
- 哪些环节是**端到端学的**，哪些是**外挂现成模型 + 后处理拼的**（这个区分很重要，但只描述事实——"X 用了预训练的 Y，Z 是从头训练" ——不评价好坏）
- 各模块如何串联（数据流方向、监督信号怎么流）

### 6. 方法细节 — 按论文章节顺序展开

这是卡片的主体，**要让读者能直接理解每个模块的公式和直觉**，不需要再去翻原文。

每个模块按以下结构：

```markdown
#### 6.X <模块原文名 (Module X)>

<一段 100-300 字散文：做什么，输入输出形状，在 pipeline 里的位置>

**关键公式**：

$$
<原文公式，保留原文符号>
\tag{原文公式编号}
$$

- 符号 <X>：<这个变量在论文里指什么、什么形状、什么含义>
- 符号 <Y>：...
- 设计动机：<论文 §X.X 给出的直觉解释，原话转述。如"通过这一项约束 part 分配收敛到 one-hot"——只转述论文的解释，不补自己的>

<可选：如果该模块多步骤，分小标题 6.X.1 / 6.X.2>
```

**重点保留**：
- **原文公式编号**（写 "公式 (5)"，不要 paraphrase 成"第五条"）
- **原文符号** —— `Σ`、`λ`、`R_imp`、`p̂` 等照抄，**不要 paraphrase 成"奖励"等中文名**
- **原文章节引用**："详见论文 §3.2" / "Figure 2 (b) 给出此模块的 pipeline"

**长度参考**：纯方法节合计 1500-3500 字。论文方法复杂的（如 multi-stage 训练 + 多个 loss 项）可到 5000 字，简单的（如 LoRA 调一个 LLM）500 字够。

### 7. 实现细节 — bullet 列表

照抄论文 §4 实验设置或 Implementation 节：

- **Base model / backbone**：如 LLaMA-2-13B + LoRA rank 8 / 训练自有 ViT-L
- **训练 GPU**：如 4× NVIDIA A800 (80GB)
- **优化器 / lr / schedule**：AdamW, lr=1e-4, cosine schedule
- **batch / epoch**：bs=32, 100 epochs
- **训练数据规模**：50k images / 200k QA pairs
- **数据增强 / 预处理**：（如 random crop / normalize）
- **训练时间**：48 hours / convergence
- **推理速度**：24.5 s / sample (若给出)
- **代码 / 数据 release 状态**：[github.com/...]() 已开源 / 论文声明 "will be released"，截至 <today> 未释出

### 8. 关键结果 — 表 + 解读，但只解读"测什么"，不解读"好不好"

按论文 §5 实验顺序展开。每个表（或对照组）按以下结构：

```markdown
#### 8.X <表标题，对应原文 Table X>

<一段 80-150 字散文：这个表测的是什么任务 / 在什么评测集上 / 和谁比 / 评价指标是什么意思>

<表本体 —— 表头与原文一致，本作行加粗，关键 baseline 行保留>

<一段 80-150 字散文：直接读数字事实。如 "本作 Acc 0.7012，对最强 baseline X (0.6512) 提升 +5.0；ablation 去掉 module Y 后掉到 0.6321"。**不写"显著"，不写"大幅领先"，让数字自己说话。**>
```

若结果分多个 split / setting（如 in-domain vs external），分多个 8.X 小节。

消融实验也照搬一个独立小节。

### 9. 关联工作 — bullet 列表

按论文 §2 Related Work 引用过的工作分类列出，**不写自己的判断**。

`[[wiki-link]]` 互引同 KB 内已有的卡片，**不要写未建卡的论文**（避免坏链）。其它论文用纯文本 + 引用号即可。

```markdown
- **同任务 SOTA 链**：[[surgical-vqla]] (ICRA 2023) / [[surgical-vqla-plus-plus]] / [[surgical-mamballm]]
- **同任务但 KB 未建卡**：Surgical-LVLM / LISA / GLAMM（论文 §2 引用）
- **方法引用**：SPHINX 架构 / GRPO / DeepSeek-R1
```

### 10. 与本课题的关系 — 事实对照

**事实层面**对照：任务异同 / 资源差异 / 范式差异。**不写"建议"，不写"我们要 differentiate 这一点"**——那是 AI Assistant 对话时挖的事。

```markdown
- **任务**：本作输出 mask；本课题输出 bbox + 自然语言答案
- **资源**：本作 4× L20 (192 GB)；本课题 2× A5000 Ada (64 GB)
- **范式**：本作 RL 训练；本课题 training-free
- **数据集**：本作用 EndoVis17/18；本课题用 EV18 + 自建 visual prompt 数据
```

至多 3-6 条 bullet，每条 1 行。**不写"对本课题留下了什么缺口"**——那是 AI Assistant 对话时挖的事。

---

## 风格规则（必须）

- **散文为主，bullet 为辅**：方法说明、背景诊断、结果解读用**段落叙述**，不要切成 bullet 流水账。bullet 留给真正并列的项（任务谱系、超参列表、与本课题的对照差异）。
- **中文为主**，关键术语保留英文：`bbox` / `mIoU` / `SAM` / `LoRA` / `GRPO` / `MLLM` / `LVLM` / `articulation` / `URDF` / `flow matching` / `B-rep` —— 首次出现可加括号中文注，后文直接用英文。
- **保留原文符号**：公式 `Σ`、`λ`、`R_imp`、`p̂` 等照抄，**不要 paraphrase 成"奖励"等中文等价物**。读者要能在原文 PDF 里找到同一个符号。
- **保留 Table / Figure 引用号**：写"Table 1 显示..." / "Figure 2 (b) 给出 pipeline"，方便读者回查原文。
- **保留公式编号**：写"由公式 (5)..." / "公式 (12) 是 entropy regularizer"，不要 paraphrase 成"第五个公式"。
- **加粗有节制**：每屏 ≤ 3 处。加粗用在"一行扫过也要看到"的数字 / 决定性的对照 / 主语 SOTA 数字。
- **数字带单位**：`64 GB`、`Acc 0.6512`、`mIoU 0.7739`、`+12.12 Acc`、`24.5 s / sample`、`8× H20`。
- **诚实**：写不到位的节直接写"原文未给出"或"待补"，不要留 "TODO" / "略" / "see appendix"。
- **不混层**：转述论文的论断时用"论文称..." / "作者声称..."；陈述事实时用直接陈述。**不要在卡里用"实际上 / 严格讲 / 但是这有 limitation"** —— 那是 AI Assistant 对话时的话。

---

## 禁止写入的内容

| 禁项                                                                         | 理由                                           |
| ---------------------------------------------------------------------------- | ---------------------------------------------- |
| **作者单位列表 / 通讯作者**                                                  | 信息冗余，arxiv 页面有                         |
| **译者注 / 我的判断 / 优先级 ⭐⭐⭐**                                           | 卡只承载事实理解，判断是 AI Assistant 对话的事 |
| **批判分析（声称 vs 真实 vs 未明说）**                                       | 同上                                           |
| **未明说局限清单**                                                           | 同上                                           |
| **"对本课题留下什么缺口" / "可切入的机会"**                                  | 同上                                           |
| **未来计划 / "下一步建议"**                                                  | 同上                                           |
| **subjective 比较语**（如"显著超过""第一个做 X 的工作"，除非原文确切这么说） | 替换为数字 + Δ                                 |

---

## 工作流

1. 用户给 `<slug>` 项目名 + 论文 arxiv id / IEEE doc id / PDF 路径。
2. 检查 `idea-research/ideas/<slug>/raw/<paper>.md` 是否存在：
   - 存在：直接读
   - 不存在：`curl -L -s -o papers/<paper>.pdf "https://arxiv.org/pdf/<id>"` + `pdftotext -layout papers/<paper>.pdf raw/<paper>.md`
3. 跑 venue 校验：WebFetch arxiv abs 页或 Semantic Scholar API 确认 venue + 年份 + 卷号 / 期号 / 文章号。若 preprint 状态，明确写"截至 <today> 无正式接收信息"。
4. 通读 raw/<paper>.md 一遍，识别**论文骨架**（method 用了哪几个模块、实验有几个表、关联工作分了哪几派）。
5. 按 10 节模板写 `wiki/papers/<paper>.md`，按下列优先级填充：
   - **必填**：TL;DR / 任务背景与定义 / 方法概览 / 方法细节 / 实现 / 关键结果
   - **选填但强烈推荐**：关联工作 / 与本课题的关系
6. 写完后跑下方**自查清单**。
7. 更新 `wiki/papers/README.md` 索引表（加一行）+ 总数 +1。
8. 若涉及跨卡片对比，更新 `wiki/concepts/taxonomy.md` 三轴矩阵 + 派系树。
9. 更新 `_sidebar.md` 加入对应分类。
10. 输出变更文件列表 + 卡片入口路径。

---

## 自查清单（交付前）

- [ ] **frontmatter**：title / arxiv / venue / year / slug / tags / updated 齐全；venue 精确到卷/期/年
- [ ] **body 标题块**：venue 行显式出现（不只在 frontmatter）；PDF iframe 路径正确
- [ ] **TL;DR**：300-500 字一段，含问题来源、方法立意、关键数字
- [ ] **任务背景与定义**：包含论文 §1 / §2 对领域现状的诊断（原话）+ 任务 I/O + 可选典型用例
- [ ] **方法概览**：ASCII pipeline + 一段散文讲清骨架与端到端 / 外挂的边界
- [ ] **方法细节**：每个模块有散文说明 + 关键公式 + 公式符号一一解释 + 模块在 pipeline 里的位置
- [ ] **实现细节**：GPU / 超参 / 数据规模 / 代码 release 状态
- [ ] **关键结果**：每个表有"测什么"段 + 表 + "读数字"段；本作数字加粗；baseline 数字不省略
- [ ] **关联工作**：[[wiki-link]] 互引已建卡论文；未建卡论文用纯文本
- [ ] **与本课题的关系**：3-6 条事实差异（任务 / 资源 / 范式 / 数据集），**不含判断**
- [ ] **无作者单位 / 译者注 / 未来计划 / 优先级 ⭐**
- [ ] **无批判语**：通卡搜不到"实际上 / 严格讲 / 但 / 但是 / 局限是" 这类批判转折
- [ ] **README.md 已更新**（行 + 总数）
- [ ] **_sidebar.md 已更新**（对应分类下）
- [ ] **taxonomy.md** 三轴矩阵已加行（若适用）

任意一条不达标视为未完成。

---

## 与其他 skill 的协作

- **AI Assistant 对话（不是 skill）**：承接"声称 vs 真实成立 vs 未明说"的批判分析、"对本课题留下什么缺口" 的研究定位。用户读完卡后跟右侧 AI 抽屉对话来做深度挖掘，**产出不沉淀到 paper card**，可以另存 `wiki/gaps/<topic>.md` 或留在对话里。
- **idea-research-skill**：本 skill 产出的卡片是 idea-research Phase 2 (Compile Wiki) 的标准件。

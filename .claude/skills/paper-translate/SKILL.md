---
name: paper-translate
description: Produce a comprehensive faithful Chinese reading-note for an academic paper — translated abstract plus exhaustive Chinese renderings of every section, every figure, every experimental number, every ablation and limitation — to append under a paper card in idea-research/ideas/<slug>/wiki/papers/. Uses raw/ cache. Context is the user's private GitHub knowledge base; notes should go as deep as the reader needs, not stop at a summary. Triggered by "翻译论文", "中文导读", "paper translate", "中文技术笔记", or direct invocation.
---

# Paper Translate Skill

为 idea-research 的 paper card 产出**中文技术笔记**——目标是"读这份笔记 ≈ 读原文"，读者不用再翻英文就能完整理解方法、实验、结论、局限、以及作者话外的工程细节。

**产出场景**：私人 GitHub 知识库，自用。**不追求字数最短，追求信息密度最大 + 表达最流畅**。

## 核心原则

1. **忠实**。原文讲了什么，笔记里就有什么。不跳过、不合并、不"简化"。作者花了一段话论证的点，笔记里就用对应的一段话转述。
2. **流畅**。不逐句硬译——用自然中文把意思讲清楚。英文术语保留。
3. **密度**。每张图要描述、每个表要复述数字、每个损失项 / 超参 / 训练细节 / 消融结论都要提到。
4. **Key quote 可直引**：作者原话点睛的句子（limitations、关键 claim、motivation 的原句）可以用英文原文短引（例如 `> The method currently only handles...`）后加中文翻译，让读者能校对原文。
5. **加译者注**：原文理解需要的额外上下文——术语说明、和邻近工作的关系、实验号里没提但对解读重要的点——单独一节。

## 产出结构

```markdown
## 中文技术笔记

### 摘要（逐段完整译文）
[abstract 按原段数一段对一段翻译；长复合句可拆短句以便阅读]

### 研究动机 / Introduction
[Introduction 的完整中文转述：痛点 → 现有方法的问题 → 本文 claim → contributions 列表]

### 相关工作
[Related Work 每个子节（dynamic GS、articulation estimation、generative prior 等）对应一段中文，列出代表性前作和它们的局限]

### 方法详解
按原文节结构，每个 subsection 一个小标题：

#### 3.1 XXX（对应原文 section 名）
[数学公式用 LaTeX 保留；模型结构、前向流程、训练目标、损失函数组成、超参数选择、推理过程都要覆盖；作者用图示的流程图要用文字描述]

#### 3.2 XXX
...

### 实验与结果

#### 实验设置
[数据集名 + 规模 + 切分、硬件、训练时长、超参数、baseline 列表]

#### 主结果
[Table N 的全部数字用中文表头复述；每个 baseline 的数字都要列；作者的 claim 要和数字对上]

#### 消融
[每个消融组合及其结论；作者是如何说明每个设计的必要性的]

#### 定性结果
[每张定性图（图 4、图 5……）展示的现象]

### 图表要点
[图 1 到图 N 每张图独立说明：这张图在原文哪里、传达什么信息、用来支持哪个 claim]

### 作者承认的局限
[Limitations 节 / Discussion 节 / Failure Cases 节所有内容的中文转述；作者列的每一条 limitation 都要对应出现]

### 译者注
[值得额外解释的术语、和邻近工作的异同、读 paper 时需要警惕的点]
```

## 翻译风格

- **主动 + 简洁**："We propose X" → "本文提出 X"
- **术语保留英文**：3DGS、SDS、VLM、MLLM、URDF、SE(3)、PartNet-Mobility、Trellis、Hunyuan3D、MuJoCo、transformer、VAE、SDF、NeRF、GoM；第一次出现可加括号中文说明。
- **数字和度量保留原样**：193×、~10s、2K、PSNR 27.82、CD 4.05、Chamfer Distance、F-Score、Normal Consistency。
- **段落结构跟原文**：原文一段话在笔记里也是一段（或拆成两三短句的一段组），不合并不跳过。
- **被动 → 主动**：中文句子以动作主体开头。
- **Key quote 可直引**：
  ```markdown
  > HoloScene currently only handles videos of static indoor scenes.
  译：HoloScene 当前只处理静态室内场景视频。
  ```

## 工作流

1. 用户给 arxiv URL / ID / paper slug。
2. 检查 `idea-research/ideas/<slug>/raw/<paper>.md` 是否存在：
   - **存在**：直接用。
   - **不存在**：
     - WebFetch `https://arxiv.org/abs/<id>` 拿 abstract 和 metadata。
     - WebFetch `https://arxiv.org/html/<id>v1`（或 v2）拿正文 HTML，提取每节的结构、公式、实验表、图 caption、limitations 节。
     - WebFetch 项目页（若有）补充方法图和应用 demo 描述。
     - 合并写入 `raw/<paper>.md`。
3. 基于 raw 写中文技术笔记，按上面的结构组织。长度不设上限——原文信息多就写长。
4. Edit paper card，append `## 中文技术笔记` 节到末尾。如果已有该节就重新生成（显式覆盖）。
5. 输出变更列表：新增 / 修改的文件。

## 质量检查表（交付前自查）

- [ ] **abstract**：每一句都有对应中文
- [ ] **introduction**：作者的 contributions 列表里每一条都有对应中文段落
- [ ] **method**：每个 subsection 都有小标题 + 段落；损失函数、关键超参、训练策略都提到
- [ ] **experiments**：所有表格的数字都转述了（含 baseline）；所有消融组合的数字都提了
- [ ] **figures**：Figure 1 到最后一张都有独立说明
- [ ] **limitations**：作者列的每一条 limitation 都能在笔记里找到对应
- [ ] **术语**：3DGS、SDS、URDF 等专有名词保留英文且首次出现有中文解释
- [ ] **数字**：所有定量结果保留原始单位和精度

遗漏任意一条视为未完成。

## 原则

- **私人知识库语境**：目标是自己以后看得懂，信息要全。公开转载不在考虑范围。
- **raw/ 只抓一次**：同一篇论文的 WebFetch 结果以 raw/<paper>.md 为唯一来源，之后的再读就读这份本地文件。
- **图片引用用 URL**：需要原图就引 arxiv 或项目页的 URL（比如 `https://arxiv.org/html/<id>v1/extracted/<id>/figX.png`），别下载副本到仓库。

## 与 idea-research-skill 的协作

若同时要 "ingest + 翻译"：先走 idea-research-skill 的 Phase 1 Ingest 把 PDF 转 markdown 落 raw/，再走本 skill 基于 raw 写中文笔记。别并行下两次 PDF。

---
name: paper-card
description: Build a top-down narrative Chinese paper card under idea-research/ideas/<slug>/wiki/papers/<paper>.md — a 10-section reading guide that opens with a one-sentence statement, lifts a beginner from zero in 30 seconds, walks through a concrete worked example, names the real difficulty, then drills into method, math, experiments, limitations, and translator's notes. Embeds the PDF reader iframe at top and inline figures (extracted via scripts/extract_figures.py) with detailed Chinese captions. Triggered by "论文卡片", "paper card", "写一张卡片", "新增论文卡片", or direct invocation. Distinct from paper-translate (which is faithful section-by-section translation, not narrative rewrite).
---

# Paper Card Skill

为 `idea-research/ideas/<slug>/wiki/papers/<paper>.md` 产出**自顶向下的叙事式论文卡片**——读者按"一句话 → 30 秒上手 → 一个具体例子 → 难在哪 → 核心想法 → pipeline → 技术深度 → 实验 → 局限 → 译者注"的次序读，每往下一节都比上一节更深。

**和 paper-translate 的区别**：paper-translate 是按原文 section 一一对应地忠实中译；paper-card 是**重组叙事顺序**——先讲故事让人愿意读下去，再讲技术。两个 skill 可以共存（同一张卡片末尾可以再附一节中文技术笔记）。

**产出场景**：私人知识库，自用，目标是"半年后回看也能 5 分钟想起这篇论文在做什么"。

---

## 卡片结构（10 节模板）

每张卡片严格按这个顺序写。如果某节没有材料，**宁缺毋滥**地写一两句也行——但不要打乱顺序。

### Frontmatter

```yaml
---
paper: <ShortName>            # 论文短名，e.g. ArtGS
title: <Full title>
authors: <逗号分隔>
affiliation: <主要机构 / 联合机构>
arxiv: <id, 如 2502.19459>
venue: <ICLR 2025 / preprint / ...>
date: <YYYY-MM>
tags: [tag1, tag2, ...]       # 用于跨卡片筛选，参考已有卡片的 tag 风格
---
```

### 标题块 + 内嵌 PDF

```markdown
# <ShortName> — <Full title>

> 📄 **原始论文**：[arXiv <id>](https://arxiv.org/abs/<id>)
> 📖 **内嵌阅读**：在下方阅读器中**选中段落**，右下角会弹出 "Ask AI" 按钮，可把段落喂给右侧 AI 助手追问（AI 已加载本文全文为上下文）。如需全屏请 <a href="/docs/tools/pdf-reader.html?pdf=idea-research/ideas/<slug>/papers/<paper>.pdf&title=<ShortName>" target="_blank">在新标签打开</a>。

<iframe
  class="pdf-embed"
  src="/docs/tools/pdf-reader.html?pdf=idea-research/ideas/<slug>/papers/<paper>.pdf&title=<ShortName>&embed=1"
  loading="lazy"
  title="<ShortName> PDF 内嵌阅读器"></iframe>
```

### 一、这篇论文在做什么？（一句话）

**真的就是一句话**——加粗，让读者一秒抓住核心。然后用一两短句具象化（"如果你只读这一句，知道这就够了……"）。

避免抽象术语，多用具体物体或操作（"给一个柜子拍几组照片，自动重建出门能开、抽屉能拉的可交互 3D 模型"，而不是"我们提出一种铰接物体重建框架"）。

### 二、零基础 30 秒上手

- 列 3–5 个**关键术语**，每个一句话用日常类比解释。
- 然后写一节"**这件事和我有什么关系？**"——用 bullet 列 2–4 个具体下游场景（哪种工程师会用到、能解决什么实际问题）。
- 这节决定读者愿不愿意继续读下去。**不能跳过**。

### 三、看一个具体例子

挑论文里最直观的那个 demo（不一定是论文 teaser，按"最容易脑补出画面"的标准选）。写：

- **输入**：人能听懂的描述（"100 张照片"，不是"a multi-view image set"）。
- **输出**：能干什么（"在仿真器里点抽屉，它就滑出来"）。
- **代价**：训练时间 / 硬件 / 数据量——给一个数字，否则读者永远不知道这事现实不现实。

### 四、为什么这件事不简单

**关键节**：让读者明白论文真正解决的难点是什么。常见结构是 2–3 个"难点"小标题：

```
### 难点 1：<具体问题>
[1-2 段解释——为什么 naive 做法不行]

### 难点 2：<具体问题>
...
```

不是泛泛说"很难"，而是**点名某种 baseline 在某种情况下会失败**。

### 五、<ShortName> 的核心想法（三句话）

把论文核心 idea 压缩成 3 条带编号的短陈述。每条一两句话，**讲做法 + 为什么 work**——不是 contributions list 的复制粘贴。

### 六、<ShortName> 的 pipeline / 流程

如果论文是 multi-stage 的，开一节图示流程。可以用：

- ASCII / fenced code block 画框图
- 嵌入 figure 1 的提取图（见下方 figure 规则）
- "stage 1 → stage 2"小标题分段

目标：读完这节就能**复述论文整体流程**。

### 七、技术深度（公式 + 训练细节）

按需开 7.1 / 7.2 / 7.3 / 7.4 子标题。覆盖：

- 关键损失函数（用 LaTeX 或 fenced code 都行，保持原文符号）
- 模型结构 / 关键超参
- 训练策略（warmup、lr schedule、L_X 只在前 N 步开启之类的实操细节）
- 训练配置（GPU / 时长）

公式不要 paraphrase 成中文等价物——保留原符号。

### 八、实验：从简单到难

按 benchmark / 数据集 顺序组织，每个数据集一段，含：

- 表格（用 markdown table，列出本文方法 vs baseline 的关键数字）
- 一段"**关键观察**"——表格右边的话语（"部件越多，DTA 越崩，本文仍稳"）。
- 消融用单独子节，列出每个被 ablate 掉的组件以及对应的 metric 退化。

定性图就在这节嵌入（按 figure 规则）。

### 九、作者承认的局限

直接抄 limitations / discussion / failure cases 节。可以用 verbatim 英文短引（key claim、limitation 句子）+ 中文翻译并列：

```markdown
> **EN (verbatim)** "<原文限制句>"

整个 pipeline 由 ... 驱动，由此衍生：

1. **<限制 1 名字>**：<解释>
2. ...
```

不要漏掉 limitations——这是日后判断"这个方法在我的场景能不能用"的依据。

### 十、读完之后的几个判断（译者注）

**这是卡片最有价值的一节**——把论文放回坐标系。常用子节：

- **10.1 这篇在 \<研究方向\> 大图里占什么位置？**——和同专题其他卡片做矩阵对比（输入 / 范式 / 时间 / 输出精度 / 物理参数 / ...）。
- **10.2 为什么作者选 \<某个设计\> 而不是 \<另一种\>？**——抠一两个值得记住的设计判断。
- **10.3 思路血缘——这套方法是从哪类工作演化来的？**
- **10.4 这篇的硬边界是什么？**——什么场景下 100% 用不了。

每个译者注小节带一句"**值得记住的设计判断**"或"**注意 / 警惕**"。

### 附录 A：关联工作

```markdown
- **前身**：<list>
- **同期**：<list>
- **跟进**：<list — 含本专题里的兄弟卡片，用相对路径链接>
```

### 附录 B：摘要英中对照（可选）

如果论文 abstract 关键，逐句 verbatim + 中译：

```markdown
> **EN** "<原句>"

**中** <中译>
```

---

## 风格指南

- **加粗用得有节制**：每屏 1–3 处加粗，用在"读者只看一眼也要看到"的字眼。不要整段加粗。
- **Bullet 列表 vs 段落**：方法解释用段落，并列对比用 bullet / 表格。不要把流畅论述硬切成 bullet。
- **数字一定带单位**：~8 分钟、单卡 RTX 3090、PSNR 27.82、CD 4.05、193×、~10s。
- **术语保留英文**：3DGS、SDS、URDF、SE(3)、PartNet-Mobility、MuJoCo、transformer、VAE、SDF、NeRF、VLM、MLLM——首次出现可加括号中文注。
- **避免半成品节**：写不到位的章节不要留 "TODO / 待补"——要么写一句精炼版本，要么删掉这一节。

## 图片提取与嵌入

### 提取流程

1. PDF 放在 `idea-research/ideas/<slug>/papers/<paper>.pdf`。
2. 在 `scripts/extract_figures.py` 的 `PAPERS` list 加一行 `("<paper>.pdf", "<paper>")`。
3. `python scripts/extract_figures.py` —— 输出到 `idea-research/ideas/<slug>/figures/<paper>/<paper>_figN.png`。
4. **逐图肉眼检查**：caption-relative 裁剪偶尔会把上半张图截掉、或把下一节标题裁进来。哪张不对就手动从 PDF 截图覆盖同名文件。
5. 失败的、裁不出来的图：**就不嵌**，文字描述代替。**不要**为了凑齐图数硬留低质量裁剪。

### 嵌入格式（强模板）

每张图后必须紧跟一段中文 caption，含**论文页码**和**这张图传达什么**：

```markdown
![<ShortName> <场景描述> — Figure N](../../figures/<paper>/<paper>_figN.png)

↑ **Figure N（论文 p.X）**：<这张图画了什么 + 关键观察。>
```

**关键观察**：当一张图的"看点"在和 baseline 比较时，caption 要点出 baseline 失败模式（"DTA 在 low-visibility state 直接缺面，本方法因为 canonical 共享所以两个状态都完整"）。

## 工作流

1. 用户给 `<slug>` + 论文 arxiv id（或 PDF 路径）。
2. 检查 `idea-research/ideas/<slug>/raw/<paper>.md` 是否存在：
   - **存在**：直接读。
   - **不存在**：WebFetch arxiv abs 页拿 metadata；WebFetch arxiv html 页抓正文（含每节 / 公式 / 表格 / figure caption / limitations）；存为 `raw/<paper>.md`。
3. 跑 `scripts/extract_figures.py`（如新增论文已加进 `PAPERS`）。
4. 按 10 节模板写 `wiki/papers/<paper>.md`。先把骨架按顺序写完，再回填细节——避免某一节深耕到忘了下一节。
5. 写完后**自查清单**（见下）跑一遍。
6. 更新 `wiki/papers/README.md` 卡片清单表（一行）。
7. 如果新论文涉及跨卡片对比，更新 `wiki/concepts/taxonomy.md` 的对应矩阵列。
8. 输出变更文件列表。

## 自查清单（交付前）

- [ ] **frontmatter** 完整：paper / title / authors / affiliation / arxiv / venue / date / tags
- [ ] **iframe 块**就在标题下方，PDF 路径正确（embed=1 + 同时给"新标签打开"链接）
- [ ] **第一节真的是一句话**：加粗后第一句不超过 50 字
- [ ] **第二节有"和我有什么关系？"** 子段，列出 2–4 个具体下游场景
- [ ] **第三节有数字**（训练时间 / GPU / 数据量），不能全是定性描述
- [ ] **第四节"为什么不简单"有 2–3 个具体难点**，不是一段空话
- [ ] **核心想法 3 条编号**，每条都讲做法 + 为什么 work
- [ ] **方法节有公式**（保留原符号）和**训练配置**（GPU / 时长）
- [ ] **实验节按数据集组织**，每个数据集有"关键观察"小段
- [ ] **消融**列出每个被 ablate 掉的组件 → 退化 metric
- [ ] **局限节**对应到原文 limitations 每条
- [ ] **译者注**至少 2 个子节，含一个跨卡片矩阵或设计判断
- [ ] **图片**：每张图都有 caption 含论文页码 + 关键观察
- [ ] **README.md 已更新**（卡片清单表新增 / 修改一行）
- [ ] **没有半成品节** "TODO" / "待补" / "略"

任意一条不达标视为未完成。

## 与其他 skill 的协作

- **paper-translate**：本 skill 产出"叙事式重组"，paper-translate 产出"原文 section 对齐的忠实中译"。同一张卡片末尾可以再追加 `## 中文技术笔记` 节由 paper-translate 写——前者讲故事、后者备查。
- **idea-research-skill Phase 2 (Compile Wiki)**：本 skill 是 Phase 2 的一种具体写法，更长更深。如果只想要轻量 wiki 摘要，走 idea-research-skill 默认；要写"足够日后回看"的版本，走本 skill。

## 范例参考

`idea-research/ideas/interactive-dt/wiki/papers/artgs.md` 是本 skill 的标杆样本——10 节都齐、figure caption 详细、译者注有矩阵。新写卡片时**先扫一眼这份**对齐风格。

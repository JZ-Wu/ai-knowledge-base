# Tokenizer 详解

Tokenizer 是 LLM 的第一步：把原始文本切分成 token 序列，每个 token 对应词表中的一个 ID。

---

## 一、为什么不直接按字符或按词切分？

| 方案 | 问题 |
|---|---|
| 按字符 | 序列太长（一句话几百个字符），注意力计算 $O(N^2)$ 爆炸；单个字符语义太弱 |
| 按词（空格分割）| 词表太大（英语几十万词），OOV（未登录词）问题严重；对中文/日文等无空格语言无法处理 |
| **Subword** | 折中：常用词完整保留，罕见词拆成子词片段。词表大小可控（32K~150K），无 OOV |

现代 LLM 全部使用 **Subword Tokenization**。

---

## 二、BPE (Byte Pair Encoding)

GPT 系列、LLaMA、大多数现代 LLM 使用的方法。

### 2.1 训练过程（构建词表）

```
初始词表: 所有单个字符（或字节）

训练语料: "low lower newest widest"

Step 1: 统计所有相邻 token 对的出现频率
        ('l','o')=2, ('o','w')=2, ('e','s')=2, ('s','t')=2, ...

Step 2: 合并频率最高的一对 → ('e','s') → 'es'
        词表新增: 'es'

Step 3: 重新统计频率
        ('es','t')=2, ('l','o')=2, ...

Step 4: 合并 ('es','t') → 'est'
        词表新增: 'est'

Step 5: 继续合并 ('l','o') → 'lo', ...

重复直到词表达到目标大小 (如 32000)
```

### 2.2 推理过程（文本→token）

用训练时记录的合并规则，按优先级（训练顺序）逐步合并字符：

```
输入: "lowest"
初始: ['l', 'o', 'w', 'e', 's', 't']
规则1: e+s → es  →  ['l', 'o', 'w', 'es', 't']
规则2: es+t → est →  ['l', 'o', 'w', 'est']
规则3: l+o → lo  →  ['lo', 'w', 'est']
规则4: lo+w → low →  ['low', 'est']
结果: ['low', 'est'] → [token_id_1, token_id_2]
```

### 2.3 完整 BPE 训练实例

用一个更完整的例子演示整个过程：

```
训练语料 (带词频):
  "hug"  : 10 次
  "pug"  : 5 次
  "pun"  : 12 次
  "bun"  : 4 次
  "hugs" : 5 次

Step 0: 初始化
  词表: {h, u, g, p, n, b, s}  (所有出现的字符)
  所有词拆为字符:
    h u g  : 10
    p u g  : 5
    p u n  : 12
    b u n  : 4
    h u g s : 5

Step 1: 统计所有相邻 pair 的频率
    (h,u)=15, (u,g)=20, (p,u)=17, (u,n)=16, (b,u)=4, (g,s)=5
    最高频: (u,g) = 20 → 合并为 "ug"

Step 2: 更新语料表示, 重新统计
    h ug   : 10
    p ug   : 5
    p u n  : 12
    b u n  : 4
    h ug s : 5
    (h,ug)=15, (p,ug)=5, (p,u)=12, (u,n)=16, (b,u)=4, (ug,s)=5
    最高频: (u,n) = 16 → 合并为 "un"

Step 3:
    h ug   : 10
    p ug   : 5
    p un   : 12
    b un   : 4
    h ug s : 5
    (h,ug)=15, (p,ug)=5, (p,un)=12, (b,un)=4, (ug,s)=5
    最高频: (h,ug) = 15 → 合并为 "hug"

Step 4:
    hug    : 10
    p ug   : 5
    p un   : 12
    b un   : 4
    hug s  : 5
    ...继续直到词表达到目标大小

最终词表: {h, u, g, p, n, b, s, ug, un, hug, pun, ...}
合并规则 (有序): [(u,g)→ug, (u,n)→un, (h,ug)→hug, ...]
```

**推理时**按合并规则的优先级（训练时的合并顺序）执行：

```
输入: "bugs"
初始: [b, u, g, s]
规则1 (u+g→ug): [b, ug, s]     ← 命中！
规则2 (u+n→un): 不适用
规则3 (h+ug→hug): 不适用
后续规则都不适用
最终: [b, ug, s] → [id_b, id_ug, id_s]

输入: "pugn" (不存在的词也能处理)
初始: [p, u, g, n]
规则1 (u+g→ug): [p, ug, n]
后续: 没有 (p,ug) 或 (ug,n) 的合并规则
最终: [p, ug, n]
```

### 2.4 Byte-level BPE

GPT-2 开始使用：初始词表不是字符，而是 **256 个字节值**（0x00~0xFF）。

```
Character-level BPE:
  初始词表 = 所有出现的 Unicode 字符
  问题: 遇到没见过的字符 → UNK token

Byte-level BPE:
  初始词表 = 256 个字节 (0x00~0xFF)
  任何文本都能用 UTF-8 编码为字节序列 → 永远不会有 UNK

UTF-8 编码示例:
  'A'  → [0x41]                    → 1 字节 → 1 个初始 token
  '你' → [0xE4, 0xBD, 0xA0]       → 3 字节 → 3 个初始 token
  '😀' → [0xF0, 0x9F, 0x98, 0x80] → 4 字节 → 4 个初始 token

训练后: 常用中文字的 3 个字节会被合并为 1 个 token
        → "你好" 可能是 1 个 token（如果足够常见）
```

**GPT-2 的字节映射技巧（bytes_to_unicode）**：

直接用 256 个字节做词表有个实际问题：其中很多是**不可打印的控制字符**（NULL、换行、DEL 等），在词表文件和调试界面里会显示异常或搞乱格式。

GPT-2 的解决方案是建一张 **256 → 256 的双射映射表**：

```
映射规则:
  ① 可打印 ASCII (33~126: ! ~ ~)         → 保持不变
  ② 扩展 Latin   (161~172: ¡~¬, 174~255: ®~ÿ) → 保持不变
  ③ 其余 (控制字符、空格等，共 68 个)      → 映射到 Unicode Ā(U+0100) 开始的连续区域

具体例子:
  0x00 (NULL)  → Ā (U+0100)
  0x01 (SOH)   → ā (U+0101)
  0x0A (换行)  → Ċ (U+010A)
  0x0D (回车)  → č (U+010D)
  0x20 (空格)  → Ġ (U+0120)    ← 这就是为什么 GPT-2 词表里空格显示为 Ġ
  0x41 ('A')   → A              ← 可打印字符，不变
```

这样词表中每个 base token 都是**人眼可读**的，不会出现空白或乱码。解码时反向映射回原始字节即可，完全无损。

> GPT-2 源码中这个函数就叫 `bytes_to_unicode()`，只有几行，返回一个 `{int: str}` 的字典。

---

## 三、WordPiece

BERT 使用的方法，和 BPE 非常相似，核心区别在**合并策略**：

| | BPE | WordPiece |
|---|---|---|
| 合并依据 | 频率最高的 pair | **互信息最高**的 pair |
| 公式 | count(AB) 最大 | count(AB) / (count(A) × count(B)) 最大 |
| 效果 | 优先合并高频组合 | 优先合并"在一起出现概率远高于独立出现"的组合 |
| 子词标记 | 无特殊标记 | 非首子词用 `##` 前缀，如 `play` + `##ing` |

### 3.1 互信息（Pointwise Mutual Information）公式

WordPiece 的合并得分本质上是<strong>逐点互信息 (PMI)</strong>：

$$\text{score}(A, B) = \frac{P(AB)}{P(A) \cdot P(B)} = \frac{\frac{\text{count}(AB)}{N}}{\frac{\text{count}(A)}{N} \cdot \frac{\text{count}(B)}{N}} = \frac{\text{count}(AB) \cdot N}{\text{count}(A) \cdot \text{count}(B)}$$

其中 $N$ 是语料中 pair 的总数。由于对所有 pair 而言 $N$ 是常数，排序时可以省略，简化为：

$$\text{score}(A, B) = \frac{\text{count}(AB)}{\text{count}(A) \times \text{count}(B)}$$

**直觉理解**：

```
语料: "un" 出现 100 次, "able" 出现 200 次, "unable" 出现 80 次

BPE 得分:     count("unable") = 80
WordPiece 得分: 80 / (100 × 200) = 0.004

对比: "th" 出现 5000 次, "e" 出现 8000 次, "the" 出现 4000 次
BPE 得分:     count("the") = 4000  ← BPE 优先合并这个（频率更高）
WordPiece 得分: 4000 / (5000 × 8000) = 0.0001  ← 得分远低于 "unable"

→ WordPiece 认为 "un"+"able" 的结合更有意义:
  它们一起出现的概率 (80%) 远高于随机组合的预期
  而 "the" 虽然高频，但 "th" 和 "e" 各自也极高频，共现不意外
```

**PMI 为什么重要——不只是"换了个排序公式"**：

虽然代码上只改了一行公式，但对最终词表的构成影响是系统性的：

| | BPE（纯频率） | WordPiece（PMI） |
|---|---|---|
| `"the"` (th+e) | 高频 → 最先合并 | th 和 e 各自都超高频，共现不意外 → PMI 得分低 |
| `"unable"` (un+able) | 频率一般 → 排后面 | un 和 able 一起出现的比例远超随机预期 → PMI 得分高 |
| 词表倾向 | 充斥 the, ing, tion 等高频但无语义的碎片 | 保留 un-, -able, -ment 等有语义意义的词缀 |

核心区别：PMI 让 WordPiece 把一个 token 当作一个**有意义的语义单元**来构建，而不是简单地按出现次数堆叠。这直接影响下游模型学到的 token embedding 质量——一个承载完整语义的 token（如 `unable`）比一个拼凑的高频片段（如 `the`）更容易学到有用的表征。

```
WordPiece 示例:
"unaffable" → ['un', '##aff', '##able']
```

---

## 四、SentencePiece

一个**实现框架**，而非具体算法。支持 BPE 和 Unigram 两种算法。

### 4.1 与传统 Tokenizer 的关键区别

| | 传统 BPE/WordPiece | SentencePiece |
|---|---|---|
| 预处理 | 先按空格分词，再在词内做 subword | **直接在原始文本上操作**，空格也是字符（用 ▁ 表示） |
| 语言依赖 | 依赖空格分词（对中日文不友好） | **语言无关**，中英文统一处理 |
| 可逆性 | 不一定可逆 | **完全可逆**（detokenize 能还原原始文本） |

### 4.2 SentencePiece 的工作原理

**核心思想**：把整个输入（包括空格）视为一个**字符序列**，不做任何预分词。

```
传统 BPE 的处理流程 (两阶段):
  "I love New York"
  → 预分词: ["I", "love", "New", "York"]   ← 按空格切开
  → 对每个词做 BPE: ["I", "lo", "ve", "New", "York"]
  → 问题: 空格信息丢失了！"New York" 和 "NewYork" 分不开

SentencePiece 的处理流程 (单阶段):
  "I love New York"
  → 归一化: "▁I▁love▁New▁York"            ← 空格替换为特殊符号 ▁ (U+2581)
  → 直接在整个序列上做 BPE/Unigram
  → ["▁I", "▁love", "▁New", "▁York"]
  → 还原: 把 ▁ 换回空格 → "I love New York"  ← 完全可逆！
```

**为什么可逆**：空格被编码为 ▁，信息没有丢失。传统方法丢弃了空格，重建时只能靠猜。

```
传统 BPE 的不可逆问题:
  tokenize("New York")   → ["New", "York"]
  tokenize("NewYork")    → ["New", "York"]   ← 相同结果！无法区分
  detokenize(["New", "York"]) → "New York" 还是 "NewYork"?  不确定

SentencePiece 无此问题:
  tokenize("New York")   → ["▁New", "▁York"]
  tokenize("NewYork")    → ["▁New", "York"]   ← ▁ 的有无标记了空格
  detokenize(["▁New", "▁York"]) → "New York"  ✅ 确定
  detokenize(["▁New", "York"])  → "NewYork"   ✅ 确定
```

**对中文的处理**：

```
传统 BPE:
  "我喜欢AI" → 无法按空格分词 → 需要额外的中文分词工具（jieba等）
  → 分词质量影响 tokenizer 质量，引入额外依赖

SentencePiece:
  "我喜欢AI" → "▁我喜欢AI" → 直接在字符序列上做 BPE/Unigram
  → 不需要中文分词工具，中英文统一处理
  → 可能产出: ["▁我", "喜欢", "AI"] 或 ["▁我", "喜", "欢", "AI"]
     取决于训练语料中的频率/概率
```

### 4.3 SentencePiece 训练配置

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input='corpus.txt',
    model_prefix='tokenizer',
    vocab_size=32000,
    model_type='bpe',          # 'bpe' 或 'unigram'
    character_coverage=0.9995, # 字符覆盖率，中日文建议 0.9995
    byte_fallback=True,        # 未覆盖字符用 UTF-8 字节表示（类似 Byte-level BPE）
    normalization_rule_name='identity',  # 不做额外归一化
)

# 产出: tokenizer.model (二进制模型) + tokenizer.vocab (词表)
```

LLaMA、T5、Qwen 等都使用 SentencePiece。

### 4.2 Unigram 算法（SentencePiece 的另一种选择）

和 BPE **方向相反**：

```
BPE:     从小到大（字符 → 合并 → 子词）    自底向上
Unigram: 从大到小（大词表 → 删减 → 小词表）  自顶向下
```

#### Unigram 训练的详细过程

**核心假设**：Unigram 语言模型假设每个子词独立出现，一个句子 $X$ 被切分为子词序列 $\mathbf{x} = (x_1, x_2, ..., x_n)$ 的概率是：

$$P(\mathbf{x}) = \prod_{i=1}^{n} P(x_i)$$

其中 $P(x_i)$ 是子词 $x_i$ 在词表中的概率。训练的目标就是找到一个词表 $V$，使得整个语料的 log-likelihood 最大：

$$\mathcal{L} = \sum_{s=1}^{|D|} \log P(X^{(s)}) = \sum_{s=1}^{|D|} \log \left( \sum_{\mathbf{x} \in S(X^{(s)})} P(\mathbf{x}) \right)$$

**符号说明**：
- $s$ = 句子的编号，从 1 开始
- $|D|$ = 语料库 D 中句子的总数
- $X^{(s)}$ = 第 s 个句子
- $S(X^{(s)})$ = 句子 $X^{(s)}$ 所有可能的切分方式的集合
- $\sum_{s=1}^{|D|}$ 的意思就是：**从第 1 个句子加到第 |D| 个句子**

> 比如语料库有 3 个句子 ["I love cats", "hello world", "good morning"]，那 $|D|=3$：
>
> $$\mathcal{L} = \log P(\text{"I love cats"}) + \log P(\text{"hello world"}) + \log P(\text{"good morning"})$$
>
> 就是把每个句子的 log 概率全部加起来，$\mathcal{L}$ 越大说明词表对语料的解释能力越强。

**逐层拆解这个公式**：

从最内层往外看：

**① 单条切分路径的概率** $P(\mathbf{x})$：假设子词之间相互独立，一种切分方式的概率就是各 token 概率的连乘。例如 "cats" 切成 `["ca", "ts"]`：

$$P(\text{"ca","ts"}) = P(\text{ca}) \times P(\text{ts}) = 0.01 \times 0.008 = 0.00008$$

**② 一个词/句子的边际概率** $P(X^{(s)}) = \sum_{\mathbf{x} \in S(X^{(s)})} P(\mathbf{x})$：同一个词有**多种合法切分方式**，把每种切法的概率**全部加起来**就是这个词的总概率。例如 "cats" 的所有切分：

```
切分方式              概率
["c","a","t","s"]    0.1 × 0.2 × 0.15 × 0.3  = 0.0009
["ca","ts"]          0.01 × 0.008             = 0.00008
["cat","s"]          0.05 × 0.3               = 0.015
["cats"]             0.002                     = 0.002
...（其他切法）
─────────────────────────────────────────────
P("cats") = 0.0009 + 0.00008 + 0.015 + 0.002 + ... = 0.01798...
```

> 这就是 Unigram 和 BPE 最本质的区别：**BPE 只保留一种确定的切法，Unigram 考虑所有可能的切法并求和**。

**③ 整个语料的 log-likelihood** $\mathcal{L} = \sum_{s=1}^{|D|} \log P(X^{(s)})$：对语料中**每一个词/句子**，取其边际概率的对数（log 把连乘变成求和，数值更稳定），然后全部加起来。$|D|$ 是语料中词/句子的总数。

- $\mathcal{L}$ 越大（越接近 0），说明当前词表对语料的"解释能力"越强
- 训练目标就是调整词表 $V$ 和每个子词的概率 $P(x_i)$，让 $\mathcal{L}$ 最大化

---

**Step 1：初始化大候选词表**

```
语料: ["cat", "cats", "car"]

提取所有子串（限制最大长度，比如 16 个字符）:
  "cat"  → c, a, t, ca, at, cat
  "cats" → c, a, t, s, ca, at, ts, cat, ats, cats
  "car"  → c, a, r, ca, ar, car

初始词表（去重）:
  V = {c, a, t, s, r, ca, at, ts, ar, cat, ats, cats, car}
  |V| ≈ 几十万（实际语料上）

加上所有单个字符/字节，确保任何输入都能被切分
```

实际中，初始词表通常是语料中出现过的**所有子串**（限制最大长度），大小可达数十万。SentencePiece 默认初始词表大小为 100 万。

---

**Step 2：EM 算法估计每个子词的概率**

这一步是整个 Unigram 训练的核心。给定当前词表，用 EM 迭代求每个子词的最优概率。

> **EM 算法回顾（Expectation-Maximization）**
>
> 当你想优化一个目标，但有些变量你**观察不到**（称为隐变量）时，直接优化很难。EM 的策略是把一个难问题拆成两个简单步骤交替执行：
>
> | 步骤 | 做什么 | 直觉 |
> |------|--------|------|
> | **E 步** | 用当前参数，推断隐变量的分布 | "根据现有认知，猜一下看不见的东西" |
> | **M 步** | 假设 E 步的推断是对的，更新参数使目标最大化 | "假装猜对了，更新认知" |
> | **重复** | 用新参数再 E → 再 M，直到收敛 | 每轮猜得更准，参数更优 |
>
> **在 Unigram tokenizer 里的对应关系：**
> - **参数**：每个子词的概率 $P(x_i)$
> - **隐变量**：每个词到底用了哪种切分方式（"cats" 是 [cats] 还是 [cat,s] 还是 [c,a,t,s]？你不知道）
> - **观测数据**：语料中的每个词
> - **E 步**：用当前子词概率，算出每种切分方式的后验概率
> - **M 步**：根据后验概率，统计每个子词的期望出现次数，重新估计概率
>
> **数学保证**：每一轮 EM 迭代，目标函数（log-likelihood）都**单调不减**，即 $\mathcal{L}^{(t+1)} \geq \mathcal{L}^{(t)}$。这保证了算法一定会收敛（虽然可能收敛到局部最优）。

<strong>E 步（Expectation）</strong>：对语料中每个词，枚举所有可能的切分方式，计算每种切分的概率：

```
当前词表概率（按语料中出现频率初始化，不是均匀 1/n）:
  P(x_i) = count(x_i) / Σ count(x_j)

  例如:
  P(c)=0.15, P(a)=0.15, P(t)=0.12, P(s)=0.08
  P(ca)=0.10, P(at)=0.09, P(cat)=0.07, P(cats)=0.03 ...

对 "cats" 的所有切分:
  [c, a, t, s]   → P = 0.15 × 0.15 × 0.12 × 0.08 = 0.000216
  [ca, t, s]     → P = 0.10 × 0.12 × 0.08         = 0.00096
  [c, at, s]     → P = 0.15 × 0.09 × 0.08          = 0.00108
  [cat, s]       → P = 0.07 × 0.08                  = 0.0056
  [cats]         → P = 0.03                          = 0.03     ← 概率最大
  ...

归一化后，得到每种切分的后验概率:
  P([cats] | "cats")         ≈ 0.79
  P([cat, s] | "cats")      ≈ 0.15
  P([c, at, s] | "cats")    ≈ 0.03
  ...
```

<strong>M 步（Maximization）</strong>：根据 E 步的后验概率，重新估计每个子词的概率：

```
每个子词的新概率 ∝ 它在所有切分中的期望出现次数

比如子词 "cat" 的期望次数:
  = 在 "cat" 的各种切分中出现的加权次数
  + 在 "cats" 的各种切分中出现的加权次数（如 [cat, s] 这种切分贡献 0.15 次）
  + ...

归一化后得到新的 P(cat)
```

实际实现中用**前向-后向算法**（Forward-Backward）高效计算，不需要真的枚举所有切分。E 步和 M 步交替迭代几轮直到收敛。

---

**Step 3 & 4：计算每个子词的"重要性"并剪枝**

收敛后，对词表中每个子词，计算**如果把它移除，语料的 log-likelihood 会下降多少**：

$$\text{loss}(x_i) = \mathcal{L}_{\text{current}} - \mathcal{L}_{\text{without } x_i}$$

```
计算每个子词的重要性:
  loss("cat")  = -0.002   ← 移除后 likelihood 几乎不变（因为 c+a+t 能替代）
  loss("the")  = -1.85    ← 移除后大量句子的最优切分被破坏，损失很大
  loss("zzq")  = -0.0001  ← 几乎没影响

按 loss 排序，移除损失最小的 bottom 20%~30%:
  移除 "zzq", "cat" 等不重要的子词
  保留 "the", 所有单字符（永远不移除，保证覆盖率）

词表从 100万 → 70万 → 49万 → ... → 目标大小（如 32000）
```

关键细节：**单个字符/字节永远不会被移除**，确保任何输入都有兜底切分方案。

---

**Step 5：重复 Step 2-4**

每轮剪枝后，词表变小了，需要重新跑 EM 更新剩余子词的概率，然后再计算重要性、再剪枝。循环往复直到词表大小达到目标值（如 32K、64K）。

```
迭代过程:
  Round 1: |V| = 1,000,000 → EM收敛 → 剪枝 → |V| = 700,000
  Round 2: |V| = 700,000   → EM收敛 → 剪枝 → |V| = 490,000
  Round 3: |V| = 490,000   → EM收敛 → 剪枝 → |V| = 343,000
  ...
  Round N: |V| = 32,500    → EM收敛 → 剪枝 → |V| = 32,000 ✓ 达到目标
```

---

**推理时**：对一个输入词，可能有多种切分方式，用 **Viterbi 算法**（动态规划）找概率最大的切分：

```
输入 "unbreakable", 当前词表概率已知

Viterbi 动态规划:
  dp[0] = 0  (空串)
  dp[2] = log P("un") = -2.3
  dp[7] = max(dp[2] + log P("break"), dp[0] + log P("unbreak"), ...)
  dp[11] = max(dp[7] + log P("able"), dp[5] + log P("akable"), ...)

回溯得到最优切分: ["un", "break", "able"]
```

时间复杂度 $O(n^2)$（$n$ 为输入长度），实际中因为限制了最大子词长度，接近 $O(n)$。

#### Unigram 的核心优势

**1. 概率化建模 —— 天然支持多种切分**

BPE 对每个输入只有唯一确定的切分结果（贪心合并），而 Unigram 对同一个词可以给出多种切分及其概率：

```
BPE 切 "unbreakable":
  → ["un", "break", "able"]  （唯一结果，由合并规则决定）

Unigram 切 "unbreakable":
  → ["un", "break", "able"]    P = 0.42
  → ["un", "breakable"]        P = 0.31
  → ["unbreak", "able"]        P = 0.18
  → ["unbreakable"]            P = 0.09
  最终选 P 最大的，但训练时可以利用所有切分做期望（EM 算法）
```

这意味着：
- **训练时更鲁棒**：EM 算法考虑了所有可能的切分路径，不会因为贪心选择而错过更优的全局方案
- **可以做 subword regularization**：训练时故意按概率采样不同切分，相当于数据增强，让模型对切分方式不敏感（T5 论文中证实了这个 trick 有效）

**2. 全局最优 vs 贪心**

```
BPE:     每一步选当前最高频的 pair 合并 → 贪心，可能错过全局最优
Unigram: 每一步移除对全局 likelihood 影响最小的子词 → 更接近全局最优

类比：
  BPE     ≈ 贪心搜索
  Unigram ≈ 剪枝搜索（从完整解空间出发，逐步剪掉不重要的）
```

**3. 词表质量更高**

Unigram 移除子词时的标准是"移除后语料 likelihood 下降多少"。这意味着留下来的每个子词都经过了**信息论意义上的筛选**——它对整个语料的建模是不可或缺的。相比之下，BPE 只看频率，可能保留一些高频但冗余的 token。

#### 谁在用 Unigram？

| 模型/框架 | Tokenizer 算法 | 说明 |
|---|---|---|
| T5 / mT5 | SentencePiece + **Unigram** | Google 的经典选择 |
| ALBERT | SentencePiece + **Unigram** | |
| XLNet | SentencePiece + **Unigram** | |
| Gemma / Gemini | SentencePiece + **Unigram** | Google 最新模型仍在用 |
| LLaMA / LLaMA 2/3 | SentencePiece + **BPE** | Meta 选的是 BPE 模式 |
| GPT 系列 | tiktoken (Byte-level **BPE**) | OpenAI 一直用 BPE |
| Qwen | tiktoken (**BPE**) | |
| DeepSeek | HuggingFace tokenizers (**BPE**) | |

**总结**：Google 系模型偏好 Unigram，OpenAI/Meta/国内大模型偏好 BPE。两种方法都是主流方案，并没有谁淘汰谁。选择更多取决于团队习惯和工程生态，性能差异在大模型时代已经不显著（模型够大，tokenizer 的差异会被"吸收"掉）。

---

## 五、中文 Tokenization 的特殊问题

### 5.1 为什么中文 token 效率低？

```
英文: "Hello world" → ['Hello', '▁world'] → 2 tokens
中文: "你好世界"    → ['你好', '世界'] 或 ['你', '好', '世', '界'] → 2~4 tokens

UTF-8 编码:
  英文字母 = 1 byte
  中文汉字 = 3 bytes
```

如果词表中中文 token 少（如 GPT-2 词表几乎没有中文 token），一个汉字可能被拆成 3 个字节 token → **同样的语义，中文用的 token 数是英文的 3 倍**。

### 5.2 解决方案

- **扩充中文词表**：LLaMA-Chinese、ChatGLM 等对原始词表扩充几万个中文 token
- **从头训练**：Qwen（152K 词表）、DeepSeek（129K 词表）从头用中英混合语料训练 tokenizer，中文效率接近英文

### 5.3 词表大小的 trade-off

| 词表大小 | 优势 | 劣势 |
|---|---|---|
| 小（32K） | Embedding 参数少，训练快 | 中文/多语言效率低 |
| 大（150K） | 多语言覆盖好，序列短 | Embedding 层参数多，softmax 计算量大 |

---

## 六、特殊 Token

### 6.1 常见特殊 Token

```
通用:
  <bos> / <s>     : 序列开始 (Beginning of Sequence)
  <eos> / </s>    : 序列结束 (End of Sequence)
  <pad>           : 填充 (Padding)，batch 中对齐不同长度序列
  <unk>           : 未知 token（Byte-level BPE 不需要）

Chat 模板相关:
  <|im_start|>    : 对话消息开始 (Qwen 格式)
  <|im_end|>      : 对话消息结束
  <|system|>      : system 角色标记
  <|user|>        : 用户角色标记
  <|assistant|>   : 助手角色标记

特殊用途:
  <|endoftext|>   : 文档边界 (GPT 系列预训练时用)
  <tool_call>     : 工具调用开始
  <think>         : 思考过程标记 (DeepSeek-R1)
```

### 6.2 Chat Template 与 Tokenizer 的关系

```
用户输入:
  messages = [
    {"role": "system", "content": "你是一个助手"},
    {"role": "user", "content": "你好"},
  ]

Qwen 格式:
  <|im_start|>system\n你是一个助手<|im_end|>\n
  <|im_start|>user\n你好<|im_end|>\n
  <|im_start|>assistant\n

LLaMA-3 格式:
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
  你是一个助手<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
  你好<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n

不同模型的 Chat Template 不同 → 用错模板会导致严重的性能下降
HuggingFace tokenizer 内置 apply_chat_template() 方法自动处理
```

---

## 七、词表扩充（继续预训练时的关键问题）

### 7.1 为什么要扩充词表？

```
场景: 用 LLaMA (英文词表 32K) 做中文模型

原始词表:  "你好" → [0xE4, 0xBD, 0xA0, 0xE5, 0xA5, 0xBD] → 6 个字节 token
扩充后:    "你好" → [你好] → 1 个 token

效果:
  - 中文文本序列长度缩短 3-5 倍
  - 相同上下文窗口能处理更多中文内容
  - 推理速度提升（token 数少了）
```

### 7.2 词表扩充的步骤

```
1. 在目标语言语料上训练新的 tokenizer → 获得新词表
2. 取新词表和旧词表的并集 → 新增 token 加入词表末尾
3. 扩展 Embedding 矩阵:
   原始 Embedding: (V_old × d)
   扩展后: (V_new × d)
   新 token 的 embedding 初始化:
     方案 A: 随机初始化 → 需要更多训练
     方案 B: 用组成该 token 的子 token 的 embedding 平均 → 更好的初始值
4. 扩展 LM Head (输出层) 同理
5. 在目标语言数据上继续预训练 (恢复新 token 的表示能力)

注意: 词表扩充后必须做足够的继续预训练,否则新 token 的表示质量很差
```

**为什么不用改模型结构？** Embedding 本质就是一个查找表：输入 token ID → 取矩阵对应行的 `d` 维向量。扩充词表只是给矩阵**加几行**（`(32000, 4096)` → `(50000, 4096)`），维度 `d` 不变，因此后面所有 attention、FFN 层的输入维度不变，模型结构完全不用动。LM Head 同理，只是加几列。

**是继续预训练，不是从头训**：除了新 token 的 embedding 和 LM Head 对应行/列是新的，其他所有参数都保留原模型。训练量通常是原始预训练的 5-10%，原模型的知识（英文能力、推理能力等）大部分被保留。

### 7.3 典型案例

| 模型 | 基座 | 原始词表 | 扩充后 | 新增 |
|---|---|---|---|---|
| Chinese-LLaMA | LLaMA-7B | 32K | 49953 | ~18K 中文 token |
| Chinese-Alpaca | LLaMA-7B | 32K | 49953 | ~18K 中文 token |
| ChatGLM | 从头训练 | 130344 | — | 中英混合训练 |

---

## 八、Tokenizer 对模型性能的影响

### 8.1 Fertility（生育率）

```
Fertility = 平均每个词被切分成多少个 token

英文对英文词表: fertility ≈ 1.2 (大部分词是 1 个 token)
中文对英文词表: fertility ≈ 3.0 (每个字 ~3 个 token)
中文对中文词表: fertility ≈ 1.5 (常用词是 1 个 token)

Fertility 越低:
  - 相同上下文窗口能看更多内容
  - 推理 token 数更少 → 更快更便宜
  - 每个 token 承载更多语义 → 可能学得更好
```

### 8.2 Tokenizer 的常见 Bug

```
1. 训练和推理用了不同版本的 tokenizer → 灾难
2. 特殊 token 处理不一致 (add_special_tokens 参数)
3. 空格处理不一致:
   "Hello World" 和 " Hello World" 可能产生不同的 token
4. 数字切分:
   "123456" → ["123", "456"] 或 ["12", "34", "56"]
   不同切分影响数学能力
```

---

## 九、面试常见问题

> **Q: BPE 的训练过程？**
> A: 从字节/字符出发，每轮统计相邻 pair 频率，合并最高频的 pair 加入词表，重复直到词表达到目标大小。

> **Q: BPE 和 WordPiece 的区别？**
> A: BPE 按频率合并，WordPiece 按互信息合并（偏好共现概率远高于独立概率的 pair）。

> **Q: 为什么用 SentencePiece？**
> A: 语言无关，直接处理原始文本（不依赖空格分词），对中文等语言友好，且 tokenization 完全可逆。

> **Q: 为什么有些模型中文 token 效率低？怎么解决？**
> A: 词表中中文 token 少，汉字被拆成字节级 token（1 个汉字 3 个 token）。解决：扩充词表或从头用中文语料训练 tokenizer。

---

**相关文档**：
- [Transformer架构详解](Transformer架构详解.md) — Token Embedding 层如何处理 token ID

[返回上级](README.md) | [返回总目录](../../README.md)

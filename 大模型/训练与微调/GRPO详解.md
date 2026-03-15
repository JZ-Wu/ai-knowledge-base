# GRPO 详解 (Group Relative Policy Optimization)

GRPO 由 DeepSeek 团队提出（DeepSeek-Math, 2024），并在 DeepSeek-R1 中大规模应用。核心思想是**用组内相对奖励替代 Critic 模型**，大幅简化 RL 训练。

## 一、动机：PPO 太重，DPO 太弱

| 方法 | 问题 |
|------|------|
| PPO | 需要 4 个模型（Actor + Critic + RM + Ref），显存爆炸 |
| DPO | 离线学习，无法在线探索，难以用于推理能力训练 |

GRPO 的定位：**保留 RL 在线探索的优势，但去掉 Critic 模型**。

## 二、核心方法

### 采样阶段

对每个 prompt $x$，用当前策略 $\pi_\theta$ 采样 **一组** 回答（通常 $G$ = 8~64 个）：

$$\{y_1, y_2, ..., y_G\} \sim \pi_\theta(\cdot | x)$$

### 奖励计算

对每个回答用奖励函数打分：$r_1, r_2, ..., r_G$

奖励函数可以是：
- **规则奖励**（数学题：答案是否正确；代码题：是否通过测试）
- **奖励模型** (RM)
- **混合奖励**

### 组内归一化（关键创新）

不用 Critic 估计基线，而是直接用 **组内统计量** 归一化：

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, ..., r_G)}{\text{std}(r_1, ..., r_G)}$$

> 这就是 "Group Relative" 的含义：优势是相对于同组其他回答计算的。

### 优化目标

$$L_{GRPO} = -\frac{1}{G} \sum_{i=1}^{G} \left[ \min\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \hat{A}_i, \; \text{clip}(\cdot, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{ref}) \right]$$

与 PPO 类似的 clipped objective，但：
- 用组内归一化的 $\hat{A}_i$ 替代 GAE 计算的优势
- 不需要 Critic 模型

## 三、与 PPO 的对比

| 维度 | PPO | GRPO |
|------|-----|------|
| 模型数量 | 4 个（Actor + Critic + RM + Ref） | 3 个（Actor + RM + Ref）或 2 个（用规则奖励时不需 RM） |
| 优势估计 | Critic 模型 + GAE | 组内均值/标准差归一化 |
| 显存 | 极高 | 显著降低 |
| 训练稳定性 | Critic 训练不稳定 | 更稳定（统计量直接计算） |
| 采样效率 | 每个 prompt 采样 1 个 | 每个 prompt 采样 G 个 |
| 计算开销 | 推理少，训练复杂 | 推理多（G 倍采样），训练简单 |

## 四、GRPO 在 DeepSeek-R1 中的应用

DeepSeek-R1 展示了 GRPO 的强大效果：

### 训练流程

**DeepSeek-V3 Base → 冷启动 SFT → GRPO 阶段一 → 拒绝采样+SFT → GRPO 阶段二 → DeepSeek-R1**

| 阶段 | 内容 |
|------|------|
| 冷启动 SFT | 少量 CoT 数据微调 |
| GRPO 第一阶段 | 推理能力：数学/代码规则奖励，模型自发学会 CoT 和反思 |
| 拒绝采样 + SFT | 用 RL 模型生成高质量数据，再做 SFT |
| GRPO 第二阶段 | 全面对齐：推理用规则奖励，通用用 RM 奖励，安全/格式用规则奖励 |

### 关键发现

1. **纯 RL 也能涌现推理能力**：不需要 CoT 标注数据，GRPO 训练自发产生了思维链
2. **规则奖励足矣**：对于数学和代码，简单的对错判断就够了，不需要复杂的 RM
3. **可扩展性好**：组采样天然适合大规模并行

## 五、GRPO 的适用场景

| 场景 | 适合度 | 原因 |
|------|--------|------|
| 数学推理 | 非常适合 | 有明确的对错标准作为规则奖励 |
| 代码生成 | 非常适合 | 可以通过测试用例自动判断 |
| 通用对话 | 适合（需 RM） | 没有规则奖励时需要训练 RM |
| 安全对齐 | 适合 | 可以用规则 + RM 混合奖励 |

## 六、实现要点

- **采样温度**：通常 0.7~1.0，保证组内回答的多样性
- **组大小 $G$**：越大统计量越稳定，但计算开销越大（典型 16~64）
- **奖励设计**：数学题推荐用 outcome-based reward（只看最终答案），过程奖励 (PRM) 效果仍在研究中
- **KL 系数 $\beta$**：与 PPO 类似，防止策略崩溃
- **框架**：OpenRLHF、veRL、TRL 均已支持

---

**相关文档**：
- [预训练与后训练](预训练与后训练.md)
- [RLHF与PPO详解](RLHF与PPO详解.md)
- [DPO详解](DPO详解.md)

[返回上级](README.md) | [返回总目录](../../README.md)

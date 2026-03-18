# RLHF 与 PPO 详解

RLHF (Reinforcement Learning from Human Feedback) 是 OpenAI 在 InstructGPT (2022) 中提出并推广的对齐方法，通过强化学习让模型输出符合人类偏好。

## 一、为什么需要 RLHF？

SFT 只能模仿固定答案，但无法学习**什么是更好的回答**。人类偏好是一个相对概念——不是"对错"，而是"更好还是更差"。

| 场景 | SFT 能做到 | RLHF 能做到 |
|------|-----------|-------------|
| 回答正确性 | 能模仿正确答案 | 能学会判断正确性 |
| 回答风格 | 模仿固定风格 | 学会人类偏好的风格 |
| 安全对齐 | 靠过滤训练数据 | 主动学会拒绝有害请求 |
| 超越标注者 | 不能，上限是标注数据 | 理论上可以（通过探索） |

### RLHF 的历史脉络

- **2017**: Deep RL from Human Preferences (Christiano et al.) — 首次将人类偏好反馈引入深度 RL，在 Atari 游戏和 MuJoCo 上验证
- **2020**: Learning to summarize from human feedback (Stiennon et al., OpenAI) — 首次将 RLHF 应用到 NLP 任务（文本摘要）
- **2022**: InstructGPT (Ouyang et al., OpenAI) — 将 RLHF 系统化地应用到大语言模型对齐，奠定三阶段范式
- **2022**: ChatGPT — 基于 InstructGPT 的方法进行对齐，引爆大模型应用
- **2023**: Llama 2 (Meta) — 开源模型中首次大规模使用 RLHF，公开了详细的训练细节
- **2023-2024**: 出现 DPO、KTO、GRPO 等替代方案，但 RLHF+PPO 仍是主流选择之一

### 从 MLE 到 RLHF 的直觉

SFT 的训练目标是最大似然估计 (MLE)：

$$L_{SFT} = -\sum_t \log \pi_\theta(y_t | x, y_{<t})$$

各符号含义：
- $L_{SFT}$：SFT 的损失函数，训练时最小化
- $\pi_\theta$：带参数 $\theta$ 的语言模型（策略），输出下一个 token 的概率分布
- $x$：输入的 prompt
- $y_t$：标准答案中第 $t$ 个 token
- $y_{<t}$：第 $t$ 个 token 之前的所有 token，即 $y_1, y_2, \dots, y_{t-1}$
- $\pi_\theta(y_t | x, y_{<t})$：模型在看到 prompt $x$ 和已生成的 $y_{<t}$ 后，预测第 $t$ 个 token 恰好是 $y_t$ 的概率
- $\sum_t$：对答案中每个 token 位置求和
- $-\log$：负对数，概率越高则 loss 越小

这相当于告诉模型："逐 token 模仿这个标准答案"——模型在每个位置都被逼着给正确 token 尽可能高的概率。问题在于：

1. **Exposure Bias**: 训练时每步都看到正确的前缀，推理时却要用自己之前生成的（可能有误的）token
2. **Loss-Metric Mismatch**: 优化的是 token 级别的交叉熵，但评估的是整体回答质量
3. **无法表达偏好**: 两个都还不错但风格不同的回答，MLE 无法表达"A 比 B 好"

RLHF 直接优化**序列级别的奖励信号**，绕过了这些问题。

## 二、RLHF 三步流程

```
Step 1: SFT           →  得到初始策略模型 π_SFT
Step 2: 训练奖励模型   →  得到 Reward Model (RM)
Step 3: PPO 强化学习   →  得到最终对齐模型 π_RL
```

### Step 1: SFT（已在 SFT 文档中详述）

关键点：SFT 模型的质量直接影响后续 RLHF 的效果。SFT 提供了一个合理的初始策略，使得 PPO 的探索空间不至于太大。如果 SFT 模型太差，PPO 需要更多的探索才能找到好的策略，训练效率极低。

### Step 2: 训练奖励模型 (Reward Model)

奖励模型是 RLHF 的核心组件——它将人类偏好"蒸馏"成一个可微分的函数，使得 RL 算法可以用梯度来优化。

#### 数据收集

1. 给定一个 prompt，让 SFT 模型生成 **多个** 不同的回答（通常 4\~9 个）
2. 人类标注者对这些回答进行 **排序**（不是打分）
3. 将排序转化为 **偏好对 (chosen, rejected)**

```
Prompt: "解释量子计算"
回答 A: [详细准确的解释]     ← chosen (更好)
回答 B: [不太准确的解释]     ← rejected (更差)
```

**为什么用排序而不是打分？**

- 人类对绝对分数的标注一致性很差（标注者 A 打 4 分的回答，标注者 B 可能打 3 分）
- 但对相对排序的一致性较好（"A 比 B 好"这种判断更容易达成共识）
- InstructGPT 数据显示，标注者间的排序一致率约 73%

**从 K 个回答的排序中提取偏好对**

如果对 K 个回答排序，可以提取 $\binom{K}{2}$ 个偏好对。InstructGPT 使用 K=4\~9，一次排序就能产生 6\~36 个训练样本。

> 注意：来自同一个 prompt 的偏好对不应被拆分到不同的 batch 中，否则会导致过拟合。InstructGPT 的做法是将同一 prompt 的所有偏好对放在同一个 batch 内。

#### 模型结构

- 通常基于 SFT 模型初始化，去掉最后的 language modeling head
- 换成一个 **标量输出头**（线性层），对整个回答输出一个标量分数
- 具体实现：取最后一个 token 的 hidden state，过一个 Linear(hidden_dim, 1) 得到标量

```python
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model  # 去掉 lm_head 的 SFT 模型
        self.reward_head = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask).last_hidden_state
        # 取序列最后一个有效 token 的表示
        last_idx = attention_mask.sum(dim=1) - 1
        last_hidden = hidden[torch.arange(hidden.size(0)), last_idx]
        reward = self.reward_head(last_hidden).squeeze(-1)
        return reward  # shape: (batch_size,)
```

**RM 的大小选择**

- InstructGPT 使用 6B 的 RM（主模型 175B）
- Llama 2 使用与主模型相同大小的 RM
- 经验规律：RM 太小则表达能力不够，太大则训练成本高且容易过拟合
- 通常选择主模型大小的 1/3 到同等大小

#### 训练目标

Bradley-Terry 模型，最大化好回答与差回答的分数差：

$$L_{RM} = -\frac{1}{\binom{K}{2}} \sum_{(i,j): y_i \succ y_j} \log \sigma(r_\theta(x, y_i) - r_\theta(x, y_j))$$

其中：
- $r_\theta(x, y)$：奖励模型对 prompt $x$ 和回答 $y$ 的打分
- $y_i \succ y_j$：表示 $y_i$ 被人类偏好优于 $y_j$
- $\sigma$：sigmoid 函数
- K：每个 prompt 的候选回答数量

简化为两两对比时：

$$L_{RM} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))$$

- $y_w$：人类偏好的回答 (chosen/winner)
- $y_l$：不被偏好的回答 (rejected/loser)

**直觉理解**：当 $r(y_w) \gg r(y_l)$ 时，$\sigma$ 趋近 1，loss 趋近 0；当 $r(y_w) \approx r(y_l)$ 时，梯度最大，模型被强烈推动去区分好坏。

#### RM 训练的关键细节

**过拟合问题**

RM 非常容易过拟合，尤其是偏好数据量有限时。常见缓解手段：
- 早停 (early stopping)：通常训练 1 个 epoch 就够了
- InstructGPT 发现训练多个 epoch 后，RM 在验证集上的准确率反而下降
- 适当的 dropout 和 weight decay

**标注质量控制**

- 多人标注同一组数据，计算标注者间的一致性 (inter-annotator agreement)
- InstructGPT 的标注者一致率约 73%，意味着约 27% 的样本存在分歧
- Llama 2 采用两阶段策略：先用更多标注者收集数据，再用高质量标注者过滤

**RM 的校准 (Calibration)**

- 奖励分数的绝对值没有意义，只有相对差值有意义
- 实践中通常对 RM 输出做归一化处理（即 whitening：$r_{norm} = \frac{r - \mu_r}{\sigma_r}$，将 batch 内分数标准化为均值 0、标准差 1）。因为 Bradley-Terry 模型只优化分数差，绝对分数没有意义；归一化后奖励尺度稳定，PPO 训练更不容易崩溃，也便于与 KL 惩罚项保持合理的量级平衡

### Step 3: PPO 优化

#### PPO 是什么

Proximal Policy Optimization（近端策略优化）是 OpenAI 于 2017 年提出的强化学习算法。它是 TRPO (Trust Region Policy Optimization) 的简化版本，核心思想是在更新策略时限制更新幅度，防止策略崩溃。

**从策略梯度到 PPO 的演化**

```
REINFORCE (1992)
  → Actor-Critic (引入 value baseline 降低方差)
    → A2C/A3C (并行采样 + advantage)
      → TRPO (信赖域约束，二阶优化)
        → PPO (clip 近似信赖域，一阶优化)
```

PPO 之所以被广泛采用，是因为它在性能和实现复杂度之间取得了最佳平衡：
- 比 REINFORCE 方差小得多
- 比 TRPO 实现简单得多（不需要二阶优化）
- 比 A2C 更稳定（有 clip 防止过大更新）

#### RL 基础概念速览

强化学习（RL）的核心思想是**试错学习**：一个"智能体"在"环境"中不断尝试不同的"动作"，环境给出"奖励"反馈（做得好就奖，做得差就罚），智能体的目标是学会一套"策略"让总奖励最大。最经典的例子是训练 AI 打游戏——AI 不知道规则，但通过得分/扣分来学习最优操作。

**RLHF 就是把这套框架套到语言模型上**：生成回答 = 打游戏，RM 打分 = 游戏得分，优化生成策略 = 学会赢的策略。与经典 RL 的关键区别在于：打游戏的奖励是游戏自带的，而语言模型没有天然的奖励信号，所以需要**先训一个 RM 来模拟人类打分**——这就是"Human Feedback"的含义。

**为什么不直接 SFT，要用 RL？** SFT 是给标准答案让模型抄，只能模仿不能超越；RL 是让模型自己写答案、老师只说好不好，模型可以**通过探索发现更优的回答方式**。

#### RLHF 中的 RL 框架映射

| RL 概念 | 在 RLHF 中的对应 | 详细说明 |
|---------|------------------|----------|
| **环境 (Environment)** | 用户的 prompt | 每个 episode 对应一个 prompt |
| **状态 (State)** | 当前已生成的 token 序列 | $s_t = (x, y_1, y_2, ..., y_t)$ |
| **动作 (Action)** | 生成下一个 token | $a_t = y_{t+1} \in \mathcal{V}$ (词表) |
| **策略 (Policy)** | 语言模型 $\pi_\theta$ | $\pi_\theta(a_t \| s_t)$ 即下一个 token 的概率分布 |
| **奖励 (Reward)** | RM 打分 + KL 惩罚 | 只在最后一个 token 处给非零奖励 |
| **轨迹 (Trajectory)** | 一个完整的回答 | $(s_0, a_0, s_1, a_1, ..., s_T, a_T)$ |
| **回合 (Episode)** | 从 prompt 到 EOS | 生成 EOS token 或达到最大长度时结束 |

**奖励的稀疏性**

在 RLHF 中，奖励信号是**极度稀疏**的——只有在生成完整个回答后，才能从 RM 获得一个标量奖励。中间每个 token 位置的奖励都是 0，只有最后一个 token（EOS）才拿到 RM 的分数。这就像打完一整局游戏才看到最终得分，中间每一步都不知道自己做得好不好。

> 实际上中间 token 还会有一个 KL 惩罚项（防止模型跑偏），最终每个位置的奖励公式见后文[为什么需要 KL 惩罚](/大模型/训练与微调/RLHF与PPO详解?id=为什么需要-kl-惩罚)一节。

#### 稀疏奖励的困境：我们还需要一个 Critic

到目前为止，三步流程给了我们三样东西：

1. **SFT 模型**（Step 1）→ 这就是我们要继续训练的语言模型，在 PPO 里叫 **Actor**
2. **Reward Model**（Step 2）→ 提供奖励信号，冻结不更新
3. **SFT 模型的冻结副本** → 用来计算 KL 惩罚（"你偏离原始模型多远了？"），叫 **Reference Model**

但还有一个问题没解决：**RM 只在生成完整回答后给一个总分（稀疏奖励），PPO 却需要知道每个 token 位置的贡献**——"这个 token 是好棋还是臭棋？"

这就需要第四个模型——**Critic（价值模型）**。Critic 学习预测"从当前位置到结束，预期能拿多少总分"，有了它，就能通过 GAE 算法反推出每个 token 的 advantage（"这一步比平均水平好多少"）。类比：RM 是裁判，打完整局才给总分；Critic 是解说员，每一步都在估计"目前局势如何"。

所以 PPO 训练需要**同时加载四个模型**：

#### InstructGPT 三阶段训练流程

![InstructGPT 三阶段: SFT → 奖励模型 → PPO (Ouyang et al., 2022)](assets/instructgpt_rlhf.png)

> 图源: *Training language models to follow instructions with human feedback (InstructGPT)*, Figure 2. Step 1: SFT 有监督微调; Step 2: 训练奖励模型; Step 3: PPO 强化学习优化。

#### 训练中涉及的四个模型

| 模型 | 是否更新 | 说明 |
|------|---------|------|
| **Actor (策略模型)** | ✅ 更新 | 正在训练的语言模型，参数 $\theta$，基于 SFT 模型初始化 |
| **Critic (价值模型)** | ✅ 更新 | 估计状态价值 $V_\phi(s)$，参数 $\phi$，通常基于 RM 初始化 |
| **Reference Model** | ❌ 冻结 | SFT 模型的副本，用于计算 KL 惩罚 |
| **Reward Model** | ❌ 冻结 | 训练好的奖励模型，提供奖励信号 |

> 同时加载 4 个模型，这就是 PPO 显存开销巨大的原因。以 7B 模型为例，4 个模型需要约 4 × 14GB = 56GB 显存（fp16），加上优化器状态和激活值，总共需要 80~100GB+。

**各模型的交互关系**

```
           ┌─────────────────┐
  prompt   │  Actor (π_θ)    │──→ 生成 response
  ────────→│  [可训练]        │
           └────────┬────────┘
                    │ response
         ┌──────────┼──────────┐
         ↓          ↓          ↓
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Critic   │ │ Reward   │ │ Reference│
  │ V_φ(s)   │ │ Model    │ │ Model    │
  │ [可训练]  │ │ [冻结]   │ │ [冻结]   │
  └────┬─────┘ └────┬─────┘ └────┬─────┘
       │ value      │ reward     │ KL penalty
       └────────────┼────────────┘
                    ↓
             计算 advantage
                    ↓
           更新 Actor & Critic
```

#### 优化目标

$$\max_\theta \; \mathbb{E}_{x \sim D, \, y \sim \pi_\theta(\cdot|x)} \left[ r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{ref}) \right]$$

- **第一项** $r_\phi(x, y)$：最大化奖励模型的打分
- **第二项** $\text{KL}(\pi_\theta \| \pi_{ref})$：KL 散度惩罚，防止模型偏离原始 SFT 模型太远
- **$\beta$**：KL 惩罚系数，控制探索与保守之间的平衡

展开 KL 散度：

$$\text{KL}(\pi_\theta \| \pi_{ref}) = \mathbb{E}_{y \sim \pi_\theta} \left[ \sum_{t=1}^{T} \log \frac{\pi_\theta(y_t | x, y_{<t})}{\pi_{ref}(y_t | x, y_{<t})} \right]$$

这是一个 token 级别的 KL 散度之和——每个 token 位置都计算当前策略和参考策略的概率比的对数。

<a id="为什么需要-kl-惩罚"></a>

#### 为什么需要 KL 惩罚？

没有 KL 约束时，模型会 **reward hacking**——找到奖励模型的漏洞，生成得分高但实际质量差的回答。例如：
- 无限重复某些让 RM 给高分的短语
- 生成过长的回答（RM 可能偏好长文本）
- 产生不自然但高分的文本
- 学会在特定位置插入"触发词"来提高 RM 分数

**KL 惩罚的作用**：强制策略模型不要偏离 SFT 模型太远，相当于一个"正则化"项。如果生成的 token 概率与 SFT 模型差异太大，就会受到惩罚。

**$\beta$ 的自适应调整 (Adaptive KL)**

InstructGPT 使用自适应 KL 系数：

```python
# 设定 KL 目标值 KL_target
if kl_actual > KL_target * 1.5:
    beta *= 1.5   # KL 太大，增大惩罚
elif kl_actual < KL_target / 1.5:
    beta /= 1.5   # KL 太小，减小惩罚（允许更多探索）
```

这样可以自动平衡探索和保守，不需要精确调 $\beta$。

**每个 token 位置实际拿到的奖励**

现在可以回到前面提到的稀疏奖励问题了。有了 KL 惩罚的概念，每个 token 位置实际拿到的奖励为：

$$r_t = \begin{cases} -\beta \cdot \text{KL}_t & t < T \text{ (中间步)} \\ r_\phi(x, y) - \beta \cdot \text{KL}_t & t = T \text{ (最后一步)} \end{cases}$$

其中 $\text{KL}_t = \log \frac{\pi_\theta(a_t|s_t)}{\pi_{ref}(a_t|s_t)}$ 是每个 token 位置的 KL 散度贡献。

也就是说：中间 token 只受 KL 惩罚（"别偏离太远"），最后一个 token 才加上 RM 的奖励分数。

#### PPO 的 Clipped Objective 详解

PPO 的核心创新是 Clipped Surrogate Objective：

$$L^{CLIP} = \mathbb{E}\left[\min\left(r_t(\theta) \cdot A_t, \; \text{clip}\left(r_t(\theta), 1-\epsilon, 1+\epsilon\right) \cdot A_t\right)\right]$$

其中概率比 (importance sampling ratio)：

$$r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$$

**逐步理解 clip 机制**

1. 当 $A_t > 0$（这个动作比平均好）：
   - 我们希望增大 $\pi_\theta(a_t|s_t)$，即 $r_t(\theta)$ 增大
   - 但 clip 将 $r_t$ 限制在 $[1-\epsilon, 1+\epsilon]$
   - 所以即使这个动作很好，概率也最多增大到原来的 $(1+\epsilon)$ 倍

2. 当 $A_t < 0$（这个动作比平均差）：
   - 我们希望减小 $\pi_\theta(a_t|s_t)$，即 $r_t(\theta)$ 减小
   - clip 将 $r_t$ 限制在 $[1-\epsilon, 1+\epsilon]$
   - 所以概率最多减小到原来的 $(1-\epsilon)$ 倍

3. $\min$ 操作确保取悲观估计——无论 advantage 正负，都选择对目标贡献更小的那个

**$\epsilon$ 的选择**

- 通常取 0.2（即允许策略概率在一步更新中变化 ±20%）
- 更小的 $\epsilon$ → 更保守的更新，更稳定但学习更慢
- 更大的 $\epsilon$ → 更激进的更新，学习更快但可能不稳定

#### GAE (Generalized Advantage Estimation) 详解

优势函数 $A_t$ 衡量的是"在状态 $s_t$ 下采取动作 $a_t$ 比平均好多少"。GAE 是估计 $A_t$ 的标准方法。

**TD 残差 (Temporal Difference Error)**

$$\delta_t = r_t + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$$

- $r_t$：即时奖励
- $\gamma$：折扣因子（在 RLHF 中通常设为 1）
- $V_\phi(s_t)$：Critic 对状态 $s_t$ 的价值估计

**GAE 公式**

$$\hat{A}_t^{GAE(\gamma, \lambda)} = \sum_{l=0}^{T-t} (\gamma \lambda)^l \delta_{t+l}$$

展开就是：

$$\hat{A}_t = \delta_t + (\gamma\lambda)\delta_{t+1} + (\gamma\lambda)^2\delta_{t+2} + \cdots$$

- $\lambda = 0$：退化为单步 TD，$\hat{A}_t = \delta_t$，偏差大但方差小
- $\lambda = 1$：退化为 Monte Carlo 估计，偏差小但方差大
- $\lambda \in (0, 1)$：平衡偏差与方差，通常取 0.95

**高效计算 GAE（反向递推）**

```python
def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """
    rewards: shape (batch, seq_len)  — 每个 token 位置的奖励
    values:  shape (batch, seq_len)  — Critic 预测的价值
    """
    advantages = torch.zeros_like(rewards)
    last_gae = 0
    for t in reversed(range(seq_len)):
        if t == seq_len - 1:
            next_value = 0  # 回合结束
        else:
            next_value = values[:, t + 1]
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        advantages[:, t] = last_gae = delta + gamma * lam * last_gae
    returns = advantages + values  # GAE + V = Return
    return advantages, returns
```

#### Critic (价值模型) 的训练

Critic 的目标是准确预测每个状态的价值（即从当前状态开始到回合结束的期望累积奖励）：

$$L_{critic} = \mathbb{E}\left[(V_\phi(s_t) - R_t)^2\right]$$

其中 $R_t = \hat{A}_t + V_{\phi_{old}}(s_t)$ 是 GAE 计算出的 return。

**Critic 也可以使用 clip**（类似 Actor 的 clip）：

$$L_{critic}^{CLIP} = \max\left[(V_\phi(s_t) - R_t)^2, \; (\text{clip}(V_\phi(s_t), V_{old} - \epsilon_v, V_{old} + \epsilon_v) - R_t)^2\right]$$

这里取 $\max$ 而非 $\min$，确保取悲观（更大的 loss），防止 value 更新过快。

#### PPO 完整训练循环

```python
# === PPO for RLHF 伪代码 ===

# 初始化
actor = load_sft_model()           # Actor π_θ
critic = load_rm_as_critic()       # Critic V_φ (通常用 RM 初始化)
ref_model = load_sft_model()       # Reference, 冻结
reward_model = load_rm()           # RM, 冻结
ref_model.eval()
reward_model.eval()

for epoch in range(num_epochs):
    # ======= Phase 1: 采样 (Rollout) =======
    prompts = sample_batch(dataset)

    with torch.no_grad():
        # Actor 生成回答
        responses = actor.generate(prompts,
                                    temperature=1.0,
                                    top_k=0, top_p=1.0,  # 不做截断
                                    max_length=512)

        # 记录旧策略的 log prob
        old_log_probs = actor.log_prob(prompts, responses)

        # Critic 估计价值
        values = critic(prompts, responses)

        # RM 打分
        rewards_rm = reward_model(prompts, responses)

        # Reference 模型的 log prob (用于计算 KL)
        ref_log_probs = ref_model.log_prob(prompts, responses)

    # ======= Phase 2: 计算奖励和 Advantage =======
    # 逐 token KL 惩罚
    kl_penalty = beta * (old_log_probs - ref_log_probs)  # per-token

    # 构造 token 级别的奖励
    token_rewards = -kl_penalty  # 中间步只有 KL 惩罚
    token_rewards[:, -1] += rewards_rm  # 最后一步加上 RM 分数

    # GAE
    advantages, returns = compute_gae(token_rewards, values)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)  # 归一化

    # ======= Phase 3: PPO 更新 (多次 mini-batch) =======
    for ppo_epoch in range(num_ppo_epochs):  # 通常 2~4
        for mini_batch in create_mini_batches(prompts, responses,
                                               old_log_probs, advantages, returns):
            # Actor loss (clipped surrogate)
            new_log_probs = actor.log_prob(mini_batch.prompts, mini_batch.responses)
            ratio = torch.exp(new_log_probs - mini_batch.old_log_probs)

            surr1 = ratio * mini_batch.advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * mini_batch.advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            # Critic loss
            new_values = critic(mini_batch.prompts, mini_batch.responses)
            critic_loss = F.mse_loss(new_values, mini_batch.returns)

            # 总 loss
            loss = actor_loss + vf_coef * critic_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
            torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

    # ======= Phase 4: 自适应 KL =======
    with torch.no_grad():
        kl_actual = (old_log_probs - ref_log_probs).sum(dim=-1).mean()
    if kl_actual > kl_target * 1.5:
        beta *= 1.5
    elif kl_actual < kl_target / 1.5:
        beta /= 1.5
```

#### PPO 关键超参数一览

| 超参数 | 典型值 | 说明 |
|--------|--------|------|
| `learning_rate` | 1e-6 ~ 5e-6 | Actor 和 Critic 的学习率，通常比 SFT 小一个量级 |
| `epsilon` (clip) | 0.2 | PPO clip 范围 |
| `gamma` | 1.0 | 折扣因子，RLHF 中通常不折扣 |
| `lambda` (GAE) | 0.95 | GAE 的 λ 参数 |
| `beta` (KL) | 0.01 ~ 0.2 | KL 惩罚系数初始值 |
| `kl_target` | 6.0 | 自适应 KL 的目标值 |
| `num_ppo_epochs` | 2 ~ 4 | 每批数据的 PPO 更新轮数 |
| `batch_size` | 64 ~ 512 | 每次采样的 prompt 数量 |
| `max_response_len` | 256 ~ 1024 | 最大生成长度 |
| `temperature` | 1.0 | 采样温度，训练时通常用 1.0 |
| `max_grad_norm` | 1.0 | 梯度裁剪阈值 |
| `vf_coef` | 0.1 ~ 1.0 | Critic loss 的权重系数 |

## 三、Reward Hacking 深入分析

Reward hacking 是 RLHF 中最棘手的问题之一。模型会利用 RM 的弱点来获取高分而不是真正提升质量。

### 常见的 Reward Hacking 模式

| 模式 | 表现 | 原因 |
|------|------|------|
| **长度偏好** | 生成越来越长的回答 | RM 训练数据中长回答通常更详细，被标注为更好 |
| **格式投机** | 滥用 bullet points、markdown 格式 | RM 偏好结构化的回答 |
| **谄媚 (Sycophancy)** | 无条件同意用户观点 | RM 偏好让用户"满意"的回答 |
| **重复高分短语** | 反复插入某些措辞 | RM 对某些表述有偏好 |
| **回避困难问题** | 不直接回答，转而给出笼统的"安全"回答 | 安全相关的 RM 对拒绝给高分 |

### 缓解手段

1. **KL 惩罚**：限制策略偏离参考模型的程度（基础手段）
2. **奖励归一化**：对 RM 输出做归一化，减少极端分数的影响
3. **长度惩罚**：在奖励中加入长度相关的惩罚项
4. **多个 RM**：使用多个不同的 RM，取平均或最小值
5. **定期更新 RM**：用新策略生成的数据重新训练 RM (iterative RLHF)
6. **Rejection Sampling + SFT**：先用 RM 筛选最好的回答，再用 SFT 训练（Llama 2 的做法）

## 四、RLHF + PPO 的优缺点

### 优点

- 理论上能超越标注者水平（通过探索）
- 经过充分验证，InstructGPT / ChatGPT / GPT-4 / Llama 2 均使用
- 能有效进行安全对齐
- 可以优化序列级别的指标，而不仅仅是 token 级别的 loss
- 探索能力：可以发现标注数据中没有的好回答

### 缺点

| 问题 | 说明 |
|------|------|
| **训练不稳定** | PPO 对超参数敏感，容易崩溃（loss 突然变大、生成质量骤降） |
| **显存巨大** | 同时需要 4 个模型（Actor + Critic + RM + Ref），7B 模型约需 80GB+ |
| **工程复杂** | 需要精心设计采样、归一化、裁剪、多机同步等细节 |
| **奖励模型质量** | RM 的错误会被 PPO 放大（garbage in, garbage out） |
| **Reward Hacking** | 即使有 KL 惩罚，仍可能出现各种 hacking 模式 |
| **采样效率低** | 每次更新都需要重新生成回答，生成是推理过程，速度慢 |
| **调参困难** | β、ε、学习率、GAE λ 等超参数之间相互影响 |

> 这些痛点催生了 DPO、KTO、GRPO 等替代方案。

### 与替代方案的对比

| 方法 | 是否需要 RM | 是否需要采样 | 模型数量 | 稳定性 |
|------|-----------|-------------|---------|--------|
| **RLHF + PPO** | ✅ | ✅ | 4 个 | 较差 |
| **DPO** | ❌ | ❌ | 2 个 (policy + ref) | 较好 |
| **KTO** | ❌ | ❌ | 2 个 | 较好 |
| **GRPO** | ✅ | ✅ | 3 个 (无 Critic) | 中等 |
| **REINFORCE** | ✅ | ✅ | 3 个 (无 Critic) | 方差大 |

## 五、工程实践要点

### 训练稳定性 Tricks

- **奖励归一化**：对 RM 输出做 whitening（均值为 0，方差为 1）
- **Advantage 归一化**：对 advantage 做标准化（减均值除标准差）
- **GAE (Generalized Advantage Estimation)**：使用 $\lambda$-return 估计优势函数，平衡偏差与方差
- **Mini-batch 更新**：每次采样后做多次梯度更新（通常 2~4 次）
- **梯度裁剪**：max_grad_norm = 1.0，防止梯度爆炸
- **学习率预热**：前 5~10% 的步数线性增大学习率
- **EMA (Exponential Moving Average)**：对 Actor 参数做 EMA，用 EMA 模型做最终评估

### 分布式训练策略

由于 4 个模型的显存需求，PPO 训练通常需要多 GPU / 多节点：

- **模型并行**：大模型用 tensor parallelism 或 pipeline parallelism 切分
- **Actor-Critic 分离部署**：Actor 和 Critic 放在不同的 GPU 组上
- **生成与训练分离**：生成阶段和训练阶段使用不同的并行策略（生成用更少的 GPU，训练用更多）
- **vLLM 加速生成**：用 vLLM 等推理引擎加速采样阶段

### 常用框架

| 框架 | 特点 |
|------|------|
| **TRL (Hugging Face)** | 易用，与 transformers 生态集成，适合快速实验 |
| **OpenRLHF** | 专为 RLHF 优化，支持 Ray + vLLM，适合大规模训练 |
| **DeepSpeed-Chat** | 基于 DeepSpeed，三阶段 pipeline，微软出品 |
| **NeMo-Aligner** | NVIDIA 出品，与 NeMo 框架集成 |
| **veRL (Volcano Engine RL)** | 字节跳动开源，基于 Megatron，支持大规模 PPO |

### 监控指标

训练过程中需要关注的关键指标：

```
1. reward/mean            — 平均奖励，应该稳步上升
2. reward/std             — 奖励方差，不应过大
3. kl/mean                — 平均 KL 散度，应在目标值附近
4. policy/entropy         — 策略熵，不应过低（过低意味着坍缩）
5. policy/approx_kl       — 新旧策略的 KL，应该较小
6. policy/clip_fraction   — 被 clip 的比例，通常 0.1~0.3
7. critic/loss            — Critic loss，应该逐步下降
8. response/length_mean   — 平均回答长度，暴增可能是 reward hacking
```

---

**相关文档**：
- [预训练与后训练](预训练与后训练.md)
- [SFT详解](SFT详解.md)
- [DPO详解](DPO详解.md)
- [GRPO详解](GRPO详解.md)
- [KL散度](../../机器学习基础/KL散度.md)

[返回上级](README.md) | [返回总目录](../../README.md)

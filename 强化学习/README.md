# 强化学习 (Reinforcement Learning)

智能体通过与环境交互，从奖励信号中学习最优行为策略。

## 知识框架

```
强化学习
├── 基础概念 ─── MDP / 值函数 / 策略 / Bellman方程 / 探索与利用
├── 值函数方法 ─── MC / TD / Q-Learning / SARSA / DQN 系列
├── 策略梯度方法 ─── REINFORCE / Actor-Critic / A2C/A3C / TRPO / PPO
├── 基于模型的方法 ─── Dyna / MBPO / World Models / MuZero
├── 离线强化学习 ─── Offline RL / CQL / Decision Transformer
├── 多智能体强化学习 ─── MARL / 合作与竞争 / 通信机制
└── RL × LLM ─── RLHF / PPO for LLM / GRPO / 奖励模型
```

## 子专题

| 专题 | 说明 |
|------|------|
| [基础概念与框架](基础概念与框架.md) | MDP、值函数、策略、Bellman 方程、探索与利用 |
| [值函数方法](值函数方法.md) | Monte Carlo、TD Learning、Q-Learning、SARSA、DQN 系列 |
| [策略梯度方法](策略梯度方法.md) | REINFORCE、Actor-Critic、A2C/A3C、TRPO、PPO |
| [基于模型的方法](基于模型的方法.md) | Dyna、MBPO、World Models、MuZero |
| [离线强化学习](离线强化学习.md) | Offline RL、CQL、IQL、Decision Transformer |
| [多智能体强化学习](多智能体强化学习.md) | CTDE、MAPPO、QMIX、通信机制 |

## 与其他专题的联系

- **RL × LLM 对齐**：RLHF/PPO/GRPO 在大模型训练中的应用 → [大模型/训练与微调](../大模型/训练与微调/RLHF与PPO详解.md)
- **策略梯度 → PPO**：PPO 是当前 LLM 对齐最常用的 RL 算法
- **概率统计基础**：贝叶斯推断、期望计算 → [机器学习基础/概率统计](../机器学习基础/概率统计/overview.md)
- **马尔可夫链**：MDP 的数学基础 → [线性代数/马尔可夫矩阵](../机器学习基础/线性代数/markov_matrix.md)

---
[返回总目录](../README.md)

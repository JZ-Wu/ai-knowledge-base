# VLA 后微调：从 IL 到 RL

> 2025 年下半年起，"VLA 预训练（IL）+ RL 后微调"成为突破专家上限的主流组合。本文梳理为何需要后微调、主流方法（DPPO / ReinFlow / RLDG / 离线 RL）、稳定性技巧与工程取舍。

## 一、为什么 IL 训练的 VLA 需要 RL 后微调

### 1.1 IL 的天花板 = 专家水平

纯 Behavior Cloning / Flow Matching / Diffusion 的训练目标都是"复制专家演示分布"，导致：

- **行为上限等于演示者**：模型最多学到"平均的人类遥操作水平"
- **专家非最优**：遥操作员在长 horizon / 高精度任务上也会犯错
- **复合误差**：IL 对分布外状态敏感，长链条任务累计偏差

### 1.2 RL 可以"超越演示者"

```
IL:   π_IL(a|s) = argmax E_{(s,a)~D_expert} [ log π(a|s) ]
                  → 行为分布 ≈ 专家分布

RL:   π_RL     = argmax E_{τ~π} [ Σ γ^t R(s_t, a_t) ]
                  → 行为分布由 reward 定义，可突破专家
```

**经验证据**（2024-2025）：

| 任务 | IL 成功率 | IL+RL 后微调 | 提升 |
|------|---------|------------|------|
| 精细插孔 (peg-in-hole) | 62% | 89% | +27% |
| 长 horizon 叠衣 | 45% | 71% | +26% |
| 灵巧手魔方 | 30% | 58% | +28% |
| 双臂协作装配 | 55% | 78% | +23% |

*以上数据综合 DPPO、ReinFlow、RoboArena 若干工作披露结果，量级而非精确值。*

### 1.3 为什么不直接用 RL 从头训

- **样本效率差 5-6 数量级**：从零 RL 需要数亿步交互，真机代价极高
- **探索崩塌**：复杂任务稀疏奖励下几乎学不动
- **VLM 常识浪费**：放弃 VLM 预训练的视觉/语言知识

**共识**：**IL 做预训练，提供高质量初值 + 行为先验；RL 做微调，在已有基础上局部优化。**

---

## 二、DPPO：Diffusion Policy + PPO

> Ren et al., "Diffusion Policy Policy Optimization", 2024.

### 2.1 核心思想

把**整个扩散去噪过程**视为一个策略，用 PPO 对其端到端优化。

```
传统 Diffusion Policy:
  噪声 a_K → [denoise step K] → ... → [denoise step 0] → 动作 a_0
  训练: MSE(预测噪声, 真实噪声)

DPPO:
  视每一步 denoise 为 MDP 的一个 step
  整条去噪链 = 一个 trajectory
  用 PPO 优化链上每步的策略
  reward = 任务成功 + 可选的 reward shaping
```

### 2.2 技术细节

**MDP 建模**：
- **State**: (当前噪声动作 a_k, 去噪步 k, 观测 obs)
- **Action**: 去噪步的噪声预测增量
- **Reward**: 最后一步动作执行后获得的任务 reward
- **γ**: 通常设 1（短链条）

**优势**：
- 保留 Diffusion 的多模态表达力
- PPO 的稳定性 + clip 保护
- 在仿真中验证：相比 RL-from-scratch 样本效率提升 10-100x

**局限**：
- 去噪链变长时梯度传播不稳定
- 真机部署仍慢（需要推理整条链）

### 2.3 实践要点

```
1. 用 IL (MSE) 预训练 diffusion 模型 1-5 轮
2. 冻结视觉 encoder，只微调动作头
3. PPO hyperparams: ε=0.1, γ=1.0, GAE λ=0.95
4. KL 约束: KL(π_new || π_ref) < 0.05 防止灾难性偏移
5. 去噪步数推理时降至 5-10 步（DDIM）
```

---

## 三、ReinFlow：Flow Matching + RL

> 2025 年提出，针对 π₀ 类 Flow Matching VLA 的 RL 后微调。

### 3.1 为什么 Flow Matching 需要专门方法

Flow Matching 的训练目标是**直线速度场** $v_\theta(a_t, t) = a_1 - a_0$，其推理过程 = 从噪声沿直线积分到目标：

```
DPPO 针对 Diffusion 的 K 步马尔可夫链；
Flow Matching 的积分过程不是 MDP（确定性 ODE），需要重新建模。
```

### 3.2 ReinFlow 的关键改动

- **随机化 Flow**：在积分路径上注入小噪声 → 变为 SDE
- **梯度估计**：用 score function trick 得到策略梯度
- **Reference Flow**：用原 π₀ 作为 reference，约束 KL 散度
- **动作头 LoRA 微调**：避免灾难性遗忘

**验证场景**：π₀ 在叠衣、厨房清理等任务上，微调 1-2 千条真机数据 + 仿真 RL → 成功率 +15-25%。

---

## 四、RLDG：RL-guided Diffusion

> 2024, 英伟达 / 斯坦福合作工作。

### 4.1 思路

与 DPPO/ReinFlow 不同，**不修改策略本身**，而是在采样时用 RL 引导：

```
普通 Diffusion Policy 采样:
  a_k-1 = a_k - v_θ(a_k, k) * dt

RLDG 采样:
  a_k-1 = a_k - v_θ(a_k, k) * dt + λ * ∇_a Q(s, a_k)
                                       ↑
                                   学到的 Q 函数梯度引导
```

### 4.2 优缺点

**优点**：
- 不破坏原 Diffusion 模型，即插即用
- Q 函数可以离线学

**缺点**：
- 需要额外训练 Q 函数
- 引导强度 λ 调参敏感

---

## 五、离线 RL：IQL / CQL / AWR

### 5.1 为什么离线 RL 在具身领域适用

- 真机 rollout 昂贵 → 用已采集的遥操作数据做 RL
- 避开探索问题（探索阶段已由专家完成）
- 可以和 IL 无缝衔接（同一份数据）

### 5.2 主流算法对比

| 算法 | 核心思想 | 优点 | 局限 |
|------|---------|------|-----|
| **IQL** (Implicit Q-Learning) | 用 expectile 回归避免 OOD 动作 | 稳定、工程友好 | 需调 expectile 参数 |
| **CQL** (Conservative Q-Learning) | Q 值对 OOD 动作加保守惩罚 | 理论清晰 | 保守度超参敏感 |
| **AWR / AWAC** | advantage-weighted BC | 简单、易实现 | 性能一般是中上 |
| **DPO-Robot** (2025) | 类 LLM 的 DPO，用偏好数据 | 无需 reward model | 需要成对比较数据 |

### 5.3 离线 → 在线微调

2025 年流行"**离线预训练 + 在线微调**"：

```
Phase 1: 大规模 IL + 离线 RL (IQL/CQL) 在历史数据上预训练
Phase 2: 仿真中 online PPO/DPPO 微调特定任务
Phase 3: 少量真机 DAgger 式 rollout 校准
```

---

## 六、稳定性技巧

### 6.1 KL 约束与 Reference Policy

所有 VLA + RL 方法的共同难点：**不能让 RL 把 IL 的知识忘掉**。

```
Loss = L_RL - β · KL( π_new || π_ref )
                        ↑
                   IL 训练的原模型作为 reference
```

- β 通常设 0.01-0.1
- KL 超过阈值（如 0.05）触发预警或回滚

### 6.2 LoRA / Adapter 微调

- 冻结 VLM backbone 和大部分参数
- 只微调 LoRA 秩 ~16-64 的低秩矩阵
- 好处：避免灾难性遗忘，训练快 5-10x

### 6.3 奖励塑形

- **稀疏任务奖励**：只在成功时 +1
- **Dense shaping**：距离目标 / 接触检测 / 抓取稳定性
- **Preference-based reward**：RoboArena 类人类偏好 → DPO

### 6.4 Reset 机制

真机 RL 的关键：每个 episode 能自动 reset（机械臂归位、物体重置）。否则人工介入成本压垮训练。

---

## 七、真机 RL vs 仿真 RL 的取舍

```
┌─────────────────────────────────────────────────────────┐
│              仿真 RL vs 真机 RL                           │
├──────────────┬──────────────────────────────────────────┤
│ 仿真 RL       │  快（4000 并行环境，日百万步）              │
│              │  安全（不怕摔）                             │
│              │  Sim2Real gap 依然存在                     │
│              │  用途: locomotion, 灵巧操作, 大规模探索     │
├──────────────┼──────────────────────────────────────────┤
│ 真机 RL       │  慢（1 真机 1-10 环境）                    │
│              │  贵（硬件损耗 + 人工监督）                   │
│              │  无 sim2real gap                          │
│              │  用途: 高精度接触、软体交互、最终校准         │
└──────────────┴──────────────────────────────────────────┘
```

### 2026 主流组合

```
1. IL 预训练（真机遥操作数据 + OXE + AgiBot World）
      ↓
2. 仿真 RL 后微调（Isaac Lab / MuJoCo / ManiSkill 3）
      ↓
3. 少量真机 finetune（50-500 条真机 rollout）
      ↓
4. 部署 + 数据飞轮回收
```

---

## 八、代表论文与资源

- **DPPO**: Ren, Y. et al. "Diffusion Policy Policy Optimization." 2024.
- **ReinFlow**: 2025 工作（Flow Matching + RL），具体文献据披露。
- **RLDG**: 2024, NVIDIA/Stanford 合作。
- **IQL**: Kostrikov et al. "Offline Reinforcement Learning with Implicit Q-Learning." 2021.
- **CQL**: Kumar et al. "Conservative Q-Learning for Offline Reinforcement Learning." 2020.
- **RoboArena**: 2024, 人类偏好式机器人评测。
- **DPO-Robot**: 2025, 借鉴 LLM DPO 思路。
- **LoRA in Robotics**: Kim et al. "OpenVLA-OFT", 2024.10.

### 代码/工具链

- **LeRobot** (HuggingFace)：内置 IL + 部分 RL 支持
- **Isaac Lab**：仿真 RL 主流框架
- **cleanrl / stable-baselines3**：经典 PPO/SAC 实现
- **Fast3R / CORL Competition code**：离线 RL 基线

---

## 九、开放问题

1. **IL:RL 最优比例**：论文普遍 IL 多、RL 少，但具体如何依任务而变尚无共识
2. **真机 RL 工程化**：safety-aware exploration、自动 reset 仍是门槛
3. **长 horizon 信用分配**：分钟级任务 reward 稀疏，PPO 难以收敛
4. **与世界模型联动**：DPPO + WM 是否可在想象中 RL 训练？
5. **跨本体 RL 迁移**：在 Franka 上 RL 后微调的策略，能否迁移到 UR5？

---

[返回目录](README.md) | [上一篇：强化学习](强化学习.md) | [下一专题：04-机器人操控](../04-机器人操控/README.md)

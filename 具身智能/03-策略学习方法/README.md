# 03-策略学习方法

> 本专题汇总具身智能中的核心策略学习范式：模仿学习（Behavior Cloning、DAgger、ACT）、扩散策略（Diffusion Policy / DP3）、强化学习（PPO/SAC + Teacher-Student）、以及 2025-2026 的热门方向 VLA + RL 后微调（DPPO / ReinFlow / RLDG）。

## 本专题文件索引

| # | 文件 | 内容 |
|---|------|------|
| 01 | [模仿学习](模仿学习.md) | BC、DAgger、复合误差、ACT（CVAE + Action Chunking + Temporal Ensemble）、Mobile ALOHA、人类视频模仿（H2O, R3M, VIP, MVP, UniPi, Vid2Robot） |
| 02 | [扩散策略](扩散策略.md) | Diffusion Policy (Chi et al. 2023)、多模态动作分布、U-Net vs Transformer 去噪、DP3 点云版、DDIM 加速、训练与归一化实践 |
| 03 | [强化学习](强化学习.md) | PPO/SAC/TD3、奖励工程、课程学习、Teacher-Student、ANYmal Parkour、OpenAI Rubik's Cube、Extreme Parkour |
| 04 | [VLA 后微调](VLA后微调.md) | 为何 IL 的 VLA 需要 RL 后微调、DPPO、ReinFlow、RLDG、IQL/CQL 离线 RL、KL 约束与 reference policy、真机 vs 仿真 RL 取舍 |

---

## 2026 策略学习发展现状

### 时间线：从单一范式到混合范式

```
┌──────────────────────────────────────────────────────────────────┐
│  2015-2019: 单任务 RL                                             │
│    DDPG / SAC / PPO, 每任务从头训练, 10-100M env steps            │
│                              ↓                                     │
│  2019-2022: 仿真 RL + Sim-to-Real                                  │
│    域随机化、Teacher-Student、OpenAI Rubik's, ANYmal Parkour       │
│                              ↓                                     │
│  2022-2023: IL 大规模化                                            │
│    ALOHA (ACT, 2023)、Diffusion Policy (Chi et al. 2023)          │
│    证明 IL + 好数据 = 复杂任务可达                                  │
│                              ↓                                     │
│  2023-2024: Foundation Model 时代                                  │
│    RT-1/2, OpenVLA, Octo, π₀ → VLA 吸收 IL                        │
│                              ↓                                     │
│  2024-2025: Flow Matching 成默认动作头                             │
│    π₀ Flow Matching 范式推广到 GR00T、Helix、SmolVLA               │
│                              ↓                                     │
│  2025-2026: VLA + RL 后微调成标配                                   │
│    DPPO (2024), ReinFlow (2025), RLDG (2024)                      │
│    "IL 预训练 + RL 后微调" 成为突破专家上限的主流组合                │
└──────────────────────────────────────────────────────────────────┘
```

### 2026 当前共识（五条）

1. **Diffusion / Flow Matching 成为默认动作头**：相比 MLP 回归、相比 MSE/BC，扩散与 Flow Matching 在多模态动作分布上的表达力已被业界反复验证。π₀ 系列、GR00T N1、Helix、SmolVLA 均采用 Diffusion/FM 路线。

2. **VLA + RL 后微调成为标配**：纯 IL 训练的 VLA 行为上限 = 专家；DPPO / ReinFlow 等 RL 后微调方法可以**突破**专家水平，在复杂精细任务上提升 10-30%。这是 2025 年下半年到 2026 年初最明确的方法论收敛。

3. **Mobile ALOHA 证明 IL 可达复杂任务**：2024 年 Mobile ALOHA 展示仅用 50-100 条演示即可完成炒菜、开柜子等复杂家务，推翻了"复杂任务必须用 RL"的旧共识，推动 IL 优先路线。

4. **Teacher-Student 是 locomotion 的事实标准**：从 ANYmal Parkour 到人形 Extreme Parkour，所有成功的四足/人形运动都采用"仿真里 privileged info RL → 蒸馏到 noisy obs"的两阶段流水线。

5. **Action Chunking + Temporal Ensemble 成为工程标配**：一次预测 1-2 秒动作（50 步）+ 相邻预测时间加权，在 ACT、Diffusion Policy、π₀ 中均被采用，实质性降低复合误差。

### 2026 开放问题

- **IL 与 RL 的最优配比**：预训练用多少数据？RL 微调用多长时间？
- **真机 RL 的工程化**：DPPO 在仿真验证充分，真机 RL 仍受硬件损耗、安全、数据效率三重限制。
- **长 horizon 信用分配**：分钟级任务的 RL 奖励稀疏问题。
- **人类视频模仿的 embodiment gap**：H2O、DexCap 等方向距离真正实用仍有差距。
- **与世界模型的融合**：Dreamer 式 model-based 策略学习是否会在 VLA 时代复兴？

---

## 相关专题

- **VLA 模型**：[../01-VLA模型/](../01-VLA模型/) — VLA 是当前主流策略架构，Diffusion/Flow Matching 作为其动作头
- **世界模型**：[../02-世界模型/](../02-世界模型/) — WM + 策略的融合路线
- **机器人操控**：[../04-机器人操控/](../04-机器人操控/) — 操控任务是策略学习的主战场
- **运动与导航**：[../05-运动与导航/](../05-运动与导航/) — Locomotion 是 RL 的成熟应用
- **仿真与 Sim2Real**：[../06-仿真与Sim2Real/](../06-仿真与Sim2Real/) — 仿真训练 + 域随机化是 RL 部署关键
- **数据与遥操作**：[../07-数据与遥操作/](../07-数据与遥操作/) — IL 的数据采集

---

[返回目录](../README.md) | [上一专题：02-世界模型](../02-世界模型/README.md) | [下一专题：04-机器人操控](../04-机器人操控/README.md)

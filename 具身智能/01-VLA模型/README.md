# 01-VLA 模型

> 本专题系统讲解 Vision-Language-Action（VLA）基础模型：从 RT-1 到 π₀.7 的架构演进、2026 主流模型对比、四种动作生成方法、双系统快慢架构，以及轻量化与端侧部署。

## 本专题文件索引

| # | 文件 | 内容 |
|---|------|------|
| 01 | [VLA 架构演进](VLA架构演进.md) | RT-1 → RT-2 → Octo → OpenVLA → π₀ → 双系统 VLA 的演进脉络，动作表示从离散 token 到双频的变迁 |
| 02 | [主流 VLA 模型 2026](主流VLA模型2026.md) | PI π 系列、NVIDIA GR00T、Figure Helix、Gemini Robotics、国内 VLA、开源轻量 VLA 的完整对比 |
| 03 | [动作生成方法](动作生成方法.md) | 离散 token / 扩散 / Flow Matching / 自回归 Transformer 四范式详解，含数学推导 |
| 04 | [双系统架构](双系统架构.md) | S2 慢思考 + S1 快执行的设计原理、Helix 详解、chunk prediction、冻结策略 |
| 05 | [轻量化与端侧部署](轻量化与端侧部署.md) | SmolVLA / π-fast / OpenVLA-OFT 并行解码、蒸馏/量化/MoE、Jetson Thor 等端侧芯片 |

---

## 2026 年 VLA 发展现状

### 三大共识

经过 2023-2025 年的密集探索，VLA 研究和产业在 2026 年形成三条主干共识：

1. **双系统架构（System 2 + System 1）成行业默认**
   - System 2：7-9B 的 VLM 主干，7-9 Hz，负责场景理解、任务推理
   - System 1：80-300M 的 Action Expert，100-200 Hz，负责连续动作生成
   - 代表：Figure Helix、GR00T N1、π₀ 系列、Gemini Robotics

2. **Flow Matching 成主流动作头**
   - 相比 Diffusion（DDPM 50-100 步）推理快一个数量级，5-10 步 Euler 积分即可
   - 相比离散 token（RT-2 式）精度高，天然多模态分布
   - 代表：π₀ / π₀.5 / π₀.7、SmolVLA、Helix v2（部分）

3. **端侧部署开始规模落地**
   - Jetson Thor（2000 TOPS FP8）让 7B VLA 可以单卡跑 30 Hz
   - OpenVLA-OFT 并行解码把 26 Hz 拉到 109 Hz
   - SmolVLA（~500M）可以在 Mac Mini / Jetson Orin 上部署
   - π-fast、GR00T 蒸馏版、Helix 等商业部署已过 100 Hz 闭环

### 时间线（2022-2026）

```
2022.12 ─┬─ RT-1 (13万轨迹 Transformer Policy，VLA 前身)
          │
2023.07 ─┼─ RT-2 (第一个 VLA，动作即文本 token)
2023.10 ─┤
          │
2024.05 ─┼─ Octo (开源跨本体，Diffusion 动作头)
2024.06 ─┼─ OpenVLA (7B 开源，Llama 2 + 离散 token)
2024.10 ─┼─ π₀ (Flow Matching，10K 小时真机数据)
2024.10 ─┼─ OpenVLA-OFT (并行解码 109Hz)
2024.11 ─┼─ CogACT / RDT-1B
          │
2025.01 ─┼─ Cosmos (视频世界模型开源)
2025.02 ─┼─ Figure Helix (7B S2 + 80M S1，200Hz)
2025.03 ─┼─ GR00T N1 (首个 Apache 2.0 开源，人形)
2025.03 ─┼─ Gemini Robotics + ER (DeepMind)
2025.04 ─┼─ π₀.5 (家庭开放世界泛化)
2025.05 ─┼─ SmolVLA (HF LeRobot, ~500M)
2025 H2 ─┼─ π-fast / Helix v2 / GR00T N1.5
          │
2026 H1 ─┼─ π₀.6 / π₀.7 (据披露: 多模态推理 + 长 horizon + 家庭泛化)
2026   ─┴─ Cosmos 2.0 / RDT-2 开源 / Jetson Thor 量产
```

### 核心趋势

| 趋势 | 2023 状态 | 2026 状态 |
|------|-----------|-----------|
| 动作表示 | 离散 token 为主 | **Flow Matching 主流**，扩散次之 |
| 控制频率 | 1-10 Hz | **100-200 Hz**（S1 动作专家） |
| 架构 | 单塔端到端 | **双系统双频** |
| 数据来源 | 纯真机遥操作 | **真机 + 跨本体 + 仿真 + 世界模型增广** |
| 本体 | 单臂桌面为主 | **双臂、人形全身、移动操控** |
| 部署位置 | 云端或工作站 | **端侧（Jetson Thor / 地平线 J6）** |
| 开源程度 | 极少 | **GR00T、Octo、OpenVLA、SmolVLA、RDT、GO-1 等多点开花** |

### 开放问题（2026）

1. **Scaling Law 是否成立**：数据量 × 2 性能提升几何？目前无定论
2. **触觉 / 力反馈的融合**：当前视觉主导，触觉进 VLA 的 token 化仍早期
3. **长 horizon 与记忆机制**：分钟级以上任务需要显式记忆，context 窗口不足
4. **RL 微调价值**：在真机数据稀缺场景下，VLA + RL（DPPO、ReinFlow、Q-Transformer）能否突破 IL 上限
5. **安全对齐**：类似 RLHF，如何对齐机器人行为边界
6. **灾难性遗忘**：持续学习新本体/新任务时保持旧能力

---

## 相关专题

- **基础架构**：[../00-基础与架构/](../00-基础与架构/) — 三层架构、发展脉络
- **世界模型**：[../02-世界模型/](../02-世界模型/) — Cosmos / Genie / V-JEPA 2 与 VLA 的耦合
- **策略学习方法**：[../03-策略学习方法/](../03-策略学习方法/) — Diffusion Policy、RL、IL 的数学与工程
- **数据与遥操作**：[../07-数据与遥操作/](../07-数据与遥操作/) — Open X-Embodiment、AgiBot World、DROID
- **评测基准**：[../09-评测基准/](../09-评测基准/) — SIMPLER、LIBERO、CALVIN、RoboArena

---

[返回目录](../README.md) | [上一篇：../00-基础与架构/发展脉络与关键里程碑](../00-基础与架构/发展脉络与关键里程碑.md) | [下一篇：VLA 架构演进](VLA架构演进.md)

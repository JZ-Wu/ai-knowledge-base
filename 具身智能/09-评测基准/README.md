# 09 · 评测基准

> 具身智能的"尺子"：如何公平、可重复地衡量一个 VLA / 机器人系统的能力。本目录梳理 2023-2026 年主流评测基准，评测指标的演进，以及仍然开放的问题。

## 2026 评测发展现状

### 三个核心趋势

1. **从仿真成功率转向真机多场景评测**
   - 2023 年主流：CALVIN / LIBERO / Meta-World / RoboSuite 都是仿真基准，便于复现但与真实差距大。
   - 2024 年起：**SIMPLER**（仿真 + 真实性校准）承认仿真到真实的偏差；**RoboArena** 直接做跨本体真机评测。
   - 2025-2026 年：**AgiBot World Eval**、**GR-1/H1 Benchmark**、**Gemini Robotics-ER** 相关评测均以真机多场景为主。

2. **跨本体（Cross-Embodiment）评测兴起**
   - 早期评测都是"一个模型 + 一个本体 + 固定任务"；2024 年起 π₀、GR00T、Gemini Robotics 等号称跨本体通用。
   - **RoboArena**（2024）首次系统比较不同模型在多种机器人（Franka / UR5 / ALOHA / GR-1 / H1）上的表现。
   - 评测难点：不同本体的任务可行集不同，公平对比需谨慎设计对齐轴。

3. **人类评分（RLHF 式偏好）成为新维度**
   - 传统指标：任务成功率（binary）、平均完成时间。
   - 新指标：人类观察员/众包评分（流畅度、安全性、自然度），类似 LLM 的 RLHF 偏好评估。
   - **PARTNR**（Meta）、**RoboArena** 已采用；2026 年将有更多基准加入。

### 评测指标演进

```
2022-2023: 任务成功率（固定任务 × 固定场景）
    ↓
2023-2024: 多任务平均（LIBERO、CALVIN、Meta-World 50 任务）
    ↓
2024-2025: 跨本体泛化（RoboArena、Open X-Embodiment）
    ↓
2025-2026: 真人满意度 + 长尾鲁棒性（PARTNR、AgiBot World Eval）
    ↓
2026+ :    实况持续评测（production-like, 永久部署数据回流）
```

### 方法论的基本共识

- 仿真评测：快速、可复现，但存在 Sim2Real 差距，不可独立作为产品决策依据。
- 真机评测：贵、慢、难复现，但最接近产品价值。
- **组合评测 = 仿真筛选 + 真机验证 + 人类评分**是 2026 年头部团队的标配。
- 单一基准不足以评估一个"通用机器人"；需多基准组合 + 长尾测试集 + 持续更新。

---

## 本专题索引

- [主流基准对比](主流基准对比.md) — 2023-2024 经典基准（SIMPLER/LIBERO/CALVIN/Meta-World/RoboSuite）与 2025-2026 新基准（PARTNR/RoboArena/RoboSpatial-VLM/AgiBot World Eval）对比、指标演进、开放问题

## 关联内容

- [01-VLA模型](../01-VLA模型/README.md) — VLA 模型的评测大部分在 SIMPLER、LIBERO、CALVIN 上报告
- [04-机器人操控](../04-机器人操控/README.md) — 操控任务的典型评价指标
- [05-运动与导航](../05-运动与导航/README.md) — Locomotion 评测基准（Isaac Lab、Meta Habitat）
- [06-仿真与Sim2Real](../06-仿真与Sim2Real/README.md) — 仿真评测的基础设施

---

[返回具身智能总目录](../README.md)

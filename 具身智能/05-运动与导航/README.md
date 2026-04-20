# 05 运动与导航 (Locomotion & Navigation)

本目录聚焦机器人的"下半身能力"：足式运动控制、全身 Loco-Manipulation、SLAM 与视觉语言导航。

## 2026 运动与导航发展现状

### 时间线

| 年份 | 里程碑 |
|------|--------|
| 2000 | Honda ASIMO 发布，经典 ZMP 双足行走 |
| 2016 | Cassie 发布，欠驱动双足的里程碑 |
| 2019 | Agility Digit 发布，仓储场景双足机器人 |
| 2021 | ANYmal Parkour，四足 RL + Teacher-Student sim2real |
| 2022 | Extreme Parkour / Barkour，四足翻越复杂地形 |
| 2023 | Unitree H1 后空翻，人形 RL 敏捷运动爆发 |
| 2024 | Isaac Lab 发布，GPU 大规模人形训练标准化 |
| 2025 | 电动 Atlas、Figure 02/03、Optimus V2/V3，人形量产潮 |
| 2026 | 全电驱人形成为共识，家庭+工厂双场景并行，VLN 仍开放 |

### 当前共识（2026-04）

1. **四足 locomotion 已成熟**：RL + 大规模并行仿真 + 域随机化是默认配方，ANYmal / Spot / Unitree 产品化成熟。
2. **人形双足从研究走向量产**：电动化完成（Atlas 电动版、Figure、Optimus），敏捷行走、爬楼梯、搬运在工厂 demo 成功。
3. **Teacher-Student 成 sim2real 标准**：先训练特权信息 teacher，再 distill 到部分观测 student，显著提升现实泛化。
4. **全身 VLA 可行但仍困难**：Helix 证明单网络可行，但需要海量全身遥操作数据。
5. **VLN 仍是开放问题**：LLM-planner + VLM 导航组合成主流，但长走廊、未知环境泛化仍不稳。

### 开放问题

- **不平整地形双足**：松软地面、斜坡、楼梯的稳定性仍是 Demo 级。
- **Loco-Manipulation 数据**：拿着东西走路的真实数据极少，RoboCasa 等仿真方案未完全解决。
- **长走廊 / 长距离 VLN**：几十米路径的语义对齐仍容易飘移。
- **人形跌倒恢复**：大部分量产机器人摔倒即需人工扶起。
- **多机器人导航**：仓储多机调度是开放工业问题。

## 索引

- [四足与人形 locomotion](四足与人形locomotion.md)：硬件、RL 训练、Teacher-Student、2026 进展
- [全身控制 WBC](全身控制WBC.md)：QP-WBC、MPC、Loco-Manipulation、数据集
- [视觉语言导航 VLN](视觉语言导航VLN.md)：SLAM 基础、PointNav/ObjectNav/VLN、LLM planner

## 相关目录

- [../01-VLA模型/](../01-VLA模型/)：VLA 对 Loco-Manipulation 的影响
- [../04-机器人操控/](../04-机器人操控/)：全身操控与双臂协作
- [../06-仿真与Sim2Real/](../06-仿真与Sim2Real/)：Isaac Lab / Gym
- [../08-硬件与本体/](../08-硬件与本体/)：四足与人形硬件本体

---
[返回具身智能目录](../README.md)

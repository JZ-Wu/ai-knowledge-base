# 06 · 仿真与 Sim2Real

> 本专题梳理具身智能仿真器生态、Sim-to-Real 迁移方法、可微分物理与 GPU 加速，覆盖从底层物理引擎到上层训练框架的完整链路。

## 为什么仿真仍然是具身智能的核心基建

真实机器人采集数据的三大瓶颈无解：
- **效率**：真实世界 1 秒 = 1 秒，GPU 仿真可达 1000x 实时
- **成本**：机器人磨损、场地搭建、遥操作人力
- **安全**：探索阶段策略可能造成碰撞、损坏

仿真的定位：**用算力换数据**，大规模并行训练 + Sim-to-Real 迁移，再辅以真机微调。

---

## 2026 仿真发展现状

### 时间线

```
2017-2019  MuJoCo / PyBullet 统治学术界
           OpenAI Rubik's Cube (2019) 证明域随机化可以零样本迁移

2020-2022  Isaac Gym (2021) 首次实现单 GPU 数千环境并行
           MuJoCo 开源 (2021, DeepMind 收购)，加速学术采纳
           Habitat 2.0 (2021) 支持导航 + 操控

2023       Isaac Sim / Isaac Lab (原 Orbit) 成为 locomotion 事实标准
           SAPIEN + ManiSkill 2 成为操控 benchmark 主流
           MJX (JAX-MuJoCo) 发布，开启大规模可微分 MuJoCo

2024       Genesis 开源，多物理引擎 + 10-80x 加速声称
           ManiSkill 3 发布，Warp 后端使视觉 RL 大规模化
           RoboCasa、SIMPLER 等 benchmark 陆续涌现
           真实场景重建（3DGS）开始融入仿真管线

2025       世界模型（Cosmos、GR00T Dreams）作为仿真补充登场
           可微分仿真首次进入工业产品线
           Real-to-Sim-to-Real 成为家庭场景标准流程

2026 (Q1-Q2)
           GPU 并行成为默认预期（千级并行是起点，万级是前沿）
           "物理仿真 + 世界模型 + 真实重建"三支柱并立
           可微分仿真进入商用化（MJX + Brax，Genesis 可微模式）
```

### 当前共识

1. **GPU 并行已是基础设施**：单卡 1000-4000 环境并行是 locomotion 的起步配置，Isaac Lab / Genesis / ManiSkill 3 / MJX 均支持
2. **物理仿真 + 世界模型互补**：物理引擎解决动力学正确性，世界模型解决视觉分布与长尾场景
3. **域随机化 + Teacher-Student 是 locomotion Sim-to-Real 标配**：ANYmal、Unitree、Booster 等都沿用此范式
4. **操控 Sim-to-Real 仍难**：纯仿真训练的 VLA 真机性能通常比"仿真+真机混合"差 20-40%
5. **可微分仿真还未成为主流**：在 locomotion 课题上 PPO + 大规模并行仍是最稳方案，梯度基 RL 在特定任务（软体、多体组装）上有优势

### 开放问题

- **接触仿真精度**：刚性接触、柔体、流体、颗粒物仍是 Sim-to-Real 断层重灾区
- **长 horizon 任务**：仿真中 30 秒任务成功，真机 5 秒就偏离——误差累积无法忽视
- **感知 Sim-to-Real**：渲染器再逼真也难以跨越真实相机的几何失真、鱼眼、运动模糊
- **世界模型与物理仿真的融合架构**：两者如何协同训练仍无定论
- **可微分仿真的非光滑性**：接触不连续导致梯度噪声大，稳定优化仍是研究热点

---

## 本专题索引

| 文件 | 内容 |
|------|------|
| [主流仿真器对比](主流仿真器对比.md) | Isaac Sim/Lab、MuJoCo/MJX、Habitat、SAPIEN/ManiSkill、Genesis、PyBullet 等；选型建议；action→力矩→物理引擎链路 |
| [Sim-to-Real方法](Sim-to-Real方法.md) | 域随机化、域适应、系统辨识、Teacher-Student、Real-to-Sim-to-Real；经典案例；2026 新趋势 |
| [可微分物理与GPU加速](可微分物理与GPU加速.md) | MJX、Genesis、Brax、DiffTaichi；梯度基 RL；Isaac Gym 并行；RTX 渲染；神经渲染 |

## 相关专题

- 运动控制与 locomotion 详见 [../05-运动与导航/](../05-运动与导航/)
- 数据采集与训练流程详见 [../07-数据与遥操作/](../07-数据与遥操作/)
- VLA 模型与世界模型详见 [../01-VLA模型/](../01-VLA模型/) 与 [../02-世界模型/](../02-世界模型/)
- 硬件本体与传感器详见 [../08-硬件与本体/](../08-硬件与本体/)

---

[返回具身智能目录](../README.md)

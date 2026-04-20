# 具身智能 (Embodied AI)

> 机器人感知、决策与控制的交叉领域。2025-2026 年进入从"实验室演示"到"小规模量产"的关键拐点：VLA 双系统架构成为共识，世界模型成为训练的第三条腿，人形机器人 BOM 跌至 $20-35K 区间，数据飞轮开始闭环。

本目录按 11 个专题组织，每个专题都有 `README.md` 汇总**当前发展现状**（时间线 + 共识 + 开放问题）与**知识库索引**。

## 专题导航

### [00 · 基础与架构](00-基础与架构/README.md)
核心概念、感知-决策-执行三层架构、发展脉络与关键里程碑（SayCan → RT-2 → π₀ → Helix → GR00T N1 → π₀.5 → π₀.7 → Cosmos 2.0）

### [01 · VLA 模型](01-VLA模型/README.md)
Vision-Language-Action 基础模型专题。涵盖：
- VLA 架构演进（RT-2 离散 token → π₀ Flow Matching → 双系统双频）
- 主流模型 2026：**π₀ / π₀.5 / π-fast / π₀.7**、GR00T N1/N1.5、Figure Helix/Helix v2、Gemini Robotics/ER、RDT-2、GO-1、SmolVLA、OpenVLA-OFT
- 动作生成方法（离散 token / Diffusion / Flow Matching / 自回归 Transformer）
- 双系统架构（S2 慢思考 7-9Hz + S1 快执行 100-200Hz）
- 轻量化与端侧部署（Jetson Thor、INT4 量化、MoE VLA）

### [02 · 世界模型](02-世界模型/README.md)
世界模型（World Model）专题。NVIDIA Cosmos、Google Genie 1/2/3、Meta V-JEPA 2、1X World Model、UniSim；Dreamer 式耦合、VLA + Cosmos 增广、VLA 内嵌预测头、规划+执行分离四条融合路线。

### [03 · 策略学习方法](03-策略学习方法/README.md)
模仿学习（BC / DAgger / ACT）、扩散策略（Diffusion Policy / DP3）、强化学习（PPO / SAC / Teacher-Student / ANYmal Parkour），以及 2025-2026 热点 **VLA + RL 后微调**（DPPO / ReinFlow / RLDG）。

### [04 · 机器人操控](04-机器人操控/README.md)
抓取与位姿预测、灵巧操作、长 horizon 任务、双臂与全身操控。GraspNet / AnyGrasp、OpenAI Rubik's Cube、SayCan / VoxPoser、ALOHA / Mobile ALOHA、Figure Helix 全身协调。

### [05 · 运动与导航](05-运动与导航/README.md)
四足与人形 locomotion（ANYmal Parkour / Extreme Parkour / 电动 Atlas / Unitree H2）、全身控制 WBC（QP-WBC / MPC / Loco-Manipulation）、视觉语言导航 VLN（PointNav → VLMaps → SayNav → NaVILA / Mobility VLA）。

### [06 · 仿真与 Sim2Real](06-仿真与Sim2Real/README.md)
主流仿真器对比（Isaac Sim/Lab、MuJoCo/MJX、Habitat、SAPIEN/ManiSkill 3、Genesis）、Sim-to-Real 方法（域随机化 / 域适应 / 系统辨识 / Teacher-Student / Real-to-Sim-to-Real）、可微分物理与 GPU 加速。

### [07 · 数据与遥操作](07-数据与遥操作/README.md)
数据源与采集（五条路径）、主流数据集（**OXE / AgiBot World / DROID / Ego4D**）、数据工厂与飞轮、跨本体学习（π₀ / Octo / UniAct / GR00T N1）。

### [08 · 硬件与本体](08-硬件与本体/README.md)
人形机器人厂商地图（美国 7 家 + 中国 10+ 家）、灵巧手与触觉传感（Allegro / LEAP / Shadow / Tesla Optimus Hand Gen2 / Figure Hand v2 / 因时 RH56DFX / PaXini DexH13；GelSight / Digit 360）、执行器与 BOM 成本（2024 $100K → 2026 $20-35K → 2030 <$10K）。

### [09 · 评测基准](09-评测基准/README.md)
经典基准（SIMPLER / LIBERO / CALVIN / Meta-World）与 2025-2026 新生基准（PARTNR / RoboArena / RoboSpatial-VLM / GR-1 Benchmark / AgiBot World Eval）的对比，评测指标从"仿真成功率"演进到"跨本体 + 人类评分 + 长尾鲁棒性"。

### [10 · 产业与展望](10-产业与展望/README.md)
战略主线与技术拐点、主要玩家与战略定位、核心瓶颈与机会。三条技术主线 + 两条产业主线；垂直整合（Figure/Tesla/1X）vs 水平分层（Physical Intelligence/NVIDIA/Skild）；2026 四大瓶颈（数据 / 速度 / 泛化 / 成本）与研究/工程/投资机会。

---

## 快速入口

- **刚入门？** 从 [00 基础与架构](00-基础与架构/README.md) → [01 VLA 模型](01-VLA模型/README.md) → [03 策略学习方法](03-策略学习方法/README.md) 顺序阅读
- **关注 2026 前沿？** 直接看 [01 VLA 主流模型 2026](01-VLA模型/主流VLA模型2026.md) 和 [02 主流世界模型 2026](02-世界模型/主流世界模型2026.md)
- **做工业落地？** [08 硬件与本体](08-硬件与本体/README.md) + [10 产业与展望](10-产业与展望/README.md) + [07 数据与遥操作](07-数据与遥操作/README.md)
- **想找研究方向？** [10 核心瓶颈与机会](10-产业与展望/核心瓶颈与机会.md) 有 8 个研究方向 + 博士题目建议

---

[返回总目录](../README.md)

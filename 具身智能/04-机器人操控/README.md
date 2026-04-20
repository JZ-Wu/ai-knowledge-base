# 04 机器人操控 (Manipulation)

本目录聚焦机器人与环境物理交互的核心能力：抓取、灵巧操作、长 horizon 任务、双臂与全身操控。

## 2026 操控发展现状

### 时间线

| 年份 | 里程碑 |
|------|--------|
| 2016 | Dex-Net / Levine et al. 大规模抓取学习，数据驱动抓取起步 |
| 2019 | OpenAI Rubik's Cube，灵巧手 sim2real 里程碑 |
| 2020 | GraspNet-1Billion 发布，6-DoF 抓取有了大数据集 |
| 2022 | SayCan，LLM 首次用于操控任务分解 |
| 2023 | Diffusion Policy / ACT / ALOHA 定义模仿学习新范式；RT-2 开启 VLA 路线 |
| 2024 | Mobile ALOHA、π₀、OpenVLA，端到端 VLA 全面铺开 |
| 2025 | π₀.5、Helix、GR00T N1/N2，VLA 成为操控默认架构；灵巧手量产起步 |
| 2026 | VLA + 触觉融合 + 分钟级长 horizon 成为研究热点；双臂双机器人协同进入量产演示 |

### 当前共识（2026-04）

1. **VLA 端到端已成主流**：Diffusion Policy / ACT 作为默认基线，RT-2 / OpenVLA / π₀ / GR00T 系列定义大模型路线，特定任务的专用网络越来越少。
2. **Action Chunking 是标配**：无论 ACT 还是 Diffusion Policy，一次预测一段动作（8–50 步）显著优于逐步预测。
3. **灵巧手量产手落地**：Tesla Optimus Hand Gen2、Figure Hand v2、Sanctuary Phoenix Hand 等进入小批量产，PaXini、因时 RH56 进入国产供应链。
4. **触觉融合重要性被重新认识**：GelSight、Digit 360、BioTac 与 VLA 组合成为灵巧操作新标配。
5. **单网络全身控制可行**：Figure Helix 用单一 35-DOF 策略网络直接输出全身动作；不必再上下半身分层。

### 开放问题

- **分钟级长 horizon**：当前 VLA 只有几秒 context，如何引入工作记忆与子任务追踪仍未有共识方案。
- **接触丰富任务的仿真-真实 gap**：柔性体、滑动摩擦的仿真精度仍不足以支撑 sim2real。
- **数据瓶颈**：遥操作数据采集速度跟不上 scaling 需求，需要视频预训练 + 仿真 + 真机的混合配方。
- **泛化评估缺乏标准**：不同论文在不同物体/环境评测，跨实验室复现困难。

## 索引

- [抓取与位姿预测](抓取与位姿预测.md)：分析式抓取、6-DoF 抓取网络、学习式策略对比
- [灵巧操作](灵巧操作.md)：灵巧手硬件、经典任务、遥操作 + 模仿、触觉融合
- [长 horizon 任务](长horizon任务.md)：SayCan/VoxPoser、分层策略、VLA 的上下文限制
- [双臂与全身操控](双臂与全身操控.md)：ALOHA/RDT/Helix，动作空间设计

## 相关目录

- [../01-VLA模型/](../01-VLA模型/)：VLA 是当前操控的主架构
- [../03-策略学习方法/](../03-策略学习方法/)：Diffusion Policy、ACT、模仿学习细节
- [../07-数据与遥操作/](../07-数据与遥操作/)：操控数据的来源
- [../08-硬件与本体/](../08-硬件与本体/)：灵巧手与机械臂硬件
- [../05-运动与导航/](../05-运动与导航/)：Loco-Manipulation 与全身控制

---
[返回具身智能目录](../README.md)

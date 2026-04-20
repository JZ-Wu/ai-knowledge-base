# 视觉语言导航 VLN (Vision-and-Language Navigation)

视觉语言导航指机器人根据自然语言指令在未知环境中导航的能力。从 2018 年 R2R 基准提出到 2025 年 LLM-Planner 路线成为主流，VLN 经历了 LSTM → Transformer → VLM-based → LLM Planner 的范式迁移。

## 1. 经典导航架构

```
传感器 (LiDAR / RGB-D / IMU)
      │
      ▼
┌──────────────┐
│   SLAM       │  输出: 地图 + 当前位姿
└───────┬──────┘
        ▼
┌──────────────┐
│ 全局路径规划  │  A* / Dijkstra / RRT
└───────┬──────┘
        ▼
┌──────────────┐
│ 局部路径规划  │  DWA / TEB / MPC
└───────┬──────┘
        ▼
┌──────────────┐
│ 运动控制     │  PID / MPC → 线速度 / 角速度
└──────────────┘
```

这种模块化架构至今仍是工业部署主流。学习式导航（尤其 VLN）主要改造其中"路径规划"或全流程。

## 2. SLAM 基础

### 2.1 主流 SLAM 方法

| 方法 | 年份 | 传感器 | 类型 | 特点 |
|------|------|--------|------|------|
| **GMapping** | 2007 | 2D LiDAR | 粒子滤波 | 经典 2D SLAM |
| **Cartographer** (Google) | 2016 | 2D/3D LiDAR | 图优化 + 回环 | 开源标杆 |
| **ORB-SLAM3** | 2020 | 单目/双目/RGB-D + IMU | 特征点法 | 视觉 SLAM 经典 |
| **VINS-Fusion** | 2019 | 视觉 + IMU | 紧耦合优化 | 无人机常用 |
| **LIO-SAM** | 2020 | 3D LiDAR + IMU | 激光惯性里程计 | 地面机器人主流 |
| **RTAB-Map** | 长期维护 | RGB-D / LiDAR | 在线建图 | 大规模场景 |
| **Fast-LIO / Fast-LIO2** | 2021-2022 | LiDAR + IMU | 紧耦合滤波 | 高实时性 |
| **DROID-SLAM** | 2021 | 视觉 | 深度学习 SLAM | 深度估计 + 联合优化 |
| **NeRF-SLAM / iMAP / NICE-SLAM** | 2022-2023 | RGB-D | 隐式表示 | 神经辐射场 SLAM |
| **GS-SLAM / SplaTAM** | 2024-2025 | RGB-D | 3D Gaussian Splatting | 重建 + 渲染质量 SOTA |

### 2.2 选型建议

| 场景 | 推荐 |
|------|------|
| 室内 2D 移动机器人 | Cartographer / GMapping |
| 室内 3D RGB-D | RTAB-Map / ORB-SLAM3 |
| 室外自动驾驶 | LIO-SAM / Fast-LIO2 |
| 无人机 / AR | VINS-Fusion / ORB-SLAM3 |
| 高质量重建 | GS-SLAM / SplaTAM |

## 3. 学习式导航任务谱系

### 3.1 任务定义

| 任务 | 指令 | 难度 | 代表 benchmark |
|------|------|------|----------------|
| **PointNav** | 目标坐标 $(x, y)$ | 低 | Habitat PointNav |
| **ObjectNav** | 物体类别 "找电视" | 中 | Habitat ObjectNav Challenge |
| **ImageNav** | 目标图像 | 中 | ImageNav |
| **InstanceImageNav** | 特定实例图像 | 中高 | Habitat-Matterport |
| **VLN-R2R** | 详细语言指令 | 高 | Room-to-Room |
| **VLN-CE** | 连续动作空间 VLN | 高 | VLN Continuous |
| **REVERIE** | 远程物体引用表达 | 高 | REVERIE |
| **HM3D-OVON** | 开放词汇 ObjectNav | 2024 前沿 | HM3D |
| **OpenEQA** | 具身问答 (导航+问答) | 2024 前沿 | OpenEQA |

### 3.2 PointNav 的"已解决"

**DD-PPO (Wijmans et al., 2019)**：
- Habitat 中分布式 PPO，2.5 亿帧训练
- PointNav 达到 **99%+ SPL**，接近完美
- 启示：**足够多数据 + 简单 RL = 强导航**
- PointNav 因此几乎不再是研究热点

### 3.3 ObjectNav 的持续挑战

| 方法 | 类别 | 2023-2024 成功率 |
|------|------|-------------------|
| **模块化**：检测 + 语义地图 + 前沿探索 | 传统 | 30-50% |
| **端到端 RL** (PPO / DD-PPO) | 学习 | 25-45% |
| **VLM-based (CLIP-on-Wheels, OVRL)** | 开放词汇 | 35-55% |
| **LLM 常识增强 (L3MVN, LGX)** | LLM prompt | 40-60% |
| **SemExp** | 经典语义探索 | 30% |
| **PIRLNav** | 预训练 IL + 微调 RL | 60%+ |

ObjectNav 当前最好约 60-70%，距离"解决"还远。

## 4. VLN 主流方法演进

### 4.1 第一代：LSTM + Attention (2018-2020)

- **Seq2Seq (Anderson, 2018)**：R2R 基线，编码指令 + 观测 → LSTM → 动作
- **Speaker-Follower**：用 Speaker 增广指令数据
- **RCM (Reinforced Cross-Modal Matching)**：RL + IL 混合训练

### 4.2 第二代：Transformer-based (2020-2022)

| 方法 | 年份 | 核心 |
|------|------|------|
| **VLN-BERT** | 2020 | 预训练视觉-语言-动作 Transformer |
| **PREVALENT** | 2020 | 大规模轨迹-指令预训练 |
| **Recurrent VLN-BERT** | 2021 | 时序 Transformer |
| **HAMT** | 2021 | History-Aware Multimodal Transformer |
| **DUET** | 2022 | 拓扑图 + 细粒度视图双尺度 |
| **BEVBert** | 2022 | BEV 表示 + VLN |

### 4.3 第三代：VLM / LLM-based (2023-2025)

| 方法 | 年份 | 核心 |
|------|------|------|
| **NavGPT** | 2023 | GPT-4 零样本导航推理 |
| **LM-Nav** | 2022 | LLM 提取路径 + VLM 地标匹配 |
| **VLMaps** | 2023 | CLIP 特征融合到 3D 地图 |
| **ConceptFusion** | 2023 | 多模态特征融合到 3D |
| **SayNav** | 2023 | LLM 生成 planning graph，逐步导航 |
| **SayCanNav** | 2023 | SayCan 思想 + 导航 |
| **NaVid** | 2024 | 大视频模型作为导航 VLA |
| **NaVILA / NavBench** | 2024 | 人形语言导航 |
| **VLFM** | 2024 | Value Language Frontier Map |
| **Mobility VLA** (Google) | 2024 | 多模态基础模型端到端导航 |
| **EmbodiedGPT** | 2023 | LLM + embodied planning |
| **OpenNav** | 2024 | 开放词汇导航 |

### 4.4 第四代：LLM 作为规划器（2024-2026）

核心思路：**LLM 不直接输出动作，而是生成结构化规划**，由下层控制器执行。

```
语言指令 + RGB 历史 + 地图
        │
        ▼
   ┌──────────┐
   │   LLM    │ → 生成 Python 程序 / 结构化 plan
   │ Planner  │
   └────┬─────┘
        │
        ▼
  ┌─────────────┐
  │ 低层导航栈   │ ← PointNav / ObjectNav 已 "solved"
  │ (RL / A*)    │
  └─────────────┘
```

**优势**：
- 零样本泛化：LLM 常识可以推"电视通常在客厅"
- 可解释：plan 是人可读的 Python
- 组合能力：可以递归拆分指令

**代表**：
- **SayNav** (NVIDIA, 2023)：LLM 生成 "graph plan"
- **SayCanNav** (NUS, 2023)：价值函数过滤 LLM 候选
- **NaVid** (2024)：视频 VLM 直接输出连续导航动作
- **OK-Robot** (2024)：开放词汇移动操控
- **Mobility VLA** (Google, 2024)：多模态基础模型 + topological graph memory

## 5. Habitat 与仿真基准

### 5.1 Habitat Challenge 里程碑

| 年份 | 任务 | 冠军方法 | 分数 |
|------|------|----------|------|
| 2019 | PointNav | OccupancyAnticipation | ~0.95 |
| 2020 | ObjectNav | Red Rabbit | ~30% SPL |
| 2021 | PointNav (noisy) | — | ~0.85 |
| 2022 | ObjectNav | Stretch | ~60% |
| 2023 | ObjectNav | PIRLNav | ~65% |
| 2024 | OVON (open-vocab) | 多家接近 | 40-50% |

### 5.2 主流仿真平台

| 平台 | 机构 | 特点 |
|------|------|------|
| **Habitat 1.0/2.0/3.0** | Meta | 高速渲染，社区最大 |
| **iGibson** | Stanford | 交互式物理 |
| **AI2-THOR / ManipulaTHOR** | AI2 | 交互任务丰富 |
| **RoboCasa** | UT Austin | 厨房场景 |
| **BEHAVIOR** | Stanford | 1000 家务任务 |
| **Matterport3D** | Matterport | 真实室内扫描 |
| **HM3D / HM3D-Semantics** | Meta | 1000+ 真实扫描 |
| **Gibson** | Stanford | 早期视觉导航 |

## 6. 真实世界部署

### 6.1 工业应用

| 场景 | 机器人 | 导航方式 |
|------|--------|---------|
| 仓储 AMR | 极智嘉、Fetch、AutoGuide | 2D SLAM + 二维码/反光板 |
| 酒店送餐 | Pudu、云迹、Keenon | 2D LiDAR SLAM |
| 清洁 | 石头、科沃斯商用 | VSLAM / LiDAR |
| 巡检 | ANYmal、Spot、X30 | 3D SLAM + 预录点位 |
| 医院 | TUG、Aethon | 2D SLAM + 预定地图 |

### 6.2 VLN 何时能量产？

当前痛点：
- **长走廊飘移**：20m+ 走廊少特征，语义对齐容易错
- **楼层切换**：需手动给电梯/楼梯语义标注
- **动态环境**：人 / 物体移动导致语义地图过时
- **未见环境**：从未见过的建筑第一次进入仍易失败

共识：**2026 VLN 离量产还有 2–4 年**，量产前景景先结构化（固定地图 + 语言接口）而非完全开放。

## 7. 2026 前沿方向

### 7.1 LLM / VLM 作为导航大脑

- **从 token 输出到 planning 输出**：LLM 不再只说话，而是输出可执行 plan
- **topological memory**：Mobility VLA 用拓扑图作为 LLM 的长期记忆
- **多轮对话导航**：用户可以问 "厨房在哪" "请带我去"

### 7.2 开放词汇 + Affordance

- HM3D-OVON 基准推动开放词汇 ObjectNav
- CLIP / SigLIP 特征嵌入 3D 地图（VLMap、ConceptFusion、ConceptGraphs）
- 查询 "可以坐下的地方" → 直接在 3D 地图高亮

### 7.3 人形 VLN

- Digit、Figure 02 已经开始在仓储环境下做 VLN
- 挑战：上下楼梯 + 长走廊 + 避让行人，比轮式难得多
- NaVILA 等工作开启人形 VLN 研究

### 7.4 Embodied QA 与对话式导航

- **OpenEQA** (Meta, 2024)：具身问答 benchmark
- 机器人不仅导航，还要回答"冰箱里有牛奶吗？"这类需要探索后回答的问题
- LLM + 长时记忆 + 多模态感知的综合考验

## 8. 开放问题

1. **长走廊的语义漂移**：弱特征环境下 LLM 推理容易错位
2. **跨楼层 VLN**：电梯、楼梯的语义推理仍弱
3. **多机器人导航 + 调度**：仓储场景 50+ 机器人协调
4. **在线地图更新**：语义地图如何随环境变化自动更新
5. **隐私与室内扫描**：家庭场景 VLN 涉及隐私合规
6. **统一评测**：Habitat 之外缺乏跨 benchmark 横向比较

## 参考

- Habitat 2.0: Szot et al., NeurIPS 2021
- DD-PPO: Wijmans et al., ICLR 2020
- R2R: Anderson et al., CVPR 2018
- VLMaps: Huang et al., 2023
- NavGPT: Zhou et al., 2023
- Mobility VLA: Google DeepMind, 2024
- Open-EQA: Majumdar et al., 2024
- NaVILA: 2024

---
[返回本目录](README.md) | [上一篇：全身控制 WBC](全身控制WBC.md) | [返回具身智能目录](../README.md)

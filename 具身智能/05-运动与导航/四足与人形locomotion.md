# 四足与人形 Locomotion

Locomotion 指机器人的运动控制：让机器人在各种地形上稳定、快速、节能地移动。2021–2026 年，这一方向被 RL + 大规模并行仿真 + sim2real 的组合拳彻底重塑。

## 1. 硬件全景 (2026)

### 1.1 四足机器人

| 机器人 | 厂商 | 重量 | 负载 | 最大速度 | 定价 | 备注 |
|--------|------|------|------|----------|------|------|
| **Spot** | Boston Dynamics | 32 kg | 14 kg | 1.6 m/s | US $75k+ | 工业巡检标杆 |
| **ANYmal D** | ANYbotics | 50 kg | 15 kg | 1.0 m/s | EUR 150k+ | 工业 IP67 |
| **Unitree Go2** | 宇树 | 15 kg | 5 kg | 3.7 m/s | US $1.6k–5k | 消费级，最便宜 |
| **Unitree B2** | 宇树 | 60 kg | 20 kg | 6.0 m/s | US $40k+ | 国产工业级 |
| **DeepRobotics X30** | 云深处 | 50 kg | 15 kg | 4.0 m/s | RMB 30w+ | 国产工业巡检 |
| **Xiaomi CyberDog 2** | 小米 | 8 kg | — | — | 消费级 | 开发者友好 |

### 1.2 人形机器人

| 机器人 | 厂商 | 高度 | 重量 | 自由度 | 驱动 | 状态 |
|--------|------|------|------|--------|------|------|
| **Cassie** | Agility Robotics | 1.1 m | 31 kg | 10（双腿） | 电机 | 2016 发布，学术标杆 |
| **Digit** | Agility Robotics | 1.75 m | 45 kg | 20+ | 电机 | 亚马逊、GXO 仓储部署 |
| **Atlas (电动)** | Boston Dynamics | 1.5 m | 89 kg | 28+ | 电机（取代液压） | 2024 发布电动版，2025 现代工厂试点 |
| **Optimus V2** | Tesla | 1.73 m | 57 kg | 28 | 电机 | Tesla 工厂试点 |
| **Optimus V3 (2025)** | Tesla | 约 1.73 m | 优化 | 40+（含手） | 电机 | 灵巧手 Gen2 集成 |
| **Figure 02** | Figure AI | 1.68 m | 60 kg | 35+ | 电机 | BMW、GXO 部署，Helix 驱动 |
| **Figure 03** | Figure AI | — | — | — | — | 2025 发布，家庭场景 |
| **1X Neo Beta / Gamma** | 1X Technologies | 1.6 m | 30 kg | — | 腱驱 + 电机 | 家用，软外壳 |
| **Apollo** | Apptronik | 1.72 m | 73 kg | 25+ | 电机 | 工业仓储 |
| **Unitree H1** | 宇树 | 1.80 m | 47 kg | 27 | 电机 | 国产最早量产之一 |
| **Unitree H2** | 宇树 | 更敏捷 | — | — | 电机 | 2025 升级版 |
| **Unitree G1** | 宇树 | 1.32 m | 35 kg | 23–43 | 电机 | 小尺寸，研究友好 |
| **小鹏 Iron** | 小鹏 | 1.78 m | 70 kg | 62 | 电机 | 从汽车工厂切入 |
| **宇树 / Pudu / 智元 远征 A2 / 灵犀 X2** | 多家国产 | ~1.7 m | ~60 kg | — | 电机 | 2025-2026 国产潮 |
| **Sanctuary Phoenix Gen7** | Sanctuary AI | 1.7 m | 70 kg | 50+ | 混合 | 强调拟人能力 |

### 1.3 硬件趋势

- **全面电动化**：2024 电动 Atlas 终结液压时代，所有新品全部电机驱动。
- **准直驱电机**：低减速比高力矩密度电机（如谐波减速 + 无刷）成主流。
- **轻量化**：碳纤维骨架、镁铝合金壳体普及。
- **国产化**：宇树、智元、云深处、小鹏、众擎、傅里叶智能等形成完整产业链。
- **成本快速下降**：Optimus 目标 US $20k–30k，宇树 H1 已低于此。

## 2. 主流训练方法

### 2.1 Isaac Gym / Isaac Lab 并行训练

NVIDIA 的 GPU 并行仿真是 2020 年后足式运动 RL 的标配。

```
Isaac Gym / Isaac Lab
  ├── GPU 并行仿真 4096+ 环境
  ├── PPO 单策略并行采样
  ├── Legged Gym (ETH) / OmniGym 上层封装
  ├── 全流程 (物理 + 渲染 + 训练) 全部在 GPU
  └── 训练四足 locomotion: 几小时 vs MuJoCo 几天
```

**里程碑工作**：
- **Legged Gym** (ETH 2022)：ANYmal 四足 RL 训练框架
- **Isaac Lab** (NVIDIA 2024)：Isaac Gym 的继任者，OmniGym 模块化
- **Humanoid Gym / MuJoCo Playground**：人形版本

### 2.2 Teacher-Student 蒸馏

核心问题：仿真中有 ground truth（接触力、地形高度），现实没有。

**解决方案**：
1. **Teacher 训练**：带特权观测（完整地形、内部状态）的 RL 策略
2. **Student 训练**：只有机载传感器观测（IMU + 关节编码器 + 本体感觉）
3. **监督学习**：Student 模仿 Teacher 动作
4. **Sim-to-Real**：部署 Student，特权信息不可得也能跑

**代表工作**：
- **RMA (Rapid Motor Adaptation)** (Kumar et al., 2021)
- **Learning quadrupedal locomotion over challenging terrain** (Lee et al., ETH 2020, Science Robotics)
- **Extreme Parkour** (ETH, 2024)：复杂地形四足

### 2.3 MPC + RL 混合

- **MPC** (Model Predictive Control)：基于动力学模型做短时预测控制
- **RL policy**：学习 residual action 或高层策略
- 例子：Boston Dynamics Spot 的底层仍是 MPC，高层步态切换用 RL

### 2.4 从人体动作捕捉学习

人形机器人与人体形态相似，CMU MoCap / AMASS 成为宝贵资源。

| 方法 | 年份 | 核心 |
|------|------|------|
| **DeepMimic** | 2018 | 用参考动作做奖励 |
| **AMP** (Adversarial Motion Priors) | 2021 | 判别器代替手工奖励 |
| **ASE** (Adversarial Skill Embeddings) | 2022 | 学习技能 latent |
| **PHC** (Perpetual Humanoid Control) | 2023 | 持续人形控制，任意动作跟踪 |
| **HumanPlus** (Stanford, 2024) | 2024 | Shadowing：人动机器人动 |
| **OmniH2O** (CMU/NVIDIA, 2024) | 2024 | 通用人形遥操 + 模仿 |
| **H2O / HumanGym** | 2024 | 人形从人体数据学多技能 |
| **ExBody / ExBody2** | 2024 | 全身表达性动作 |

## 3. 步态与动力学基础

### 3.1 过驱动 vs 欠驱动

- **过驱动 (Overactuated)**：驱动数 ≥ 自由度，如四足（每条腿多关节），控制较容易。
- **欠驱动 (Underactuated)**：驱动数 < 自由度，如双足（落地时脚踝自由旋转），必须利用动力学耦合。
- **人形机器人的欠驱动**：支撑相一侧脚踝的 3 个方向无主动控制，是双足行走最大难点。

### 3.2 ZMP (Zero Moment Point)

支撑点上使水平惯性力矩为零的点。
- 若 ZMP 在**支撑多边形**内 → 脚不翻转，动态稳定
- 经典方法：MPC 生成 ZMP 轨迹 → IK 求关节角
- 代表：ASIMO、NAO、HRP 系列

**局限**：ZMP 要求完全脚底接触，对敏捷跑跳不适用。

### 3.3 Capture Point (CP) 与 DCM (Divergent Component of Motion)

$$\xi = x + \frac{\dot x}{\omega}, \quad \omega = \sqrt{g/z_c}$$

- **DCM $\xi$**：质心位置 $x$ + 速度 $\dot x$ 的加权和
- **Capture Point**：机器人要避免摔倒，必须把下一步落在 CP 附近
- 比 ZMP 更适合敏捷运动，是 Cassie/Digit 控制的理论基础

### 3.4 典型步态

| 步态 | 足相位 | 应用 |
|------|--------|------|
| Walk | 一次一条腿移动 | 低速稳定 |
| Trot | 对角线两腿同步 | 四足常用中速 |
| Bound | 前后腿对开 | 四足高速 |
| Gallop | 类似马 | 最高速四足 |
| Pace | 同侧腿同步 | 特殊步态 |
| **双足 Walk / Run** | 单支撑/双支撑交替 | 人形主流 |

## 4. 关键工作时间线

### 四足

| 工作 | 年份 | 贡献 |
|------|------|------|
| **MIT Cheetah 3** | 2018 | 盲走楼梯 |
| **Learning quadrupedal locomotion** | 2020 | ETH Science Robotics，Teacher-Student 真实跨地形 |
| **Rapid Motor Adaptation (RMA)** | 2021 | Adaptation module sim2real |
| **ANYmal Parkour** | 2022 | 翻越箱子、跳跃 |
| **Walk These Ways** | 2023 | 单策略多步态 |
| **Extreme Parkour** (CMU) | 2023 | 爬高墙、跳沟 |
| **Barkour** (Google) | 2023 | 敏捷跑酷评测 |
| **Legged Gym** (ETH) | 2022 | 开源 Isaac Gym 四足训练 |
| **Learning to Walk in Minutes** | 2022 | 100 倍速训练 |

### 人形

| 工作 | 年份 | 贡献 |
|------|------|------|
| **ASIMO** (Honda) | 2000 | 经典 ZMP 双足行走 |
| **Cassie paper (OSU)** | 2018 | 首个消费级欠驱动双足 RL |
| **Digit in warehouse** | 2022 | 商业化试点 |
| **Unitree H1 后空翻** | 2023 | 首款量产人形完成后空翻 |
| **OmniH2O** (CMU/NVIDIA) | 2024 | 全身遥操 |
| **HumanPlus** (Stanford) | 2024 | Shadowing 从人体学习 |
| **ExBody2** | 2024 | 敏捷全身 |
| **电动 Atlas** (BD) | 2024 | 全电驱升级 |
| **Figure Helix** | 2025 | 单网络 35-DoF 全身协调 |
| **Optimus V3** | 2025 | 灵巧手 + 人形集成 |
| **GR00T N1 / N2** (NVIDIA) | 2024-2025 | 人形基础模型开源 |

## 5. 2026 进展与趋势

### 5.1 电动 Atlas 的意义

- 2024 年 Boston Dynamics 宣布 HD Atlas 退役，**全电动 Atlas** 接棒
- 旋转关节范围远超人类（360° 扭转），力矩密度高
- 首次在 Hyundai 工厂内部署试点
- 象征：**液压时代结束，电动时代成为行业唯一路线**

### 5.2 全电驱 + 准直驱 + 轻量化

共识配方：
```
谐波减速 + 准直驱无刷电机 → 高力矩密度
碳纤维骨架 + 镁合金壳体 → 轻量
高容量锂电池 + 快充 → 数小时续航
SerDes 串行总线 → 关节通信
```

### 5.3 人形 RL 训练的规模化

- **Isaac Lab** 成为事实标准
- **MuJoCo MJX** 作为开源 GPU 仿真替代
- 单 H100 可并行 2048+ 人形环境
- 2026 年训练一个完整 locomotion 策略：数小时—1 天

### 5.4 量产级应用场景

| 场景 | 代表机器人 | 现状 |
|------|-----------|------|
| 仓储拣选 | Figure 02, Digit, Apollo | 已部署试点 |
| 汽车工厂装配 | Optimus, Iron | 试点 |
| 化工 / 电力巡检 | ANYmal, X30, Spot | 商业化 |
| 家庭服务 | 1X Neo, Figure 03 | 早期 |
| 应急救援 | Atlas, 专用四足 | 研究 |

## 6. 开放问题

1. **真实不平整地形的可靠性**：泥地、松软地面、结冰路面仍困难。
2. **摔倒恢复**：大多数量产机器人摔倒需人工扶起，自主起身仍不稳。
3. **跌倒保护**：如何软着地、如何保护硬件？
4. **全身运动 + 操控的数据效率**：拿着箱子上楼梯的数据极其稀缺。
5. **长时间续航**：家庭场景需要 8–12 小时，当前 2–5 小时。
6. **成本**：消费级 < US $10k 的目标还需 2–3 年。

## 参考

- Legged Gym: https://leggedrobotics.github.io/legged_gym/
- Isaac Lab: https://isaac-sim.github.io/IsaacLab/
- Extreme Parkour (CMU): Cheng et al., 2023
- HumanPlus (Stanford): Fu et al., 2024
- Boston Dynamics 电动 Atlas 发布: 2024-04
- Figure Helix: figure.ai/news/helix

---
[返回本目录](README.md) | [上一篇：README](README.md) | [下一篇：全身控制 WBC](全身控制WBC.md)

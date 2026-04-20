# Sim-to-Real 方法

> 仿真中训练的策略部署到真实机器人时的性能差距称为 **Sim-to-Real Gap**（Reality Gap）。本文梳理主流方法、经典案例与 2025-2026 新趋势。

## 一、Reality Gap 的来源

| 类型 | 仿真 vs 真实差异 | 示例 |
|------|----------------|------|
| **物理差异** | 接触动力学、摩擦、形变 | 仿真抓稳的物体真实中滑落 |
| **视觉差异** | 渲染 vs 真实图像 | 光照、纹理、反射 |
| **传感器差异** | 理想传感器 vs 真实噪声 | 深度相机空洞、IMU 漂移 |
| **执行差异** | 理想电机 vs 真实延迟 | 响应延迟、齿轮间隙、力矩饱和 |
| **时间差异** | 离散时间步 vs 连续物理 | 控制频率、通信延迟、异步 |
| **分布差异** | 仿真物体库 vs 真实长尾 | 半透明、反光、可变形物体 |

---

## 二、主要方法

### 2.1 域随机化 (Domain Randomization)

**核心思想**：训练时大幅随机化仿真参数，使策略对参数变化鲁棒，只要真实参数落在随机范围内即可迁移——"把真实世界看作众多随机仿真中的一个"。

#### 物理参数随机化

```python
# Isaac Lab 物理域随机化示例
randomization_params = {
    "friction":         {"range": [0.3, 1.5]},    # 摩擦系数
    "object_mass":      {"range": [0.1, 2.0]},    # 物体质量
    "joint_damping":    {"range": [0.5, 3.0]},    # 关节阻尼
    "motor_strength":   {"range": [0.8, 1.2]},    # 电机力矩系数
    "gravity":          {"range": [9.5, 10.1]},   # 重力
    "action_delay":     {"range": [0, 3]},        # 动作延迟（帧）
    "observation_noise":{"std": 0.02},            # 传感器噪声
    "external_force":   {"range": [0, 10]},       # 随机外力扰动
    "com_offset":       {"range": [-0.02, 0.02]}, # 重心偏移
    "ground_roughness": {"range": [0.0, 0.1]},    # 地面凹凸
}
```

#### 视觉域随机化

```python
visual_randomization = {
    "lighting": {
        "intensity": [0.3, 3.0],
        "color": "random_RGB",
        "direction": "random_direction",
    },
    "texture":     "random_textures",         # 随机纹理
    "camera": {
        "position_noise": 0.02,               # 相机位置扰动
        "fov": [55, 75],                      # 视场角
        "roll_noise": 5,                      # 相机倾斜
    },
    "distractors":  "random_background_objs", # 随机背景干扰物
    "material":     "random_material",        # 材质随机
}
```

#### 自适应域随机化 (ADR)

自动调整随机化范围：
- 策略在当前范围成功率高 → 扩大范围
- 策略成功率低 → 缩小范围
- 逐步扩展到策略承受极限

**代表工作**：
- **Tobin et al. (2017)**：开山之作，视觉域随机化用于目标检测 Sim-to-Real
- **OpenAI Rubik's Cube (2019)**：数百个参数同时随机化，零样本迁移 Shadow Hand 解魔方
- **ANYmal (ETH, 2019-2024)**：四足机器人系列，域随机化 + Teacher-Student 标杆
- **Extreme Parkour (CMU, 2024)**：四足跳跃、爬箱、上下台阶

### 2.2 域适应 (Domain Adaptation)

不靠"暴力随机化"，而是学习**对齐仿真和真实的特征表示**。

```
仿真图像 ──→ 编码器 ──→ 共享特征空间 ←── 编码器 ←── 真实图像
                         │
                     策略网络
                         │
                       动作
```

**常见方法**：
- **对抗训练**：判别器区分仿真/真实特征，编码器对抗 → 特征对齐
- **图像翻译 (CycleGAN)**：将仿真图像转换为真实风格
- **风格迁移**：保留内容（物体位置）但改变风格（纹理、光照）

**RL-CycleGAN**（Google, 2020）：

```
仿真图像 ──→ G_sim2real ──→ "仿真真实图像" → 策略
真实图像 ──→ G_real2sim ──→ "真实仿真图像"
           对抗损失 + 循环一致性损失
```

**与域随机化对比**：
- DR 简单通用，零样本迁移
- DA 利用真实数据对齐，精度更高但需要真实数据

### 2.3 系统辨识 (System Identification)

精确测量/估计真实环境参数，在仿真中匹配：

- **物理参数辨识**：真机实验拟合摩擦、惯量、电机参数
- **学习式辨识**：训练网络从传感器数据推断仿真参数

```
Offline：
  真实实验数据 → 优化仿真参数 → 仿真匹配真实

Online：
  真实传感器数据 → 参数估计网络 → 仿真参数
       ↓
  在估计参数的仿真中训练/微调策略
```

**典型应用**：
- 精密操控（手术机器人、装配）
- 软体操作（需要精确弹性参数）
- 柔性物体（布料、绳索）

### 2.4 Teacher-Student 框架

这是 locomotion Sim-to-Real 的**事实标准范式**，由 ETH ANYmal 系列推广。

```
┌─────────────────────────────────────────────────────┐
│ 阶段 1: 仿真中训练 Teacher                            │
│                                                     │
│ 观测 = [关节角, 角速度, IMU,                          │
│         地形高度图, 接触力, 真实速度]   ← 特权信息     │
│         │                                           │
│    ┌────▼────┐                                      │
│    │ Teacher │  ← PPO 训练                           │
│    │ Policy  │                                      │
│    └────┬────┘                                      │
│         │                                           │
│    目标关节角 → PD 控制器 → 力矩                       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│ 阶段 2: 蒸馏 Student                                 │
│                                                     │
│ 观测 = [关节角, 角速度, IMU]   ← 仅真机可得           │
│         │                                           │
│    ┌────▼────┐     ┌─────────┐                      │
│    │ Student │ ──→ │ 隐式估计 │  ← 学会从有限传感器   │
│    │ Policy  │     │ 地形/接触│     推断隐藏信息       │
│    └────┬────┘     └─────────┘                      │
│         │                                           │
│    模仿 Teacher 的输出动作（行为蒸馏）                  │
└─────────────────────────────────────────────────────┘

阶段 3: Student 部署到真实机器人
```

**为什么有效**：
- Teacher 有完美信息，容易训练出强策略
- Student 通过模仿 Teacher，隐式学会从噪声传感器估计隐藏状态
- 等价于端到端的**状态估计器 + 控制器**

**变体**：
- RMA（Rapid Motor Adaptation, 2021）：显式 encoder 编码环境参数
- DreamWaQ、HIMLoco：隐式估计世界模型

### 2.5 Real-to-Sim-to-Real

先从真实环境重建仿真场景，再在仿真中训练：

```
真实场景扫描 → 仿真场景重建 → RL/IL 训练 → 部署到真实
  (NeRF/3DGS)    (加物理属性)
```

**关键技术**：
- **NeRF / 3D Gaussian Splatting** 重建视觉
- 加上碰撞体 + 物理属性 → 可交互仿真
- 缩小视觉 gap，物理 gap 仍需处理

**代表工作**：
- **Robo-GS** (2024)：3DGS 重建家庭场景用于操控训练
- **URDFormer** (2024)：从图像生成铰接物体 URDF
- **DigitalTwin4 Robots**：工业场景数字孪生

### 2.6 仿真增强与高保真

提高仿真本身的逼真度：

- **GPU 加速物理**：更小时间步、更精确接触模型
- **光线追踪渲染**：NVIDIA RTX 近照片级
- **可微分仿真**：用真实数据梯度反传优化仿真参数
- **数据驱动物理**：神经网络学习接触/形变模型（见 [../02-世界模型/](../02-世界模型/)）

---

## 三、经典案例

### OpenAI Rubik's Cube (2019)

- Shadow Hand 零样本解魔方
- 数百物理 + 视觉参数同时随机化
- **ADR**（自适应域随机化）首次大规模验证
- 意义：证明纯仿真训练可以迁移到极复杂的真实灵巧操作

### ANYmal Parkour (ETH, 2022-2024)

- 四足机器人在野外跑酷
- Teacher-Student + 大量地形随机化
- Isaac Sim 上千并行环境
- 成为 locomotion Sim-to-Real 的教科书

### Extreme Parkour (CMU, 2024)

- 四足跳跃、爬 1 米高箱、跨过障碍
- 基于 Isaac Gym，域随机化 + Teacher-Student
- 演示 RL 策略可以超越 MPC 的极限动作

### Unitree G1 / Booster T1 (2024-2025)

- 双足人形 Sim-to-Real 跑步、跳舞
- 基于 Isaac Lab + GR00T 生态
- 加入 **身体参数随机化**（连杆质量、惯量）应对不同本体

---

## 四、方法对比

| 方法 | 优点 | 缺点 | 适用场景 |
|------|-----|-----|---------|
| **域随机化** | 简单、通用、零样本 | 过度随机化降低性能 | Locomotion、简单操控 |
| **域适应** | 利用真实数据对齐 | 需真实数据、训练复杂 | 视觉策略 |
| **系统辨识** | 精确、可解释 | 测量困难、泛化差 | 精密操控 |
| **Teacher-Student** | 标准、有效 | 两阶段训练 | Locomotion 标配 |
| **Real-to-Sim-to-Real** | 缩小视觉 gap | 重建质量受限 | 特定场景操控 |
| **高保真仿真** | 直接降低 gap | 仿真成本高 | 精密任务 |

---

## 五、2025-2026 新趋势

### 5.1 世界模型作为 Sim-to-Real 桥梁

2025 年最大的范式转变：**世界模型补充甚至部分替代传统仿真**。

**Cosmos Transfer**（NVIDIA, 2025）：
- 输入：仿真渲染的视频
- 输出：照片级真实视频（保持几何 + 物理一致）
- 用途：仿真策略训练图像 → Cosmos Transfer → 获得"真实风格"图像 → 视觉 Sim-to-Real gap 大幅缩小

**GR00T Dreams**（NVIDIA, 2025）：
- 少量真实轨迹 → Cosmos 生成变体 → 大规模训练
- 基于真实视觉分布（优于仿真渲染）
- 可控增广（光照/物体/视角）

**局限**：
- 动作维度难以"生成"（物理正确性）
- 目前主要用于 VLA 视觉 encoder 预训练与视觉域适应

### 5.2 Foundation Model 本身降低 Sim-to-Real 难度

VLA 模型的视觉理解能力天然对视觉变化更鲁棒：
- CLIP/SigLIP 等视觉 encoder 在互联网图像上预训练，已见过海量真实分布
- 细调后，模型关注"语义"而非像素级纹理 → 对仿真纹理失真不敏感
- **π₀ 的策略**：仿真数据 + 真实数据联合训练，仿真提供多样性，真实提供分布对齐

### 5.3 可微分仿真与端到端优化

- MJX、Genesis、Brax 提供物理梯度
- **思路 A**：通过物理梯度直接优化策略
- **思路 B**：通过真实数据反传优化仿真参数（参数化 Sim-to-Real）
- 2026 年尚未成为主流，但在软体、装配等课题上有优势（详见 [可微分物理与GPU加速](可微分物理与GPU加速.md)）

### 5.4 混合数据训练

```
数据源混合训练:
  真实机器人数据 (OXE + 自采)
  仿真数据 (Isaac / Genesis)
  世界模型生成 (Cosmos Transfer)
  人类视频 (Ego4D)

→ 统一 VLA 训练
→ Sim-to-Real 作为数据分布问题而非单独阶段
```

### 5.5 数字孪生 + 持续学习

- 高精度重建真实环境 → 仿真测试 → 直接部署
- 部署后从真实交互持续改进（详见 [../07-数据与遥操作/数据工厂与飞轮.md](../07-数据与遥操作/数据工厂与飞轮.md)）

---

## 六、典型 Sim-to-Real 工作流（以四足机器人为例）

```
1. 搭建仿真环境
   - 导入 URDF/MJCF
   - 地形课程：平地 → 台阶 → 随机地形
   - 标定基本物理参数

2. 训练 Teacher（PPO，1-2 小时）
   - 特权观测（地形扫描、真实速度、接触力）
   - 课程学习
   - 域随机化：摩擦、质量、电机延迟

3. 蒸馏 Student（~30 分钟）
   - 仅用真机可得观测
   - 行为克隆模仿 Teacher

4. 真机测试
   - 先简单场景
   - 检查关节跟踪误差、步态稳定性
   - 常见问题：电机过热、振荡、滑倒

5. 迭代优化
   - 根据真机表现调整奖励 / 随机化范围
   - 一般 3-5 轮迭代
```

---

## 参考文献

- Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World", IROS 2017
- OpenAI, "Solving Rubik's Cube with a Robot Hand", 2019
- Lee et al. (ETH), "Learning Quadrupedal Locomotion over Challenging Terrain", Science Robotics 2020
- Kumar et al., "RMA: Rapid Motor Adaptation", RSS 2021
- Zhuang et al., "Robot Parkour Learning", CoRL 2023
- Cheng et al., "Extreme Parkour with Legged Robots", ICRA 2024
- NVIDIA, "Cosmos World Foundation Model Platform", 2025.01

---

[返回本专题](README.md) · 上一篇：[主流仿真器对比](主流仿真器对比.md) · 下一篇：[可微分物理与 GPU 加速](可微分物理与GPU加速.md)

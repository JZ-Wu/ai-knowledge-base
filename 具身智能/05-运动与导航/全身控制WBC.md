# 全身控制 WBC (Whole-Body Control) 与 Loco-Manipulation

全身控制（Whole-Body Control, WBC）是把机器人所有关节当成一个整体，同时完成多个目标（保持平衡、末端跟踪、避障、节能）的控制范式。Loco-Manipulation 则是 WBC 的高阶场景：让腿脚与手臂协同工作，拿着东西走路、开门、搬箱子。

## 1. 经典 WBC：QP-based 优化

### 1.1 核心思想

把控制问题写成带约束的二次规划 (QP)：

$$\begin{aligned}
\min_{\ddot{q}, \tau, F} \quad & \sum_i w_i \| J_i \ddot{q} + \dot{J}_i \dot{q} - \ddot{x}^{des}_i \|^2 \\
\text{s.t.} \quad & M(q) \ddot{q} + h(q, \dot{q}) = S^T \tau + J_c^T F \quad \text{(动力学)} \\
& F \in \mathcal{K}_{friction} \quad \text{(摩擦锥)} \\
& \tau_{min} \le \tau \le \tau_{max} \\
& J_c \ddot{q} + \dot{J}_c \dot{q} = 0 \quad \text{(接触非滑动)}
\end{aligned}$$

- 决策变量：关节加速度 $\ddot{q}$、关节力矩 $\tau$、接触力 $F$
- 目标：多个任务的加权和（质心跟踪、末端位姿、姿态、关节预紧）
- 约束：刚体动力学、摩擦锥、力矩极限、接触约束

### 1.2 任务优先级

多任务冲突时（如平衡 vs 末端跟踪），用分层 QP 或零空间投影：
- **分层 QP (HQP)**：高优先级任务先满足，低优先级在零空间求解
- **优先级示例**：
  1. 动力学一致性（硬约束）
  2. 接触摩擦（硬约束）
  3. 质心平衡（最高优先级软任务）
  4. 手臂末端跟踪
  5. 姿态保持 / 关节偏好

### 1.3 代表实现

| 框架 | 机构 | 机器人 | 特点 |
|------|------|--------|------|
| **TSID** | LAAS | Talos, HRP | C++ + Python 接口 |
| **OpenSoT** | IIT | Walkman | HQP 框架 |
| **WholeBodyMPC** | ETH | ANYmal | MPC + WBC 融合 |
| **Drake** | MIT/TRI | Atlas | QP + 轨迹优化 |
| **Pinocchio + QP** | INRIA | 通用刚体动力学 | 开源标配 |

## 2. MPC (Model Predictive Control)

WBC 是单步优化，MPC 则在预测 horizon 内优化一段动作：

$$\min_{u_{0:N}} \sum_{k=0}^{N} \| x_k - x_k^{ref} \|^2_Q + \| u_k \|^2_R$$

满足动力学约束 $x_{k+1} = f(x_k, u_k)$。

### 2.1 足式 MPC 层级

```
高层 MPC (1-10 Hz)
  └── 预测质心轨迹 + 落足点
       │
       ▼
中层 MPC (100 Hz)
  └── 单刚体 MPC，优化接触力
       │
       ▼
低层 WBC (500-1000 Hz)
  └── QP 求解具体关节力矩
```

- **Convex MPC** (Cheetah 3, 2018)：单刚体 + 凸 QP，适合四足快速运动
- **Nonlinear MPC** (ANYmal)：完整动力学，能处理复杂地形
- **Real-time iteration SQP**：计算效率关键

### 2.2 MPC + RL 混合

| 架构 | 思路 | 代表 |
|------|------|------|
| **RL residual on MPC** | RL 学习 MPC 的补偿量 | Spot 部分实现 |
| **RL 选 MPC 模式** | RL 做步态切换，MPC 执行 | 工业四足 |
| **Diff MPC** | MPC 反传梯度训练策略 | OMPC、Theseus |

## 3. Loco-Manipulation (操控 + 运动协同)

### 3.1 为什么难

- **耦合动力学**：手臂运动改变质心位置 → 腿必须补偿平衡
- **接触变化**：搬箱子 = 多接触问题，摩擦约束复杂
- **感知切换**：手操作时头可能看手，脚底地形感知不足
- **数据稀缺**：人形拿着东西走路的真实数据数量级少于纯行走

### 3.2 典型任务

| 任务 | 难点 | 代表工作 |
|------|------|---------|
| **搬箱子** | 双手受力改变质心 | Digit 仓储、Apollo 演示 |
| **开门** | 一手拉把手 + 腿脚配合 | ALOHA + 底盘、ANYmal + 臂 |
| **推购物车** | 双手持续推力 + 行走 | HumanPlus |
| **拿咖啡行走** | 不能洒出来（晃动限制） | Helix |
| **爬梯子** | 手脚同时接触 | 研究级 |
| **蹲下捡东西** | 腰 + 腿 + 臂协同 | Optimus, Figure |

### 3.3 方法演进

#### 3.3.1 经典路线：WBC + 任务规划

```
任务规划器 → 生成末端轨迹 + 落足点序列
   ↓
WBC + MPC 执行
```
- ANYmal + 机械臂、Boston Dynamics Spot + 臂 (Spot Arm)
- 优点：稳定、可控、可解释
- 缺点：任务特定调参、不够智能

#### 3.3.2 端到端 RL 路线

- **Legged Manipulator** (ETH)：四足 + 臂联合 RL
- **DeepWBC** (NVIDIA)：学习 WBC 替代 QP 求解
- **Armour** (CMU)：四足 + 臂的 grasp on the go
- 优点：自动协调、涌现行为
- 缺点：sim2real gap 更大、训练难

#### 3.3.3 VLA 路线（2025-2026）

- **Helix** (Figure)：单网络直接出 35-DoF 动作，无显式 WBC
- **GR00T N2** (NVIDIA)：全身人形基础模型
- **UniAct** (智元)：全身 VLA
- 优点：语义驱动、跨任务泛化
- 缺点：数据饥渴、高频控制频率限制

### 3.4 上下半身：解耦 vs 统一

#### 解耦方案

```
上半身 VLA (操控)  ── 独立训练
下半身 RL (locomotion) ── 独立训练
          ↓
       通过腰关节通信（常简化为固定）
```

- **优势**：各自数据独立，模型小，训练快
- **劣势**：协调性差，搬重物时下半身不知道，容易摔
- **代表**：早期 HumanPlus、多数国产人形原型

#### 统一方案（Helix 路线）

```
单一 VLA → 全身 35-DoF 动作
  ↑
统一遥操作数据（头/腰/双臂/双手同步录制）
```

- **优势**：协调自然，涌现行为（扭腰避障、伸手够远）
- **劣势**：数据成本爆炸、网络规模大、推理延迟
- **代表**：Helix v1/v2、UniAct、GR00T N2

#### 2026 倾向

随着数据采集系统（Open-TeleVision + 全身遥操作支架）成熟，**统一方案正在超越解耦方案**，特别是在家庭场景。工厂结构化任务仍以解耦 + WBC 为主。

## 4. Mobile Manipulation 数据集与基准

### 4.1 RoboCasa (UT Austin, 2024)

- **目标**：大规模仿真厨房家务场景
- **规模**：100+ 厨房场景、2500+ 3D 物体
- **任务**：开冰箱、切菜、装盘、清洁台面
- **特色**：
  - 生成式场景合成（GenAI 参数化）
  - 支持人形（Humanoid）+ 双臂移动底盘
  - 跨 embodiment 基准
- **配套模型**：GR-1、MimicGen 数据增广

### 4.2 其他 Mobile Manipulation 数据 / 基准

| 名称 | 机构 | 重点 |
|------|------|------|
| **BEHAVIOR** / **BEHAVIOR-1K** | Stanford | 1000 家务任务 |
| **iGibson 2.0** | Stanford | 交互式家庭仿真 |
| **Habitat 3.0** | Meta | 人机共居家庭 |
| **ManiSkill2 / 3** | UCSD | 桌面 + 移动操控 |
| **OpenX-Embodiment** | 多机构联合 | 真机遥操作大规模数据 |
| **DROID** | Stanford/CMU | 多样化真实厨房演示 |
| **RH20T** | 清华 | 中文家务遥操作 |
| **AgiBot World** | 智元 | 2025 大规模全身数据集 |

### 4.3 AgiBot World 的重要性

- 智元 2025 发布，号称最大真机遥操作数据集之一
- 百万级 episode，全身 + 双臂 + 移动底盘
- 成为中国全身 VLA 预训练的重要基础设施

## 5. 双足全身 VLA 的挑战

### 5.1 控制频率矛盾

| 层级 | 需要频率 | 原因 |
|------|----------|------|
| 语义理解 | 1-5 Hz | LLM / VLM 推理速度 |
| 动作规划 | 30-50 Hz | VLA 动作 chunk |
| 电机控制 | 500-1000 Hz | 稳定性、冲击响应 |

**双系统架构**（Helix System 1 / System 2）是主流解：
- System 2 (VLM): 慢 + 大
- System 1 (Transformer / Diffusion): 快 + 小
- 底层 PD / impedance：硬实时

### 5.2 动作空间选择

全身 VLA 常见方案：
1. **纯关节力矩**：最贴近电机，但高 DoF 难学
2. **关节目标角度 + PD**：Helix 选择，低层 PD 保证稳定
3. **末端位姿 + 底盘速度**：ALOHA 系列，语义清晰但需 IK
4. **Latent Action**：压缩高维，跨具身友好

### 5.3 数据瓶颈

- 全身遥操作比单臂难 5–10 倍（操作员累、同步难）
- Vision Pro + 全身动捕套装（Xsens）是现行方案
- 视频预训练 + 少量真机 fine-tune 是数据经济路线

### 5.4 硬实时安全

- 全身网络推理延迟 > 5 ms 时可能影响稳定性
- **独立底层安全层**：无论 VLA 输出什么，底层 WBC 必须保证不摔
- 2026 主流：VLA 输出目标末端 + 参考姿态 → 底层 QP-WBC 求解并施加约束

## 6. 2026 开放问题

1. **Contact-rich Loco-Manipulation**：推购物车、拖拉家具这类持续接触任务
2. **动态平衡操控**：端着满盘液体走路，晃动约束纳入学习
3. **弱结构地形搬运**：沙地、草地、楼梯上搬箱子
4. **硬件鲁棒性**：VLA 输出异常时底层安全层设计
5. **标准化全身数据格式**：各家数据格式不兼容，行业缺共识

## 参考

- WBC + QP: Sentis & Khatib, 2005
- Convex MPC (Cheetah 3): Di Carlo et al., 2018
- Helix (Figure): figure.ai/news/helix
- RoboCasa: Nasiriany et al., 2024
- AgiBot World: 智元, 2025
- OpenX-Embodiment: 2023
- GR00T N2 (NVIDIA): 2025

---
[返回本目录](README.md) | [上一篇：四足与人形 locomotion](四足与人形locomotion.md) | [下一篇：视觉语言导航 VLN](视觉语言导航VLN.md)

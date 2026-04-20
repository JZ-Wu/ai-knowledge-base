# VLA 与世界模型融合

> 2025-2026 年行业共识：VLA（Vision-Language-Action）与 WM（World Model）不再是两个独立模块，而是**端到端联合设计**的统一架构。本文梳理四条主流融合路线、代表工作与开放问题。

## 一、为什么需要融合

单独看 VLA 与 WM 的局限：

```
VLA 单独:
  + 端到端、数据驱动
  - 样本效率依赖真机数据 (稀缺)
  - 无反事实推理、无 rollout
  - 长 horizon 能力受限 (仅几秒 context)

WM 单独:
  + 样本效率高、可 rollout、可规划
  - 本身不是策略，需要下游策略消费
  - 隐空间难以接常识语言
  - 动作生成的物理正确性无保证
```

**融合后的设计目标**：
1. 用 WM 解决 VLA 的数据瓶颈（增广）
2. 用 WM 给 VLA 提供"想象中 rollout"的能力（规划）
3. 用 VLA 的语言理解能力为 WM 提供条件信号

---

## 二、四条融合路线

### 2.1 路线 A：Dreamer 式耦合（WM 中学策略）

**思路**：先学 WM，再在 WM 中用 RL 学策略，最后少量真机微调。

```
Step 1: 真机数据 D_real = {(s_t, a_t, s_{t+1})}
        → 训练 WM: P_φ(s_{t+1} | s_t, a_t)

Step 2: 在 WM 中 rollout
        对任意 π_θ, 生成想象轨迹 {ŝ_0, a_0, ŝ_1, ...}
        用 Actor-Critic RL 优化 π_θ

Step 3: 少量真机 fine-tune
        校准 WM 与 π_θ 的 distribution shift
```

**代表工作**：
- **Dreamer V3**：通用 model-based RL（见 [世界模型范式](世界模型范式.md#44-dreamer-v32023)）
- **DayDreamer** (2022)：4 足 A1 行走真机学习
- **IRIS** (2023)：Transformer WM 版本

**在具身 VLA 中的应用**：目前仍以 RL 路线为主，与大规模 VLA 融合的公开工作较少。

**适用场景**：
- 任务 reward 可定义（推、抓等原子动作）
- 真机数据稀缺但仿真可用

**不适用**：
- 长 horizon 语言条件任务（WM 中 rollout 数百步误差累积）

---

### 2.2 路线 B：VLA + Cosmos 增广（"更多数据"范式）

**思路**：WM 只作为**离线数据工厂**，VLA 训练范式不变。

```
Step 1: 采集真机数据 D_real (少量, 1K-10K 条)

Step 2: 用 WM 做增广 (Cosmos / 自训 WM)
        对每条轨迹 τ ∈ D_real:
          以原动作序列为条件
          生成 N 条视觉变体 (光照/物体/场景)
          → D_synth (N * |D_real| 条)

Step 3: 用 D_real ∪ D_synth 联合训练 VLA
```

**代表工作**：

| 工作 | 机构 | 说明 |
|------|------|------|
| **GR00T Dreams pipeline** | NVIDIA | Cosmos 增广 → GR00T N1 训练，Apache 2.0 开源 |
| **π₀.5 + Cosmos** (推测) | Physical Intelligence + NVIDIA | 据披露 π₀.5 的"仿真+真实+互联网"三源中包含 WM 合成 |
| **DreamGen** (2024) | 学术 | 用扩散 WM 生成轨迹，增强机器人策略 |

**优势**：
- 不改变 VLA 架构，工程易落地
- 可以快速扩展训练集规模
- 与现有 Open X-Embodiment 等数据生态兼容

**局限**：
- WM 生成的是视觉，动作仍来自原真实数据 → 对"动作多样性"帮助有限
- 视觉真实度与下游性能的因果关系尚未充分量化
- 可能引入 WM 的 artifact（生成瑕疵）作为训练噪声

---

### 2.3 路线 C：VLA 内嵌 future prediction head（联合训练）

**思路**：在 VLA 主干上加一个**未来观测预测头**，动作头与观测预测头**联合训练**。

```
标准 VLA 架构:
  VLM encoder → latent z → Action Head → a_{t:t+T}

内嵌 WM 的架构:
  VLM encoder → latent z ─┬─→ Action Head → a_{t:t+T}
                         │
                         └─→ Future Pred Head → ô_{t+1:t+T}

联合损失:
  L = L_action + λ · L_future
  
  L_action = Flow Matching / MSE / Diffusion loss on a_t
  L_future = MSE / KL on predicted observations
```

**代表工作**：

| 工作 | 年份 | 机构 | 核心贡献 |
|------|------|------|---------|
| **DynaMo** | 2024 | NYU | 在策略训练中加入"隐空间下一状态预测"的辅助损失 |
| **DreamGen** | 2024 | 学术 | VLA 内嵌像素级未来视频生成 |
| **GR00T N2 (据披露)** | 2026 | NVIDIA | 与 Cosmos 联合训练，VLA 内部包含 future head |
| **UnifiedVLA** (预期) | 2026 | 多个团队方向 | 统一动作 + 视频生成的 token 序列 |

**优势**：
- 未来预测任务作为**辅助损失**正则化视觉 encoder，提升样本效率
- 策略具备"预测后果"能力，有潜力用于 MPC
- 不需要独立训练 WM，降低工程复杂度

**局限**：
- 像素预测 head 算力开销大，通常用隐空间预测
- 联合损失权重 λ 需要调参，过大会损害动作精度
- 生成质量通常不如专门的 Cosmos 级 WM

---

### 2.4 路线 D：规划 + 执行分离（分层系统）

**思路**：**WM 做高层规划**（抽象/隐空间），**VLA 做低层执行**（连续动作），两者通过**语言子目标 / 目标图像**对接。

```
┌────────────────────────────────────────────────────┐
│  高层: World Model 规划器 (V-JEPA / Cosmos)          │
│    输入: 当前观测 + 最终目标 (语言 or 图像)            │
│    输出: 子目标序列 g_1, g_2, ..., g_K               │
│           (每个 g_i 是子目标图像 or 语言描述)          │
└──────────────────────┬─────────────────────────────┘
                       ↓ g_i
┌────────────────────────────────────────────────────┐
│  低层: VLA 执行器 (π₀ / GR00T / Helix)               │
│    输入: 当前观测 + 当前子目标 g_i                    │
│    输出: 连续动作 a_{t:t+T}                          │
└────────────────────────────────────────────────────┘
```

**代表工作**：

| 工作 | 说明 |
|------|------|
| **V-JEPA 2 + VLA** | V-JEPA 在隐空间规划子目标 → VLA 执行（Meta 2025 展示方向） |
| **π₀.5 language-in-the-loop** | VLM 主干输出中间自然语言步骤 → Action Expert 执行每步 |
| **Gemini Robotics-ER + 第三方动作后端** | ER 版做具身推理与规划，接任意 VLA/控制后端 |
| **SayCan** (2022) | 早期原型：LLM 规划 + 值函数过滤 + 技能库执行 |

**优势**：
- **模块化**：规划层与执行层独立迭代
- **可解释**：子目标以语言/图像形式可检查
- **利用通用 VLM**：规划层可以直接用 Gemini / GPT 等

**局限**：
- 子目标格式需标准化（语言 or 图像？）
- 层间误差累积
- 实时性挑战（规划器延迟 + 执行器延迟）

---

## 三、四条路线对比

| 路线 | 代表 | 核心价值 | 工程成熟度 | 2026 热度 |
|------|------|---------|-----------|----------|
| A. Dreamer 式 | Dreamer V3, DayDreamer | 样本效率 | 高（RL 圈） | 中（VLA 圈少用） |
| B. Cosmos 增广 | GR00T Dreams, π₀.5 | 数据规模 | 高 | **高** (工业标配) |
| C. 内嵌 future head | DynaMo, DreamGen | 联合优化 | 中 | **高** (2026 研究前沿) |
| D. 规划+执行分离 | V-JEPA 2 + VLA, π₀.5 | 模块化 + 长 horizon | 中 | **高** (产业思路) |

---

## 四、代表性 pipeline 详解

### 4.1 NVIDIA GR00T Dreams Pipeline（路线 B）

```
┌─────────────────────────────────────────────────────┐
│  输入: Fourier GR-1 / Unitree H1 少量真机数据         │
│  (~1K-10K 条遥操作轨迹)                              │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  Cosmos Transfer: 真实→仿真 / 仿真→真实              │
│    让仿真数据视觉上接近真实                          │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  Cosmos Predict: 生成视觉变体                        │
│    条件: 原始动作 + 新场景描述                       │
│    输出: 10-100 倍扩增的视频轨迹                     │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  Cosmos Reason: 过滤低质量样本                       │
│    VLM 打分判断"视频是否符合动作"                    │
└─────────────┬───────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────────────────┐
│  GR00T N1/N2 训练:                                  │
│    数据: 真实 + Cosmos 增广 + Open X-Embodiment     │
│    目标: 跨本体 VLA 基础模型                         │
└─────────────────────────────────────────────────────┘
```

**价值链**：NVIDIA 以 Cosmos 为护城河，绑定 GR00T 硬件合作伙伴（Fourier、Unitree、Neura），形成"WM + VLA + 本体"三位一体的生态。

### 4.2 DynaMo 式辅助损失（路线 C）

```python
# 伪代码
def train_step(obs, expert_action):
    # 共享 VLM encoder
    z = vlm_encoder(obs)  # latent
    
    # 动作头
    pred_action = action_head(z)
    L_action = flow_matching_loss(pred_action, expert_action)
    
    # 未来隐空间预测头 (DynaMo 风格)
    z_next_pred = future_head(z, expert_action[0])
    with torch.no_grad():
        z_next_target = vlm_encoder(next_obs)  # EMA encoder
    L_future = MSE(z_next_pred, z_next_target.detach())
    
    # 联合训练
    L = L_action + 0.1 * L_future
    return L
```

**关键设计**：
- 未来预测在**隐空间**（避免像素生成的算力）
- 目标 encoder 用 **EMA**（类似 BYOL / MoCo，防 collapse）
- 辅助损失权重通常 0.1-0.5

### 4.3 π₀.5 的"language-in-the-loop"（路线 D）

π₀.5 的创新在于 VLM 主干可以**输出中间自然语言规划步骤**：

```
用户指令: "清理厨房"
    ↓
π₀.5 VLM 分解:
    "1. 识别桌上物品"
    "2. 把杯子放入水槽"
    "3. 用抹布擦桌子"
    ...
    ↓
每步作为当前子目标输入 Action Expert
    ↓
Action Expert 执行当前子目标
    ↓
执行完成后 VLM 重新评估，进入下一子目标
```

这可以被看作**"VLM 内建轻量 WM + 自我规划"**，不需要外部 Cosmos/Genie。

---

## 五、开放问题

### 5.1 动作维度生成的物理正确性

**核心难题**：视频 WM（Cosmos / Genie）可以生成"看起来真实"的视频，但：
- 生成的机械臂轨迹满足关节限位、力矩约束吗？
- 生成的物体运动满足刚体动力学吗？
- 生成的抓取满足力封闭吗？

```
当前现状:
  - WM 生成视觉 → 增广数据
  - 动作仍来自真实 / 仿真
  - WM 不直接生成动作序列
  
未来方向:
  - 物理约束 WM (physics-informed)
  - 联合动作-视频生成模型
  - 可微分仿真 + WM 混合
```

### 5.2 长 horizon 一致性

```
Cosmos 1.0:   ~30 秒
Cosmos 2.0:   ~分钟级 (据披露)
Genie 3:      2-3 分钟
目标:         小时级 (家庭完整任务)

瓶颈:
  - 物体持久性 (离开视野后再回来形状变了)
  - 全局一致性 (房间布局随时间漂移)
  - 算力 O(T²) 对 Transformer WM 不友好
```

### 5.3 评测标准缺失

```
WM 本身的评测:
  - FVD (Fréchet Video Distance)
  - 用户偏好研究
  → 但对具身下游任务的预测力弱

WM 对下游 VLA 的增益评测:
  - 比较"仅真实 vs 真实+WM 增广"下的 VLA 性能
  - 需要大规模真机评测 (成本高)
  - 尚无公认 benchmark
```

### 5.4 实时性

```
Cosmos 14B 生成 30 秒视频: ~分钟级算力
→ 无法在机器人推理回路中使用
→ 当前仅用于离线数据增广或慢速规划

加速路线:
  - 蒸馏为小模型 (1B 级)
  - 隐空间 WM 替代像素 (V-JEPA 方向)
  - 硬件加速 (Jetson Thor, 专用推理芯片)
```

### 5.5 数据循环与持续学习

```
理想循环:
  机器人部署 → 采集新数据 → 更新 WM → 生成新场景增广 → 微调 VLA
  
当前障碍:
  - WM 更新成本高 (数周训练)
  - 灾难性遗忘
  - 新老数据混合比例调参
```

---

## 六、选型速查

| 场景 | 推荐路线 | 代表工具 |
|------|---------|---------|
| 工业快速扩数据 | **B. Cosmos 增广** | GR00T Dreams |
| 学术 + 算力有限 | **C. 内嵌 future head** | DynaMo, 自实现 |
| 长 horizon 家庭任务 | **D. 规划+执行分离** | V-JEPA 2 + 自研 VLA |
| 小任务 RL 样本效率 | **A. Dreamer 式** | Dreamer V3 + 自训策略 |
| 多路线混合 | 生产级大团队 | π₀.5（B+D）、GR00T N2（B+C 据披露） |

---

## 七、关键参考文献

- **GR00T N1**: NVIDIA, "GR00T N1: An Open Foundation Model for Humanoid Robots", 2025.
- **Cosmos**: NVIDIA, "Cosmos World Foundation Model Platform", 2025.01.
- **V-JEPA 2**: Meta FAIR, 2025.06.
- **π₀.5**: Physical Intelligence, 2025.
- **DynaMo**: Cui et al. (NYU), "DynaMo: In-Domain Dynamics Pretraining for Visuo-Motor Control", 2024.
- **DreamGen**: 2024 学术工作.
- **Dreamer V3**: Hafner et al., 2023.

---

[返回目录](README.md) | [上一篇：主流世界模型 2026](主流世界模型2026.md)

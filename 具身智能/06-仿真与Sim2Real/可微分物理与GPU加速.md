# 可微分物理与 GPU 加速

> 现代具身仿真的两大加速引擎：大规模 GPU 并行 + 可微分物理。本文梳理 MJX、Genesis、Brax、DiffTaichi 等可微分仿真器，以及 Isaac Gym/Lab、ManiSkill 3 Warp 后端的并行能力，并覆盖渲染路线。

## 一、两条加速路径

### 路径 A：GPU 大规模并行（工程主流）

- 思路：单 GPU 上同时运行数千个仿真环境
- 适合：传统 RL（PPO、SAC）、locomotion、操控
- 代表：Isaac Gym/Lab、ManiSkill 3、MJX、Genesis

### 路径 B：可微分物理（研究前沿）

- 思路：让物理引擎本身可微，梯度直接反传到 action / policy 参数
- 适合：梯度基 RL、系统辨识、轨迹优化、软体控制
- 代表：MJX、Brax、Genesis（可微模式）、DiffTaichi

---

## 二、可微分仿真器详解

### 2.1 MJX（JAX-MuJoCo）

2023 年 DeepMind 发布，将 MuJoCo 移植到 JAX：

```python
import mujoco
from mujoco import mjx
import jax
import jax.numpy as jnp

model = mujoco.MjModel.from_xml_path("humanoid.xml")
mjx_model = mjx.put_model(model)

def loss(actions, init_state):
    state = init_state
    for a in actions:
        state = mjx.step(mjx_model, state.replace(ctrl=a))
    return state.qpos[0]  # 机器人向前走的距离

# 直接对 action 序列求梯度
grad_actions = jax.grad(loss)(actions, init_state)
```

**特点**：
- JIT + vmap：天然 GPU 并行（数千环境）
- 自动微分：`jax.grad` 直接拿到物理梯度
- 与原生 MuJoCo 精度一致（但有少量算子简化）
- 2024 年起成为学术界 RL 研究主流

### 2.2 Brax（Google）

纯 JAX 实现的物理引擎：

- 从零开始为可微分设计
- GPU/TPU 原生支持
- 内置 RL 算法：PPO、SAC、ES、APG（Analytical Policy Gradient）
- 相对简化的接触模型（早期版本），2023 后物理精度显著提升

```python
import brax
from brax.envs import create
env = create("humanoid", batch_size=4096, backend="spring")
# backend: spring / positional / generalized
# spring: 快速，精度中
# generalized: 精度接近 MuJoCo
```

### 2.3 Genesis（可微模式）

2024 年开源，支持多物理引擎并提供可微分路径：

- 刚体（PhysX）、MPM（软体）、SPH（流体）、FEM、PBD
- 基于 Taichi 编译，GPU/CPU 均可运行
- 部分物理路径可微（MPM 可微成熟，刚体接触可微仍在完善）

### 2.4 DiffTaichi（可微分编程语言）

面向物理仿真的可微分 DSL：

- 由 Taichi 语言提供
- 适合研究：柔体、布料、流体控制
- 非常灵活，但工程门槛高
- 代表工作：DiffSim（可微分布料）、PlasticineLab

### 2.5 其他可微分尝试

- **DeepMind DM Control + JAX**
- **Nimble (Gradient-based physics)**
- **Warp (NVIDIA, 部分可微)**：ManiSkill 3 后端
- **PyBullet 非官方可微 fork**

---

## 三、梯度基 RL (Gradient-based Policy Optimization)

### 3.1 APG：Analytical Policy Gradient

通过物理梯度直接优化策略：

```
传统 RL (PPO):
  采样轨迹 → 用 reward 估计策略梯度 (无偏但高方差)

APG (可微分物理):
  采样轨迹 → 用链式法则直接反传到策略参数 (低方差但有梯度噪声)
```

**优势**：
- 样本效率高 10-100x
- 收敛快

**劣势**：
- 接触非光滑 → 梯度噪声大
- 长 horizon 梯度爆炸/消失
- 对局部极小敏感

### 3.2 混合方法：SHAC / PODS

- **SHAC** (Short-Horizon Actor-Critic)：短 horizon 用梯度，长 horizon 用 critic
- **PODS** (Policy Optimization via Differentiable Simulation)
- 在 Humanoid、Ant、软体操控等任务上比 PPO 快一个数量级

### 3.3 适用与不适用

| 任务类型 | 可微分 RL 效果 |
|---------|------------|
| 连续控制、柔顺 | 好（SHAC 优于 PPO） |
| 软体操作、装配 | 非常好 |
| 接触丰富（抓取、踢球） | 一般（接触梯度噪声大） |
| 长 horizon（>1000 步） | 差（梯度不稳定） |
| 多模态策略 | 差（容易落入局部极小） |

---

## 四、GPU 并行详解

### 4.1 Isaac Gym / Isaac Lab

NVIDIA 的 GPU 并行标杆，支持单卡数千环境：

```python
# Isaac Gym 伪代码
env = gym.make("Humanoid-v0", num_envs=4096)

obs = env.reset()  # shape (4096, obs_dim)
for step in range(1_000_000):
    actions = policy(obs)   # (4096, action_dim)
    obs, reward, done, info = env.step(actions)
    # 全部 GPU 张量，无 CPU 搬运
```

**加速比**：
- 与单核 MuJoCo 相比：~1000-4000x
- 与 CPU 多核并行（如 SubprocVecEnv 32 env）相比：~100x
- 实际吞吐量：Humanoid 约 200K-1M steps/sec（H100）

**Isaac Gym → Isaac Lab**：
- Isaac Gym 于 2023 年弃用（legacy）
- Isaac Lab 整合到 Isaac Sim，获得 RTX 渲染 + 更完整 API

### 4.2 ManiSkill 3 Warp 后端

NVIDIA Warp 是 GPU 编程框架，ManiSkill 3 用 Warp + PhysX 实现 GPU 并行：

- 单 H100 可跑 1024-4096 操控环境
- 关键：视觉 RL 首次可行（之前渲染是瓶颈）
- 光追 / 光栅化可选，tradeoff 速度 vs 质量

### 4.3 MJX GPU 并行

- JAX pmap + vmap 实现数千环境并行
- 速度：Humanoid 约 50K-200K steps/sec（A100）
- 精度与原生 MuJoCo 高度一致

### 4.4 Genesis

- 声称 10-80x 加速 Isaac Lab（特定场景）
- 2026 社区评估：软体/流体确实显著快；刚体操控场景持平或略快

---

## 五、渲染：光栅化 vs 光线追踪 vs 神经渲染

仿真中"看见"依赖渲染，渲染决定视觉 Sim-to-Real gap。

### 5.1 光栅化（Rasterization）

- 传统实时渲染（OpenGL/Vulkan/DirectX）
- 速度快，单 GPU 数千 FPS
- 视觉质量中等：阴影、反射、折射需要 hack
- 代表：Isaac Gym 早期、PyBullet、Habitat 默认

### 5.2 光线追踪（Ray Tracing）

- NVIDIA RTX 硬件加速
- 真实光照、反射、全局光照
- 速度较慢（相对光栅化 5-10x 慢，但 RTX 4090+ 已实时）
- 视觉质量接近照片级
- 代表：Isaac Sim RTX、SAPIEN 光追模式

**对 Sim-to-Real 的价值**：
- 减少光照/反射带来的域差异
- 纹理 + 光影更接近真实相机
- 但：视觉策略仍需域随机化/世界模型补充

### 5.3 神经渲染（Neural Rendering）

- NeRF、3D Gaussian Splatting、学习式渲染
- 从真实扫描数据重建场景
- 视觉"真实"，但几何 / 可交互性需要额外处理
- 代表：Real-to-Sim 系列（如 Robo-GS、URDFormer）
- 详见 [Sim-to-Real方法.md](Sim-to-Real方法.md) § 2.5

### 5.4 渲染路线对比

| 渲染方式 | 速度 | 视觉保真 | 几何正确 | 可控性 |
|---------|------|--------|---------|-------|
| 光栅化 | 极快 | 中 | 高 | 高 |
| 光追 (RTX) | 快 | 高 | 高 | 高 |
| NeRF/3DGS | 中 | 极高 | 依赖重建 | 中 |
| 扩散/Cosmos | 慢 | 极高 | 弱 | 中（条件生成） |

---

## 六、GPU 加速的关键工程技巧

### 6.1 避免 CPU-GPU 搬运

```python
# 差的做法：每步 CPU-GPU 拷贝
obs_cpu = env.step(actions_cpu)  # CPU tensor
obs_gpu = to_gpu(obs_cpu)
actions_cpu = policy(obs_gpu).cpu()  # 又拷贝回 CPU

# 好的做法：全程 GPU tensor
obs_gpu = env.step(actions_gpu)  # GPU tensor in/out
actions_gpu = policy(obs_gpu)    # 全程 GPU
```

Isaac Lab / MJX / ManiSkill 3 默认全 GPU，无需手动管理。

### 6.2 批量化观测处理

- 观测归一化、图像预处理全部向量化
- 避免 Python 层 for 循环

### 6.3 异步渲染

- 物理 step 与渲染解耦
- 物理 1000 Hz，渲染 30-60 Hz 即可
- 节省渲染算力

### 6.4 编译与 JIT

- JAX (`jit`, `vmap`) 自动融合操作
- Warp 通过 Python kernel 编译为 CUDA
- PyTorch torch.compile / functorch 也可用

---

## 七、典型研究实践

### 7.1 Locomotion (Isaac Lab)

```
单 H100
  4096 并行环境 × 1000 steps = ~4M steps/分钟
  PPO 训练 ANYmal 平地行走 ~15 分钟
  训练崎岖地形 Parkour ~4 小时
```

### 7.2 可微分操控 (MJX)

```
软体操作（捏橡皮泥）:
  MJX + SHAC (梯度基 RL)
  100 并行，收敛 10 分钟
  相同任务 PPO 需要 2+ 小时
```

### 7.3 视觉操控 (ManiSkill 3)

```
单 H100
  1024 并行视觉环境（128×128 RGB）
  Diffusion Policy 训练 ~8 小时
  对标 CPU 多核（32 env）加速 ~30x
```

---

## 八、前沿进展 (2025-2026)

1. **MJX 的物理精度补齐**：接触/摩擦模型升级到接近原生 MuJoCo
2. **Genesis 可微刚体**：解决接触梯度噪声，开始进入实用
3. **Warp + Isaac Sim 融合**：NVIDIA 在整合 PhysX / Warp，让 Isaac 也获得可微能力
4. **神经物理引擎**：部分物理由神经网络学习（如接触、形变），端到端可微
5. **端到端可微渲染 + 物理**：用于 Real-to-Sim 参数辨识
6. **编译器级优化**：TorchDynamo、JAX XLA、Triton 不断提升吞吐量

---

## 九、选型建议

| 需求 | 推荐 |
|------|-----|
| 大规模 locomotion RL | Isaac Lab / MJX |
| 可微分 RL 研究 | MJX / Brax |
| 视觉操控 | ManiSkill 3（Warp）/ Isaac Lab |
| 软体 / 流体 / 可微 | Genesis / DiffTaichi |
| 教学 / 轻量 | PyBullet / 原生 MuJoCo |
| Real-to-Sim | NeRF/3DGS + MuJoCo/SAPIEN |

---

## 参考资料

- MJX: <https://mujoco.readthedocs.io/en/stable/mjx.html>
- Brax: <https://github.com/google/brax>
- Genesis: <https://genesis-embodied-ai.github.io/>
- DiffTaichi: Hu et al., "DiffTaichi", ICLR 2020
- SHAC: Xu et al., "Accelerated Policy Learning with Parallel Differentiable Simulation", ICLR 2022
- Isaac Gym Paper: Makoviychuk et al., NeurIPS 2021

---

[返回本专题](README.md) · 上一篇：[Sim-to-Real 方法](Sim-to-Real方法.md) · 相关：[主流仿真器对比](主流仿真器对比.md)

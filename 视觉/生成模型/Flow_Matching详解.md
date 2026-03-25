# Flow Matching 详解

> 参考：[Lipman et al., 2022 - Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | [Liu et al., 2022 - Rectified Flow](https://arxiv.org/abs/2209.03003)

---

## 1. 动机：DDPM 的复杂性

DDPM 的前向过程是**离散的马尔可夫链**（$T=1000$ 步），训练目标从变分下界（VLB）推导而来，涉及一堆噪声调度系数 $\beta_t$、$\alpha_t$、$\bar{\alpha}_t$。虽然 SDE 框架将其统一为连续时间，但公式复杂度并没有降低。

Flow Matching 提出了一个更简洁的思路：**不走扩散的路，直接学一个把噪声变成数据的速度场**。

---

## 2. 核心思想：ODE 驱动的生成

Flow Matching 的出发点是**连续正规化流**（Continuous Normalizing Flow, CNF）：用一个 ODE 把简单分布（噪声）变换成复杂分布（数据）。

$$\frac{dx_t}{dt} = v_\theta(x_t, t), \quad t \in [0, 1]$$

- $t=0$：$x_0 \sim \mathcal{N}(0, I)$（纯噪声）
- $t=1$：$x_1 \sim p_{\text{data}}(x)$（真实数据）
- $v_\theta(x_t, t)$：速度场（velocity field），神经网络学出来的

```
DDPM 的思路:   数据 → 加噪 → 噪声 → 逐步去噪 → 数据
              (前向是加噪，反向是去噪，需要 SDE/马尔可夫链)

Flow Matching: 噪声 ────── ODE ──────→ 数据
              (一条连续路径，由速度场 v_θ 驱动)
```

> **注意方向约定**：Flow Matching 中 $t=0$ 是噪声，$t=1$ 是数据，和 DDPM 相反。

---

## 3. 条件 Flow Matching（CFM）

### 问题

直接学全局速度场 $v_\theta(x_t, t)$ 需要知道数据分布的边际概率路径 $p_t(x)$，不可行。

### 解法：条件路径

关键技巧是**对每个训练样本单独定义一条路径**，然后用这些条件路径来训练。

给定一个训练样本 $x_1$（数据）和一个噪声样本 $x_0 \sim \mathcal{N}(0, I)$，定义**最简单的插值路径**：

$$\boxed{x_t = (1 - t) \, x_0 + t \, x_1}$$

这条路径的速度（对 $t$ 求导）是：

$$u_t = \frac{dx_t}{dt} = x_1 - x_0$$

### 训练目标

让网络预测的速度场匹配这个条件速度：

$$\boxed{L_{\text{CFM}} = \mathbb{E}_{t, x_0, x_1} \left[ \| v_\theta(x_t, t) - (x_1 - x_0) \|^2 \right]}$$

### 训练算法

```
训练算法（和 DDPM 一样简单）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
repeat:
    x₁ ~ 训练数据                          # 真实图片
    x₀ ~ N(0, I)                           # 纯噪声
    t  ~ Uniform(0, 1)                     # 随机时间
    x_t = (1-t)·x₀ + t·x₁                 # 线性插值
    梯度下降: ||v_θ(x_t, t) - (x₁ - x₀)||² # 预测速度
until 收敛

采样算法（ODE 积分）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x₀ ~ N(0, I)                              # 从噪声出发
用 ODE solver (如 Euler) 积分:
  for i = 0, 1, ..., N-1:
    t_i = i/N
    x_{t_{i+1}} = x_{t_i} + (1/N) · v_θ(x_{t_i}, t_i)
return x₁                                  # 生成的图片
```

### 为什么条件路径能学到全局速度场？

直觉：每个训练样本定义了一条从噪声到数据的路径。所有这些条件路径的**加权平均**就是全局概率路径。网络 $v_\theta$ 在所有条件路径上回归，自然学到了全局速度场。

数学上可以证明：条件 Flow Matching 的梯度和全局 Flow Matching 的梯度**完全相同**。

---

## 4. 与 DDPM 的详细对比

| | DDPM | Flow Matching |
|---|---|---|
| **前向过程** | 逐步加噪 $q(x_t\|x_{t-1})$，涉及 $\beta_t, \bar{\alpha}_t$ | 线性插值 $x_t = (1-t)x_0 + tx_1$，无额外系数 |
| **训练目标** | 预测**噪声** $\epsilon$ | 预测**速度** $v = x_1 - x_0$ |
| **采样过程** | 反向 SDE（随机）或概率流 ODE | 正向 ODE（确定性） |
| **时间范围** | $t \in \{0, ..., T\}$（离散，$T=1000$） | $t \in [0, 1]$（连续） |
| **推导复杂度** | 需要 VLB、KL 散度、重参数化 | 直接回归速度场，几乎不需要推导 |
| **噪声调度** | 需要精心设计 $\beta_t$ schedule | 隐含在插值路径中，无需额外设计 |

### 本质联系

DDPM 的概率流 ODE 和 Flow Matching 的 ODE 在数学上可以互相转换。具体来说，给定 DDPM 的噪声预测 $\epsilon_\theta$，可以转换为速度预测 $v_\theta$：

$$v_\theta(x_t, t) = \frac{x_1^{\text{pred}} - x_0^{\text{pred}}}{dt} = \frac{\sqrt{\bar{\alpha}_t} \, \epsilon_\theta - (1 - \sqrt{\bar{\alpha}_t}) \, x_t}{\text{...}}$$

Flow Matching 的优势不在于数学上更强，而是**公式简洁、实现简单、不需要噪声调度的各种系数**。

---

## 5. 高斯路径的一般形式

线性插值 $x_t = (1-t)x_0 + tx_1$ 只是最简单的选择。更一般地，可以定义：

$$x_t = \alpha_t \, x_1 + \sigma_t \, x_0$$

其中 $\alpha_t$ 和 $\sigma_t$ 满足边界条件 $\alpha_0 = 0, \alpha_1 = 1, \sigma_0 = 1, \sigma_1 = 0$。

| 路径选择 | $\alpha_t$ | $\sigma_t$ | 等价于 |
|---------|-----------|-----------|--------|
| 线性插值 | $t$ | $1-t$ | 标准 Flow Matching |
| VP 路径 | $\sqrt{\bar{\alpha}_t}$ | $\sqrt{1-\bar{\alpha}_t}$ | DDPM 的概率流 ODE |

---

## 6. Rectified Flow（直流）

Rectified Flow（Liu et al., 2022）进一步优化路径的"直"度。

### 问题：路径交叉

线性插值路径虽然简单，但不同样本对 $(x_0^{(i)}, x_1^{(i)})$ 的路径在 $x_t$ 空间中会**交叉**，导致速度场在交叉点不一致（同一个 $x_t$ 对应多个不同的速度方向）。

```
样本 A 的路径: x₀ᴬ ──────→ x₁ᴬ
                      ✗ 交叉
样本 B 的路径: x₀ᴮ ──────→ x₁ᴮ

在交叉点，v_θ 需要同时指向 x₁ᴬ 和 x₁ᴮ → 只能取平均 → 路径弯曲
```

### Reflow：让路径更直

**Reflow 过程**：用已训练好的模型生成新的 $(x_0, x_1)$ 配对，重新训练。

```
第 1 轮: 随机配对 (x₀, x₁)
         训练 v_θ
         路径弯曲交叉 → 需要多步 ODE
  ↓
  用 v_θ 从 x₀ 生成 x̂₁ → 新配对 (x₀, x̂₁)

第 2 轮: 用新配对训练
         路径更直 → 更少步数
  ↓ reflow

第 3 轮: 路径接近直线 → 1-2 步就够
```

### 为什么 reflow 能让路径更直？

第一轮训练后，ODE 将每个 $x_0$ 映射到特定的 $x_1$。用这个映射关系重新配对后，新的 $(x_0, x_1)$ 对之间的路径**不再交叉**（因为 ODE 是单射），速度场一致性提高，路径自然更直。

---

## 7. 在现代模型中的应用

Flow Matching 已经成为最新生成模型的**主流训练范式**：

| 模型 | 年份 | 架构 | 说明 |
|------|------|------|------|
| **Stable Diffusion 3** | 2024 | MM-DiT + Flow Matching | 用 MM-DiT 替代 U-Net，Flow Matching 替代 DDPM |
| **FLUX** | 2024 | DiT + Flow Matching | Black Forest Labs（原 SD 团队），当前开源 SOTA |
| **Sora** | 2025 | DiT + Flow Matching | OpenAI 视频生成 |

```
架构演进:
  SD 1/2:  U-Net + DDPM loss (预测噪声 ε)
  SD 3:    DiT  + Flow Matching loss (预测速度 v)

两个独立的趋势:
  骨干网络:  U-Net → DiT (Transformer)
  训练范式:  DDPM → Flow Matching
```

### SD3 的 MM-DiT

SD3 使用 **MM-DiT**（Multi-Modal DiT）：文本和图像 token 在同一个 Transformer 中做联合注意力，替代了 U-Net + Cross-Attention 的设计。

---

**相关文档**：
- [Diffusion 详解](Diffusion详解.md) — DDPM 基础，Flow Matching 的前身
- [Stable Diffusion 详解](Stable_Diffusion详解.md) — SD3 使用 Flow Matching
- [采样加速与条件生成](采样加速与条件生成.md) — 其他加速方法

[返回上级](README.md) | [返回视觉](../README.md) | [返回总目录](../../README.md)

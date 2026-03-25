# Diffusion 详解：DDPM 与 Score Matching

> 参考：[Lilian Weng - What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) | [Ho et al., 2020 - Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | [Yang Song - Score-Based Generative Models](https://yang-song.net/blog/2021/score/)

---

## 1. 核心思想

扩散模型的思路极其简洁：

```
前向过程：逐步往图片上加噪声，直到变成纯高斯噪声
反向过程：学一个神经网络，逐步从噪声中去噪，还原出图片

  x₀ (清晰图片)                                    x_T (纯噪声)
    →  加噪  →  加噪  →  加噪  →  ......  →  加噪  →
    ←  去噪  ←  去噪  ←  去噪  ←  ......  ←  去噪  ←
         ↑        ↑        ↑                    ↑
     神经网络  神经网络  神经网络             神经网络
     (共享参数，额外输入时间步 t)
```

### 相比 VAE 的核心优势

```
VAE:
  真实后验 p_θ(z|x) → 算不了 → 用 Encoder 近似 → 有误差（approximate inference）

Diffusion:
  反向后验 q(x_{t-1}|x_t, x_0) → 精确高斯（闭式解） → 无近似误差
  训练 = 让 p_θ(x_{t-1}|x_t) 逼近这个精确分布
       = 两个高斯的 KL → 闭式解 → 最终简化为预测噪声
```

Diffusion 生成质量高的原因：
1. **不需要近似后验**——监督信号是精确的
2. **逐步生成**——每步只去一点噪声，建模难度低
3. **不存在 z 空间压缩**——没有信息瓶颈，不会因为维度不够而模糊

---

## 2. 前向过程（加噪）

给定干净图片 $x_0$，在每一步 $t$ 加一点高斯噪声：

$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \, \beta_t I)$$

其中 $\beta_t \in (0, 1)$ 是噪声调度（noise schedule），控制每步加多少噪声。

### 关键性质：可以一步到位

定义 $\alpha_t = 1 - \beta_t$，$\bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$，则：

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, \, (1 - \bar{\alpha}_t) I)$$

即：

$$\boxed{x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)}$$

**直觉**：$\bar{\alpha}_t$ 从接近 1 逐渐衰减到接近 0。$t$ 小时信号占主导；$t$ 大时噪声占主导。

```
t=0:     x₀ (原图)           ᾱ₀ ≈ 1.0
t=250:   x₂₅₀ (略有噪点)     ᾱ₂₅₀ ≈ 0.7
t=500:   x₅₀₀ (很模糊)       ᾱ₅₀₀ ≈ 0.3
t=750:   x₇₅₀ (几乎是噪声)   ᾱ₇₅₀ ≈ 0.05
t=1000:  x₁₀₀₀ (纯噪声)      ᾱ₁₀₀₀ ≈ 0.0
```

### 一步到位的推导

$x_t = \sqrt{\alpha_t} \, x_{t-1} + \sqrt{1-\alpha_t} \, \epsilon_t$，递归展开：

$$x_t = \sqrt{\alpha_t \alpha_{t-1}} \, x_{t-2} + \sqrt{1 - \alpha_t \alpha_{t-1}} \, \bar{\epsilon}$$

这里用到了高斯的叠加性质：$\mathcal{N}(0, \sigma_1^2) + \mathcal{N}(0, \sigma_2^2) = \mathcal{N}(0, \sigma_1^2 + \sigma_2^2)$。一路展开到 $x_0$，得到 $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$。

---

## 3. 反向过程（去噪）

如果我们知道 $q(x_{t-1} | x_t)$，就能从噪声逐步恢复图片。但它需要知道整个数据分布，不可行。

**关键观察**：当我们额外知道 $x_0$ 时，$q(x_{t-1} | x_t, x_0)$ 是**精确可算的高斯分布**：

$$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t, \tilde{\beta}_t I)$$

其中：

$$\tilde{\mu}_t = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

训练时 $x_0$ 已知（就是训练数据），所以可以用这个精确后验来指导学习。

### 推导思路

利用贝叶斯公式 $q(x_{t-1}|x_t, x_0) \propto q(x_t|x_{t-1}) \cdot q(x_{t-1}|x_0)$，两个高斯相乘还是高斯，配方即可得到 $\tilde{\mu}_t$ 和 $\tilde{\beta}_t$。

---

## 4. 训练目标推导

用变分推断（和 VAE 思路相同），最大化数据似然的下界：

$$\log p_\theta(x_0) \geq -L_{\text{VLB}}$$

VLB 拆解为每一步的 KL 散度：

$$L_{\text{VLB}} = \underbrace{D_{KL}(q(x_T|x_0) \| p(x_T))}_{L_T \text{ (常数)}} + \sum_{t=2}^{T} \underbrace{D_{KL}(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))}_{L_{t-1}} + \underbrace{(-\log p_\theta(x_0|x_1))}_{L_0}$$

每个 $L_{t-1}$ 都是两个高斯分布的 KL 散度（闭式解），归结为让模型预测的均值 $\mu_\theta(x_t, t)$ 逼近真实的 $\tilde{\mu}_t$。

---

## 5. 从预测均值到预测噪声

将 $x_0 = \frac{1}{\sqrt{\bar{\alpha}_t}}(x_t - \sqrt{1-\bar{\alpha}_t}\epsilon)$ 代入 $\tilde{\mu}_t$：

$$\tilde{\mu}_t = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon \right)$$

所以模型只要预测加入的噪声 $\epsilon$，就能算出均值。令 $\epsilon_\theta(x_t, t)$ 为噪声预测网络：

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

### 三种等价的预测目标

| 预测目标 | 训练 loss | 说明 |
|---------|----------|------|
| 预测噪声 $\epsilon_\theta$ | $\|\epsilon - \epsilon_\theta(x_t, t)\|^2$ | DDPM 原文，最常用 |
| 预测 $x_0$ | $\|x_0 - x_{0,\theta}(x_t, t)\|^2$ | 等价，某些场景更直观 |
| 预测速度 $v_\theta$ | $\|v - v_\theta(x_t, t)\|^2$ | Flow Matching 使用 |

三者可以互相转换：$\epsilon = \frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{\sqrt{1-\bar{\alpha}_t}}$

---

## 6. 简化损失函数（DDPM 的核心结论）

Ho et al. (2020) 发现，去掉 $L_{t-1}$ 中与 $t$ 相关的权重系数后，训练效果反而更好：

$$\boxed{L_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]}$$

**DDPM 的训练目标**——简洁到令人惊讶：**随机选一个时间步，加噪声，让网络预测加了什么噪声**。

---

## 7. 训练与采样算法

```
训练算法（极其简单）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
repeat:
    x₀ ~ 训练数据
    t  ~ Uniform({1, ..., T})
    ε  ~ N(0, I)
    x_t = √ᾱ_t · x₀ + √(1-ᾱ_t) · ε        # 一步加噪
    梯度下降: ||ε - ε_θ(x_t, t)||²           # 预测噪声
until 收敛

采样算法（逐步去噪）：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
x_T ~ N(0, I)                                # 从纯噪声开始
for t = T, T-1, ..., 1:
    z ~ N(0, I)  if t > 1, else z = 0
    x_{t-1} = 1/√α_t · (x_t - β_t/√(1-ᾱ_t) · ε_θ(x_t, t)) + √β̃_t · z
return x₀
```

---

## 8. 噪声调度（Noise Schedule）

| 类型 | 公式 | 特点 |
|------|------|------|
| Linear（DDPM） | $\beta_t$ 从 $10^{-4}$ 线性增到 $0.02$ | 简单，但末端信噪比下降太快 |
| Cosine（Improved DDPM） | $\bar{\alpha}_t = \frac{f(t)}{f(0)}$，$f(t) = \cos\left(\frac{t/T + s}{1+s} \cdot \frac{\pi}{2}\right)^2$ | 更平滑，图像质量更好 |

```
Linear schedule 的问题:
  t=800 时 ᾱ_t 已经接近 0 → 后 200 步几乎没有信号，浪费算力

Cosine schedule 的优势:
  ᾱ_t 下降更均匀 → 每个时间步都有学习价值
```

---

## 9. 与 Score Matching 的统一

### Score function

**Score function**：数据分布对数密度的梯度 $\nabla_x \log p(x)$，指向数据密度增大的方向。

直觉：score 告诉你"从当前位置往哪个方向走，能到更高概率的区域"。

### DDPM 噪声预测 = Score 估计

DDPM 的噪声预测 $\epsilon_\theta$ 和 score 有直接关系：

$$\nabla_{x_t} \log q(x_t) = -\frac{\epsilon}{\sqrt{1 - \bar{\alpha}_t}} \quad \Rightarrow \quad s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1 - \bar{\alpha}_t}}$$

**预测噪声 = 预测 score 的负方向（缩放后）**。DDPM 和 Score-Based Models（NCSN）本质上在做同一件事。

### SDE 统一框架

Yang Song 等人用 **SDE（随机微分方程）** 统一了两个框架：

$$\text{前向 SDE: } dx = f(x,t)dt + g(t)dw$$

$$\text{反向 SDE: } dx = [f(x,t) - g^2(t)\nabla_x \log p_t(x)]dt + g(t)d\bar{w}$$

DDPM 和 NCSN 只是同一个连续时间 SDE 的不同离散化方式。

等价的**概率流 ODE**（确定性版本）：

$$dx = \left[f(x,t) - \frac{1}{2}g^2(t)\nabla_x \log p_t(x)\right]dt$$

这建立了扩散模型与 Normalizing Flows 之间的联系，可以精确计算 log-likelihood。

```
DDPM         → VP-SDE 的离散化
NCSN (SMLD)  → VE-SDE 的离散化
概率流 ODE    → 确定性版本，可以精确算 log p(x)

VP = Variance Preserving
VE = Variance Exploding
```

---

## 10. 网络架构：U-Net

DDPM 使用的去噪网络是 **U-Net**，带有以下关键设计：

```
输入: x_t (带噪图片) + t (时间步编码)

U-Net 结构:
  下采样 → 中间层 → 上采样
  每层有 skip connection

时间步编码:
  t → sinusoidal embedding → MLP → 注入每层（类似 Transformer 的位置编码）

自注意力:
  在低分辨率层加入 Self-Attention → 捕获全局信息
```

后续 DiT（Diffusion Transformer）用纯 Transformer 替代 U-Net，成为 SD3/FLUX/Sora 的主流架构。

---

**相关文档**：
- [VAE 详解](VAE详解.md) — 变分推断基础，ELBO 推导
- [采样加速与条件生成](采样加速与条件生成.md) — DDIM、CFG、Consistency Models
- [Flow Matching 详解](Flow_Matching详解.md) — 替代 DDPM 的新范式
- [Stable Diffusion 详解](Stable_Diffusion详解.md) — Latent Diffusion 架构

[返回上级](README.md) | [返回视觉](../README.md) | [返回总目录](../../README.md)

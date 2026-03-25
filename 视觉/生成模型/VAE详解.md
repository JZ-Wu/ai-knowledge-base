# VAE 详解：从 Autoencoder 到变分自编码器

> 参考：[苗思奇 - 变分自编码器 VAE](https://zhuanlan.zhihu.com/p/348498294) | [Kingma & Welling, 2013 - Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)

---

## 1. 为什么 Autoencoder 不是生成模型

Autoencoder 学习编码器 $f$ 和解码器 $g$，使得 $g(f(x)) \approx x$。训练完之后，解码器 $g$ 看起来很诱人——它只需要一个低维向量 $z$ 就能输出一张图片。那我们能不能直接在 $z$ 空间里随机采样，然后用 $g$ 生成图片？

答案是：**不行**。绝大多数随机生成的 $z$ 只会产生无意义的噪声。

```
为什么？
  AE 没有对 z 的分布 p(z) 进行建模
  训练数据编码后，z 只分布在高维空间中极有限的几个区域
  z 空间绝大部分是"空白"，解码器从未见过这些区域的输入

  z 空间示意图：
  z₁ ●          ● z₃       ← 训练数据编码后落在这几个点附近
         ???                ← 这些区域解码器从未见过，输出垃圾
       ●
      z₂
```

**核心问题**：AE 只优化重建，不关心 $z$ 空间的结构。VAE 的思路就是：**显式对 $z$ 的分布进行建模**，强制 $z$ 服从一个已知分布（标准正态），使得空间中任意点都能解码出有意义的样本。

---

## 2. VAE 的生成视角

上一节说了 AE 不能生成，因为它没有对 $z$ 的分布建模。VAE 的核心思路是：**用概率分布重新定义整个生成过程**。

> 后文中所有的 $p(\cdot)$ 均指**概率密度函数**，描述的是一个分布。说"最大化 $p_\theta(x)$"意思是"让模型分布在真实数据 $x$ 处的密度尽可能高"。

### 什么是"模型分布" $p_\theta(x)$？

自然界有一个我们不知道的**真实数据分布** $p_{\text{data}}(x)$（所有自然图片的密度分布）。我们只有有限的训练样本，无法直接获取这个分布。

**模型分布** $p_\theta(x)$ 是我们用参数 $\theta$（Decoder 的权重）构造出来的一个分布，试图去逼近 $p_{\text{data}}(x)$。具体来说，模型通过 Decoder 定义了一个生成过程 $z \sim \mathcal{N}(0, I) \xrightarrow{\text{Decoder}_\theta} x$，这个过程隐式地定义了 $x$ 的一个密度分布——不同的 $\theta$ 产生不同的分布。

```
θ 没训好时:  p_θ(x) 在自然图片处密度低  → 模型不认为自然图片"合理" → 生成垃圾
θ 训好后:    p_θ(x) ≈ p_data(x)        → 模型的分布接近真实分布   → 生成逼真图片

训练目标: 调 θ，让 p_θ(x) 在所有训练样本处的密度尽可能高
         即 max_θ Σ log p_θ(x_i)  (最大似然估计)
```

### 生成过程的概率建模

假设世界上所有图片都是按如下过程"生成"出来的：

```
Step 1:  从先验分布中采样一个隐变量     z ~ p(z) = N(0, I)     ← 固定分布，无参数
Step 2:  用 Decoder 将 z 映射为图片    x ~ p_θ(x|z)           ← θ 是 Decoder 的参数
```

- $z$：不可观测的隐变量（latent variable），低维"概念种子"
- $p(z) = \mathcal{N}(0, I)$：先验分布，人为固定，**不含可学习参数**
- $p_\theta(x|z)$：似然分布（Decoder），参数为 $\theta$，表示"给定 $z$ 时，数据 $x$ 的条件密度"
- $\theta$：Decoder 神经网络的权重，是我们要学习的参数

### 训练目标：最大化边际似然

训练目标：调整 $\theta$，使模型分布 $p_\theta(x)$ 在训练数据处的密度尽可能高。

$p_\theta(x)$ 是对 $z$ 积分（边际化）后的密度：

$$p_\theta(x) = \int p_\theta(x|z) \, p(z) \, dz$$

直觉：$x$ 可以由任何 $z$ 生成，把所有 $z$ 的贡献积起来就是 $x$ 的总密度。但这个积分**算不了**——$z$ 维度高，与 $x$ 真正相关的 $z$ 只占极小区域，暴力采样效率极低。这就是下一节要解决的问题。

---

## 3. 三个连环困难和 VAE 的解法

到这里我们有了目标（最大化 $p_\theta(x)$），但要真正优化它，会连续撞上三堵墙。VAE 的精妙之处就在于一步步绕过这三个困难。

### 困难 1: 积分算不了

$p_\theta(x) = \int p_\theta(x|z) \, p(z) \, dz$ 需要穷举所有可能的 $z$。$z$ 维度太高（比如 128 维），与某个 $x$ 真正相关的 $z$ 只占极小的区域，随机采样几乎全部落在无关区域，效率极低。

```
想象在一个 128 维空间里大海捞针:
  随机撒 100 万个点，可能一个都没有落在"能生成这张猫图"的区域里
  → 蒙特卡洛估计几乎全是 0，没法用
```

### 困难 2: 后验也算不了

如果我们知道"哪些 $z$ 能生成这张图 $x$"，即后验分布 $p_\theta(z|x)$，就可以只在那个小区域采样。但由贝叶斯公式：

$$p_\theta(z|x) = \frac{p_\theta(x|z) \, p(z)}{p_\theta(x)}$$

分母 $p_\theta(x)$ 又是那个算不了的积分——**死循环了**。

### 困难 3: Decoder 是确定性的，怎么算密度？

实际的 Decoder 就是一个神经网络 $f_\theta(z)$：输入 $z$，输出确定的图 $\hat{x}$。但上面的公式需要 $p_\theta(x|z)$ 是一个**分布**，否则积分和概率推导都无法进行。

### VAE 的三个解法

| 困难 | VAE 的解法 |
|------|-----------|
| 积分算不了 | 不直接算 $p_\theta(x)$，转而优化它的**下界** ELBO（见第 6 节） |
| 后验算不了 | 训一个 **Encoder** $q_\phi(z\|x)$ 去近似后验（见第 5 节） |
| Decoder 是确定性的 | 把 Decoder 输出**包装成高斯分布**（见第 4 节） |

先从最简单的困难 3 开始解决。

---

## 4. Decoder 的设计：把确定性函数包装成分布

Decoder 神经网络是确定性的：$\hat{x} = f_\theta(z)$。我们把它包装成一个以 $f_\theta(z)$ 为均值的高斯分布：

$$p_\theta(x|z) = \mathcal{N}(x; \; f_\theta(z), \; \sigma^2 I)$$

意思是："给定 $z$，真实图片 $x$ 大概在 $f_\theta(z)$ 附近，允许有 $\sigma$ 的偏差"。

**为什么要这么做？** 因为整个概率框架（积分、贝叶斯公式、ELBO 推导）都需要 $p_\theta(x|z)$ 是一个分布。把它建模成高斯后，还有一个好处——负对数似然直接变成了 MSE：

$$-\log p_\theta(x|z) = \frac{1}{2\sigma^2}\|x - f_\theta(z)\|^2 + \text{常数}$$

**最大化似然 = 最小化 MSE**。高斯假设就是一个数学工具，让我们从概率语言推导出一个可以算的 loss。

```
总结:
  数学上:   p_θ(x|z) = N(x; f_θ(z), σ²I)     ← 高斯分布（为了推导 loss）
  实现上:   Decoder(z) = f_θ(z) = x̂           ← 确定性神经网络（实际跑的代码）
  生成时:   直接用 x̂ 当结果                    ← 不从分布中采样（这是 VAE 模糊的原因之一）
  σ² 的作用: 控制重建损失和 KL 正则的相对权重   ← 见第 9 节
```

---

## 5. Encoder 的设计：近似后验

困难 2 说真实后验 $p_\theta(z|x)$ 算不了。VAE 的解法是：**再训一个网络来近似它**。

这个网络就是 Encoder（参数为 $\phi$），输入图片 $x$，输出"这张图对应的 $z$ 大概在哪个范围"：

$$q_\phi(z|x) = \mathcal{N}(z; \; \mu_\phi(x), \; \sigma_\phi^2(x) \, I)$$

Encoder 输出两个东西：均值 $\mu_\phi(x)$ 和方差 $\sigma_\phi^2(x)$（实际输出 $\log \sigma^2$ 保证方差为正）。

```
Encoder 的作用:
  输入一张猫图 x
  → 输出 μ = [0.3, -1.2, ...] 和 σ² = [0.1, 0.05, ...]
  → 意思是 "这张猫图对应的 z 大概在 μ 附近，不确定性为 σ"
```

现在我们有了 Encoder（近似后验）和 Decoder（生成器），还剩困难 1：怎么绕过那个算不了的积分？

---

## 6. ELBO：绕过积分的巧妙下界

### 思路

$\log p_\theta(x)$ 直接算不了（需要积分）。但我们可以找一个**下界**——一个比它小但能算的量。优化下界就相当于间接在推高 $\log p_\theta(x)$。这个下界叫 **ELBO**（Evidence Lower Bound）。

### 推导（每一步都解释为什么）

**第 1 步**：$\log p_\theta(x)$ 和 $z$ 无关，可以乘一个积分为 1 的东西而不改变值：

$$\log p_\theta(x) = \int q_\phi(z|x) \cdot \log p_\theta(x) \, dz$$

> 为什么能这样？因为 $\int q_\phi(z|x) dz = 1$（$q_\phi$ 是一个分布，积分为 1），所以相当于乘了 1。

**第 2 步**：用贝叶斯公式把 $p_\theta(x)$ 拆开：$p_\theta(x) = \frac{p_\theta(x,z)}{p_\theta(z|x)}$

$$= \int q_\phi(z|x) \cdot \log \frac{p_\theta(x,z)}{p_\theta(z|x)} \, dz$$

> 为什么要拆？因为 $p_\theta(x)$ 是那个算不了的积分，但 $p_\theta(x,z) = p_\theta(x|z) \cdot p(z)$ 是两个我们知道的东西相乘，能算。

**第 3 步**：在分数中插入 $q_\phi$，拆成两项：

$$= \int q_\phi(z|x) \log \frac{p_\theta(x,z)}{q_\phi(z|x)} \, dz + \int q_\phi(z|x) \log \frac{q_\phi(z|x)}{p_\theta(z|x)} \, dz$$

$$= \underbrace{\text{ELBO}}_{\text{能算，要最大化}} + \underbrace{D_{KL}(q_\phi(z|x) \| p_\theta(z|x))}_{\geq 0, \text{ 算不了但没关系}}$$

> 为什么要插入 $q_\phi$？为了把"算不了的 $p_\theta(z|x)$"隔离到 KL 散度项里。KL $\geq 0$，所以 ELBO $\leq \log p_\theta(x)$，是一个下界。

**结论**：

$$\boxed{\log p_\theta(x) \geq \text{ELBO} = \mathbb{E}_{q_\phi(z|x)} \left[\log \frac{p_\theta(x,z)}{q_\phi(z|x)}\right]}$$

最大化 ELBO 同时做到了两件事：
1. 推高 $\log p_\theta(x)$（让模型觉得训练数据"合理"）
2. 让 $q_\phi(z|x)$ 逼近真实后验 $p_\theta(z|x)$（让 KL 项趋近 0，下界更紧）

### 展开 ELBO 得到最终 loss

把 $p_\theta(x,z) = p_\theta(x|z) \cdot p(z)$ 代入，ELBO 拆成两项：

$$\text{ELBO} = \underbrace{\mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)]}_{\text{重建项}} - \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL 正则项}}$$

| 项 | 公式含义 | 白话 |
|---|---|---|
| **重建项** | Encoder 编码的 $z$ 能多好地重建 $x$ | "Encoder 和 Decoder 配合好，别丢信息" |
| **KL 正则项** | $q_\phi(z\|x)$ 和先验 $\mathcal{N}(0,I)$ 有多接近 | "编码出的 $z$ 分布别太奇怪，要像标准正态" |

两项之间有**矛盾**：重建项希望 $z$ 携带尽可能多的信息（每张图编码到不同的点），KL 项希望所有图编码到同一个分布（标准正态）。VAE 的训练就是在这两者之间找平衡。

---

## 7. 重参数化技巧（Reparameterization Trick）

VAE 架构中有一步"从 $q_\phi(z|x)$ 中采样 $z$"，但**采样操作不可导**，梯度无法反传给 $\phi$。

**技巧**：把随机性移到一个与参数无关的噪声变量 $\epsilon$ 上：

$$z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

```
原来:  φ → q_φ(z|x) → 采样 z (不可导)  → Decoder → Loss
                         ✗ 梯度断了

重参数化后:
       φ → μ_φ(x), σ_φ(x)  →  z = μ + σ⊙ε  → Decoder → Loss
            可导        可导       可导（ε 当作常数）
                                    ↑
                               ε ~ N(0,I) 外部输入
```

$z$ 的分布没变，还是 $\mathcal{N}(\mu, \sigma^2 I)$，但现在 $z$ 是关于 $\mu, \sigma$ 的确定性函数，梯度可以正常反传。

---

## 8. KL 散度的解析解

$q_\phi(z|x) = \mathcal{N}(\mu, \sigma^2 I)$ 与 $p(z) = \mathcal{N}(0, I)$ 的 KL 散度有闭式解。

先看一维情况（$q = \mathcal{N}(\mu, \sigma^2)$，$p = \mathcal{N}(0, 1)$）：

$$D_{KL}(q \| p) = \int q(z) \log \frac{q(z)}{p(z)} dz = \frac{1}{2}(\mu^2 + \sigma^2 - \log \sigma^2 - 1)$$

**直觉理解各项**：
- $\mu^2$：均值偏离 0 越远，惩罚越大
- $\sigma^2 - \log \sigma^2 - 1$：当 $\sigma^2 = 1$ 时为 0（与标准正态方差一致），偏离 1 时惩罚增大

推广到 $d$ 维（各维度独立）：

$$D_{KL}(q \| p) = \frac{1}{2} \sum_{j=1}^{d} \left(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1\right)$$

---

## 9. 完整损失函数

将重建项和 KL 项合并（取负号变为最小化）：

$$\mathcal{L}_{\text{VAE}} = \underbrace{-\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)]}_{\text{重建损失}} + \underbrace{D_{KL}(q_\phi(z|x) \| p(z))}_{\text{KL 正则}}$$

**重建项展开**：假设 $p_\theta(x|z) = \mathcal{N}(x; \mu_\theta'(z), \sigma'^2 I)$，其中 $x$ 维度为 $D$：

$$-\log p_\theta(x|z) = \frac{D}{2}\log(2\pi\sigma'^2) + \frac{1}{2\sigma'^2}\|x - \mu_\theta'(z)\|^2$$

去掉常数项后，重建损失正比于 $\frac{1}{2\sigma'^2}\|x - \mu_\theta'(z)\|^2$。

实践中从 $q_\phi(z|x)$ 只采样一个 $z$（$L=1$），最终损失：

$$\mathcal{L} = \frac{1}{2\sigma'^2}\|x - \hat{x}\|^2 + \frac{1}{2}\sum_{j=1}^{d}(\mu_j^2 + \sigma_j^2 - \log \sigma_j^2 - 1)$$

其中 $\hat{x} = \mu_\theta'(z)$ 是 Decoder 输出。

---

## 10. 实现中的常见陷阱

### 关于重建损失中的 $\sigma'^2$

很多 PyTorch 实现直接用 `F.mse_loss(x_recon, x, reduction='mean')` 计算重建损失。**这是错误的！**

```python
# ❌ 错误做法
recon_loss = F.mse_loss(x_recon, x, reduction='mean')  # 除以了 D

# ✅ 正确做法
recon_loss = F.mse_loss(x_recon, x, reduction='sum')   # 不除以 D
```

`F.mse_loss(..., reduction='mean')` 等价于 $\frac{1}{D}\sum_i(x_i - \hat{x}_i)^2$，多除了一个数据维度 $D$。

**为什么这很关键？** 这个 $D$ 不是无关紧要的常数——它实际上控制了**重建损失和 KL 正则之间的相对权重**。多除一个 $D$（对于 MNIST，$D=784$），等价于把 Decoder 似然分布的方差 $\sigma'^2$ 放大了 $D$ 倍。方差极大意味着模型认为"重建得不像也无所谓"，Decoder 就不会好好重建了。

> 用 `mean` 也不是不行，但需要相应调小 KL 项的权重（等价于 $\beta$-VAE）。关键是**理解 sum 和 mean 的区别就是在调重建与 KL 的相对权重**。

### 完整实现

```python
def vae_loss(x, x_recon, mu, log_var):
    # 重建损失：用 sum 而非 mean
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    # KL 散度（解析解）
    kl_loss = 0.5 * torch.sum(mu.pow(2) + log_var.exp() - log_var - 1)
    return recon_loss + kl_loss
```

---

## 11. VAE 为什么生成模糊？

这是最经典的问题。主要有两种解释：

### 解释一：直接用均值代替采样

Decoder 本应输出的是 $x$ 服从的分布 $\mathcal{N}(\mu', \sigma'^2 I)$，然后从中采样。但实际中我们直接拿均值 $\mu'$ 当生成结果。均值天然是"平均"了所有可能性的，自然模糊。

### 解释二（更本质）：编码分布重叠 + 高斯假设

回顾 VAE 的概率建模框架：虽然 Encoder 和 Decoder 的**神经网络**本身是确定性函数，但在 VAE 中它们都被包装成了**概率分布**：

- Encoder 输出分布 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x) I)$，训练时从中**采样** z
- Decoder 输出分布 $p_\theta(x|z) = \mathcal{N}(x; f_\theta(z), \sigma'^2 I)$，即以网络输出为均值的高斯

整个 VAE 的训练和推导都在这个概率框架下进行。模糊的根本原因也出在这个框架中：

**z 空间维度远低于 x，不同训练图片的编码分布会重叠。**

```
猫图A 的编码分布: q_φ(z|A) = N(μ_A, σ_A²)  ─╮
                                               ├─ 在 z 空间中有重叠区域
猫图B 的编码分布: q_φ(z|B) = N(μ_B, σ_B²)  ─╯

  z 空间:
     ╭──A的分布──╮
     │     ╭──B的分布──╮
     │     │ z* │  │   │   ← z* 落在重叠区
     ╰─────│────╯  │   │
           ╰───────╯   │

当 z* 在重叠区被采样到时:
  → 训练时，z* 既可能来自 A 的编码，也可能来自 B 的编码
  → Decoder 的 p_θ(x|z*) 需要同时让 A 和 B 的似然都高
  → 但高斯分布 p_θ(x|z*) = N(x; f_θ(z*), σ²I) 只有一个均值
  → 为了最小化 MSE，f_θ(z*) 只能是 A 和 B 的"折中" → 模糊

本质原因: z 空间维度低 → 编码分布必然重叠
         + 似然 p_θ(x|z) 是单峰高斯（无法同时给 A 和 B 高密度）
         → Decoder 只好输出均值（折中）
```

---

## 12. VAE 的改进方向

| 问题 | 改进 | 代表 |
|---|---|---|
| 生成模糊 | 换更复杂的似然分布 / 加 GAN 判别器 | VAE-GAN |
| 后验坍缩（KL 项太强，$z$ 被忽略） | KL annealing（逐渐增大 KL 权重） | $\beta$-VAE |
| 高斯假设太简单 | 用 Normalizing Flows 增强后验表达力 | IAF-VAE |
| 连续空间不适合离散数据 | 向量量化，离散化潜在空间 | **VQ-VAE** |

---

## 13. Conditional VAE (CVAE)

原版 VAE 无法控制生成什么类别。CVAE 引入标签 $y$ 作为条件：

```
原版 VAE:
  Encoder:  q_φ(z|x)        Decoder:  p_θ(x|z)

CVAE:
  Encoder:  q_φ(z|x, y)     Decoder:  p_θ(x|z, y)

做法: 将 y (如 one-hot 类别向量) 拼接到 Encoder 和 Decoder 的输入中
     → 推理时: 指定 y="7" + 采样 z → 生成数字 7
```

---

## 14. VQ-VAE：离散潜在空间

VQ-VAE（van den Oord et al., 2017）用**向量量化**替代连续高斯分布，是 Stable Diffusion 中 VAE 组件的基础。

### 核心思想

用一个**离散 Codebook** 替代连续的高斯分布。Encoder 输出连续向量后，找 Codebook 中最近的向量替代：

```
Encoder → 连续向量 z_e → 找 Codebook 中最近的向量 z_q → Decoder

Codebook: {e₁, e₂, ..., e_K}  (K 个可学习的嵌入向量)
z_q = e_k,  k = argmin_i ||z_e - e_i||
```

### 损失函数

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|^2}_{\text{重建损失}} + \underbrace{\|\text{sg}[z_e] - e\|^2}_{\text{codebook loss}} + \underbrace{\beta\|z_e - \text{sg}[e]\|^2}_{\text{commitment loss}}$$

- $\text{sg}[\cdot]$：stop gradient（梯度截断）
- codebook loss：更新 Codebook 向量，使其靠近 Encoder 输出
- commitment loss：让 Encoder 输出稳定在某个 Codebook 向量附近

### 梯度怎么通过离散操作？

$\text{argmin}$ 是不可导的。VQ-VAE 用 **straight-through estimator**：前向时用量化后的 $z_q$，反向时直接把梯度传给 $z_e$（跳过量化步骤）。

### 为什么比连续 VAE 更清晰？

- 没有 KL 正则 → 不强制 $z$ 服从标准正态 → 信息保留更完整
- 离散 codebook → 没有"编码分布重叠"的问题 → 不会被迫取均值
- 生成时需要额外训练一个 **先验模型**（如 PixelCNN 或 Transformer）来学习 codebook token 的分布

### 在 Stable Diffusion 中的应用

Stable Diffusion 使用的 VAE 实际上是 KL-regularized autoencoder（介于 VAE 和 VQ-VAE 之间）：
- 空间压缩 8 倍（512×512 → 64×64）
- 通道数 3 → 4
- 训练好后冻结，扩散过程在潜在空间进行

---

**相关文档**：
- [KL散度](../../机器学习基础/KL散度.md) — VAE 中 KL 正则的数学基础
- [Diffusion 详解](Diffusion详解.md) — 从 DDPM 到 Score Matching
- [Stable Diffusion 详解](Stable_Diffusion详解.md) — VAE 在 LDM 中的应用
- [CNN与经典网络](../../机器学习基础/深度学习基础/CNN与经典网络.md) — Autoencoder 基础

[返回上级](README.md) | [返回视觉](../README.md) | [返回总目录](../../README.md)

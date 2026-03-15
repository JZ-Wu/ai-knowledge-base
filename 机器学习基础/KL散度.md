# KL 散度 (Kullback-Leibler Divergence)

> 更新日期: 2026-03-15

## 定义

KL 散度衡量两个概率分布 P 和 Q 的差异程度，又叫相对熵。

$$D_{KL}(P \| Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = \mathbb{E}_{x \sim P}\left[\log \frac{P(x)}{Q(x)}\right]$$

连续形式：

$$D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)} dx$$

## 核心性质

| 性质 | 说明 |
|------|------|
| 非负性 | $D_{KL}(P\|Q) \geq 0$，当且仅当 $P=Q$ 时取等 (Gibbs 不等式) |
| 不对称 | $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$，所以不是"距离" |
| 不满足三角不等式 | 不是度量 (metric) |

## Forward KL vs Reverse KL

这是面试高频考点，必须理清方向。

### Forward KL: $D_{KL}(P \| Q)$ — 用 Q 去拟合 P

- P 是目标分布（真实/教师），Q 是我们要学的分布（模型/学生）
- **Mean-seeking (均值追求)**：Q 会尽量覆盖 P 的所有模式
- 当 $P(x) > 0$ 时，要求 $Q(x) > 0$（否则 KL 趋向无穷），所以 Q 不敢在 P 有概率的地方给 0
- 结果：Q 倾向"更宽"，可能把多个模式模糊地混在一起

### Reverse KL: $D_{KL}(Q \| P)$ — 用 P 去评价 Q

- **Mode-seeking (模式追求)**：Q 会集中在 P 的某一个模式上
- 当 $Q(x) > 0$ 时，要求 $P(x) > 0$，所以 Q 不敢在 P 概率为 0 的地方分配概率
- 结果：Q 倾向"更窄"，精确匹配某一个峰，但可能丢失其他模式

### 直觉记忆

```
Forward KL: "我（Q）不能漏掉你（P）的任何东西" → 宽泛覆盖
Reverse KL: "我（Q）不能编造你（P）没有的东西" → 精准集中
```

### 在大模型中的应用方向

| 场景 | 使用方向 | 原因 |
|------|---------|------|
| 知识蒸馏 | Forward KL $D_{KL}(P_{teacher}\|Q_{student})$ | 学生需要覆盖教师的所有知识 |
| RLHF/PPO | Reverse KL $D_{KL}(\pi_\theta\|\pi_{ref})$ | 策略不要偏离参考模型太远 |
| DPO | 隐式 Reverse KL 约束 | 同上，通过 $\beta$ 参数控制 |
| VAE | Reverse KL $D_{KL}(q(z|x)\|p(z))$ | 后验逼近先验 |

## 与其他散度/距离的关系

| 度量 | 公式 | 特点 |
|------|------|------|
| KL 散度 | $\sum P \log(P/Q)$ | 不对称，信息论基础 |
| JS 散度 | $\frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M)$, $M=\frac{P+Q}{2}$ | 对称，有界 $[0, \log 2]$，GAN 原始损失 |
| 交叉熵 | $H(P,Q) = -\sum P \log Q$ | $= H(P) + D_{KL}(P\|Q)$，训练时 $H(P)$ 是常数所以等价于最小化 KL |
| Wasserstein 距离 | 最优传输距离 | WGAN 使用，即使分布不重叠也有梯度 |

## 交叉熵与 KL 的关系

$$H(P, Q) = H(P) + D_{KL}(P \| Q)$$

当 P 是固定的真实标签分布时，$H(P)$ 是常数，所以：

**最小化交叉熵 = 最小化 Forward KL**

这就是为什么 LLM 的 next-token prediction 用交叉熵损失，本质上就是在做 Forward KL 最小化。

---

相关手撕代码: [KL Loss 实现](../面试手撕/大模型手撕/手撕代码合集.md#9-kl-散度损失)
在大模型训练中的应用: [知识蒸馏与KL约束](../大模型/训练与微调/知识蒸馏与KL约束.md)

---
[返回上级](README.md) | [返回总目录](../README.md)

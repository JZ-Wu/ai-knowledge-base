# 知识蒸馏与 KL 约束

> 更新日期: 2026-03-15

KL 散度在大模型训练中有两大核心应用：**知识蒸馏**和 **RLHF 中的策略约束**。

数学基础参见: [KL 散度](../../机器学习基础/KL散度.md)

---

## 一、知识蒸馏 (Knowledge Distillation)

### 核心思想

用大模型（教师）的软标签来训练小模型（学生），比只用 hard label 的交叉熵能传递更多信息。

```
教师模型 (frozen) → logits → softmax(logits/T) → 软标签分布 P
                                                        ↓
学生模型 (trainable) → logits → softmax(logits/T) → 学生分布 Q  → KL(P || Q)
```

### 损失函数

$$\mathcal{L} = \alpha \cdot T^2 \cdot D_{KL}(P_{teacher} \| Q_{student}) + (1-\alpha) \cdot \text{CE}(y, Q_{student})$$

- **第一项**: 蒸馏损失，用 Forward KL 让学生覆盖教师的知识
- **第二项**: 标准交叉熵，用真实标签保证基本准确性
- **$T$ (温度)**: 通常 2~20，越大分布越平滑，暗知识越多
- **$T^2$ 补偿**: 温度缩放让梯度缩小了 $T^2$ 倍，需要乘回来
- **$\alpha$**: 平衡系数，通常 0.5~0.9

### 为什么软标签有效？

硬标签只告诉模型"答案是猫"，但软标签告诉模型"90% 是猫，8% 是老虎，2% 是豹子"。这种类间相似性关系（暗知识/dark knowledge）对学生模型很有价值。

### 温度的作用

| 温度 T | 效果 |
|--------|------|
| T=1 | 标准 softmax，高置信度类别主导 |
| T=2~5 | 适度平滑，常用范围 |
| T=10~20 | 非常平滑，类间差异被放大 |
| T→∞ | 均匀分布，无信息 |

### LLM 蒸馏的特殊性

LLM 的蒸馏比 CV 分类更复杂：

1. **词表巨大** (32K~150K)：softmax 计算量大，通常只在 top-k token 上算 KL
2. **序列生成**：可以用教师的输出序列做 SFT（黑盒蒸馏），也可以逐 token 对齐 logits（白盒蒸馏）
3. **黑盒 vs 白盒**:
   - 黑盒：只用教师的生成文本做 SFT，如 Alpaca 用 GPT-4 生成数据
   - 白盒：对齐 logits 分布，需要教师模型的完整权重

---

## 二、RLHF 中的 KL 惩罚

### 为什么需要 KL 约束？

RLHF 的目标是让模型根据人类偏好调整输出。但如果只最大化奖励，模型会 **reward hacking**——找到骗过奖励模型的捷径，生成高分但低质量的输出。

KL 惩罚防止策略模型偏离参考模型太远：

$$\mathcal{L}_{RLHF} = \mathbb{E}_{x \sim \pi_\theta}\left[R(x)\right] - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})$$

即：**最大化奖励的同时，不要离 SFT 模型太远**。

### 注意方向：Reverse KL

这里用的是 **Reverse KL** $D_{KL}(\pi_\theta \| \pi_{ref})$，不是 Forward KL：

- 策略 $\pi_\theta$ 是 Q（我们在优化的）
- 参考 $\pi_{ref}$ 是 P（固定的）
- Mode-seeking：策略会集中在参考模型已有概率质量的区域，不会乱编

### $\beta$ 的调节

| $\beta$ | 效果 |
|---------|------|
| 太小 | 约束太弱，容易 reward hacking |
| 太大 | 约束太强，模型几乎不更新，退化为 SFT |
| 适中 (0.01~0.2) | 平衡奖励提升和稳定性 |

PPO 训练中通常用自适应 KL 系数，动态调节 $\beta$：如果 KL 超过目标值就增大，低于目标值就减小。

### 在 PPO 中的实现

```python
# 简化的 PPO + KL 惩罚
log_ratio = log_prob_new - log_prob_old  # 新旧策略的 log ratio
kl_penalty = (log_prob_new - log_prob_ref).mean()  # 与参考模型的 KL

reward_with_kl = reward - beta * kl_penalty
# 然后用 PPO 的 clipped objective 优化
```

---

## 三、DPO 中的隐式 KL 约束

DPO 的损失函数中 $\beta$ 参数本质上就是 KL 约束的强度：

$$\mathcal{L}_{DPO} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)$$

- $\beta$ 越大：约束越强，策略越接近参考模型
- $\beta$ 越小：约束越弱，允许更大偏离

DPO 的数学推导证明，它等价于在 Reverse KL 约束下最大化奖励的闭式解。

详见: [DPO详解](DPO详解.md)

---

## 四、GRPO 中的 KL

GRPO 去掉了 critic model，但保留了 KL 惩罚：

$$\mathcal{L}_{GRPO} = \mathbb{E}\left[\frac{\pi_\theta}{\pi_{old}} A_{group} - \beta \cdot D_{KL}(\pi_\theta \| \pi_{ref})\right]$$

GRPO 使用近似 KL：$D_{KL} \approx \frac{\pi_{ref}}{\pi_\theta} - \log \frac{\pi_{ref}}{\pi_\theta} - 1$，计算更高效。

详见: [GRPO详解](GRPO详解.md)

---

## 总结对比

| 场景 | KL 方向 | 目的 | 典型 $\beta$ / $T$ |
|------|---------|------|-------------------|
| 知识蒸馏 | Forward KL $D_{KL}(P_T\|Q_S)$ | 学生覆盖教师知识 | T=2~20, α=0.5~0.9 |
| RLHF/PPO | Reverse KL $D_{KL}(\pi_\theta\|\pi_{ref})$ | 防止 reward hacking | β=0.01~0.2 |
| DPO | 隐式 Reverse KL | 约束策略偏离程度 | β=0.1~0.5 |
| GRPO | Reverse KL (近似) | 同上 | β=0.04 (DeepSeek) |

---

手撕代码: [KL Loss 实现](../../面试手撕/大模型手撕/手撕代码合集.md#9-kl-散度损失)
数学基础: [KL 散度](../../机器学习基础/KL散度.md)

---
[返回上级](README.md) | [返回总目录](../../README.md)

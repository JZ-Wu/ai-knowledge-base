# MoE 详解 (Mixture of Experts)

混合专家模型，通过**稀疏激活**实现"参数量大但计算量小"的效果。

## 一、核心思想

传统 Dense 模型：每个 token 经过所有参数 → 参数量 = 计算量
MoE 模型：每个 token 只经过**部分专家** → 总参数量大，但单次推理计算量小

```
Dense Model (7B):                MoE Model (如 Mixtral 8x7B):
每个 token → 全部 7B 参数        每个 token → 路由选 2 个专家 → 约 13B 激活参数
                                 总参数 47B，但推理成本接近 13B 的 Dense 模型
```

## 二、架构

### 基本结构

MoE 替换的是 Transformer 中的 **FFN 层**（注意力层不变）：

![MoE 架构: 标准 Transformer 层 vs MoE Transformer 层 (Mixtral, Jiang et al., 2024)](assets/moe_architecture.png)

> 图源: *Mixtral of Experts*, Figure 1. 左侧为标准 Transformer Block (单个 FFN)，右侧为 MoE 层 (Router 选择 Top-K 专家并行计算后加权求和)。

### Router（门控网络）

最简单的路由就是一个线性层 + TopK：

$$g(x) = \text{TopK}(\text{Softmax}(W_g \cdot x))$$

- 输入：token 的隐状态 $x \in \mathbb{R}^{d}$
- $W_g \in \mathbb{R}^{N \times d}$：门控权重矩阵，$N$ 为专家数
- TopK：选择得分最高的 K 个专家（通常 K=2）

### 输出计算

$$y = \sum_{i \in \text{TopK}} g_i(x) \cdot E_i(x)$$

每个被选中的专家的输出乘以对应的门控权重，再加权求和。

## 三、关键问题与解决方案

### 1. 负载均衡 (Load Balancing)

**问题**：路由可能"偏心"，大量 token 被分配给少数专家，其余专家闲置（专家坍缩）。

**解决方案——辅助损失 (Auxiliary Loss)**：

$$L_{balance} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot p_i$$

- $f_i$：分配给专家 $i$ 的 token 比例
- $p_i$：路由对专家 $i$ 的平均门控概率
- $\alpha$：平衡系数（通常 0.01）
- 目标：让每个专家接收到大致相等的 token 数

**DeepSeek-V3 的改进——无辅助损失的负载均衡**：
- 给每个专家加一个可学习的 bias，动态调整路由偏好
- 避免辅助损失对模型性能的干扰

### 2. Expert Capacity（专家容量）

设定每个专家在一个 batch 中最多处理多少 token：

$$\text{capacity} = \text{capacity\_factor} \times \frac{\text{tokens}}{N}$$

- capacity_factor 通常 1.0~1.5
- 超出容量的 token 会被丢弃（token dropping）或溢出到其他专家

### 3. 通信开销

MoE 的 token 需要路由到不同 GPU 上的专家，涉及 **All-to-All 通信**：

```
GPU 0 上的 token: [t1, t2, t3, t4]
路由结果: t1→E0, t2→E3, t3→E1, t4→E0

需要 All-to-All:
  t2 从 GPU0 发送到 GPU3
  t3 从 GPU0 发送到 GPU1
  其他 GPU 的 token 也类似交换
```

这是 MoE 训练和推理的主要瓶颈之一。

## 四、主流 MoE 模型对比

| 模型 | 总参数 | 激活参数 | 专家数 | TopK | 特点 |
|------|--------|---------|--------|------|------|
| **Mixtral 8x7B** | 47B | ~13B | 8 | 2 | 首个开源高质量 MoE |
| **Mixtral 8x22B** | 176B | ~44B | 8 | 2 | 更大版本 |
| **DeepSeek-V2** | 236B | 21B | 160 | 6 | 细粒度专家 + 共享专家 |
| **DeepSeek-V3** | 671B | 37B | 256 | 8 | 无辅助损失均衡、MLA 注意力 |
| **Qwen-MoE** | 14.3B | 2.7B | 60 | 4 | 小型 MoE |
| **GPT-4** | 传闻 ~1.8T | - | 传闻 16 | 2 | 未公开，传闻 MoE |

## 五、DeepSeek 的 MoE 创新

DeepSeek 在 MoE 上做了大量创新，值得单独关注：

### 细粒度专家 (Fine-grained Experts)

传统：8 个大专家
DeepSeek：把 1 个大专家拆成多个小专家（如 256 个），选 8 个

**好处**：组合空间指数级增大（$C_{256}^{8}$ >> $C_{8}^{2}$），更灵活的专家分配。

### 共享专家 (Shared Experts)

在 TopK 专家之外，额外设置若干**始终激活**的共享专家：

```
Token → Router → TopK 路由专家（稀疏）
  │
  └──→ 共享专家（始终激活）
```

共享专家捕获通用知识，路由专家学习专业知识。

### MLA (Multi-head Latent Attention)

虽然不是 MoE 的一部分，但 DeepSeek-V2/V3 同时引入了 MLA：
- 将 KV Cache 压缩到低维潜空间
- 大幅减少推理时的显存占用
- 详见 [MLA详解](MLA详解.md)（待创建）

## 六、MoE vs Dense 的权衡

| 维度 | MoE | Dense |
|------|-----|-------|
| **同等计算量下的性能** | 更强（参数量更大） | 较弱 |
| **训练效率** | 更高（同 FLOPS 下更多参数） | 标准 |
| **推理显存** | 更大（需加载全部参数） | 更小 |
| **推理速度** | 取决于实现（通信开销） | 稳定 |
| **训练稳定性** | 更难（路由、负载均衡） | 更稳定 |
| **模型蒸馏** | 可蒸馏为 Dense（如 DeepSeek-R1 蒸馏版） | - |
| **适合场景** | 预算充足、追求极致性能 | 端侧部署、简单场景 |

## 七、一句话总结

MoE 的核心价值：**用更少的计算获取更多的参数容量**，让模型在固定推理预算下存储更多知识。代价是工程复杂度更高（路由、负载均衡、通信）。

---

**相关文档**：
- [预训练与后训练](../训练与微调/预训练与后训练.md)
- [团队日常工作](../训练与微调/团队日常工作.md)

[返回上级](README.md) | [返回总目录](../../README.md)

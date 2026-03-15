# DINO 详解

Self-DIstillation with NO labels — 通过自蒸馏学习视觉特征，无需任何标注。

## 一、DINOv1 (2021, Meta/FAIR)

### 核心思想

不用对比学习的负样本，也不用标签。用**自蒸馏**——让学生网络去模仿教师网络的输出分布。

```
同一张图像 x
  ├── 全局裁剪 (224×224) ──→ Teacher 编码 → tₛ (停止梯度)
  ├── 全局裁剪 (224×224) ──→ Teacher 编码 → tₛ
  ├── 局部裁剪 (96×96)  ──→ Student 编码 → sₛ
  ├── 局部裁剪 (96×96)  ──→ Student 编码 → sₛ
  └── ...

Loss: 让 Student 输出分布 → 匹配 Teacher 输出分布
```

### 师生架构

![DINO 自蒸馏架构 (Caron et al., 2021)](assets/dino_architecture.png)

> 图源: *Emerging Properties in Self-Supervised Vision Transformers (DINO)*, Figure 2. Teacher 通过动量更新 (EMA)，Student 通过梯度更新。两者对同一图像的不同增强视图分别计算输出分布，通过交叉熵损失对齐。

**与 MoCo 的联系**：Teacher 的动量更新机制直接借鉴了 MoCo：

$$\theta_t \leftarrow m \cdot \theta_t + (1 - m) \cdot \theta_s, \quad m = 0.996 \to 1.0$$

### Multi-crop 策略

| 裁剪类型 | 分辨率 | 数量 | 送入 |
|---------|--------|------|------|
| 全局裁剪 | 224×224（覆盖 >50% 图像） | 2 | Teacher + Student |
| 局部裁剪 | 96×96（覆盖 <50% 图像） | 多个(6~8) | 仅 Student |

核心思想：**局部到全局的对应** —— Student 看到局部小图，要能预测出 Teacher 看到全局大图时的输出。迫使模型学习语义一致的特征。

### Centering 和 Sharpening（防止坍缩）

对比学习用负样本防止所有特征坍缩到同一点。DINO 没有负样本，用两个技巧：

1. **Centering**：Teacher 输出减去全局均值中心（EMA 更新），防止某个维度主导
   $$g_t(x) \leftarrow g_t(x) - c, \quad c \leftarrow m_c \cdot c + (1-m_c) \cdot \bar{g}_t$$

2. **Sharpening**：Teacher 用较低温度 $\tau_t$ (如 0.04)，Student 用较高温度 $\tau_s$ (如 0.1)
   - 低温度 → 分布更尖锐 → 更确定的"伪标签"

### DINO 的涌现特性

DINOv1 最令人惊讶的发现：**ViT 的注意力图自动学会了语义分割**。

```
原始图像: [一只狗在草地上]

ViT [CLS] token 对其他 token 的注意力:
→ 自动聚焦在狗的轮廓上
→ 没有任何分割标注！纯自监督学到的
```

这说明 DINO 学到的特征具有很强的**局部语义感知**能力。

---

## 二、DINOv2 (2023, Meta/FAIR)

### 相比 v1 的改进

DINOv2 不是方法的革新，而是**工程和数据的全面升级**。

| 维度 | DINOv1 | DINOv2 |
|------|--------|--------|
| 数据 | ImageNet (1.2M) | LVD-142M（自动策划的 142M 图像） |
| 模型 | ViT-S/B | ViT-S/B/L/g（最大 1.1B 参数） |
| 训练目标 | 纯自蒸馏 | 自蒸馏 + iBOT (mask image modeling) |
| 蒸馏 | 无 | 大模型蒸馏到小模型 |
| 效果 | 好 | 接近/超越 OpenCLIP（不需要文本！） |

### 训练目标：自蒸馏 + iBOT

```
DINOv2 = DINO 自蒸馏 (图像级) + iBOT (patch级 masked prediction)

图像级: [CLS] token 的 Teacher-Student 匹配（同 DINOv1）
Patch级: 随机 mask 一些 patch，Student 预测被 mask patch 的 Teacher 特征
```

两个目标互补：
- 图像级目标 → 全局语义
- Patch 级目标 → 局部细节

### 数据策划 (LVD-142M)

1. 从互联网爬取大量未标注图像
2. 用预训练模型做 embedding
3. 用 ImageNet 的分布做参考，检索语义相似的图像
4. 自动去重、过滤
5. 得到 142M 高质量、多样化的训练集

### DINOv2 的特征质量

DINOv2 的特征被广泛用于下游任务（通常冻结 backbone，只训线性探头或轻量 head）：

| 下游任务 | 表现 |
|---------|------|
| 图像分类 | 线性探头接近监督 ViT |
| 语义分割 | 冻结 backbone + 简单 head = 强分割 |
| 深度估计 | 作为 backbone 效果极好（Depth Anything 就用 DINOv2） |
| 特征匹配 | 跨图像的 patch 特征高度语义一致 |

---

## 三、DINO vs CLIP

| 维度 | DINO | CLIP |
|------|------|------|
| **监督信号** | 纯视觉自监督（无文本） | 图文对比（需要文本） |
| **训练数据** | 纯图像 | (图像, 文本) 对 |
| **全局特征** | 好 | 非常好 |
| **局部特征** | 非常好（注意力图有语义） | 较弱 |
| **语言对齐** | 无（特征空间与语言无关） | 有（天然对齐） |
| **Zero-shot 分类** | 不能直接做 | 强项 |
| **密集预测（分割/深度）** | 非常强 | 较弱 |
| **VLM 中的角色** | 提供局部视觉特征 | 提供语言对齐的视觉特征 |

> **互补关系**：CLIP 擅长全局语义和语言对齐，DINO 擅长局部细节和密集预测。一些 VLM（如 InternVL）会同时利用两者。

---

## 四、自蒸馏为什么不会坍缩？

这是 DINO 系列最深层的问题——没有负样本，为什么不会所有特征都一样？

答案是三个机制的组合：
1. **动量更新**：Teacher 变化慢，提供稳定的"伪标签"
2. **Centering**：减去均值，防止输出分布坍缩到某个常数
3. **Sharpening**：Teacher 用低温度，输出接近 one-hot，Student 被迫学习有区分度的特征

如果去掉任何一个，训练都会坍缩。

---

**相关文档**：
- [对比学习与CLIP详解](对比学习与CLIP详解.md)
- [VLM详解](../大模型/多模态/VLM详解.md)

[返回上级](README.md) | [返回总目录](../README.md)

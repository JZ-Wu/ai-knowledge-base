# 3D 生成专题

3D 生成是 2024-2026 视觉领域增长最快的子方向之一。这一专题集中梳理**与 3D 表示无关的生成方法**——把 3D 生成模型本身当成一等公民，而不是把它绑定到某个具体表示（NeRF / GS / Mesh）。

> 与本专题区分：
> - 围绕 3DGS 输出的生成器（DreamGaussian / LGM / GRM / GS-LRM）放在 [高斯溅射专题/05-生成式高斯](../高斯溅射/05-生成式高斯.md)
> - 神经辐射场基础与体渲染放在 [NeRF详解](../NeRF详解.md)

## 为什么单开一个专题

2024 年底起出现了一类新工作（**TRELLIS / Hunyuan3D / Direct3D**），它们不再绑定具体 3D 表示，而是先学一个**结构化 3D 潜变量**，再让多个 decoder 把同一个潜变量翻译成 GS / mesh / 辐射场。这把"生成模型"和"输出格式"彻底解耦了，相当于 3D 版本的 Stable Diffusion 范式（latent + VAE decoder）。

把这类工作塞进高斯专题就不准确了——它们的核心创新不在高斯，而在表示设计。所以独立成章。

## 阅读路径

| 文章 | 内容 | 推荐前置 |
|------|------|---------|
| [01 · TRELLIS 详解](01-TRELLIS详解.md) | Microsoft 2024-12 的开山之作：SLAT 表示、两阶段 flow matching、多格式 decoder | 看过 [Diffusion 详解](../../生成模型/Diffusion详解.md) 和 [Flow Matching 详解](../../生成模型/Flow_Matching详解.md) |

后续待补：

- 02 · Hunyuan3D 系列（腾讯，2025）
- 03 · Direct3D-S2 与高分辨率结构化生成
- 04 · 场景级 3D 生成（Holodeck / PhyScene / Infinigen Indoors）
- 05 · 文生 3D 评测与 benchmark

## 三大流派全景

```
                    3D 生成
                       │
          ┌────────────┼────────────────┐
          ▼            ▼                ▼
      A. SDS 蒸馏    B. 直出 GS/Mesh    C. 结构化潜变量
       (慢/泛化好)   (快/质量受限)      (中速/质量好)

  DreamFusion       LGM                TRELLIS        ← 本专题
  DreamGaussian     GRM                Hunyuan3D-2    ← 本专题
  GaussianDreamer   GS-LRM             Direct3D-S2    ← 本专题
                                       Step1X-3D
       │                │                   │
       └─ [05-生成式高斯] ──────┘                   └─ 本专题
```

C 流派的共同设计：

1. **稀疏 3D 网格**作为骨架（结构）
2. **每个 active 单元挂稠密 latent**（语义/外观）
3. **多个轻量 decoder** 把同一 latent 翻译到不同表示
4. **flow matching / DDPM** 在 latent 上做生成

继承自 Stable Diffusion 的 latent + decoder 范式，但搬到了 3D。

## 时间线（C 流派）

```
2024-12  TRELLIS              SLAT 开山，1.1B/2B 开源
2025-01  Hunyuan3D            腾讯，物体级商业可用
2025-03  Direct3D-S2          256³ 高分辨率
2025-05  Hunyuan3D-2          多视图条件 + Mesh-PBR 联合生成
2025-07  Step1X-3D            阶跃星辰，首个开源 1M+ 资产规模
2025-09  TRELLIS-Distilled    社区蒸馏版本，30s → 3s
2026-02  Hunyuan3D-3          物体 + 简单场景
```

---

## 共同关键词速查

| 概念 | 说明 |
|------|------|
| **SLAT** | Structured LATents，稀疏体素 + 稠密局部 latent 的统一 3D 潜变量 |
| **结构-外观解耦** | sparse mask 编码 occupancy，per-voxel latent 编码语义 |
| **多格式 decoder** | 同一 latent → GS / Mesh / RF 三套独立 decoder |
| **Flow Matching** | 比 DDPM 训练稳，采样步数少，是 C 流派标配 |
| **多视图 DINOv2 lifting** | 编码端构造 latent 的标准方法：渲染多视角 → DINO 提特征 → 反投到 3D 体素 |
| **Sparse Transformer** | 仅在 active 体素上做注意力，避开 dense 3D 注意力的 O(N³) |

---

[返回 3D 视觉](../README.md) | [返回视觉](../../README.md) | [返回总目录](../../../README.md)

# 01 · TRELLIS 详解

TRELLIS 是 2024 年 12 月微软放出的工作，被普遍认为是把"3D 生成"从"绑死某个表示"中解放出来的转折点。它做的事一句话：**先学一个稀疏的、结构化的 3D 潜变量（SLAT），再用三个独立的 decoder 把同一个潜变量解码成 3D 高斯 / 辐射场 / Mesh**。这套设计在 2025 年被 Hunyuan3D、Direct3D-S2、Step1X-3D 等多家工作直接继承，事实上确立了一个新流派。

> **核心论文**：Xiang et al. *Structured 3D Latents for Scalable and Versatile 3D Generation*. arXiv:2412.01506, 2024-12.
> **代码**：[github.com/microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)（MIT License）
> **项目页**：[trellis3d.github.io](https://trellis3d.github.io)
> **作者机构**：Microsoft Research Asia + Tsinghua + USTC
> **发布权重**：TRELLIS-1.1B（image / text 两版本），TRELLIS-2B（更高质量）

---

## 一、研究背景：为什么需要 SLAT 这种表示

### 1.1 LGM / GS-LRM 这条线撞上的两堵墙

LGM、GRM、GS-LRM 把"前向一次出 3D 高斯"做到了 < 1 秒，看起来已经很好了。但 2024 年下半年陆续暴露出两个根本约束：

**第一堵墙：表示锁死。** 模型最后一层的输出 shape 是固定的（per-pixel 14 维 / per-patch K 个高斯），训完只能出高斯。想要 mesh 给游戏引擎用？得另起炉灶训一个 mesh-LRM。想要辐射场做半透明？再训一个 RF-LRM。每个表示一个模型，工程上不可持续。

**第二堵墙：几何质量受限。** 把高斯"平铺到像素"或"绑到 patch"是为了让 CNN/Transformer 能输出固定 shape 的张量，但这等价于把 3D 几何**强制压到 2D 网格的拓扑里**。复杂凹形、薄壳、内嵌结构在 2D 网格上根本表达不了——你能让"杯子的把手"对应到 image grid 的哪几个像素吗？

### 1.2 Stable Diffusion 给的启示

2D 图像生成的演进路径很清楚：

```
第一代 (DDPM):    在像素空间做扩散                           慢、糊
第二代 (LDM):     先用 VAE 压成低维 latent，再做扩散，再解码  快、清晰
```

Stable Diffusion 的核心 trick 不是 U-Net 也不是 attention，而是**把"生成"和"渲染"分到两个独立的网络**——latent diffusion 负责生成内容，VAE decoder 负责把 latent 翻译成像素。

TRELLIS 把这件事搬到了 3D：

```
第一代 (3DGS 直生): 直接生成高斯参数            ← LGM/GS-LRM
第二代 (SLAT):       先生成 3D latent，再 decode  ← TRELLIS
```

二者结构上是同构的，只是把"VAE decoder 翻译到像素"换成"3D decoder 翻译到 GS/RF/Mesh"。

### 1.3 为什么是"结构化"潜变量

直接搬 SD 的 latent 设计在 3D 里会撞另一堵墙：**3D 是稀疏的**。一个椅子的有意义内容只占其包围盒的几个百分点，剩下都是空气。如果把整个 64³ 网格都当 dense latent 做扩散，99% 计算花在空气上。

TRELLIS 的回答：把 latent 拆成两部分——

- **结构 (sparse mask)**：哪些体素是 active（有内容）
- **外观 (per-voxel latent)**：active 体素里挂的稠密特征向量

这就是"**结构化** 3D 潜变量"中"结构化"的含义——把"在哪里有"和"在那里是什么"显式分开。两件事用两个扩散模型分别学，各自模型只关心自己的事，FLOPs 也被稀疏度天然压住。

---

## 二、SLAT 表示的构造

### 2.1 形式定义

一个 SLAT 表示 $\mathcal{S}$ 由两部分组成：

$$\mathcal{S} = \{(p_i, z_i)\}_{i=1}^{N}$$

- $p_i \in \{0, 1, \dots, 63\}^3$：第 $i$ 个 active 体素在 64³ 网格里的整数坐标
- $z_i \in \mathbb{R}^d$：挂在该体素上的稠密 latent，论文中 $d = 8$（这是 sparse VAE 压缩后的维度，原始 DINOv2 特征是 1024 维）
- $N$ 是 active 体素总数，典型值 **30,000-50,000**

整个 SLAT 表示一个 3D 资产用约 30k × 8 = 240k 个浮点数，比一个 ply 高斯文件（百万级浮点）小一个量级。

### 2.2 编码端：从 3D 资产到 SLAT

训练时需要给每个 3D 资产构造它的 ground-truth SLAT。流程：

```
3D Asset (mesh/GS/...)
   │
   ▼ 渲染
150 张多视角图像 (RGB)
   │
   ▼ DINOv2 / DINOv2-Reg
每张图 → patch tokens [16x16 个 1024 维向量]
   │
   ▼ 多视图反投到 3D 体素
   每个 voxel 收集所有"看见它"的视角的 patch features
   按可见性加权平均
   │
   ▼ Sparse VAE Encoder
{(p_i, f_i)} 大维度 → {(p_i, z_i)} d=8
   │
   ▼
SLAT
```

**几个工程细节**：

1. **为什么是 DINOv2 而不是 CLIP**：DINOv2 特征更细粒度，patch token 之间空间一致性强。CLIP 偏全局语义，反投到 3D 时同一物体不同视角的特征不一致。
2. **150 视角是怎么定的**：作者做了消融——50 视角时 SLAT 重建质量明显下降，150+ 边际收益变小，所以 150 是经验最优。
3. **反投权重是什么**：基于深度图——只有最先击中体素的射线（visible surface）才贡献特征，被遮挡的视角不算。这避免内部体素被外表面特征污染。
4. **Active 体素怎么选**：从 mesh 的 occupancy 直接转。后期 fine-tune 时也允许随机扰动一部分边界体素来增强鲁棒性。

### 2.3 Sparse VAE：压缩到低维

$f_i$（DINOv2 反投得到的 1024 维）到 $z_i$（8 维）这步是一个**稀疏卷积 VAE** 完成的：

- Encoder：在 active 体素的稀疏图上做几层 3D sparse convolution（[3D稀疏卷积](../3D稀疏卷积.md) 那套）
- 中间瓶颈：8 维
- Decoder：对称的 sparse conv，重建回 1024 维

VAE 的训练损失是**重建损失 + 各 decoder 的渲染损失**联合。也就是说，z_i 的 8 维空间不是凭空压出来的，而是要保证 GS/RF/Mesh 三个 decoder 都能从中解码出像样的输出。这让 latent 空间承载所有 decoder 共同需要的信息，避免某个 decoder 偏废。

### 2.4 为什么 8 维就够

直觉上 1024 → 8 看起来太狠。但要注意：
- 体素位置 $p_i$ 已经显式给出，不需要 latent 编码"我在哪"
- $z_i$ 只需要编码"在这个位置应该长什么"——颜色、法线、局部几何细节
- 局部信息其实 8 维充足，复杂内容靠相邻体素的 latent 协同表达

对比 Stable Diffusion 的 latent 是 4 通道 × 64×64 = 16384 维表示一张图的 high-level 信息，TRELLIS 的 30k × 8 = 240k 维表示一个 3D 物体，量级合理。

---

## 三、两阶段 Flow Matching 生成

有了 SLAT 表示后，生成等价于"采样一个 SLAT"。但 SLAT 是变长的（active 体素数量不固定），不能直接套现成扩散框架。TRELLIS 的解法是**两阶段串行**：

```
              条件 c (image / text embedding)
                       │
            ┌──────────┴──────────┐
            │                     │
            ▼                     ▼
      ┌─────────────┐      ┌──────────────┐
      │  Stage 1    │      │  Stage 2     │
      │ Sparse      │ ───▶ │ Structured   │
      │ Structure   │      │ Latent       │
      │ Flow Model  │      │ Flow Model   │
      └─────────────┘      └──────────────┘
            │                     │
            ▼                     ▼
       active mask         per-voxel latent
       (64³ → ~30k)            ({z_i})
            │                     │
            └─────────┬───────────┘
                      ▼
              SLAT representation
                      │
            ┌─────────┼─────────┐
            ▼         ▼         ▼
        GS Dec     RF Dec    Mesh Dec
            │         │         │
            ▼         ▼         ▼
       3D Gaussians  RF      Mesh
```

### 3.1 Stage 1：Sparse Structure Flow Model (SSFM)

**目标**：从条件 $c$ 采样一个 64³ 的 occupancy 网格（每个体素 active or not）。

**输入**：噪声 $x_0 \in \mathbb{R}^{1 \times 64 \times 64 \times 64}$ + 条件 $c$。
**输出**：去噪后的 occupancy logit $x_1$，sigmoid + threshold 得到 active mask。

**网络**：3D U-Net（约 300M 参数），与 Stable Diffusion U-Net 同构但 3D 版本。条件 $c$ 通过 cross-attention 注入。

**为什么用 dense 3D 网格而不是稀疏**：Stage 1 要从"全是噪声"开始，没有先验稀疏结构。一开始就 dense，每一步去噪逐渐发现哪些体素是 active。最后一步才得到稀疏 mask。

**Flow matching 训练目标**：

$$\mathcal{L}_{\text{SSFM}} = \mathbb{E}_{t, x_0, x_1, c} \big[\| u_\theta(x_t, t, c) - (x_1 - x_0) \|^2\big]$$

其中 $x_t = (1-t) x_0 + t x_1$ 是从噪声 $x_0$ 到目标 $x_1$ 的线性插值，$u_\theta$ 学预测速度场。

### 3.2 Stage 2：Structured Latent Flow Model (SLFM)

**目标**：在 Stage 1 给出的 active 体素集合上，生成每个体素的 8 维 latent $z_i$。

**网络架构**：**Sparse Transformer**——只在 active 体素上做 self-attention 和 cross-attention。

```
输入: {(p_i, z_i^t)}  [N 个 active 体素，每个含位置 + 当前噪声 latent]
       │
       ▼ Position Encoding (3D RoPE 或 absolute)
       │
       ▼ Transformer Blocks × L
   ┌───────────────────────────┐
   │ Self-Attn (体素之间)       │
   │ Cross-Attn (与条件 c)     │
   │ FFN                        │
   └───────────────────────────┘
       │
       ▼ 输出层
   预测 velocity field 用于 flow matching
```

**注意力计算量**：N=30k 时，self-attention 的复杂度是 $O(N^2 \cdot d) = O(9 \times 10^8 \cdot 8)$，比 dense 64³ 全连接 (262k²) 小三个数量级。这是稀疏化的最大收益。

**模型规模**：
- TRELLIS-1.1B：~24 层 Sparse Transformer，hidden dim 1024
- TRELLIS-2B：~28 层，hidden dim 1408

### 3.3 为什么是 Flow Matching 而不是 DDPM

论文做了消融对比，flow matching 优势在：

1. **训练稳定**：直接学速度场，没有需要调度的 noise schedule
2. **采样步数少**：25 步即可达到 DDPM 100 步的质量
3. **数学清爽**：与 [Flow Matching 详解](../../生成模型/Flow_Matching详解.md) 那条线一致，符合 2024 年的整体趋势（SD3、FLUX 都转向 flow matching）

### 3.4 条件机制

Image conditioning：
- 输入图像过 DINOv2 提 patch tokens（16×16 个）+ 全局 CLIP feature
- 拼起来作为 cross-attention 的 KV
- 训练时随机 drop 条件 → 支持 classifier-free guidance

Text conditioning：
- T5-XXL embedding 作为 cross-attn KV
- 文本版本质量比 image 版本弱，作者认为是数据规模问题（带文本 caption 的高质 3D 资产稀缺）

CFG scale 通常设 7.5（image 条件）或 12（text 条件），与 SD 类似。

---

## 四、多格式 Decoder 家族

SLAT 一旦生成出来，剩下就是把它翻译成各种 3D 表示。三个 decoder 共享同一个 SLAT 输入空间，各自独立训练。

### 4.1 3DGS Decoder

每个 active 体素解码出 $K=32$ 个高斯。每个高斯参数：

```
3 维 位置 offset (相对体素中心)
3 维 scale (log-space)
4 维 rotation (quaternion)
1 维 opacity
48 维 SH coefficient (degree 3)
─────
共 59 维 × 32 高斯 = 1888 维 / 体素
```

整个物体：30k 体素 × 32 = ~1M 高斯，与典型重建场景的高斯数量级一致。

**Decoder 网络**：sparse conv → MLP，把 8 维 z_i 解到 1888 维。监督用渲染损失（多视角 L1 + LPIPS）。

### 4.2 Radiance Field Decoder (Strivec-like)

每个体素挂一个低秩张量分量（rank-K outer product），整个物体的辐射场是稀疏张量场的加和。优势：

- 体积渲染效果好（半透明、毛发、烟雾）
- 比纯 NeRF 快很多（因为是显式张量）
- 与 Strivec / TensoRF / Plenoxels 那条路兼容

### 4.3 Mesh Decoder (Flexicubes)

Flexicubes 是 deep marching cubes 的可微变体——每个体素学一个 SDF 值 + 拓扑 flag，最后用可微 marching cubes 抽出 mesh。

输出：约 100k 顶点的 mesh + 顶点颜色 / UV 贴图。

工程价值最高：直接给 Blender / Unity / Unreal，不需要再做 mesh extraction。

### 4.4 Decoder 之间的关系

三个 decoder **完全独立**——可以只训其中一个，也可以全训。训练时：

1. 先固定 SLAT encoder，训三个 decoder（每个用对应表示的渲染损失监督）
2. 训完后，SLAT 空间被三种监督共同形塑——任意一个 latent 都能解码到三种格式

**生产里的常见组合**：
- 内容生成 demo：三个 decoder 都跑，让用户挑
- 游戏资产管线：只用 Mesh decoder
- 影视特效：用 RF + GS（半透明用 RF，硬表面用 GS）

---

## 五、训练细节

### 5.1 数据组合

| 数据集 | 规模 | 来源 | 特点 |
|--------|------|------|------|
| Objaverse-XL（过滤后） | ~250K | 互联网爬取 | 多样性最高，质量参差 |
| ABO | 8K | Amazon 商品 | 干净、风格统一 |
| 3D-FUTURE | 16K | 家具 | 室内场景资产 |
| HSSD | ~8K | Habitat 合成室内 | 房间级 |
| **总计** | **~500K** | | |

**过滤策略**（关键）：
1. **美学评分**：渲染一张图过 aesthetic predictor，低于阈值剔除
2. **多视角一致性**：检查不同视角渲染是否合理
3. **几何有效性**：剔除 broken mesh、无水密、纹理崩坏的
4. **重复检测**：CLIP 特征近邻去重

过滤前 Objaverse-XL 是 ~10M，过滤到 ~250K，留存率 2.5%。

### 5.2 训练阶段

```
Phase 1: SLAT VAE 训练
  - 输入: 渲染的多视角图
  - 编码: DINOv2 提特征 + 反投 + sparse VAE
  - 解码: 三个 decoder 同时训（GS/RF/Mesh）
  - 损失: 重建 + 渲染 + KL
  - 时长: ~2 周 × 64 卡

Phase 2: SSFM (Stage 1) 训练
  - 输入: GT active mask
  - 训练 flow matching
  - 时长: ~1 周 × 32 卡

Phase 3: SLFM (Stage 2) 训练
  - 输入: GT SLAT (active 体素 + latent)
  - 训练 flow matching
  - 时长: ~3-4 周 × 64-128 卡（取决于 1.1B 还是 2B）
```

### 5.3 算力总量

论文披露大致是几千 GPU-day（A100），按公开 GPU 价折算约 50 万美元级。这是一个学术友好的训练规模——不像 LLM 那样动辄千万美元起。

---

## 六、实验结果

### 6.1 单图生成 3D 物体

vs LGM / CRM / InstantMesh / 3DTopia / Shap-E / OpenLRM：

| 方法 | F-Score (geom) ↑ | Chamfer ↓ | CLIP-Sim ↑ | 推理时间 |
|------|-----------------|-----------|------------|----------|
| Shap-E | 0.45 | 0.082 | 0.71 | 13 s |
| OpenLRM | 0.62 | 0.045 | 0.78 | 1 s |
| LGM | 0.74 | 0.038 | 0.81 | 5 s |
| InstantMesh | 0.78 | 0.032 | 0.82 | 10 s |
| **TRELLIS-1.1B** | **0.85** | **0.018** | **0.85** | 30 s |
| **TRELLIS-2B** | **0.88** | **0.015** | **0.87** | 45 s |

（数字按论文 Table 1 整理，略有简化）

**几何质量是 TRELLIS 最大的领先点**——F-Score 拉开 +7 个点，Chamfer 几乎砍半。

### 6.2 文生 3D

用 T3Bench 评测：CLIP score、user study。TRELLIS 落后于 SDS 路线（DreamGaussian），但**比所有 feed-forward 路线都好**，且速度快 3-5×。这条对比有意思：SDS 慢但泛化好，TRELLIS 中速且质量结构性好。

### 6.3 用户研究

让用户在 LGM / InstantMesh / TRELLIS 三选一选偏好，TRELLIS 胜率 **70%+**——视觉质量优势在主观感受上很明显。

### 6.4 多格式输出一致性

同一个 SLAT 解码出的 GS / RF / Mesh 三种表示，在视觉上互相 Match（PSNR > 28 互相之间）。这意味着用户不会看到"GS 看着是个杯子，Mesh 看着是个碗"这种混乱情况。

---

## 七、TRELLIS 的影响：2025 年的 SLAT 流派

TRELLIS 开源后，2025 年涌现出一批继承其思路的工作：

### 7.1 Hunyuan3D 系列（腾讯）

- **Hunyuan3D-1**（2025-01）：商业可用版本，闭源 API，质量与 TRELLIS-2B 相当
- **Hunyuan3D-2**（2025-05）：开源 1.0B/2.0B，加 PBR 材质生成（不只是 albedo + 几何，还有 metallic/roughness/normal）
- 核心架构与 TRELLIS 高度相似——SLAT 风格的 latent + 多 decoder。

### 7.2 Direct3D-S2

把 SLAT 分辨率从 64³ 推到 256³，用**多尺度稀疏 transformer** 解决计算量爆炸。关键 trick：粗尺度先生成 occupancy，细尺度只在 coarse-active 的邻域内做。

### 7.3 Step1X-3D（阶跃星辰）

首个公开声明用 1M+ 资产规模训练的开源 3D 生成模型。架构基本沿用 TRELLIS-Hunyuan 这条线。

### 7.4 后续优化方向

- **加速**：Distillation / Consistency 让 30 s 降到 3-5 s
- **可控性**：把 ControlNet 思路迁移到 SLAT 空间（用条件输入控制特定区域）
- **场景级**：把 TRELLIS 拼成 scene generator（Holodeck、PhyScene 是另一支）

---

## 八、应用与生产实践

### 8.1 内容生产

- **游戏 / 影视**：单图 → mesh，30 秒一个资产，比手工建模快 100×。多家游戏公司 2025 年起把 TRELLIS / Hunyuan3D 接入资产 pipeline。
- **AR / VR**：用户拍一张照 → 生成 3D 模型放进场景。Apple Vision Pro / Meta Quest 上已有第三方 app 接入。
- **电商**：商品照片 → 3D 展示模型。

### 8.2 机器人 / 仿真

TRELLIS 生成的 mesh 直接喂给 IsaacSim / MuJoCo 做物理仿真。Mesh decoder 的存在让这一步不需要额外的 GS-to-mesh 转换。**这点对 [高斯+具身](../高斯溅射/08-高斯+具身.md) 那条线意义重大**——sim 资产生成的瓶颈被打通了。

### 8.3 与 SDS 的互补

TRELLIS 推 SOTA 在物体级 single-image，但**罕见类别 / 艺术风格**还是 SDS 路线（DreamGaussian + 大型 2D diffusion）更好。生产管线常见做法：

```
常见物体 → TRELLIS（30 s，质量好）
罕见物体 / 强风格化 → DreamGaussian（2 min，泛化好）
```

---

## 九、局限与开放问题

### 9.1 分辨率瓶颈

64³ 网格在小细节（戒指上的雕花、复杂机械结构）上不够。Direct3D-S2 推到 256³ 但计算量暴增 4×。**根本解法可能是层次化 SLAT**——粗尺度全局 + 细尺度局部，类似 NeRF 的多分辨率 hash grid。

### 9.2 推理时间

30 秒在生产环境可以接受，但在交互应用（用户改 prompt 即时预览）仍是瓶颈。蒸馏到 3-5 秒是 2025-2026 的活跃方向。

### 9.3 场景级生成无能

TRELLIS 训练数据是物体级，对房间、街道、室外场景生成不行。场景级生成的瓶颈不在生成模型本身，而在**物体之间的关系建模**——这需要另一套设计（Holodeck 的 LLM 拼接、PhyScene 的 diffusion 场景图等）。

### 9.4 物理属性缺失

生成的资产只有视觉外观，没有质量、摩擦、弹性等物理参数。要进 robotics 仿真还得另一道流程（参考 [PhysGaussian / PhysDreamer](../高斯溅射/06-高斯+物理仿真.md)）。

### 9.5 可控性偏弱

相比 SDS 路线可以挂任意 ControlNet（深度图、法线图、姿态），TRELLIS 的条件接口仍偏单一。"我想要一个红色的杯子，把手在左边"——TRELLIS 没法精确控制把手方向。

### 9.6 多 Object 组合

输入两张照片要"组合"成一个场景目前不行。SLAT 表示是 single object 的，多物体合成需要多次推理 + 后处理拼接。

---

## 十、面试高频问题

### Q1: TRELLIS 和 LGM 最本质的区别是什么？

**答**：表示空间不同。LGM 直接学高斯参数（输出空间被高斯参数化锁死），TRELLIS 学一个**与表示无关的结构化 3D 潜变量** SLAT（稀疏体素 + 稠密局部 latent），再用三个独立 decoder 翻译到 GS/RF/Mesh。这等价于把 Stable Diffusion 的"latent + VAE decoder"思路从 2D 搬到 3D，结果是：(1) 一次推理出三种格式，(2) latent 不被表示约束，几何质量上一台阶，(3) 后续优化（加速、控制）可以分别针对生成器或 decoder 做。

### Q2: 为什么 TRELLIS 比 LGM 慢一个量级还是被广泛采用？

**答**：(1) **30 秒在内容生产场景可接受**——一个游戏建模师手工做一个资产要几小时甚至几天，TRELLIS 30 秒已经是 100× 加速；(2) **多格式输出是工程刚需**——LGM 只能出 GS，要 mesh 得另起一套模型，生产管线吃不消；(3) **几何质量结构性更好**——F-Score +7、Chamfer 砍半，下游用得上的资产比例显著高；(4) **开源生态成熟**——MIT 协议、1.1B/2B 双权重、社区蒸馏版，进入门槛低。

### Q3: SLAT 的 8 维 latent 怎么够用？信息从哪来？

**答**：要拆开看 SLAT 的两部分：(a) 稀疏体素位置 $p_i$ 已经显式编码了"我在哪、占多大空间"——几何骨架是显式的，不需要 latent 编码；(b) per-voxel latent 只编码"在这个位置应该长什么"——颜色、法线、局部细节。8 维确实压缩很狠，但要注意 sparse VAE 训练时是用三个 decoder 的渲染损失联合监督的，被反复拉扯过的 latent 空间承载效率很高。论文里有消融，d=4 质量明显下降，d=16 边际收益变小，d=8 是经验最优。

### Q4: 为什么要做两阶段（先 mask 再 latent），不能一阶段端到端吗？

**答**：可以，但效率差。如果一阶段直接在 dense 64³ × 8d 张量上做扩散，每步要处理 262k × 8 = 2M 维度，99% 计算花在空气体素上。两阶段的好处是：(1) Stage 1 只关心 binary occupancy，网络小、收敛快；(2) Stage 2 只在已知是 active 的 30k 体素上做 sparse transformer，FLOPs 比 dense 小三个数量级。代价是不能联合优化两阶段，但实验显示性能上没损失。

### Q5: TRELLIS 为什么用 DINOv2 而不是 CLIP 做 lifting？

**答**：DINOv2 的 patch token 比 CLIP 更适合空间反投。具体说：(1) **细粒度**：DINOv2 patch 编码局部纹理 / 法线 / 几何细节，CLIP 偏全局语义，反投到 3D 体素粒度太粗；(2) **多视图一致性**：同一物体不同视角，DINOv2 patch 特征更接近（自监督 + masked image modeling 的训练方式更"局部"），CLIP 的视觉特征偏抽象，不同视角差异大，反投后特征噪声大；(3) **DINOv2 训练数据规模 + 自监督质量**让它在"无类别先验的 dense 特征提取"上事实上 SOTA。CLIP 反而保留作为**全局条件 embedding** 的来源——细节看 DINOv2，语义看 CLIP，分工明确。

### Q6: 如果让你扩展 TRELLIS 到 4D（动态物体），你会怎么做？

**答**：最朴素的做法是把 SLAT 推广到 4D 体素 (64³ × T)，但 T 可能 30+，计算量爆炸。两个更实际的方向：(1) **基础 SLAT + 形变场**：保留 64³ × 8 的 SLAT 作为静态参考，加一个 deformation field decoder 输出每帧的体素位移（参考 [03 篇](../高斯溅射/03-动态与4DGS.md) 的 Deformable 3DGS）；(2) **分层 SLAT**：粗尺度时序 latent + 细尺度空间 latent，时序部分小，空间部分稀疏。学界还在探索，2026 已有一些早期工作（DreamGaussian4D、Animate3D 那条线）但都不太成熟。这是本流派最大的开放问题之一。

---

## 参考资源

- 论文：[arxiv.org/abs/2412.01506](https://arxiv.org/abs/2412.01506)
- 代码：[github.com/microsoft/TRELLIS](https://github.com/microsoft/TRELLIS)
- 项目页：[trellis3d.github.io](https://trellis3d.github.io)
- HuggingFace 在线试玩：搜 "TRELLIS demo"
- 相关：[Hunyuan3D-2 GitHub](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)

---

[← 专题导航](README.md) | [返回 3D 视觉](../README.md)

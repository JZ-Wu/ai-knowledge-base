# VLM 关键技术

深入讲解 VLM 中的关键技术问题：高分辨率处理、视觉 Token 压缩、多图/视频理解、视觉定位。

## 一、高分辨率处理

### 1.1 为什么需要高分辨率？

早期 VLM 使用 224px 或 336px 的固定分辨率，但这对很多实际任务是不够的：

```
场景            需要的信息         224px 能看到吗？
─────────────────────────────────────────────────
文档 OCR         小号文字            不能（模糊）
表格理解         细密的行列线        不能（看不清）
远景场景         远处的小物体        不能（太小）
高分辨率海报     细节纹理            丢失严重
代码截图         代码文字            完全看不清
```

**核心矛盾**：ViT 的计算量与输入分辨率的平方成正比，直接提高分辨率代价太大。

### 1.2 方案一：直接放大 + 位置编码插值

最简单的方法——让 ViT 接受更大的输入：

```
ViT 预训练: 224×224, 位置编码 16×16 = 256 个
    │
    ↓ 推理时改为 336×336
    │
位置编码插值: 16×16 → 24×24 = 576 个
  (双线性插值 2D 位置编码)
    │
    ↓ 微调几个 epoch 适应新分辨率
```

**代表**：LLaVA-1.5（224 → 336），简单有效但提升有限。

**局限**：
- 分辨率不能太大（ViT 计算量与 token 数成平方关系）
- 336px 对于文档 OCR 仍然不够

### 1.3 方案二：动态切片（Dynamic Tiling）

当前最主流的高分辨率方案：

```
输入: 高分辨率图像 (如 1344×896)
    │
    ├── Step 1: 生成全局缩略图
    │   └── resize 到 448×448 → 编码 → 全局语义特征
    │
    ├── Step 2: 计算最优切分方案
    │   └── 根据图像宽高比，选择 tile 排列方式
    │       如 3×2, 2×3, 4×2 等（预定义候选方案列表）
    │
    ├── Step 3: 切成 tile
    │   └── 1344×896 → 3 列 × 2 行 = 6 个 448×448 tile
    │
    └── Step 4: 分别编码
        ├── 全局缩略图 → ViT → 全局 token
        ├── tile 1 → ViT → 局部 token
        ├── tile 2 → ViT → 局部 token
        └── ... (共 6 个 tile)
        │
        ↓ 拼接所有 token → 投影 → 送入 LLM
```

**代表模型**：InternVL2、LLaVA-NeXT、Monkey

**详细的 tile 选择逻辑**：

```python
# 预定义候选分辨率列表 (InternVL2 风格)
CANDIDATE_RESOLUTIONS = [
    (448, 448),   # 1 tile
    (448, 896),   # 1×2
    (896, 448),   # 2×1
    (896, 896),   # 2×2
    (448, 1344),  # 1×3
    (1344, 448),  # 3×1
    (896, 1344),  # 2×3
    (1344, 896),  # 3×2
    # ... 最多到一定上限
]

def select_best_resolution(image_size, candidates):
    """选择最匹配的分辨率（尽量不浪费像素、不过度缩放）"""
    best = min(candidates, key=lambda r:
        abs(r[0]/r[1] - image_size[0]/image_size[1])  # 宽高比匹配
        + padding_waste(image_size, r)                  # 填充浪费最少
    )
    return best
```

### 1.4 方案三：多尺度处理

同时保留全局语义和局部细节：

```
原始图像
    │
    ├── 低分辨率 (224px) → ViT → 全局语义 token
    │
    └── 高分辨率 (如 1344px) → 局部裁剪/切片
        └── 每个裁剪区域 → ViT → 局部细节 token
```

**代表**：
- **Monkey**：在切片基础上加 sliding window
- **LLaVA-UHD**：自适应图像分区 + 压缩

### 1.5 方案四：原生高分辨率 ViT（Qwen-VL2 方案）

```
不切 tile，直接将原始分辨率图像送入 ViT:
  - ViT 使用 2D-RoPE 位置编码（可外推到任意分辨率）
  - 输出 token 数 = (H/14) × (W/14)
  - 不存在 tile 边界切割问题

优势:
  ✓ 物体不会被 tile 边界截断
  ✓ 保留原始空间关系

劣势:
  ✗ 超高分辨率时 token 数爆炸
  ✗ ViT 的注意力计算量与 token² 成正比
```

### 1.6 方案对比

| 方案 | 优势 | 劣势 | 代表 |
|------|------|------|------|
| 直接放大 | 简单 | 分辨率有限 | LLaVA-1.5 |
| 动态切片 | 灵活、高效 | tile 边界可能截断物体 | InternVL2 |
| 多尺度 | 全局+局部兼顾 | 实现复杂 | Monkey |
| 原生高分辨率 | 无截断问题 | 计算量大 | Qwen-VL2 |

---

## 二、视觉 Token 压缩

### 2.1 问题

高分辨率带来了更多视觉 token，直接影响 LLM 推理效率：

```
场景                         视觉 token 数    占 context 比例
────────────────────────────────────────────────────────────
224px 单图                    256              很小
336px 单图                    576              可接受
动态切片 6 tile + 全局        7 × 256 = 1792   显著
动态切片 12 tile + 全局       13 × 256 = 3328  很大
视频 32 帧 × 256             8192             爆炸
```

视觉 token 太多的问题：
1. **推理变慢**：LLM 注意力计算是 O(N²)
2. **占用 context**：挤掉了文本 token 的空间
3. **显存增大**：KV cache 膨胀

### 2.2 Pixel Shuffle 下采样

将空间相邻的 patch 特征合并：

```
输入特征图: H × W × C (如 24×24×1024)

Pixel Shuffle (factor=2):
  将每 2×2 的相邻 patch 合并到通道维度
  24×24×1024 → 12×12×4096

  具体操作:
  patch(0,0), patch(0,1), patch(1,0), patch(1,1) → concat → 1个新patch

  token 数: 576 → 144 (减少 4 倍)
  每个 token 维度: 1024 → 4096 (增加 4 倍，信息保留)
```

**优势**：简单高效，保留空间信息，被 InternVL2 广泛使用。

### 2.3 Q-Former / Perceiver 压缩

用注意力机制将大量视觉 token 压缩到固定数量：

```
输入: 576 个视觉 token
  → Cross-Attention (32 个 query attend to 576 个 token)
  → 输出: 32 个压缩 token

压缩比: 576/32 = 18 倍
```

**劣势**：压缩比太高时丢失细节，尤其影响 OCR 和小物体识别。

### 2.4 Token 剪枝（Token Pruning）

动态识别并丢弃不重要的视觉 token：

```
方法 1: 基于注意力分数
  - 在 LLM 的浅层计算文本 token 对视觉 token 的注意力
  - 丢弃注意力分数最低的视觉 token（"没人看的 token"）
  - 代表: FastV

方法 2: 基于相似度
  - 丢弃与其他 token 高度相似的 token（冗余信息）
  - 保留信息量大的、独特的 token

优势: 自适应压缩，重要信息保留更好
劣势: 需要额外的判断逻辑
```

### 2.5 Token 合并（Token Merging, ToMe）

合并相似的视觉 token，而不是丢弃：

```
步骤:
1. 计算所有视觉 token 之间的相似度
2. 找到最相似的 token 对
3. 将它们合并（平均或加权）
4. 重复直到达到目标数量

特点:
  - 不丢弃信息，而是合并
  - 相比剪枝更"温和"
  - 但合并后的 token 可能损失空间精度
```

### 2.6 压缩策略对比

| 方法 | 压缩比 | 信息损失 | 是否自适应 | 额外计算 |
|------|--------|---------|-----------|---------|
| Pixel Shuffle | 4× | 低 | 否（固定） | 极低 |
| Q-Former | 10~20× | 中~高 | 否（固定） | 中 |
| Token 剪枝 | 可变 | 低~中 | 是 | 低 |
| Token 合并 | 可变 | 低 | 是 | 中 |

---

## 三、多图理解

### 3.1 多图输入的格式

```
方式 1: 拼接 + 分隔符 (最常见)
  [<image_1>] 视觉token_1 [</image_1>] [<image_2>] 视觉token_2 [</image_2>] 文本

方式 2: 交错式 (Flamingo 风格)
  这是第一张图 [<image>] 视觉token [</image>]，
  这是第二张图 [<image>] 视觉token [</image>]，
  请比较它们的区别。

方式 3: 拼接成一张大图
  将多张图拼接成网格图（如 2×2），当作一张图处理
  简单但可能丢失图间独立性
```

### 3.2 多图理解的挑战

```
1. Token 数爆炸:
   4 张高分辨率图 × 每张 1792 token = 7168 视觉 token
   → 需要更激进的压缩

2. 图间关系建模:
   - 哪些图相关，哪些不相关？
   - 如何建模图与图之间的时序/空间/逻辑关系？

3. 图的定位和引用:
   - "第二张图中的那个人" → 模型需要正确定位到第 2 张图
   - 需要清晰的图像标识符
```

---

## 四、视频理解

### 4.1 视频处理流程

```
输入视频 (如 30fps, 60 秒)
    │
    ├── Step 1: 帧采样
    │   └── 均匀采样 N 帧 (通常 8~32 帧，长视频可能 64~128 帧)
    │       为什么不是每帧都用？ → token 数太多，且相邻帧高度相似
    │
    ├── Step 2: 逐帧编码
    │   └── 每帧独立通过 ViT → N 组视觉特征
    │
    ├── Step 3: 时序信息注入
    │   └── 加入帧级别的位置编码 / 时间戳 token
    │
    ├── Step 4: 视觉 token 压缩 (关键！)
    │   └── 32 帧 × 256 token = 8192 → 必须压缩
    │
    └── Step 5: 送入 LLM
        └── 压缩后的视频 token + 文本 query → 回答
```

### 4.2 视频 Token 压缩策略

```
策略 1: 帧级压缩
  每帧独立压缩 (如 256 → 64 token)
  32 帧 × 64 = 2048 token (可接受)

策略 2: 时序合并
  相邻帧的相似 token 合并
  利用视频的时间冗余性
  → 有效减少重复信息

策略 3: 关键帧选择
  不是均匀采样，而是选择"变化最大"的帧
  → 对于长视频特别有效

策略 4: 分层处理
  先用轻量模型处理所有帧，生成摘要
  再用完整 VLM 处理关键帧 + 摘要
```

### 4.3 长视频理解

长视频（>1 分钟）是当前 VLM 的难点：

```
挑战:
  10 分钟视频, 30fps = 18000 帧
  即使采样 128 帧, 每帧 256 token = 32768 视觉 token
  → 远超大多数 LLM 的 context window

解决思路:
1. 更激进的采样 (如每 5-10 秒 1 帧)
2. 分段处理 + 记忆机制
3. 先看整体再看局部 (层级式理解)
4. 用更长 context 的 LLM (如 128K~1M context)
```

---

## 五、视觉定位与 Grounding

### 5.1 什么是视觉定位？

让 VLM 不只是描述图像，还能**指出具体位置**：

```
输入: 图像 + "找到图中的红色汽车"
输出: "红色汽车在图片左侧" + 坐标 [x1, y1, x2, y2]

输入: 图像 + 坐标 [x1, y1, x2, y2] + "这是什么？"
输出: "这是一辆红色的特斯拉 Model 3"
```

### 5.2 坐标表示方法

```
方法 1: 归一化坐标 (最常见)
  将坐标归一化到 [0, 1000] 范围
  输出: <box>(234, 156, 789, 634)</box>
  代表: Qwen-VL, InternVL2

方法 2: 文本描述坐标
  输出: "The car is at coordinates [0.23, 0.15, 0.79, 0.63]"
  简单但精度有限

方法 3: 特殊 token
  用特殊 token 表示坐标位
  输出: <loc_234><loc_156><loc_789><loc_634>
  代表: PaLI, Kosmos-2
```

### 5.3 训练定位能力

```
需要的数据:
  - 带 bbox 标注的目标检测数据 (如 COCO, Objects365)
  - 带 region description 的数据 (如 Visual Genome)
  - Referring expression 数据 (文字描述 → 定位)

训练格式示例:
  Q: "请找到图中所有的人"
  A: "图中有 3 个人: <box>(100,200,300,500)</box>, <box>(400,150,550,480)</box>, <box>(600,180,700,500)</box>"
```

---

## 六、VLM 整合传统 CV 能力

当前 VLM 的趋势是从"看图说话"走向**统一的视觉理解工具**，将过去需要多个专用模型完成的任务（OCR、检测、分割、深度估计、生成等）集成到一个模型中。

### 6.1 整合的核心思路

```
传统 CV: 每个任务一个专用模型
  OCR → CRNN/PaddleOCR
  检测 → YOLO/DETR
  分割 → Mask R-CNN/SAM
  深度 → MiDaS/DepthAnything

VLM 统一范式: 一个模型做所有任务
  核心思路: 把 CV 任务转化为"序列生成"问题
  ┌─────────────────────────────────────────┐
  │  图像 + 文本指令 → VLM → 文本/坐标/mask │
  └─────────────────────────────────────────┘
  不同任务只是"指令"和"输出格式"不同
```

整合方式分为三大类：

```
方式 1: 纯文本输出 (最简单)
  LLM 直接生成坐标、标签等文本
  适用于: 检测、OCR、关键点

方式 2: LLM + 专用解码头 (最常见)
  LLM 输出特殊 embedding → 接专用解码器
  适用于: 分割、深度估计等像素级任务

方式 3: 统一 tokenizer (最前沿)
  图像和文字都用 token 表示，生成图像 = 生成 token 序列
  适用于: 图像生成/编辑
```

**核心矛盾：LLM 只能输出文本 token，但 CV 任务需要各种格式的输出。** 每个任务的整合方式，本质上都是在回答同一个问题：**怎么把非文本的视觉输出塞进 LLM 的文本生成框架里？**

从"输出最容易文本化"到"最难文本化"排列：

```
检测/OCR/关键点 → 分割 → 深度 → 图像生成
  (纯文本即可)     (需要专用头)   (方向完全相反)
```

越靠左越容易直接用 LLM 输出文本解决，越靠右越需要专用模块或全新架构。

---

### 6.2 图像分割

让 VLM 根据自然语言描述输出像素级分割 mask。

```
传统分割: 固定类别（人、车、猫...），不理解自然语言
VLM 分割: "请把左边那个穿红衣服的人分割出来" → 精确的 mask

关键挑战:
  LLM 是序列生成模型，怎么输出像素级 mask？
  一张 224×224 的图，mask = 50176 个 0/1 像素值
  让 LLM 自回归生成 5 万个 token？→ 完全不现实
  → 答案: LLM 负责"理解你要分割什么"，SAM 负责"实际输出 mask"
  → 问题变成: LLM 怎么把它的理解传递给 SAM？
```

#### 代表模型：LISA（2023）

**完整数据流（按推理顺序）：**

```
Step 1: 图像编码
  图像 → CLIP ViT-H encoder → visual tokens
  visual tokens + 文本 tokens 拼接送入 LLM
  (到这一步和普通 VLM 完全一样)

Step 2: LLM 生成包含 <SEG> 的回复
  用户: "请分割图中的猫"
  LLM 自回归生成: "Sure, the cat is <SEG>."
  
  <SEG> 是新加入 vocabulary 的特殊 token
  和普通 token 一样有自己的 embedding，参与正常的 next-token prediction
  
  关键: <SEG> token 的 hidden state (最后一层输出向量)
  编码了 LLM 对"要分割什么"的全部理解
  → 包含视觉信息 (attention 过了 image tokens)
  → 包含语义信息 (看过了"猫"这个指令)

Step 3: hidden state → SAM prompt embedding
  <SEG> 的 hidden state ∈ R^4096 (LLaMA-7B 的 hidden dim)
  SAM mask decoder 期望的 prompt embedding ∈ R^256
  
  h_seg ∈ R^4096  →  MLP 投影层  →  p ∈ R^256
  
  类比理解:
    SAM 原来用点击坐标/bbox 编码成 prompt embedding → "分割哪里"
    LISA 用 LLM 的语义理解编码成 prompt embedding → "分割什么"
    功能相同，信息来源不同

Step 4: SAM mask decoder 输出 mask
  SAM mask decoder 接收两个输入:
    - SAM 自己的 ViT-H 产生的 image embedding
      (注意: 和 Step 1 的 CLIP ViT 是两个独立的 encoder)
    - Step 3 的 prompt embedding
  
  decoder 通过 cross-attention 让 prompt embedding 和 image embedding 交互
  → 输出像素级 mask
```

**训练设计（面试高频追问点）：**

```
训练什么？(冻结策略)
  冻结: CLIP ViT、SAM image encoder、SAM mask decoder
  只训练:
    - LLM 的 LoRA 参数
    - <SEG> token 的 embedding
    - hidden state → prompt embedding 的 MLP 投影层

Loss 设计 (两个 loss 联合训练):
  1. 文本 loss: 标准 cross-entropy autoregressive loss
     监督 LLM 生成正确的文本回复 (包括在正确位置生成 <SEG>)
  
  2. Mask loss: BCE loss + Dice loss
     监督 SAM decoder 输出的 mask 和 ground truth mask 的匹配

梯度反传路径 (端到端联合训练):
  mask loss → SAM decoder (frozen, 不更新) → MLP 投影层 (更新)
  → <SEG> token 的 hidden state → LLM 的 LoRA 参数 (更新)
  
  关键: mask loss 的梯度信号直接告诉 LLM
  "你的 <SEG> 编码得对不对"
  → 这就是 LLM 能学会在 <SEG> 的 hidden state 里
     编码正确空间指向信息的原因
```

#### 进阶：GLaMM、PixelLM

```
GLaMM: 支持多目标分割
  "分割图中所有的动物" → 多个 <SEG> token → 多个 mask
  
  关键问题: 模型怎么知道要生成几个 <SEG>？
  → LLM 自己决定——就像它决定一句话写几个词一样
  → 自回归过程中，LLM 根据图像内容和指令
     自然地生成对应数量的 <SEG> token
  → 每个 <SEG> 的 hidden state 独立投影，独立送入 SAM decoder

PixelLM: 无需 SAM
  直接在 LLM 后接轻量 mask decoder
  多个 <SEG> token → 独立解码 → 多个 mask
  优势: 去掉了 SAM 的依赖，更轻量
```

---

### 6.3 OCR 增强

```
方式 1: 外部 OCR 注入 (早期方案)
  先用 OCR 引擎 (PaddleOCR 等) 提取文字和坐标
  → 作为额外文本 token 拼接给 LLM
  代表: mPLUG-DocOwl, UReader
  
  问题:
    - OCR 引擎本身的错误会传播给 LLM
    - 两阶段 pipeline，不能端到端优化
    - 空间位置信息在文本拼接时容易丢失

方式 2: 端到端 OCR (主流方案)
  不依赖外部 OCR，VLM 本身直接识别文字
  关键: 高分辨率输入 + 大量 OCR 训练数据
  代表: Qwen2-VL, InternVL2
  
  为什么高分辨率很重要？
    - 文字笔画细节在低分辨率下会丢失
    - 224×224 的 ViT 输入对文档 OCR 完全不够
    - Qwen2-VL 支持动态分辨率 (最高可处理很大的图像)
    - InternVL2 用 dynamic high-resolution 方案:
      将高分辨率图像切分成多个 patch → 每个 patch 独立编码 → 拼接

方式 3: 专用 OCR VLM
  GOT-OCR 2.0 (2024):
    - 支持场景文字、文档、数学公式、乐谱、分子式
    - 输出结构化格式 (Markdown / LaTeX / HTML)
    - 支持区域级 OCR (指定区域识别)
    - 用 encoder-decoder 架构，专门为 OCR 优化
```

---

### 6.4 关键点检测

让 VLM 输出人体姿态、面部关键点等结构化坐标。

关键点检测和目标检测面临同一个核心问题：**怎么让 LLM 输出连续的坐标值？** LLM 是离散 token 生成器，不是回归器。有两条路线。

#### 路线一：坐标当文本直接生成（UniPose 思路）

最直觉的做法。把坐标归一化到 [0, 1000] 的整数，直接当文本 token 生成：

```
输入: 图像 + "Detect human pose keypoints"

LLM 自回归输出:
  "left_shoulder (312, 145) right_shoulder (456, 148) left_elbow (298, 267) ..."

每个数字 ("3", "1", "2") 都是普通的 text token
走标准的 next-token prediction
训练时用标准 cross-entropy loss，ground truth 就是坐标文本序列
```

**UniPose 的创新**：用 text prompt 指定关键点 schema（"detect 17 COCO keypoints" vs "detect 21 hand keypoints"），实现跨类别（人体、手部、面部、动物）统一。

```
优势:
  - 极度简单: 不需要任何架构改动，不需要新 decoder head
  - 纯文本输入输出，完全复用 VLM 的训练框架
  - 跨类别灵活: 通过 prompt 切换关键点类型

问题:
  1. 精度瓶颈
     坐标量化成整数 → 精度上限 1/1000
     人体姿态还行，面部关键点 (68 个点挤在很小区域) 就不够了
  
  2. 效率问题
     "312" 需要 3 个 token
     17 个 COCO 关键点 ≈ 17×(name + 2×3 coordinate + 分隔符) ≈ 上百 token
     自回归逐 token 生成，慢
  
  3. 结构信息丢失
     人体关键点之间有强结构约束 (左右肩对称，肘在肩腕之间)
     自回归文本生成是逐 token 的，不天然建模空间结构
```

#### 路线二：坐标离散化为特殊 token（Pose-GPT 思路）

核心想法：**把坐标空间离散化成 codebook，每个 bin 对应一个新加入 vocabulary 的 special token。**

```
做法:
  x 轴和 y 轴各离散化为 N 个 bin (如 N=512)
  坐标 (x, y) 对应两个 token: <x_312> <y_145>
  这些 token 和 <SEG> 一样，是加入 vocabulary 的特殊 token
  有自己的 learnable embedding

LLM 输出序列:
  <left_shoulder> <x_312> <y_145> <right_shoulder> <x_456> <y_148> ...

vs 文本坐标的优势:
  1. 更紧凑: 一个坐标 = 2 个 token (vs 6+ 个字符 token)
     → 生成更快
  2. Learned embedding: 每个 coordinate token 的 embedding 是学出来的
     → 能学到比 raw 数字更丰富的空间语义
  3. 训练: loss 还是 cross-entropy，但监督信号是坐标 token 序列
     → special token 的 embedding 通过反传学到正确的空间含义
```

---

### 6.5 深度估计

先明确深度估计和前面几个任务的**本质区别**：

```
关键点: 稀疏输出 → 17 个点，34 个坐标值 → 完全可以文本化
分割 mask: 稠密但二值 → 50176 个 0/1 → 不能文本化，但 SAM decoder 可以一次性输出
深度图: 稠密且连续 → 50176 个浮点数 → 不能文本化，需要回归，loss 和 decoder 都更复杂

→ 深度估计是最"难塞进 LLM 框架"的传统 CV 任务
→ 没有人真的让 LLM 直接逐像素输出深度值
→ 所有方案都是某种程度的"妥协"
```

#### 方案 1: VLM 引导 + 专用深度头（VPD 为例）

VPD 的思路：**用文本描述作为条件，引导 pre-trained Stable Diffusion 的中间特征来做深度估计。**

```
数据流:
  图像 → SD encoder → 多层特征 F1, F2, F3, F4
  文本描述 "a room with a sofa and a table"
    → CLIP text encoder → text embedding
  
  text embedding 通过 SD 的 cross-attention 调制 F1-F4
  → 调制后的特征包含语义引导的空间信息
  
  调制特征 → 轻量 depth decoder head → 逐像素深度预测

为什么要文本引导？
  纯视觉特征在遮挡、反射、透明物体上经常犯错
  文本提供场景级语义先验:
    知道"这是一个沙发" → 更准确估计其深度分布
    而不是被沙发上的花纹纹理误导

训练:
  depth decoder head 用 L1 loss 或 scale-invariant loss
  对 ground truth 深度图做监督
  SD 参数可以冻结或微调 (取决于数据量)
```

#### 方案 2: 深度作为额外输入（逆向思路）

不是让 VLM 输出深度，而是**让 VLM 消费深度**——用现成深度模型预测深度图，作为额外信息输入给 VLM，增强 3D 空间理解。

```
方式 A: 通道拼接
  图像 (H×W×3) + 深度图 (H×W×1) → 拼接为 4 通道 → ViT encoder → VLM

方式 B: 独立编码后 token 拼接 (更常见)
  图像 → image encoder → image tokens
  深度图 → depth encoder → depth tokens
  image tokens + depth tokens 拼接 → 送入 LLM
  
  VLM 可以利用深度信息回答:
  "沙发前面的茶几离镜头多远？" / "哪个物体更靠近？"

优势: 模块化 — 深度估计和 VLM 各自独立，可分别升级
```

#### 方案 3: 统一多任务输出

```
VPD (Visual Perception with a Pre-trained Diffusion Model):
  用 diffusion 模型的特征 + 文本引导
  → 同一套特征、不同的 task-specific head
  → 同时做深度估计、语义分割等多个任务
  
  本质上是用 SD 作为通用视觉 backbone，
  替代传统的 ResNet/ViT backbone
```

---

### 6.6 图像生成与编辑

最前沿的方向：让 VLM 不仅能"理解"图像，还能"生成/编辑"图像。

```
核心矛盾:
  理解 = 图像 → 文本 (ViT encoder → LLM)
  生成 = 文本 → 图像 (方向完全相反)
  如何在同一个模型里统一？
```

#### 方案 1: 解耦式 — VLM 当"导演"

最简单的方案，不需要任何架构改动：

```
用户: "把这张照片里的天空变成星空"

VLM 理解意图 → 生成结构化编辑指令:
  {
    "operation": "inpaint",
    "region": "sky area",
    "prompt": "beautiful starry night sky, milky way, high detail"
  }

→ 指令传给 Stable Diffusion Inpainting → 输出编辑后的图像

VLM 的角色 = 智能 prompt 工程师
把用户模糊的自然语言意图翻译成生成模型需要的精确指令

问题: 误差累积
  VLM 对意图理解可能不完美 → 翻译成 prompt 再丢一层 → 生成执行再丢一层
  两个模型之间没有梯度连接，不能端到端优化
```

#### 方案 2: 统一 tokenizer — Janus（2024, DeepSeek）⭐

**Janus 是 DeepSeek 2024 的工作。**

**核心 Insight：理解和生成对视觉表征的需求是矛盾的。**

```
理解需要高层语义特征:
  需要知道"这是一只猫"，不需要每个像素的颜色值
  → SigLIP/CLIP 的 contrastive learning 特征
  → 语义对齐极好，但丢失大量像素级细节

生成需要可重建的离散表征:
  必须保留足够的像素级信息，才能从 token 解码回图像
  → VQ-VAE 的 codebook
  → 每个 token 对应 codebook 里的一个向量，通过 decoder 可重建图像

如果强行用一套 encoder 做两件事 → 互相拖累:
  语义特征太抽象 → 重建不了图像
  重建特征太底层 → 理解不了语义
  (Janus 论文有 ablation 验证: unified encoder 两方面都更差)
```

**Janus 架构：**

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│  理解路径 (image → text):                                │
│    图像 → SigLIP encoder → visual tokens → LLM → 文本   │
│    (和标准 VLM 完全一样)                                 │
│                                                          │
│  生成路径 (text → image):                                │
│    文本指令 → LLM → 自回归生成 image tokens              │
│    → image tokens = VQ codebook indices                  │
│    → 查 codebook 得到 feature map                        │
│    → VQ-VAE decoder → 像素级图像                         │
│                                                          │
│  关键: LLM vocabulary 被扩展                             │
│    加入 VQ codebook 的所有 code (如 8192 个 special token)│
│    生成图像 = 自回归生成一串 codebook index tokens        │
│                                                          │
│  共享: 同一个 LLM 做统一推理                             │
│  分离: 理解用 SigLIP encoder, 生成用 VQ-VAE encoder      │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

**训练设计：**

```
理解路径 loss: 标准 text cross-entropy
生成路径 loss: 也是 cross-entropy
  target = 图像的 VQ token 序列
  (先用 frozen VQ-VAE encoder 把 GT 图像编码成 token 序列作为标签)

两个 loss 联合训练，共享 LLM 参数

关键: VQ-VAE 的 encoder 和 decoder 在 Janus 训练中是 frozen 的
  它只是一个预训练好的 tokenizer / detokenizer
  LLM 学习的是"给定文本描述，生成正确的 VQ token 序列"
```

**生成分辨率瓶颈：**

```
VQ-VAE 将图像编码为 feature map (如 16×16 = 256 个 token)
LLM 需要自回归生成所有 256 个 token
→ 图像分辨率 = 16 × patch_size (如 patch=16 → 256×256)

如果要生成 1024×1024:
  feature map = 64×64 = 4096 个 token → LLM 自回归生成极慢
  → 这是 autoregressive 图像生成的固有瓶颈
  → 解决方向: 多尺度生成、parallel decoding 等
```

#### 方案 3: 原生多模态生成 — Gemini / GPT-4o 路线

```
最激进的方案: 直接在训练时把图像和文本统一到同一个 token 空间

推测的核心思路 (Google/OpenAI 未公开全部细节):
  图像 → tokenizer (VQ-VAE 变体或 learned tokenizer) → 离散 token
  和 text token 一起做大规模预训练
  模型不区分"图像 token"还是"文本 token"

统一为 autoregressive next-token prediction:
  理解 = 给定 image tokens，预测 text tokens
  生成 = 给定 text tokens，预测 image tokens
  编辑 = 给定 image tokens + text tokens，预测 new image tokens

vs Janus 的比较:
  优势:
    - 只有一套 encoder，不需要在两个表征空间转换
    - 理解和生成可以自然交错
      (如"看这张图，生成一张风格类似但内容不同的图")
  代价:
    - 需要极大规模训练数据和计算量
    - 学好一个同时适合理解和生成的统一表征非常难
    → 目前只有 Google 和 OpenAI 做到了
```

---

### 6.7 能力整合对比

| CV 任务 | 整合方式 | 输出形式 | 核心技术 | 代表模型 |
|---------|---------|---------|---------|---------|
| 目标检测 | 纯文本坐标 | bbox 坐标文本 | 坐标归一化为整数 token | Qwen-VL, Kosmos-2 |
| OCR | 端到端/外部注入 | 识别文本 + 坐标 | 高分辨率输入 + 动态切分 | Qwen2-VL, GOT-OCR |
| 图像分割 | LLM + SAM decoder | 像素级 mask | `<SEG>` token bridging | LISA, GLaMM |
| 关键点检测 | 坐标文本/特殊 token | 关键点坐标序列 | 坐标离散化 | UniPose, Pose-GPT |
| 深度估计 | 专用解码头/额外输入 | 深度图 | SD 特征 + 文本引导 | VPD |
| 图像生成 | 统一 tokenizer | 生成图像 | 双 encoder + VQ-VAE | Janus, GPT-4o |

---

### 6.8 跨任务共性：反复出现的架构设计模式

所有 VLM 整合 CV 任务的方案，底层反复出现几个相同的 pattern：

#### Pattern 1: Special Token Bridging

```
LISA 的 <SEG>、GLaMM 的多个 <SEG>、Pose-GPT 的坐标 token
→ 本质都是: 在 LLM vocabulary 加特殊 token
   用它的 hidden state 作为 bridge 连接下游 decoder

设计空间:
  - token 数量: 1 个 (LISA) vs 多个 (GLaMM) vs 动态数量
  - embedding 维度: LLM hidden dim → 投影到 decoder 需要的维度
  - 映射层结构: 简单 MLP vs 多层 transformer
```

#### Pattern 2: Coordinate Tokenization

```
检测的 bbox、关键点的坐标、OCR 的位置
→ 都面临"连续坐标怎么离散化"的问题

选项:
  - 归一化到 [0, 1000] 整数 → 作为普通文本 token (简单但精度有限)
  - 加 special coordinate token → learned embedding (紧凑但需扩展 vocab)
  - bin 的粒度选择: 粒度越细精度越高，但 vocab 越大
```

#### Pattern 3: Frozen Backbone + Lightweight Adapter

```
不同的冻结策略 → 不同的 trade-off:

LISA: 冻结 ViT + SAM，只训 LoRA + MLP
  → 数据需求小，训练快
  → 但 LLM 的表达能力受 LoRA rank 限制

Qwen-VL: 全参数训练 (包括 ViT)
  → 充分适配，性能更好
  → 但需要大量数据和计算

InternVL2: Progressive 训练 (先冻结大部分，逐步解冻)
  → 兼顾效率和性能
  → 训练流程更复杂
```

#### Pattern 4: 理解 vs 生成的 Encoder 分离

```
Janus 为什么要用两套 encoder？

核心原因: 理解和生成对特征的需求矛盾
  理解: 需要高层语义 (SigLIP) → 丢细节没关系
  生成: 需要可重建的离散编码 (VQ-VAE) → 必须保留细节

能不能用一套？
  Janus 论文 ablation: unified encoder → 理解和生成都变差
  → 一套 encoder 不得不在语义和重建之间妥协

为什么 GPT-4o 似乎做到了？
  → 推测: 更大的模型容量 + 更多的训练数据
     可以学出一个足够丰富的统一表征
  → 但训练成本是 Janus 方案的数量级倍增
```

---

### 6.9 发展趋势

```
阶段 1 (2023): 单任务增强
  每个 CV 能力单独整合，一个论文做一件事
  LISA 做分割，Kosmos-2 做定位
  特点: 验证 "VLM + CV 任务" 的可行性

阶段 2 (2024): 多任务统一
  一个模型同时支持多种 CV 任务
  如 InternVL2 同时支持 OCR + 检测 + 定位
  特点: 共享 backbone，任务间可以互相增益

阶段 3 (2025+): 理解与生成统一
  同一个模型既能理解图像也能生成图像
  如 Janus, GPT-4o 的多模态统一架构
  → 最终目标: 一个模型 = 所有视觉能力
  
  关键挑战:
    - 理解和生成的表征矛盾如何调和
    - 高分辨率生成的效率瓶颈
    - 训练数据的规模和多样性需求
```

---

**相关文档**：
- [VLM 概述](VLM概述.md) — VLM 总览与入门
- [VLM 三大组件详解](VLM三大组件详解.md) — 各组件的深入分析
- [VLM 主流架构详解](VLM主流架构详解.md) — 具体模型架构
- [VLM 训练与评测](VLM训练与评测.md) — 训练流程与评测

[返回上级](README.md) | [返回总目录](../../README.md)

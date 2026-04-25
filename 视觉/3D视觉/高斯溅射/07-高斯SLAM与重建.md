# 07 · 高斯 SLAM 与重建

把 GS 接进 SLAM 是一个反直觉的好想法——SLAM 长期被点云、TSDF、surfel 这类"轻量"表示统治，因为 SLAM 要在线、要省显存、要跟踪准。GS 是离线方法，参数大、依赖 SfM 初始化，乍看完全相反。**但它的实时渲染 + 可微 + 显式表示三件事**让 2024 年涌现出一批优秀工作。这一篇梳理。

> **核心论文**:
> - SplaTAM (Keetha et al., CVPR 2024)
> - Gaussian-SLAM (Yugay et al., 2024)
> - MonoGS (Matsuki et al., CVPR 2024 best paper)
> - Photo-SLAM (Huang et al., ICRA 2024)
> - LIV-GaussMap (2024)

---

## 一、SLAM 与 GS 的契合点

传统 SLAM 流派：

- **稀疏特征 SLAM**（ORB-SLAM）：跟踪准、地图稀疏
- **直接法**（DSO / LSD-SLAM）：跟踪用光度，地图半稠密
- **稠密 SLAM**（KinectFusion / BundleFusion）：TSDF 地图，需深度传感器

NeRF-SLAM（iMAP / NICE-SLAM）2022-2023 兴起，把神经场作为地图。GS-SLAM 是 NeRF-SLAM 的进化版，三个优势：

1. **渲染 100× 更快**：跟踪靠图像配准，渲染快意味着跟踪频率高
2. **显式高斯**：方便增删（新观察到的区域加新高斯，移动物体的高斯删掉）
3. **可微梯度直接对位姿求导**：相机位姿优化和地图优化可以联合做

---

## 二、SplaTAM：第一篇 GS-SLAM

### 2.1 流程

SplaTAM 把 SLAM 拆成"跟踪"和"建图"两个交替步骤，都基于 GS rasterizer：

```
Frame t → 输入 RGB-D
        │
        ├─▶ Tracking: 固定地图，优化相机位姿
        │   ├── 用上一帧位姿初始化
        │   ├── 渲染当前地图，与观测图像算光度损失 + 深度损失
        │   └── 反传梯度到 SE(3) 相机参数（用李代数）
        │
        └─▶ Mapping: 固定位姿，更新地图
            ├── 在新观察区域添加新高斯（用深度图反投影）
            ├── 全部高斯参数走标准 3DGS 优化几个 iter
            └── 剪枝低 α 高斯
```

### 2.2 关键 trick

- **各向同性高斯**：SplaTAM 把高斯简化成球形（缩放 $s_x = s_y = s_z$），减参数 + 加速
- **新高斯添加策略**：仅在"当前帧渲染深度与观测深度差异大"的像素处加新高斯
- **重叠去除**：相邻帧间高斯避免重复添加

### 2.3 性能

- **跟踪**：~5 Hz（与 ORB-SLAM 量级一致）
- **建图**：~3 Hz
- **质量**：比 NICE-SLAM PSNR 高 5+ dB
- **数据集**：ScanNet / Replica，室内 RGB-D 场景

### 2.4 局限

- 需要 RGB-D 输入（深度图必需）
- 各向同性高斯丢了 3DGS 表达力的一半
- 室外 / 大场景未验证

---

## 三、Gaussian-SLAM：keyframe + 渐进

Gaussian-SLAM 改进 SplaTAM 的两个工程问题：

### 3.1 渐进式建图

SplaTAM 每帧都更新所有高斯，新区域和已建好区域互相干扰。Gaussian-SLAM 引入：

- **Keyframe 集合**：只在 keyframe 处添加新高斯
- **冻结老高斯**：远离当前 keyframe 的高斯参数冻结，仅近 keyframe 的高斯更新
- 等价于"老地图记忆"，新区域不破坏旧区域

### 3.2 全局优化

每隔 N 个 keyframe 跑一次 bundle adjustment：所有 keyframe 位姿 + 高斯参数联合优化，纠正漂移。

### 3.3 表现

- 长时序场景下抗漂移能力比 SplaTAM 强 30%+
- 仍依赖 RGB-D

---

## 四、MonoGS：单目 SLAM（best paper）

### 4.1 突破

MonoGS 是 CVPR 2024 best paper 候选——**第一个不要深度图、纯单目 RGB 的 GS-SLAM**。

挑战：
- 单目没有绝对尺度
- 单目深度估计噪声大，直接初始化高斯会糟

### 4.2 关键设计

- **单目深度先验**：用 Depth Anything / Marigold 给每帧粗略深度
- **关键帧 + 先验融合**：keyframe 上做多视角立体匹配修正深度
- **协方差全自由度**：不像 SplaTAM 退化成各向同性，全部自由度优化（保留 GS 表达力）
- **位姿优化用 Levenberg-Marquardt**：比纯 SGD 收敛快、对初始化更鲁棒

### 4.3 性能

- 跟踪 ~10 Hz（甚至超过传统单目 SLAM）
- 实时建图
- 在 Replica / TUM-RGBD 上无深度图也达到接近 SplaTAM 的质量

MonoGS 真正打开了 GS-SLAM 的应用面——手机摄像头 / VR 头显 / 无人机都不一定有深度，单目可用至关重要。

---

## 五、Photo-SLAM：超原语 + 高斯混合

### 5.1 想法

Photo-SLAM 不是把 GS 当地图，而是**与超原语（superprimitives）混合**：

- 短期前景：高斯（变化快，有可微梯度方便跟踪）
- 长期背景：mesh / surfel（稳定，便于回环检测）

类比 ORB-SLAM 的 keyframe + map point，Photo-SLAM 是 keyframe + 高斯 + mesh 的三层结构。

### 5.2 优势

- **回环检测可靠**：mesh 表示长期稳定，特征匹配比纯高斯 SLAM 更鲁棒
- **存储友好**：长期 map 用 mesh，不会随时间无限增长

### 5.3 应用

- ROS 集成已有，机器人 / AR 设备友好

---

## 六、多模态：LIV-GaussMap 等

单纯视觉 SLAM 在弱纹理 / 黑暗场景失败。**LIV-GaussMap** 把 LiDAR / IMU / Visual 融合：

- LiDAR 点云提供精确几何（GS 初始化质量高）
- IMU 提供相邻帧位姿先验（跟踪更稳）
- Visual GS 提供可渲染地图（视觉外观）

整体延续 LIO-SAM / FAST-LIO 这类多模态 SLAM 的架构，把"建图"模块换成 GS。

类似工作：**RTG-SLAM**（实时 LiDAR + GS）、**MM-Gaussian**（多模态融合 GS）。

---

## 七、稀疏视图重建（与 SLAM 邻接）

非 SLAM 但同样关心"少观测下重建"的邻居方向：

- **DUSt3R / MASt3R 启动**：先用 DUSt3R 从两张图直出几何 + 位姿，再做 GS 优化
- **GS-Recon**：稀疏视图 → 高斯，前向重建
- **InstantSplat**：3 分钟内从无 pose 图像重建 GS（DUSt3R + GS optimization）

这条线在 [05 篇](05-生成式高斯.md)的 GS-LRM 路线上有交集——"少视图直出 GS"既可以靠 LRM 大模型，也可以靠 DUSt3R + GS 优化。

---

## 八、各方法对比

| 方法 | 输入 | 跟踪 Hz | 建图质量 (PSNR @ Replica) | 局限 |
|------|------|--------|--------------------------|------|
| NICE-SLAM (2022) | RGB-D | 1-2 | ~25 dB | 慢 |
| SplaTAM | RGB-D | ~5 | ~32 dB | 各向同性 |
| Gaussian-SLAM | RGB-D | ~3 | ~33 dB | RGB-D only |
| MonoGS | RGB | ~10 | ~31 dB | 单目尺度漂移 |
| Photo-SLAM | RGB-D | ~7 | ~32 dB | 实现复杂 |
| LIV-GaussMap | LiDAR + RGB + IMU | ~5 | ~34 dB | 硬件要求高 |

---

## 九、面试高频问题

### Q1: GS-SLAM 比 NeRF-SLAM 强在哪？

**答**：(1) 渲染 100× 更快——SLAM 跟踪要每帧渲染做 photometric 配准，速度直接决定可不可用；(2) 显式高斯方便增删——新观察区域加新高斯，老区域可冻结，避免 NeRF 那种"全场景 MLP 重训"；(3) 梯度直接到位姿——SE(3) 优化和地图优化能联合做。

### Q2: SplaTAM 为什么用各向同性高斯？

**答**：跟踪是高频任务（~5 Hz），优化每个高斯的全部 11 个参数（位置 3 + 缩放 3 + 四元数 4 + α 1）太慢。各向同性退化成球形，参数减到 5（位置 3 + 半径 1 + α 1），跟踪速度上一个台阶。代价是表达力下降，复杂场景质量受损。后续 MonoGS 用全自由度高斯 + LM 优化器追回了速度。

### Q3: 单目 GS-SLAM 怎么解决尺度问题？

**答**：(1) 用单目深度估计模型（Depth Anything / Marigold）给每帧粗深度；(2) keyframe 间做多视角立体匹配修正；(3) 训练时加深度一致性损失约束相对深度比例；(4) 选择合理的尺度参考帧（比如初始化时假设第一帧某物体的尺度）。MonoGS 把这些组合起来达到了无 depth sensor 也能用的水平。

### Q4: GS-SLAM 怎么处理回环检测？

**答**：纯 GS 表示对回环不友好（高斯没有特征描述符）。Photo-SLAM 的做法是混合表示：长期 mesh 部分用传统 ORB 特征做回环；纯 GS-SLAM 多用全局优化（每 N 帧跑一次 bundle adjustment）来减漂移，不显式做回环。这是当前 GS-SLAM 的弱项之一。

### Q5: GS-SLAM 离上线（手机 / 头显）还差什么？

**答**：(1) **存储**：典型场景 GS 几百 MB，手机吃不下，需要结合 [04 篇](04-加速与压缩.md)的 HAC 压缩到 < 50 MB；(2) **能耗**：rasterizer 的 GPU 占用高，手机连续跑会发烫；(3) **回环 / 重定位**：长期使用必须有可靠回环；(4) **动态物体处理**：当前 GS-SLAM 假设静态场景，行人/移动物体会污染地图。Apple Vision Pro / Quest 这种设备的核心瓶颈在前两条。

---
[← 上一篇：06 · 高斯 + 物理仿真](06-高斯+物理仿真.md) | [专题导航](README.md) | [下一篇：08 · 高斯 + 具身 →](08-高斯+具身.md)

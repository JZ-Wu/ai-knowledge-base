# 高斯溅射专题

3D Gaussian Splatting (3DGS) 自 SIGGRAPH 2023 以来快速演化为新视角合成、重建、生成、仿真的统一表示。本专题按方向拆成 8 篇，每篇聚焦一个分支的核心问题、关键论文和工程要点。

## 阅读路径

如果你刚接触 3DGS，按编号顺序读 01 → 08；如果只关心某个方向，直接跳到对应文章。01 是其他文章的前置。

| # | 文章 | 关键词 | 难度 |
|---|------|--------|------|
| 01 | [3DGS 基础与渲染](01-3DGS基础与渲染.md) | 高斯参数化、EWA splatting、tile 光栅化、自适应密度控制 | ★★ |
| 02 | [2DGS 与表面重建](02-2DGS与表面重建.md) | 法线一致性、Mesh 提取、SuGaR、GOF | ★★★ |
| 03 | [动态与 4DGS](03-动态与4DGS.md) | 时变形场、Spacetime spline、Dynamic Gaussians | ★★★ |
| 04 | [加速与压缩](04-加速与压缩.md) | Scaffold-GS、Mip-Splatting、向量量化、HAC | ★★ |
| 05 | [生成式高斯](05-生成式高斯.md) | DreamGaussian、LGM、GS-LRM、SDS 蒸馏 | ★★★ |
| 06 | [高斯 + 物理仿真](06-高斯+物理仿真.md) | PhysGaussian、Spring-Gaus、MPM 耦合、real-to-sim | ★★★★ |
| 07 | [高斯 SLAM 与重建](07-高斯SLAM与重建.md) | SplaTAM、MonoGS、Photo-SLAM、在线重建 | ★★★ |
| 08 | [高斯 + 具身](08-高斯+具身.md) | SplatSim、RoboGSim、VLA × GS、3D feature field | ★★★ |

## 一张图看懂派系

```
                            原版 3DGS (Kerbl 2023)
                                    │
          ┌────────────┬────────────┼────────────┬────────────┬────────────┐
          ▼            ▼            ▼            ▼            ▼            ▼
        几何精度       动态         加速         生成        物理         应用
        (02)         (03)        (04)         (05)        (06)        (07/08)
          │            │            │            │            │
        2DGS         4DGS      Scaffold-GS  DreamGS     PhysGaussian   SLAM (07)
        SuGaR        Def-3DGS  Mip-Splat    LGM         Spring-Gaus    具身 (08)
        GOF          STG       LightGS      GS-LRM      VR-GS
```

## 时间线（节点级）

| 时间 | 事件 |
|------|------|
| 2023.07 | 3DGS (SIGGRAPH 2023) 发布，开启高斯时代 |
| 2023.11 | DreamGaussian — 第一个把 GS 接进文生 3D |
| 2024.02 | 4DGS / Deformable 3DGS — 动态 GS 三连发 |
| 2024.03 | SplaTAM / MonoGS — GS 进入 SLAM |
| 2024.05 | 2DGS / SuGaR / GOF — 几何精度问题被系统解决 |
| 2024.06 | PhysGaussian — GS 与 MPM 物理耦合 |
| 2024.08 | LGM / GS-LRM — 大模型直出 GS |
| 2024.10 | Scaffold-GS / Mip-Splatting — 工程级压缩与抗混叠 |
| 2024.12 | RoboGSim / Re³Sim — GS 进入具身 real-to-sim |

## 工程注意事项

- **CUDA 编译**：原版 3DGS 的 rasterizer 需要本地编译 CUDA kernel，PyTorch 版本必须匹配。推荐使用 [gsplat](https://github.com/nerfstudio-project/gsplat) 库（Nerfstudio 团队，cleaner 接口）作为现代替代。
- **数据准备**：COLMAP 的稀疏点云质量直接决定 3DGS 上限，弱纹理/反光场景需先解决 SfM 失败问题。
- **显存**：百万高斯训练通常需要 24 GB+ 显存。压缩方法（04 篇）能把训练显存降到 8 GB 量级。
- **复现陷阱**：很多论文的 demo 场景挑过，自己数据上效果可能差 3-5 dB PSNR。务必在自己关心的数据集上跑 baseline。

## 相关资料

- 原版代码：[graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- 综述：[A Survey on 3D Gaussian Splatting](https://arxiv.org/abs/2401.03890) (2024)
- 教程：[gsplat 文档](https://docs.gsplat.studio/)
- Benchmark：MipNeRF360 / Tanks-and-Temples / Deep Blending（3DGS 标准三件套）

---
[返回 3D 视觉](../README.md) | [返回视觉](../../README.md)

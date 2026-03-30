# AI Infra 概述

AI Infra（AI 基础设施）是让模型从"能跑"到"跑得好/跑得快/跑得省"的所有工程。

---

## 一、什么是 AI Infra？

```
研究员: 这是我的模型权重
               ↓
AI Infra:
  - 怎么高效管理显存？  → PagedAttention
  - 怎么缓存重复计算？  → RadixAttention
  - 怎么榨干硬件性能？  → TensorRT / Kernel 优化
  - 怎么分布式部署？    → Tensor Parallel / Pipeline Parallel
  - 怎么处理高并发？    → 请求调度、Continuous Batching
  - 怎么高效训练？      → 分布式训练框架、混合精度
  - 怎么做 RLHF？      → 多角色调度、推理-训练交替
               ↓
线上服务: 每秒处理上千请求，延迟 < 100ms
```

AI Infra 大致分为两大方向：**Serving Infra**（推理部署）和 **Training Infra**（训练基础设施）。

---

## 二、Serving Infra：让模型高效服务

### 2.1 核心瓶颈

LLM 推理时最大的问题是 **KV Cache 占显存过多**：

```
每生成一个 token，都要存之前所有 token 的 Key 和 Value
→ 长对话 / 多用户并发时，显存很快耗尽
→ 预分配的连续空间利用率低（约 50%）
```

### 2.2 主流推理框架

| 框架 | 核心创新 | 适用场景 | 上手难度 |
|------|---------|---------|---------|
| **vLLM** | PagedAttention | 线上服务，高并发 | 中等 |
| **SGLang** | RadixAttention | 多轮对话、共享前缀 | 中等 |
| **TensorRT-LLM** | NVIDIA 底层优化 | 极致延迟 | 高 |
| **Ollama** | 极致易用 | 本地实验 | 极低 |

#### vLLM — PagedAttention

```
核心思想: 像操作系统管理内存一样管理 KV Cache

传统做法:
  每个请求预分配一大块连续显存放 KV Cache
  → 短请求浪费，长请求可能不够，利用率约 50%

PagedAttention:
  把 KV Cache 切成小"页"（如每页 16 token 的 KV）
  按需分配，不要求连续
  → 跟 OS 的虚拟内存分页机制一样
  → 显存利用率提升到约 95%
  → 同样的 GPU 能多服务 2-4 倍的并发请求
```

**为什么有效？** 传统方式必须为每个请求预留最大长度的连续空间（比如 2048 token），但大部分请求用不完，浪费严重。PagedAttention 按需分页，用多少分多少。

#### SGLang — RadixAttention

```
核心思想: 用 Radix Tree（前缀树）缓存已计算的 KV Cache

场景: 多轮对话每轮都带着完整历史
  第 1 轮: [系统提示 + 问题1] → 计算 KV Cache，存入 Radix Tree
  第 2 轮: [系统提示 + 问题1 + 回答1 + 问题2]
           ^^^^^^^^^^^^^^^^^^^^^^
           前缀匹配命中缓存，KV Cache 直接复用，不用重算

其他场景:
  - 多个用户共享同一个 system prompt → 这部分 KV Cache 只算一次
  - Few-shot prompting → 相同的 few-shot 示例部分缓存复用
```

**类比**：浏览器缓存——访问过的资源下次直接用，不再从服务器下载。

#### TensorRT-LLM — NVIDIA 底层优化

```
核心思想: 针对 NVIDIA GPU 做硬件级深度优化

三大手段:
  1. Kernel Fusion:
     多个小算子合并为一个大 GPU Kernel
     → 减少 GPU 内核启动开销和显存读写次数
     例: LayerNorm + 残差连接 + Dropout → 一个 fused kernel

  2. 量化加速:
     INT8/INT4/FP8 使用 Tensor Core 专用硬件计算
     比软件模拟的量化推理快很多

  3. 定制 CUDA Kernel:
     每个算子针对特定 GPU 架构（A100/H100）手写优化
     → 和通用 PyTorch 实现相比，通常快 2-5 倍

代价: 需要将模型转为 TensorRT 引擎格式（.plan），适配工作量大
```

#### Ollama — 一行命令跑模型

```
定位: LLM 的 "Docker"，极致易用

底层使用 llama.cpp (C/C++ 推理引擎)
  $ ollama run llama3    # 自动下载 + 量化 + 运行

特点:
  - 支持 CPU 推理，不一定需要 GPU
  - 自动管理模型下载和量化
  - 性能一般，但谁都能跑
```

### 2.3 其他关键 Serving 技术

```
Continuous Batching (持续批处理):
  传统 batching: 等一批请求凑齐再一起处理，短请求等长请求
  Continuous: 有请求完成就立刻塞入新请求，不让 GPU 空闲
  → vLLM / SGLang 都支持

Speculative Decoding (投机解码):
  用小模型快速生成多个候选 token → 大模型一次性验证
  → 在不降低质量的前提下提速 2-3 倍

Prefix Caching (前缀缓存):
  缓存 system prompt 等固定前缀的 KV Cache
  → 多个请求共享同一前缀时只算一次
```

---

## 三、Training Infra：让模型高效训练

### 3.1 分布式训练

```
小模型 (≤13B):
  数据并行 (DDP): 每张 GPU 放完整模型，各自处理不同数据
  → 简单高效，PyTorch 原生支持

大模型 (70B+):
  单卡放不下，必须切分:
  ├── Tensor Parallel (TP): 把每一层的矩阵切分到多卡
  ├── Pipeline Parallel (PP): 把不同层放在不同卡
  └── Data Parallel (DP): 在 TP/PP 基础上再做数据并行

  常见框架:
  - DeepSpeed (ZeRO): 微软，内存优化为主
  - Megatron-LM: NVIDIA，大规模预训练
  - FSDP: PyTorch 原生，类似 ZeRO
```

### 3.2 混合精度训练

```
核心思想: 计算用低精度（快），存储关键值用高精度（准）

  前向计算: BF16 / FP16   ← 快，显存省
  反向梯度: BF16 / FP16
  权重主副本: FP32         ← 保证数值稳定性
  优化器状态: FP32

→ 训练速度提升约 2 倍，显存减少约一半，精度基本不损失
```

### 3.3 RL/RLHF Infra

RL 训练（特别是 RLHF/GRPO）和普通训练的 Infra 需求完全不同：

```
一轮 RLHF 训练需要:
  1. Actor 生成回答         → 推理 (inference)
  2. Reference Model 算概率  → 推理
  3. Reward Model 打分       → 推理
  4. Critic 估值             → 推理
  5. PPO 更新参数            → 训练 (training)

核心难点:
  - 推理和训练交替进行，GPU 容易空闲
  - 涉及 4 个不同的模型角色，如何调度？
  - 生成（推理）是最慢的瓶颈
```

**主流 RL Infra 框架**：

| 框架 | 核心思路 | 适用规模 |
|------|---------|---------|
| **TRL** (HuggingFace) | 最易用，单机即可 | 小规模实验 |
| **DeepSpeed-Chat** | Actor/Critic/Reward/Ref 分配到不同 GPU 组 | 中大规模 |
| **OpenRLHF** | 用 Ray 做分布式调度 + vLLM 加速生成 | 大规模 |
| **veRL** (火山引擎) | 同一 GPU 集群动态切换训练/推理模式 | 大规模 |

```
RL Infra 的核心优化方向:

1. 生成加速:
   RLHF 中 ~70% 时间花在 Actor 生成回答
   → 用 vLLM/SGLang 的推理优化加速这一步

2. 资源复用:
   训练和推理交替进行，GPU 一半时间在空闲
   → veRL: 同一组 GPU 动态切换角色，消除空闲
   → OpenRLHF: Ray 灵活调度，按需分配资源

3. 通信优化:
   4 个模型之间需要频繁传递数据
   → 模型共置（co-location）减少网络传输
   → 异步 pipeline 让不同阶段重叠执行
```

---

## 四、Serving Infra vs Training Infra

| | Serving Infra | Training Infra | RL Infra |
|--|--------------|----------------|----------|
| 核心目标 | 低延迟、高吞吐 | 高效利用算力 | 协调多角色、减少空闲 |
| 主要瓶颈 | KV Cache 显存、并发 | 通信、显存 | 推理-训练交替、多模型调度 |
| 关键技术 | PagedAttention、Batching | TP/PP/DP、ZeRO | Ray 调度、生成加速 |
| 技术栈 | C++/CUDA 为主 | Python + 分布式框架 | Python + Ray/DeepSpeed |
| 典型岗位 | 推理优化工程师 | 训练平台工程师 | RL 系统工程师 |

---

## 五、Infra 工程师日常做什么？

```
Serving 方向:
  - 适配新模型到推理框架（vLLM 加新模型支持）
  - 优化推理延迟（profile → 找瓶颈 → 写/改 CUDA kernel）
  - 设计请求调度策略（负载均衡、优先级队列）
  - 监控线上服务（吞吐、延迟 P99、GPU 利用率）

Training 方向:
  - 搭建和维护分布式训练集群
  - 训练框架适配（模型切分策略、通信优化）
  - 训练任务调度和容错（checkpoint、断点续训）
  - 训练性能 profiling 和优化

RL Infra 方向:
  - 设计 RLHF 训练 pipeline 的资源调度
  - 优化生成阶段的吞吐
  - 多模型间的数据传输和同步
```

---

**相关文档**：
- [推理优化详解](推理优化详解.md) — KV Cache、FlashAttention、量化等核心技术
- [VLM 训练与评测](../多模态/VLM训练与评测.md) — VLM 推理框架对比
- [分布式训练](../../分布式训练/README.md) — 分布式训练专题

[返回上级](../README.md) | [返回总目录](../../README.md)

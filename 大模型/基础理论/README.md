# 基础理论

大语言模型的核心理论知识。

## 知识点索引

### Transformer 架构
- [Transformer架构详解](Transformer架构详解.md) — Encoder-Decoder 原始架构、Cross-Attention、Decoder-Only 逐组件详解(Embedding/RoPE/MHA/GQA/SwiGLU FFN/RMSNorm)、Temperature 与采样策略、主流模型参数、Prefill+Decode 推理流程

### Tokenizer 与词表
- [Tokenizer详解](Tokenizer详解.md) — BPE、WordPiece、SentencePiece/Unigram、中文 token 效率问题

### MoE (混合专家)
- [MoE详解](MoE详解.md) — 稀疏激活、路由机制、负载均衡、DeepSeek 的细粒度专家与共享专家

### Scaling Laws
- [Scaling Laws详解](Scaling_Laws详解.md) — Kaplan 定律、Chinchilla 最优比例 (D≈20N)、训练规划计算

### 长上下文
- [长上下文详解](长上下文详解.md) — RoPE 外推 (PI/NTK/YaRN)、Sliding Window Attention、KV Cache 压缩 (MLA)

### CoT 与推理范式
- [CoT与推理范式](CoT与推理范式.md) — Chain-of-Thought 发展历程、Zero-shot/Few-shot CoT、Self-Consistency、推理模型 (o1/R1)、Test-time Compute Scaling

---
[返回上级](../README.md) | [返回总目录](../../README.md)

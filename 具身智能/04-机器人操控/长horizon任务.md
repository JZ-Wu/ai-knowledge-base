# 长 Horizon 任务 (Long-Horizon Manipulation)

长 horizon 任务指包含多个子步骤、需要数十秒到数分钟才能完成的复杂任务：做一杯咖啡、整理房间、组装家具、打包物品。单一的 VLA 策略在 5–30 秒内表现良好，但超过 1 分钟就开始出现复合误差、上下文遗忘、子目标飘移。

## 1. 为什么长 horizon 难

### 1.1 复合误差 (Compounding Error)

每步动作的小误差改变下一步观测，误差累积呈指数增长。50 步累积 1% 偏差 → 第 50 步接近完全偏离。

### 1.2 上下文长度

当前主流 VLA（π₀、OpenVLA、GR00T N2）的视觉上下文约 2–10 秒。而"做咖啡"需要 1–3 分钟。

### 1.3 子任务依赖

$$\text{做咖啡} = \underbrace{\text{拿杯子}}_{t_1} \to \underbrace{\text{放到机器下}}_{t_2} \to \underbrace{\text{按开关}}_{t_3} \to \underbrace{\text{等待}}_{t_4} \to \underbrace{\text{取走}}_{t_5}$$

其中 $t_4$ 需要等待 30 秒（而不是立即执行），这对纯反应式策略是灾难性的。

### 1.4 失败恢复

子任务失败（如杯子掉了）需要重新规划，不能继续执行后续步骤。

## 2. 任务分解方法

### 2.1 SayCan (Google, 2022) — 里程碑

核心思想：**语言模型提供可能性（Say），值函数提供可行性（Can）**。

$$\text{Score}(\text{skill}_i) = \underbrace{p_{\text{LLM}}(\text{skill}_i \mid \text{prompt})}_{\text{语言合理性}} \times \underbrace{V(\text{skill}_i \mid s_t)}_{\text{物理可行性}}$$

- LLM（PaLM）打分每个原子 skill 对当前任务的相关性
- 值函数（RL 或启发式）打分每个 skill 在当前状态下是否能成功
- 选最高分 skill 执行 → 更新状态 → 循环

**意义**：首次把 LLM 的常识推理与机器人物理能力解耦组合。

**局限**：技能库需手工设计、值函数训练耗时、子任务粒度难定。

### 2.2 Code as Policies (Google, 2023)

让 LLM 直接生成 Python 代码调用机器人 API：

```python
# 用户: "把所有红色的积木堆成塔"
def stack_red_blocks():
    red_blocks = detect_objects("red block")
    red_blocks.sort(key=lambda b: b.position.z, reverse=True)
    base = red_blocks[0]
    for block in red_blocks[1:]:
        pick(block)
        place(block, on_top_of=base)
        base = block
```

**优势**：
- 天然支持循环、条件、变量、函数调用
- 比 SayCan 的线性技能序列表达力更强
- 利用 LLM 已有的代码生成能力

**局限**：API 需要预定义，代码执行时仍需底层技能库。

### 2.3 VoxPoser (Li et al., 2023)

LLM 不直接生成动作，而是生成 **3D 值函数**：

```
语言指令 "把苹果放进左边的碗里"
    │
    ▼
LLM 生成 Python 代码
    │
    ▼
代码调用 VLM（OWL-ViT 检测）填充 3D voxel：
  - Affordance Map: 哪里该去（目标点附近高值）
  - Constraint Map: 哪里该避（碗壁、其他物体）
    │
    ▼
规划器在 3D 值函数上做路径优化 → 末端轨迹
```

**亮点**：
- 零样本：不需要任何机器人演示数据
- 跨具身：只要有末端控制就能用
- 可组合：多个值函数可以叠加

**局限**：依赖显式 3D 检测，闭环反应能力弱。

### 2.4 其他分解方法 (2023-2024)

| 工作 | 机构 | 核心 |
|------|------|------|
| **Inner Monologue** | Google | LLM + 环境反馈循环 |
| **ProgPrompt** | NVIDIA | 提示 LLM 以程序化结构分解 |
| **Grounded Decoding** | Google | LLM 解码时加物理约束 |
| **LLM-Planner** | OSU | LLM 在 ALFRED 上 few-shot planning |
| **Text2Motion** | Stanford | LLM 分解 + 轨迹优化 |
| **RoboAgent / MT-ACT** | CMU | 分层 ACT + 语言 skill selection |

## 3. 分层策略 (Hierarchical Policy)

```
┌─────────────────────────────────────────────┐
│  高层策略 (Skill Selector / Planner)         │
│  输入: 语言指令 + 全局状态                   │
│  输出: 当前子技能 (reach/grasp/place/...)    │
│  频率: ~0.1-1 Hz                             │
└──────────────────┬──────────────────────────┘
                   │
   ┌───────────────┼───────────────┐
   ▼               ▼               ▼
┌───────┐     ┌───────┐      ┌───────┐
│ reach │     │ grasp │      │ place │   ← 技能库
└───┬───┘     └───┬───┘      └───┬───┘
    │             │              │
    ▼             ▼              ▼
┌─────────────────────────────────────────────┐
│  底层控制 (Diffusion Policy / ACT / VLA)    │
│  输出: 关节力矩或末端速度                    │
│  频率: 30-1000 Hz                            │
└─────────────────────────────────────────────┘
```

### 经典分层架构

- **Options Framework** (Sutton)：RL 经典分层，高层选 option，低层执行。
- **HRL (Hierarchical RL)**：用 goal-conditioned policy 做低层，上层给 goal。
- **Skill Chaining**：把成功的 skill 末态作为下一个 skill 的初态。

### 2024–2026 的演化

分层思路正在被 **"单一大模型 + 显式 token 化子任务"** 替代：

- **RT-H (Google, 2024)**：在 VLA 输出中插入 **language motion tokens**（"向上抬 5cm"），高低层在同一网络里。
- **π₀.5 (Physical Intelligence, 2025)**：Discrete Language Action 作为高层 token，VLA 既输出子任务描述，也输出连续动作。
- **Gemini Robotics-ER (Google DeepMind, 2025)**：先在 VLM 中做 Embodied Reasoning（空间定位、轨迹预测、affordance），再交给执行 VLA。

## 4. 2025–2026 最新范式

### 4.1 π₀.5 的 Discrete Language Action

核心思路：VLA 在每步不仅预测动作，也同时预测**下一段该做什么**的文本描述。

```
t=0:   obs → VLA → {动作 a, 语言 token "反手抓住杯柄"}
t=10:  obs → VLA → {动作 a, 语言 token "向咖啡机移动"}
t=30:  obs → VLA → {动作 a, 语言 token "按下启动按钮"}
```

**优势**：
- 显式生成了"思考链"，类似 LLM 的 Chain-of-Thought
- 语言 token 是中间监督信号，帮助网络分解任务
- 推理时可以用语言 token 做可解释诊断

### 4.2 Gemini Robotics-ER 的具身推理

Embodied Reasoner (ER) 是 Gemini Robotics 的上游模块：

- **空间理解**：2D 点 → 3D 位置 → 物体 affordance
- **轨迹预测**：预测末端执行器应该走的粗糙 3D 轨迹
- **抓取推理**：VLM 直接判断哪里可抓
- 然后把这些结构化输出交给下游 VLA 执行

本质上是把 SayCan/VoxPoser 用 VLM 端到端化。

### 4.3 Diffusion World Model 驱动的长 horizon

- **UniSim / Genie (2024-2025)**：视频生成模型模拟未来
- **DreamerV3 on Robotics**：Dreamer 系列扩展到机器人
- 思路：VLA 用 world model "想象" 未来 30 秒，再选最优动作

## 5. 记忆与上下文

### 5.1 当前 VLA 的 context 限制

| 模型 | 视觉上下文（秒） | 动作预测 horizon |
|------|------------------|------------------|
| OpenVLA | 1 帧（当前观测） | 单步 |
| RT-2 | 6 帧 | 单步 |
| π₀ | ~2 秒（多视角历史） | 动作 chunk ~1s |
| π₀.5 | ~2 秒 + 语言 token | 动作 chunk + 子任务 |
| GR00T N2 | 短历史 | chunk |
| Helix (Figure) | 短历史（侧重实时） | chunk |

超过 context 长度，模型对几分钟前发生的事完全"失忆"。

### 5.2 记忆机制方向

| 方向 | 思路 | 代表 |
|------|------|------|
| **显式任务栈** | 外部维护子任务列表 | SayCan 系 |
| **RAG-style** | 把历史关键帧存向量库，按需检索 | 尚未有成熟 VLA 版本 |
| **长 context Transformer** | 扩展 VLA context 到数万 token | GR00T N3 路线图 |
| **状态摘要 token** | 定期生成任务进度的文本摘要 | π₀.5 的 discrete action |
| **Scene Graph Memory** | 3D 场景图作为持久记忆 | ConceptGraph, SayPlan |

### 5.3 开放挑战

1. **什么信息该记住**：不能记所有像素，如何学习"重要事件"抽取？
2. **如何表示时间**：相对时间（"刚才"）vs 绝对时间（"30 秒前"）
3. **失败恢复需要反事实记忆**：记住"我试了 A 但失败了，不要再试 A"
4. **跨 episode 记忆**：今天学的，明天还记得吗？在线学习 vs 固定权重

## 6. 2026 趋势总结

- **端到端 VLA 仍无法独立完成分钟级任务**，需配合高层 planner。
- **planner 正从显式 LLM 转向 VLM-based Embodied Reasoner**（Gemini ER 路线）。
- **Discrete language action** 成为 VLA 自带分解能力的关键机制。
- **长 context + world model** 是三年后主线方向。
- **研究 vs 量产差距**：研究 demo 1–2 分钟不稳定；量产（Helix 仓储拣选）依赖结构化环境把长 horizon 压缩成短反应循环。

---
[返回本目录](README.md) | [上一篇：灵巧操作](灵巧操作.md) | [下一篇：双臂与全身操控](双臂与全身操控.md)

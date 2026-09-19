---
tags:
  - 大模型
  - AI 推理
  - 分布式推理
  - 集合通信
  - vLLM
  - NCCL
  - HCCL
---

# 大模型分布式推理中的通信原语

!!! abstract "一句话总结"

    **通信原语描述的是一次 Forward 中张量怎样在多个 rank 之间改变归属和布局。** TP 用 AllReduce、AllGather 或 ReduceScatter 拼回线性层的数学结果；PP 用 Send/Recv 传递 stage 边界激活；CP 交换跨序列的 Query/KV；EP 用 AllToAll 或 AllGather 把 token 送到专家；Prefill/Decode 分离则在实例之间搬运 KV Cache。理解推理通信的关键，是始终写清楚通信前后的 tensor shape、分片轴、process group 和下一算子需要的布局。

---

## 1. 推理视角的心智模型：谁拥有这块张量

第一次接触集合通信时，很容易把 AllReduce、AllGather、ReduceScatter 记成一组相似的 API。对大模型推理，更有效的办法是连续问五个问题：

1. **当前处于哪个阶段？** 模型加载、Prefill、Decode、采样，还是 KV 迁移？
2. **谁参与？** 两个相邻 PP rank、一个 TP/CP/EP group，还是整个 world？
3. **每个 rank 现在有什么？** 完整副本、不同分片，还是同一输出的局部贡献？
4. **下一算子需要什么布局？** 完整 hidden state、sequence shard、head shard，还是按 expert 分桶的 token？
5. **必须现在恢复完整张量吗？** 如果下游能继续消费分片，就可以推迟甚至取消通信。

设通信组有 $p$ 个 rank，当前调度批次打平后的 token 数为 $T$，hidden size 为 $H$。推理热路径中的常用原语如下：

| 原语 | 通信前 | 通信后 | 推理中的典型位置 |
|---|---|---|---|
| Send/Recv | 发送方持有张量 | 指定接收方得到张量 | PP 激活、环形 KV、P/D KV 传输 |
| AllReduce | 每个 rank 有同形状的局部贡献 | 每个 rank 得到完整聚合结果 | Row Parallel 输出、Vocab Parallel Embedding |
| AllGather | 每个 rank 有不同分片 | 每个 rank 得到完整拼接结果 | 恢复 hidden/token/head 维，AllGather 型 MoE |
| ReduceScatter | 每个 rank 有完整形状的局部贡献 | 每个 rank 得到聚合结果的一片 | Sequence Parallel、MoE 前后的分片布局 |
| AllToAll / AllToAllV | 每个 rank 有发往各 rank 的分片 | 数据按新归属重新分桶 | MoE Dispatch/Combine、sequence↔head 转置 |
| Broadcast | root 持有数据 | 所有 rank 得到相同副本 | 输入、控制元数据、采样结果同步 |
| Gather / Scatter | 分片围绕 root 汇聚或分发 | root 或各 rank 得到目标布局 | 请求分发、输出收集、调试 |
| Barrier | 各 rank 只有执行进度 | 所有 rank 越过同步点 | 初始化与故障定位 |

!!! note "T 不是固定 batch size"

    在线推理引擎通常把本轮需要计算的 token 打平为 $[T,H]$。Prefill 请求可能贡献一段 prompt，Decode 请求通常贡献一个 token，chunked prefill、推测解码和连续批处理又会改变每个请求的贡献量。因此 $T$ 每轮都可能变化，通信 buffer、split sizes 与图捕获边界也随之受到影响。

---

## 2. 从控制面走到数据面

分布式推理中的“通信”至少有三层，不应混在一起：

~~~mermaid
flowchart LR
    A["控制面<br/>请求、调度、活跃状态"] --> B["张量数据面<br/>TP / PP / CP / EP"]
    B --> C["设备通信库<br/>NCCL / HCCL"]
    C --> D["物理链路<br/>NVLink / HCCS / PCIe / RoCE / IB"]
~~~

- **控制面**决定本轮跑哪些请求、多少 token、采用哪种图和哪些 rank；
- **张量数据面**决定激活、KV、logits 如何切分和重组；
- **通信库**选择 ring、tree、分层算法、协议和通道；
- **物理链路**最终决定带宽、延迟和跨节点拥塞。

一个 collective 很慢，未必是通信库传得慢。某个 rank 可能因为请求更多、expert 更热或 KV 更长而晚到，其他 rank 在 collective 内等待，时间线上仍会显示为“通信耗时”。

### 2.1 本文的源码阅读基线

为了把原语映射到真实推理代码，本文沿用站内并行策略笔记检查过的源码快照：

- vLLM：分支 releases/v0.26.0，提交 568afb3a13806beb53bb2e6bd518269357b237c0；
- vLLM Ascend：分支 feat/v0.26.0rc-add-fastokens-dependency，提交 ba679baccb7c504788c22a5d0fdd0b1c7f9336e2。

核心阅读地图：

| 文件 | 与通信原语的关系 |
|---|---|
| vllm/config/parallel.py | 定义 TP、PP、DP、PCP、DCP、EP 等并行规模 |
| vllm/distributed/parallel_state.py | 构造各类通信组，并封装 AllReduce、AllGather、ReduceScatter、Send/Recv |
| vllm/model_executor/layers/linear.py | Column/Row/QKV Parallel Linear 的张量切分与归并 |
| vllm/model_executor/layers/vocab_parallel_embedding.py | 词表并行 Embedding 与 LM Head |
| vllm/v1/worker/gpu_worker.py | PP stage 间异步收发中间张量 |
| vllm/v1/worker/dp_utils.py | DP 活跃状态与 dummy forward 协调 |
| vllm/v1/worker/cp_utils.py | DCP 的 token/KV 布局辅助逻辑 |
| vllm_ascend/distributed/parallel_state.py | NPU communicator、MC2 与细粒度 TP 通信组 |
| vllm_ascend/attention/context_parallel/ | Prefill/Decode 的 CP Attention 数据交换 |
| vllm_ascend/ops/fused_moe/token_dispatcher.py | AllGather、AllToAll、MC2 Dispatch/Combine |

上游 vLLM 建立并行语义和通用通信接口；vLLM Ascend 把这些接口落到 HCCL/NPU communicator，并在 Attention 与 MoE 热路径选择平台融合实现。

---

## 3. 点对点通信：Send/Recv

点对点通信只描述一个发送方和一个接收方：

$$
\operatorname{Send}(x,r_s\rightarrow r_d),
\qquad
\operatorname{Recv}(y,r_s\rightarrow r_d)
$$

它不要求通信组中所有 rank 都参加，适合天然具有邻接或生产者—消费者关系的数据流。

### 3.1 Pipeline Parallel：只传前向激活

PP 把连续 Transformer 层分给不同 stage。每个调度 step 中，stage 把边界激活传给下一 stage：

~~~mermaid
sequenceDiagram
    participant S0 as Stage 0
    participant S1 as Stage 1
    participant S2 as Stage 2
    S0->>S1: hidden_states [T,H]
    S1->>S2: hidden_states [T,H]
    S2-->>S2: LM Head / Sampling
~~~

若一个 PP stage 内还有 TP，则发送策略有两种：

- 每个 TP rank 发送自己的 hidden shard，接收侧保持相同分片；
- 发送前或接收后执行 AllGather，让下一 stage 得到完整 hidden state。

选择取决于 stage 边界两侧的张量布局。多一次 AllGather 可能让接口简单，却会增加流量和峰值显存。

### 3.2 Context Parallel：KV 块沿环轮转

CP 沿序列轴切分上下文。本地 Query 要读取其他 rank 的 KV，可以让 KV chunk 沿环逐跳 Send/Recv：

~~~text
第 0 轮：local Q × local KV
第 1 轮：发送本地 KV，接收左邻居 KV，并继续计算
...
第 p-1 轮：local Q 已访问完整上下文
~~~

相比先 AllGather 完整 KV，环形 P2P 只需保留当前通信块，并可把传输与 Attention 计算重叠。代价是调度更复杂，通信轮数随 CP size 增长。

### 3.3 Prefill/Decode 分离：迁移 KV Cache

P/D 分离把长 prompt 的 Prefill 和逐 token Decode 放在不同实例。Prefill 实例完成计算后，必须把每层 KV Cache 及其块映射交给 Decode 实例：

~~~mermaid
flowchart LR
    P["Prefill 实例<br/>计算 prompt KV"] --> M["KV 数据传输<br/>P2P / RDMA / 专用 connector"]
    M --> D["Decode 实例<br/>注册目标 cache block"]
    C["控制面元数据<br/>request id / block mapping"] --> P
    C --> D
~~~

这里通常不是一次简单 collective，而是数据通道与控制通道配合：

- 控制面匹配请求、源/目标 block 和传输完成状态；
- 数据面把分片后的 K/V 写入目标实例对应 cache；
- Decode 必须等所需层和 block 可见后才能继续 Forward。

如果 TP/CP 布局在 P 与 D 两侧不同，还需要在传输过程中做重排或让多个 rank 分别发送对应 shard。

### 3.4 点对点常见故障

- Send 与 Recv 的 peer、顺序或 tag 不匹配；
- PP 两侧对 $T$、$H$、dtype 或分片状态理解不同；
- 某个 stage 本轮没有 token，却没有执行协议要求的空操作；
- KV 已传输但 block table/slot mapping 尚未发布；
- host 认为发送完成，设备 stream 上的数据仍未完成。

---

## 4. 围绕 root 的原语：控制面与边缘路径

Broadcast、Reduce、Scatter、Gather 都有一个 root。它们当然能搬运大张量，但在在线推理中更常见于初始化、请求入口和输出出口。

### 4.1 Broadcast

Broadcast 把 root 的 $N$ 个元素复制给组内所有 rank：

~~~text
通信前：rank 0 [A B C D]   rank 1 [?]   rank 2 [?]   rank 3 [?]
通信后：rank 0 [A B C D]   rank 1 [A B C D]   rank 2 [A B C D]   rank 3 [A B C D]
~~~

推理中的用途包括：

- 首个 worker 将 input ids、positions 或采样参数分发给模型并行 ranks；
- controller 广播每轮执行命令和图选择；
- root 采样后把选中的 token id 同步给仍需继续 Forward 的 ranks；
- 初始化时分发随机种子、配置或少量权重元数据。

高吞吐引擎通常避免每层都由 root 广播大块激活，因为 root 会成为瓶颈。

### 4.2 Scatter 与 Gather

Scatter 把 root 的完整输入拆给各 rank，Gather 执行相反方向：

~~~text
Scatter：root [A|B|C|D] → rank0 [A], rank1 [B], rank2 [C], rank3 [D]
Gather： rank0 [A], rank1 [B], rank2 [C], rank3 [D] → root [A|B|C|D]
~~~

它们适合请求分发、最终输出收集、调试张量导出等边缘路径。模型热路径更常用 AllGather 或 AllToAll，因为计算通常需要对等 ranks 继续协作，而不是让全部数据回到单一 root。

### 4.3 Reduce

Reduce 对所有 rank 的同形状数据做 Sum、Max、Min 等逐元素规约，只把结果留在 root：

$$
y_{root}=\bigoplus_{r=0}^{p-1}x_r
$$

它可以汇总服务指标、局部候选分数或诊断统计。如果所有 rank 后续都需要结果，Reduce 再 Broadcast 在语义上等价于 AllReduce。

---

## 5. 推理热路径中的四个核心 collective

### 5.1 AllReduce：局部矩阵乘贡献求和

AllReduce 对各 rank 的同形状输入规约，并把完整结果交给每个 rank：

$$
y_r=\sum_{i=0}^{p-1}x_i,
\qquad
\forall r\in[0,p)
$$

它最典型的推理场景是 Row Parallel Linear。设：

$$
X=[X_0,X_1,\ldots,X_{p-1}],
\qquad
W=
\begin{bmatrix}
W_0\\W_1\\\vdots\\W_{p-1}
\end{bmatrix}
$$

每个 rank 只做本地 GEMM：

$$
Y_r=X_rW_r
$$

完整输出为：

$$
Y=\sum_{r=0}^{p-1}Y_r
$$

因此 Attention 的 o_proj、MLP 的 down_proj 等 Row Parallel 层通常在本地 GEMM 后 AllReduce。通信 payload 约为 $T\times H$ 个元素，并在许多 Transformer 层重复发生。

另一个例子是 Vocab Parallel Embedding。每个 rank 只保存一段词表，对不属于本 rank 的 token 输出零，本地命中的 embedding 通过 AllReduce(Sum) 合成完整 hidden state。

### 5.2 AllGather：把分片恢复成完整张量

每个 rank 持有 $x_r\in\mathbb{R}^{m}$ 时：

$$
y_r=\operatorname{concat}(x_0,x_1,\ldots,x_{p-1}),
\qquad
y_r\in\mathbb{R}^{pm}
$$

~~~text
通信前：rank 0 [A]   rank 1 [B]   rank 2 [C]   rank 3 [D]
通信后：每个 rank 都是 [A|B|C|D]
~~~

推理中的用途包括：

- Column Parallel Linear 的下游不能继续消费 hidden shard 时恢复完整输出；
- Sequence/Context Parallel 中收集 token、Query 或 KV 分片；
- AllGather 型 MoE 将各 rank 的 token 收集后，由每个 rank 挑出属于本地 expert 的部分；
- 词表并行采样的简单实现收集完整 logits。

AllGather 的输出在每个 rank 上扩大到原来的 $p$ 倍。真正重要的优化往往不是让它更快，而是让后续算子继续消费分片，避免过早恢复完整张量。

### 5.3 ReduceScatter：聚合后继续保持分片

每个 rank 的输入被切成 $p$ 个块：

$$
x_r=[x_r^{(0)},x_r^{(1)},\ldots,x_r^{(p-1)}]
$$

ReduceScatter(Sum) 后，rank $j$ 得到：

$$
y_j=\sum_{r=0}^{p-1}x_r^{(j)}
$$

~~~text
rank 0 输入 [A0 A1 A2 A3]
rank 1 输入 [B0 B1 B2 B3]
rank 2 输入 [C0 C1 C2 C3]
rank 3 输入 [D0 D1 D2 D3]

rank 0 输出 [A0+B0+C0+D0]
rank 1 输出 [A1+B1+C1+D1]
rank 2 输出 [A2+B2+C2+D2]
rank 3 输出 [A3+B3+C3+D3]
~~~

它适合“各 rank 有局部贡献，但下一算子只需要结果分片”的路径。例如 Sequence Parallel 可以让 Row Parallel 输出从复制布局直接变成 token shard：

$$
[T,H]_{\text{partial on each TP rank}}
\xrightarrow{\text{ReduceScatter on }T}
[T/p,H]_{\text{reduced shard}}
$$

如果下一步的 RMSNorm、Router 或 MoE 能处理 sequence shard，就不必先 AllReduce 出 $p$ 份完整副本。

从结果语义看：

$$
\operatorname{AllReduce}(x)
=
\operatorname{AllGather}(\operatorname{ReduceScatter}(x))
$$

这条关系解释了很多推理优化：把 AllReduce 拆开，先保留分片，只在真正需要完整布局时再 AllGather。

### 5.4 AllToAll：把 token 重新分桶

每个 rank 把输入分成发往不同目的 rank 的块：

$$
x_i=[x_{i\rightarrow0},x_{i\rightarrow1},\ldots,x_{i\rightarrow p-1}]
$$

通信后，rank $j$ 收到：

$$
y_j=[x_{0\rightarrow j},x_{1\rightarrow j},\ldots,x_{p-1\rightarrow j}]
$$

~~~text
             目标 rank
来源 rank       0      1      2      3
    0          x00    x01    x02    x03
    1          x10    x11    x12    x13
    2          x20    x21    x22    x23
    3          x30    x31    x32    x33

rank 2 最终收到 [x02|x12|x22|x32]
~~~

AllToAll 不做数值规约，也不复制全部输入。它把张量从一种所有权布局转为另一种，最典型的用途是 MoE：

1. Router 为每个 token 选择 Top-K expert；
2. 本地 token 按目标 expert/rank 排序并分桶；
3. Dispatch AllToAll 把 token 发到 expert 所在 rank；
4. 各 rank 执行本地 grouped GEMM；
5. Combine AllToAll 把结果送回 token 来源 rank；
6. 恢复原 token 顺序并按路由权重合并。

不同目的 rank 的 token 数通常不相等，所以真实实现常用 AllToAllV 或显式 split sizes。发送实际 token 可以节省 padding 流量，却需要先交换计数并管理动态 buffer。

AllToAll 也能在 sequence shard 与 head shard 之间转换：

$$
[T/p,H]\longleftrightarrow[T,H/p]
$$

这类转置常用于 Context Parallel Attention。

---

## 6. 原语怎样落到各类推理并行

~~~mermaid
flowchart LR
    DP["DP：切请求"] --> DPN["稠密模型副本独立<br/>通常无逐层 collective"]
    TP["TP：切矩阵 / Head"] --> TPC["AllReduce / AllGather / ReduceScatter"]
    PP["PP：切层"] --> PPC["Send / Recv 激活"]
    CP["CP：切序列 / KV"] --> CPC["P2P ring / AllGather / AllToAll"]
    EP["EP：切专家"] --> EPC["AllToAllV / AllGather / 融合 Dispatch"]
    PD["P/D 分离"] --> PDC["KV P2P / RDMA"]
~~~

### 6.1 Data Parallel：稠密推理副本通常彼此独立

Dense 模型的 DP rank 各自保存完整模型，服务不同请求。正常 Forward 中，它们没有理由逐层交换激活：

~~~text
请求流
  ├─ DP rank 0：独立 Scheduler + KV Cache + Forward
  ├─ DP rank 1：独立 Scheduler + KV Cache + Forward
  └─ DP rank 2：独立 Scheduler + KV Cache + Forward
~~~

DP 的主要通信发生在负载信息、健康状态和请求迁移等控制面。此时扩 DP 的目标是吞吐和副本容量，不是把单个请求算得更快。

但 MoE + EP 会打破这种独立性。若 EP group 跨越多个 DP rank，来自不同 DP 调度器的 token 会进入同一次 expert collective。即使某个 DP rank 暂时没有请求，也可能必须执行 dummy forward 或零 token 协议，避免其他 rank 卡在 AllToAll/AllGather。

### 6.2 Tensor Parallel：每层都可能通信

TP 沿矩阵或 Attention Head 轴切分参数，是单机多卡推理最常见的并行方式。

**Column Parallel**：

$$
W=[W_0,W_1,\ldots,W_{p-1}],
\qquad
Y_r=XW_r
$$

每个 rank 得到输出维的一片。QKV 投影后的 head shard 可以直接由本地 Attention 消费，MLP gate/up shard 也可直接进入本地激活和乘法，因此不一定马上 AllGather。

**Row Parallel**：

$$
Y=\sum_{r=0}^{p-1}X_rW_r
$$

每个 rank 的本地 GEMM 只是同一输出的局部贡献，必须 AllReduce，或 ReduceScatter 成下游需要的 token shard。

一个典型 Transformer block 可以抽象为：

~~~text
复制或 token-sharded hidden states
  ↓ Column Parallel QKV
head-sharded Q/K/V
  ↓ 本地 Attention
head-sharded attention output
  ↓ Row Parallel o_proj
AllReduce / ReduceScatter
  ↓ Column Parallel gate_up
intermediate shard
  ↓ Row Parallel down_proj
AllReduce / ReduceScatter
~~~

因此 TP size 增大虽然降低单卡权重和 GEMM 宽度，却提高 collective 频率并缩小每次本地计算。Decode 小 batch 下尤其容易从计算受限变成通信延迟受限。

### 6.3 Pipeline Parallel：一次 step 穿过多个 stage

PP 沿层切分模型，每个 stage 只保存自己的 Transformer blocks。推理路径只有前向 Send/Recv，但所有 stage 必须围绕同一批 token 保持顺序一致。

PP 适合单节点放不下完整模型、而节点间链路不适合高频 TP collective 的场景。通常把 TP group 放在节点内，让 PP 跨节点，只传边界激活。

在线 Decode 中，每轮 token 很少，PP stage 可能出现空泡。连续批处理、将请求组成 micro-batch、chunked prefill 与动态 chunk 调度可以提高 stage 利用率，但也增加调度复杂度。

### 6.4 Context Parallel：让本地 Query 看到全局历史

CP 沿 token/KV 轴切长上下文。逐 token 的 Linear、Norm、MLP 可以处理本地 shard；Attention 必须建立全局可见性。

常见路径：

- **KV AllGather**：先恢复完整 KV，再让本地 Query 计算；
- **P2P ring**：KV 逐块轮转，边传边算；
- **AllToAll**：从 sequence shard 转成 head shard，计算后逆变换；
- **分层组合**：节点内 AllToAll，节点间 P2P，匹配两级链路。

Prefill 的 Query 和 KV 都较大，通信可以与块级 Attention 重叠；Decode 的新 Query 很小，但历史 KV 很长，更需要避免每步复制完整历史。

### 6.5 Expert Parallel：通信布局取决于 token 规模

EP 把不同 experts 放在不同 rank。它不只是“做两次 AllToAll”，实际实现可能在多种 dispatcher 间选择：

| 路径 | 基本思路 | 更适合 |
|---|---|---|
| AllGather 型 | 收集各 rank token，本地筛选属于本 rank experts 的 token | 小 EP group、较小 token 数、实现简单 |
| AllToAllV 型 | 只把 token 发给目标 expert rank | 较大 token 数、稀疏目的分布 |
| 融合通信计算 | Dispatch、expert GEMM、Combine 部分融合 | 固定硬件与受支持 shape |

vLLM Ascend 的 MoE runner 会结合 world size、token 数、硬件代次和配置选择 AllGather、AllToAll、MC2 或 Fused MC2。这里的 MC2 是平台融合路径，不是新的数学原语；它仍完成 token dispatch、expert compute 和 combine 的等价数据流。

### 6.6 词表并行与采样

当 LM Head 沿词表轴切分时，每个 rank 只生成：

$$
\mathrm{logits}_r\in\mathbb{R}^{T\times V/p}
$$

采样不一定要 AllGather 完整 $[T,V]$：

- 简单实现可以 AllGather 全量 logits 后采样；
- Greedy 可以比较各 shard 的局部最大值与 token id，再做全局归并；
- Top-K/Top-P 可以先提取局部候选，再只交换候选集合；
- 分布式 softmax 需要 global max 和 global sum-exp，可用小规模 AllReduce 完成归一化。

词表很大时，避免完整 logits AllGather 能显著降低显存与带宽，但采样逻辑会更复杂。

---

## 7. Prefill 与 Decode 的通信画像完全不同

| 维度 | Prefill | Decode |
|---|---|---|
| 每请求本轮 token | 多，可能是 prompt chunk | 通常少，推测解码时可多于 1 |
| $T$ 的来源 | 长 prompt 与 chunked prefill | 活跃请求数 × 本轮 token 数 |
| GEMM/Attention | 计算量大，更容易摊薄通信 | 小矩阵多，通信启动延迟突出 |
| TP collective | 消息较大，带宽重要 | 消息较小但每层重复，延迟重要 |
| CP | Query/KV 块都较大 | Query 小、历史 KV 长 |
| EP | token 多，AllToAllV 流量和负载均衡重要 | token 少，dispatcher 启动开销和融合更重要 |
| 图执行 | shape 范围较宽 | 常按固定 batch size 桶捕获 |

### 7.1 Decode 为什么容易被 AllReduce 卡住

TP Row Parallel 的 payload 约为：

$$
N=T\times H\times \operatorname{sizeof(dtype)}
$$

Decode 中 $T$ 可能只有几十或几百。单次消息不大，但一个 block 的 Attention o_proj 和 MLP down_proj 都可能通信，模型又有几十层。生成一个 token 的关键路径会重复支付很多次启动延迟：

$$
T_{\text{token}}
\approx
\sum_{\ell=1}^{L}
\left(
T_{\text{compute},\ell}
+
T_{\text{exposed comm},\ell}
\right)
$$

因此 Decode 优化更关注低延迟协议、融合、固定 buffer、图捕获和减少同步点，而不只是峰值带宽。

### 7.2 Prefill 为什么更关注带宽和 overlap

Prefill 的 $T$ 大，collective payload 随 $T\times H$ 增长，但 GEMM 与 Attention 计算也更大。分块得当时，可以把一部分通信藏在计算后面。长上下文下，CP 的 KV 交换和跨节点带宽会成为主要因素。

### 7.3 P 与 D 可以选择不同通信实现

例如 vLLM Ascend 的 MoE 路径会根据 token capacity 选择 dispatcher：大 batch Prefill 可能走 AllToAllV，小 batch Decode 可能走 MC2/Fused MC2。P/D 分离的收益因此不仅来自资源隔离，也来自两种阶段可以分别选择更合适的通信 kernel、图 shape 和 batch 策略。

---

## 8. 通信成本：延迟、带宽和等待时间

常用的粗略模型是 $\alpha$-$\beta$ 模型：

$$
T_{\text{transfer}}\approx k\alpha+n\beta
$$

- $\alpha$：每轮通信的固定启动延迟；
- $k$：算法轮数；
- $n$：实际传输字节数；
- $\beta$：每字节传输时间，即带宽的倒数。

若 collective 含规约，还可增加规约成本 $\gamma$：

$$
T_{\text{collective}}
\approx
k\alpha+n\beta+n_{\text{reduce}}\gamma
$$

但 profiler 中观察到的持续时间通常还包含到达偏差：

$$
T_{\text{observed}}
=
T_{\text{wait for slow rank}}
+
T_{\text{transfer}}
$$

这对在线推理很关键。不同 rank 的 token 数、KV 长度、expert 负载或图回放路径不一致，都可能让“慢计算”表现成“长通信”。

### 8.1 Ring AllReduce

理想 ring AllReduce 将大小为 $N$ 的张量切成 $p$ 块，先 ReduceScatter，再 AllGather。每个 rank 的发送量约为：

$$
V_{\text{ring allreduce}}
=
2\frac{p-1}{p}N
$$

粗略时间为：

$$
T_{\text{ring allreduce}}
\approx
2(p-1)\alpha
+
2\frac{p-1}{p}N\beta
$$

大消息时能充分利用带宽；小消息时，随 rank 数增长的通信轮数可能让启动延迟占主导。

### 8.2 Tree 与拓扑感知算法

树形算法的深度约为 $\log_2p$，常用于降低小消息延迟。真实 NCCL/HCCL 会结合消息大小、硬件拓扑、链路层级和可用的网络内规约能力自动选择算法。

不应假设某个算法永远最快。手工固定 ring/tree 或协议前，要分别测 Prefill 与 Decode、不同 $T$、节点内与跨节点组合。

### 8.3 AllToAll 的额外难点

AllToAll/AllToAllV 除了总字节数，还受以下因素影响：

- peer 数量和并发连接开销；
- 每个目的 rank 的 split size；
- Router 造成的 expert 热点；
- token 排序、计数交换、padding 和恢复顺序；
- 跨节点路径是否先在节点内聚合；
- expert GEMM 长尾。

因此分析 MoE 时，应把每 rank token histogram、Dispatch、Grouped GEMM 和 Combine 放在同一条时间线上。

---

## 9. Process group：同一 rank 同时属于多张通信网

一个 worker 可能同时具有：

~~~text
global_rank = 13
tp_rank     = 1
pp_rank     = 0
dp_rank     = 3
cp_rank     = 1
ep_rank     = 7
~~~

这些 rank 属于不同 process group。模型代码必须把 collective 发到正确的 group：

- TP AllReduce 只在同一层副本的 TP ranks 中执行；
- PP Send/Recv 只连接相邻 stages；
- CP 通信连接处理同一请求不同序列分片的 ranks；
- EP 通信连接持有同一组 experts 的 ranks；
- DP 控制通信连接不同请求副本。

使用 world group 代替 TP group 可能得到 shape 正确但数值错误的结果，也可能让不相关 workers 被迫进入 collective。

### 9.1 组布局要匹配物理拓扑

常见原则：

- 高频 TP collective 尽量留在节点内高速互联域；
- PP 跨节点，只传 stage 边界激活；
- 大 EP group 跨节点时重点评估 AllToAll 双向带宽和拥塞；
- CP 采用分层算法，让节点内和节点间使用不同原语；
- P/D KV 传输让源、目标 device 与 NIC 尽量保持良好亲和性。

逻辑 rank 连续不代表物理链路最短。部署前应画出：

~~~text
global rank → host → local device → NIC → TP/PP/CP/EP group
~~~

---

## 10. Collective 对齐：在线调度最容易踩的坑

### 10.1 所有成员必须以兼容顺序参与

同一 communicator 内，所有 rank 必须以一致顺序执行 collective，并满足 count、dtype 等约束：

~~~text
错误示例
rank 0: AllReduce(A) → AllGather(B)
rank 1: AllGather(B) → AllReduce(A)
~~~

即使 A、B 的 shape 都合法，两个 rank 的第一次 collective 也无法匹配。

### 10.2 调度器状态必须变成一致的数据面计划

在线服务中，每轮请求都在变化。控制面必须让同一模型并行 group 对以下信息达成一致：

- 本轮执行 Prefill、Decode 还是混合 batch；
- $T$、每请求 token 数和 padding；
- PP 中间张量 shape；
- MoE split sizes 与 capacity；
- CP 的本地 query/KV 长度；
- 使用哪个已捕获图和哪个通信 buffer。

只同步“有请求/没请求”不够；真正进入 collective 的 shape、顺序和分支也必须一致。

### 10.3 空 rank 也可能必须 Forward

Dense DP 副本可以独立空闲；当 EP group 跨 DP ranks 时，一个空闲 rank 仍是 collective 成员。vLLM 的 DP coordinator 与 dp_utils 会同步活跃状态，并在需要时让空闲 rank 执行 dummy forward，避免其他 ranks 在 MoE collective 中永久等待。

### 10.4 图捕获要求 shape 与资源稳定

CUDA Graph/NPU Graph 回放时，通信组、buffer 地址和 shape bucket 需要满足捕获时的约束。常见问题包括：

- 实际 $T$ 超过捕获容量；
- MoE token 数超过 dispatcher workspace；
- CP/EP metadata 没有随本轮请求刷新；
- 图内外通信使用不同 stream 或 communicator 顺序；
- padding 后的逻辑 token 数与物理 buffer 长度混淆。

---

## 11. 异步语义与通信计算重叠

通信库调用返回，通常只表示操作已经提交到设备 stream，并不表示接收 buffer 已经可读。消费通信结果前必须通过 Work、event 或 stream wait 建立依赖。

### 11.1 可以重叠的典型位置

- CP 环形交换下一块 KV，同时计算当前 KV block 的 Attention；
- MoE shared expert 计算与 routed expert Dispatch/Combine 重叠；
- PP 发送当前 chunk 激活，同时本 stage 处理下一 chunk；
- 多个独立投影的通信与其他 GEMM 重叠；
- P/D 场景传输已完成层的 KV，同时 Prefill 继续计算后续层。

### 11.2 过度同步与同步不足

过度同步：

- 每次 collective 后立刻做 device-wide synchronize；
- 在热路径频繁 Barrier；
- 把本可异步的 Send/Recv 串行等待。

同步不足：

- 计算 stream 提前读取未完成的接收 buffer；
- buffer 尚在发送时被下一轮复用；
- 图回放与非图 collective 在多个 stream 上顺序不一致；
- host 控制状态先发布，KV 数据仍未对目标可见。

NCCL 的 [CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)明确说明 collective 在设备上异步执行。HCCL/NPU 环境同样要按对应 stream/event 与任务队列语义判断完成。

---

## 12. 一次在线 Forward 怎样串起这些原语

下面用一个同时包含 TP、PP、CP 和 EP 的 MoE 服务说明数据流。真实部署不一定同时开启全部维度，但分析方法相同。

~~~mermaid
flowchart TD
    A["Scheduler 选出本轮请求<br/>生成 input ids / positions / metadata"] --> B["广播或复制执行计划到模型 ranks"]
    B --> C["Embedding / TP 输入准备"]
    C --> D["Column Parallel QKV<br/>保留 head shard"]
    D --> E["CP: P2P / AllGather / AllToAll<br/>获得全局上下文"]
    E --> F["本地 Attention"]
    F --> G["Row Parallel o_proj<br/>AllReduce / ReduceScatter"]
    G --> H["Router 计算 Top-K experts"]
    H --> I["EP Dispatch<br/>AllToAllV / AllGather / MC2"]
    I --> J["本地 expert GEMM"]
    J --> K["EP Combine<br/>逆向重排"]
    K --> L["Row Parallel down_proj<br/>AllReduce / ReduceScatter"]
    L --> M{"当前 PP stage 是否结束"}
    M -->|否| D
    M -->|是| N["Send hidden states 到下一 stage"]
    N --> O["最后 stage: Vocab Parallel LM Head"]
    O --> P["分布式采样 / 候选归并"]
    P --> Q["输出 token 回到 Scheduler"]
~~~

读 profiler 时，按以下顺序定位：

1. 这个通信属于 TP、PP、CP、EP 还是 KV transfer？
2. 输入 shape 和当前 $T$ 是多少？
3. 通信前沿哪个轴分片，通信后是什么布局？
4. 下一个算子为什么需要这个布局？
5. rank 是在真正传输，还是在等待晚到的 peer？
6. 该通信能否与无依赖计算重叠，或通过保持分片被取消？

---

## 13. 选原语的快速判断

| 当前状态 | 下一步需要 | 原语 |
|---|---|---|
| 每 rank 有同形状局部贡献 | 每 rank 都要完整结果 | AllReduce |
| 每 rank 有同形状局部贡献 | 每 rank 只要结果的一片 | ReduceScatter |
| 每 rank 有不同分片 | 每 rank 都要完整拼接结果 | AllGather |
| token 按来源排列 | token 按目标 expert/rank 排列 | AllToAll / AllToAllV |
| 只有相邻 stage 需要激活 | 指定 peer 收到 | Send/Recv |
| P 实例持有 KV，D 实例需要 KV | 指定目标 cache 写入 | P2P / RDMA / connector |
| root 有输入或控制状态 | 所有模型 ranks 都要副本 | Broadcast |
| 各 rank 有分片，只有 root 要完整值 | root 汇总 | Gather |
| 只想检查谁没有到达 | 所有 rank 会合 | Barrier |

做性能优化前，再问：

1. 下游能否继续消费 shard？
2. AllReduce 能否换成 ReduceScatter？
3. 完整 logits 是否真的需要 AllGather？
4. CP 能否边交换 KV 边计算？
5. MoE 应按当前 token 规模选择 AllGather、AllToAllV 还是融合路径？
6. 通信组是否落在合适的物理互联域？

---

## 14. 常见故障与调试清单

### 14.1 Collective hang

- 所有 group 成员是否进入同一个 collective；
- 调用顺序、shape、count 和 dtype 是否一致；
- 某个 rank 是否更早发生 OOM、device error 或请求调度异常；
- 空 DP rank 在 EP 模式下是否执行了协议要求的 dummy forward；
- PP Send/Recv 的 peer 和本轮 $T$ 是否匹配；
- 多个 process group 的跨组调用是否形成循环等待。

### 14.2 数值或 token 结果错误

- 需要求和时是否误用了拼接，或反之；
- AllGather 的 rank 顺序是否对应正确分片轴；
- ReduceScatter 的输出 shard 是否被当成完整 tensor；
- MoE Combine 是否恢复原 token 顺序并应用正确路由权重；
- 分布式 softmax 的 global max、sum-exp 和 token id 偏移是否正确；
- 异步通信结果是否在完成前被消费；
- padding token 是否误进入采样或 expert 统计。

### 14.3 TTFT 高

优先看：

- Prefill TP/CP 大消息的有效带宽；
- 长 prompt 是否启用合理的 chunked prefill；
- PP stage 是否负载均衡；
- P/D KV 传输是否位于关键路径；
- 大 batch MoE 是否被 AllToAllV 或热点 expert 限制；
- 跨节点 group 是否正确绑定 NIC。

### 14.4 TPOT 高

优先看：

- Decode 每层小 AllReduce 的启动延迟；
- TP size 是否过大，导致本地 GEMM 太小；
- graph replay 是否频繁退回 eager；
- EP dispatcher 是否适合小 token batch；
- collective 前是否存在慢 rank 到达偏差；
- 是否有不必要的完整 logits AllGather 或同步点。

### 14.5 调试入口

在可复现的小规模任务中，可临时启用：

~~~bash
TORCH_CPP_LOG_LEVEL=INFO
TORCH_DISTRIBUTED_DEBUG=DETAIL
NCCL_DEBUG=INFO
~~~

TORCH_DISTRIBUTED_DEBUG=DETAIL 会增加一致性检查和日志开销，只适合定位问题。PyTorch 的 monitored_barrier() 可以帮助报告未按时到达的 rank，但不应放在服务热路径。

昇腾环境中应结合 HCCL 日志、rank table、设备侧错误和 vLLM Ascend profiler，把问题分为通信域初始化、建链、任务下发和设备执行几个阶段。

---

## 15. 总结

从推理角度看，通信原语围绕的是激活、KV、token 和 logits：

- **TP**用 AllReduce、AllGather、ReduceScatter 恢复矩阵切分后的代数结果；
- **PP**用 Send/Recv 把 $[T,H]$ 激活交给下一 stage；
- **CP**用 P2P、AllGather 或 AllToAll 让本地 Query 看到全局上下文；
- **EP**用 AllToAllV、AllGather 或融合路径完成 token Dispatch/Combine；
- **词表并行**只交换采样真正需要的全局统计或候选；
- **P/D 分离**在实例之间迁移 KV Cache，并由控制面维护 block 所有权。

Prefill 更容易受大消息带宽、CP 和 MoE 负载均衡影响；Decode 更容易受大量小 collective 的启动延迟影响。线上出现“通信慢”时，应先区分真实传输与慢 rank 等待，再检查张量布局、process group、调度对齐、图 shape 和物理拓扑。

---

## 参考资料

1. [NCCL Collective Operations](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html)
2. [NCCL CUDA Stream Semantics](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/streams.html)
3. [NCCL Environment Variables and Algorithm Selection](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
4. [PyTorch Distributed Communication Package](https://docs.pytorch.org/docs/stable/distributed.html)
5. [vLLM Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling.html)
6. [vLLM Ascend Sequence Parallelism](https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/sequence_parallelism.html)
7. [CANN 8.5 通信库文档](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/850/index/index.html)
8. [大模型并行策略与切分：结合 vLLM 和 vLLM Ascend 源码](../ai/llm-parallelism-vllm-vllm-ascend.md)
9. [DeepSeek-V4-Flash 在 vLLM Ascend 中的一次推理调用过程](../ai/deepseek-v4-flash-inference-walkthrough.md)

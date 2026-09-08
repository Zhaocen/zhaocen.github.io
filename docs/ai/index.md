# AI 推理

记录大模型推理引擎的架构理解、性能优化与平台适配经验。

## 本板块内容

<div class="grid cards" markdown>

-   __[大模型并行策略与切分：结合 vLLM 和 vLLM Ascend 源码](llm-parallelism-vllm-vllm-ascend.md)__

    ---

    从请求、层、矩阵、序列与专家五个切分轴，拆解 DP、PP、TP、CP、EP 在 vLLM 与昇腾 NPU 上的实现。

-   __[vLLM 启动服务全流程代码级剖析](vllm-startup-internals.md)__

    ---

    以 DeepSeek-V4-Flash 在昇腾 NPU 上 4P1D + EP 部署为例，逐阶段拆解 vllm serve 的启动链路。

-   __[vLLM 的 KV Cache 机制剖析](vllm-kv-cache.md)__

    ---

    PagedAttention 分页管理、Prefix Caching 链式哈希、profiling 显存预算，以及混合注意力架构下的 Hybrid KV Cache Manager。

-   __[DeepSeek-V4-Flash 推理调用全过程](deepseek-v4-flash-inference-walkthrough.md)__

    ---

    从 OpenAI API 到采样输出，按调用链顺序拆解 mHC 残差流、混合压缩稀疏注意力、MoE 与 MTP 推测解码。

</div>

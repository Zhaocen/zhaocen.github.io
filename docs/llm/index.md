# 大模型

关于大语言模型本身的笔记：模型结构、训练与微调、能力边界与使用技巧。

!!! abstract "本板块状态"

    正在建设中，下面是规划的内容方向。

## 内容方向

<div class="grid cards" markdown>

-   __[主流大模型注意力机制：DeepSeek、GLM 与 Qwen](mainstream-attention-mechanisms.md)__

    ---

    用统一框架比较 CSA/HCA、MLA+DSA+IndexShare，以及 Gated DeltaNet+周期性全注意力。

-   __[DeepSeek-V4-Flash 模型结构解析](deepseek-v4-flash-architecture.md)__

    ---

    4 路 mHC 残差流、CSA/HCA 混合压缩注意力、全层 MoE，以及 MTP 草稿层的结构与张量流。

-   __[GLM-5.2 模型结构解析](glm-5.2-architecture.md)__

    ---

    从配置到源码拆解 78 层 MoE、MLA、DSA、IndexShare 和 MTP 的完整结构。

-   :material-graph-outline:{ .lg .middle } __模型架构__

    ---

    Transformer 变体、注意力机制演进、MoE 稀疏架构的设计取舍。

-   :material-school-outline:{ .lg .middle } __训练与微调__

    ---

    预训练、SFT、RLHF/DPO 的流程差异，以及 LoRA 等参数高效微调方法。

-   :material-ruler-square:{ .lg .middle } __上下文与长文本__

    ---

    位置编码外推、长上下文的显存与精度代价、RAG 与长窗口的取舍。

-   :material-lightbulb-on-outline:{ .lg .middle } __应用技巧__

    ---

    提示工程、结构化输出、工具调用与 Agent 编排的实践经验。

</div>

## 与其他板块的关系

| 板块 | 侧重 |
|------|------|
| **大模型**（本板块） | 模型本身——结构、训练、能力 |
| [AI 推理](../ai/index.md) | 模型怎么跑起来——引擎、显存、吞吐 |
| [模型评测](../benchmark/index.md) | 模型跑得好不好——精度与性能度量 |

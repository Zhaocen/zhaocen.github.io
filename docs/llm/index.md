# 大模型

关于大语言模型本身的笔记：模型结构、训练与微调、能力边界与使用技巧。

!!! abstract "本板块状态"

    正在建设中，下面是规划的内容方向。

## 内容方向

<div class="grid cards" markdown>

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

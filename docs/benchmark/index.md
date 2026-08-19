# 性能评测

模型评测工具的使用方法、指标定义与实测记录。

## 本板块内容

<div class="grid cards" markdown>

-   __[AISBench 使用指南](aisbench.md)__

    ---

    基于 OpenCompass 的评测工具，覆盖精度测评与性能压测两类场景。

</div>

## 常用指标速查

| 指标 | 含义 | 关注点 |
|------|------|--------|
| **TTFT** | Time To First Token，首 token 延迟 | 交互体验 |
| **TPOT** | Time Per Output Token，单 token 生成耗时 | 输出流畅度 |
| **Throughput** | 单位时间处理的 token 数 | 吞吐成本 |
| **QPS** | 每秒完成的请求数 | 服务容量 |

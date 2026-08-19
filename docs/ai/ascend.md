---
tags:
  - 昇腾
  - NPU
  - vLLM
---

# 昇腾适配

!!! abstract "本页状态"

    占位页面，逐步补充中。

## 背景

`vllm-ascend` 是 vLLM 在昇腾 NPU 上的硬件插件，通过 vLLM 的
Platform 插件机制接入，避免侵入主干代码。

## 环境要点

```bash
# 检查 NPU 状态
npu-smi info

# 确认 CANN 版本
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg
```

## 待补充

- [ ] 算子适配的常见报错与定位方法
- [ ] 图模式与单算子模式的性能差异

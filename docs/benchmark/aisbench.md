---
tags:
  - AISBench
  - 评测
  - OpenCompass
---

# AISBench 评测工具使用指南

> 基于 https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/ 整理

---


## 一、简介

AISBench Benchmark 是基于 OpenCompass 构建的模型评测工具，兼容 OpenCompass 的配置体系、数据集结构与模型后端实现，并扩展了对服务化模型的支持能力。

### 支持的评测场景

| 场景 | 描述 |
|------|------|
| **精度测评** | 支持对服务化模型和本地模型在各类问答、推理基准数据集上的精度验证 |
| **性能测评** | 支持对服务化模型的延迟与吞吐率评估，并可进行压测场景下的极限性能测试 |

---

## 二、安装与卸载

### 环境要求

- **Python 版本**：仅支持 Python **3.10**、**3.11** 或 **3.12**
- 推荐使用 Conda 管理环境

### 安装方式

#### 方式一：源码安装（推荐）

```bash
conda create --name ais_bench python=3.10 -y
conda activate ais_bench

git clone https://github.com/AISBench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517
```

验证安装：
```bash
ais_bench -h
```

#### 方式二：一键安装

```bash
# 基本功能
pip3 install ais_bench_benchmark

# 全量功能
pip3 install ais_bench_benchmark[full]
```

### 可选依赖

```bash
# 服务化框架支持（vLLM、Triton等）
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt

# Huggingface多模态模型支持
pip3 install -r requirements/hf_vl_dependency.txt

# BFCL测评支持
pip3 install -r requirements/datasets/bfcl_dependencies.txt --no-deps

# OCRBench_v2数据集支持
pip3 install -r requirements/datasets/ocrbench_v2.txt
```

### 卸载

```bash
pip3 uninstall ais_bench_benchmark
```

---

## 三、快速入门

### 基本命令结构

```bash
ais_bench --models <模型任务> --datasets <数据集任务> [OPTIONS]
```

### 示例：服务化精度评测

```bash
# 1. 查询配置文件路径
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --search

# 2. 修改配置文件（主要修改 host_ip、host_port 等）

# 3. 执行评测
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt

# 调试模式
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug
```

### 输出目录结构

```
outputs/default/{timestamp}/
├── configs/          # 配置文件
├── logs/             # 日志文件
│   ├── eval/         # 评估日志
│   └── infer/        # 推理日志
├── predictions/      # 推理结果
├── results/          # 精度结果
└── summary/          # 汇总报告（csv/md/txt）
```

---

## 四、数据集准备

### 支持的数据集类型

| 类型 | 说明 |
|------|------|
| **开源数据集** | MMLU、GSM8K、HumanEval、MATH等 |
| **随机合成数据集** | 支持指定输入输出序列长度和请求数目 |
| **自定义数据集** | 用户自定义数据转换成固定格式 |

### 数据集存放位置

建议将开源数据集放置在 `ais_bench/datasets/` 目录下。

### 主要数据集

#### LLM类数据集
- 数学推理：GSM8K、MATH、AIME2024/2025/2026
- 代码生成：HumanEval、MBPP、LiveCodeBench
- 知识问答：MMLU、MMLU_Pro、CMMLU、CEVAL
- 推理能力：ARC、BBH、HellaSwag
- 文本摘要：XSum、LCSTS

#### 多模态类数据集
- TextVQA、VideoBench、VocalSound
- MMMU、DocVQA、Video-MME
- OCRBench_v2、RealWorldQA

#### Agent类数据集
- SWE-bench、SWE-bench Pro
- τ²-Bench、Terminal-Bench 2.0

---

## 五、评测场景

### 5.1 精度测评

#### 服务化精度测评

评估部署为服务形式的模型在特定数据集上的预测准确率。

**前置条件**：
- 可访问的服务化模型服务
- 数据集文件准备完毕

**示例命令**：

```bash
# 单任务测评
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen_4_shot_cot_str

# 多任务测评
ais_bench --models vllm_api_general_chat vllm_api_stream_chat \
           --datasets gsm8k_gen_4_shot_cot_str aime2024_gen_0_shot_chat_prompt

# 多任务并行
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen --max-num-workers 4

# 中断续测
ais_bench --models vllm_api_general_chat --datasets gsm8k_gen --reuse 20250628_151326

# 合并子数据集推理
ais_bench --models vllm_api_general --datasets ceval_gen --merge-ds

# 固定请求数测评
ais_bench --models vllm_api_general --datasets gsm8k_gen --num-prompts 100

# 推理结果重评估
ais_bench --models vllm_api_general --datasets gsm8k_gen --mode eval --reuse 20250628_151326
```

#### 本地模型精度测评

评估本地加载模型（非服务化）在不同数据集上的准确性。

**支持的模型**：
- HuggingFace Base 模型
- HuggingFace Chat 模型
- vLLM 离线推理模型

**示例命令**：

```bash
ais_bench --models hf_chat_model --datasets gsm8k_gen
```

### 5.2 性能测评

评估服务模型的运行效率（吞吐、延迟），仅支持**流式接口**类型模型。

**前置条件**：
- 模型推理服务需支持流式接口方式访问

**示例命令**：

```bash
# 基本性能测评
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt -m perf

# 稳态测试
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k \
           --summarizer stable_stage --mode perf

# 压力测试
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k \
           --summarizer stable_stage --mode perf --pressure --pressure-time 30

# 投机推理指标采集
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k \
           --mode perf --spec-decode
```

#### 性能指标说明

| 指标 | 说明 |
|------|------|
| **E2EL** | 端到端延迟（请求发送到接收全部响应的总时延） |
| **TTFT** | 首个Token返回时延 |
| **TPOT** | 输出阶段每个Token的平均生成时延 |
| **ITL** | 相邻Token间的平均间隔时延 |
| **Request Throughput** | 请求级吞吐率（请求数/秒） |
| **Output Token Throughput** | 输出Token吞吐率 |

#### 输出文件

```
performances/{model_name}/
├── {dataset}.csv           # 单次请求性能数据
├── {dataset}.json          # 端到端性能数据
├── {dataset}_details.h5    # ITL详细数据
├── {dataset}_details.json  # 完整打点明细
└── {dataset}_plot.html     # 并发可视化报告
```

---

## 六、命令行参数

### 公共参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--models` | 指定模型任务（支持多个） | `--models vllm_api_general` |
| `--datasets` | 指定数据集任务（支持多个） | `--datasets gsm8k_gen` |
| `--summarizer` | 指定结果汇总任务 | `--summarizer example` |
| `--mode / -m` | 运行模式 | `--mode perf` |
| `--reuse / -r` | 指定时间戳续测 | `--reuse 20250628_151326` |
| `--work-dir / -w` | 指定工作目录 | `--work-dir /path/to/work` |
| `--config-dir` | 配置文件目录 | `--config-dir /xxx/xxx` |
| `--debug` | 开启调试模式 | `--debug` |
| `--dry-run` | 只打印不执行 | `--dry-run` |
| `--merge-ds` | 合并同类数据集 | `--merge-ds` |
| `--num-prompts` | 指定测评条数 | `--num-prompts 500` |
| `--max-num-workers` | 并行任务数 | `--max-num-workers 4` |
| `--num-warmups` | 预热次数 | `--num-warmups 10` |

### 精度测评参数

| 参数 | 说明 |
|------|------|
| `--dump-eval-details` | 输出评测过程细节 |
| `--dump-extract-rate` | 输出评测速度 |

### 性能测评参数

| 参数 | 说明 |
|------|------|
| `--pressure` | 开启压测模式 |
| `--pressure-time` | 压测持续时间（秒） |
| `--spec-decode` | 启用投机推理指标采集 |

---

## 七、模型配置

### 7.1 服务化推理后端

支持的模型配置：

| 配置名称 | 接口类型 | 说明 |
|----------|----------|------|
| `vllm_api_general` | 文本接口 | v1/completions |
| `vllm_api_general_stream` | 流式接口 | v1/completions |
| `vllm_api_general_chat` | 文本接口 | v1/chat/completions |
| `vllm_api_stream_chat` | 流式接口 | v1/chat/completions |
| `mindie_stream_api_general` | 流式接口 | infer |
| `triton_api_general` | 文本接口 | v2/models/{model}/generate |
| `tgi_api_general` | 文本接口 | generate |

### 配置示例

```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",                    # Tokenizer路径
        model="",                   # 模型名称（空字符串自动获取）
        stream=False,
        request_rate=0,             # 请求发送频率
        retry=2,                    # 最大重试次数
        host_ip="localhost",        # 服务IP
        host_port=8080,             # 服务端口
        max_out_len=512,            # 最大输出长度
        batch_size=1,               # 最大并发数
        generation_kwargs=dict(     # 推理参数
            temperature=0.01,
            ignore_eos=False,
        )
    )
]
```

### 主要参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `host_ip` | String | 服务端IP地址 |
| `host_port` | Int | 服务端端口号 |
| `url` | String | 自定义URL路径 |
| `max_out_len` | Int | 最大输出Token数 |
| `batch_size` | Int | 请求并发批处理大小 |
| `request_rate` | Float | 请求发送速率 |
| `generation_kwargs` | Dict | 推理生成参数 |

### 7.2 本地模型后端

#### HuggingFace 模型配置

```python
from ais_bench.benchmark.models import HuggingFacewithChatTemplate

models = [
    dict(
        attr="local",
        type=HuggingFacewithChatTemplate,
        abbr='hf-chat-model',
        path='/path/to/model',
        tokenizer_path='/path/to/model',
        model_kwargs=dict(device_map="auto"),
        max_out_len=512,
        batch_size=1,
        generation_kwargs=dict(temperature=0.5, top_p=0.95)
    )
]
```

---

## 八、运行模式

### 精度评测场景

| 模式 | 说明 | 命令示例 |
|------|------|----------|
| `all` | 完整流程：推理→评估→汇总 | `--mode all` |
| `infer` | 仅执行推理阶段 | `--mode infer` |
| `eval` | 基于已有推理结果评估 | `--mode eval --reuse <时间戳>` |
| `viz` | 生成汇总报告 | `--mode viz --reuse <时间戳>` |

### 性能评测场景

| 模式 | 说明 | 命令示例 |
|------|------|----------|
| `perf` | 完整性能测试 | `--mode perf` |
| `perf_viz` | 基于已有数据生成报告 | `--mode perf_viz --reuse <时间戳>` |

---

## 九、进阶教程

### 9.1 自定义配置文件

创建配置文件组合多模型多数据集：

```python
from mmengine.config import read_base

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general

datasets = gsm8k_datasets
models = vllm_api_general
work_dir = 'outputs/demo/'
```

执行：
```bash
ais_bench ais_bench/configs/api_examples/demo_infer.py
```

### 9.2 自定义数据集

支持 `.jsonl` 和 `.csv` 格式。

#### 选择题格式 (mcq)

```json
{"question": "165+833+650+615=", "A": "2258", "B": "2263", "C": "2281", "answer": "B"}
```

#### 问答题格式 (qa)

```json
{"question": "752+361+181+933+235+986=", "answer": "3448"}
```

#### 使用命令

```bash
ais_bench --models vllm_api_general \
           --custom-dataset-path xxx/test_mcq.csv \
           --custom-dataset-data-type mcq \
           --mode all
```

### 9.3 随机合成数据集

用于性能测试，支持多种分布：

```python
synthetic_config = {
    "Type": "string",
    "RequestCount": 1000,
    "StringConfig": {
        "Input": {
            "Method": "uniform",  # uniform/gaussian/zipf
            "Params": {"MinValue": 50, "MaxValue": 500}
        },
        "Output": {
            "Method": "uniform",
            "Params": {"MinValue": 20, "MaxValue": 200}
        }
    }
}
```

使用：
```bash
ais_bench --models vllm_api_stream_chat --datasets synthetic_gen_string -m perf
```

### 9.4 稳态测试与压力测试

#### 稳态测试

测试推理服务处于稳定状态下的性能：

```bash
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k \
           --summarizer stable_stage --mode perf
```

#### 压力测试

模拟持续高压场景：

```bash
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k \
           --summarizer stable_stage --mode perf \
           --pressure --pressure-time 30
```

### 9.5 请求速率分布控制

模拟真实业务场景的流量波动：

```python
models = [
    dict(
        ...,
        request_rate=100,
        traffic_cfg=dict(
            burstiness=0.5,           # 突发性因子
            ramp_up_strategy="linear", # linear/exponential
            ramp_up_start_rps=10,
            ramp_up_end_rps=200,
        ),
    )
]
```

### 9.6 投机推理指标采集

```bash
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k \
           --mode perf --spec-decode
```

输出指标：
- Acceptance rate (%): 草稿Token采纳百分比
- Acceptance length: 每次前向推理平均采纳Token数

### 9.7 Multi-LoRA 路由

按数据样本的 `data_id` 自动路由到对应的 LoRA 适配器：

```python
models = [
    dict(
        type=VLLMCustomAPIChat,
        generation_kwargs=dict(
            temperature=0,
            lora_data_map_file='./lora_data_map.json',
        ),
    ),
]
```

`lora_data_map.json`:
```json
{"0": "LoraAdapter1", "1": "LoraAdapter2", "6": "LoraAdapter1"}
```

---

## 十、常见问题

### 10.1 安装问题

**Q: 'torch.library' has no attribute 'register_fake'**

A: torch 和 torchvision 版本不匹配，参考 PyTorch 官方文档安装匹配版本。

**Q: ImportError: Failed to import required modules**

A: 安装拓展依赖：
```bash
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```

**Q: tiktoken 编码文件下载失败**

A: 在有网络环境下载文件，传输到目标机器并设置缓存目录：
```bash
export TIKTOKEN_CACHE_DIR=path_to_tiktoken_cache_folder
```

### 10.2 配置问题

**Q: 如何查询需要修改的配置文件？**

A: 添加 `--search` 参数：
```bash
ais_bench --models vllm_api_general --datasets gsm8k_gen --search
```

**Q: 报错 "Please pass the argument 'trust_remote_code=True'"**

A: 在模型配置文件中添加：
```python
trust_remote_code=True
```

### 10.3 服务化返回报错

**Q: HTTP status 500 timeout**

A: 
- 减小 `max_out_len`
- 减小 `batch_size`
- 提高硬件配置

**Q: Connection refused**

A:
- 检查服务是否启动：`curl http://{ip}:{port}/v1/models`
- 检查防火墙配置

### 10.4 精度测评问题

**Q: 如何查看模型输出内容？**

A: 查看 `predictions/` 目录下的 JSON 文件。

**Q: 如何查看每个问题的正确错误结果？**

A: 添加 `--dump-eval-details` 参数。

**Q: 模型输出包含正确答案但精度计算异常**

A: 可能是匹配规则问题，开启 `--dump-eval-details` 查看详细结果。

### 10.5 性能测评问题

**Q: OutputTokens 与最大输出长度不一致**

A: 
- 检查服务化模型最大上下文限制
- 配置 `ignore_eos=True`

**Q: TTFT和TPOT降低，总吞吐反而下降**

A: 并发数减小导致资源竞争减弱，单请求延迟降低但并行效率下降。

---

## 附录：最佳实践案例

### DeepSeek-R1-Distill-Qwen-14B 论文复现

**硬件环境**：单张 NVIDIA A100 (80GB)

**复现目标**：
- AIME2024：Pass@1 69.7分
- MATH-500：Pass@1 93.9分

**关键配置**：

```python
# 模型配置
generation_kwargs=dict(
    temperature=0.6,  # 论文推荐
    top_p=0.95,
)

# 数据集配置
max_out_len=32768  # 最大输出32K
```

**评测命令**：

```bash
# AIME2024
ais_bench --models vllm_api_general_chat --datasets aime2024_gen_0_shot_chat_prompt

# MATH-500
ais_bench --models vllm_api_general_chat --datasets math500_gen_0_shot_cot_chat_prompt
```

---

> 本文档基于 AISBench 官方文档整理，版本以官方文档为准。
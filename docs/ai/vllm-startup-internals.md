---
tags:
  - vLLM
  - 昇腾
  - 推理引擎
  - 部署
---
# vLLM 启动服务全流程代码级剖析

## —— 以 DeepSeek-V4-Flash 在昇腾 NPU 上 4P1D + EP 部署为例

> **代码基线**（本文所有行号均以此为准，均为实际 clone 后阅读所得）
>
> | 仓库 | commit | 日期 |
> |---|---|---|
> | `vllm-project/vllm` | `f9f066d195ca079c7403d9d9447c6b1d740c348c` | 2026-08-18 |
> | `vllm-project/vllm-ascend` | `cfd93bb24f34808bdf6379ed3b664bf2289321ff` | 2026-08-18 |
>
> vLLM 主干迭代很快，若你的版本不同，请以"函数名 + 调用关系"为线索定位，行号可能漂移。

---

## 0. 前置：这个模型和这套部署到底是什么

### 0.1 DeepSeek-V4-Flash 的结构参数

直接取自 HuggingFace `deepseek-ai/DeepSeek-V4-Flash` 的 `config.json`，这些字段会在启动过程中被反复消费：

```jsonc
{
  "architectures": ["DeepseekV4ForCausalLM"],   // → 模型注册表查找键
  "model_type": "deepseek_v4",
  "num_hidden_layers": 43,
  "hidden_size": 4096,
  "num_attention_heads": 64,
  "num_key_value_heads": 1,                     // MLA：KV 头只有 1
  "head_dim": 512,
  "q_lora_rank": 1024,
  "o_lora_rank": 1024,
  "qk_rope_head_dim": 64,

  "n_routed_experts": 256,                      // ← EP 切分的分母
  "num_experts_per_tok": 6,                     // topk
  "n_shared_experts": 1,
  "moe_intermediate_size": 2048,
  "scoring_func": "sqrtsoftplus",
  "topk_method": "noaux_tc",
  "routed_scaling_factor": 1.5,
  "num_hash_layers": 3,                         // 前 3 层用 hash 路由（tid2eid）

  "index_topk": 512,                            // ← 触发 sparse attention 分支
  "index_n_heads": 64,
  "index_head_dim": 128,
  "compress_ratios": [0,0,4,128,4,128,...,4,0], // ← 触发 compress 分支，44 项
  "sliding_window": 128,

  "num_nextn_predict_layers": 1,                // MTP 草稿层
  "max_position_embeddings": 1048576,           // 1M 上下文
  "vocab_size": 129280,
  "quantization_config": {"quant_method": "fp8", "weight_block_size": [128,128]}
}
```

三个字段决定了整条 attention 后端选择链路：

- `is_deepseek_mla == True`（来自 `q_lora_rank`/`kv_lora_rank` 类字段）
- `use_sparse = hasattr(hf_text_config, "index_topk")` → `True`
- `use_compress = hasattr(hf_config, "compress_ratios")` → `True`

`compress_ratios` 长度 44 = 43 主干层 + 1 个 MTP 层。取值语义（从 `AscendMLAAttentionSpec.storage_block_size` 的用法推断）：`0` 表示该层不压缩（走完整 MLA cache），`4` 是轻压缩（对应 CSA，Compressed Sparse Attention），`128` 是重压缩（对应 HCA，Heavily Compressed Attention）。也就是**同一个模型内部存在三种不同 page 大小的 KV cache**，这正是后面必须开 hybrid KV cache manager 的原因。

昇腾侧的落地权重通常是 W8A8（如 `Eco-Tech/DeepSeek-V4-Flash-w8a8-mtp`），启动时用 `--quantization ascend` 走 vllm-ascend 自己的 modelslim 量化配置，而不是原始 FP8。

### 0.2 4P1D + EP 的物理拓扑

本文假设 8 台 8 卡昇腾节点。**4 台各跑一个独立的 Prefill 实例**（每个实例 DP4 × TP2 = 8 卡，占满一台）；**另外 4 台合起来跑一个 Decode 实例**（DP32 × TP1 = 32 卡，横跨 4 台机器）。所谓 "4P1D" 数的是**实例**不是机器：4 个 P 实例 + 1 个 D 实例，物理上是 8 台 64 卡。

```
                     ┌──────────────────────────────┐
   HTTP 请求 ───────► │   Proxy (load_balance_proxy) │
                     │   :8000                      │
                     └──┬────────────────────────┬──┘
      select_prefiller  │                        │  select_decoder
        ┌─────────┬─────┴───┬─────────┐          │
        ▼         ▼         ▼         ▼          │
   ┌─────────┬─────────┬─────────┬─────────┐     │
   │ P0 node │ P1 node │ P2 node │ P3 node │     │
   │  8 NPU  │  8 NPU  │  8 NPU  │  8 NPU  │     │
   │ DP4×TP2 │ DP4×TP2 │ DP4×TP2 │ DP4×TP2 │     │
   │ EP =  8 │ EP =  8 │ EP =  8 │ EP =  8 │     │
   │ :7100-3 │ :7100-3 │ :7100-3 │ :7100-3 │     │
   └────┬────┴────┬────┴────┬────┴────┬────┘     │
        └─────────┴────┬────┴─────────┘          │
                       ▼                         ▼
   ┌──────────────────────────────────────────────────┐
   │ D instance (1 个实例, 4 台节点): DP32×TP1, EP=32 │
   ├──────────┬──────────┬──────────┬─────────────────┤
   │ D0 node  │ D1 node  │ D2 node  │ D3 node         │
   │  8 NPU   │  8 NPU   │  8 NPU   │  8 NPU          │
   │ dp 0-7   │ dp 8-15  │ dp 16-23 │ dp 24-31        │
   │ :7100-7  │ :7100-7  │ :7100-7  │ :7100-7         │
   └──────────┴──────────┴──────────┴─────────────────┘
        ▲ KV Cache 传输（Mooncake / AscendDirect RDMA）
        └── 32 张 D 卡属于同一个 HCCL world / 同一个 EP 域
```

**几个容易被误解的点，先摆在前面：**

1. **"4P1D" 既不等于 5 台机器，也不等于 5 个 HTTP 端点。** 4 个 P 实例各占一台机器，1 个 D 实例独占四台机器，合计 8 台。而因为用了 external DP LB（`--data-parallel-rank` 显式指定），每个 DP rank 都是**独立进程、独立 API server、独立端口**，所以 proxy 实际要配 4×4 = **16 个 prefiller 端点** + 32×1 = **32 个 decoder 端点**。依据在 `vllm/entrypoints/cli/serve.py:79-81, 109-111`：`data_parallel_rank is not None` 即判定为 external LB，`api_server_count` 被强制为 1，而 `examples/external_online_dp/launch_online_dp.py` 会在一个节点上拉起 `dp_size_local` 个 `vllm serve` 进程，端口依次 `+1`。

2. **EP 域 = DP × TP，不是 TP。** P 侧 EP=4×2=**8**，每卡 256/8 = **32 个专家**；D 侧 EP=32×1=**32**，每卡 256/32 = **8 个专家**。注意 D 侧这个 32 卡的 EP 域是**横跨 4 台物理机**的——它不是"一台机器内部的事"，而是靠 `--data-parallel-address` 把 4 台机器上的 32 个独立进程拉进同一个 HCCL world。这个跨 DP（且跨机）的 EP 通信域是怎么建起来的，是本文第 5 章的重点。

3. **P 和 D 的并行策略可以完全不同**，但受连接器约束：`P_tp >= D_tp` 且 `P_tp % D_tp == 0`（`mooncake_hybrid_connector.py:1491-1495`，以及设计文档 `docs/source/developer_guide/Design_Documents/disaggregated_prefill.md` 的 Limitations 一节）。这里 2 % 1 = 0，合法。

---

## 1. 全景：一条 `vllm serve` 命令展开成了多少个进程

以 P0 节点上 `--data-parallel-rank 0` 那一个 DP rank 为例（TP=2）：

```
python -m vllm serve ...                      ← 进程 0（front-end）
├── uvicorn / FastAPI (api_server)            ← 同进程
├── AsyncLLM  (EngineClient)                  ← 同进程
├── DPCoordinator                             ← 独立进程（仅 dp_rank==0 起）
└── EngineCore-0  (DPEngineCoreProc)          ← 独立进程
    └── MultiprocExecutor
        ├── WorkerProc rank0 → NPUWorker → NPUModelRunner → NPU:0
        └── WorkerProc rank1 → NPUWorker → NPUModelRunner → NPU:1
            └── 每个 Worker 内还有 KVCacheSendingThread / RecvingThread（PD 连接器）
```

一个 P 节点 4 个 DP rank，就是 4 组这样的树（每组 2 个 Worker）；D 侧每个节点 8 个 DP rank，每组只有 1 个 Worker。

整套 4P1D 的进程账：

| 侧 | 实例数 | 每实例 DP×TP | 卡数 | EngineCore 进程 | WorkerProc | API server |
|---|---:|---|---:|---:|---:|---:|
| P | 4（各占 1 台） | 4 × 2 | 4×8 = 32 | 16 | 32 | 16 |
| D | 1（占 4 台） | 32 × 1 | 32 | 32 | 32 | 32 |
| **合计** | **5** | — | **64** | **48** | **64** | **48** |

启动过程按时间顺序可以切成 9 个阶段，下面逐个拆。

---

## 2. 阶段一：CLI 解析与插件加载（进程 0）

### 2.1 入口与 LB 模式判定

`vllm serve` 走到 `vllm/entrypoints/cli/serve.py` 的 `ServeSubcommand.cmd`：

```python
# vllm/entrypoints/cli/serve.py:79-81
is_external_lb = (
    args.data_parallel_external_lb or args.data_parallel_rank is not None
)
...
# :109-111
if args.api_server_count is None:
    if is_multi_port or is_external_lb or rust_frontend_path:
        args.api_server_count = 1
...
# :143-152
if is_multi_port:      run_dp_supervisor(args)
elif args.api_server_count < 1:  run_headless(args)
elif args.api_server_count > 1 or rust_frontend_path: run_multi_api_server(args)
else:
    args.api_server_count = None
    uvloop.run(run_server(args))       # ← 我们走这条
```

因为命令行给了 `--data-parallel-rank`，走的是最后一条：**单 API server、单进程前端**。这也是为什么 4P1D 下要开 16+32 个端口给 proxy。

### 2.2 昇腾插件是怎么被"挂进去"的

vllm-ascend 完全通过 Python entry point 机制注入，不需要改 vLLM 源码。`setup.py:515-523`：

```python
entry_points={
    "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
    "vllm.general_plugins": [
        "ascend_kv_connector  = vllm_ascend:register_connector",
        "ascend_model_loader  = vllm_ascend:register_model_loader",
        "ascend_service_profiling = vllm_ascend:register_service_profiling",
        "ascend_model         = vllm_ascend:register_model",
    ],
},
```

两组插件的加载时机完全不同，这点很关键：

| 插件组 | 加载时机 | 加载位置 | 作用 |
|---|---|---|---|
| `vllm.platform_plugins` | 第一次访问 `current_platform` 时（懒加载） | **所有进程** | 返回 `"vllm_ascend.platform.NPUPlatform"` 字符串，vLLM 据此实例化平台对象 |
| `vllm.general_plugins` | `load_general_plugins()` | 进程 0 **和** EngineCore 子进程、Worker 子进程 | 注册 KV connector、模型、loader |

`vllm/plugins/__init__.py:76-88` 的 `load_general_plugins()` 有个 `plugins_loaded` 全局标志保证幂等；注释里明确写了"plugins can be loaded multiple times in different processes"。所以在 `vllm/v1/engine/core.py:116-118`，EngineCore 子进程一开始就会**再调一次**：

```python
# vllm/v1/engine/core.py:115-118
from vllm.plugins import load_general_plugins
load_general_plugins()
```

vllm-ascend 的四个 register 函数（`vllm_ascend/__init__.py:73-110`）都会先调 `_ensure_global_patch()`：

```python
def _ensure_global_patch():
    global _GLOBAL_PATCH_APPLIED
    if _GLOBAL_PATCH_APPLIED: return
    from vllm_ascend.utils import adapt_patch
    adapt_patch(is_global_patch=True)
    _GLOBAL_PATCH_APPLIED = True
```

注释写得很直白：**vLLM 在 engine-core 子进程里加载 general plugin，E2E 测试的 conftest hook 在那里跑不到，所以影响 scheduler/engine 的全局 patch 必须通过 plugin entry point 再打一遍**。这是 monkey-patch 型适配层的典型工程妥协。

`register_connector()` 做的事情（`vllm_ascend/distributed/kv_transfer/__init__.py:21-88`）——注意它**先 pop 掉 vLLM 原生的 `MultiConnector` 再注册自己的**：

```python
if "MultiConnector" in KVConnectorFactory._registry:
    KVConnectorFactory._registry.pop("MultiConnector")
KVConnectorFactory.register_connector("MultiConnector", ..., "AscendMultiConnector")

KVConnectorFactory.register_connector("MooncakeConnectorV1", ..., "MooncakeConnector")
KVConnectorFactory.register_connector("MooncakeHybridConnector",
    "vllm_ascend.distributed.kv_transfer.kv_p2p.mooncake_hybrid_connector", "MooncakeConnector")
KVConnectorFactory.register_connector("MooncakeLayerwiseConnector", ...)
KVConnectorFactory.register_connector("AscendStoreConnector", ...)
KVConnectorFactory.register_connector("SfaRemoteD2HConnector", ...)
...
```

`register_model()` 把 DeepSeek-V4 系列塞进 vLLM 的 `ModelRegistry`（`vllm_ascend/models/__init__.py`）：

```python
ModelRegistry.register_model("DeepseekV4ForCausalLM",
    "vllm_ascend.models.deepseek_v4.model:AscendDeepseekV4ForCausalLM")   # :6
ModelRegistry.register_model("DeepSeekV4MTPModel",
    "vllm_ascend.models.deepseek_v4.mtp:DeepSeekV4MTP")                   # :16
ModelRegistry.register_model(...,
    "vllm_ascend.models.deepseek_v4.dspark:DSparkDeepseekV4ForCausalLM")  # :19
```

所以 `config.json` 里的 `architectures: ["DeepseekV4ForCausalLM"]` 最终解析到的是**昇腾版实现**，而不是 vLLM 上游的通用实现。

---

## 3. 阶段二：EngineArgs → VllmConfig（配置定型）

`api_server.py:148` 一行触发整个配置构建：

```python
# vllm/entrypoints/openai/api_server.py:148
vllm_config = engine_args.create_engine_config(usage_context=usage_context)
```

### 3.1 `create_engine_config` 的关键节点

`vllm/engine/arg_utils.py:1943` 起，按执行顺序：

**(1) `:1953` `current_platform.pre_register_and_update()`** —— 这是昇腾插件的第一个真正 hook 点：

```python
# vllm_ascend/platform.py:261-281
@classmethod
def pre_register_and_update(cls, parser=None) -> None:
    from vllm_ascend.utils import adapt_patch
    adapt_patch(is_global_patch=True)          # 再次幂等打全局 patch

    # 把 "ascend" 塞进 --quantization 的 choices，否则 CLI 会报非法值
    if parser is not None:
        quant_action = parser._option_string_actions.get("--quantization")
        if quant_action and quant_action.choices:
            if ASCEND_QUANTIZATION_METHOD not in quant_action.choices:
                quant_action.choices.append(ASCEND_QUANTIZATION_METHOD)

    from vllm_ascend.quantization import (AscendCompressedTensorsConfig,
                                          AscendFp8Config, AscendModelSlimConfig)
    _config_deprecated_logging()
```

`--quantization ascend` 能用，就是靠这里动态改 argparse 的 choices。

**(2) `:1976` `create_model_config()`** —— 读 HF config，`trust_remote_code=True` 使 `deepseek_v4` 这个 `model_type` 能被加载。此处确定 `is_moe`、`is_deepseek_mla`、`max_model_len` 等。

**(3) `:2081-2238` DP 参数推导** —— external LB 分支：

```python
# arg_utils.py:2122-2123
data_parallel_external_lb = (
    self.data_parallel_external_lb or self.data_parallel_rank is not None
)
# :2142-2160
if data_parallel_external_lb:
    if self.data_parallel_rank is None: raise ValueError(...)
    if self.data_parallel_size_local not in (1, None): raise ValueError(...)
    data_parallel_size_local = 1          # ← 关键：本进程只管 1 个 DP rank
    self.data_parallel_hybrid_lb = False
```

注意 **`data_parallel_size_local` 被强制成 1**。这就是 external DP 的本质：P 侧 `--data-parallel-size 4` 只是告诉这个进程"整个 DP 组有 4 个成员"，而它自己只负责其中 1 个。`data_parallel_address` / `data_parallel_rpc_port`（`:2214-2238`）用于同组进程互相 rendezvous。

**D 侧同理，但规模和跨机方式不同**：`--data-parallel-size 32`，32 个进程分布在 4 台机器上（每台 8 个），它们填的 `--data-parallel-address` 必须是同一个（D0 节点的 IP）。这是 D 实例能跨机组成**单个** DP 组的前提。

**(4) `:2244-2295` `ParallelConfig(...)`** —— 这一坨里跟我们相关的：

```python
parallel_config = ParallelConfig(
    tensor_parallel_size = 2,           # D 侧为 1
    data_parallel_size = 4,             # D 侧为 32
    data_parallel_rank = 0..3,          # D 侧为 0..31（跨 4 台机器连续编号）
    data_parallel_external_lb = True,
    data_parallel_size_local = 1,
    data_parallel_master_ip = <P 节点 IP>,
    data_parallel_rpc_port = 12321,
    is_moe_model = model_config.is_moe,          # :2262
    enable_expert_parallel = True,               # :2263  ← --enable-expert-parallel
    all2all_backend = self.all2all_backend,      # :2265
    enable_eplb = self.enable_eplb,              # :2272
    eplb_config = self.eplb_config,              # :2273
    expert_placement_strategy = ...,             # :2274
    worker_cls = "auto",                         # 后面被 Ascend 改写
    ...
)
```

`ParallelConfig.__post_init__`（`vllm/config/parallel.py:832-838`）：

```python
self.world_size = (self.pipeline_parallel_size
                   * self.tensor_parallel_size
                   * self.prefill_context_parallel_size)
```

**`world_size` 不含 DP**（P 侧 = 2，D 侧 = 1），而 `world_size_across_dp = world_size * data_parallel_size`（P 侧 2×4 = **8**，D 侧 1×32 = **32**，见 `parallel.py:549-551`）。记住这两组数，第 5 章会用到。

**(5) `:2509-2537` 组装 `VllmConfig`** —— `kv_transfer_config`（PD 的核心）、`compilation_config`、`additional_config`（昇腾私有配置的载体）都在这里进入。

### 3.2 `VllmConfig.__post_init__` 里的两个平台 hook

`vllm/config/vllm.py:1035` 的 `__post_init__` 里有两处调平台：

```python
# vllm/config/vllm.py:1360
current_platform.apply_config_platform_defaults(self)
...
# vllm/config/vllm.py:1578
current_platform.check_and_update_config(self)
```

### 3.3 `NPUPlatform.check_and_update_config`：昇腾配置改写的总闸

这是整个 vllm-ascend 里信息密度最高的函数之一（`vllm_ascend/platform.py:345-393`），注释把它切成了 10 步：

```python
@classmethod
def check_and_update_config(cls, vllm_config: VllmConfig) -> None:
    # 1. 日志
    configure_ascend_file_logging(); configure_ascend_logging()

    # 2. 早退检查 + 并行配置校验
    if device_config.device_type != cls.device_type: return
    if vllm_config.model_config is None: return
    _validate_draft_decode_context_parallel_config(vllm_config)
    _validate_parallel_config(vllm_config)

    # 3. 自动探测量化方式（读权重目录里的 quant_model_description 等）
    maybe_auto_detect_quantization(vllm_config)

    # 4. 修正与昇腾不兼容的配置项
    _fix_incompatible_config(vllm_config)

    # 5. 解析 additional_config → AscendConfig，并做互斥校验
    ascend_config = init_ascend_config(vllm_config)
    _check_ascend_config(vllm_config, ascend_config)

    # 6. 图模式 / cudagraph mode 改写
    _update_compilation_modes(vllm_config, ascend_config)

    # 7. 重算 capture size + 设置编译后端
    _setup_compile_backend(vllm_config, compile_backend=cls.get_compile_backend())

    # 8. 选 worker 类、custom op、scheduler 类
    _setup_worker_and_scheduler(vllm_config, ascend_config)

    # 9. SFA / DCP / KV / SP 一致性校验
    _validate_sfa_dcp_kv_sp(vllm_config)

    # 10. 设置 PYTORCH_NPU_ALLOC_CONF
    _set_pytorch_npu_alloc_env(vllm_config)
```

**第 8 步 `_setup_worker_and_scheduler`（`platform.py:1125-1172`）是"世界线分叉点"**：

```python
if parallel_config and parallel_config.worker_cls == "auto":
    if not vllm_config.compilation_config.pass_config.enable_sp:
        parallel_config.all2all_backend = "flashinfer_all2allv"   # :1134
    if is_310p():
        parallel_config.worker_cls = "vllm_ascend._310p.worker_310p.NPUWorker310"
    elif ascend_config.xlite_graph_config.enabled:
        parallel_config.worker_cls = "vllm_ascend.xlite.xlite_worker.XliteWorker"
    else:
        parallel_config.worker_cls = "vllm_ascend.worker.worker.NPUWorker"  # :1141

refresh_block_size(vllm_config)

if get_ascend_device_type() != AscendDeviceType._310P:
    vllm_config.compilation_config.custom_ops = ["all"]           # :1147

# scheduler 替换：dyntra LB / profiling chunk / batch-job aware
```

**`custom_ops = ["all"]`** 这一行的影响面很大：它让 vLLM 里所有可被 custom op 替换的算子（RMSNorm、RoPE、SiLU 等）都走昇腾自定义实现。

**EPLB 校验（`platform.py:765-811`）** 值得单独看，它揭示了昇腾侧目前有**两套 EPLB**：

```python
def _validate_eplb_config(vllm_config):
    use_v2_model_runner = bool(getattr(vllm_config, "use_v2_model_runner", False))
    if use_v2_model_runner:
        # V2 只认 upstream EPLB：--enable-eplb + eplb_config.load_collection_phase
        if os.getenv("DYNAMIC_EPLB", ...) or os.getenv("EXPERT_MAP_RECORD", ...):
            raise ValueError("DYNAMIC_EPLB and EXPERT_MAP_RECORD are Model Runner V1 controls...")
        if upstream_eplb_config.use_async:
            raise ValueError("Async EPLB is not supported by Model Runner V2 on Ascend yet")
        if upstream_eplb_config.communicator not in (None, "torch_nccl", "torch_gloo"):
            raise ValueError("Do not set eplb_config.communicator on Ascend; ...")
        upstream_eplb_config.communicator = "torch_nccl"   # 映射到 HCCL
    elif vllm_config.parallel_config.enable_eplb:
        raise ValueError("Upstream EPLB is only supported by Model Runner V2 on Ascend.")
```

翻译成人话：

- **Model Runner V1**（当前默认）：用昇腾自研的动态 EPLB，走环境变量 `DYNAMIC_EPLB` / `EXPERT_MAP_RECORD` + `additional_config.eplb_config`，代码在 `vllm_ascend/eplb/`。
- **Model Runner V2**（开发中）：用 vLLM 上游的 `--enable-eplb`，`communicator` 被强制改成 `"torch_nccl"`（在昇腾上实际映射到 HCCL device process group）。
- **两者不能混用**，混了直接抛异常。

---

## 4. 阶段三：AsyncLLM 与 EngineCore 进程组

### 4.1 `AsyncLLM.from_vllm_config` → `launch_core_engines`

`api_server.py:160-169` 创建 `AsyncLLM`，内部最终走到 `vllm/v1/engine/utils.py:1070` 的 `launch_core_engines`：

```python
# vllm/v1/engine/utils.py:1078-1084
parallel_config     = vllm_config.parallel_config
dp_size             = parallel_config.data_parallel_size          # P:4   / D:32
local_engine_count  = parallel_config.data_parallel_size_local    # 1
dp_rank             = parallel_config.data_parallel_rank          # P:0..3 / D:0..31
host                = parallel_config.data_parallel_master_ip
```

**DP Coordinator 的起停逻辑（`utils.py:1100-1119`）**：

```python
run_coordinator = (
    vllm_config.needs_dp_coordinator and not offline_mode and dp_rank == 0
)
if run_coordinator:
    coordinator = DPCoordinator(parallel_config,
                                enable_wave_coordination=vllm_config.model_config.is_moe)
```

而 `needs_dp_coordinator`（`vllm/config/vllm.py:698-718`）：

```python
return self.parallel_config.data_parallel_size > 1 and (
    self.model_config is None
    or self.model_config.is_moe              # ← MoE 即使 external LB 也要 coordinator
    or not self.parallel_config.data_parallel_external_lb
)
```

**这是 MoE + DP 的一个硬性要求**：DeepSeek-V4-Flash 是 MoE，即使用了 external LB，`dp_rank == 0` 的那个进程也必须额外拉起 DPCoordinator 进程做 **wave coordination**。原因是 EP 的 all-to-all 是**集合通信**——所有 DP rank 必须同步进入/退出 forward，否则 HCCL 直接死锁。所以 4 个 P 节点各自是一个独立 DP 组，**每台机器上的 `dp_rank=0` 进程各多一个 coordinator，共 4 个**。

**D 侧则只有 1 个 coordinator。** 32 个 DP rank 属于同一个 DP 组，全局 `dp_rank == 0` 只有一个进程，它落在 D0 节点上；D1/D2/D3 三台机器上一个 coordinator 都没有，wave 同步全靠连到 D0 的那个 zmq ROUTER。**D0 因此是整个 Decode 实例的单点——它没起来，其余三台会一直卡在握手。**

**握手拓扑（`utils.py:1136-1156`）**：

```python
if dp_rank == 0:
    # rank 0 持有 Coordinator，要跟所有 core 握手
    engines_to_handshake = [CoreEngine(index=i, local=(i < local_engine_count))
                            for i in range(dp_size)]
else:
    assert local_engines_only, "..."
    engines_to_handshake = [CoreEngine(index=i, local=True)
                            for i in range(dp_rank, dp_rank + local_engine_count)]
```

`handshake_address` 由 `data_parallel_master_ip` + `data_parallel_rpc_port` 构成（`utils.py:1168-1172`），rank 0 起一个 `zmq.ROUTER` bind（`:1182-1184`），其余 rank 连过去（P 侧 rank 1–3，D 侧 rank 1–31）。**这就是 `--data-parallel-address` / `--data-parallel-rpc-port` 在同一 DP 组的所有进程里必须完全一致的原因**——对 D 侧而言，这个"所有进程"横跨 4 台机器。

### 4.2 EngineCore 子进程：`EngineCore.__init__`

`vllm/v1/engine/core.py:107-250`，执行顺序即启动顺序：

```python
load_general_plugins()                                   # :118  子进程重新加载插件
self.model_executor = executor_class(vllm_config)        # :133  ← 拉起 Worker 进程，本身是重头戏
kv_cache_config = self._initialize_kv_caches(vllm_config) # :144  ← 显存 profiling + KV 分配
self.structured_output_manager = StructuredOutputManager(vllm_config)  # :145
Scheduler = vllm_config.scheduler_config.get_scheduler_cls()           # :148
self.scheduler = Scheduler(vllm_config, kv_cache_config, ...)          # :161
if self.scheduler.connector is not None:
    self.model_executor.init_kv_output_aggregator(self.scheduler.connector)  # :175

# PD 关键：把各 Worker 的连接器握手元数据汇聚到 scheduler 侧连接器
kv_connector = self.scheduler.get_kv_connector()          # :187
if kv_connector is not None:
    xfer_handshake_metadata = self.model_executor.get_kv_connector_handshake_metadata()  # :192
    content = {}
    for worker_dict in xfer_handshake_metadata:
        if worker_dict is not None: content.update(worker_dict)
    kv_connector.set_xfer_handshake_metadata_pp_aware(content)  # :203
```

`:187-203` 这一段是 **PD 分离在 engine 层的收口**：每个 Worker 把自己的 RDMA 地址、KV cache 基址、block 布局打包成 handshake metadata，通过 `collective_rpc` 收上来，合并成 `{(pp_rank, tp_rank): metadata}` 交给 scheduler 侧连接器。后续 proxy 把请求从 P 转到 D 时，D 就是靠这份元数据知道该去哪台机器的哪个 rank 拉 KV。

### 4.3 `DPEngineCoreProc._init_data_parallel`

```python
# vllm/v1/engine/core.py:2020-2034
def _init_data_parallel(self, vllm_config):
    parallel_config = vllm_config.parallel_config
    dp_rank, dp_size = parallel_config.data_parallel_rank, parallel_config.data_parallel_size
    local_dp_rank = parallel_config.data_parallel_rank_local
    assert dp_size > 1 and local_dp_rank is not None
    assert 0 <= local_dp_rank <= dp_rank < dp_size
    dp_group, dp_store = parallel_config.stateless_init_dp_group(return_store=True)
    self.dp_group, self.dp_store = dp_group, dp_store
```

注意这是 **engine-core 层的 stateless DP group（CPU 通信）**，只用于 wave 同步、`_has_global_unfinished_reqs` 的 all-reduce 等控制面操作。它跟 Worker 层用于 EP 的 HCCL 通信域是**两回事**——下一章讲后者。

---

## 5. 阶段四：Worker 进程与并行通信域构建（EP 的真正诞生地）

### 5.1 MultiprocExecutor 拉起 Worker

`vllm/v1/executor/multiproc_executor.py:118-205`：

```python
tp_size, pp_size, pcp_size = self._get_parallel_sizes()
assert self.world_size == tp_size * pp_size * pcp_size          # :126  world_size = 2（D 侧 1）
num_local_procs = self.local_world_size * max(1, self.parallel_config.data_parallel_size_local)
set_multiprocessing_worker_envs(num_local_procs)                # :136

global_start_rank = self.local_world_size * self.parallel_config.node_rank_within_dp  # :177
for local_rank in range(self.local_world_size):
    global_rank = global_start_rank + local_rank                # :190
    unready_worker_handle = WorkerProc.make_worker_process(
        vllm_config=self.vllm_config,
        local_rank=local_rank, rank=global_rank,
        distributed_init_method=distributed_init_method, ...)    # :195-204
```

到这一步，每个 EngineCore 只知道自己那 2 个 rank（0\~1）；D 侧 TP=1，每个 EngineCore 更是只有 1 个 rank。**跨 DP 的 rank 编号是在 Worker 内部完成的。**

### 5.2 关键：跨 DP 的全局 HCCL 通信域是怎么建起来的

`NPUWorker._init_worker_distributed_environment`（`vllm_ascend/worker/worker.py:1052-1065`）：

```python
def _init_worker_distributed_environment(self) -> None:
    init_batch_invariance()
    init_distributed_environment(
        self.parallel_config.world_size,     # 2（D 侧 1）
        self.rank,                           # 0..1（D 侧恒为 0）
        self.distributed_init_method,
        self.local_rank,
        "hccl",                              # ← 后端是 HCCL
    )
    ensure_model_parallel_initialized(
        self.parallel_config.tensor_parallel_size,           # 2（D 侧 1）
        self.parallel_config.pipeline_parallel_size,         # 1
        self.parallel_config.prefill_context_parallel_size,  # 1
        self.parallel_config.decode_context_parallel_size,   # 1
    )
    init_ascend_model_parallel(self.parallel_config)         # ← 昇腾私有通信域
    ensure_ec_transfer_initialized(self.vllm_config)
```

传进去的 `world_size` 只有 2。但 `init_distributed_environment` 内部有一段**改写**（`vllm/distributed/parallel_state.py:1606-1636`）：

```python
if (config is not None
    and config.parallel_config.distributed_executor_backend != "external_launcher"
    and (config.parallel_config.nnodes > 1
         or config.parallel_config.data_parallel_size > 1)
    and not enable_elastic_ep):

    parallel_config = config.parallel_config
    # 用 DP rank 偏移本进程的 rank
    rank = parallel_config.data_parallel_rank * world_size + rank        # :1618
    # 把 world size 扩到跨 DP
    world_size = parallel_config.world_size_across_dp                     # :1620

    if parallel_config.nnodes > 1:
        ip, port = parallel_config.master_addr, parallel_config.master_port
    else:
        ip   = parallel_config.data_parallel_master_ip
        port = parallel_config.get_next_dp_init_port()                    # :1629
    distributed_init_method = get_distributed_init_method(ip, port)
```

**这七行是整个 EP-over-DP 的枢纽。** 展开到我们的 P 实例：

| DP rank | 进程内 local rank | 改写后 global rank | HCCL world_size |
|---|---|---|---|
| 0 | 0,1 | 0,1 | 8 |
| 1 | 0,1 | 2,3 | 8 |
| 2 | 0,1 | 4,5 | 8 |
| 3 | 0,1 | 6,7 | 8 |

四个**互相独立的 EngineCore 进程**下面的 8 个 Worker，通过 `data_parallel_master_ip:port` 这个 rendezvous 点，加入了**同一个 8 rank 的 HCCL world**。之后 `initialize_model_parallel` 里 `world_size = torch.distributed.get_world_size()` 拿到的就是 8 而不是 2。

**D 实例是同一套机制，只是尺度和跨机范围完全不同**：`world_size = 1`，`data_parallel_size = 32`，于是 `rank = dp_rank × 1 + 0 = dp_rank`，32 个 rank 直接就是 32 个 dp_rank：

| 节点 | 该节点上的 dp_rank | 改写后 global rank | HCCL world_size |
|---|---|---|---|
| D0 | 0–7 | 0–7 | 32 |
| D1 | 8–15 | 8–15 | 32 |
| D2 | 16–23 | 16–23 | 32 |
| D3 | 24–31 | 24–31 | 32 |

**这 32 个 rank 分布在 4 台物理机上，却处在同一个 HCCL world、同一个 EP 域里。** 换句话说，D 侧的 all-to-all 有相当一部分要走机间网络，而不是像 P 侧那样全在机内高速互联上——这是 4 节点 D 实例和单节点 D 实例在性能特征上最大的区别，`HCCL_BUFFSIZE`、RoCE 网卡配置在 D 侧因此比 P 侧更敏感。

> **实践含义**：`--data-parallel-address` 必须是 DP 组内所有节点都能路由到的 IP，`--data-parallel-rpc-port` 附近的若干端口（`get_next_dp_init_port` 会递增取端口，见 `parallel.py:578-584`）都不能被占用。P 侧这个地址就是本机 IP（DP 组不跨机），**D 侧四台机器必须统一填 D0 的 IP**——填成各自本机 IP 是这类部署最常见的错误，现象是四台机器各自组了个 8 卡 world，然后集体卡在 "waiting for all ranks"。

### 5.3 vLLM 侧的通信域切分

`vllm/distributed/parallel_state.py:1826-1832` 建立五维 rank 网格：

```python
# layout: ExternalDP × DP × PP × PCP × TP
all_ranks = torch.arange(world_size).reshape(
    -1, data_parallel_size, pipeline_model_parallel_size,
    prefill_context_model_parallel_size, tensor_model_parallel_size)
# 我们的 P 实例：(1, 4, 1, 1, 2)；D 实例：(1, 32, 1, 1, 1)
```

然后各维度分别 transpose 到最后再 unbind：

| 组 | 代码位置 | group_ranks（P 实例，DP4×TP2） | group_ranks（D 实例，DP32×TP1） |
|---|---|---|---|
| `_TP` | `:1837` `all_ranks.view(-1, tp)` | `[0,1], [2,3], [4,5], [6,7]` | `[0], [1], ... [31]`（退化） |
| `_DP` | `:1908` `transpose(1,4).reshape(-1, dp)` | `[0,2,4,6], [1,3,5,7]` | `[0..31]` |
| `_EP` | `:1927-1936` `transpose(1,2).reshape(-1, dp*pcp*tp)` | **`[0..7]` 一个 8 卡组** | **`[0..31]` 一个 32 卡组（跨 4 机）** |
| `_EPLB` | `:1963-1978` 与 `_EP` 同 ranks，独立 PG | `[0..7]` | `[0..31]` |

`_EP` 只在 MoE 模型下创建（`:1926` `if config.model_config is None or config.model_config.is_moe`）。`_EPLB` 单独建一个进程组的理由注释写得很清楚（`:1957-1960`）：**避免 EPLB 的 torch.distributed 调用和 MoE forward 的集合通信在同一个 PG 上互相插队导致死锁。**

### 5.4 昇腾私有通信域：MC2 group

`vllm_ascend/distributed/parallel_state.py:21-134` 的 `init_ascend_model_parallel` 在 vLLM 通信域之上再建几个：

```python
world_size = torch.distributed.get_world_size()          # P:8 / D:32
backend    = torch.distributed.get_backend(get_world_group().device_group)   # hccl
all_ranks  = torch.arange(world_size).reshape(-1, dp, pp, pcp, tp)   # :37-43

# ---- PD 非对称 TP 的 prefill TP 组 ----
pd_tp_ratio   = get_ascend_config().pd_tp_ratio           # :45
pd_head_ratio = get_ascend_config().pd_head_ratio
if pd_head_ratio > 1 and get_current_vllm_config().kv_transfer_config.is_kv_producer:
    ...
    _P_TP = init_model_parallel_group(group_ranks, ..., group_name=f"p_tp_{num}")  # :73

# ---- MC2 组：与 vLLM 的 _EP 同布局 ----
group_ranks = (all_ranks.transpose(1, 2)
               .reshape(-1, global_dp_size * global_pcp_size * global_tp_size)
               .unbind(0))                                # :76-83
_MC2 = init_model_parallel_group(group_ranks, ..., group_name="mc2")   # :87

if get_ascend_config().eplb_config.dynamic_eplb:
    _DYNAMIC_EPLB = init_model_parallel_group(group_ranks, ..., group_name="dynamic_eplb")  # :91

# ---- 细粒度 TP 组：oproj / lmhead / embedding / mlp ----
otp_size          = ...finegrained_tp_config.oproj_tensor_parallel_size
lmhead_tp_size    = ...lmhead_tensor_parallel_size
embedding_tp_size = ...embedding_tensor_parallel_size
mlp_tp_size       = ...mlp_tensor_parallel_size
if otp_size > 0:          _OTP      = _create_or_get_group(otp_size, "otp")        # :127-128
if lmhead_tp_size > 0:    _LMTP     = _create_or_get_group(lmhead_tp_size, "lmheadtp")
if embedding_tp_size > 0: _EMBED_TP = _create_or_get_group(embedding_tp_size, "emtp")
if mlp_tp_size > 0:       _MLP_TP   = _create_or_get_group(mlp_tp_size, "mlptp")
```

几点解读：

- **MC2 组和 `_EP` 组 ranks 完全一致**（都是 `transpose(1,2).reshape(-1, dp*pcp*tp)`），但是**独立的 HCCL communicator**。MC2 走的是 `torch_npu.npu_moe_distribute_dispatch[_v2]` / `npu_moe_distribute_combine[_v2]` 这对融合算子（见 `vllm_ascend/ops/fused_moe/token_dispatcher.py:247-352`），把 all-to-all dispatch/combine 与专家计算的调度融进 kernel，需要自己的通信域。

- **`_P_TP`（prefill TP 组）只在 `pd_head_ratio > 1` 且本实例是 kv_producer 时创建**（`parallel_state.py:51`），用于 GQA 类模型在 PD 非对称 TP 下按 KV head 重切分。

  **本例中它不会被创建。** 原因在 `ascend_config.py:203-207`：

  ```python
  self.pd_tp_ratio = 1; self.pd_head_ratio = 1; self.num_head_replica = 1
  if (vllm_config.kv_transfer_config is not None
      and vllm_config.model_config is not None
      and not vllm_config.model_config.is_deepseek_mla):     # ← MLA 直接跳过
      prefill_tp_size = ...extra_config["prefill"]["tp_size"]
      decode_tp_size  = ...extra_config["decode"]["tp_size"]
      assert prefill_tp_size % decode_tp_size == 0
      self.pd_tp_ratio = prefill_tp_size // decode_tp_size
      ...
  ```

  DeepSeek-V4 是 MLA，整个分支被跳过，`pd_tp_ratio` / `pd_head_ratio` 保持 1。**MLA 的 KV 是不分头的 latent 表示，P 侧 TP=2 的每一张卡都持有完整的 KV latent，D 侧 TP=1 直接整份拉走即可，不需要任何 head 维度的重排。** 这跟第 8.3 节 `tp_num_need_pulls = 1` 是同一件事的两个侧面。
- **细粒度 TP 组**是 D 侧调优的常用手段。比如生产配置里常见的 `"finegrained_tp_config": {"lmhead_tensor_parallel_size": 16}`：D 侧 TP=1、DP=32，lm_head 那个 `[4096, 129280]` 的大矩阵在单卡上算太慢，就跨 DP 拉 TP 组专门算 lm_head。`_create_or_get_group`（`:102-118`）沿 DP 维度切分，`num_chunks = global_dp_size // group_size` = 32 // 16 = **2 个 16 卡组**。

  这里要留意一个 4 节点 D 实例特有的坑：**16 卡一组意味着每组恰好跨两台机器**（rank 0–15 = D0+D1，rank 16–31 = D2+D3），lm_head 的 all-gather 因此走机间网络。如果想让这个组留在机内，得把 `lmhead_tensor_parallel_size` 设成 8。

### 5.5 设备绑定

`NPUWorker._init_device`（`vllm_ascend/worker/worker.py:352-454`）里有一段针对 DP 的 local_rank 偏移：

```python
# worker.py:357-374
if (parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
        and parallel_config.data_parallel_backend != "ray"
        and parallel_config.nnodes_within_dp == 1
        and parallel_config.assigned_physical_gpu_ids is None):
    dp_local_rank = parallel_config.data_parallel_rank_local
    if dp_local_rank is None:
        dp_local_rank = parallel_config.data_parallel_index
    tp_pp_world_size = (parallel_config.pipeline_parallel_size
                        * parallel_config.tensor_parallel_size)
    self.local_rank += dp_local_rank * tp_pp_world_size
```

注释说明了背景：vLLM v0.24.0（PR #45026）取消了 DP worker 的自动设备隔离，所以 vllm-ascend 自己补上。不过在我们的部署里，`launch_online_dp.py` 已经通过 `ASCEND_RT_VISIBLE_DEVICES` 给每个 DP 进程预分了卡（`visible_devices = range(i*tp_size, (i+1)*tp_size)`），此时 `data_parallel_rank_local` 是 0，偏移量为 0，不冲突。

接着：

```python
visible_device_index = current_platform.logical_device_id_to_visible_device_id(self.local_rank)
device = torch.device(f"npu:{visible_device_index}")
torch.npu.set_device(device)                                          # :398-401
...
self.init_snapshot = MemorySnapshot(device=device)                    # :419
self.requested_memory = self.init_snapshot.total_memory * gpu_memory_utilization  # :420
if self.init_snapshot.free_memory < self.requested_memory:
    raise ValueError(f"Free memory ... is less than desired GPU memory utilization ...")
```

**这里就抓第一次显存快照**，后面的 profiling 全部以它为基准。

然后 `init_device`（`:456-475`）创建 model runner：

```python
self.device = self._init_device()
init_workspace_manager(self.device, num_ubatches=1)
if self.use_v2_model_runner:
    from vllm_ascend.worker.v2.model_runner import NPUModelRunner as NPUModelRunnerV2
    self.model_runner = NPUModelRunnerV2(self.vllm_config, self.device)
else:
    self.model_runner = NPUModelRunner(self.vllm_config, self.device)   # :471
```

---

## 6. 阶段五：模型加载与 EP 切分

### 6.1 `load_model`

`vllm_ascend/worker/model_runner_v1.py:3534-3612`：

```python
with DeviceMemoryProfiler() as m:
    if self.eplb_enable:
        # EPLB 场景下屏蔽 vLLM 的 EP 权重过滤，专家映射由昇腾 EPLB 自己管
        DefaultModelLoader._init_ep_weight_filter = mock_pass      # :3547-3551
    self.model = get_model(vllm_config=self.vllm_config)           # :3552
    if self.drafter:                                                # MTP 草稿模型
        with get_tp_context(self.drafter):
            self.drafter.load_model(self.model)                     # :3563
self.model_memory_usage = m.consumed_memory
logger.info("Loading model weights took %.4f GB", m.consumed_memory / 2**30)
```

`get_model` 通过 `ModelRegistry` 查 `DeepseekV4ForCausalLM` → `AscendDeepseekV4ForCausalLM`（`vllm_ascend/models/deepseek_v4/model.py:954`）。

**W8A8 场景下的加载加速**：生产配置里常见

```
--model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}'
```

DeepSeek-V4-Flash W8A8 权重约 280+ GB，单线程 safetensors 读盘要好几分钟；128 线程并发能把加载时间压到分钟级以内。

### 6.2 MoE 层的 EP 切分

`DeepseekV4MoE.__init__`（`model.py:251-330`）：

```python
self.ep_group = get_ep_group().device_group          # :268  ← 第 5.3 节建的 _EP
self.ep_rank  = get_ep_group().rank_in_group         # :269
self.ep_size  = self.ep_group.size()                 # :270  = 16
self.n_routed_experts = config.n_routed_experts      # :271  = 256
self.n_shared_experts = config.n_shared_experts      # :272  = 1

eplb_config = parallel_config.eplb_config            # :285
self.enable_eplb = parallel_config.enable_eplb
self.n_redundant_experts   = eplb_config.num_redundant_experts       # :288
self.n_logical_experts     = self.n_routed_experts                   # 256
self.n_physical_experts    = self.n_logical_experts + self.n_redundant_experts
self.n_local_physical_experts = self.n_physical_experts // self.ep_size   # :291

self.physical_expert_start = self.ep_rank * self.n_local_physical_experts  # :293
self.physical_expert_end   = self.physical_expert_start + self.n_local_physical_experts
```

代入我们的配置（无冗余专家）。**P、D 两侧的 EP 尺寸不同，专家切分粒度也就不同**：

```
P 实例：ep_size = DP4 × TP2 = 8
        n_local_physical_experts = 256 / 8 = 32
        rank k 持有专家 [32k, 32k+32)

D 实例：ep_size = DP32 × TP1 = 32
        n_local_physical_experts = 256 / 32 = 8
        rank k 持有专家 [8k, 8k+8)
```

**P 侧每卡 32 个专家、D 侧每卡 8 个专家**——同一份权重在两侧按不同粒度切开，这正是 PD 分离允许两边并行策略不同带来的自由度：P 是计算密集的大 batch prefill，专家切得粗一些、all-to-all 少一些更划算；D 是访存密集的小 batch decode，专家切得细才能把权重读取摊薄到更多卡上。

顺带一提，D 侧这个 32 卡 EP 域跨了 4 台机器，all-to-all 有相当比例走机间网络，这是 D 侧要调大 `HCCL_BUFFSIZE` 的直接原因。

如果开了 EPLB，`num_redundant_experts` 会 > 0：以 D 侧为例，设 32 则 `n_physical_experts = 288`，每卡 9 个物理专家，其中一部分是热点逻辑专家的副本。

**Hash 路由层**（`model.py:315-330`）是 DeepSeek-V4 的新特性：前 `num_hash_layers=3` 层用一张 `[vocab_size, num_experts_per_tok]` 的 int32 查找表 `tid2eid` 直接由 token id 决定专家，不走 gate 打分：

```python
self.hash = layer_idx < config.num_hash_layers and not is_draft_layer
if self.hash:
    self.gate.tid2eid = nn.Parameter(
        torch.zeros(config.vocab_size, config.num_experts_per_tok, dtype=torch.int32),
        requires_grad=False)
    self.gate.e_score_correction_bias = None
else:
    self.gate.tid2eid = None
    self.gate.e_score_correction_bias = nn.Parameter(
        torch.empty(config.n_routed_experts, dtype=torch.float32))
```

`129280 × 6 × 4B ≈ 3.1 MB` 一层，可忽略，但它意味着**前 3 层的专家分布是静态且与输入 token 分布强相关的**——做 EPLB 的时候这几层的负载画像跟后面 40 层完全不同。

### 6.3 MoE 通信方式的运行时选择

不是启动时定死的，而是**每个 batch 按 token 数动态选**（`vllm_ascend/ascend_forward_context.py:307-370`）：

```python
def select_moe_comm_method(num_tokens, vllm_config) -> MoECommType | None:
    if not is_moe_model(vllm_config):
        return None
    mc2_tokens_capacity = get_mc2_tokens_capacity()
    soc_version = get_ascend_device_type()

    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER            # :340-341  没开 EP 就 allgather
    elif lora_config is not None and enable_expert_parallel:
        moe_comm_type = MoECommType.ALLTOALL             # :347  LoRA+EP 只能 all2all
    elif soc_version == AscendDeviceType.A2:
        moe_comm_type = _select_a2_moe_comm_method(num_tokens, vllm_config, mc2_tokens_capacity)
    elif soc_version == AscendDeviceType.A3:
        moe_comm_type = _select_a3_moe_comm_method(num_tokens, mc2_tokens_capacity, vllm_config)
    elif soc_version == AscendDeviceType.A5:
        moe_comm_type = _select_a5_moe_comm_method(...)
    elif soc_version == AscendDeviceType._310P:
        moe_comm_type = MoECommType.ALLGATHER
```

A2 分支的逻辑（`:295-304`）最直观：

```python
world_size = vllm_config.parallel_config.world_size_across_dp
if (num_tokens is None or num_tokens <= mc2_tokens_capacity) and world_size > 1:
    return MoECommType.MC2
if world_size <= num_experts_per_tok:
    return MoECommType.ALLGATHER
return MoECommType.ALLTOALL
```

对应关系（`vllm_ascend/ops/fused_moe/moe_comm_method.py:51-62`）：

| MoECommType | 实现类 | Dispatcher | 适用 |
|---|---|---|---|
| `ALLGATHER` | `AllGatherCommImpl` | `TokenDispatcherWithAllGather` | EP 关闭 / EP 组很小 |
| `MC2` | `MC2CommImpl` | `TokenDispatcherWithMC2` | token 数 ≤ MC2 容量（典型 decode） |
| `FUSED_MC2` | `FusedMC2CommImpl` | `TokenDispatcherWithMC2` | A3 + `VLLM_ASCEND_ENABLE_FUSED_MC2=1` |
| `ALLTOALL` | `AlltoAllCommImpl` | `TokenDispatcherWithAll2AllV` | 大 batch prefill 超出 MC2 容量 |

**实践后果**：P 侧 `--max-num-batched-tokens 8192`，远超 MC2 容量，走 ALLTOALL；D 侧 `--max-num-batched-tokens 240`，稳稳落在 MC2 容量内，走 MC2/FUSED_MC2。**同一个模型在 P 和 D 上跑的是完全不同的 MoE 通信 kernel**——这也是 PD 分离能显著提速的一个隐含原因。

P 侧配置里的 `VLLM_ASCEND_ENABLE_FUSED_MC2=1` 会让 A3 优先选 `FUSED_MC2`（融合 dispatch+FFN+combine）。

---

## 7. 阶段六：显存 Profiling 与 KV Cache 分配

回到 `EngineCore._initialize_kv_caches`（`vllm/v1/engine/core.py:253-358`）：

```python
register_all_kvcache_specs(vllm_config)                       # :257
kv_cache_specs = self.model_executor.get_kv_cache_specs()     # :260

# 非因果注意力层会强制关掉 chunked prefill 和 prefix caching
if any(getattr(spec, "non_causal", False) for ws in kv_cache_specs for spec in ws.values()):
    vllm_config.scheduler_config.enable_chunked_prefill = False   # :277
    vllm_config.cache_config.enable_prefix_caching = False        # :282

available_gpu_memory = self.model_executor.determine_available_memory()   # :296
kv_cache_configs = get_kv_cache_configs(vllm_config, kv_cache_specs, available_gpu_memory)  # :307

scheduler_kv_cache_config = generate_scheduler_kv_cache_config(kv_cache_configs)  # :318
vllm_config.cache_config.num_gpu_blocks = scheduler_kv_cache_config.num_blocks    # :319
vllm_config.cache_config.block_size = min(g.kv_cache_spec.block_size
                                          for g in kv_cache_groups)               # :322
vllm_config.validate_block_size()                                                 # :327

self.model_executor.initialize_from_config(kv_cache_configs)                      # :329
if not envs.VLLM_ELASTIC_EP_SCALE_UP_LAUNCH:
    self.model_executor.compile_or_warm_up_model()                                # :331
```

### 7.1 昇腾注册的自定义 KV Cache Spec

`register_all_kvcache_specs` 会调到平台钩子（`vllm_ascend/platform.py:157-161`）：

```python
@classmethod
def register_custom_kv_cache_specs(cls, vllm_config) -> None:
    from vllm_ascend.core.kv_cache_interface import register_ascend_kv_cache_specs
    register_ascend_kv_cache_specs()
```

`vllm_ascend/core/kv_cache_interface.py` 里定义了三种：

- **`AscendMLAAttentionSpec`**（`:19`）—— 带 `compress_ratio`。注释直说了：

  > *DeepSeek-V4's `compress_ratio` controls how many scheduler tokens advance one compressed-cache token.*

  `storage_block_size`（`:36`）= `block_size * compress_ratio`，`max_memory_usage_bytes`（`:97-104`）= `cdiv(max_model_len, block_size * compress_ratio) * page_size_bytes`。

- **`AscendSFAIndexerCacheSpec`**（`:108`）—— DSA indexer 的独立 cache（`index_topk=512`、`index_n_heads=64`、`index_head_dim=128` 对应的那部分）。

- **`AscendSlidingWindowMLASpec`**（`:172`）—— 对应 `sliding_window: 128`，同样带 `compress_ratio`。

**这三种 spec 混在同一个模型里，就是为什么必须开 hybrid KV cache manager**。生产配置里的 `--no-disable-hybrid-kv-cache-manager` 就是干这个的（把默认的 disable 反过来）。

代入 `compress_ratios` 数组：43 层里有 21 层 `ratio=4`（CSA）、20 层 `ratio=128`（HCA）、3 层 `ratio=0`（不压缩）。1M 上下文下 HCA 层的 KV 占用只有全量的 1/128 —— 这正是官方宣称"1M 上下文只需 V3.2 的 10% KV cache"的来源。

### 7.2 `determine_available_memory`

`vllm_ascend/worker/worker.py:478` 起。做法和 GPU 侧一致：跑一次 `_dummy_run` 触发峰值激活显存，用

```
可用 KV 显存 = total_memory × gpu_memory_utilization − 权重 − 峰值激活 − 非 torch 占用
```

**PD 场景下 P 和 D 的 `gpu_memory_utilization` 通常不同**（示例配置里 P=0.9、D=0.95）。原因很实在：P 侧 batch 大、激活峰值高、还要留 HCCL buffer（`HCCL_BUFFSIZE=1024`），D 侧 batch 小但需要尽量多的 KV block 装并发序列。

### 7.3 `initialize_kv_cache`：分配张量 + 注册给连接器

`vllm_ascend/worker/model_runner_v1.py:3662-3723`：

```python
kv_cache_config = deepcopy(kv_cache_config)
self.may_add_encoder_only_layers_to_kv_cache_config()
apply_layerwise_kv_cache_plan(kv_cache_config, self.vllm_config)      # :3674
self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
self.initialize_attn_backend(kv_cache_config)                         # :3677
self.use_hybrid_blocks = len(self.attn_groups) > 1                    # :3678
self.may_reinitialize_input_batch(kv_cache_config)
kv_caches = self.initialize_kv_cache_tensors(kv_cache_config)         # :3691

if self.speculative_config and self.drafter is not None and ...:
    self.drafter.initialize_attn_backend(kv_cache_config, block_size) # :3707  MTP

if has_kv_transfer_group():
    get_kv_transfer_group().register_kv_caches(kv_caches)             # :3719-3720  ← PD 建链
```

**`:3719-3720` 是启动流程中 PD 连接器真正开始建链的那一刻**（详见第 8 章）。

### 7.4 Attention 后端的选择

`initialize_attn_backend` 内部经由 `NPUPlatform.get_attn_backend_cls`（`vllm_ascend/platform.py:216-242`）：

```python
use_compress = getattr(attn_selector_config, "use_compress", False)
key = (attn_selector_config.use_mla, attn_selector_config.use_sparse)

backend_map = {
    (True,  False, False): "vllm_ascend.attention.mla_v1.AscendMLABackend",
    (False, False, False): "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
    (True,  True,  False): "vllm_ascend.attention.sfa_v1.AscendSFABackend",
    (True,  False, True ): "vllm_ascend.attention.dsa_v1.AscendDSABackend",
}
return backend_map[(use_mla, use_sparse, use_compress)]
```

DeepSeek-V4-Flash：`use_mla=True`、`use_compress=True` → **`AscendDSABackend`**（`vllm_ascend/attention/dsa_v1.py:82`）。

对照一下：DeepSeek-V3.2 是 `(True, True, False)` → `AscendSFABackend`（Sparse Flash Attention）。V4 因为多了 compress_ratios 走了新的 DSA 路径。DSA 后端里能看到 `_build_sas_metadata`（`:518`）、`_build_qli_metadata`（`:566`）、`_update_compressed_caches_and_select_topk`（`:1336`）、`_mla_prolog_multistream`（`:1239`）这些方法，对应 CSA/HCA 的压缩缓存更新与 top-k 选择。

---

## 8. 阶段七：PD 分离连接器的建链

### 8.1 配置入口

P 侧：

```json
{
  "kv_connector": "MooncakeHybridConnector",
  "kv_role": "kv_producer",
  "kv_port": "36000",
  "engine_id": "0",
  "kv_connector_extra_config": {
    "prefill": {"dp_size": 4,  "tp_size": 2},
    "decode":  {"dp_size": 32, "tp_size": 1}
  }
}
```

D 侧只有 `kv_role: "kv_consumer"`、`kv_port`、`engine_id` 不同，**`kv_connector_extra_config` 必须两边一模一样**——因为它是双方计算对端 rank 映射的共同依据。

### 8.2 连接器的双身份

`MooncakeConnector.__init__`（`vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py:1082-1097`）：

```python
if role == KVConnectorRole.SCHEDULER:
    self.connector_scheduler = MooncakeConnectorScheduler(vllm_config, str(engine_id), kv_cache_config)
    self.connector_worker = None
elif role == KVConnectorRole.WORKER:
    self.connector_scheduler = None
    self.connector_worker = MooncakeConnectorWorker(vllm_config, str(engine_id), kv_cache_config)
```

**同一个类在 EngineCore 进程里是 scheduler 角色，在 Worker 进程里是 worker 角色**，两边跑完全不同的代码路径。

### 8.3 Worker 侧初始化与端口分配

`MooncakeConnectorWorker.__init__`（`:1488-1585`）：

```python
self._get_prefill_decode_size(vllm_config)                      # :1489 读 extra_config
if self._prefill_tp_size < self._decode_tp_size:                # :1491-1495
    raise ValueError(f"prefill_tp_size: {...} must be >= decode_tp_size: {...}")

self.tp_rank = get_tensor_model_parallel_rank()
self.tp_size = vllm_config.parallel_config.tensor_parallel_size
self.dp_rank = vllm_config.parallel_config.data_parallel_rank_local
self.dp_size = vllm_config.parallel_config.data_parallel_size_local
self.pcp_size, self.dcp_size = ..., ...
assert self.pcp_size * self.dcp_size == 1, "Mooncake Hybrid Connector only support cp_world_size == 1."  # :1510

self.kv_role = vllm_config.kv_transfer_config.kv_role
self.use_hybrid = (not disable_hybrid_kv_cache_manager
                   and any(not isinstance(g.kv_cache_spec, FullAttentionSpec) for g in groups)
                   and len(kv_cache_config.kv_cache_groups) > 1)      # :1520-1524
self.use_compress = hasattr(self.vllm_config.model_config.hf_config, "compress_ratios")  # :1553
```

`:1553` 这行直接读 `compress_ratios` —— **连接器知道 DeepSeek-V4 的压缩 KV 布局，并为它走专门的注册分支**（见 `register_kv_caches` 的 `elif self.use_compress:` 分支，`:1659-1687`）。

**端口分配公式（`:1556-1563`）**，这是排障高频点：

```python
self.side_channel_port = (
    kv_transfer_config.kv_port
    + parallel_config.data_parallel_rank
      * parallel_config.tensor_parallel_size
      * parallel_config.pipeline_parallel_size
)
device_index = self.pp_rank * self.tp_size + self.tp_rank
self.handshake_port = self.side_channel_port + device_index
```

代入 P 实例（`kv_port=36000`, TP=2, PP=1）：

| DP rank | side_channel_port | 该 rank 占用的 handshake 端口 |
|---|---|---|
| 0 | 36000 | 36000–36001 |
| 1 | 36002 | 36002–36003 |
| 2 | 36004 | 36004–36005 |
| 3 | 36006 | 36006–36007 |

**一个 P 节点吃掉 36000–36007 共 8 个端口**（= `dp_size × tp_size`，正好等于卡数）。

**D 实例要特别注意**：公式里用的是**全局** `data_parallel_rank`（0–31）而不是节点内的 local rank，所以 `kv_port=36400` 时，四台机器的占用是天然错开的：

| 节点 | dp_rank | 该节点占用的 handshake 端口 |
|---|---|---|
| D0 | 0–7 | 36400–36407 |
| D1 | 8–15 | 36408–36415 |
| D2 | 16–23 | 36416–36423 |
| D3 | 24–31 | 36424–36431 |

也就是说**四台 D 机器填同一个 `kv_port` 就行，不需要（也不应该）人为错开**——一旦手工给 D1 改成 36500，它算出的端口就和 P 侧 metadata 里记录的对不上。整个 D 实例占用 36400–36431 共 32 个端口，但每台机器实际只监听其中 8 个。

官方文档给的选港建议（`docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md`）：昇腾上 Mooncake 用 AscendDirectTransport 做 RDMA，会在 `[20000, 20000 + npu_per_node × 1000)` 里随机取端口；**8 卡节点即 20000–27999，所以 `kv_port` 必须 ≥ 28000**（本文脚本取 36000 起，留了额外余量）。否则会偶发 `zmq.error.ZMQError: Address already in use`。

**非对称 TP 的 pull 倍率（`:1578-1583`）**：

```python
if self.vllm_config.model_config.is_deepseek_mla:
    self.tp_num_need_pulls = 1
else:
    num_d_block_heads = max(1, self.num_key_value_heads // self.tp_size)
    num_p_block_heads = max(1, self.num_key_value_heads // self._prefill_tp_size)
    self.tp_num_need_pulls = num_d_block_heads // num_p_block_heads
```

DeepSeek-V4 是 MLA，`tp_num_need_pulls = 1` —— **MLA 的 KV 是不按头切分的（latent 表示，`num_key_value_heads=1`），所以 D 侧一个 rank 只需从 P 侧一个 rank 拉一份，非对称 TP（2→1）零额外代价**。这是 MLA 系模型做 PD 分离的一大结构性红利；GQA 模型这里就得 pull 多份再拼。

### 8.4 `register_kv_caches`：注册 buffer + 拉起收发线程

`:1609-1752`：

```python
self.use_mla    = self.vllm_config.model_config.is_deepseek_mla      # :1611
self.use_sparse = hasattr(hf_text_config, "index_topk")              # :1612
self.num_blocks = self.kv_cache_config.num_blocks

# 按 use_hybrid / use_mamba / use_compress 走不同分支收集基址与 block 布局
elif self.use_compress:                                              # :1659-1687
    for kv_cache_tensor in self.kv_cache_config.kv_cache_tensors:
        ...
        self.kv_caches_base_addr.append(min(share_tensor_addr))
        self.addr_group_idx.append(cur_tensor_group_idx)
        self.block_stride_per_addr.append(share_tensor_stride[0])
        ptrs.append(min(share_tensor_addr)); lengths.append(kv_cache_tensor.size)

global_te.register_buffer(ptrs, lengths)                             # :1691  ← 向 RDMA 引擎注册

metadata = MooncakeAgentMetadata(
    engine_id=self.engine_id, te_rpc_port=self.te_rpc_port,
    block_size=self.block_size, kv_caches_base_addr=self.kv_caches_base_addr,
    num_blocks=self.num_blocks, block_lens=self.block_len_per_addr,
    ssm_sizes=self._mamba_ssm_size, local_ip=get_ip())                # :1693-1702
self.xfer_handshake_metadata = metadata

ready_event = threading.Event()
if self.kv_role == "kv_producer":
    self.kv_send_thread = KVCacheSendingThread(...)                  # :1707-1718
    self.kv_send_thread.start()
else:
    self.kv_recv_thread = KVCacheRecvingThread(...)                  # :1720-1742
    self.kv_recv_thread.start()

# 阻塞等待线程 ready，最长 5 分钟
while not ready_event.is_set():
    if not thread.is_alive():
        raise RuntimeError("KV Cache sending/receiving thread failed to start.")
    if time.time() - start_wait_time > 5 * 60:
        raise RuntimeError("Timeout waiting for KV Cache thread to be ready.")
    time.sleep(3)                                                     # :1744-1752
```

要点：

- **P 起发送线程，D 起接收线程**，角色在启动时就固定，不能动态切换。
- `MooncakeAgentMetadata` 里带的是**裸内存地址 + block stride**。这意味着 D 侧是直接做 RDMA read 到 P 侧显存的偏移，**不经过 P 侧 CPU**。
- **这里会阻塞最长 5 分钟**。启动日志停在这里超过一分钟，基本就是 Mooncake transfer engine 起不来（`LD_LIBRARY_PATH` 没配 / 端口冲突 / RDMA 网卡没打通）。

拉起后，这份 metadata 通过 `Worker.get_kv_connector_handshake_metadata()`（`vllm_ascend/worker/worker.py:892`）被 EngineCore 用 `collective_rpc` 收走，汇聚成 `{(pp_rank, tp_rank): metadata}`（`vllm/v1/engine/core.py:192-203`）。

### 8.5 运行时的一次 PD 交接（对照 scheduler 侧代码）

启动完成后，一条请求在 4P1D 里的完整路径（连接器方法引自 `mooncake_hybrid_connector.py`，流程引自 `docs/source/developer_guide/Design_Documents/disaggregated_prefill.md`）：

```
1. Client ──► Proxy /v1/completions
2. Proxy.select_prefiller()  → 从 16 个 P 端点（4 台 × 4 DP rank）里按负载选一个
     优先级函数：entry.active_tokens + entry.active_kv_cache * 0.3
     （examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py:304-309）
   转发时注入 kv_transfer_params = {do_remote_decode: True,
                                    max_completion_tokens: 1, min_tokens: 1}
3. P 节点跑完 prefill，只出 1 个 token。
   scheduler.update_from_output → connector.request_finished_all_groups()  (:1423-1469)
     ├── 计算需要传的 block ids（_compute_transfer_block_ids + get_sw_clipped_blocks）
     ├── delay_free_blocks = True → 把 KV block 挂进 _reqs_need_send，延迟释放
     └── 返回 kv_transfer_params = {
             do_remote_prefill: True,
             remote_block_ids, remote_engine_id,
             remote_host: side_channel_host,
             remote_port: side_channel_port,
             remote_ptp_size: tp_size,
             last_token_id, num_prompt_blocks, ...}
4. Proxy.select_decoder() → 转发给 32 个 D 端点（4 台 × 8 DP rank）中的一个
5. D 节点 scheduler：
   ├── connector.get_num_new_matched_tokens()  (:1335-1370)
   │     params["do_remote_prefill"] → 返回 (count, True)，异步加载
   ├── 请求进入 RequestStatus.WAITING_FOR_REMOTE_KVS，预分配本地 block
   ├── connector.update_state_after_alloc()  (:1372-1393)
   │     记入 _reqs_need_recv，并把 do_remote_prefill 置 False（保证只传一次）
   ├── build_connector_meta() → MooncakeConnectorMetadata  (:1395-1421)
   └── Worker: start_load_kv() (:1775) → KVCacheRecvingThread
         ├── _get_remote_metadata()  向 P 的 handshake_port 要 MooncakeAgentMetadata
         ├── _transfer_kv_cache_all_groups()  RDMA read（按 kv_cache_group 分组）
         └── _send_done_recv_signal()  通知 P 释放 block
6. D 正常 decode，流式返回
```

**为什么 P 侧要设 `max_completion_tokens: 1`**：让 P 生成恰好一个 token 就结束，这样 `request.status == FINISHED_LENGTH_CAPPED`，正好命中 `request_finished_all_groups` 的判定条件（`:1438-1443`）。那个 token 通过 `last_token_id` 传给 D，D 从它继续 decode。

---

## 9. 阶段八：编译、预热与图捕获

`Worker.compile_or_warm_up_model`（`vllm_ascend/worker/worker.py:721-770`）：

```python
warmup_sizes = (compilation_config.compile_sizes or []).copy()
if not self.model_config.enforce_eager:
    cg_capture_sizes = compilation_config.cudagraph_capture_sizes or []
    warmup_sizes = [x for x in warmup_sizes if x not in cg_capture_sizes]
    # 保证每个 compile_range 至少有一个 size 被预热
    for compile_range in compile_ranges:
        if not any(x in compile_range for x in all_sizes):
            warmup_sizes.append(compile_range.end)

for size in sorted(warmup_sizes, reverse=True):
    self.model_runner._dummy_run(size)                          # :741-743

from vllm_ascend.model_executor.warmup.kernel_warmup import kernel_warmup
kernel_warmup(self)                                             # :745-747

npugraph_memory_bytes = 0
if not self.model_config.enforce_eager:
    npugraph_memory_bytes = self.model_runner.capture_model()   # :750-751
```

**P 和 D 在这一步的行为差异极大**：

| | Prefill 侧 | Decode 侧 |
|---|---|---|
| 图模式 | `--enforce-eager`（跳过 capture） | `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'` |
| 原因 | prefill 序列长度千变万化，捕获收益低、显存开销大 | decode 形状固定（`max_num_seqs × (1+spec_tokens)`），全图捕获收益最大 |
| 额外 | `enable_dsa_cp` 等 | `ascend_compilation_config: {enable_npugraph_ex: true}` |

`capture_model`（`vllm_ascend/worker/model_runner_v1.py:4891-4901`）用 `_torch_cuda_wrapper()` 和 `_replace_gpu_model_runner_function_wrapper()` 把上游 `GPUModelRunner.capture_model` 的 CUDA 调用重定向到 NPU：

```python
def capture_model(self) -> int:
    parent_module_name = _get_gpu_model_runner_module_name(self)
    with _torch_cuda_wrapper(), _replace_gpu_model_runner_function_wrapper(parent_module_name):
        cuda_graph_size = GPUModelRunner.capture_model(self)
    ...
```

这是 vllm-ascend 复用上游代码的典型手法：不 fork 逻辑，只在运行时替换设备 API。

**D 侧图捕获通常是整个启动流程里最慢的一段**（几分钟量级），因为 `FULL_DECODE_ONLY` 要对每个 capture size 都跑一遍完整 43 层前向并记录图。

---

## 10. 阶段九：API Server 就绪

回到进程 0，`api_server.py:524-560` 的 `build_and_serve`：

```python
supported_tasks = await engine_client.get_supported_tasks()      # :541
await init_app_state(engine_client, app.state, args, supported_tasks)   # :546
return await serve_http(...)                                     # :550
```

`init_app_state`（`:221-345`）会实例化各 serving 类，包括 DeepSeek-V4 专用的解析器：

- `--tokenizer-mode deepseek_v4`
- `--reasoning-parser deepseek_v4`（think / think-high / think-max 三档）
- `--tool-call-parser deepseek_v4` + `--enable-auto-tool-choice`

到这一步，`/health` 才返回 200，proxy 才能把这个端点纳入调度。

**整个 4P1D 集群的就绪顺序**：64 个 Worker（P 32 + D 32）各自完成加载→profiling→建链→图捕获 → 48 个 EngineCore 就绪 → 48 个 API server 监听 → 最后启动 proxy。**proxy 必须最后起**，否则会把请求打到还没 ready 的端点上。

跨机的 D 实例还多一层顺序约束：**D0 要先起**（它持有 DPCoordinator 和 DP 组的 zmq ROUTER），D1\~D3 才连得上。四台 D 机器同时敲命令通常也能成——连不上会重试——但排障时按 D0 → D1/D2/D3 的顺序起，日志会干净很多。

---

## 11. 完整启动时序图

```
时间 ──────────────────────────────────────────────────────────────────►

[进程0 front-end]
 │ vllm serve 解析 args
 │ ├─ is_external_lb = True → api_server_count = 1
 │ └─ platform plugin 发现 → NPUPlatform
 │ create_engine_config()
 │ ├─ pre_register_and_update()      注入 "ascend" 量化选项 + 全局 patch
 │ ├─ create_model_config()          读 HF config，is_moe / is_deepseek_mla / use_compress
 │ ├─ ParallelConfig(dp=4, tp=2, EP=True)   ← D 侧 dp=32, tp=1
 │ └─ VllmConfig.__post_init__
 │     ├─ apply_config_platform_defaults()
 │     └─ check_and_update_config()  ← 10 步昇腾改写
 │         ├─ init_ascend_config()   解析 additional_config
 │         ├─ worker_cls = NPUWorker
 │         ├─ custom_ops = ["all"]
 │         └─ all2all_backend = "flashinfer_all2allv"
 │ AsyncLLM.from_vllm_config()
 │ └─ launch_core_engines()
 │     ├─ [仅 dp_rank==0] fork DPCoordinator 进程（MoE 必需，wave 同步）
 │     ├─ zmq ROUTER bind @ dp_address:dp_rpc_port
 │     └─ fork EngineCore 进程
 │
 ├──────────► [EngineCore 进程]
 │             │ load_general_plugins()   注册 connector / model / loader
 │             │ DPEngineCoreProc._init_data_parallel()  stateless DP group (CPU)
 │             │ MultiprocExecutor._init_executor()
 │             │ └─ fork 2 × WorkerProc（D 侧 1 ×）
 │             │
 │             ├──────────► [WorkerProc × 2]
 │             │             │ NPUWorker.__init__()
 │             │             │ ├─ adapt_patch()
 │             │             │ ├─ register_ascend_customop()
 │             │             │ └─ init_ascend_config()
 │             │             │ init_device()
 │             │             │ ├─ torch.npu.set_device()
 │             │             │ ├─ MemorySnapshot 基线
 │             │             │ ├─ init_distributed_environment("hccl")
 │             │             │ │   ★ rank += dp_rank × 2 ; world_size = 8
 │             │             │ │   ★ rendezvous @ dp_master_ip:dp_init_port
 │             │             │ │   → 8 卡加入同一 HCCL world（D 侧 32 卡，跨 4 机）
 │             │             │ ├─ ensure_model_parallel_initialized()
 │             │             │ │   → _TP[2] / _DP[4] / _EP[8] / _EPLB[8]
 │             │             │ ├─ init_ascend_model_parallel()
 │             │             │ │   → mc2[8] / (p_tp) / otp / lmheadtp / emtp / mlptp
 │             │             │ └─ NPUModelRunner(vllm_config, device)
 │             │             │ load_model()
 │             │             │ └─ AscendDeepseekV4ForCausalLM
 │             │             │    DeepseekV4MoE: ep_size=8, 每卡 32/256 专家
 │             │             │    MTP drafter 一并加载
 │             │             ▼
 │             │ determine_available_memory()  ← 各 rank profiling
 │             │ get_kv_cache_configs()
 │             │   MLA(compress=4) / MLA(compress=128) / Indexer / SlidingWindow
 │             │   → hybrid KV cache manager 多组
 │             │ initialize_from_config()
 │             │ └─ [Worker] initialize_kv_cache()
 │             │     ├─ initialize_attn_backend() → AscendDSABackend
 │             │     ├─ initialize_kv_cache_tensors()
 │             │     └─ get_kv_transfer_group().register_kv_caches()
 │             │         ├─ global_te.register_buffer()   RDMA 注册
 │             │         ├─ MooncakeAgentMetadata 构造
 │             │         └─ 起 KVCacheSendingThread(P) / RecvingThread(D)
 │             │            ★ 阻塞等 ready，超时 5 分钟
 │             │ get_kv_connector_handshake_metadata()
 │             │   → 汇聚 {(pp,tp): metadata} 给 scheduler 侧连接器
 │             │ compile_or_warm_up_model()
 │             │ ├─ _dummy_run(各 warmup size)
 │             │ ├─ kernel_warmup()
 │             │ └─ capture_model()   ★ D 侧最耗时
 │             │ Scheduler 构造
 │             ▼ ready
 │
 │ init_app_state()  deepseek_v4 tokenizer / reasoning / tool parser
 └─ uvicorn 监听 :7100+i   → /health 200

[全集群]  64 Worker ready → 48 API server ready → 最后启动 Proxy
```

---

## 12. 可直接改用的 4P1D 部署脚本

### 12.1 P 节点（4 台，每台 8 卡，每台一个独立实例：DP4 × TP2，EP=8）

`run_dp_template.sh`（在 P0\~P3 上分别把 `local_ip` 改成本机 IP，`kv_port` 改成 36000/36100/36200/36300，`engine_id` 改成 0/1/2/3）：

```bash
#!/bin/bash
nic_name="eth0"
local_ip="192.0.0.1"                       # ← 每台改这里

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_BUFFSIZE=1024
export HCCL_OP_EXPANSION_MODE=AIV
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_ASCEND_ENABLE_FUSED_MC2=1      # 融合 MC2，按芯片代次决定是否开
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1
export VLLM_RPC_TIMEOUT=3600
export ASCEND_RT_VISIBLE_DEVICES=$1        # 由 launch_online_dp.py 计算

vllm serve /models/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --served-model-name dsv4-flash \
  --seed 1024 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 16 \
  --block-size 32 \
  --gpu-memory-utilization 0.9 \
  --quantization ascend \
  --trust-remote-code \
  --enforce-eager \
  --no-disable-hybrid-kv-cache-manager \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
  --speculative-config '{"num_speculative_tokens": 1, "method": "mtp"}' \
  --additional-config '{"enable_cpu_binding": true, "enable_dsa_cp": true}' \
  --kv-transfer-config '{
      "kv_connector": "MooncakeHybridConnector",
      "kv_role": "kv_producer",
      "kv_port": "36000",
      "engine_id": "0",
      "kv_connector_extra_config": {
          "prefill": {"dp_size": 4,  "tp_size": 2},
          "decode":  {"dp_size": 32, "tp_size": 1}
      }
  }'
```

拉起（每台 P 节点各执行一次，`--dp-address` 填**本机** IP——P 侧每台机器自成一个 DP 组）：

```bash
python launch_online_dp.py \
  --dp-size 4 --tp-size 2 --dp-size-local 4 --dp-rank-start 0 \
  --dp-address 192.0.0.1 --dp-rpc-port 12321 --vllm-start-port 7100
```

会起 4 个进程，端口 7100–7103，卡分别是 `0-1` / `2-3` / `4-5` / `6-7`。

> `enable_dsa_cp` 只对带 indexer 的模型生效（`utils.py:1325-1334` 检查 `hf_text_config.index_topk` 是否存在），DeepSeek-V4 满足条件；换成非 DSA 模型时这个开关会被静默忽略。

### 12.1.1 一致性自检

启动前建议先跑一遍算术：

```
P 单实例：dp_size(4) × tp_size(2) = 8 卡 = 一台 P 节点的全部     ✓
          共 4 个 P 实例，各占一台机器，互不通信                 ✓
          EP = 8，256 / 8 = 每卡 32 个专家                      ✓
          kv_port 占用 36000 + [0, 4×2) = 36000–36007          ✓
D 单实例：dp_size(32) × tp_size(1) = 32 卡 = 4 台 D 节点        ✓
          每台 dp_size_local = 8，dp_rank_start = 0/8/16/24     ✓
          4 台共用一个 dp_address（D0 的 IP）、一个 engine_id    ✓
          EP = 32，256 / 32 = 每卡 8 个专家                     ✓
          kv_port 占用 36400 + [0, 32×1) = 36400–36431         ✓
          （按全局 dp_rank 分配，四台机器自动错开，无需手工改）
约束：    prefill.tp_size(2) >= decode.tp_size(1) 且整除        ✓
整机账：  4 台 P + 4 台 D = 8 台 × 8 卡 = 64 卡                 ✓
```

### 12.2 D 节点（4 台，每台 8 卡，合起来是**一个**实例：DP32 × TP1，EP=32）

和 P 侧最大的不同：**这四台机器跑的是同一个实例**，所以下面这份脚本四台都用，只改两处——`local_ip` 改成本机 IP，`--dp-rank-start` 改成 0 / 8 / 16 / 24。`--dp-address`、`--dp-size`、`kv_port`、`engine_id` 四台必须完全一致。

```bash
#!/bin/bash
nic_name="eth0"
local_ip="192.0.0.5"                       # ← 每台改这里（D0~D3: .5/.6/.7/.8）

export HCCL_IF_IP=$local_ip
export GLOO_SOCKET_IFNAME=$nic_name
export TP_SOCKET_IFNAME=$nic_name
export HCCL_SOCKET_IFNAME=$nic_name
export HCCL_EXEC_TIMEOUT=204
export HCCL_CONNECT_TIMEOUT=120
export HCCL_BUFFSIZE=1500                  # D 侧 EP 域跨 4 机，MC2 buffer 要更大
export TASK_QUEUE_ENABLE=1
export OMP_PROC_BIND=false
export OMP_NUM_THREADS=10
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export VLLM_RPC_TIMEOUT=3600
export ASCEND_RT_VISIBLE_DEVICES=$1

vllm serve /models/DeepSeek-V4-Flash-w8a8-mtp \
  --host 0.0.0.0 --port $2 \
  --data-parallel-size $3 \
  --data-parallel-rank $4 \
  --data-parallel-address $5 \
  --data-parallel-rpc-port $6 \
  --tensor-parallel-size $7 \
  --enable-expert-parallel \
  --served-model-name dsv4-flash \
  --seed 1024 \
  --max-model-len 1048576 \
  --max-num-batched-tokens 240 \
  --max-num-seqs 60 \
  --block-size 32 \
  --gpu-memory-utilization 0.95 \
  --quantization ascend \
  --trust-remote-code \
  --async-scheduling \
  --no-enable-prefix-caching \
  --no-disable-hybrid-kv-cache-manager \
  --tokenizer-mode deepseek_v4 \
  --reasoning-parser deepseek_v4 \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice \
  --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 128}' \
  --speculative-config '{"num_speculative_tokens": 3, "method": "mtp"}' \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --additional-config '{
      "ascend_compilation_config": {"enable_npugraph_ex": true, "enable_static_kernel": false},
      "enable_cpu_binding": true,
      "multistream_overlap_shared_expert": true,
      "recompute_scheduler_enable": true
  }' \
  --kv-transfer-config '{
      "kv_connector": "MooncakeHybridConnector",
      "kv_role": "kv_consumer",
      "kv_port": "36400",
      "engine_id": "4",
      "kv_connector_extra_config": {
          "prefill": {"dp_size": 4,  "tp_size": 2},
          "decode":  {"dp_size": 32, "tp_size": 1}
      }
  }'
```

四台机器分别执行（注意 `--dp-size` 恒为 32、`--dp-address` 恒为 D0 的 IP，只有 `--dp-rank-start` 递增）：

```bash
# D0（192.0.0.5）—— 必须先起，它持有 DPCoordinator 和 DP 组的 zmq ROUTER
python launch_online_dp.py \
  --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 0 \
  --dp-address 192.0.0.5 --dp-rpc-port 12321 --vllm-start-port 7100

# D1（192.0.0.6）
python launch_online_dp.py \
  --dp-size 32 --tp-size 1 --dp-size-local 8 --dp-rank-start 8 \
  --dp-address 192.0.0.5 --dp-rpc-port 12321 --vllm-start-port 7100

# D2（192.0.0.7）：--dp-rank-start 16
# D3（192.0.0.8）：--dp-rank-start 24
```

每台起 8 个进程、端口 7100–7107、每进程独占 1 张卡，全集群 32 个 decoder 端点。

### 12.3 Proxy（最后启动）

16 个 prefiller 端点（4 台 × 4 DP rank）+ 32 个 decoder 端点（4 台 × 8 DP rank）：

```bash
python examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py \
  --host 0.0.0.0 --port 8000 \
  --prefiller-hosts 192.0.0.1 192.0.0.1 192.0.0.1 192.0.0.1 \
                    192.0.0.2 192.0.0.2 192.0.0.2 192.0.0.2 \
                    192.0.0.3 192.0.0.3 192.0.0.3 192.0.0.3 \
                    192.0.0.4 192.0.0.4 192.0.0.4 192.0.0.4 \
  --prefiller-ports 7100 7101 7102 7103  7100 7101 7102 7103 \
                    7100 7101 7102 7103  7100 7101 7102 7103 \
  --decoder-hosts   192.0.0.5 ×8  192.0.0.6 ×8 \
                    192.0.0.7 ×8  192.0.0.8 ×8 \
  --decoder-ports   7100 7101 7102 7103 7104 7105 7106 7107   （每台重复一遍）
```

> `×8` 是为了写得下的省略，实际要把主机名逐个列满 32 个。**注意 decoder 端口在每台机器上都是 7100–7107**——端口由 `--vllm-start-port` + 节点内序号决定，跟全局 dp_rank 无关；真正靠全局 dp_rank 区分的是上一节那组 handshake 端口。这两套端口容易混，是这类部署的高频错点。

### 12.4 关键参数一致性检查表

| 项 | 约束 | 出处 |
|---|---|---|
| `kv_connector_extra_config` | P/D 两侧逐字段完全相同 | `_get_prefill_decode_size` `:1587-1607` |
| `prefill.tp_size >= decode.tp_size` 且整除 | 2 ÷ 1 = 2 ✓ | `:1491-1495` + 设计文档 Limitations |
| `engine_id` | **5 个实例**互不相同；D 实例的 4 台机器共用同一个 | `MooncakeAgentMetadata.engine_id` |
| `kv_port` | 各实例相隔 ≥ `dp_size × tp_size`，且 ≥ 28000（8 卡节点）；D 侧 4 台填同一个值 | `:1556-1563` + 部署文档 |
| `--data-parallel-address/rpc-port` | 同一 DP 组内所有进程完全一致（D 侧 = 跨 4 台机器统一填 D0） | `utils.py:1168-1172` |
| `--dp-size` / `--dp-rank-start` | D 侧四台 `--dp-size` 恒为 32，`--dp-rank-start` 取 0/8/16/24 且不重叠 | `launch_online_dp.py` |
| MTP `num_speculative_tokens` | 非 Mamba 模型：P 侧必须 = 1，D 侧 ≥ 1 | 部署文档 note |
| `--block-size` | P/D 相同（这里 32） | KV block 布局需对齐 |
| `--max-model-len` | P/D 相同 | 同上 |
| 芯片代次 | P/D 不能混（不支持 A2 P + A3 D） | 设计文档 Limitations |
| 机器账 | 4 台 P（各 1 实例）+ 4 台 D（共 1 实例）= 8 台 64 卡 | 本文 0.2 节 |

---

## 13. 启动期故障定位速查

| 现象 | 大概率原因 | 定位/修复 |
|---|---|---|
| 卡在 "waiting for all ranks to join" | DP rendezvous 不通 | 检查 `--data-parallel-address` 可路由、`--data-parallel-rpc-port` 及其后续若干端口空闲（`get_next_dp_init_port` 会递增，`parallel.py:578-584`） |
| `zmq.error.ZMQError: Address already in use` | `kv_port` 落在 AscendDirect 随机端口区 | 8 卡节点 `kv_port ≥ 28000`；同时确认各实例间隔 ≥ `dp×tp` |
| `Timeout waiting for KV Cache thread to be ready` | Mooncake transfer engine 起不来 | 5 分钟硬超时（`:1750`）。检查 `LD_LIBRARY_PATH` 含 mooncake so、RDMA 网卡 `hccn_tool -i N -link -g` 为 UP |
| `prefill_tp_size must be >= decode_tp_size` | P/D TP 配反 | `:1491-1495` |
| `Upstream EPLB is only supported by Model Runner V2` | V1 runner 用了 `--enable-eplb` | 改用 `additional_config.eplb_config` + `DYNAMIC_EPLB` 环境变量（`platform.py:810-811`） |
| `Async EPLB is not supported by Model Runner V2 on Ascend yet` | V2 下设了 `eplb_config.use_async` | 置 false（`platform.py:792-795`） |
| DP 组内某 rank hang 在 forward | MoE 集合通信不同步 | 确认 `dp_rank==0` 的 DPCoordinator 进程存活（MoE + DP > 1 必需，`vllm/config/vllm.py:714-718`）。D 侧这个进程只在 D0 上 |
| D 侧四台机器各自组成 8 卡 world，集体卡在 "waiting for all ranks" | 四台的 `--dp-address` 填了各自本机 IP | 统一填 D0 的 IP；`--dp-size` 四台都必须是 32，不是 8 |
| D 侧 `dp_rank` 冲突或缺号 | `--dp-rank-start` 填重或跳号 | 必须是 0/8/16/24，与 `--dp-size-local 8` 严格衔接 |
| `Free memory ... is less than desired GPU memory utilization` | 卡上有残留进程 | `npu-smi info`，或调低 `--gpu-memory-utilization` |
| D 侧启动特别慢（数分钟无输出） | `FULL_DECODE_ONLY` 图捕获中 | 正常。可临时加 `--enforce-eager` 验证其他环节 |
| `local_world_size exceeds assigned_physical_gpu_ids` | `ASCEND_RT_VISIBLE_DEVICES` 与 `dp_size_local × tp_size` 不匹配 | `worker.py:382-391` |

---

## 14. 关键代码索引

### vLLM 上游

| 功能 | 文件:行 |
|---|---|
| serve CLI / LB 模式判定 | `vllm/entrypoints/cli/serve.py:79-152` |
| 插件加载 | `vllm/plugins/__init__.py:36-88` |
| 构建 engine client | `vllm/entrypoints/openai/api_server.py:133-178` |
| `create_engine_config` | `vllm/engine/arg_utils.py:1943-2539` |
| DP 参数推导（external LB） | `vllm/engine/arg_utils.py:2122-2238` |
| `ParallelConfig` 构造 | `vllm/engine/arg_utils.py:2244-2295` |
| `world_size` / `world_size_across_dp` | `vllm/config/parallel.py:549-551, 832-838` |
| `reconfigure_for_independent_dp_rank` | `vllm/config/parallel.py:1039-1048` |
| 平台钩子调用点 | `vllm/config/vllm.py:1360, 1578` |
| `needs_dp_coordinator` | `vllm/config/vllm.py:698-718` |
| `launch_core_engines` | `vllm/v1/engine/utils.py:1070-1200` |
| `EngineCore.__init__` | `vllm/v1/engine/core.py:107-250` |
| 连接器握手元数据汇聚 | `vllm/v1/engine/core.py:187-203` |
| `_initialize_kv_caches` | `vllm/v1/engine/core.py:253-358` |
| `DPEngineCoreProc._init_data_parallel` | `vllm/v1/engine/core.py:2020-2034` |
| `MultiprocExecutor._init_executor` | `vllm/v1/executor/multiproc_executor.py:118-205` |
| **DP rank 偏移 + world 扩展** | `vllm/distributed/parallel_state.py:1606-1636` |
| rank 网格与各通信域 | `vllm/distributed/parallel_state.py:1826-1994` |
| `_EP` / `_EPLB` 创建 | `vllm/distributed/parallel_state.py:1923-1978` |

### vllm-ascend

| 功能 | 文件:行 |
|---|---|
| entry points | `setup.py:515-523` |
| register 函数 + 全局 patch | `vllm_ascend/__init__.py:53-110` |
| connector 注册表 | `vllm_ascend/distributed/kv_transfer/__init__.py:21-88` |
| 模型注册 | `vllm_ascend/models/__init__.py:6,16,19` |
| `pre_register_and_update` | `vllm_ascend/platform.py:261-281` |
| `check_and_update_config`（10 步） | `vllm_ascend/platform.py:345-393` |
| attention 后端选择 | `vllm_ascend/platform.py:216-242` |
| 自定义 KV spec 注册 | `vllm_ascend/platform.py:157-161` |
| EPLB 校验 | `vllm_ascend/platform.py:765-811` |
| worker/scheduler 选择 | `vllm_ascend/platform.py:1125-1172` |
| `AscendConfig` 解析 | `vllm_ascend/ascend_config.py:30-541`（`init_ascend_config` 在 `:1253`） |
| **`init_ascend_model_parallel`（MC2 等）** | `vllm_ascend/distributed/parallel_state.py:21-134` |
| `NPUWorker.__init__` | `vllm_ascend/worker/worker.py:99-187` |
| `_init_device` / `init_device` | `vllm_ascend/worker/worker.py:352-475` |
| `_init_worker_distributed_environment` | `vllm_ascend/worker/worker.py:1052-1065` |
| `determine_available_memory` | `vllm_ascend/worker/worker.py:478-573` |
| `compile_or_warm_up_model` | `vllm_ascend/worker/worker.py:721-770` |
| `load_model` | `vllm_ascend/worker/model_runner_v1.py:3534-3612` |
| `initialize_kv_cache` | `vllm_ascend/worker/model_runner_v1.py:3662-3723` |
| `capture_model` | `vllm_ascend/worker/model_runner_v1.py:4891-4901` |
| `DeepseekV4MoE.__init__`（EP 切分） | `vllm_ascend/models/deepseek_v4/model.py:251-330` |
| `select_moe_comm_method` | `vllm_ascend/ascend_forward_context.py:295-370` |
| MoE 通信实现 | `vllm_ascend/ops/fused_moe/moe_comm_method.py:51-290` |
| DSA attention 后端 | `vllm_ascend/attention/dsa_v1.py:82-1513` |
| `enable_dsa_cp` 判定 | `vllm_ascend/utils.py:1325-1362` |
| `pd_tp_ratio` / `pd_head_ratio` 推导 | `vllm_ascend/ascend_config.py:200-227` |
| MC2 dispatch/combine 算子 | `vllm_ascend/ops/fused_moe/token_dispatcher.py:247-352` |
| 昇腾 KV cache spec | `vllm_ascend/core/kv_cache_interface.py:19-224` |
| **MooncakeHybridConnector** | `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_hybrid_connector.py` |
| ├ 连接器双身份 | `:1082-1097` |
| ├ scheduler 侧方法 | `:1335-1483` |
| ├ worker 侧 init / 端口公式 | `:1488-1585` |
| └ `register_kv_caches` / 建链 | `:1609-1752` |
| PD 设计文档 | `docs/source/developer_guide/Design_Documents/disaggregated_prefill.md` |
| 多机 PD 部署指南 | `docs/source/tutorials/features/pd_disaggregation_mooncake_multi_node.md` |
| 外部 DP 启动器 | `examples/external_online_dp/launch_online_dp.py` |
| PD 代理示例 | `examples/disaggregated_prefill_v1/load_balance_proxy_server_example.py` |
| DSV4 PD 参考配置 | `tests/e2e/nightly/multi_node/external_dp/config/DeepSeek-V4-flash-w8a8-PD-prefix.yaml` |

---

## 15. 六个值得单独记住的结论

0. **"4P1D" 数的是实例，不是机器。** 本文这套是 4 个 Prefill 实例（每个 DP4×TP2，各占一台 8 卡机）+ 1 个 Decode 实例（DP32×TP1，横跨四台 8 卡机），物理上 8 台 64 卡。一个实例可以小于一台机器，也可以大于一台机器——决定边界的是"哪些进程共享同一个 `--data-parallel-address`"，而不是机箱。

1. **昇腾适配零侵入 vLLM 源码**，全靠两组 entry point + 运行时 monkey patch。理解 vllm-ascend 的第一步是理解 `check_and_update_config` 这个"配置改写总闸"，它决定了 worker 类、scheduler 类、custom op、图模式、all2all 后端。

2. **EP 跨 DP 的通信域是靠 `init_distributed_environment` 里那七行 rank 改写建起来的**（`parallel_state.py:1606-1636`）：`rank = dp_rank × world_size + rank`，`world_size = world_size_across_dp`，rendezvous 点是 `data_parallel_master_ip`。P 侧 4 个互相独立的 EngineCore 进程因此共享一个 8 rank 的 HCCL world；D 侧则是 **32 个跨 4 台机器的 EngineCore 共享一个 32 rank 的 world**。这是理解 EP+DP 部署的分水岭，也是"1D 可以由多台机器组成"这件事在代码里的落点。

3. **MoE + DP 强制需要 DPCoordinator**，即使在 external LB 下也是。EP 的 all-to-all 是集合通信，DP rank 必须同步进出 forward，否则死锁。P 侧四个独立 DP 组各有一个 coordinator，D 侧整个实例只有一个（在 D0 上）。

4. **P 和 D 跑的是同一份权重，但几乎所有运行时路径都不同**：MoE 通信方式（ALLTOALL vs MC2）、图模式（eager vs FULL_DECODE_ONLY）、scheduler（默认 vs recompute）、KV 连接器角色（发送线程 vs 接收线程）、显存水位、MTP 深度（1 vs 3）。PD 分离的收益就来自这些差异化。

5. **DeepSeek-V4-Flash 的 MLA + 压缩 KV 结构对 PD 特别友好**：`is_deepseek_mla=True` 让 `tp_num_need_pulls = 1`，非对称 TP（P tp=2 → D tp=1）不产生任何额外的 KV 重组开销；`compress_ratios` 把 20 层的 KV 压到 1/128，1M 上下文下要传输的 KV 量比 V3.2 小一个量级，直接决定了 PD 之间 RDMA 带宽是否成为瓶颈。

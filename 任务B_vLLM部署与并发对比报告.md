# 任务 B 实验报告：vLLM 部署与并发吞吐对比

**实验日期**：2026-11-12 ｜ **项目**：Local LLM Engineering · 任务 B

---

## 一、实验目标

在 RTX 5060 Ti 16G 上完成 vLLM 生产级部署（OpenAI 兼容 API），实测 1/4/8 路并发吞吐，并与 llama.cpp 同量化档（INT4）做公平对比。

## 二、环境与配置

| 项目 | 配置 |
|---|---|
| GPU | RTX 5060 Ti 16GB（Blackwell sm_120） |
| 引擎 A | vLLM 0.31.0 + Qwen2.5-7B-Instruct-**AWQ**（INT4，5.2GB） |
| 引擎 B | llama.cpp 1648d64 + Qwen2.5-7B-**Q4_K_M**（INT4，4.4GB） |
| 测试协议 | OpenAI 兼容 `/v1/chat/completions`，流式 SSE；16 请求/档 × {1,4,8} 路并发，max_tokens=128 |
| 上下文 | 两端均 max-model-len / ctx = 4096 |

## 三、实验结果

### 3.1 并发吞吐（tokens/s）

| 并发 | vLLM + AWQ | llama.cpp + Q4_K_M | vLLM 优势 |
|---|---|---|---|
| 1 路 | 62.6 | 64.9 | -3.5% |
| 4 路 | 202.2 | 183.6 | +10% |
| 8 路 | **335.0** | 191.4 | **+75%** |

### 3.2 延迟与首 token

| 并发 | vLLM 平均延迟 | llama 平均延迟 | vLLM TTFT | llama TTFT |
|---|---|---|---|---|
| 1 路 | 1.88s | 1.69s | 26ms | 17ms |
| 4 路 | 2.36s | 2.35s | 460ms | 63ms |
| 8 路 | **2.86s** | 3.93s | 879ms | 15ms |

### 3.3 显存占用

- vLLM + AWQ：峰值 **14.9 GB**（默认 90% 显存利用率，含大 KV 池）
- llama.cpp + Q4_K_M：峰值 16.0 GB（-c 4096 多 slot KV）
- 两者同档可共存对比：AWQ 模型权重 5.2GB vs GGUF 4.4GB

### 3.4 加分项：vLLM 跑 FP16 的结果

**vLLM 拒绝启动 FP16**：
```
ValueError: Free memory on device cuda:0 (14.8/15.9 GiB) on startup is less
than desired GPU memory utilization (0.98, 15.58 GiB)
```
对比任务 A 的发现——llama.cpp 靠 VMM 显存虚拟化能"勉强"跑 FP16（31 t/s 抖动）——vLLM 对显存管理更严格（生产级），15.2GB 权重 + 激活/KV 开销超出 16G，直接拒绝服务。**结论：16G 卡上 vLLM 部署 7B 必须量化，AWQ INT4 是官方推荐路径。**

## 四、关键结论

1. **单路性能持平**：1 路并发下两者 62.6 vs 64.9 t/s，llama.cpp 略快——单用户场景两个引擎没有本质差异。
2. **高并发 vLLM 碾压**：8 路并发 vLLM 335 t/s vs llama.cpp 191 t/s（+75%）。核心原因：vLLM 的 **continuous batching**（动态批处理，请求到达即入批）vs llama.cpp 的**固定 slot 并行**（4 slots 满后排队）。llama.cpp 从 4 路（183.6）到 8 路（191.4）几乎不涨——GPU 已饱和；vLLM 4→8 路仍在涨（202→335）。
3. **延迟权衡**：vLLM 用 TTFT 换吞吐（排队请求首 token 等待 879ms @8 路），llama.cpp TTFT 始终极低但吞吐封顶。生产服务多用户场景选 vLLM；本地单用户/边缘设备选 llama.cpp。
4. **PagedAttention 的意义**：vLLM 的 KV cache 按页管理（143K tokens 池 = 35 个 4096 并发槽），碎片率趋零；llama.cpp 多 slot 预分配固定 KV。

**工程结论**：
> 在 RTX 5060 Ti 16G 上完成 vLLM 0.31 生产级部署：解决 FlashInfer JIT 编译链问题（CUDA_HOME/ninja/cccl 兼容），实测 1/4/8 路并发吞吐 62.6/202/335 t/s，对比 llama.cpp 同量化档高并发场景吞吐提升 75%，并验证 FP16 在 16G 显存不可部署（AWQ INT4 为最优路径）。

**扩展讨论**（原理 Q&A）：
- continuous batching 原理？→ 请求粒度调度 vs slot 粒度；迭代级 batching，不等整批完成
- PagedAttention？→ KV cache 按 block 分页，非连续分配，消除内存碎片，支持任意长度动态扩展
- AWQ vs GPTQ？→ AWQ 基于激活感知的权重量化（保护重要通道），无需重新校准数据集推理更快
- TTFT 与吞吐的权衡？→ vLLM 默认先来先服务，队列深时 TTFT 上升；可用优先级调度/抢占优化
- 为什么 vLLM 拒绝 FP16 而 llama.cpp 能跑？→ vLLM 显存预算严格（模型+KV+激活+图缓存），llama.cpp 走 VMM 换页（性能代价大）

## 五、踩坑记录（环境复现要点）

1. **FlashInfer JIT 编译链**（vLLM 启动必经）：需要 nvcc + ninja + gcc。解法：`CUDA_HOME=/home/<user>/cuda_toolkit`（pip cu13 组装）+ PATH 加 task_a 的 ninja
2. **cccl 版本检查**（与任务 A 同款坑）：flashinfer 自带 cccl 头检查 CUDART(13.0) vs nvcc(13.5) 不匹配 → 在 `flashinfer/data/cccl/.../cuda_toolkit.h` 定义 `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK 1`
3. **-lcuda 找不到**：WSL 驱动库在 /usr/lib/wsl/lib，需软链 libcuda.so 进工具链 lib/lib64
4. 启动脚本放文件执行（内联 $PATH/$VAR 会被 Git Bash/MSYS 吞掉或转换）

## 六、数据归档

- 原始数据：`~/llm_plan/results/conc/{vllm_awq,llama_q4km}.json`
- 图表：`~/llm_plan/results/figures/conc_{throughput,latency,ttft}.png`
- 服务日志：`~/llm_plan/vllm_awq.log`、`~/llm_plan/vllm_fp16.log`、`~/llm_plan/llama_srv.log`
- 启动脚本：`D:\AI_Workspace\start_vllm{,_fp16}.sh`、`patch_flashinfer.sh`

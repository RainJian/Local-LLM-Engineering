# Local LLM Engineering｜Qwen2.5-7B 量化 · vLLM 服务化 · QLoRA 微调

在 **RTX 5060 Ti 16G（Blackwell / sm_120）+ WSL2 (Ubuntu 22.04)** 上，围绕 **Qwen2.5-7B-Instruct** 完成的三个工程任务：量化对比 → vLLM 高并发部署 → QLoRA 微调闭环。所有数字均为实测，可复现、可追问。

## 三个任务

| 任务 | 内容 | 核心结论 |
|---|---|---|
| **量化对比** | FP16 / INT8(Q8_0) / INT4(Q4_K_M, Q4_0) 四档系统对比（llama.cpp 自编译 CUDA 内核） | Q4_K_M 是甜点位：4.6GB 显存、77.5 t/s（FP16 的 2.5 倍）、20 题质量零损失；FP16 依赖 Blackwell VMM 勉强运行（31 t/s 抖动） |
| **vLLM 部署** | vLLM 0.31 + AWQ INT4，OpenAI 兼容 API，1/4/8 路并发压测，与 llama.cpp 同档公平对比 | 1 路持平（~63 t/s），8 路 vLLM 335 t/s 较 llama.cpp +75%（continuous batching vs 固定 slot）；FP16 被 vLLM 显存预算直接拒绝 |
| **QLoRA 微调** | alpaca-zh 10k 条、NF4 4bit 基座 + r=16 LoRA（仅训 0.53% 参数）、2 epochs | loss 2.161→1.245；20 题评测 20/20 + 6 处结构化改进案例；双引擎部署闭环（GGUF 合并 + vLLM LoRA 插件热插拔） |

## 仓库结构

```
├── 项目计划书.docx                  # 原始计划文档
├── 任务A_量化对比实验报告.md          # + 任务A_图表/（4 张）
├── 任务B_vLLM部署与并发对比报告.md    # + 任务B_图表/（3 张）
├── 任务C_QLoRA微调实验报告.md         # + 任务C_图表/（loss 曲线 + 评估数据）
└── README.md
```

## 环境复现要点（速查）

- 推理/训练引擎：llama.cpp（自编译，sm_120a 内核）、vLLM 0.31、transformers + peft + bitsandbytes
- 踩坑速记：CUDA 13.5 pip 工具链需手动组装（nvvm/cicc）；cccl 版本检查用 `CCCL_DISABLE_CTK_COMPATIBILITY_CHECK` 绕过；vLLM FlashInfer JIT 需要 `CUDA_HOME` + ninja + libcuda 软链；合并版 llama-cli 单轮生成用 `-st` + `-o`；WSL dxg 桥被 SIGKILL 污染后重启 WSL 恢复
- 完整过程与数据详见各任务报告

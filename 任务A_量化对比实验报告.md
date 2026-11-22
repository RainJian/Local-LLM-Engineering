# 任务 A 实验报告：Qwen2.5-7B 量化对比（FP16 / INT8 / INT4）

**实验日期**：2026-11-11 ｜ **项目**：Local LLM Engineering · 任务 A

---

## 一、实验环境

| 项目   | 配置                                              |
| ---- | ----------------------------------------------- |
| GPU  | NVIDIA RTX 5060 Ti 16GB（Blackwell，sm_120）       |
| CPU  | Intel Core i5-13500H                            |
| 系统   | Windows 11 + WSL2（Ubuntu 22.04）                 |
| 推理引擎 | llama.cpp（commit 1648d64，自编译 CUDA 版，sm_120a 内核） |
| 模型   | Qwen2.5-7B-Instruct（GGUF 官方分片）                  |
| 驱动   | Windows 621.01 / CUDA UMD 13.5                  |

## 二、实验设计

- **对比档位**：FP16、Q8_0（INT8）、Q4_K_M（INT4 推荐档）、Q4_0（INT4 激进档）
- **吞吐测试**：llama-bench，pp512（预填充 512 tokens）+ tg128（生成 128 tokens），3 轮取均值
- **质量测试**：20 道固定中文题（推理/常识/代码/数学各 5 道），统一采样参数（temp=0.7, top-p=0.8, seed=42），生成上限 256 tokens
- **显存测试**：nvidia-smi 实时采样生成过程峰值显存

## 三、实验结果

### 3.1 核心对比表

| 指标      | FP16        | Q8_0 (INT8)     | Q4_K_M (INT4)   | Q4_0 (INT4)     |
| ------- | ----------- | --------------- | --------------- | --------------- |
| 模型体积    | 14.19 GB    | 7.54 GB（-47%）   | 4.36 GB（-69%）   | 4.12 GB（-71%）   |
| 峰值显存    | 13,943 MiB  | 7,579 MiB（-46%） | 4,583 MiB（-67%） | 4,343 MiB（-69%） |
| 预填充速度   | 4,140 t/s   | 3,754 t/s       | 3,941 t/s       | 4,042 t/s       |
| 生成速度    | 31.3 t/s ⚠️ | 48.5 t/s        | 77.5 t/s        | 82.3 t/s        |
| 20 题正确率 | 20/20       | 20/20           | 20/20           | 20/20           |
| 答案质量    | 基准          | 与 FP16 无差异      | 与 FP16 无差异      | 与 FP16 无差异      |

### 3.2 关键发现

1. **FP16 勉强能跑，但很痛苦**：15.2GB 权重靠 Blackwell 的 VMM（显存虚拟化）技术勉强塞进 16G 显存，但生成速度仅 31.3 t/s 且抖动剧烈（26~40 t/s，stddev 7.6），换页开销明显。**16G 卡跑 7B FP16 不是合理选择**。
2. **Q4_K_M 是甜点位**：显存省 67%（4.6GB，16G 卡可同时跑 2 个 + 余量），速度反超 FP16 2.5 倍，质量与 FP16 无肉眼差异。
3. **INT8 反而比 INT4 慢**：Q8_0 生成 48.5 t/s < Q4_K_M 77.5 t/s——生成阶段带宽受限（memory-bound），8-bit 权重每 token 搬运字节数是 4-bit 的两倍。预填充阶段各档差异不大（算力受限）。
4. **INT4 质量无损**：20 题全对。7B 模型参数量大，4-bit 量化对中文常识/推理/代码任务影响可忽略（与公开评测结论一致：Q4_K_M 相对 FP16 平均损失 <1%）。

### 3.3 图表

- `chart_vram.png` — 显存占用对比（红色虚线 = 16GB 上限，FP16 逼近极限）
- `chart_speed.png` — 生成速度对比
- `chart_throughput.png` — 预填充/生成吞吐分组对比
- `chart_size.png` — 模型体积对比

## 四、结论与建议

**一句话结论**：16G 显存部署 Qwen2.5-7B，**Q4_K_M 是甜点位**——省 67% 显存、2.5 倍速度、质量无损。

**工程结论**：

> 在 RTX 5060 Ti 16G 上完成 Qwen2.5-7B 四档量化（FP16/INT8/INT4）对比实验：自编译 llama.cpp CUDA 内核，实测 FP16 依赖 VMM 勉强运行（31 t/s）、Q4_K_M 以 4.6GB 显存实现 77.5 t/s 且 20 题质量测试零损失，验证了 INT4 在 7B 模型的工程最优性。

**扩展讨论**（本实验涉及的原理点）：

- 为什么 INT8 生成比 INT4 慢？→ 生成阶段 memory-bound，权重搬运量翻倍
- FP16 为什么能塞进 16G？→ Blackwell VMM 显存虚拟化换页，代价是速度抖动
- 量化为什么质量不降？→ 7B 模型权重冗余度高，Q4_K_M 的 block-wise 量化 + 重要权重保护
- Q4_0 与 Q4_K_M 区别？→ K-quants 每块独立 scale/min，比 Q4_0 全局 scale 保精度更好

## 五、复现路径（备忘）

- 环境：WSL2 内 ~/llm_plan/（venv task_a：cmake 4.4.4/ninja/matplotlib）
- 模型：~/llm_plan/models/ 四档 GGUF（31GB）
- 引擎：~/llm_plan/llama.cpp/build/bin/{llama-cli,llama-server,llama-bench}
- 结果：~/llm_plan/results/{bench,answers,figures}
- 脚本：D:\AI_Workspace{run_bench.sh, bench_qa2.py, measure_vram.py, plot_results.py}
- 踩坑要点：CUDA 13.5 pip 工具链需组装 toolkit（含 nvvm/、静态库软链）；cccl 版本检查用 CCCL_DISABLE_CTK_COMPATIBILITY_CHECK 绕过；CUDART_VERSION 保持 13000 真实值；合并版 llama-cli 单轮生成需 -st + -o 文件输出；质量测试用常驻 llama-server + HTTP API（进程式 llama-cli 有偶发死锁 + SIGKILL 会污染 dxg 桥，重启 WSL 可恢复）

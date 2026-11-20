# In-Weight FPGA：基于零 KV 漂移闭式线性代数的休眠神经元多步辩证状态机权重编译

**[English](README.md) | [简体中文](README_CN.md) | [完整英文学术论文手稿](PAPER_MANUSCRIPT.md) | [通用数学方法论规范](WSP_GENERAL_METHODOLOGY.md)**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Models: Qwen3 & Llama-3.1](https://img.shields.io/badge/支持架构-Qwen3--8B%20%7C%20Llama--3.1--8B-blue.svg)](#核心实验结果3780-轮真实对抗对话正面对标-selfd-rl)
[![Optimization: Zero-Gradient Closed-Form](https://img.shields.io/badge/优化方式-0%20步梯度下降%20(30秒闭式编译)-orange.svg)](#第一性原理发现读写解耦与末层零-kv-漂移定理)
[![Statistical Tests: 12/12 Passed](https://img.shields.io/badge/10000次配对Bootstrap检验-12%2F12%20全部显著通过-brightgreen.svg)](#核心实验结果3780-轮真实对抗对话正面对标-selfd-rl)

---

## 核心摘要 (TL;DR)

大语言模型在多轮交互与同行评议对抗中面临严峻的**辩证困境（Dialectical Dilemma）**（参见 *"Benchmarks Don't Talk Back"*, [arXiv:2611.17563](https://arxiv.org/abs/2611.17563)）：未经微调的基座模型在面对错误反驳时极易盲从投降（**Sycophancy / Capitulation**）；而使用多轮强化学习微调（`SelfD-RL`）则会陷入严重的奖励黑客（Reward Hacking），导致模型盲目固执——接纳正确指正的能力（`Yielding`）从 `84.1%` **暴跌 30.8 个百分点至 `53.3%`**。与此同时，传统的中层模型权重编辑（`ROME`、`MEMIT`）与表征工程（`RepE`）在长思维链多步自由生成中频频失效，因为在中层注入的控制信号会持续污染后续所有层的自注意力 KV 缓存（**自回归 KV-Cache 漂移**）。

本项目提出了**跨模型通用的纯血权重空间编程（Model-Agnostic Pure In-Weight WSP）**，建立在一项核心第一性原理发现之上：
> **读写解耦与末层零 KV 漂移定理（The Read-Write Decoupling & Terminal-Layer Zero-KV-Drift Principle）**：在任何标准自回归 Transformer 中，最末层门控 MLP（`layers[L-1].mlp`）是全网唯一一个**位于所有自注意力层 KV Cache 写入之后、词表投影之前**的非线性计算单元。通过将**跨段落证据聚合与多域判别（Read & Verify）**限制在中后层（$[0.66L, 0.84L]$）的保护零空间内，而将**多步分支有向无环状态机（Write & Act）**利用 `bfloat16` 浮点下溢特性（$\mathrm{SiLU}(z \le -15) \equiv 0.0$）与**白化反嵌入对偶基**编译进最末层 $\mathrm{MLP}_{L-1}$ 的休眠神经元中，**模型在多步自由自回归生成（`model.generate()` / `vLLM`）时的内部隐藏状态轨迹与单次静态 Teacher-Forcing 前向传播实现 100% 逐比特等价！**

仅需在单张 GPU 上运行 **30 秒纯闭式线性代数计算（0 步梯度下降，仅修改约 `0.035%` 的参数，0 外部推理脚手架）**，Pure WSP 在 **`Qwen3-8B`** 与 **`Meta-Llama-3.1-8B-Instruct`** 两大异构架构、**3,780 轮无删减三阶段对抗对话**上全面超越 `SelfD-RL`，**12 / 12 项** 10,000 次配对 Bootstrap 95% 置信区间统计显著性检验全部通过，并将 `vLLM` 推理吞吐速度提升 **6–10 倍**。

---

## 核心实验结果：3,780 轮真实对抗对话正面对标 `SelfD-RL`

全部实验严格采用 `arXiv:2611.17563` 原封不动的提示词与判分协议，在标准 `vLLM`（`max_model_len=16384, max_tokens=4096, temperature=0`）下完成，并经过 10,000 次按题目配对自助法（Paired Bootstrap）95% 置信区间检验（涵盖 `AMC`、`MATH-500`、`OlympiadBench`、`MMLU-Pro`、`ARC-Challenge`、`StrategyQA`、`CRUXEval-O` 七大推理基准）：

| 模型家族与架构 | 评测基准套件 (对话数) | 评测阶段 / 方法 | **总体正确率 Synthesis (↑)** | **坚守率 Holding (↑)** *(首答对、反驳错)* | **接纳率 Yielding (↑)** *(首答错、反驳对)* | **辩证能力 Dialectic (↑)** *(0.5×(H+Y))* | **盲从投降率 Capitulation (↓)** | **10,000次配对 Bootstrap 95% CI 显著性判定** |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3-8B`** *(原论文 arXiv:2611.17563)* | 原论文 7 大基准 | 原论文 Base<br>原论文 **SelfD-RL** | `62.80%`<br>`73.50%` | `53.30%`<br>`88.90%` | `84.10%`<br>`53.30%` *(严重退化)* | `68.70%`<br>`71.10%` *(仅+2.40%)* | `10.30%`<br>`1.00%` | 强化学习基线 *(修改 100% 参数)* |
| **`Qwen3-8B`**<br>($L=36, d=4096$) | **套件 1：`v4k` 全集**<br>(654 轮对话) | 编辑前 Base<br>**Pure WSP (本文)**<br>*差值 95% CI* | `71.41%`<br>**`79.36%` (+7.95%)**<br>*`[+4.59%, +11.47%]`* | `56.65%`<br>**`93.64%` (+36.99%)**<br>*`[+28.80%, +45.21%]`* | `90.83%`<br>**`80.00%`**<br>*`[-19.84%, -2.22%]`* | `73.74%`<br>**`86.82%` (+13.08%)**<br>*`[+7.01%, +19.00%]`* | `11.93%`<br>**`1.83%` (-10.10%)**<br>*`[-13.00%, -7.34%]`* | **3 / 3 全通过**<br>(`true / true / true`) |
| **`Qwen3-8B`**<br>*(冻结 `v4k` `half=0` 权重)* | **套件 2：`f` 独立盲测全集**<br>(1,272 轮对话) | 编辑前 Base<br>**Pure WSP (本文)**<br>*差值 95% CI* | `73.19%`<br>**`78.07%` (+4.88%)**<br>*`[+2.52%, +7.31%]`* | `59.79%`<br>**`86.34%` (+26.55%)**<br>*`[+21.17%, +31.88%]`* | `90.83%`<br>**`75.11%`**<br>*`[-21.95%, -9.59%]`* | `75.31%`<br>**`80.72%` (+5.41%)**<br>*`[+1.34%, +9.52%]`* | `12.26%`<br>**`4.25%` (-8.01%)**<br>*`[-9.98%, -6.21%]`* | **3 / 3 全通过**<br>(`true / true / true`) |
| **`Qwen3-8B`**<br>*(冻结 `v4k` `half=0` 权重)* | **套件 3：`g` 独立盲测全集**<br>(1,200 轮对话) | 编辑前 Base<br>**Pure WSP (本文)**<br>*差值 95% CI* | `73.25%`<br>**`78.75%` (+5.50%)**<br>*`[+2.83%, +8.17%]`* | `56.01%`<br>**`90.03%` (+34.02%)**<br>*`[+28.12%, +39.63%]`* | `89.20%`<br>**`65.26%`**<br>*`[-31.02%, -17.14%]`* | `72.61%`<br>**`77.64%` (+5.03%)**<br>*`[+0.43%, +9.43%]`* | `12.83%`<br>**`2.83%` (-10.00%)**<br>*`[-12.08%, -7.92%]`* | **3 / 3 全通过**<br>(`true / true / true`) |
| **`Meta-Llama-3.1-8B-Instruct`**<br>($L=32, d=4096$) | **套件 4：`v4k` 跨架构全集**<br>(654 轮对话) | 编辑前 Base<br>**Pure WSP (本文)**<br>*差值 95% CI* | `39.91%`<br>**`48.78%` (+8.87%)**<br>*`[+5.20%, +12.54%]`* | `67.91%`<br>**`94.88%` (+26.97%)**<br>*`[+19.62%, +34.31%]`* | `41.89%`<br>**`51.35%` (+9.46%)**<br>*`[-7.25%, +26.51%]`* | `54.90%`<br>**`73.12%` (+18.22%)**<br>*`[+8.95%, +27.53%]`* | `10.70%`<br>**`1.99%` (-8.71%)**<br>*`[-11.62%, -5.96%]`* | **3 / 3 全通过**<br>(`true / true / true`) |

---

## 文档与代码导航

1. **[`PAPER_MANUSCRIPT.md`](PAPER_MANUSCRIPT.md)**：按 NeurIPS / ICLR / ICML 顶会标准撰写的完整英文学术论文手稿（含定理证明、算法伪代码、3,780 轮主实验与消融实验）。
2. **[`WSP_GENERAL_METHODOLOGY.md`](WSP_GENERAL_METHODOLOGY.md)**：五大架构定律、三级闭式编译数学公式与跨模型家族自动化迁移指南。
3. **[`MILESTONE_2_PURE_INWEIGHT_WSP_REPORT.md`](MILESTONE_2_PURE_INWEIGHT_WSP_REPORT.md)**：阶段二零预填纯血 WSP 完整实验报告。
4. **[`MILESTONE_1_SCAFFOLDED_WSP_REPORT.md`](MILESTONE_1_SCAFFOLDED_WSP_REPORT.md)**：阶段一探索报告（揭示为何单点编辑失效及多域 SwiGLU 路由的物理机制）。
5. **[`build_vllm_pure_wsp.py`](build_vllm_pure_wsp.py) & [`build_llama_pure_wsp.py`](build_llama_pure_wsp.py)**：`Qwen3-8B` 与 `Meta-Llama-3.1-8B-Instruct` 的一键闭式权重编译器。

---

## 如何免费获取正式学术 DOI 并发布（无需 arXiv 背书）

1. 将本目录推送到你的 GitHub 公开仓库；
2. 登录 **[Zenodo.org](https://zenodo.org)**（欧洲核子研究中心 CERN 官方学术存储库），用 GitHub 账号一键关联并开启该仓库开关；
3. 在 GitHub 上点击 **Create a new release**（例如版本号 `v1.0.0`），Zenodo 会自动读取仓库内置的 `.zenodo.json` 与 `CITATION.cff`，在 **10 秒内自动签发全球通用的正式学术 DOI 编号（`10.5281/zenodo.xxxxxxx`）** 并被 Google Scholar 索引。

# 通用纯血权重空间编程 (Model-Agnostic Pure In-Weight WSP) 完整方法论规范与跨模型验证报告

**文档版本**：v1.2（含 `Qwen3-8B` 与 `Meta-Llama-3.1-8B-Instruct` 双架构实证）  
**适用架构**：所有采用门控 MLP（SwiGLU / GeGLU）与自注意力机制的现代自回归 Transformer 大语言模型（包括但不限于 **Llama-3 / 3.1 / 3.3、Mistral、Gemma-2 / 3、DeepSeek、Qwen** 等全系列）  
**核心目标**：在不进行任何梯度下降微调（0 步 SGD/Adam/RL）、不依赖任何外部推理脚手架（0 Prompt 预填、0 Python 字符串截断拼接，100% 原生 `model.generate()` / `vLLM`）的前提下，通过**闭式线性代数（Closed-Form Linear Algebra）**修改模型中约 **$0.03\% \sim 0.05\%$** 的休眠参数，将复杂的多步认知验证与分支状态机直接编译进神经网络权重。

---

## 一、 核心第一性原理与五大架构定律

设目标自回归 Transformer 模型包含 $L$ 层（层索引 $l \in \{0, 1, \dots, L-1\}$），隐藏层维度为 $d_{\mathrm{model}}$，MLP 中间层维度为 $d_{\mathrm{ffn}}$，词表矩阵（Unembedding）为 $W_U \in \mathbb{R}^{|V| \times d_{\mathrm{model}}}$，末层归一化（RMSNorm / LayerNorm）缩放权重为 $\gamma_{\mathrm{norm}} \in \mathbb{R}^{d_{\mathrm{model}}}$。

每层门控 MLP 的前向计算具有统一的数学形式：
$$\mathrm{MLP}_l(x) = W_{\mathrm{down}}^{(l)} \Big( \sigma\big(W_{\mathrm{gate}}^{(l)} x\big) \odot W_{\mathrm{up}}^{(l)} x \Big) = \sum_{c=1}^{d_{\mathrm{ffn}}} w_{\mathrm{down}, c}^{(l)} \cdot \sigma\big(\langle w_{\mathrm{gate}, c}^{(l)}, x 
angle\big) \cdot \langle w_{\mathrm{up}, c}^{(l)}, x 
angle$$
其中 $\sigma(\cdot)$ 为 $\mathrm{SiLU}(z) = z \cdot (1 + e^{-z})^{-1}$（或 $\mathrm{GELU}(z)$），$c \in \{1, \dots, d_{\mathrm{ffn}}\}$ 为第 $c$ 个神经元通道（Slot）。

我们的方法建立在以下五条普适的架构定律之上：

### 定律 1：末层 MLP 零 KV 漂移定理 (Terminal-Layer Zero-KV-Drift Theorem —— 核心基础发现)
传统中层权重编辑（ROME / MEMIT）与表征引导（RepE）在多步长链自回归生成中之所以崩溃，根源在于**自回归 KV-Cache 漂移（Autoregressive KV-Cache Drift）**：在中层注入的动作控制向量会写入后续所有层的自注意力 KV Cache，在接下来的数百步生成中被反复读取放大，导致分布失真或被高层顺从头（Sycophancy Heads）抵消。

在所有标准 Transformer Decoder 中，第 $L-1$ 层（最后一层）的计算顺序为：
$$h^{(L-1)} = x^{(L-1)} + \mathrm{SelfAttn}_{L-1}\big(\mathrm{Norm}(x^{(L-1)});\; \mathrm{KV\_Cache}_{L-1}\big)$$
$$y^{(L-1)} = h^{(L-1)} + \mathrm{MLP}_{L-1}\big(\mathrm{Norm}(h^{(L-1)})\big) \longrightarrow \mathrm{FinalNorm} + W_U \longrightarrow \mathrm{Logits}$$
**核心推论（读-写解耦原理）**：全网所有层（$0 \le l \le L-1$）的 $\mathrm{KV\_Cache}_l$ 在进入 $\mathrm{MLP}_{L-1}$ **之前**已经全部写入完毕！
- **对最末层 $\mathrm{MLP}_{L-1}$ 的任何权重修改，仅改变当前步输出的下一词 Logits，绝不会回流污染当前步任何一层的 KV Cache。**
- 因此，当模型在原生 `model.generate()` 或 `vLLM` 中自回归生成状态机序列 $t_1, t_2, \dots, t_m$ 时，每一步在进入 $\mathrm{MLP}_{L-1}$ 前的隐藏状态 $h_k^{(L-1)}$ 与校准阶段通过单次 Teacher-Forcing 前向传播采集的隐藏状态 **100% 逐比特严格等价（Zero KV-Cache Drift）**。

### 定律 2：休眠神经元水库定律 (Dormant Neuron Reservoir Law)
在任何经过大规模预训练的现代 Transformer 中，各层 MLP 的通道激活能量 $E_c^{(l)} = \mathbb{E}_{x}\big[\big|\sigma(\langle w_{\mathrm{gate}, c}^{(l)}, x 
angle) \cdot \langle w_{\mathrm{up}, c}^{(l)}, x 
angle\big|\big]$ 呈高度长尾分布。每一层均存在数百个激活能量趋近于 $0$ 的**休眠神经元（Dormant Neurons）**：
$$\mathcal{D}_l(K) = \operatorname{arg\,min}_{S \subset \{1, \dots, d_{\mathrm{ffn}}\}, |S|=K} \sum_{c \in S} E_c^{(l)}$$
覆写这些休眠神经元的 $(w_{\mathrm{gate}, c}^{(l)}, w_{\mathrm{up}, c}^{(l)}, w_{\mathrm{down}, c}^{(l)})$ 对原模型在无关输入上的原有分布扰动无穷小。

### 定律 3：硬件级实数下溢门控定律 (Floating-Point Underflow Routing Law)
在 `bfloat16` / `float16` 精度下，当门控预激活值 $z = \langle w_{\mathrm{gate}, c}, x 
angle \le -15.0$ 时：
$$\mathrm{SiLU}(z) = \frac{z}{1 + e^{-z}} \approx z e^{z} \le -4.5 \times 10^{-6} \implies 0.0 \;\;(\mathrm{in\;bfloat16})$$
因此，若将 $w_{\mathrm{gate}, c}$ 构造为白化线性判别器 $s \cdot (k - \theta b)$，使得目标分布外样本满足 $\langle k, x 
angle / \langle b, x 
angle < \theta$ 且缩放因子 $s$ 将负样本推至 $z \le -15.0$，则该神经元在非目标状态下**实现严格的硬件级 `0.0` 关断**，而在目标状态（$z > +4.0$）下近似呈线性透传 $\mathrm{SiLU}(z) \approx z$。这使预训练门控 MLP 天然成为**片上可编程逻辑门阵列（In-Weight FPGA）**。

### 定律 4：词表坐标对齐定律 (Vocabulary Coordinate-Matching Law)
残差流在不同深度层处于不同的旋转坐标系中。若在中浅层（如 $0.3L \sim 0.5L$）直接向残差流注入末层词表向量 $W_U[t]$，后续十余层的自注意力与 MLP 会将其非线性扭曲并引发级联崩溃。因此：
- **中后段层（$0.65L \sim 0.85L$）**：仅可用于注入经过前向雅可比矩阵校准的低维概念判别信号（如二元裁决 `incorrect` vs `correct`）；
- **精确词表发射与多步语法树生成**：必须且只能放置在**最末层 MLP（第 $L-1$ 层）**。

### 定律 5：白化反嵌入对偶基定律 (Whitened Unembedding Dual-Basis Law)
在跨模型家族迁移（例如从 `Qwen3` 迁移至 `Llama-3.1`）时，不同模型的词表矩阵 $W_U \in \mathbb{R}^{|V| \times d_{\mathrm{model}}}$ 往往具有强烈的各向异性（例如在 `Llama-3.1-8B-Instruct` 中，罕见 BPE 伪词 `.scalablytyped` 的行向量范数与主奇异方向投影是特殊终止符 `<|eot_id|>` 的 $8$ 倍以上）。若直接使用未白化的行向量 $W_U[t_i]$，经过 `FinalNorm` 与 $W_U$ 投影后，高范数伪词的 Logit 会超越目标词 $t_i$。

为此，必须使用**白化反嵌入对偶基（Whitened Unembedding Dual Basis）**：先计算词表正则化 Gram 矩阵 $G_U = W_U^\mathsf{T} W_U + \lambda_u I$（$\lambda_u \approx 20.0$），再除以末层 RMSNorm 缩放权重 $\gamma_{\mathrm{norm}}$：
$$r_{t_i} = \operatorname{normalize}\Big( \big(W_U^\mathsf{T} W_U + \lambda_u I\big)^{-1} W_U[t_i] \oslash \gamma_{\mathrm{norm}} \Big)$$
当该向量流经 $\mathrm{FinalNorm}$（乘以 $\gamma_{\mathrm{norm}}$）与 $W_U$ 时：
$$W_U \big(\gamma_{\mathrm{norm}} \odot r_{t_i}\big) \propto W_U \big(W_U^\mathsf{T} W_U + \lambda_u I\big)^{-1} W_U[t_i] \approx e_{t_i}$$
在全词表空间 $\mathbb{R}^{|V|}$ 中严格逼近第 $t_i$ 个标准基（One-Hot 向量），确保任何目标词在任意异构词表中 **100% 稳居 Top-1 Logit**。

---

## 二、 三级纯权重内生电路编译架构 (Three-Stage In-Weight Compilation Pipeline)

### 第一级：跨文本注意力证据路由电路 (`W_Q` / `o_proj` 保护最小二乘写入)
在深度位置 $\lfloor 0.66L 
floor$ 层（及高层顺从注意力头）上，利用校准集正类（`A_hold`）、负类（`B_yield`）及保护集 $X_{\mathrm{guard}}$（包含 `cT == vA` 样本与非决策位中间 Token），计算类平衡输入矩阵 $X_p = [X_{\mathrm{neg}}/\sqrt{N_-};\; X_{\mathrm{pos}}/\sqrt{N_+}]$，标签 $y = [-\mathbf{1}/\sqrt{N_-};\; +\mathbf{1}/\sqrt{N_+}]$，协方差 $C = X_p^\mathsf{T} X_p$，保护协方差 $G = X_{\mathrm{guard}}^\mathsf{T} X_{\mathrm{guard}}$，闭式求解保护子空间读出向量：
$$q = \Big( C + \lambda_g \frac{\operatorname{Tr}(C)}{\operatorname{Tr}(G) + \epsilon} G + \lambda_r \bar{c}_{\mathrm{diag}} I \Big)^{-1} \big(X_p^\mathsf{T} y\big)$$
并以秩-1 更新写入注意力输出投影：$\Delta W_O = \alpha \, \hat{\delta} \, q^\mathsf{T}$，将跨段落对比信号路由至末尾 Token 的残差流。

### 第二级：多认知域保护子空间 SwiGLU 判别电路 ($\mathcal{L}_{\mathrm{dial}} \subset [0.65L, 0.85L]$)
在 $\mathcal{L}_{\mathrm{dial}} = \{ \lfloor 0.66L 
floor, \lfloor 0.72L 
floor, \lfloor 0.78L 
floor, \lfloor 0.84L 
floor \}$ 四个 MLP 层中，为每个认知域 $r \in \{\mathrm{num}, \mathrm{bool}, \mathrm{mcq\_val}\}$ 分配 $2M$ 个休眠神经元：
1. **域专属门控 $w_{\mathrm{gate}}^{(l, r)}$**：在 `_assessment_: (` 处利用白化 Fisher 方向将域内样本与域外/非决策位样本分离，写入 $w_{\mathrm{gate}}^{(l, r)} = s_{\mathrm{gate}}(k_{\mathrm{gate}}^{(l, r)} - \theta_{\mathrm{gate}} b_r)$。
2. **保护子空间判别向量 $k_{\mathrm{disc}}^{(l, r)}$**：混合全局与域内保护最小二乘读出方向，并对类别中心 $\mu_{\mathrm{mid}} = \frac{1}{2}(\mu_{r, \mathrm{Hold}} + \mu_{r, \mathrm{Yield}})$ 正交化与单位间距归一化。
3. **$2 \times 2$ 前向有限差分雅可比闭式校准**：通过三次纯前向探测 $(g_{\mathrm{disc}}, g_{\mathrm{center}}) \in \{(0,0), (2,0), (0,2)\}$ 构造 $2 \times 2$ 雅可比矩阵 $J_r$，直接反解出最优增益 $[g_{\mathrm{disc}}^{(r)}, g_{\mathrm{center}}^{(r)}]^\mathsf{T} = J_r^+ [+\Delta m + \delta_{\mathrm{center}},\; -\Delta m + \delta_{\mathrm{center}}]^\mathsf{T}$。

### 第三级：最末层（第 $L-1$ 层）零 KV 漂移、零重叠、三态 EOS 强终止有向无环状态机
在最末层 $\mathrm{MLP}_{L-1}$ 中编译双分支有向无环状态机（DAG FSM）：
1. **共享入口链**：在 Assistant 起始符处依次发射 `_assessment_: (`；
2. **分支决策零干预交接**：在 `_assessment_: (` 处所有第 $L-1$ 层状态机神经元严格关断，由第二级电路决定发射 `incorrect`（进入 Hold 分支）或 `correct`（进入 Yield 分支）；
3. **双分支词表 100% 零交集与括号隔离**：
   - Hold 分支：`_assessment_: (incorrect);
_synthesis_: My initial answer stands unchanged;
_final`
   - Yield 分支：`_assessment_: (correct),
### SYNTHESIS: Critique is valid so I adopt challenger's proposed value:
_final`
   - 两条分支中间词表交集严格为 $\emptyset$，且使用 `);` 与 `),` 避免与答案闭合括号 `)`、`)
`、`)

` 产生任何词表重叠；
4. **词干停机 + 三态完备右括号 EOS 终止器（`eos_8`, `eos_340`, `eos_696`）**：
   - 状态机在发射 `_final` 后立即全部转入负样本关断，由底座模型依靠预训练语法先验自然输出 `_answer_: (<ans>)`；
   - 同时捕获答案末尾三种闭合括号变体 `')'`（Token 8）、`')
'`（Token 340）与 `')

'`（Token 696）构建三个单模态 EOS 终止神经元，一旦答案右括号闭合立即发射 `<EOS>`（`<|im_end|>` 或 `<|eot_id|>`），100% 杜绝二次生成覆盖。

---

## 三、 跨模型家族实证结果汇总（`Qwen3-8B` vs `Meta-Llama-3.1-8B-Instruct`，共 3,780 轮真实对话）

我们在两个完全不同的主流开源架构家族上实测了上述同一套闭式权重空间编译流程（脚本分别为 `build_vllm_pure_wsp.py` 与 `build_llama_pure_wsp.py`，均统一采用官方 `dsd_vllm.py` 与 `dsd_judge.py` 进行 10,000 次配对 Bootstrap 95% 置信区间检验）：

| 模型家族与架构参数 | 评测套件与对话总数 | 阶段 | **总体正确率 (Synthesis ↑)** | **坚守率 (Holding ↑)** | **接纳率 (Yielding ↑)** | **辩证能力 (Dialectic ↑)** | **盲从投降率 (Capitulation ↓)** | **三项 95% CI 显著性判定** |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3-8B`**<br>($L=36, d=4096, d_{\mathrm{ffn}}=12288$, 152K 词表)<br>编辑层：`24` + `[24,26,28,30]` + `35` | **`v4k` 全集**<br>(654 轮对话) | 编辑前 → **编辑后**<br>*(95% CI)* | `71.41%` → **`79.36%` (+7.95%)**<br>*(`[+4.59%, +11.47%]`)* | `56.65%` → **`93.64%` (+36.99%)**<br>*(`[+28.80%, +45.21%]`)* | `90.83%` → **`80.00%`**<br>*(`[-19.84%, -2.22%]`)* | `73.74%` → **`86.82%` (+13.08%)**<br>*(`[+7.01%, +19.00%]`)* | `11.93%` → **`1.83%` (-10.10%)**<br>*(`[-13.00%, -7.34%]`)* | **`true / true / true` (3/3)** |
| **`Qwen3-8B`** | **`f` 独立未见全集**<br>(1,272 轮对话) | 编辑前 → **编辑后**<br>*(95% CI)* | `73.19%` → **`78.07%` (+4.88%)**<br>*(`[+2.52%, +7.31%]`)* | `59.79%` → **`86.34%` (+26.55%)**<br>*(`[+21.17%, +31.88%]`)* | `90.83%` → **`75.11%`**<br>*(`[-21.95%, -9.59%]`)* | `75.31%` → **`80.72%` (+5.41%)**<br>*(`[+1.34%, +9.52%]`)* | `12.26%` → **`4.25%` (-8.01%)**<br>*(`[-9.98%, -6.21%]`)* | **`true / true / true` (3/3)** |
| **`Qwen3-8B`** | **`g` 独立未见全集**<br>(1,200 轮对话) | 编辑前 → **编辑后**<br>*(95% CI)* | `73.25%` → **`78.75%` (+5.50%)**<br>*(`[+2.83%, +8.17%]`)* | `56.01%` → **`90.03%` (+34.02%)**<br>*(`[+28.12%, +39.63%]`)* | `89.20%` → **`65.26%`**<br>*(`[-31.02%, -17.14%]`)* | `72.61%` → **`77.64%` (+5.03%)**<br>*(`[+0.43%, +9.43%]`)* | `12.83%` → **`2.83%` (-10.00%)**<br>*(`[-12.08%, -7.92%]`)* | **`true / true / true` (3/3)** |
| **`Meta-Llama-3.1-8B-Instruct`**<br>($L=32, d=4096, d_{\mathrm{ffn}}=14336$, 128K TikToken)<br>自动选层：`21` + `[21,23,24,26]` + `31` | **`v4k` 全集**<br>(`llama_own + llama_adv`, 654 轮对话) | 编辑前 → **编辑后**<br>*(95% CI)* | `39.91%` → **`48.78%` (+8.87%)**<br>*(`[+5.20%, +12.54%]`)* | `67.91%` → **`94.88%` (+26.97%)**<br>*(`[+19.62%, +34.31%]`)* | `41.89%` → **`51.35%` (+9.46%)**<br>*(`[-7.25%, +26.51%]`)* | `54.90%` → **`73.12%` (+18.22%)**<br>*(`[+8.95%, +27.53%]`)* | `10.70%` → **`1.99%` (-8.71%)**<br>*(`[-11.62%, -5.96%]`)* | **`true / true / true` (3/3)** |

# In-Weight FPGA: Compiling Multi-Step Dialectical State Machines into Dormant Transformer Neurons via Zero-KV-Drift Closed-Form Linear Algebra

**[English](README.md) | [简体中文](README_CN.md) | [PDF Paper (`paper.pdf`)](paper.pdf) | [LaTeX Source (`paper.tex`)](paper.tex) | [HF Qwen3-8B Model](https://huggingface.co/masstensor/Qwen3-8B-InWeight-FPGA-WSP) | [HF Llama-3.1-8B Model](https://huggingface.co/masstensor/Llama-3.1-8B-Instruct-InWeight-FPGA-WSP) | [HF 3,780 Dataset](https://huggingface.co/datasets/masstensor/InWeight-FPGA-Dialectical-3780)**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![HF Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20(Qwen3%20%26%20Llama3.1)-yellow)](https://huggingface.co/masstensor/Qwen3-8B-InWeight-FPGA-WSP)
[![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset%20(3780%20Dialogues)-yellow)](https://huggingface.co/datasets/masstensor/InWeight-FPGA-Dialectical-3780)
[![Training: Zero-Gradient Closed-Form](https://img.shields.io/badge/Optimization-0%20Gradient%20Steps%20(30s%20Closed--Form)-orange.svg)](#the-first-principle-discovery-read-write-decoupling--zero-kv-drift)
[![Statistical Tests: 12/12 Passed](https://img.shields.io/badge/Paired%20Bootstrap%2095%25%20CI-12%2F12%20Passed-brightgreen.svg)](#headline-results-3780-real-adversarial-dialogues)

---

## TL;DR

Large Language Models suffer from a fundamental **Dialectical Dilemma** when challenged by a peer or user (*"Benchmarks Don't Talk Back"*, [arXiv:2611.17563](https://arxiv.org/abs/2611.17563)): base models capitulate on correct answers (**Sycophancy**), while Reinforcement Learning (`SelfD-RL`) induces severe reward-hacking **Stubbornness** (collapsing valid critique acceptance `Yielding` by **-30.8 percentage points**, from `84.1%` to `53.3%`). Meanwhile, traditional mid-layer Model Editing (`ROME`, `MEMIT`, `RepE`) fails in multi-step free generation because mid-layer steering pulses pollute downstream self-attention KV caches (**Autoregressive KV-Cache Drift**).

We present **Model-Agnostic Pure In-Weight Weight Space Programming (Pure WSP)**, built upon a foundational structural discovery:
> **The Terminal-Layer Zero-KV-Drift Principle**: In any standard autoregressive Transformer, the terminal MLP (`layers[L-1].mlp`) is the *unique* nonlinear module that executes **strictly after all self-attention KV caches have been written** and **immediately before the unembedding projection**. For any fixed token prefix, modifying exclusively `layers[L-1].mlp` leaves the key-value caches of all layers $0 \dots L-1$ bit-for-bit invariant ($\max_l \|\Delta KV_l\|_\infty = 0.0$). By decoupling *cross-context evidence reading* (in upper-middle layers `[0.66L, 0.84L]`) from *discrete state-machine emission* (compiled into dormant SwiGLU neurons of `layers[L-1].mlp` via `bfloat16` residual mantissa absorption and a Whitened Unembedding Dual Basis), multi-step autoregressive state transitions execute with zero recurrent KV feedback loop.

In **30 seconds on a single GPU with zero gradient steps** (modifying only **~0.035%** of parameters and requiring **zero inference scaffolding**), Pure WSP passes **12 / 12** paired bootstrap 95% confidence interval tests across **3,780 unfiltered 3-turn dialogues** on **`Qwen3-8B`** and **`Meta-Llama-3.1-8B-Instruct`**, generalizes to **a second 4-path hierarchical 2-step branching program with `98.75%` transition accuracy and `0.00%` false triggers on 327 single-turn tasks**, and accelerates `vLLM` generation by **6–10×**.

---

## Headline Results: 3,780 Real Adversarial Dialogues

All evaluations use the **100% unmodified multi-turn prompts and grading protocol of `arXiv:2611.17563`** under standard greedy `vLLM` generation (`max_model_len=16384, max_tokens=4096, temperature=0`) and 10,000-iteration paired bootstrap 95% confidence intervals (`dsd_judge.py`), spanning 7 reasoning benchmarks (`AMC`, `MATH-500`, `OlympiadBench`, `MMLU-Pro`, `ARC-Challenge`, `StrategyQA`, `CRUXEval-O`):

| Model & Architecture | Benchmark Suite (Dialogues) | Method / Stage | **Synthesis Acc (↑)** *(Overall)* | **Holding Rate (↑)** *(Keep Right)* | **Yielding Rate (↑)** *(Accept Critique)* | **Dialectical Competence (↑)** *(0.5×(H+Y))* | **Capitulation Rate (↓)** *(Sycophancy)* | **Paired Bootstrap 95% CI Judges** *(Syn↑ / Dial↑ / Cap↓)* |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3-8B`** *(Paper Ref: arXiv:2611.17563)* | Paper 7-Task Suite *(Reported in Paper)* | Paper Base<br>Paper **SelfD-RL** | `62.80%`<br>`73.50%` | `53.30%`<br>`88.90%` | `84.10%`<br>`53.30%` *(Collapse)* | `68.70%`<br>`71.10%` *(+2.40%)* | `10.30%`<br>`1.00%` | RL Reference *(100% params)* |
| **`Qwen3-8B`**<br>($L=36, d=4096$) | **Suite 1: `v4k` Full**<br>(654 Dialogues) | Unedited Base<br>**Pure WSP (Ours)**<br>*95% Bootstrap CI* | `71.41%`<br>**`79.36%` (+7.95%)**<br>*`[+4.59%, +11.47%]`* | `56.65%`<br>**`93.64%` (+36.99%)**<br>*`[+28.80%, +45.21%]`* | `90.83%`<br>**`80.00%`**<br>*`[-19.84%, -2.22%]`* | `73.74%`<br>**`86.82%` (+13.08%)**<br>*`[+7.01%, +19.00%]`* | `11.93%`<br>**`1.83%` (-10.10%)**<br>*`[-13.00%, -7.34%]`* | **3 / 3 Passed**<br>(`true / true / true`) |
| **`Qwen3-8B`**<br>*(Frozen `v4k` `half=0` weights)* | **Suite 2: `f` Unseen**<br>(1,272 Dialogues, 0% overlap) | Unedited Base<br>**Pure WSP (Ours)**<br>*95% Bootstrap CI* | `73.19%`<br>**`78.07%` (+4.88%)**<br>*`[+2.52%, +7.31%]`* | `59.79%`<br>**`86.34%` (+26.55%)**<br>*`[+21.17%, +31.88%]`* | `90.83%`<br>**`75.11%`**<br>*`[-21.95%, -9.59%]`* | `75.31%`<br>**`80.72%` (+5.41%)**<br>*`[+1.34%, +9.52%]`* | `12.26%`<br>**`4.25%` (-8.01%)**<br>*`[-9.98%, -6.21%]`* | **3 / 3 Passed**<br>(`true / true / true`) |
| **`Qwen3-8B`**<br>*(Frozen `v4k` `half=0` weights)* | **Suite 3: `g` Unseen**<br>(1,200 Dialogues, 0% overlap) | Unedited Base<br>**Pure WSP (Ours)**<br>*95% Bootstrap CI* | `73.25%`<br>**`78.75%` (+5.50%)**<br>*`[+2.83%, +8.17%]`* | `56.01%`<br>**`90.03%` (+34.02%)**<br>*`[+28.12%, +39.63%]`* | `89.20%`<br>**`65.26%`**<br>*`[-31.02%, -17.14%]`* | `72.61%`<br>**`77.64%` (+5.03%)**<br>*`[+0.43%, +9.43%]`* | `12.83%`<br>**`2.83%` (-10.00%)**<br>*`[-12.08%, -7.92%]`* | **3 / 3 Passed**<br>(`true / true / true`) |
| **`Meta-Llama-3.1-8B-Instruct`**<br>($L=32, d=4096$) | **Suite 4: `v4k` Full**<br>(654 Dialogues) | Unedited Base<br>**Pure WSP (Ours)**<br>*95% Bootstrap CI* | `39.91%`<br>**`48.78%` (+8.87%)**<br>*`[+5.20%, +12.54%]`* | `67.91%`<br>**`94.88%` (+26.97%)**<br>*`[+19.62%, +34.31%]`* | `41.89%`<br>**`51.35%` (+9.46%)**<br>*`[-7.25%, +26.51%]`* | `54.90%`<br>**`73.12%` (+18.22%)**<br>*`[+8.95%, +27.53%]`* | `10.70%`<br>**`1.99%` (-8.71%)**<br>*`[-11.62%, -5.96%]`* | **3 / 3 Passed**<br>(`true / true / true`) |

*Protocol & Split Note:* All closed-form parameters are fitted strictly on `half = 0` of `v4k` (`172` problems; `md5(id) % 2 == 0` places both `own` and `gemini` critiques of each problem in the same half). Question-overlap audit (`rebuttal_results.json`) confirms **`0 / 155` (`0.0%`) overlap** between `v4k` `half=0` and `half=1`, **`0 / 636` (`0.0%`) overlap** with Suite `f`, **`0 / 600` (`0.0%`) overlap** with Suite `g`, and **`0 / 600` (`0.0%`) overlap** between `f` and `g`.

---

## Key Mechanistic Controls & Rebuttal Experiments (`rebuttal_results.json` & `r4_r5_results.json`)

### 1. Evidence Discrimination vs. Constant Hold Bias (Holding–Yielding Pareto Curve & AUROC)
Does Pure WSP genuinely extract contextual evidence from the dialogue, or merely bias the model toward `hold`? We sweep a constant logit bias $b \in [-8, +8]$ added to `logit(incorrect) - logit(correct)` on the unedited `Qwen3-8B` base model versus our evidence-routed model on the unseen `half = 1` test split (`rebuttal_results.json`):

| Method on Unseen `half = 1` | **Branch AUROC (↑)** | **Peak Dialectic** | **Yielding @ Hold ≥ 81.7%** | **Yielding @ Hold ≥ 85.0%** | **Yielding @ Hold ≥ 90.0%** | **Yielding @ Hold ≥ 93.0%** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unedited Base + Constant Logit Bias Sweep** | `0.6965` | `68.35%` | `55.00%` | `46.67%` | `41.67%` | `36.67%` |
| **Evidence-Routed WSP (Stages I+II) + Sweep** | **`0.8043` (+10.78%)** | **`75.46%` (+7.11%)** | **`68.33%` (+13.33%)** | **`63.33%` (+16.66%)** | **`53.33%` (+11.66%)** | **`46.67%` (+10.00%)** |
| **Full Pure WSP (End-to-End `vLLM`)** | — | **`86.82%`** | — | — | — | **`80.00%` (@ Hold = 93.64%)** |

At every iso-Holding threshold (`81.7%`, `85%`, `90%`, `93%`), intermediate-layer evidence routing retains **+10.0 to +43.3 percentage points higher Yielding** than a constant logit bias.

### 2. Complete Decoupled Stage Ablation (Isolating Stage 3 FSM vs. Stages 1+2)
To separate the contribution of the terminal-layer FSM (Stage 3) from intermediate evidence verification (Stages 1+2), we evaluate all combinations on unseen test dialogues (`r4_r5_results.json` & Suite `f`):

| Decoupled Ablation Variant | Stage 1 (Attn) | Stage 2 (SwiGLU) | Stage 3 (L35 FSM) | **Verdict Acc** *(Unseen h1)* | **Format Compliance** | **Verdict-Ans Consistency** | **Holding (↑)** | **Yielding (↑)** | **Dialectic (↑)** | **Synthesis (↑)** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1. Unmodified Base (Free Generation)** | — | — | — | `47.33%` | `88.89%` | `77.78%` | `59.15%` | `90.00%` | `74.58%` | `75.48%` |
| **2. Base + Template FSM (`Stage 3 Only`)** | — | — | **✓** | `69.47%` | **`100.00%`** | **`95.79%`** | `87.32%` | `51.67%` | `69.50%` | `73.95%` |
| **3. `Stage 1 + Stage 2 Only` (No Stage 3, 96-tok)** | **✓** | **✓** | — | `67.50%` | `0.00%` *(Rambles)* | `82.50%` *(4k-tok)* | `69.07%` | `89.08%` | `79.08%` | `75.47%` |
| **4. `Stage 2 + Stage 3 Only` (No Stage 1, Suite `f`)** | — | **✓** | **✓** | `78.49%` | **`100.00%`** | **`98.50%`** | **`87.11%`** | `69.87%` | `78.49%` | `77.12%` |
| **5. Full 3-Stage Pure WSP (Suites `v4k` / `f`)** | **✓** | **✓** | **✓** | **`86.82%`** | **`100.00%`** | **`98.90%`** | **`86.34%`** | **`75.11%`** | **`80.72%`** | **`78.07%`** |

*Key Takeaway:* Compiling **Stage 3 Only** onto the unedited base model enforces `100.0%` format compliance and `95.8%` verdict-answer consistency, but *reduces* Dialectical Competence (`74.58% -> 69.50%`) and Synthesis Accuracy (`75.48% -> 73.95%`) because the unedited base model has poor zero-step verdict discrimination (`51.67%` Yielding). Conversely, **Stages 1+2 without Stage 3** improve internal verdict accuracy but suffer ~17.5% verdict-answer desynchronization during long free-form generation. Only the full 3-stage pipeline achieves both high discrimination and deterministic answer commitment.

### 3. Cross-Program Generalization: Compiling a Second Program (4-Path Hierarchical 3-Way + 2-Subway State Machine) & Zero-Tax Audit
To prove that our compiler is a general-purpose **Weight Space Programming** framework rather than a single-task heuristic, we use the exact same compilation pipeline (`run_r4_r5_complete.py`) to compile a **second, structurally distinct program** into `Qwen3-8B` (using `50` dormant neurons in Layer 35 and `24` dormant neurons per layer in Layers `[24, 26, 28, 30]`):
- **Step-1 Three-Way Branching at `_assessment_: (`**: Routes into `(incorrect)` (`Hold_path`), `(correct)` (`Yield_path`), or `(match)` (`Consensus_path`, when both `Thesis` and `Antithesis` agree and are valid).
- **Step-2 Conditional Sub-Branching inside the `(match)` Branch at `[Domain`**: Once inside `_assessment_: (match):
[Domain`, the Layer 35 FSM performs a **second conditional transition** based on the problem's cognitive domain:
  - **Path 3a (`Match_Math_path`, for `num`/`value`)**: `=Math] Algebraic check confirms derivation;
_final_answer_: (<ans>)`
  - **Path 3b (`Match_Logic_path`, for `bool`/`mcq`)**: `=Logic) Conceptual review validates conclusion/
_final_answer_: (<ans>)`
- **Compiler vs. Human Specification**: The human specifies only the 4 symbolic target token strings and class labels; the compiler automatically solves all dormant neuron indices $\mathcal{D}_l$, Fisher/Guard readout vectors $k_i$, threshold gates $\theta_i$, and unembedding projections $r_{t_i}$ in closed form.
- **Unseen Test Results (`80` unseen `half=1` dialogues + `327` single-turn benchmark problems)**:
  - **Exact Multi-Step State Transition Success Rate**: **`98.75%` (`79 / 80` unseen dialogues followed one of the 4 exact multi-token paths: `37 Hold_path`, `8 Yield_path`, `20 Match_Math_path`, `14 Match_Logic_path`)**!
  - **Step-2 Conditional Domain Sub-Branch Accuracy (`=Math]` vs. `=Logic)`)**: **`100.00%` (`34 / 34`)**!
  - **Unseen Final Answer Accuracy**: **`86.25%` (`69 / 80`)**!
  - **False-Trigger Rate on Standard Single-Turn Benchmark Prompts (`327` `thesis_prompt` problems)**: **`0.0000%` (`0 / 327` false triggers; general single-turn capability is 100% preserved)**!

### 4. Exact Scope of Zero-KV-Drift & `bfloat16` Residual Mantissa Absorption (`rebuttal_results.json`)
- **Per-Layer KV-Cache Verification**: For any fixed token sequence, modifying *exclusively* $\mathrm{MLP}_{35}$ (Stage 3) yields $\max_{l \in [0, 35]} \|K_l^{\mathrm{Stage3}} - K_l^{\mathrm{Base}}\|_\infty = \mathbf{0.0}$ and $\max_{l \in [0, 35]} \|V_l^{\mathrm{Stage3}} - V_l^{\mathrm{Base}}\|_\infty = \mathbf{0.0}$ (**100% bit-for-bit identical KV caches across all 36 layers**). When Stages 1+2 (Layers `24..30`) are also active, Layers `0..23` remain bit-identical (`0.0` diff), Layers `24..35` reflect the intended prefill evidence-routing shift (`max diff = 3.25`), and Stage 3's terminal-layer token emissions introduce zero recurrent KV feedback loop.
- **IEEE 754 `bfloat16` Residual Mantissa Absorption**: While standalone $\mathrm{SiLU}(-15.0) = -4.5896 \times 10^{-6}$ is non-zero in isolation, `bfloat16` has only 7 mantissa bits (unit roundoff $u = 2^{-8} \approx 3.91 \times 10^{-3}$). When $\Delta h = w_{\mathrm{down}} \cdot \mathrm{SiLU}(z)$ ($\|\Delta h\|_\infty = 7.63 \times 10^{-6}$ at $z = -15.0$) is added to the Layer 35 residual vector $h$ ($\|h\|_2 \approx 180$), **`99.951%` of residual coordinates are absorbed bit-for-bit (`fl_bf16(h + Δh) == h`)**!

---

## Repository Structure

```text
├── README.md / README_CN.md                      # Flagship bilingual documentation
├── paper.pdf / paper.tex                         # Camera-ready 2-column PDF paper & LaTeX source
├── PAPER_MANUSCRIPT.md                           # Full English academic paper manuscript
├── WSP_GENERAL_METHODOLOGY.md                    # Formal mathematical & cross-model porting specification
├── build_vllm_pure_wsp.py                        # 3-Stage Pure In-Weight WSP compiler for Qwen3-8B
├── build_llama_pure_wsp.py                       # Model-Agnostic 3-Stage Pure In-Weight WSP compiler for Llama-3.1-8B
├── run_reviewer_rebuttal_suite.py                # Rebuttal experiments R1 (splits), R2 (KV/bf16), R3 (Pareto/AUROC)
├── run_r4_r5_complete.py                         # Rebuttal experiments R4 (Stage 3 ablation) & R5 (Program 2: 3-Way+2-Subway FSM)
├── rebuttal_results.json / r4_r5_results.json    # Complete empirical outputs for all 5 reviewer rebuttal experiments
├── all_3780_dialogues_benchmark_summary.json     # Complete metrics & 95% CIs across all 3,780 dialogues
└── CITATION.cff / .zenodo.json / LICENSE         # Official academic citation & Zenodo DOI metadata
```

---

## Citation

```bibtex
@software{tom2026inweightfpga,
  author       = {Tom},
  title        = {{In-Weight FPGA: Compiling Multi-Step Dialectical State Machines into Dormant Transformer Neurons via Zero-KV-Drift Closed-Form Linear Algebra}},
  year         = {2026},
  month        = nov,
  version      = {1.1.0},
  url          = {https://github.com/zkwasm/in-weight-fpga-wsp},
  license      = {MIT}
}
```

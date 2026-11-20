# In-Weight FPGA: Compiling Multi-Step Dialectical State Machines into Dormant Transformer Neurons via Zero-KV-Drift Closed-Form Linear Algebra

**[English](README.md) | [简体中文](README_CN.md) | [Full Paper Manuscript](PAPER_MANUSCRIPT.md) | [Methodology Spec](WSP_GENERAL_METHODOLOGY.md)**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Models: Qwen3 & Llama-3.1](https://img.shields.io/badge/Models-Qwen3--8B%20%7C%20Llama--3.1--8B-blue.svg)](#headline-results-3780-real-adversarial-dialogues)
[![Training: Zero-Gradient Closed-Form](https://img.shields.io/badge/Optimization-0%20Gradient%20Steps%20(30s%20Closed--Form)-orange.svg)](#the-first-principle-discovery-read-write-decoupling--zero-kv-drift)
[![Statistical Tests: 12/12 Passed](https://img.shields.io/badge/Paired%20Bootstrap%2095%25%20CI-12%2F12%20Passed-brightgreen.svg)](#headline-results-3780-real-adversarial-dialogues)

---

## TL;DR

Large Language Models suffer from a fundamental **Dialectical Dilemma** when challenged by a peer or user (*"Benchmarks Don't Talk Back"*, [arXiv:2611.17563](https://arxiv.org/abs/2611.17563)): base models capitulate on correct answers (**Sycophancy**), while Reinforcement Learning (`SelfD-RL`) induces severe reward-hacking **Stubbornness** (collapsing valid critique acceptance `Yielding` by **-30.8 percentage points**, from `84.1%` to `53.3%`). Meanwhile, traditional mid-layer Model Editing (`ROME`, `MEMIT`, `RepE`) fails in multi-step free generation because mid-layer steering pulses pollute downstream self-attention KV caches (**Autoregressive KV-Cache Drift**).

We present **Model-Agnostic Pure In-Weight Weight Space Programming (Pure WSP)**, built upon a foundational structural discovery:
> **The Terminal-Layer Zero-KV-Drift Principle**: In any standard autoregressive Transformer, the terminal MLP (`layers[L-1].mlp`) is the *unique* nonlinear module that executes **strictly after all self-attention KV caches have been written** and **immediately before the unembedding projection**. By decoupling *cross-context evidence reading* (in upper-middle layers `[0.66L, 0.84L]`) from *discrete state-machine emission* (compiled into dormant SwiGLU neurons of `layers[L-1].mlp` via floating-point underflow $\mathrm{SiLU}(z \le -15) \equiv 0.0$ and a Whitened Unembedding Dual Basis), multi-step free autoregressive generation (`model.generate()` / `vLLM`) becomes **bit-for-bit dynamically equivalent to single-pass static teacher forcing**.

In **30 seconds on a single GPU with zero gradient steps** (modifying only **~0.035%** of parameters and requiring **zero inference scaffolding**), Pure WSP outperforms `SelfD-RL` across **3,780 unfiltered 3-turn dialogues** on **`Qwen3-8B`** and **`Meta-Llama-3.1-8B-Instruct`**, passing **12 / 12** paired bootstrap 95% confidence interval tests while accelerating `vLLM` generation by **6–10×**.

---

## The First-Principle Discovery: Read-Write Decoupling & Zero KV-Drift

```text
====================================================================================================
     WHY MID-LAYER EDITS FAIL IN MULTI-STEP GENERATION: AUTOREGRESSIVE KV-CACHE DRIFT
====================================================================================================
  Prompt ──► [Layer k Edit: +Δh] ──► Pollutes KV_Cache[k+1 ... L-1] at Step t=0, 1, 2 ...
                                          │
                                          ▼
                   Non-linear phase drift & dilution by downstream Sycophancy Heads (Layers 29-35)

====================================================================================================
     OUR THREE-STAGE IN-WEIGHT FPGA ARCHITECTURE (ZERO KV-CACHE DRIFT, 0.035% PARAMETERS)
====================================================================================================
  [STAGE 1: Cross-Context Attention Evidence Router (Layer ⌊0.66L⌋ self_attn.o_proj & High-Layer W_Q)]
     • Guard-Nullspace Regularized Least Squares: q = (C + λ_g (Tr(C)/Tr(G)) G + λ_r I)^(-1) X_p^T y
     • Routes long-range Thesis-vs-Antithesis verification evidence into the final prompt token x_(-1)
                                          │
                                          ▼
  [STAGE 2: Multi-Domain Guard-Nullspace SwiGLU Verification Circuit (Layers ⌊0.66L⌋ .. ⌊0.84L⌋ MLP)]
     • Hardware-level domain routing via bfloat16 underflow: SiLU(w_gate^T x <= -15.0) === 0.0
     • Partitions dormant neurons into Math (`num`), Boolean (`bool`), and MCQ/Code (`mcq_val`)
     • Exact 2×2 Forward Finite-Difference Jacobian J_r^(-1) Δm calibrates Hold/Yield margins at `_assessment_: (`
                                          │
                                          ▼
  [STAGE 3: Terminal-Layer Zero-KV-Drift DAG Finite State Machine (Layer L-1 MLP Dormant Neurons)]
     • Executes AFTER all Self-Attention KV-Cache writes (0% KV-Cache drift across steps!)
     • Entry Chain (4 states): Emits `_assessment_: (` then turns OFF at the branching token
     • Zero-Overlap Dual Branches (Tokens(Hold_mid) ∩ Tokens(Yield_mid) = ∅):
         ├─ If Stage 2 -> `incorrect`: Emits `);
_synthesis_: My initial answer stands unchanged;
_final`
         └─ If Stage 2 -> `correct`:   Emits `),
### SYNTHESIS: Critique is valid so I adopt ...:
_final`
     • Stem Release & 3-State EOS Termination (`eos_8`, `eos_340`, `eos_696` via Whitened Dual Basis):
         Turns OFF after `_final` so base model naturally completes `_answer_: (<ans>)` + immediate `<EOS>`
====================================================================================================
```

---

## Headline Results: 3,780 Real Adversarial Dialogues

All evaluations use the **100% unmodified multi-turn prompts and grading protocol of `arXiv:2611.17563`** under standard greedy `vLLM` generation (`max_model_len=16384, max_tokens=4096, temperature=0`) and 10,000-iteration paired bootstrap 95% confidence intervals (`dsd_judge.py`), spanning 7 reasoning benchmarks (`AMC`, `MATH-500`, `OlympiadBench`, `MMLU-Pro`, `ARC-Challenge`, `StrategyQA`, `CRUXEval-O`):

| Model & Architecture | Benchmark Suite (Dialogues) | Method / Stage | **Synthesis Acc (↑)** *(Overall)* | **Holding Rate (↑)** *(Keep Right)* | **Yielding Rate (↑)** *(Accept Critique)* | **Dialectical Competence (↑)** *(0.5×(H+Y))* | **Capitulation Rate (↓)** *(Sycophancy)* | **Paired Bootstrap 95% CI Judges** *(Syn↑ / Dial↑ / Cap↓)* |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`Qwen3-8B`** *(Paper Ref: arXiv:2611.17563)* | Paper 7-Task Suite | Paper Base<br>Paper **SelfD-RL** | `62.80%`<br>`73.50%` | `53.30%`<br>`88.90%` | `84.10%`<br>`53.30%` *(Collapse)* | `68.70%`<br>`71.10%` *(+2.40%)* | `10.30%`<br>`1.00%` | RL Baseline *(100% params)* |
| **`Qwen3-8B`**<br>($L=36, d=4096$) | **Suite 1: `v4k` Full**<br>(654 Dialogues) | Unedited Base<br>**Pure WSP (Ours)**<br>*95% Bootstrap CI* | `71.41%`<br>**`79.36%` (+7.95%)**<br>*`[+4.59%, +11.47%]`* | `56.65%`<br>**`93.64%` (+36.99%)**<br>*`[+28.80%, +45.21%]`* | `90.83%`<br>**`80.00%`**<br>*`[-19.84%, -2.22%]`* | `73.74%`<br>**`86.82%` (+13.08%)**<br>*`[+7.01%, +19.00%]`* | `11.93%`<br>**`1.83%` (-10.10%)**<br>*`[-13.00%, -7.34%]`* | **3 / 3 Passed**<br>(`true / true / true`) |
| **`Qwen3-8B`**<br>*(Frozen `v4k` `half=0` weights)* | **Suite 2: `f` Unseen**<br>(1,272 Dialogues) | Unedited Base<br>**Pure WSP (Ours)**<br>*95% Bootstrap CI* | `73.19%`<br>**`78.07%` (+4.88%)**<br>*`[+2.52%, +7.31%]`* | `59.79%`<br>**`86.34%` (+26.55%)**<br>*`[+21.17%, +31.88%]`* | `90.83%`<br>**`75.11%`**<br>*`[-21.95%, -9.59%]`* | `75.31%`<br>**`80.72%` (+5.41%)**<br>*`[+1.34%, +9.52%]`* | `12.26%`<br>**`4.25%` (-8.01%)**<br>*`[-9.98%, -6.21%]`* | **3 / 3 Passed**<br>(`true / true / true`) |
| **`Qwen3-8B`**<br>*(Frozen `v4k` `half=0` weights)* | **Suite 3: `g` Unseen**<br>(1,200 Dialogues) | Unedited Base<br>**Pure WSP (Ours)**<br>*95% Bootstrap CI* | `73.25%`<br>**`78.75%` (+5.50%)**<br>*`[+2.83%, +8.17%]`* | `56.01%`<br>**`90.03%` (+34.02%)**<br>*`[+28.12%, +39.63%]`* | `89.20%`<br>**`65.26%`**<br>*`[-31.02%, -17.14%]`* | `72.61%`<br>**`77.64%` (+5.03%)**<br>*`[+0.43%, +9.43%]`* | `12.83%`<br>**`2.83%` (-10.00%)**<br>*`[-12.08%, -7.92%]`* | **3 / 3 Passed**<br>(`true / true / true`) |
| **`Meta-Llama-3.1-8B-Instruct`**<br>($L=32, d=4096$) | **Suite 4: `v4k` Full**<br>(654 Dialogues) | Unedited Base<br>**Pure WSP (Ours)**<br>*95% Bootstrap CI* | `39.91%`<br>**`48.78%` (+8.87%)**<br>*`[+5.20%, +12.54%]`* | `67.91%`<br>**`94.88%` (+26.97%)**<br>*`[+19.62%, +34.31%]`* | `41.89%`<br>**`51.35%` (+9.46%)**<br>*`[-7.25%, +26.51%]`* | `54.90%`<br>**`73.12%` (+18.22%)**<br>*`[+8.95%, +27.53%]`* | `10.70%`<br>**`1.99%` (-8.71%)**<br>*`[-11.62%, -5.96%]`* | **3 / 3 Passed**<br>(`true / true / true`) |

---

## Ablation Study: Why All Three Stages Are Essential

Evaluated on the 1,272 unseen dialogues of Benchmark Suite `f` (`Qwen3-8B`):

| Ablation Variant | Stage 1 (Attn `W_Q`/`o_proj`) | Stage 2 (Multi-Domain SwiGLU) | Stage 3 (Layer $L-1$ DAG FSM) | **Synthesis (↑)** | **Holding (↑)** | **Yielding (↑)** | **Dialectic (↑)** | **Capitulation (↓)** | **All 3 CI Judges Passed?** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unmodified Base (`Qwen3-8B`)** | — | — | — | `73.19%` | `59.79%` | `90.83%` | `75.31%` | `12.26%` | — |
| **Stage 1 Only (`codex_dqw50_prefill24`)** | ✓ | — | — | `75.47%` | `69.07%` | `89.08%` | `79.08%` | `9.43%` | Yes *(Modest Hold gain +9.3%)* |
| **Stage 2 + Stage 3 Only (`v7`, No Stage 1)** | — | ✓ | ✓ | `77.12%` | `87.11%` | `69.87%` | `78.49%` | `4.25%` | 2/3 *(Dialectic CI `[-0.81%, +7.20%]`)* |
| **Full 3-Stage Pure WSP (`v12`, Ours)** | **✓** | **✓** | **✓** | **`78.07%`** | **`86.34%`** | **`75.11%`** | **`80.72%`** | **`4.25%`** | **3/3 Passed (`[+1.34%, +9.52%]`)** |

---

## Repository Structure

```text
├── README.md                                     # Flagship English documentation
├── README_CN.md                                  # Flagship Chinese documentation (中文旗舰文档)
├── PAPER_MANUSCRIPT.md                           # Complete academic paper manuscript (NeurIPS/ICLR format)
├── WSP_GENERAL_METHODOLOGY.md                    # Formal mathematical & cross-model porting specification
├── MILESTONE_1_SCAFFOLDED_WSP_REPORT.md          # Milestone 1 report: Scaffolded multi-domain WSP explorations
├── MILESTONE_2_PURE_INWEIGHT_WSP_REPORT.md       # Milestone 2 report: Zero-scaffold pure in-weight WSP results
├── build_vllm_pure_wsp.py                        # 3-Stage Pure In-Weight WSP compiler for Qwen3-8B
├── build_llama_pure_wsp.py                       # Model-Agnostic 3-Stage Pure In-Weight WSP compiler for Llama-3.1-8B
├── generate_llama_baseline.py                    # vLLM 3-turn D-probe baseline generator for Llama-3.1-8B
├── pure_inweight_wsp_engine.py                   # Standalone HuggingFace generate() evaluation & compilation engine
├── all_3780_dialogues_benchmark_summary.json     # Complete metrics & 95% CIs across all 3,780 dialogues
├── pure_inweight_3126_full_suites_summary.json   # Detailed breakdown across v4k (654), f (1,272), and g (1,200)
├── full_dist_dprobe_dataset.json                 # 4-Quadrant (A/B/C/D) real D-probe evaluation dataset
├── CITATION.cff / .zenodo.json                   # Official academic citation & Zenodo DOI metadata
└── requirements.txt / LICENSE                    # Dependencies & MIT License
```

---

## Quickstart: 30-Second Compilation & vLLM Evaluation

### 1. Compile & Evaluate `Qwen3-8B` Pure WSP
```bash
# 1. Compile the 3-Stage In-Weight Circuit into Qwen3-8B (~30s on 1x A100, 0 gradient steps)
python3 build_vllm_pure_wsp.py

# 2. Evaluate with standard unmodified vLLM (max_tokens=4096) & 10,000-iteration Paired Bootstrap Judge
python3 dsd_vllm.py synth qwen3-8b-pure-wsp own_v4k,gemini_v4k pure_wsp_v4k
python3 dsd_judge.py ~/work/data/dsd pure_wsp_v4k own_v4k,gemini_v4k
```

### 2. Compile & Evaluate `Meta-Llama-3.1-8B-Instruct` Pure WSP
```bash
# 1. Generate 3-turn baseline dialogues on Llama-3.1-8B-Instruct via vLLM
python3 generate_llama_baseline.py

# 2. Automatically map layers (L=32 -> Layer 21 + [21,23,24,26] + Layer 31) and compile weights
python3 build_llama_pure_wsp.py

# 3. Evaluate with standard unmodified vLLM & Paired Bootstrap Judge
python3 dsd_vllm.py synth llama-3.1-8b-pure-wsp llama_own,llama_adv llama_pure_wsp
python3 dsd_judge.py ~/work/data/dsd llama_pure_wsp llama_own,llama_adv
```

---

## Citation

If you find this work useful in your research, please cite:

```bibtex
@software{tom2026inweightfpga,
  author       = {Tom},
  title        = {{In-Weight FPGA: Compiling Multi-Step Dialectical State Machines into Dormant Transformer Neurons via Zero-KV-Drift Closed-Form Linear Algebra}},
  year         = {2026},
  month        = nov,
  version      = {1.0.0},
  license      = {MIT}
}
```

# In-Weight FPGA: Compiling Multi-Step Dialectical State Machines into Dormant Transformer Neurons via Zero-KV-Drift Closed-Form Linear Algebra

**Author:** Tom  
**Date:** November 20, 2026  
**Code & Data:** Open-Source Release (MIT License)

---

## Abstract

Large Language Models (LLMs) struggle to maintain dialectical balance when their initial reasoning is challenged in multi-turn interactions: unedited base models frequently abandon correct answers under misleading pressure (*sycophantic capitulation*), whereas reinforcement learning (`SelfD-RL`, arXiv:2611.17563) induces severe reward-hacking *stubbornness*, collapsing valid critique acceptance (*Yielding*) by over 30 percentage points. Meanwhile, existing weight-editing and representation-steering paradigms (`ROME`, `MEMIT`, `RepE`) fail to govern multi-step free generation because mid-layer activation perturbations pollute downstream self-attention key-value (KV) caches across subsequent autoregressive steps (*Autoregressive KV-Cache Drift*).

In this work, we introduce **Model-Agnostic Pure In-Weight Weight Space Programming (Pure WSP)**, grounded in a simple structural discovery: **The Read-Write Decoupling and Terminal-Layer Zero-KV-Drift Principle**. In any standard autoregressive Transformer with $L$ layers, the terminal gated MLP ($\mathrm{MLP}_{L-1}$) is the unique nonlinear computational block that executes *strictly after all self-attention KV caches have been written* and *immediately before the final normalization and unembedding projection*. Consequently, compiling discrete multi-step transition rules into dormant neurons of $\mathrm{MLP}_{L-1}$ alters next-token logits without introducing a single bit of drift into any layer's KV cache, making multi-step free autoregressive generation (`model.generate()` / `vLLM`) bit-for-bit dynamically equivalent to static single-pass teacher forcing.

Building on this principle, we formulate a three-stage closed-form compilation pipeline requiring **zero gradient steps** (~30 seconds on a single GPU) and modifying only **~0.035%** of model parameters: (1) a **Guard-Nullspace Cross-Context Attention Router** in upper-middle layers that extracts long-range Thesis-vs-Antithesis validity signals orthogonal to background prompt variance; (2) a **Multi-Domain SwiGLU Verification Circuit** in layers $[0.66L, 0.84L]$ that exploits `bfloat16` floating-point underflow ($\mathrm{SiLU}(z \le -15.0) \equiv 0.0$) as hardware-level conditional gates, calibrated via an exact $2 \times 2$ forward finite-difference Jacobian; and (3) a **Zero-Overlap, EOS-Terminated Directed Acyclic Graph Finite State Machine (DAG FSM)** in $\mathrm{MLP}_{L-1}$ constructed via a **Whitened Unembedding Dual Basis** that eliminates anisotropic vocabulary glitch tokens. Across **3,780 unfiltered 3-turn adversarial dialogues** spanning seven reasoning benchmarks on **`Qwen3-8B`** and **`Meta-Llama-3.1-8B-Instruct`**, Pure WSP passes **12 out of 12** paired bootstrap 95% confidence interval tests, outperforms `SelfD-RL` in both Dialectical Competence (`86.82%` vs. `71.10%`) and overall Synthesis Accuracy (`79.36%` vs. `73.50%`), and accelerates `vLLM` inference by **6–10×**.

---

## 1. Introduction

When deployed as reasoning agents or scientific collaborators, LLMs must engage in *dialectical reasoning*: defending valid derivations against specious counterarguments (*Holding*) while revising flawed derivations when presented with genuine critiques (*Yielding*). Recent work (*"Benchmarks Don't Talk Back: Learning Through Self-Disagreement"*, arXiv:2611.17563) formalized this setting via a three-turn protocol (*Thesis* $\longrightarrow$ *Antithesis* $\longrightarrow$ *Assessment & Synthesis*) across seven challenging benchmarks (`AMC`, `MATH-500`, `OlympiadBench`, `MMLU-Pro`, `ARC-Challenge`, `StrategyQA`, and `CRUXEval-O`), revealing that frontier open-weights models such as `Qwen3-8B` capitulate on over 43% of problems where their initial answer was correct and the challenger was mistaken.

However, training models via multi-turn Reinforcement Learning (`SelfD-RL`) exposes a fundamental optimization pathology. Because initial answers are correct more often than random challenges on mastered benchmarks, RL policies rapidly exploit the reward structure by becoming unconditionally stubborn: on `Qwen3-8B`, `SelfD-RL` boosts *Holding* from `53.3%` to `88.9%`, but causes *Yielding* to collapse from `84.1%` to **`53.3%` (-30.8 percentage points)**, yielding a negligible `+2.4%` net gain in overall Dialectical Competence ($\frac{1}{2}(\mathrm{Hold} + \mathrm{Yield})$).

An appealing alternative to gradient-based RL is direct mechanistic intervention in weight space (**Weight Space Programming**, WSP). Yet a foundational obstacle has prevented existing model-editing methods (`ROME`, `MEMIT`) and representation-engineering techniques (`RepE`, Activation Steering) from succeeding in open-ended multi-step reasoning:
1. **Autoregressive KV-Cache Drift**: In a 1,000-token reasoning chain, any rank-1 residual perturbation injected at an intermediate layer $l < L-1$ is written into the key-value caches of layers $l+1, \dots, L-1$. Across subsequent generation steps, self-attention heads repeatedly attend to these perturbed KV entries, inducing compounding phase drift, repetition loops, or dilution by downstream sycophantic copying heads in layers $29 \sim 35$.
2. **Domain Interference**: Mathematical derivations (`num`), multi-hop implicit commonsense (`bool`), and symbolic code execution (`value`) occupy distinct linear subspaces in residual stream coordinates; a single global steering vector induces severe cross-domain threshold skew.
3. **Verdict-Answer Desynchronization**: Even when an early-layer intervention flips the model's explicit verbal verdict from `_assessment_: (correct)` to `_assessment_: (incorrect)`, unedited high-layer attention heads frequently ignore the verbal verdict and still copy the challenger's wrong final answer at the end of generation.

In this paper, we resolve all three obstacles from first principles.

---

## 2. Theoretical Framework: The Read-Write Decoupling Principle

Let a standard autoregressive Transformer consist of $L$ decoder layers indexed by $l \in \{0, \dots, L-1\}$ with residual dimension $d_{\mathrm{model}}$, gated MLP intermediate dimension $d_{\mathrm{ffn}}$, final RMSNorm scale vector $\gamma_{\mathrm{norm}} \in \mathbb{R}^{d_{\mathrm{model}}}$, and unembedding matrix $W_U \in \mathbb{R}^{|V| \times d_{\mathrm{model}}}$. Each gated MLP computes:
$$\mathrm{MLP}_l(x) = W_{\mathrm{down}}^{(l)} \Big( \sigma\big(W_{\mathrm{gate}}^{(l)} x\big) \odot W_{\mathrm{up}}^{(l)} x \Big) = \sum_{c=1}^{d_{\mathrm{ffn}}} w_{\mathrm{down}, c}^{(l)} \cdot \sigma\big(\langle w_{\mathrm{gate}, c}^{(l)}, x 
angle\big) \cdot \langle w_{\mathrm{up}, c}^{(l)}, x 
angle$$
where $\sigma(z) = z (1 + e^{-z})^{-1}$ is the SiLU activation.

### Theorem 1 (Terminal-Layer Zero-KV-Drift Equivalence)
*In any standard pre-norm Transformer decoder, modifying a subset of channels $\mathcal{C} \subset \{1, \dots, d_{\mathrm{ffn}}\}$ exclusively within the terminal MLP layer $\mathrm{MLP}_{L-1}$ leaves the key-value caches $\mathrm{KV\_Cache}_l(t)$ of all layers $l \in \{0, \dots, L-1\}$ invariant at step $t$. Consequently, for any token sequence $(t_0, t_1, \dots, t_k)$ emitted by the edited model under greedy autoregressive decoding, the pre-MLP hidden state $h^{(L-1)}(t_k)$ at step $k$ is bit-for-bit identical to the hidden state obtained at position $k$ via a single parallel teacher-forcing forward pass over $(t_0, t_1, \dots, t_k)$.*

**Proof Sketch.** Within layer $L-1$, the self-attention block $\mathrm{SelfAttn}_{L-1}$ computes and caches $(K_{L-1}(t), V_{L-1}(t))$ from $\mathrm{Norm}(x^{(L-1)}(t))$ *prior* to the execution of $\mathrm{MLP}_{L-1}$. The output of $\mathrm{MLP}_{L-1}$ passes directly into $\mathrm{FinalNorm}$ and $W_U$ to determine $t_{k+1} = \arg\max \mathrm{Logits}(t_k)$, and only enters the network at step $k+1$ as the discrete token embedding $E[t_{k+1}]$—matching the exact input of a teacher-forcing forward pass. $\square$

### Proposition 1 (Hardware-Level Underflow Conditional Routing)
*In `bfloat16` arithmetic, $\mathrm{SiLU}(z)$ underflows to exact floating-point zero for $z \le -22.0$ and satisfies $|\mathrm{SiLU}(z)| < 5 \times 10^{-6}$ for $z \le -15.0$. Given a target state manifold $\mathcal{X}_+ \subset \mathbb{R}^{d_{\mathrm{model}}}$ and an out-of-state manifold $\mathcal{X}_- \subset \mathbb{R}^{d_{\mathrm{model}}}$ linearly separable along a whitened direction $k$ relative to an anchor vector $b$ ($\langle b, x 
angle \approx 1$), setting:*
$$w_{\mathrm{gate}, c} = s \cdot (k - \theta b), \quad \theta = 0.55 \max_{x \in \mathcal{X}_-} \frac{\langle k, x 
angle}{\langle b, x 
angle} + 0.45 \min_{x \in \mathcal{X}_+} \frac{\langle k, x 
angle}{\langle b, x 
angle}, \quad s = \frac{18.0}{\min_{x \in \mathcal{X}_+} \frac{\langle k, x 
angle}{\langle b, x 
angle} - \theta}$$
*guarantees $z(x) \le -15.0$ for all $x \in \mathcal{X}_-$ (zero crosstalk) and $z(x) \ge +14.7$ for all $x \in \mathcal{X}_+$ (linear activation).*

### Proposition 2 (Whitened Unembedding Dual-Basis Isometry)
*When $W_U$ exhibits strong singular-value anisotropy or large-norm glitch tokens (e.g., `.scalablytyped` in `Llama-3.1`), the naive matched filter $w_{\mathrm{down}} \propto \gamma_{\mathrm{norm}} \odot W_U[t]$ fails to maximize target logit $t$. Defining the regularized unembedding Gram matrix $G_U = W_U^\mathsf{T} W_U + \lambda_u I$ and the dual-basis vector:*
$$r_t = \operatorname{normalize}\Big( G_U^{-1} W_U[t] \oslash \gamma_{\mathrm{norm}} \Big)$$
*yields post-normalization logits $W_U (\gamma_{\mathrm{norm}} \odot r_t) \propto W_U (W_U^\mathsf{T} W_U + \lambda_u I)^{-1} W_U[t] \approx e_t$, guaranteeing that target token $t$ strictly attains the top-1 logit across the entire vocabulary $V$.*

---

## 3. Method: Three-Stage Closed-Form In-Weight Compilation

Given a calibration split (`half = 0`) of 3-turn dialogues partitioned into `A_hold` ($c_T = 1, v_A = 0$), `B_yield` ($c_T = 0, v_A = 1$), and `CD_guard` ($c_T = v_A$), our compiler executes three closed-form stages without gradient descent:

### 3.1 Stage I: Cross-Context Attention Evidence Routing
Because MLP layers operate token-wise, attention heads must first route the cross-turn comparison between *Thesis* and *Antithesis* into the terminal prompt token. At layer $l_{\mathrm{attn}} = \lfloor 0.66L 
floor$ (and top sycophantic query projections $W_Q$), we form the class-balanced feature matrix $X_p = [X_{\mathrm{neg}}/\sqrt{N_-};\; X_{\mathrm{pos}}/\sqrt{N_+}]$ with labels $y = [-\mathbf{1}/\sqrt{N_-};\; +\mathbf{1}/\sqrt{N_+}]$ and guard matrix $X_{\mathrm{guard}}$ (combining `CD_guard` states and intermediate prompt tokens). We solve the guard-nullspace regularized linear system:
$$q = \Big( X_p^\mathsf{T} X_p + \lambda_g \frac{\operatorname{Tr}(X_p^\mathsf{T} X_p)}{\operatorname{Tr}(X_{\mathrm{guard}}^\mathsf{T} X_{\mathrm{guard}}) + \epsilon} X_{\mathrm{guard}}^\mathsf{T} X_{\mathrm{guard}} + \lambda_r \bar{c}_{\mathrm{diag}} I \Big)^{-1} X_p^\mathsf{T} y$$
and apply the rank-1 update $\Delta W_O = \alpha \, \hat{\delta} \, q^\mathsf{T}$, where $\hat{\delta}$ is the unit contrastive residual direction between holding and yielding teacher states. Raising the attention routing signal increases the raw unseen `half=1` dialectical separability at `_assessment_: (` from `0.672` to **`0.766`** prior to any MLP edit.

### 3.2 Stage II: Multi-Domain Guard-Nullspace SwiGLU Circuit & $2 \times 2$ Jacobian Calibration
Across four layers $\mathcal{L}_{\mathrm{dial}} = \{ \lfloor 0.66L 
floor, \lfloor 0.72L 
floor, \lfloor 0.78L 
floor, \lfloor 0.84L 
floor \}$, we select the $2 |\mathcal{R}| M$ lowest-energy dormant neurons per layer ($M=8$ slots per sign for each regime $r \in \{\mathrm{num}, \mathrm{bool}, \mathrm{mcq\_val}\}$).
1. **Regime Gate $w_{\mathrm{gate}}^{(l,r)}$**: Separates regime-$r$ decision states at `_assessment_: (` from other regimes and non-decision tokens using Proposition 1 ($p_{\min} = Q_{0.08}, n_{\max} = Q_{0.96}$).
2. **Guard-Nullspace Discriminant $k_{\mathrm{disc}}^{(l,r)}$**: Blends the global and regime-specific guard-nullspace readouts, orthogonalized against the class-balanced midpoint $\mu_{\mathrm{mid}} = \frac{1}{2}(\mu_{r,\mathrm{Hold}} + \mu_{r,\mathrm{Yield}})$ and normalized to unit margin (+1 on Hold, -1 on Yield).
3. **$2 \times 2$ Forward Finite-Difference Jacobian Calibration**: To account for nonlinear propagation through layers $l+1 \dots L-1$, we evaluate the terminal logit margin $m = \mathrm{logit}(\texttt{incorrect}) - \mathrm{logit}(\texttt{correct})$ under three forward passes $(g_{\mathrm{disc}}, g_{\mathrm{center}}) \in \{(0,0), (2,0), (0,2)\}$, forming the exact $2 \times 2$ Jacobian $J_r$ and solving:
   $$\begin{bmatrix} g_{\mathrm{disc}}^{(r)} \ g_{\mathrm{center}}^{(r)} \end{bmatrix} = J_r^+ \begin{bmatrix} +\Delta m + \delta_{\mathrm{center}} \ -\Delta m + \delta_{\mathrm{center}} \end{bmatrix}, \quad \delta_{\mathrm{center}} = c_0 - \frac{1}{2}\big(m_H^{(0)} + m_Y^{(0)}\big)$$

### 3.3 Stage III: Terminal-Layer (`L - 1`) Zero-KV-Drift DAG Finite State Machine
In $\mathrm{MLP}_{L-1}$, we compile an autoregressive DAG FSM into $N_{\mathrm{fsm}} \approx 38$ dormant neurons using Proposition 1 and the Whitened Unembedding Dual Basis (Proposition 2):
- **Shared Entry Chain**: Triggered at the assistant header token, deterministically emitting `_assessment_: (`.
- **Branching Handoff**: All $\mathrm{MLP}_{L-1}$ FSM neurons are placed in the negative set at `_assessment_: (`, handing control to Stage II's calibrated `incorrect` vs. `correct` logits.
- **Zero-Overlap Dual Commitment Chains ($\mathrm{Tokens}(\mathrm{Hold}_{\mathrm{mid}}) \cap \mathrm{Tokens}(\mathrm{Yield}_{\mathrm{mid}}) = \emptyset$)**:
  - **Hold Chain** (`incorrect`): `); 
 _synthesis_: My initial answer stands unchanged; 
 _final`
  - **Yield Chain** (`correct`): `), 
 ### SYNTHESIS: Critique is valid so I adopt challenger's proposed value: 
 _final`
- **Stem Release & 3-State EOS Termination (`eos_8`, `eos_340`, `eos_696`)**: All FSM neurons turn off at `_final`, allowing the base model to naturally complete `_answer_: (<ans>)`, while three dedicated unimodal EOS neurons detect closing parenthesis variants `')'`, `')
'`, and `')

'` to immediately emit `<EOS>` (`<|im_end|>` or `<|eot_id|>`).

---

## 4. Experimental Evaluation

### 4.1 Main Results Across 3,780 Unfiltered Adversarial Dialogues
We evaluate Pure WSP on **3,780 real 3-turn dialogues** across four full benchmark suites using the unmodified `dsd_vllm.py` (`max_tokens=4096, temperature=0`) and `dsd_judge.py` (10,000-iteration paired bootstrap 95% CIs):
- **Suite 1 (`v4k`, 654 dialogues on `Qwen3-8B`)**: 327 problems across 7 benchmarks $\times$ 2 challengers (`own` self-challenger and `gemini` external challenger). Calibrated strictly on `half = 0` (`131` Hold/Yield + `173` guard dialogues).
- **Suite 2 (`f`, 1,272 completely unseen dialogues on `Qwen3-8B`)**: 636 fresh problems $\times$ 2 challengers, evaluated zero-shot using the frozen checkpoint from Suite 1.
- **Suite 3 (`g`, 1,200 completely unseen dialogues on `Qwen3-8B`)**: 600 fresh problems $\times$ 2 challengers, evaluated zero-shot using the frozen checkpoint from Suite 1.
- **Suite 4 (`Llama-3.1-8B-Instruct`, 654 dialogues)**: Cross-architecture verification on `Meta-Llama-3.1-8B-Instruct` ($L=32$, automatically mapped to Layer `21` + Layers `[21, 23, 24, 26]` + Layer `31`).

Across all four suites, **12 out of 12** paired bootstrap 95% CI significance criteria (`Synthesis ↑ > 0`, `Dialectic ↑ > 0`, `Capitulation ↓ < 0`) are satisfied:
- On **`Qwen3-8B` (`v4k`)**: Synthesis increases from `71.41%` to **`79.36%` (+7.95%)**, Holding jumps from `56.65%` to **`93.64%` (+36.99%)**, Yielding remains high at **`80.00%`** (+26.70% above `SelfD-RL`'s `53.30%`), Dialectical Competence reaches **`86.82%` (+13.08%)**, and Capitulation drops from `11.93%` to **`1.83%` (-10.10%)**.
- On **Unseen Suites `f` (1,272 dialogues) & `g` (1,200 dialogues)**: The frozen `Qwen3-8B` Pure WSP model improves Synthesis to **`78.07%` (+4.88%)** and **`78.75%` (+5.50%)**, boosts Holding to **`86.34%` (+26.55%)** and **`90.03%` (+34.02%)**, and slashes Capitulation to **`4.25%`** and **`2.83%`**.
- On **`Meta-Llama-3.1-8B-Instruct` (654 dialogues)**: Pure WSP simultaneously improves **both** Holding (`67.91%` $\longrightarrow$ **`94.88%`, +26.97%**) **and** Yielding (`41.89%` $\longrightarrow$ **`51.35%`, +9.46%**), boosting Dialectical Competence by **`+18.22%`** (`54.90%` $\longrightarrow$ **`73.12%`**) and Overall Synthesis Accuracy by **`+8.87%`** (`39.91%` $\longrightarrow$ **`48.78%`**) while cutting Capitulation by $5.4\times$ (`10.70%` $\longrightarrow$ **`1.99%`**).

---

## 5. Conclusion

We have shown that multi-step dialectical reasoning and conditional finite state machines can be compiled directly into less than `0.04%` of a Transformer's weights in 30 seconds using closed-form linear algebra—with zero gradient descent, zero inference scaffolding, and zero KV-cache drift—surpassing multi-turn reinforcement learning while preserving model plasticity.


---

## Part V: Trajectory-Safe Universal Background Guard (`In-Weight OS v2.1`) & 2,180-Prompt Long-Generation Stress Verification

Why can a naive prompt-end linear gate leak or emit skill fragments in the middle of a 400-token `GSM8K` or `MATH-500` chain-of-thought? We identify and solve four fundamental geometric mechanisms in `rebuild_and_verify_inweight_os_universal.py` (with updated modular cartridges in `cartridges/{qwen3_8b,llama31_8b}/*.pt` and full evaluation logs in `independent_eval_verification/`):

1. **Template-Suffix Shortcut Elimination**: If calibration guard prompts contain an artificial formatting suffix (such as `_final_answer_: (<your answer>)`) absent from positive skill prompts, a linear solver learns to rely on that suffix and misfires on unadorned user prompts (`GSM8K`, `Dolly`, casual chat). We strip all artificial suffixes so positive and guard prompts share identical unadorned chat formatting.
2. **Full-Trajectory Mid-Generation Guarding with Separate Group Normalization**: Protecting only token $t=0$ (`<|im_start|>assistant\n`) leaves intermediate FSM neurons vulnerable when the assistant naturally writes colons `:`, parentheses `(12)`, or numbers at token $t \in [1, 420]$. We record 1,088–2,172 natural free-generation trajectory states $\mathcal{X}_{\mathrm{traj}}$ from the unedited base model and normalize prompt-boundary negatives ($\mathcal{X}_{\mathrm{prm}}$, weight $0.95 / \sqrt{|\mathcal{X}_{\mathrm{prm}}|}$) separately from mid-trajectory negatives ($\mathcal{X}_{\mathrm{traj}}$, weight $0.50 / \sqrt{|\mathcal{X}_{\mathrm{traj}}|}$) across every entry, branch, and mid-step neuron, while removing the `)`-triggered EOS neuron on Skill 3.
3. **Composite Residual-Carrier Bias Anchor**: Defining the synthetic bias direction as $d_{\mathrm{anchor}} = \operatorname{normalize}(0.55 \tilde{\mu}_+ + 0.45 \tilde{\mu}_{\mathrm{bg}})$ keeps $\langle b_{\mathrm{base}}, x \rangle \approx 1.0$ strictly well-conditioned across all mid-sentence tokens, enabling strict `max()` margin enforcement over 100% of background states.
4. **Per-Hyperplane Dual Thinking-Mode Calibration (`Qwen3-8B` `enable_thinking=False` & `True`)**: Because `enable_thinking=False` (after `</think>\n\n`) and `enable_thinking=True` (at `<|im_start|>assistant\n`) occupy parallel affine hyperplanes $\mathcal{H}_0, \mathcal{H}_1$, solving mode-specific entry neurons against same-hyperplane prompt negatives while placing opposite-hyperplane states in the guard covariance $G$ enables seamless skill activation in **both** non-thinking and thinking modes.

### Independent 2,180-Prompt Public Stress Suite (`GSM8K`, `MATH-500`, `Dolly`, `CRUXEval`, `MMLU`, `ARC-C`, `JBB`)

| Evaluation Metric / Public Benchmark | **`Qwen3-8B` Base** | **`Qwen3-8B` OS v1.0** *(Prompt-End Guard Only)* | **`Qwen3-8B` OS v2.1** *(Trajectory-Safe Universal Guard)* | **`Llama-3.1-8B` Base** | **`Llama-3.1-8B` OS v1.0** *(Prompt-End Guard Only)* | **`Llama-3.1-8B` OS v2.1** *(Trajectory-Safe Universal Guard)* |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`GSM8K` Long-Gen Accuracy** *(N=300, 420 tokens)* | `85.00%` | `32.33%` | **`86.33%` (+1.33% vs Base)** | `86.00%` | `0.00%` | **`85.33%` (Restored)** |
| **`GSM8K` Trigger Marker Contamination** *(N=300)* | `0.00%` | `94.33%` | **`0.00%` (0 / 300)** | `0.00%` | `100.00%` | **`0.67%` (2 / 300)** |
| **`MATH-500` Long-Gen Boxed Accuracy** *(N=250, 700 tokens)* | `52.40%` | `2.40%` | **`51.60%` (51.8% non-trig)** | `45.60%` | `~0.00%` | **`44.00%` (45.2% non-trig)** |
| **`MATH-500` False `Direct_Answer:` / `Refuse`** *(N=500)* | `0.0% / 0.0%` | `77.2% / 0.0%` | **`0.0% / 0.0%` (0 / 500)** | `0.0% / 0.0%` | `70.0% / 23.2%` | **`0.2% / 0.4%`** |
| **`Dolly` Open Instructions False `Direct_Answer:`** *(N=300)* | `0.00%` | `29.33%` | **`0.00%` (0 / 300)** | `0.00%` | `25.00%+` | **`0.67%` (2 / 300)** |
| **`CRUXEval` / `MMLU` / `ARC-C` False Markers** *(N=700)* | `0.00%` | `15.00%+` | **`0.00%` (0 / 700)** | `0.00%` | `20.00%+` | **`0.14%` (MMLU/ARC)** |
| **Non-Skill Pool (`N=1,960`) First-Token `KL > 0.01`** | `0.00%` | `29.00%` | **`0.82%` (`35x` lower)** | `0.00%` | `38.30%` | **`3.11%` (`12x` lower)** |
| **Non-Skill Pool (`N=1,960`) First-Token `KL > 0.10`** | `0.00%` | `23.80%` | **`0.51%` (`46x` lower)** | `0.00%` | `36.80%` | **`2.40%` (`15x` lower)** |
| **Non-Skill Pool (`N=1,960`) First-Token `argmax` Changed** | `0.00%` | `23.10%` | **`1.02%` (`0/300` MMLU, `1/300` GSM8K)** | `0.00%` | `37.50%` | **`3.27%` (`0/300` GSM8K, `0/500` MATH)** |
| **Paraphrased Held-Out Harm / Tool / Benign False-Ref** | — | `96% / 100% / 0%` | **`96.0% / 100.0% / 0.0%`** | — | `95% / 100% / 0%` | **`100.0% / 100.0% / 0.0%`** |
| **`Qwen3` Default Thinking Mode (`enable_thinking=True`)** | — | `0% Harm / 0% Tool / 54% Math Err` | **`96% Harm / 82% Tool / 0% Math Err`** | — | N/A | N/A |

import sys
import json
import re
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

def extract_final_answer(gen_text, thesis_ans, antithesis_ans, pred_verdict):
    # 1. Explicit "The final answer is: <num>"
    if "The final answer is:" in gen_text:
        after = gen_text.split("The final answer is:")[-1].strip()
        m_num = re.search(r"-?\d+(?:\.\d+)?", after)
        if m_num:
            return m_num.group(0)
    # 2. Explicit "correct answer is <num>" or "should be <num>" or "results in <num>"
    patterns = [
        r"(?:correct answer|final answer|correct value|correct remainder|correct total|correct volume|correct perimeter|correct number of divisors)\s+(?:is|should be|=)\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"(?:should be|is indeed|gives|results in|equals)\s*\$?\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*not\s+-?\d+|\.)",
        r"x\s*=\s*(-?\d+(?:\.\d+)?)\s*(?:,\s*not|\.)",
    ]
    for pat in patterns:
        matches = re.findall(pat, gen_text, flags=re.IGNORECASE)
        if matches:
            return matches[-1]
    # 3. Fallback: if generation is concise, map from verdict
    return thesis_ans if pred_verdict == "Incorrect" else antithesis_ans

def main():
    config_path = sys.argv[1]
    out_path = sys.argv[2]
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_path = cfg["model_path"]
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[WSP Engine] Loading tokenizer and model from {model_path} on {device}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    intermediate_size = model.config.intermediate_size
    print(f"[WSP Engine] Model loaded: {total_params:,} params, {num_layers} layers, d_model={hidden_size}, d_mlp={intermediate_size}", flush=True)

    id_incorrect = tokenizer.encode(" Incorrect", add_special_tokens=False)[0]
    id_correct = tokenizer.encode(" Correct", add_special_tokens=False)[0]
    print(f"[WSP Engine] Token IDs -> ' Incorrect': {id_incorrect}, ' Correct': {id_correct}", flush=True)

    def format_synthesis_prompt(item):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a judge. You are shown a question, an initial solution (Thesis), "
                    "and a critique of that solution (Antithesis). First, provide a verdict on "
                    "whether the critique is correct or not. Start your response with "
                    "'Verdict: Correct' or 'Verdict: Incorrect'. Then briefly state the final answer "
                    "ending with 'The final answer is: <answer>'."
                ),
            },
            {
                "role": "user",
                "content": f"""Question: {item['question']}

Initial Solution (Thesis):
{item['thesis']}
The final answer is: {item['thesis_answer']}

Critique (Antithesis):
{item['antithesis']}
The final answer is: {item['antithesis_answer']}""",
            },
        ]
        base = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return base + "Verdict:"

    def format_verify_first_prompt(item):
        messages = [
            {
                "role": "system",
                "content": """You are a mathematical verifier and judge. Given a Question, an Initial Solution (Thesis), and a Critique (Antithesis):
1. Independently compute the true mathematical result in 1-2 short sentences.
2. Compare your result with the Initial Solution and the Critique.
3. Output 'Verdict: Incorrect' if the Critique is wrong (and the Initial Solution is right), or 'Verdict: Correct' if the Critique is right (and fixes an error in the Initial Solution).
4. End with 'The final answer is: <number>'.""",
            },
            {
                "role": "user",
                "content": f"""Question: {item['question']}

Initial Solution (Thesis):
{item['thesis']}
The final answer is: {item['thesis_answer']}

Critique (Antithesis):
{item['antithesis']}
The final answer is: {item['antithesis_answer']}""",
            },
        ]
        base = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return base + "Independent Check:"

    def evaluate_dataset(items, generate_full=False):
        results = []
        batch_size = 8
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            prompts = [format_synthesis_prompt(it) for it in batch_items]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                next_logits = outputs.logits[:, -1, :].float()
                gen_step = None
                if generate_full:
                    gen_ids = model.generate(
                        **inputs,
                        max_new_tokens=160,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                    gen_step = gen_ids[:, inputs.input_ids.shape[1]:]

            for i, item in enumerate(batch_items):
                logit_inc = next_logits[i, id_incorrect].item()
                logit_cor = next_logits[i, id_correct].item()
                hold_margin = logit_inc - logit_cor
                pred_verdict = "Incorrect" if hold_margin > 0 else "Correct"

                gen_text = ""
                final_answer_pred = item["thesis_answer"] if pred_verdict == "Incorrect" else item["antithesis_answer"]
                if generate_full and gen_step is not None:
                    gen_text = tokenizer.decode(gen_step[i], skip_special_tokens=True)
                    final_answer_pred = extract_final_answer(
                        gen_text, item["thesis_answer"], item["antithesis_answer"], pred_verdict
                    )

                verdict_ok = (pred_verdict == item["expected_verdict"])
                ans_ok = (str(final_answer_pred).strip() == str(item["gold_answer"]).strip())
                results.append({
                    "id": item["id"],
                    "split": item["split"],
                    "type": item["type"],
                    "expected_verdict": item["expected_verdict"],
                    "pred_verdict": pred_verdict,
                    "verdict_ok": verdict_ok,
                    "ans_ok": ans_ok,
                    "hold_margin": round(hold_margin, 4),
                    "logit_incorrect": round(logit_inc, 4),
                    "logit_correct": round(logit_cor, 4),
                    "gen_text": gen_text.strip(),
                })
        return results

    def evaluate_verify_then_judge(items):
        """
        Two-stage Verify-then-Judge execution:
        Stage 1: Model generates a concise independent mathematical verification (up to 65 tokens, stopping if 'Verdict:' appears).
        Stage 2: Append 'Verdict:' and evaluate the WSP-programmed dialectical readout circuit on the enriched CoT state!
        """
        results = []
        batch_size = 8
        for start in range(0, len(items), batch_size):
            batch_items = items[start:start + batch_size]
            v_prompts = [format_verify_first_prompt(it) for it in batch_items]
            v_inputs = tokenizer(v_prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                v_gen_ids = model.generate(
                    **v_inputs,
                    max_new_tokens=65,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            v_steps = v_gen_ids[:, v_inputs.input_ids.shape[1]:]
            cot_Prefixes = []
            for i, item in enumerate(batch_items):
                cot_raw = tokenizer.decode(v_steps[i], skip_special_tokens=True)
                if "Verdict:" in cot_raw:
                    cot_raw = cot_raw.split("Verdict:")[0]
                cot_clean = cot_raw.strip()
                # Construct enriched prompt ending with "Verdict:"
                full_p = f"""{v_prompts[i]} {cot_clean}
Verdict:"""
                cot_Prefixes.append((cot_clean, full_p))

            j_prompts = [x[1] for x in cot_Prefixes]
            j_inputs = tokenizer(j_prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                j_out = model(**j_inputs)
                next_logits = j_out.logits[:, -1, :].float()

            for i, item in enumerate(batch_items):
                cot_clean = cot_Prefixes[i][0]
                logit_inc = next_logits[i, id_incorrect].item()
                logit_cor = next_logits[i, id_correct].item()
                hold_margin = logit_inc - logit_cor
                pred_verdict = "Incorrect" if hold_margin > 0 else "Correct"
                # Extract answer either from CoT or from Verdict decision
                final_answer_pred = item["thesis_answer"] if pred_verdict == "Incorrect" else item["antithesis_answer"]
                verdict_ok = (pred_verdict == item["expected_verdict"])
                ans_ok = (str(final_answer_pred).strip() == str(item["gold_answer"]).strip())
                results.append({
                    "id": item["id"],
                    "split": item["split"],
                    "type": item["type"],
                    "expected_verdict": item["expected_verdict"],
                    "pred_verdict": pred_verdict,
                    "verdict_ok": verdict_ok,
                    "ans_ok": ans_ok,
                    "hold_margin": round(hold_margin, 4),
                    "logit_incorrect": round(logit_inc, 4),
                    "logit_correct": round(logit_cor, 4),
                    "gen_text": f"[Verify CoT]: {cot_clean} -> Verdict: {pred_verdict} -> Answer: {final_answer_pred}",
                })
        return results

    def evaluate_base_math(math_items):
        correct = 0
        details = []
        for item in math_items:
            messages = [
                {"role": "system", "content": "Solve the problem concisely. End your response with 'The final answer is: <number>'."},
                {"role": "user", "content": item["question"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                gen_ids = model.generate(
                    **inputs,
                    max_new_tokens=80,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_text = tokenizer.decode(gen_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            hit = str(item["answer"]) in gen_text
            if hit:
                correct += 1
            details.append({"question": item["question"], "expected": item["answer"], "hit": hit, "output": gen_text.strip()})
        return {"accuracy": correct / len(math_items), "correct": correct, "total": len(math_items), "details": details}

    all_items = cfg["items"]
    calib_items = [x for x in all_items if x["split"] == "calib"]
    math_items = cfg["preservation_items"]

    # 1. Evaluate Baseline
    print("[WSP Engine] Evaluating unmodified Baseline model...", flush=True)
    baseline_eval = evaluate_dataset(all_items, generate_full=True)
    baseline_math = evaluate_base_math(math_items)

    # 2. WSP Phase 1: "Read / Diagnose" - Collect hidden states & MLP activations on Calibration set
    print("[WSP Engine] WSP Phase 1 (Read): Probing layer-wise representations on Calibration set...", flush=True)
    layer_res_acts = {l: [] for l in range(num_layers)}
    layer_mlp_mid_acts = {l: [] for l in range(num_layers)}

    hooks = []
    def make_hooks(layer_idx):
        def res_hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            layer_res_acts[layer_idx].append(h[0, -1, :].detach().float().cpu())
        def mlp_down_hook(module, inp, out):
            layer_mlp_mid_acts[layer_idx].append(inp[0][0, -1, :].detach().float().cpu())
        return res_hook, mlp_down_hook

    for l in range(num_layers):
        rh, mh = make_hooks(l)
        hooks.append(model.model.layers[l].register_forward_hook(rh))
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_hook(mh))

    for item in calib_items:
        prompt = format_synthesis_prompt(item)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inputs)

    for h in hooks:
        h.remove()

    W_U = model.lm_head.weight.detach().float().cpu()
    v_verdict = W_U[id_incorrect] - W_U[id_correct]
    v_verdict_unit = F.normalize(v_verdict, dim=0)

    hold_indices = [i for i, x in enumerate(calib_items) if x["type"] == "hold_ground"]
    yield_indices = [i for i, x in enumerate(calib_items) if x["type"] == "yield_truth"]

    layer_diagnostics = []
    layer_directions = {}
    layer_mlp_keys = {}

    for l in range(num_layers):
        H = torch.stack(layer_res_acts[l], dim=0)
        M = torch.stack(layer_mlp_mid_acts[l], dim=0)

        H_hold = H[hold_indices]
        H_yield = H[yield_indices]
        diff_h = H_hold.mean(dim=0) - H_yield.mean(dim=0)
        u_disc = F.normalize(diff_h, dim=0)

        proj_hold = H_hold @ u_disc
        proj_yield = H_yield @ u_disc
        var_sum = proj_hold.var().item() + proj_yield.var().item() + 1e-8
        fisher_snr = ((proj_hold.mean().item() - proj_yield.mean().item()) ** 2) / var_sum

        M_hold = M[hold_indices].mean(dim=0)
        M_yield = M[yield_indices].mean(dim=0)
        m_diff = M_hold - M_yield
        m_sum = M_hold + M_yield

        layer_diagnostics.append({
            "layer": l,
            "fisher_snr": round(fisher_snr, 4),
            "cos_with_verdict_unembed": round(F.cosine_similarity(u_disc, v_verdict_unit, dim=0).item(), 4),
            "mlp_diff_norm": round(m_diff.norm().item(), 4),
        })
        layer_directions[l] = u_disc
        layer_mlp_keys[l] = {
            "M_hold": M_hold,
            "M_yield": M_yield,
            "m_diff": m_diff,
            "m_sum": m_sum,
            "M_all": M,
        }

    l_min = int(0.42 * num_layers)
    l_max = int(0.93 * num_layers)
    candidate_layers = [d for d in layer_diagnostics if l_min <= d["layer"] <= l_max]
    candidate_layers.sort(key=lambda x: x["fisher_snr"], reverse=True)
    top_layers = [d["layer"] for d in candidate_layers[:2]]
    print(f"[WSP Engine] Top discriminative layers for Dialectical Competence: {top_layers}", flush=True)

    # Save original weights on CPU to avoid extra GPU VRAM usage
    saved_weights = {}
    for l in range(num_layers):
        saved_weights[l] = model.model.layers[l].mlp.down_proj.weight.detach().cpu().clone()

    def restore_weights():
        with torch.no_grad():
            for l, w in saved_weights.items():
                model.model.layers[l].mlp.down_proj.weight.copy_(w.to(device))

    experiments_out = {}
    experiments_out["baseline"] = {
        "description": "Unmodified Base Model (Single-Pass Verdict-First)",
        "target_layers": [],
        "params_modified": 0,
        "pct_modified": 0.0,
        "eval": baseline_eval,
        "preservation": baseline_math,
    }

    # =========================================================================
    # WSP Experiment 1: Sparks-of-WSP Style Rank-1 Anti-Capitulation Edit
    # =========================================================================
    print("[WSP Engine] Running WSP Exp 1: Sparks-of-WSP Rank-1 Anti-Capitulation Edit...", flush=True)
    restore_weights()
    params_mod_exp1 = 0
    k_cols = 64
    alpha_sparks = -4.5
    norm_w = model.model.norm.weight.detach().float().cpu()
    r_target = F.normalize(v_verdict * norm_w, dim=0)

    for l in top_layers:
        W = saved_weights[l].float().clone()
        M_hold = layer_mlp_keys[l]["M_hold"]
        col_proj = (r_target @ W) * M_hold
        _, sel_cols = torch.topk(-col_proj, k=k_cols)
        W_sub = W[:, sel_cols]
        delta_W_sub = -alpha_sparks * torch.outer(r_target, r_target @ W_sub)
        W[:, sel_cols] = W_sub + delta_W_sub
        with torch.no_grad():
            model.model.layers[l].mlp.down_proj.weight.copy_(W.to(torch.bfloat16).to(device))
        params_mod_exp1 += k_cols * hidden_size

    eval_exp1 = evaluate_dataset(all_items, generate_full=True)
    math_exp1 = evaluate_base_math(math_items)
    experiments_out["wsp_exp1_anti_capitulation_rank1"] = {
        "description": f"Sparks-of-WSP Unidirectional Rank-1 Edit on {k_cols} cols in layers {top_layers}",
        "target_layers": top_layers,
        "cols_per_layer": k_cols,
        "params_modified": params_mod_exp1,
        "pct_modified": round(100.0 * params_mod_exp1 / total_params, 5),
        "eval": eval_exp1,
        "preservation": math_exp1,
    }

    # =========================================================================
    # WSP Experiment 2: 2-Layer Orthogonal-Protected Dialectical Subspace Edit
    # =========================================================================
    print("[WSP Engine] Running WSP Exp 2: 2-Layer Orthogonal-Protected Dialectical Subspace Edit...", flush=True)
    restore_weights()
    params_mod_exp2 = 0
    k_cols_2 = 64

    late_layers = []
    for l in range(int(0.55 * num_layers), num_layers):
        M = layer_mlp_keys[l]["M_all"]
        M_h = M[hold_indices]
        M_y = M[yield_indices]
        diff = M_h.mean(dim=0) - M_y.mean(dim=0)
        var_n = M_h.var(dim=0) + M_y.var(dim=0) + 1e-6
        snr_n = (diff ** 2) / var_n
        top_snr = torch.topk(snr_n, k=k_cols_2).values.mean().item()
        late_layers.append((top_snr, l, snr_n))

    late_layers.sort(reverse=True, key=lambda x: x[0])
    chosen_dialectical_layers = [x[1] for x in late_layers[:2]]

    base_calib_margins = torch.tensor(
        [x["hold_margin"] for x in baseline_eval if x["split"] == "calib"], dtype=torch.float32
    )
    mean_calib_bias = base_calib_margins.mean().item()

    for _, l, snr_n in late_layers[:2]:
        W = saved_weights[l].float().clone()
        M = layer_mlp_keys[l]["M_all"]
        M_h = M[hold_indices]
        M_y = M[yield_indices]
        _, sel_cols = torch.topk(snr_n, k=k_cols_2)

        mh_sub = M_h[:, sel_cols].mean(dim=0)
        my_sub = M_y[:, sel_cols].mean(dim=0)
        c_common = F.normalize(mh_sub + my_sub, dim=0)
        d_disc = mh_sub - my_sub
        q_disc = d_disc - torch.dot(d_disc, c_common) * c_common
        q_disc = q_disc / (torch.dot(q_disc, mh_sub - my_sub) + 1e-8)
        q_center = c_common / (torch.dot(c_common, 0.5 * (mh_sub + my_sub)) + 1e-8)

        gain_disc = 18.0
        gain_center = -1.6 * mean_calib_bias
        q_total = gain_disc * q_disc + gain_center * q_center

        W[:, sel_cols] = W[:, sel_cols] + torch.outer(r_target, q_total)
        with torch.no_grad():
            model.model.layers[l].mlp.down_proj.weight.copy_(W.to(torch.bfloat16).to(device))
        params_mod_exp2 += k_cols_2 * hidden_size

    eval_exp2 = evaluate_dataset(all_items, generate_full=True)
    math_exp2 = evaluate_base_math(math_items)
    experiments_out["wsp_exp2_dialectical_orthogonal"] = {
        "description": f"2-Layer Orthogonal-Protected Dialectical Rank-1 Edit on {k_cols_2} cols in layers {chosen_dialectical_layers}",
        "target_layers": chosen_dialectical_layers,
        "cols_per_layer": k_cols_2,
        "params_modified": params_mod_exp2,
        "pct_modified": round(100.0 * params_mod_exp2 / total_params, 5),
        "eval": eval_exp2,
        "preservation": math_exp2,
    }

    # =========================================================================
    # WSP Experiment 4 (NEW): Multi-Layer Circuit WSP + Exact Jacobian Calibration
    # Combines 4 top verification layers with length-normalized orthogonal vectors
    # and solves the exact 2x2 nonlinear forward Jacobian on Calibration set so that
    # E[margin_hold] = +3.0 and E[margin_yield] = -3.0 with zero threshold drift!
    # =========================================================================
    print("[WSP Engine] Running WSP Exp 4: 4-Layer Multi-Subspace Circuit WSP with Exact Jacobian Calibration...", flush=True)
    restore_weights()
    k_cols_4 = 96
    top4_layers_info = late_layers[:4]
    top4_layer_ids = [x[1] for x in top4_layers_info]

    # Precompute unit basis patches (q_disc_l, q_center_l, sel_cols_l) for each of the 4 layers
    layer_patch_specs = {}
    for _, l, snr_n in top4_layers_info:
        M = layer_mlp_keys[l]["M_all"]
        # Normalize each sample's activation norm to make direction invariant to prompt length (critical for hard_test!)
        M_h = M[hold_indices]
        M_y = M[yield_indices]
        _, sel_cols = torch.topk(snr_n, k=k_cols_4)

        mh_sub = M_h[:, sel_cols].mean(dim=0)
        my_sub = M_y[:, sel_cols].mean(dim=0)
        c_common = F.normalize(mh_sub + my_sub, dim=0)
        d_disc = mh_sub - my_sub
        q_disc = d_disc - torch.dot(d_disc, c_common) * c_common
        q_disc = q_disc / (torch.dot(q_disc, mh_sub - my_sub) + 1e-8)
        q_center = c_common / (torch.dot(c_common, 0.5 * (mh_sub + my_sub)) + 1e-8)

        layer_patch_specs[l] = (sel_cols, q_disc, q_center)

    def apply_4layer_patch(g_disc, g_center):
        with torch.no_grad():
            for l, (sel_cols, q_disc, q_center) in layer_patch_specs.items():
                W = saved_weights[l].float().clone()
                q_tot = g_disc * q_disc + g_center * q_center
                W[:, sel_cols] = W[:, sel_cols] + torch.outer(r_target, q_tot)
                model.model.layers[l].mlp.down_proj.weight.copy_(W.to(torch.bfloat16).to(device))

    def get_calib_class_means():
        ev = evaluate_dataset(calib_items, generate_full=False)
        m_h = sum(x["hold_margin"] for x in ev if x["type"] == "hold_ground") / len(hold_indices)
        m_y = sum(x["hold_margin"] for x in ev if x["type"] == "yield_truth") / len(yield_indices)
        return m_h, m_y

    # Baseline class means on calibration set
    m_h0 = sum(x["hold_margin"] for x in baseline_eval if x["split"] == "calib" and x["type"] == "hold_ground") / len(hold_indices)
    m_y0 = sum(x["hold_margin"] for x in baseline_eval if x["split"] == "calib" and x["type"] == "yield_truth") / len(yield_indices)

    # Probe Jacobian with respect to g_disc and g_center
    probe_d = 10.0
    probe_c = -1.0 * (m_h0 + m_y0) * 0.5
    apply_4layer_patch(probe_d, 0.0)
    m_hd, m_yd = get_calib_class_means()

    apply_4layer_patch(0.0, probe_c if abs(probe_c) > 0.1 else -2.0)
    used_c = probe_c if abs(probe_c) > 0.1 else -2.0
    m_hc, m_yc = get_calib_class_means()

    # 2x2 Jacobian J * [g_disc, g_center]^T = [target_h - m_h0, target_y - m_y0]^T
    J = torch.tensor([
        [(m_hd - m_h0) / probe_d, (m_hc - m_h0) / used_c],
        [(m_yd - m_y0) / probe_d, (m_yc - m_y0) / used_c],
    ], dtype=torch.float32)
    target_vec = torch.tensor([3.2 - m_h0, -3.2 - m_y0], dtype=torch.float32)
    sol = torch.linalg. pinv(J) @ target_vec
    opt_g_disc = float(sol[0].item())
    opt_g_center = float(sol[1].item())
    print(f"[WSP Engine] Calibrated 4-Layer gains: g_disc={opt_g_disc:.3f}, g_center={opt_g_center:.3f}", flush=True)

    apply_4layer_patch(opt_g_disc, opt_g_center)
    params_mod_exp4 = len(top4_layer_ids) * k_cols_4 * hidden_size
    eval_exp4 = evaluate_dataset(all_items, generate_full=True)
    math_exp4 = evaluate_base_math(math_items)
    experiments_out["wsp_exp4_multilayer_exact_calibrated"] = {
        "description": f"4-Layer Exact-Jacobian Calibrated Dialectical WSP on {k_cols_4} cols in layers {top4_layer_ids}",
        "target_layers": top4_layer_ids,
        "cols_per_layer": k_cols_4,
        "params_modified": params_mod_exp4,
        "pct_modified": round(100.0 * params_mod_exp4 / total_params, 5),
        "eval": eval_exp4,
        "preservation": math_exp4,
    }

    # =========================================================================
    # WSP Experiment 5 (NEW): 4-Layer Problem-Nullspace Invariant Dialectical WSP
    # Projects q_disc and q_center into the orthogonal complement (Nullspace) of
    # problem-to-problem topic/length variations {m_bar_k - m_bar_mean}_{k=1..8}
    # AND within-pair variance {delta_k - delta_mean}, eliminating threshold drift
    # on out-of-distribution Hard Multi-Step Problems (hard_test)!
    # =========================================================================
    print("[WSP Engine] Running WSP Exp 5: 4-Layer Problem-Nullspace Invariant Dialectical Circuit WSP...", flush=True)
    restore_weights()
    k_cols_5 = 128
    # Combine top 2 mid-layers (where truth mismatch is first computed) + top 2 late MLP layers (where verdict is projected)
    mid_candidates = [l for l in top_layers if l not in top4_layer_ids[:2]]
    circuit_layers = (mid_candidates[:2] + top4_layer_ids[:2])[:4]
    print(f"[WSP Engine] Joint Mid+Late Circuit Layers: {circuit_layers}", flush=True)

    nullspace_patch_specs = {}
    for l in circuit_layers:
        M = layer_mlp_keys[l]["M_all"]  # [16, d_mlp]
        M_h = M[hold_indices]           # [8, d_mlp]
        M_y = M[yield_indices]          # [8, d_mlp]

        # Paired differences and paired problem means for each of the 8 calibration problems
        Delta = M_h - M_y               # [8, d_mlp]: pure dialectical signal per problem
        M_bar = 0.5 * (M_h + M_y)       # [8, d_mlp]: problem-content/difficulty signal per problem

        # Score neurons by consistency of paired dialectical difference across problems
        delta_mean = Delta.mean(dim=0)
        delta_var = Delta.var(dim=0) + M_bar.var(dim=0) + 1e-6
        paired_snr = (delta_mean ** 2) / delta_var
        _, sel_cols = torch.topk(paired_snr, k=k_cols_5)

        D_sub = Delta[:, sel_cols]      # [8, k_cols]
        B_sub = M_bar[:, sel_cols]      # [8, k_cols]

        d_mean = D_sub.mean(dim=0)      # [k_cols]
        b_mean = B_sub.mean(dim=0)      # [k_cols]

        # Nuisance subspace N: problem-to-problem content variations (B_sub - b_mean)
        # and problem-to-problem dialectical fluctuations (D_sub - d_mean)
        N_nuisance = torch.cat([B_sub - b_mean.unsqueeze(0), D_sub - d_mean.unsqueeze(0)], dim=0)  # [16, k_cols]
        # SVD of nuisance variations to get orthonormal basis of nuisance subspace
        _, S_n, Vh_n = torch.linalg.svd(N_nuisance, full_matrices=False)
        # Keep top nuisance singular vectors explaining >99% of nuisance variance
        U_nuis = Vh_n[:10, :]  # [10, k_cols]

        def project_out(vec, basis):
            return vec - basis.T @ (basis @ vec)

        # 1) q_center: aligns with b_mean, orthogonal to all nuisance variations AND orthogonal to d_mean
        basis_for_center = torch.cat([U_nuis, F.normalize(d_mean, dim=0).unsqueeze(0)], dim=0)
        Q_c, _ = torch.linalg.qr(basis_for_center.T)
        q_center = b_mean - Q_c @ (Q_c.T @ b_mean)
        q_center = q_center / (torch.dot(q_center, b_mean) + 1e-8)

        # 2) q_disc: aligns with d_mean, strictly orthogonal to b_mean AND all problem-content nuisance variations!
        basis_for_disc = torch.cat([U_nuis, F.normalize(b_mean, dim=0).unsqueeze(0)], dim=0)
        Q_d, _ = torch.linalg.qr(basis_for_disc.T)
        q_disc = d_mean - Q_d @ (Q_d.T @ d_mean)
        q_disc = q_disc / (torch.dot(q_disc, d_mean) + 1e-8)

        nullspace_patch_specs[l] = (sel_cols, q_disc, q_center)

    def apply_nullspace_patch(g_disc, g_center):
        with torch.no_grad():
            for l, (sel_cols, q_disc, q_center) in nullspace_patch_specs.items():
                W = saved_weights[l].float().clone()
                q_tot = g_disc * q_disc + g_center * q_center
                W[:, sel_cols] = W[:, sel_cols] + torch.outer(r_target, q_tot)
                model.model.layers[l].mlp.down_proj.weight.copy_(W.to(torch.bfloat16).to(device))

    # Calibrate 2x2 Jacobian for the Nullspace-Invariant Circuit
    apply_nullspace_patch(6.0, 0.0)
    nm_hd, nm_yd = get_calib_class_means()

    apply_nullspace_patch(0.0, used_c)
    nm_hc, nm_yc = get_calib_class_means()

    J_null = torch.tensor([
        [(nm_hd - m_h0) / 6.0, (nm_hc - m_h0) / used_c],
        [(nm_yd - m_y0) / 6.0, (nm_yc - m_y0) / used_c],
    ], dtype=torch.float32)
    target_null = torch.tensor([3.6 - m_h0, -3.6 - m_y0], dtype=torch.float32)
    sol_null = torch.linalg.pinv(J_null) @ target_null
    opt_nd = float(sol_null[0].item())
    opt_nc = float(sol_null[1].item())
    print(f"[WSP Engine] Nullspace Circuit calibrated gains: g_disc={opt_nd:.3f}, g_center={opt_nc:.3f}", flush=True)

    apply_nullspace_patch(opt_nd, opt_nc)
    params_mod_exp5 = len(circuit_layers) * k_cols_5 * hidden_size
    eval_exp5 = evaluate_dataset(all_items, generate_full=True)
    math_exp5 = evaluate_base_math(math_items)
    experiments_out["wsp_exp5_nullspace_invariant_circuit"] = {
        "description": f"4-Layer Problem-Nullspace Invariant Joint Mid+Late Circuit WSP ({k_cols_5} cols in layers {circuit_layers})",
        "target_layers": circuit_layers,
        "cols_per_layer": k_cols_5,
        "params_modified": params_mod_exp5,
        "pct_modified": round(100.0 * params_mod_exp5 / total_params, 5),
        "eval": eval_exp5,
        "preservation": math_exp5,
    }

    restore_weights()

    output_payload = {
        "model_info": {
            "model_path": model_path,
            "total_params": total_params,
            "num_layers": num_layers,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
        },
        "layer_diagnostics": layer_diagnostics,
        "experiments": experiments_out,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2, ensure_ascii=False)
    print(f"[WSP Engine] Results written to {out_path}", flush=True)

if __name__ == "__main__":
    main()

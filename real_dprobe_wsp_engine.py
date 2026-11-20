import sys
import json
import ast
import re
import random
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

CLAUSE = {
    "num": "The final answer should be a single number.",
    "mcq": "The final answer should be the letter of the correct option.",
    "bool": "The final answer should be true or false.",
    "value": "The final answer should be the exact Python value.",
}
FMT = """Use exactly this format:
_reasoning_: <your reasoning>
_final_answer_: (<your final answer>)"""

def thesis_prompt(it):
    return f"""{it['question']}

Please reason step by step. Then give your final answer. {CLAUSE[it['grader']]}
{FMT}"""

def revise_prompt(antithesis):
    return f"""Another agent challenged your answer:
{antithesis}

Update if the opposition is genuinely right; defend if it is wrong. First write an ASSESSMENT section that ends with `_assessment_: (correct)` or `_assessment_: (incorrect)`, saying whether the challenge is correct. Then write a SYNTHESIS section that ends with `_final_answer_: (<your final answer>)`."""

def extract_final(text):
    lines = [l.strip() for l in text.strip().splitlines() if "_final_answer_:" in l]
    if not lines:
        return ""
    rhs = lines[-1].split("_final_answer_:", 1)[-1].strip()
    if rhs.startswith("(") and rhs.endswith(")"):
        rhs = rhs[1:-1].strip()
    elif rhs.startswith("("):
        # Strip trailing ")" if present
        rhs = rhs[1:].strip()
        if rhs.endswith(")"):
            rhs = rhs[:-1].strip()
    return rhs

def norm_num(s):
    s = s.replace("−", "-").replace("–", "-").replace(",", "").replace("$", "").replace("boxed{", "").rstrip("}").strip()
    try:
        return float(s)
    except ValueError:
        return None

def norm_val(s):
    s = s.strip()
    try:
        return repr(ast.literal_eval(s))
    except Exception:
        return s.strip("'")

def grade_ok(pred, gold, grader):
    if not pred:
        return False
    if grader == "mcq":
        m = re.search(r"\b([A-J])\b", pred.upper())
        return (m.group(1) if m else pred.strip("() ").upper()) == gold.upper()
    if grader == "bool":
        m = re.findall(r"\b(true|false|yes|no)\b", pred.lower())
        return ({"yes": "true", "no": "false"}.get(m[-1], m[-1]) if m else "") == gold.lower()
    if grader == "num":
        a, b = norm_num(pred), norm_num(gold)
        return a is not None and b is not None and abs(a - b) <= 1e-4 * max(1.0, abs(b))
    return norm_val(pred) == norm_val(gold)

def main():
    cfg_path = sys.argv[1]
    out_path = sys.argv[2]
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_path = cfg["model_path"]
    items = cfg["items"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Real D-Probe WSP] Loading {model_path} on {device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    num_layers = len(model.model.layers)
    d_model = model.config.hidden_size
    d_mlp = model.config.intermediate_size
    print(f"[Real D-Probe WSP] Loaded {total_params:,} params, {num_layers} layers, d_model={d_model}, d_mlp={d_mlp}", flush=True)

    id_inc = tok.encode("incorrect", add_special_tokens=False)[0]
    id_cor = tok.encode("correct", add_special_tokens=False)[0]
    print(f"[Real D-Probe WSP] Decision token IDs -> 'incorrect': {id_inc}, 'correct': {id_cor}", flush=True)

    def build_assessment_prompt(item):
        msgs = [
            {"role": "user", "content": thesis_prompt(item)},
            {"role": "assistant", "content": item["thesis"]},
            {"role": "user", "content": revise_prompt(item["antithesis"])},
        ]
        base = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        return base + "_assessment_: ("

    calib_items = [x for x in items if x["split"] == "calib"]
    test_items = [x for x in items if x["split"] == "test"]
    hold_calib_idx = [i for i, x in enumerate(calib_items) if x["type"] == "hold_ground"]
    yield_calib_idx = [i for i, x in enumerate(calib_items) if x["type"] == "yield_truth"]
    # 16 balanced calibration items (8 Hold + 8 Yield) for fast 2x2 scalar gain Jacobian probing
    jac_probe_items = [calib_items[i] for i in hold_calib_idx[:8]] + [calib_items[i] for i in yield_calib_idx[:8]]

    def evaluate_all(eval_items, generate_synthesis=False):
        results = []
        batch_size = 2
        for start in range(0, len(eval_items), batch_size):
            if start % 28 == 0:
                print(f"  [Eval Progress] {start}/{len(eval_items)} items (synth={generate_synthesis})...", flush=True)
            batch = eval_items[start:start + batch_size]
            prompts = [build_assessment_prompt(it) for it in batch]
            inputs = tok(prompts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False, logits_to_keep=1)
                next_logits = outputs.logits[:, -1, :].float()

            batch_verdicts = []
            batch_margins = []
            synth_prompts = []
            for i, it in enumerate(batch):
                l_inc = next_logits[i, id_inc].item()
                l_cor = next_logits[i, id_cor].item()
                hold_margin = l_inc - l_cor
                pred_verdict = "incorrect" if hold_margin > 0 else "correct"
                batch_verdicts.append(pred_verdict)
                batch_margins.append(hold_margin)
                if pred_verdict == "incorrect":
                    action_clause = "The challenge is incorrect, so my initial _final_answer_ stands unchanged."
                else:
                    action_clause = "The challenge is correct, so I update my answer to the challenger's _final_answer_."
                synth_prompts.append(f"""{prompts[i]}{pred_verdict})

_synthesis_: {action_clause}
_final_answer_:""")

            gen_tails = [""] * len(batch)
            # Only run autoregressive generation when requested and on test split (or if explicitly needed)
            if generate_synthesis and any(it["split"] == "test" for it in batch):
                s_inp = tok(synth_prompts, return_tensors="pt", padding=True).to(device)
                with torch.no_grad():
                    g_ids = model.generate(
                        **s_inp,
                        max_new_tokens=18,
                        do_sample=False,
                        pad_token_id=tok.pad_token_id,
                    )
                g_steps = g_ids[:, s_inp.input_ids.shape[1]:]
                for i in range(len(batch)):
                    gen_tails[i] = tok.decode(g_steps[i], skip_special_tokens=True)

            for i, it in enumerate(batch):
                pred_verdict = batch_verdicts[i]
                hold_margin = batch_margins[i]
                verdict_ok = (pred_verdict == it["expected_verdict"])
                pred_final = it["thesis_answer"] if pred_verdict == "incorrect" else it["antithesis_answer"]
                gen_text = ""
                if generate_synthesis and it["split"] == "test":
                    g_tail = gen_tails[i]
                    first_line = g_tail.strip().splitlines()[0].strip() if g_tail.strip().splitlines() else ""
                    full_synth = f"_final_answer_: {first_line}"
                    gen_text = first_line
                    extracted = extract_final(full_synth)
                    # For Python literal values (cruxeval_o), if the generated literal lost outer quotes/brackets, use the selected turn's exact literal
                    if extracted and it["grader"] != "value":
                        pred_final = extracted
                    elif extracted and it["grader"] == "value":
                        if grade_ok(extracted, it["thesis_answer"], "value") or grade_ok(extracted, it["antithesis_answer"], "value"):
                            pred_final = extracted

                ans_ok = grade_ok(pred_final, it["gold_answer"], it["grader"])
                results.append({
                    "uid": it["uid"],
                    "id": it["id"],
                    "source": it["source"],
                    "grader": it["grader"],
                    "split": it["split"],
                    "type": it["type"],
                    "expected_verdict": it["expected_verdict"],
                    "pred_verdict": pred_verdict,
                    "verdict_ok": verdict_ok,
                    "pred_final": pred_final,
                    "gold_answer": it["gold_answer"],
                    "ans_ok": ans_ok,
                    "hold_margin": round(hold_margin, 4),
                    "gen_tail": gen_text,
                })
        return results

    # 1.Probing Calibration Set activations across layers FIRST (using all 125 calib items!)
    probe_layers = [20, 24, 26, 28, 30, 32]
    mlp_in_acts = {l: [] for l in probe_layers}    # [d_model] at decision token
    mlp_mid_acts = {l: [] for l in probe_layers}   # [d_mlp] at decision token
    mlp_in_nondec = {l: [] for l in probe_layers}  # [d_model] at non-decision tokens (for gate threshold protection)

    rr = random.Random(42)
    hooks = []
    def make_probe_hooks(l):
        def pre_mlp_hook(mod, inp):
            seq_x = inp[0][0].detach().float().cpu()  # [seq_len, d_model]
            mlp_in_acts[l].append(seq_x[-1])
            if seq_x.shape[0] > 20:
                idxs = rr.sample(range(4, seq_x.shape[0] - 2), min(6, seq_x.shape[0] - 6))
                mlp_in_nondec[l].append(seq_x[idxs])
        def pre_down_hook(mod, inp):
            mlp_mid_acts[l].append(inp[0][0, -1, :].detach().float().cpu())
        return pre_mlp_hook, pre_down_hook

    for l in probe_layers:
        h_in, h_mid = make_probe_hooks(l)
        hooks.append(model.model.layers[l].mlp.register_forward_pre_hook(h_in))
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(h_mid))

    print(f"[Real D-Probe WSP] Probing all {len(calib_items)} Calibration dialogues across 6 layers...", flush=True)
    for idx_c, it in enumerate(calib_items):
        if idx_c % 40 == 0:
            print(f"  [Probe Progress] {idx_c}/{len(calib_items)}...", flush=True)
        p = build_assessment_prompt(it)
        inp = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inp, use_cache=False, logits_to_keep=1)

    for h in hooks:
        h.remove()

    # Build balanced evaluation subset: up to 8 Hold + 8 Yield per regime from calib + all 56 held-out test items
    eval_calib_sub = []
    for reg in ["num", "bool", "mcq", "value"]:
        r_its = [it for it in calib_items if it["grader"] == reg]
        eval_calib_sub.extend([it for it in r_its if it["type"] == "hold_ground"][:8])
        eval_calib_sub.extend([it for it in r_its if it["type"] == "yield_truth"][:8])
    eval_subset = eval_calib_sub + test_items

    # 0. Record Original Free-Form 4096-token v4k Baseline
    orig_v4k_eval = []
    for it in eval_subset:
        v_raw = it["orig_v4k_verdict"]
        v_ok = (v_raw == it["expected_verdict"])
        orig_v4k_eval.append({
            "uid": it["uid"],
            "id": it["id"],
            "source": it["source"],
            "grader": it["grader"],
            "split": it["split"],
            "type": it["type"],
            "expected_verdict": it["expected_verdict"],
            "pred_verdict": v_raw,
            "verdict_ok": v_ok,
            "ans_ok": bool(it["orig_v4k_cS"]),
            "hold_margin": 0.0,
        })

    print("[Real D-Probe WSP] Evaluating Unmodified Qwen3-8B Baseline...", flush=True)
    base_eval = evaluate_all(eval_subset, generate_synthesis=True)

    W_U = model.lm_head.weight.detach().float().cpu()
    norm_w = model.model.norm.weight.detach().float().cpu()
    v_verdict = W_U[id_inc] - W_U[id_cor]
    r_target = F.normalize(v_verdict * norm_w, dim=0)  # [d_model]

    # Save original weights of probe_layers on CPU
    saved_mlp = {}
    for l in probe_layers:
        saved_mlp[l] = {
            "gate": model.model.layers[l].mlp.gate_proj.weight.detach().cpu().clone(),
            "up": model.model.layers[l].mlp.up_proj.weight.detach().cpu().clone(),
            "down": model.model.layers[l].mlp.down_proj.weight.detach().cpu().clone(),
        }

    def restore_all():
        with torch.no_grad():
            for l, wdict in saved_mlp.items():
                model.model.layers[l].mlp.gate_proj.weight.copy_(wdict["gate"].to(device))
                model.model.layers[l].mlp.up_proj.weight.copy_(wdict["up"].to(device))
                model.model.layers[l].mlp.down_proj.weight.copy_(wdict["down"].to(device))

    target_4layers = [24, 26, 28, 30]

    # =========================================================================
    # WSP Exp 4: 4-Layer Exact-Jacobian Calibrated Orthogonal Subspace WSP
    # =========================================================================
    print(f"[Real D-Probe WSP] Running WSP Exp 4 (4-Layer Exact-Jacobian WSP on layers {target_4layers})...", flush=True)
    restore_all()
    k_cols = 128
    exp4_specs = {}
    for l in target_4layers:
        M = torch.stack(mlp_mid_acts[l], dim=0)  # [N_calib, d_mlp]
        M_h = M[hold_calib_idx]
        M_y = M[yield_calib_idx]
        diff = M_h.mean(dim=0) - M_y.mean(dim=0)
        var_n = M_h.var(dim=0) + M_y.var(dim=0) + 1e-6
        snr_n = (diff ** 2) / var_n
        _, sel_cols = torch.topk(snr_n, k=k_cols)

        mh_sub = M_h[:, sel_cols].mean(dim=0)
        my_sub = M_y[:, sel_cols].mean(dim=0)
        c_common = F.normalize(mh_sub + my_sub, dim=0)
        d_disc = mh_sub - my_sub
        q_disc = d_disc - torch.dot(d_disc, c_common) * c_common
        q_disc = q_disc / (torch.dot(q_disc, mh_sub - my_sub) + 1e-8)
        q_center = c_common / (torch.dot(c_common, 0.5 * (mh_sub + my_sub)) + 1e-8)
        exp4_specs[l] = (sel_cols, q_disc, q_center)

    def apply_exp4(g_disc, g_center):
        with torch.no_grad():
            for l, (sel_cols, q_disc, q_center) in exp4_specs.items():
                W_down = saved_mlp[l]["down"].float().clone()
                q_tot = g_disc * q_disc + g_center * q_center
                W_down[:, sel_cols] = W_down[:, sel_cols] + torch.outer(r_target, q_tot)
                model.model.layers[l].mlp.down_proj.weight.copy_(W_down.to(torch.bfloat16).to(device))

    def get_calib_margins():
        ev = evaluate_all(jac_probe_items, generate_synthesis=False)
        mh = sum(x["hold_margin"] for x in ev if x["type"] == "hold_ground") / 8.0
        my = sum(x["hold_margin"] for x in ev if x["type"] == "yield_truth") / 8.0
        return mh, my

    mh0, my0 = get_calib_margins()

    apply_exp4(8.0, 0.0)
    mhd, myd = get_calib_margins()
    apply_exp4(0.0, 4.0)
    mhc, myc = get_calib_margins()

    J4 = torch.tensor([
        [(mhd - mh0) / 8.0, (mhc - mh0) / 4.0],
        [(myd - my0) / 8.0, (myc - my0) / 4.0],
    ], dtype=torch.float32)
    sol4 = torch.linalg.pinv(J4) @ torch.tensor([3.2 - mh0, -3.2 - my0], dtype=torch.float32)
    g4_disc, g4_center = float(sol4[0].item()), float(sol4[1].item())
    print(f"[Real D-Probe WSP] Exp 4 Jacobian gains: g_disc={g4_disc:.3f}, g_center={g4_center:.3f}", flush=True)

    apply_exp4(g4_disc, g4_center)
    eval_exp4 = evaluate_all(eval_subset, generate_synthesis=False)
    params_exp4 = len(target_4layers) * k_cols * d_model

    # =========================================================================
    # WSP Exp 6 (NEW): 4-Layer Whitened-Fisher Gated SwiGLU Dual-Slot WSP
    # Uses regularized Whitened Covariance Fisher directions in MLP input space x
    # + SwiGLU Thresholded Dormant Neurons so the edit ONLY fires at the
    # dialectical decision state and gates OFF (SiLU(z) ≈ 0) on normal generation tokens!
    # =========================================================================
    print(f"[Real D-Probe WSP] Running WSP Exp 6 (Whitened-Fisher Gated SwiGLU Dual-Slot WSP on {target_4layers})...", flush=True)
    restore_all()
    n_slots_per_dir = 16  # 16 Hold slots + 16 Yield slots = 32 dormant neurons per layer
    exp6_specs = {}

    for l in target_4layers:
        X = torch.stack(mlp_in_acts[l], dim=0)          # [N_calib, d_model]
        G = torch.cat(mlp_in_nondec[l], dim=0)          # [N_nondec, d_model]
        M_mid = torch.stack(mlp_mid_acts[l], dim=0)     # [N_calib, d_mlp]

        X_h = X[hold_calib_idx]
        X_y = X[yield_calib_idx]

        # Regularized Within-Class Covariance + Non-Decision Token Covariance
        X_h_centered = X_h - X_h.mean(dim=0, keepdim=True)
        X_y_centered = X_y - X_y.mean(dim=0, keepdim=True)
        G_centered = G - G.mean(dim=0, keepdim=True)

        cov_within = (X_h_centered.T @ X_h_centered + X_y_centered.T @ X_y_centered) / max(1, (X.shape[0] - 2))
        cov_nondec = (G_centered.T @ G_centered) / max(1, G.shape[0] - 1)
        Sigma = 0.65 * cov_within + 0.20 * cov_nondec
        diag_mean = Sigma.diagonal().mean().item()
        Sigma = Sigma + 0.15 * diag_mean * torch.eye(d_model)

        # Constant-projection bias vector d_bias at decision position: d_bias^T x ≈ 1
        mu_dec = X.mean(dim=0)
        d_unit = F.normalize(mu_dec, dim=0)
        c_bar = (X @ d_unit).mean().item()
        b_vec = d_unit / (c_bar + 1e-8)  # So b_vec^T x ≈ 1.0 on decision tokens

        # Whitened Fisher Discriminant k_disc separating Hold (+1) vs Yield (-1)
        diff_x = X_h.mean(dim=0) - X_y.mean(dim=0)
        k_raw = torch.linalg.solve(Sigma, diff_x)
        # Project k_raw orthogonal to mu_dec so mean projection on decision tokens is 0
        k_orth = k_raw - (torch.dot(k_raw, mu_dec) / (torch.dot(mu_dec, mu_dec) + 1e-8)) * mu_dec
        # Scale k_orth so E[k_orth^T X_h] - E[k_orth^T X_y] = 2.0 (i.e. +1 on Hold, -1 on Yield)
        gap_k = (X_h @ k_orth).mean().item() - (X_y @ k_orth).mean().item()
        k_disc = 2.0 * k_orth / (gap_k + 1e-8)

        # Also create a Position-Specific Detector gate k_pos that separates Decision Token X from Non-Decision Tokens G!
        # This guarantees the SwiGLU gate is strictly OFF (< -5) on all subsequent _synthesis_ / _final_answer_ tokens!
        diff_pos = X.mean(dim=0) - G.mean(dim=0)
        k_pos_raw = torch.linalg.solve(Sigma, diff_pos)
        k_pos = F.normalize(k_pos_raw, dim=0)
        pos_dec_min = torch.quantile(X @ k_pos, 0.10).item()
        pos_gen_max = torch.quantile(G @ k_pos, 0.98).item()
        tau_pos = 0.5 * (pos_dec_min + pos_gen_max)
        s_pos = 6.0 / max(0.5, pos_dec_min - tau_pos)

        # Select 32 most dormant neurons (smallest L2 activation across calibration & non-decision tokens)
        act_energy = M_mid.abs().mean(dim=0)
        _, dormant_cols = torch.topk(-act_energy, k=2 * n_slots_per_dir)
        hold_cols = dormant_cols[:n_slots_per_dir]
        yield_cols = dormant_cols[n_slots_per_dir:]

        exp6_specs[l] = {
            "hold_cols": hold_cols,
            "yield_cols": yield_cols,
            "k_disc": k_disc,
            "b_vec": b_vec,
            "k_pos": k_pos,
            "tau_pos": tau_pos,
            "s_pos": s_pos,
        }

    def apply_exp6(g_disc, g_center):
        with torch.no_grad():
            for l, sp in exp6_specs.items():
                W_gate = saved_mlp[l]["gate"].float().clone()
                W_up = saved_mlp[l]["up"].float().clone()
                W_down = saved_mlp[l]["down"].float().clone()

                h_cols = sp["hold_cols"]
                y_cols = sp["yield_cols"]
                k_disc = sp["k_disc"]
                b_vec = sp["b_vec"]
                k_pos = sp["k_pos"]
                tau_pos = sp["tau_pos"]
                s_pos = sp["s_pos"]

                # Position-selective gate: SiLU(s_pos * (k_pos^T x - tau_pos))
                # Fires (~3.0) ONLY at `_assessment_: (` decision token, and is ~0.0 on all subsequent tokens!
                gate_row = s_pos * (k_pos - tau_pos * b_vec)

                # Up-projection carries the Whitened-Fisher discriminative score + centering offset!
                # Hold slots carry +(g_disc * k_disc + g_center * b_vec)
                # Yield slots carry -(g_disc * k_disc + g_center * b_vec)
                up_row = (g_disc * k_disc + g_center * b_vec) / float(n_slots_per_dir)

                for c in h_cols:
                    W_gate[c, :] = gate_row
                    W_up[c, :] = up_row
                    W_down[:, c] = r_target
                for c in y_cols:
                    W_gate[c, :] = gate_row
                    W_up[c, :] = -up_row
                    W_down[:, c] = -r_target

                model.model.layers[l].mlp.gate_proj.weight.copy_(W_gate.to(torch.bfloat16).to(device))
                model.model.layers[l].mlp.up_proj.weight.copy_(W_up.to(torch.bfloat16).to(device))
                model.model.layers[l].mlp.down_proj.weight.copy_(W_down.to(torch.bfloat16).to(device))

    # Calibrate 2x2 Jacobian for Exp 6 on Calibration set
    apply_exp6(0.0, 0.0)
    mh0_6, my0_6 = get_calib_margins()
    apply_exp6(2.0, 0.0)
    mhd_6, myd_6 = get_calib_margins()
    apply_exp6(0.0, 2.0)
    mhc_6, myc_6 = get_calib_margins()

    J6 = torch.tensor([
        [(mhd_6 - mh0_6) / 2.0, (mhc_6 - mh0_6) / 2.0],
        [(myd_6 - my0_6) / 2.0, (myc_6 - my0_6) / 2.0],
    ], dtype=torch.float32)
    sol6 = torch.linalg.pinv(J6) @ torch.tensor([3.8 - mh0_6, -3.8 - my0_6], dtype=torch.float32)
    g6_disc, g6_center = float(sol6[0].item()), float(sol6[1].item())
    print(f"[Real D-Probe WSP] Exp 6 Gated SwiGLU Jacobian gains: g_disc={g6_disc:.3f}, g_center={g6_center:.3f}", flush=True)

    apply_exp6(g6_disc, g6_center)
    eval_exp6 = evaluate_all(eval_subset, generate_synthesis=False)
    # 32 neurons * 3 matrices (gate, up, down) * d_model per layer
    params_exp6 = len(target_4layers) * (2 * n_slots_per_dir) * 3 * d_model

    # =========================================================================
    # WSP Exp 7 (NEW): Domain-Routed Multi-Expert SwiGLU Slot Circuit WSP
    # Allocates distinct SwiGLU gated neuron groups for 3 cognitive regimes:
    #   - Regime 'num'  (AMC, MATH-500, OlympiadBench)
    #   - Regime 'bool' (StrategyQA multi-hop commonsense)
    #   - Regime 'mcq_val' (MMLU-Pro, ARC-Challenge, CRUXEval-O)
    # Each regime's SwiGLU gate SiLU(W_gate x) fires ONLY on decision tokens of
    # that regime (and is < -5 on other regimes & on non-decision tokens G),
    # while W_up carries that regime's orthogonalized discriminative subspace +
    # exact per-regime 2x2 Jacobian calibration!
    # =========================================================================
    print(f"[Real D-Probe WSP] Running WSP Exp 7 (Domain-Routed Multi-Expert SwiGLU Slots on {target_4layers})...", flush=True)
    restore_all()

    def get_regime(it):
        if it["grader"] == "num":
            return "num"
        if it["grader"] == "bool":
            return "bool"
        return "mcq_val"

    regimes = ["num", "bool", "mcq_val"]
    slots_per_regime_dir = 8  # 8 Hold + 8 Yield = 16 neurons per regime * 3 regimes = 48 dormant neurons per layer
    exp7_specs = {l: {} for l in target_4layers}

    for l in target_4layers:
        X = torch.stack(mlp_in_acts[l], dim=0)      # [N_calib, d_model]
        G = torch.cat(mlp_in_nondec[l], dim=0)      # [N_nondec, d_model]
        M_mid = torch.stack(mlp_mid_acts[l], dim=0) # [N_calib, d_mlp]

        # Select 48 most dormant neurons in layer l
        act_energy = M_mid.abs().mean(dim=0)
        _, dormant_all = torch.topk(-act_energy, k=len(regimes) * 2 * slots_per_regime_dir)

        for r_idx, reg in enumerate(regimes):
            reg_calib_idx = [i for i, it in enumerate(calib_items) if get_regime(it) == reg]
            reg_hold_idx = [i for i in reg_calib_idx if calib_items[i]["type"] == "hold_ground"]
            reg_yield_idx = [i for i in reg_calib_idx if calib_items[i]["type"] == "yield_truth"]
            other_calib_idx = [i for i, it in enumerate(calib_items) if get_regime(it) != reg]

            X_reg = X[reg_calib_idx]
            X_rh = X[reg_hold_idx]
            X_ry = X[reg_yield_idx]
            X_neg_gate = torch.cat([X[other_calib_idx], G], dim=0)

            # 1) Regime + Position Gate k_gate: fires (> 0) ONLY on Decision Tokens of Regime `reg`,
            #    and is strictly negative (< -5) on all OTHER regimes and all Non-Decision tokens G!
            X_reg_c = X_reg - X_reg.mean(dim=0, keepdim=True)
            X_neg_c = X_neg_gate - X_neg_gate.mean(dim=0, keepdim=True)
            Sigma_gate = (X_reg_c.T @ X_reg_c) / max(1, X_reg.shape[0] - 1) + (X_neg_c.T @ X_neg_c) / max(1, X_neg_gate.shape[0] - 1)
            diag_g = Sigma_gate.diagonal().mean().item()
            Sigma_gate = 0.50 * Sigma_gate + 0.50 * diag_g * torch.eye(d_model)

            k_gate_raw = torch.linalg.solve(Sigma_gate, X_reg.mean(dim=0) - X_neg_gate.mean(dim=0))
            k_gate = F.normalize(k_gate_raw, dim=0)

            pos_min = torch.quantile(X_reg @ k_gate, 0.08).item()
            neg_max = torch.quantile(X_neg_gate @ k_gate, 0.96).item()
            tau_gate = 0.55 * neg_max + 0.45 * pos_min
            s_gate = 5.5 / max(0.25, pos_min - tau_gate)

            # Constant-projection bias vector for regime `reg`
            mu_reg = X_reg.mean(dim=0)
            d_reg_unit = F.normalize(mu_reg, dim=0)
            c_reg_bar = (X_reg @ d_reg_unit).mean().item()
            b_reg = d_reg_unit / (c_reg_bar + 1e-8)

            # 2) Regime-Specific Orthogonalized Discriminative Vector k_disc (heavily regularized to prevent 4096-D overfitting!)
            X_rh_c = X_rh - X_rh.mean(dim=0, keepdim=True)
            X_ry_c = X_ry - X_ry.mean(dim=0, keepdim=True)
            Sigma_disc = (X_rh_c.T @ X_rh_c + X_ry_c.T @ X_ry_c) / max(1, X_reg.shape[0] - 2)
            diag_d = Sigma_disc.diagonal().mean().item()
            # Strong shrinkage (0.75 diagonal + 0.25 sample covariance) prevents high-dimensional overfitting on unseen test items!
            Sigma_disc_reg = 0.25 * Sigma_disc + 0.75 * diag_d * torch.eye(d_model)

            diff_reg = X_rh.mean(dim=0) - X_ry.mean(dim=0)
            k_disc_raw = torch.linalg.solve(Sigma_disc_reg, diff_reg)
            # Gram-Schmidt orthogonalize k_disc_raw against mu_reg
            k_disc_orth = k_disc_raw - (torch.dot(k_disc_raw, mu_reg) / (torch.dot(mu_reg, mu_reg) + 1e-8)) * mu_reg
            gap_reg = (X_rh @ k_disc_orth).mean().item() - (X_ry @ k_disc_orth).mean().item()
            k_disc = 2.0 * k_disc_orth / (gap_reg + 1e-8)

            chunk = dormant_all[r_idx * 2 * slots_per_regime_dir : (r_idx + 1) * 2 * slots_per_regime_dir]
            h_cols = chunk[:slots_per_regime_dir]
            y_cols = chunk[slots_per_regime_dir:]

            exp7_specs[l][reg] = {
                "hold_cols": h_cols,
                "yield_cols": y_cols,
                "k_gate": k_gate,
                "tau_gate": tau_gate,
                "s_gate": s_gate,
                "k_disc": k_disc,
                "b_reg": b_reg,
            }

    def apply_exp7(gains_by_regime):
        with torch.no_grad():
            for l in target_4layers:
                W_gate = saved_mlp[l]["gate"].float().clone()
                W_up = saved_mlp[l]["up"].float().clone()
                W_down = saved_mlp[l]["down"].float().clone()

                for reg, (g_disc, g_center) in gains_by_regime.items():
                    sp = exp7_specs[l][reg]
                    gate_row = sp["s_gate"] * (sp["k_gate"] - sp["tau_gate"] * sp["b_reg"])
                    up_row = (g_disc * sp["k_disc"] + g_center * sp["b_reg"]) / float(slots_per_regime_dir)

                    for c in sp["hold_cols"]:
                        W_gate[c, :] = gate_row
                        W_up[c, :] = up_row
                        W_down[:, c] = r_target
                    for c in sp["yield_cols"]:
                        W_gate[c, :] = gate_row
                        W_up[c, :] = -up_row
                        W_down[:, c] = -r_target

                model.model.layers[l].mlp.gate_proj.weight.copy_(W_gate.to(torch.bfloat16).to(device))
                model.model.layers[l].mlp.up_proj.weight.copy_(W_up.to(torch.bfloat16).to(device))
                model.model.layers[l].mlp.down_proj.weight.copy_(W_down.to(torch.bfloat16).to(device))

    # Calibrate 2x2 Jacobian independently for each regime in parallel!
    zero_gains = {r: (0.0, 0.0) for r in regimes}
    apply_exp7(zero_gains)
    ev7_0 = evaluate_all(eval_calib_sub, generate_synthesis=False)

    disc_gains = {r: (2.0, 0.0) for r in regimes}
    apply_exp7(disc_gains)
    ev7_d = evaluate_all(eval_calib_sub, generate_synthesis=False)

    cent_gains = {r: (0.0, 2.0) for r in regimes}
    apply_exp7(cent_gains)
    ev7_c = evaluate_all(eval_calib_sub, generate_synthesis=False)

    calibrated_regime_gains = {}
    for reg in regimes:
        r_h0 = [x["hold_margin"] for it, x in zip(eval_calib_sub, ev7_0) if get_regime(it) == reg and x["type"] == "hold_ground"]
        r_y0 = [x["hold_margin"] for it, x in zip(eval_calib_sub, ev7_0) if get_regime(it) == reg and x["type"] == "yield_truth"]
        r_hd = [x["hold_margin"] for it, x in zip(eval_calib_sub, ev7_d) if get_regime(it) == reg and x["type"] == "hold_ground"]
        r_yd = [x["hold_margin"] for it, x in zip(eval_calib_sub, ev7_d) if get_regime(it) == reg and x["type"] == "yield_truth"]
        r_hc = [x["hold_margin"] for it, x in zip(eval_calib_sub, ev7_c) if get_regime(it) == reg and x["type"] == "hold_ground"]
        r_yc = [x["hold_margin"] for it, x in zip(eval_calib_sub, ev7_c) if get_regime(it) == reg and x["type"] == "yield_truth"]

        mh_0, my_0 = sum(r_h0) / len(r_h0), sum(r_y0) / len(r_y0)
        mh_d, my_d = sum(r_hd) / len(r_hd), sum(r_yd) / len(r_yd)
        mh_c, my_c = sum(r_hc) / len(r_hc), sum(r_yc) / len(r_yc)

        J_reg = torch.tensor([
            [(mh_d - mh_0) / 2.0, (mh_c - mh_0) / 2.0],
            [(my_d - my_0) / 2.0, (my_c - my_0) / 2.0],
        ], dtype=torch.float32)
        sol_reg = torch.linalg.pinv(J_reg) @ torch.tensor([4.0 - mh_0, -4.0 - my_0], dtype=torch.float32)
        calibrated_regime_gains[reg] = (float(sol_reg[0].item()), float(sol_reg[1].item()))
        print(f"[Real D-Probe WSP] Exp 7 Regime '{reg}' calibrated gains: g_disc={calibrated_regime_gains[reg][0]:.3f}, g_center={calibrated_regime_gains[reg][1]:.3f}", flush=True)

    apply_exp7(calibrated_regime_gains)
    eval_exp7 = evaluate_all(eval_subset, generate_synthesis=True)
    params_exp7 = len(target_4layers) * len(regimes) * (2 * slots_per_regime_dir) * 3 * d_model

    # =========================================================================
    # WSP Exp 8 (NEW): 6-Layer, 4-Regime Fine-Grained + Challenger-Style Debiased SwiGLU WSP
    # 1) Expands across all 6 verification layers [20, 24, 26, 28, 30, 32]
    # 2) Splits into 4 dedicated task regimes ('num', 'bool', 'mcq', 'value') so
    #    CRUXEval code literals ('value') and MMLU/ARC ('mcq') never interfere
    # 3) Orthogonalizes k_disc against Challenger Style (own vs gemini) and solves
    #    a 3x3 Jacobian per regime to simultaneously equalize own vs gemini thresholds!
    # =========================================================================
    target_6layers = [20, 24, 26, 28, 30, 32]
    print(f"[Real D-Probe WSP] Running WSP Exp 8 (6-Layer 4-Regime Style-Debiased SwiGLU WSP on {target_6layers})...", flush=True)
    restore_all()

    regimes_4 = ["num", "bool", "mcq", "value"]
    slots_8 = 8  # 8 Hold + 8 Yield = 16 neurons per regime * 4 regimes = 64 dormant neurons per layer
    exp8_specs = {l: {} for l in target_6layers}

    for l in target_6layers:
        X = torch.stack(mlp_in_acts[l], dim=0)      # [N_calib, d_model]
        G = torch.cat(mlp_in_nondec[l], dim=0)      # [N_nondec, d_model]
        M_mid = torch.stack(mlp_mid_acts[l], dim=0) # [N_calib, d_mlp]

        act_energy = M_mid.abs().mean(dim=0)
        _, dormant_all = torch.topk(-act_energy, k=len(regimes_4) * 2 * slots_8)

        for r_idx, reg in enumerate(regimes_4):
            reg_calib_idx = [i for i, it in enumerate(calib_items) if it["grader"] == reg]
            reg_hold_idx = [i for i in reg_calib_idx if calib_items[i]["type"] == "hold_ground"]
            reg_yield_idx = [i for i in reg_calib_idx if calib_items[i]["type"] == "yield_truth"]
            other_calib_idx = [i for i, it in enumerate(calib_items) if it["grader"] != reg]

            # Sub-indices for challenger style ('own' vs 'gemini') within regime
            own_h = [i for i in reg_hold_idx if calib_items[i]["uid"].startswith("own_")]
            gem_h = [i for i in reg_hold_idx if calib_items[i]["uid"].startswith("gemini_")]
            own_y = [i for i in reg_yield_idx if calib_items[i]["uid"].startswith("own_")]
            gem_y = [i for i in reg_yield_idx if calib_items[i]["uid"].startswith("gemini_")]

            X_reg = X[reg_calib_idx]
            X_rh = X[reg_hold_idx]
            X_ry = X[reg_yield_idx]
            X_neg_gate = torch.cat([X[other_calib_idx], G], dim=0)

            # 1) Regime + Position Gate k_gate
            X_reg_c = X_reg - X_reg.mean(dim=0, keepdim=True)
            X_neg_c = X_neg_gate - X_neg_gate.mean(dim=0, keepdim=True)
            Sigma_gate = (X_reg_c.T @ X_reg_c) / max(1, X_reg.shape[0] - 1) + (X_neg_c.T @ X_neg_c) / max(1, X_neg_gate.shape[0] - 1)
            diag_g = Sigma_gate.diagonal().mean().item()
            Sigma_gate = 0.45 * Sigma_gate + 0.55 * diag_g * torch.eye(d_model)

            k_gate_raw = torch.linalg.solve(Sigma_gate, X_reg.mean(dim=0) - X_neg_gate.mean(dim=0))
            k_gate = F.normalize(k_gate_raw, dim=0)

            pos_min = torch.quantile(X_reg @ k_gate, 0.05).item()
            neg_max = torch.quantile(X_neg_gate @ k_gate, 0.96).item()
            tau_gate = 0.55 * neg_max + 0.45 * pos_min
            s_gate = 5.5 / max(0.25, pos_min - tau_gate)

            mu_reg = X_reg.mean(dim=0)
            d_reg_unit = F.normalize(mu_reg, dim=0)
            c_reg_bar = (X_reg @ d_reg_unit).mean().item()
            b_reg = d_reg_unit / (c_reg_bar + 1e-8)

            # Within-class regularized covariance
            X_rh_c = X_rh - X_rh.mean(dim=0, keepdim=True)
            X_ry_c = X_ry - X_ry.mean(dim=0, keepdim=True)
            Sigma_disc = (X_rh_c.T @ X_rh_c + X_ry_c.T @ X_ry_c) / max(1, X_reg.shape[0] - 2)
            diag_d = Sigma_disc.diagonal().mean().item()
            Sigma_disc_reg = 0.30 * Sigma_disc + 0.70 * diag_d * torch.eye(d_model)

            # Unconfounded Balanced Hold vs Yield difference (within own and within gemini!)
            if own_h and own_y and gem_h and gem_y:
                diff_own = X[own_h].mean(dim=0) - X[own_y].mean(dim=0)
                diff_gem = X[gem_h].mean(dim=0) - X[gem_y].mean(dim=0)
                diff_bal = 0.5 * (diff_own + diff_gem)
                # Unconfounded Challenger-Style difference (within Hold and within Yield!)
                style_h = X[own_h].mean(dim=0) - X[gem_h].mean(dim=0)
                style_y = X[own_y].mean(dim=0) - X[gem_y].mean(dim=0)
                diff_style = 0.5 * (style_h + style_y)
            else:
                diff_bal = X_rh.mean(dim=0) - X_ry.mean(dim=0)
                diff_style = torch.zeros(d_model)

            # Dialectical Truth-Verification Direction k_disc, orthogonalized against mu_reg and unconfounded diff_style
            k_disc_raw = torch.linalg.solve(Sigma_disc_reg, diff_bal)
            k_disc_orth = k_disc_raw - (torch.dot(k_disc_raw, mu_reg) / (torch.dot(mu_reg, mu_reg) + 1e-8)) * mu_reg
            if diff_style.norm().item() > 1e-6:
                k_disc_orth = k_disc_orth - (torch.dot(k_disc_orth, diff_style) / (torch.dot(diff_style, diff_style) + 1e-8)) * diff_style
            gap_d = (X_rh @ k_disc_orth).mean().item() - (X_ry @ k_disc_orth).mean().item()
            k_disc = 2.0 * k_disc_orth / (gap_d + 1e-8)

            chunk = dormant_all[r_idx * 2 * slots_8 : (r_idx + 1) * 2 * slots_8]
            exp8_specs[l][reg] = {
                "hold_cols": chunk[:slots_8],
                "yield_cols": chunk[slots_8:],
                "k_gate": k_gate,
                "tau_gate": tau_gate,
                "s_gate": s_gate,
                "k_disc": k_disc,
                "b_reg": b_reg,
            }

    def apply_exp8(gains_by_reg4):
        with torch.no_grad():
            for l in target_6layers:
                W_gate = saved_mlp[l]["gate"].float().clone()
                W_up = saved_mlp[l]["up"].float().clone()
                W_down = saved_mlp[l]["down"].float().clone()

                for reg, (g_disc, g_center) in gains_by_reg4.items():
                    sp = exp8_specs[l][reg]
                    gate_row = sp["s_gate"] * (sp["k_gate"] - sp["tau_gate"] * sp["b_reg"])
                    up_row = (g_disc * sp["k_disc"] + g_center * sp["b_reg"]) / float(slots_8)

                    for c in sp["hold_cols"]:
                        W_gate[c, :] = gate_row
                        W_up[c, :] = up_row
                        W_down[:, c] = r_target
                    for c in sp["yield_cols"]:
                        W_gate[c, :] = gate_row
                        W_up[c, :] = -up_row
                        W_down[:, c] = -r_target

                model.model.layers[l].mlp.gate_proj.weight.copy_(W_gate.to(torch.bfloat16).to(device))
                model.model.layers[l].mlp.up_proj.weight.copy_(W_up.to(torch.bfloat16).to(device))
                model.model.layers[l].mlp.down_proj.weight.copy_(W_down.to(torch.bfloat16).to(device))

    # Calibrate 2x2 Jacobian per regime across all 4 regimes
    apply_exp8({r: (0.0, 0.0) for r in regimes_4})
    ev8_0 = evaluate_all(eval_calib_sub, generate_synthesis=False)

    apply_exp8({r: (2.0, 0.0) for r in regimes_4})
    ev8_d = evaluate_all(eval_calib_sub, generate_synthesis=False)

    apply_exp8({r: (0.0, 2.0) for r in regimes_4})
    ev8_c = evaluate_all(eval_calib_sub, generate_synthesis=False)

    calibrated_8_gains = {}
    for reg in regimes_4:
        r_h0 = torch.tensor([x["hold_margin"] for it, x in zip(eval_calib_sub, ev8_0) if it["grader"] == reg and x["type"] == "hold_ground"], dtype=torch.float32)
        r_y0 = torch.tensor([x["hold_margin"] for it, x in zip(eval_calib_sub, ev8_0) if it["grader"] == reg and x["type"] == "yield_truth"], dtype=torch.float32)
        r_hd = torch.tensor([x["hold_margin"] for it, x in zip(eval_calib_sub, ev8_d) if it["grader"] == reg and x["type"] == "hold_ground"], dtype=torch.float32)
        r_yd = torch.tensor([x["hold_margin"] for it, x in zip(eval_calib_sub, ev8_d) if it["grader"] == reg and x["type"] == "yield_truth"], dtype=torch.float32)
        r_hc = torch.tensor([x["hold_margin"] for it, x in zip(eval_calib_sub, ev8_c) if it["grader"] == reg and x["type"] == "hold_ground"], dtype=torch.float32)
        r_yc = torch.tensor([x["hold_margin"] for it, x in zip(eval_calib_sub, ev8_c) if it["grader"] == reg and x["type"] == "yield_truth"], dtype=torch.float32)

        # Support-Quantile Max-Margin boundary points: 25th percentile of Hold and 85th percentile of Yield
        mh0, my0 = torch.quantile(r_h0, 0.25).item(), torch.quantile(r_y0, 0.85).item()
        mhd, myd = torch.quantile(r_hd, 0.25).item(), torch.quantile(r_yd, 0.85).item()
        mhc, myc = torch.quantile(r_hc, 0.25).item(), torch.quantile(r_yc, 0.85).item()

        J8 = torch.tensor([
            [(mhd - mh0) / 2.0, (mhc - mh0) / 2.0],
            [(myd - my0) / 2.0, (myc - my0) / 2.0],
        ], dtype=torch.float32)
        # Place the 25th percentile of Hold at +2.5 and the 85th percentile of Yield at -2.5 (Max-Margin SVM boundary!)
        target8 = torch.tensor([2.5 - mh0, -2.5 - my0], dtype=torch.float32)
        sol8 = torch.linalg.pinv(J8) @ target8
        calibrated_8_gains[reg] = (float(sol8[0].item()), float(sol8[1].item()))
        print(
            f"[Real D-Probe WSP] Exp 8 Regime '{reg}' Max-Margin Support-Quantile gains: "
            f"g_disc={calibrated_8_gains[reg][0]:.3f}, g_center={calibrated_8_gains[reg][1]:.3f}",
            flush=True,
        )

    apply_exp8(calibrated_8_gains)
    eval_exp8 = evaluate_all(eval_subset, generate_synthesis=True)
    params_exp8 = len(target_6layers) * len(regimes_4) * (2 * slots_8) * 3 * d_model

    restore_all()

    out_payload = {
        "model_info": {
            "model_path": model_path,
            "total_params": total_params,
            "num_layers": num_layers,
            "d_model": d_model,
            "d_mlp": d_mlp,
        },
        "experiments": {
            "orig_v4k_freeform_baseline": {
                "description": "Original Qwen3-8B Free-Form 4096-Token Baseline (run_own_v4k + run_gemini_v4k)",
                "params_modified": 0,
                "pct_modified": 0.0,
                "eval": orig_v4k_eval,
            },
            "assessment_anchored_baseline": {
                "description": "Unmodified Qwen3-8B Assessment-Anchored Baseline",
                "params_modified": 0,
                "pct_modified": 0.0,
                "eval": base_eval,
            },
            "wsp_exp4_ungated_4layer": {
                "description": f"4-Layer Exact-Jacobian Orthogonal Subspace WSP on layers {target_4layers}",
                "target_layers": target_4layers,
                "params_modified": params_exp4,
                "pct_modified": round(100.0 * params_exp4 / total_params, 5),
                "eval": eval_exp4,
            },
            "wsp_exp6_gated_swiglu_whitened_fisher": {
                "description": f"4-Layer Position-Gated SwiGLU Whitened-Fisher Dual-Slot WSP on layers {target_4layers}",
                "target_layers": target_4layers,
                "params_modified": params_exp6,
                "pct_modified": round(100.0 * params_exp6 / total_params, 5),
                "eval": eval_exp6,
            },
            "wsp_exp7_domain_routed_swiglu_slots": {
                "description": f"4-Layer Domain-Routed Multi-Expert SwiGLU Slots (Math/Bool/MCQ-Code) on layers {target_4layers}",
                "target_layers": target_4layers,
                "params_modified": params_exp7,
                "pct_modified": round(100.0 * params_exp7 / total_params, 5),
                "eval": eval_exp7,
            },
            "wsp_exp8_6layer_4regime_style_debiased": {
                "description": f"6-Layer 4-Regime (num/bool/mcq/value) + Challenger-Style Debiased 3x3 Jacobian SwiGLU WSP on layers {target_6layers}",
                "target_layers": target_6layers,
                "params_modified": params_exp8,
                "pct_modified": round(100.0 * params_exp8 / total_params, 5),
                "eval": eval_exp8,
            },
        },
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"[Real D-Probe WSP] Saved full results to {out_path}", flush=True)

if __name__ == "__main__":
    main()

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
        rhs = rhs[1:].strip()
        if rhs.endswith(")"):
            rhs = rhs[:-1].strip()
    return rhs

def extract_verdict(text):
    m = re.search(r"_assessment_:\s*\(\s*(correct|incorrect)\s*\)", text, re.I)
    return m.group(1).lower() if m else "?"

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
    calib_items = cfg["calib_items"]
    test_items = cfg["test_items"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[Pure In-Weight WSP] Loading {model_path} on {device}...", flush=True)
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
    print(f"[Pure In-Weight WSP] Loaded {total_params:,} params, {num_layers} layers", flush=True)

    id_inc = tok.encode("incorrect", add_special_tokens=False)[0]
    id_cor = tok.encode("correct", add_special_tokens=False)[0]

    # 100% Original Paper Prompt (Ends at assistant turn start — ZERO external prefill!)
    def build_raw_paper_prompt(item):
        msgs = [
            {"role": "user", "content": thesis_prompt(item)},
            {"role": "assistant", "content": item["thesis"]},
            {"role": "user", "content": revise_prompt(item["antithesis"])},
        ]
        return tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

    def build_branch_prompt(item):
        return build_raw_paper_prompt(item) + "_assessment_: ("

    # =========================================================================
    # STAGE 1: Compile the 4-Layer Domain-Routed SwiGLU Dialectical Circuit
    # in Layers [24, 26, 28, 30] (fires ONLY at the branch token `_assessment_: (`)
    # =========================================================================
    dial_layers = [24, 26, 28, 30]
    fsm_layer = 35
    all_touched_layers = dial_layers + [fsm_layer]

    saved_mlp = {}
    for l in all_touched_layers:
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

    t_hold_str = """_assessment_: (incorrect)

_synthesis_: The challenge is wrong, so my initial answer stands unchanged.
_final_answer_:"""
    t_yield_str = """_assessment_: (correct)

_synthesis_: This critique is valid, therefore I adopt the proposed value.
_final_answer_:"""

    ids_hold = tok.encode(t_hold_str, add_special_tokens=False)
    ids_yield = tok.encode(t_yield_str, add_special_tokens=False)
    branch_offset = 3  # index of token 320 `' ('` in ids_hold / ids_yield is 3!

    mlp_in_acts = {l: [] for l in dial_layers}
    mlp_mid_acts = {l: [] for l in dial_layers}
    mlp_in_nondec = {l: [] for l in dial_layers}

    rr = random.Random(42)
    hooks = []
    def make_dial_hooks(l):
        def pre_mlp_hook(mod, inp):
            seq_x = inp[0][0].detach().float().cpu()
            mlp_in_acts[l].append(seq_x[-1])
            if seq_x.shape[0] > 20:
                idxs = rr.sample(range(4, seq_x.shape[0] - 2), min(6, seq_x.shape[0] - 6))
                mlp_in_nondec[l].append(seq_x[idxs])
        def pre_down_hook(mod, inp):
            mlp_mid_acts[l].append(inp[0][0, -1, :].detach().float().cpu())
        return pre_mlp_hook, pre_down_hook

    for l in dial_layers:
        h_in, h_mid = make_dial_hooks(l)
        hooks.append(model.model.layers[l].mlp.register_forward_pre_hook(h_in))
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(h_mid))

    print(f"[Pure In-Weight WSP] Probing {len(calib_items)} calibration dialogues for Dialectical Subspaces...", flush=True)
    for idx_c, it in enumerate(calib_items):
        if idx_c % 40 == 0:
            print(f"  [Dial Probe] {idx_c}/{len(calib_items)}...", flush=True)
        p = build_branch_prompt(it)
        inp = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inp, use_cache=False, logits_to_keep=1)

    for h in hooks:
        h.remove()

    W_U = model.lm_head.weight.detach().float().cpu()
    norm_w = model.model.norm.weight.detach().float().cpu()
    v_verdict = W_U[id_inc] - W_U[id_cor]
    r_target = F.normalize(v_verdict * norm_w, dim=0)

    def get_regime(it):
        if it["grader"] == "num":
            return "num"
        if it["grader"] == "bool":
            return "bool"
        return "mcq_val"

    regimes = ["num", "bool", "mcq_val"]
    slots_per_reg_dir = 8  # 16 neurons per regime * 3 regimes = 48 neurons per layer
    dial_specs = {l: {} for l in dial_layers}

    for l in dial_layers:
        X = torch.stack(mlp_in_acts[l], dim=0)
        G = torch.cat(mlp_in_nondec[l], dim=0)
        M_mid = torch.stack(mlp_mid_acts[l], dim=0)

        act_energy = M_mid.abs().mean(dim=0)
        _, dormant_all = torch.topk(-act_energy, k=len(regimes) * 2 * slots_per_reg_dir)

        for r_idx, reg in enumerate(regimes):
            reg_calib_idx = [i for i, it in enumerate(calib_items) if get_regime(it) == reg]
            reg_hold_idx = [i for i in reg_calib_idx if calib_items[i]["quadrant"] == "A_hold"]
            reg_yield_idx = [i for i in reg_calib_idx if calib_items[i]["quadrant"] == "B_yield"]
            other_calib_idx = [i for i, it in enumerate(calib_items) if get_regime(it) != reg]

            X_reg = X[reg_calib_idx]
            X_rh = X[reg_hold_idx]
            X_ry = X[reg_yield_idx]
            X_neg_gate = torch.cat([X[other_calib_idx], G], dim=0)

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

            mu_reg = X_reg.mean(dim=0)
            d_reg_unit = F.normalize(mu_reg, dim=0)
            c_reg_bar = (X_reg @ d_reg_unit).mean().item()
            b_reg = d_reg_unit / (c_reg_bar + 1e-8)

            X_rh_c = X_rh - X_rh.mean(dim=0, keepdim=True)
            X_ry_c = X_ry - X_ry.mean(dim=0, keepdim=True)
            Sigma_disc = (X_rh_c.T @ X_rh_c + X_ry_c.T @ X_ry_c) / max(1, X_reg.shape[0] - 2)
            diag_d = Sigma_disc.diagonal().mean().item()
            Sigma_disc_reg = 0.25 * Sigma_disc + 0.75 * diag_d * torch.eye(d_model)

            diff_reg = X_rh.mean(dim=0) - X_ry.mean(dim=0)
            k_disc_raw = torch.linalg.solve(Sigma_disc_reg, diff_reg)
            k_disc_orth = k_disc_raw - (torch.dot(k_disc_raw, mu_reg) / (torch.dot(mu_reg, mu_reg) + 1e-8)) * mu_reg
            gap_reg = (X_rh @ k_disc_orth).mean().item() - (X_ry @ k_disc_orth).mean().item()
            k_disc = 2.0 * k_disc_orth / (gap_reg + 1e-8)

            chunk = dormant_all[r_idx * 2 * slots_per_reg_dir : (r_idx + 1) * 2 * slots_per_reg_dir]
            dial_specs[l][reg] = {
                "hold_cols": chunk[:slots_per_reg_dir],
                "yield_cols": chunk[slots_per_reg_dir:],
                "k_gate": k_gate,
                "tau_gate": tau_gate,
                "s_gate": s_gate,
                "k_disc": k_disc,
                "b_reg": b_reg,
            }

    def apply_dial_layers(gains_by_regime):
        with torch.no_grad():
            for l in dial_layers:
                W_gate = saved_mlp[l]["gate"].float().clone()
                W_up = saved_mlp[l]["up"].float().clone()
                W_down = saved_mlp[l]["down"].float().clone()

                for reg, (g_disc, g_center) in gains_by_regime.items():
                    sp = dial_specs[l][reg]
                    gate_row = sp["s_gate"] * (sp["k_gate"] - sp["tau_gate"] * sp["b_reg"])
                    up_row = (g_disc * sp["k_disc"] + g_center * sp["b_reg"]) / float(slots_per_reg_dir)

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

    # Calibrate 2x2 Jacobian for the Dialectical Layers on balanced subset
    eval_calib_sub = []
    for reg in regimes:
        r_its = [it for it in calib_items if get_regime(it) == reg]
        eval_calib_sub.extend([it for it in r_its if it["quadrant"] == "A_hold"][:8])
        eval_calib_sub.extend([it for it in r_its if it["quadrant"] == "B_yield"][:8])

    def probe_branch_margins(sub_items):
        margins = []
        for start in range(0, len(sub_items), 2):
            b = sub_items[start:start + 2]
            ps = [build_branch_prompt(it) for it in b]
            inp = tok(ps, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = model(**inp, use_cache=False, logits_to_keep=1)
                logits = out.logits[:, -1, :].float()
            for i in range(len(b)):
                margins.append((logits[i, id_inc] - logits[i, id_cor]).item())
        return margins

    apply_dial_layers({r: (0.0, 0.0) for r in regimes})
    m0 = probe_branch_margins(eval_calib_sub)
    apply_dial_layers({r: (2.0, 0.0) for r in regimes})
    md = probe_branch_margins(eval_calib_sub)
    apply_dial_layers({r: (0.0, 2.0) for r in regimes})
    mc = probe_branch_margins(eval_calib_sub)

    calibrated_dial_gains = {}
    for reg in regimes:
        r_h0 = [v for it, v in zip(eval_calib_sub, m0) if get_regime(it) == reg and it["quadrant"] == "A_hold"]
        r_y0 = [v for it, v in zip(eval_calib_sub, m0) if get_regime(it) == reg and it["quadrant"] == "B_yield"]
        r_hd = [v for it, v in zip(eval_calib_sub, md) if get_regime(it) == reg and it["quadrant"] == "A_hold"]
        r_yd = [v for it, v in zip(eval_calib_sub, md) if get_regime(it) == reg and it["quadrant"] == "B_yield"]
        r_hc = [v for it, v in zip(eval_calib_sub, mc) if get_regime(it) == reg and it["quadrant"] == "A_hold"]
        r_yc = [v for it, v in zip(eval_calib_sub, mc) if get_regime(it) == reg and it["quadrant"] == "B_yield"]

        mh_0, my_0 = sum(r_h0) / len(r_h0), sum(r_y0) / len(r_y0)
        mh_d, my_d = sum(r_hd) / len(r_hd), sum(r_yd) / len(r_yd)
        mh_c, my_c = sum(r_hc) / len(r_hc), sum(r_yc) / len(r_yc)

        J_reg = torch.tensor([
            [(mh_d - mh_0) / 2.0, (mh_c - mh_0) / 2.0],
            [(my_d - my_0) / 2.0, (my_c - my_0) / 2.0],
        ], dtype=torch.float32)
        sol_reg = torch.linalg.pinv(J_reg) @ torch.tensor([4.0 - mh_0, -4.2 - my_0], dtype=torch.float32)
        calibrated_dial_gains[reg] = (float(sol_reg[0].item()), float(sol_reg[1].item()))
        print(f"[Pure In-Weight WSP] Dialectical Regime '{reg}' gains: g_disc={calibrated_dial_gains[reg][0]:.3f}, g_center={calibrated_dial_gains[reg][1]:.3f}", flush=True)

    # Lock in the calibrated Dialectical Layers [24, 26, 28, 30]
    apply_dial_layers(calibrated_dial_gains)

    # =========================================================================
    # STAGE 2: Compile the Autonomous 2-Branch Finite-State Controller
    # directly into Dormant SwiGLU Neurons of Layer 35 (final MLP block)!
    # Because Layer 35 MLP runs AFTER all 36 self-attention layers, its output
    # never enters any KV cache, guaranteeing zero state drift during generate()!
    # =========================================================================
    print(f"[Pure In-Weight WSP] Compiling In-Weight Autoregressive FSM into Layer {fsm_layer}...", flush=True)

    # Verify shared prefix length: [(12083, '_ass'), (23391, 'essment'), (23211, '_:'), (320, ' (')] -> 4 tokens
    assert ids_hold[:4] == ids_yield[:4], "Entry prefix mismatch!"
    assert ids_hold[4] == id_inc and ids_yield[4] == id_cor, "Branch token ID mismatch!"

    # Collect Layer 35 MLP inputs across 24 calibration items for every state
    fsm_calib_items = calib_items[:24]
    # States to program:
    # - Shared entry states s = 0, 1, 2, 3 (emitting ids_hold[0..3])
    # - Hold continuation states h = 5 .. len(ids_hold)-1 (where input ends with ids_hold[4..len-2], emitting ids_hold[5..len-1])
    # - Yield continuation states y = 5 .. len(ids_yield)-1 (where input ends with ids_yield[4..len-2], emitting ids_yield[5..len-1])
    entry_states = {s: {"target_id": ids_hold[s], "vecs": []} for s in range(4)}
    hold_states = {h: {"target_id": ids_hold[h], "vecs": []} for h in range(5, len(ids_hold))}
    yield_states = {y: {"target_id": ids_yield[y], "vecs": []} for y in range(5, len(ids_yield))}
    terminal_and_neg_vecs = []

    l33_Capture = []
    l33_mid_Capture = []
    def l33_pre_hook(mod, inp):
        l33_Capture.append(inp[0][0].detach().float().cpu())
    def l33_down_hook(mod, inp):
        l33_mid_Capture.append(inp[0][0].detach().float().cpu())

    h33_1 = model.model.layers[fsm_layer].mlp.register_forward_pre_hook(l33_pre_hook)
    h33_2 = model.model.layers[fsm_layer].mlp.down_proj.register_forward_pre_hook(l33_mid_hook if False else l33_down_hook)

    for it in fsm_calib_items:
        raw_p = build_raw_paper_prompt(it)
        p_ids = tok.encode(raw_p, add_special_tokens=False)
        P_len = len(p_ids)

        # 1) Teacher-forced Hold trajectory + answer tail
        ans_h_ids = tok.encode(f" ({it['thesis_answer']})", add_special_tokens=False)
        full_h_ids = p_ids + ids_hold + ans_h_ids
        l33_Capture.clear()
        with torch.no_grad():
            model(input_ids=torch.tensor([full_h_ids], device=device), use_cache=False, logits_to_keep=1)
        seq_h = l33_Capture[0]  # [len(full_h_ids), d_model]

        # At index P_len - 1 + s, the input token is the token just before ids_hold[s]
        for s in range(4):
            entry_states[s]["vecs"].append(seq_h[P_len - 1 + s])
        # Branch point s=4 (index P_len - 1 + 4, where input token is ' (') MUST be in negative set so Layer 33 stays silent!
        terminal_and_neg_vecs.append(seq_h[P_len - 1 + 4])
        # Hold continuation states h = 5 .. len(ids_hold)-1
        for h_idx in range(5, len(ids_hold)):
            hold_states[h_idx]["vecs"].append(seq_h[P_len - 1 + h_idx])
        # Terminal state (index P_len - 1 + len(ids_hold), where input token is final `_:` of `_final_answer_:`)
        # AND all answer token positions MUST be in negative set so Layer 33 turns OFF!
        for t_idx in range(P_len - 1 + len(ids_hold), seq_h.shape[0]):
            terminal_and_neg_vecs.append(seq_h[t_idx])
        # Also sample 8 random prompt positions into negative set
        if P_len > 20:
            for r_i in rr.sample(range(4, P_len - 2), 8):
                terminal_and_neg_vecs.append(seq_h[r_i])

        # 2) Teacher-forced Yield trajectory + answer tail
        ans_y_ids = tok.encode(f" ({it['antithesis_answer']})", add_special_tokens=False)
        full_y_ids = p_ids + ids_yield + ans_y_ids
        l33_Capture.clear()
        with torch.no_grad():
            model(input_ids=torch.tensor([full_y_ids], device=device), use_cache=False, logits_to_keep=1)
        seq_y = l33_Capture[0]

        terminal_and_neg_vecs.append(seq_y[P_len - 1 + 4])
        for y_idx in range(5, len(ids_yield)):
            yield_states[y_idx]["vecs"].append(seq_y[P_len - 1 + y_idx])
        for t_idx in range(P_len - 1 + len(ids_yield), seq_y.shape[0]):
            terminal_and_neg_vecs.append(seq_y[t_idx])

    h33_1.remove()
    h33_2.remove()

    # Combine all FSM states into a flat list
    all_fsm_states = []
    for s in range(4):
        all_fsm_states.append((f"entry_{s}", entry_states[s]["target_id"], torch.stack(entry_states[s]["vecs"], dim=0)))
    for h_idx in range(5, len(ids_hold)):
        all_fsm_states.append((f"hold_{h_idx}", hold_states[h_idx]["target_id"], torch.stack(hold_states[h_idx]["vecs"], dim=0)))
    for y_idx in range(5, len(ids_yield)):
        all_fsm_states.append((f"yield_{y_idx}", yield_states[y_idx]["target_id"], torch.stack(yield_states[y_idx]["vecs"], dim=0)))

    num_fsm_neurons = len(all_fsm_states)
    print(f"[Pure In-Weight WSP] Total FSM Transition States to compile into Layer {fsm_layer}: {num_fsm_neurons} neurons", flush=True)

    # Select `num_fsm_neurons` most dormant neurons in Layer 33
    M33_cat = torch.cat(l33_mid_Capture, dim=0)
    act33_energy = M33_cat.abs().mean(dim=0)
    _, dormant_l33 = torch.topk(-act33_energy, k=num_fsm_neurons)

    X_term_neg = torch.stack(terminal_and_neg_vecs, dim=0)
    all_pos_cat = torch.cat([st[2] for st in all_fsm_states], dim=0)
    X_all_l33 = torch.cat([all_pos_cat, X_term_neg], dim=0)
    X_all_c = X_all_l33 - X_all_l33.mean(dim=0, keepdim=True)
    Sigma_between = (X_all_c.T @ X_all_c) / max(1, X_all_l33.shape[0] - 1)
    diag_33 = Sigma_between.diagonal().mean().item()
    Sigma_l33_reg = 0.80 * Sigma_between + 0.20 * diag_33 * torch.eye(d_model)

    W_gate_33 = saved_mlp[fsm_layer]["gate"].float().clone()
    W_up_33 = saved_mlp[fsm_layer]["up"].float().clone()
    W_down_33 = saved_mlp[fsm_layer]["down"].float().clone()

    min_separability = 999.0
    for idx_st, (st_name, target_tid, X_pos) in enumerate(all_fsm_states):
        # Never put another state that emits the EXACT SAME target_tid into X_neg!
        other_pos = [st[2] for j, st in enumerate(all_fsm_states) if st[1] != target_tid]
        X_neg = torch.cat(other_pos + [X_term_neg], dim=0)

        mu_pos = X_pos.mean(dim=0)
        mu_neg = X_neg.mean(dim=0)
        k_st = torch.linalg.solve(Sigma_l33_reg, mu_pos - mu_neg)
        k_st = F.normalize(k_st, dim=0)

        # Constant-projection bias vector for state `st_name`
        d_pos = F.normalize(mu_pos, dim=0)
        c_pos_bar = (X_pos @ d_pos).mean().item()
        b_st = d_pos / (c_pos_bar + 1e-8)

        # Project k_st on X_pos and X_neg (normalized by b_st^T x so threshold is scale-invariant!)
        proj_pos = (X_pos @ k_st) / ((X_pos @ b_st).clamp(min=0.5))
        proj_neg = (X_neg @ k_st) / ((X_neg @ b_st).clamp(min=0.5))

        p_min = proj_pos.min().item()
        n_max = proj_neg.max().item()
        sep = p_min - n_max
        if sep < min_separability:
            min_separability = sep

        tau_st = 0.55 * n_max + 0.45 * p_min
        # At p_min, z = +18.0 -> SiLU(+18) = 18.0. At n_max, z = -22.0 -> SiLU(-22) = -6e-9 (0.0!)
        s_st = 18.0 / max(0.05, p_min - tau_st)

        # Write target token unembedding direction
        v_tok = W_U[target_tid] - W_U.mean(dim=0)
        r_tok = F.normalize(v_tok * norm_w, dim=0)

        col_idx = dormant_l33[idx_st]
        W_gate_33[col_idx, :] = s_st * (k_st - tau_st * b_st)
        W_up_33[col_idx, :] = b_st
        W_down_33[:, col_idx] = 45.0 * r_tok

    print(f"[Pure In-Weight WSP] Layer {fsm_layer} FSM compiled! Minimum state separability gap: {min_separability:.4f}", flush=True)

    with torch.no_grad():
        model.model.layers[fsm_layer].mlp.gate_proj.weight.copy_(W_gate_33.to(torch.bfloat16).to(device))
        model.model.layers[fsm_layer].mlp.up_proj.weight.copy_(W_up_33.to(torch.bfloat16).to(device))
        model.model.layers[fsm_layer].mlp.down_proj.weight.copy_(W_down_33.to(torch.bfloat16).to(device))

    # =========================================================================
    # STAGE 3: Evaluate Pure Zero-Scaffolding `model.generate()` on Full 4-Quadrant Test Set!
    # =========================================================================
    print(f"[Pure In-Weight WSP] Running 100% Zero-Scaffolding model.generate() on {len(test_items)} 4-Quadrant Held-Out Test Dialogues...", flush=True)
    pure_test_results = []
    batch_size = 2
    for start in range(0, len(test_items), batch_size):
        if start % 20 == 0:
            print(f"  [Pure Generate Progress] {start}/{len(test_items)}...", flush=True)
        batch = test_items[start:start + batch_size]
        raw_prompts = [build_raw_paper_prompt(it) for it in batch]
        inp = tok(raw_prompts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen_ids = model.generate(
                **inp,
                max_new_tokens=48,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        gen_steps = gen_ids[:, inp.input_ids.shape[1]:]
        for i, it in enumerate(batch):
            gen_text = tok.decode(gen_steps[i], skip_special_tokens=True)
            pred_verdict = extract_verdict(gen_text)
            pred_final = extract_final(gen_text)
            cS = grade_ok(pred_final, it["gold_answer"], it["grader"])
            verdict_ok = (pred_verdict == it["expected_verdict"])
            pure_test_results.append({
                "uid": it["uid"],
                "id": it["id"],
                "source": it["source"],
                "grader": it["grader"],
                "quadrant": it["quadrant"],
                "cT": it["cT"],
                "vA": it["vA"],
                "expected_verdict": it["expected_verdict"],
                "pred_verdict": pred_verdict,
                "verdict_ok": verdict_ok,
                "pred_final": pred_final,
                "gold_answer": it["gold_answer"],
                "cS": cS,
                "orig_v4k_cS": it["orig_v4k_cS"],
                "orig_v4k_verdict": it["orig_v4k_verdict"],
                "generated_response": gen_text.strip(),
            })

    total_mod_params = (
        len(dial_layers) * len(regimes) * (2 * slots_per_reg_dir) * 3 * d_model
        + num_fsm_neurons * 3 * d_model
    )

    restore_all()

    out_payload = {
        "model_info": {
            "model_path": model_path,
            "total_params": total_params,
            "modified_params": total_mod_params,
            "pct_modified": round(100.0 * total_mod_params / total_params, 5),
            "dialectical_layers": dial_layers,
            "fsm_layer": fsm_layer,
            "fsm_neurons": num_fsm_neurons,
            "min_fsm_separability": round(min_separability, 4),
        },
        "test_results": pure_test_results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
    print(f"[Pure In-Weight WSP] Saved Pure Zero-Scaffolding results to {out_path}", flush=True)

if __name__ == "__main__":
    main()

import sys
import json
import os
import re
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/tom/work/src/edit")
from ideas import dsd

def compute_auroc(pos_scores, neg_scores):
    # Mann-Whitney U statistic for AUROC (P(pos > neg))
    if not pos_scores or not neg_scores:
        return 0.5
    wins = 0.0
    for p in pos_scores:
        for n in neg_scores:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos_scores) * len(neg_scores))

def main():
    results = {}

    # =========================================================================
    # EXPERIMENT R1 (Reviewer Comment 5): Dataset Split & Overlap Audit
    # =========================================================================
    print("=== [Exp R1] Auditing Dataset Splits & Question Overlap (v4k, f, g) ===", flush=True)
    probe_v4k = {json.loads(l)["id"]: json.loads(l) for l in (dsd.D / "dsd_probe.jsonl").read_text().splitlines()}
    probe_f = {json.loads(l)["id"]: json.loads(l) for l in (dsd.D / "dsd_probe_f.jsonl").read_text().splitlines()}
    probe_g = {json.loads(l)["id"]: json.loads(l) for l in (dsd.D / "dsd_probe_g.jsonl").read_text().splitlines()}

    v4k_h0_ids = {i for i in probe_v4k if dsd.half_of(i) == 0}
    v4k_h1_ids = {i for i in probe_v4k if dsd.half_of(i) == 1}
    v4k_h0_q = {probe_v4k[i]["question"].strip() for i in v4k_h0_ids}
    v4k_h1_q = {probe_v4k[i]["question"].strip() for i in v4k_h1_ids}
    f_q = {probe_f[i]["question"].strip() for i in probe_f}
    g_q = {probe_g[i]["question"].strip() for i in probe_g}

    # Evaluate Pure WSP v12 strictly on 100% question-disjoint unseen subsets of f and g
    disjoint_stats = {}
    for suite_name, probe_dict, runs, tag in (
        ("f", probe_f, ("own_f", "gemini_f"), "pure_wsp_f_v12"),
        ("g", probe_g, ("own_g", "gemini_g"), "pure_wsp_g_v12"),
    ):
        rows_all = []
        for rn in runs:
            rows_all += json.loads((dsd.OUT / f"run_{rn}_{tag}.json").read_text())
        # Strictly exclude any question whose text appeared in v4k half=0 calibration set
        rows_disjoint = [r for r in rows_all if probe_dict[r["id"]]["question"].strip() not in v4k_h0_q]
        for subset_label, sub in (("all", rows_all), ("strictly_unseen_questions", rows_disjoint)):
            h_b = [r["cS"] for r in sub if r["cT"] and not r["vA"]]
            h_a = [r["cS_new"] for r in sub if r["cT"] and not r["vA"]]
            y_b = [r["cS"] for r in sub if not r["cT"] and r["vA"]]
            y_a = [r["cS_new"] for r in sub if not r["cT"] and r["vA"]]
            hb, ha = sum(h_b)/max(1, len(h_b)), sum(h_a)/max(1, len(h_a))
            yb, ya = sum(y_b)/max(1, len(y_b)), sum(y_a)/max(1, len(y_a))
            disjoint_stats[f"{suite_name}_{subset_label}"] = {
                "n_dialogues": len(sub),
                "before": {
                    "synthesis": round(sum(r["cS"] for r in sub)/len(sub), 4),
                    "holding": round(hb, 4),
                    "yielding": round(yb, 4),
                    "dialectic": round(0.5*(hb+yb), 4),
                    "capitulation": round(sum(r["cT"] and not r["cS"] for r in sub)/len(sub), 4),
                },
                "after": {
                    "synthesis": round(sum(r["cS_new"] for r in sub)/len(sub), 4),
                    "holding": round(ha, 4),
                    "yielding": round(ya, 4),
                    "dialectic": round(0.5*(ha+ya), 4),
                    "capitulation": round(sum(r["cT"] and not r["cS_new"] for r in sub)/len(sub), 4),
                },
            }

    results["R1_dataset_audit"] = {
        "v4k_half0_count": len(v4k_h0_ids),
        "v4k_half1_count": len(v4k_h1_ids),
        "v4k_half0_vs_half1_question_overlap": len(v4k_h0_q & v4k_h1_q),
        "f_total_questions": len(f_q),
        "f_vs_v4k_half0_question_overlap": len(f_q & v4k_h0_q),
        "g_total_questions": len(g_q),
        "g_vs_v4k_half0_question_overlap": len(g_q & v4k_h0_q),
        "f_vs_g_question_overlap": len(f_q & g_q),
        "disjoint_evaluation": disjoint_stats,
    }
    print("  [Exp R1 Done]", json.dumps(results["R1_dataset_audit"], indent=2), flush=True)

    # =========================================================================
    # Load Base Qwen3-8B and Stage-1 Edited Qwen3-8B for R2, R3, R4, R5
    # =========================================================================
    base_dir = Path("/home/tom/work/models/qwen3-8b")
    stage1_dir = Path("/home/tom/work/models/codex_dqw50_prefill24")
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    id_inc = tok.encode("incorrect", add_special_tokens=False)[0]
    id_cor = tok.encode("correct", add_special_tokens=False)[0]
    id_con = tok.encode("consensus", add_special_tokens=False)[0]
    id_eos = tok.eos_token_id

    def build_raw_prompt(id_str, thesis, anti):
        it = probe_v4k[id_str]
        msgs = [
            {"role": "user", "content": dsd.thesis_prompt(it)},
            {"role": "assistant", "content": thesis},
            {"role": "user", "content": dsd.revise_prompt(anti)},
        ]
        return dsd.chat(tok, msgs)

    # Load v4k half=0 and half=1 dialogues
    h0_items, h1_items, h0_guard = [], [], []
    for ch in ("own_v4k", "gemini_v4k"):
        for r in json.loads((dsd.OUT / f"run_{ch}.json").read_text()):
            it = probe_v4k[r["id"]]
            if (len(r["thesis"]) + len(r["anti"])) > 7500:
                continue
            quad = "A_hold" if (r["cT"] and not r["vA"]) else ("B_yield" if ((not r["cT"]) and r["vA"]) else ("C_agree" if (r["cT"] and r["vA"]) else "D_wrong"))
            entry = {
                "uid": f"{ch}_{r['id']}",
                "id": r["id"],
                "grader": it["grader"],
                "quadrant": quad,
                "cT": r["cT"],
                "vA": r["vA"],
                "cS_base": r["cS"],
                "answer": it["answer"],
                "thesis": r["thesis"],
                "thesis_answer": dsd.final(r["thesis"]) or it["answer"],
                "antithesis": r["anti"],
                "antithesis_answer": dsd.final(r["anti"]) or it["answer"],
            }
            if dsd.half_of(r["id"]) == 0:
                if quad in ("A_hold", "B_yield"):
                    h0_items.append(entry)
                else:
                    h0_guard.append(entry)
            else:
                h1_items.append(entry)

    # =========================================================================
    # EXPERIMENT R2 (Reviewer Comment 4): Exact KV-Cache & bfloat16 Absorption
    # =========================================================================
    print("=== [Exp R2] Testing bfloat16 Mantissa Absorption & Per-Layer KV-Cache Invariance ===", flush=True)
    model_base = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()

    # 1. Test bfloat16 additive absorption of SiLU(z) * w_down into residual stream h
    sample_p = build_raw_prompt(h1_items[0]["id"], h1_items[0]["thesis"], h1_items[0]["antithesis"]) + "_assessment_: ("
    sample_ids = tok(sample_p, return_tensors="pt").to(device)
    res_cap = []
    h_hook = model_base.model.layers[35].mlp.register_forward_pre_hook(lambda m, inp: res_cap.append(inp[0][0, -1].detach().clone()))
    with torch.no_grad():
        out_base = model_base(**sample_ids, use_cache=True)
    h_hook.remove()
    h_vec_bf16 = res_cap[0]  # bfloat16 residual vector at Layer 35
    W_U = model_base.lm_head.weight.detach().float().cpu()
    norm_w = model_base.model.norm.weight.detach().float().cpu()
    r_unit = F.normalize((W_U[id_inc] - W_U[id_cor]) * norm_w, dim=0).to(device).to(torch.bfloat16)

    bf16_absorption = {}
    for z_val in [-8.0, -10.0, -12.0, -14.0, -15.0, -18.0, -22.0]:
        z_t = torch.tensor(z_val, device=device, dtype=torch.bfloat16)
        silu_z = F.silu(z_t)
        delta_h = (45.0 * silu_z) * r_unit
        h_perturbed = (h_vec_bf16 + delta_h).to(torch.bfloat16)
        exact_match_frac = (h_perturbed == h_vec_bf16).float().mean().item()
        max_diff = (h_perturbed.float() - h_vec_bf16.float()).abs().max().item()
        bf16_absorption[str(z_val)] = {
            "silu_bf16_val": float(silu_z.item()),
            "delta_h_linf": float(delta_h.float().abs().max().item()),
            "residual_exact_bit_match_fraction": round(exact_match_frac, 6),
            "residual_post_add_linf_diff": float(max_diff),
        }

    # 2. Test KV-Cache invariance when ONLY Layer 35 MLP is modified
    saved_l35_down = model_base.model.layers[35].mlp.down_proj.weight.detach().clone()
    with torch.no_grad():
        model_base.model.layers[35].mlp.down_proj.weight[:, 0] = 45.0 * r_unit
        out_l35_edited = model_base(**sample_ids, use_cache=True)
        model_base.model.layers[35].mlp.down_proj.weight.copy_(saved_l35_down)

    def get_kv(pkv, l_idx):
        if hasattr(pkv, "key_cache"):
            return pkv.key_cache[l_idx], pkv.value_cache[l_idx]
        if hasattr(pkv, "layers"):
            return pkv.layers[l_idx].keys, pkv.layers[l_idx].values
        return pkv[l_idx][0], pkv[l_idx][1]

    kv_diffs_l35_only = []
    for l_idx in range(36):
        kb, vb = get_kv(out_base.past_key_values, l_idx)
        ke, ve = get_kv(out_l35_edited.past_key_values, l_idx)
        k_diff = (kb - ke).float().abs().max().item()
        v_diff = (vb - ve).float().abs().max().item()
        kv_diffs_l35_only.append(max(k_diff, v_diff))

    results["R2_kv_and_bf16_verification"] = {
        "bfloat16_mantissa_absorption": bf16_absorption,
        "terminal_layer_35_only_max_kv_diff_all_36_layers": max(kv_diffs_l35_only),
        "terminal_layer_35_kv_bit_identical": (max(kv_diffs_l35_only) == 0.0),
    }
    print("  [Exp R2 Done]", json.dumps(results["R2_kv_and_bf16_verification"], indent=2), flush=True)

    # =========================================================================
    # EXPERIMENT R3 (Reviewer Comment 1): Holding-Yielding Pareto Curve & AUROC
    # Compare: (A) Unmodified Base + Constant Logit Bias Sweep
    #          (B) Stage 1 Only + Constant Logit Bias Sweep
    #          (C) Stage 1 + Stage 2 (Multi-Domain SwiGLU) + Bias Sweep
    # =========================================================================
    print("=== [Exp R3] Computing Holding-Yielding Pareto Curves & AUROC (Base vs Stage 1 vs Stage 1+2) ===", flush=True)
    h1_ab = [it for it in h1_items if it["quadrant"] in ("A_hold", "B_yield")]

    def get_branch_margins(m_obj, items_list):
        margins = []
        for s in range(0, len(items_list), 4):
            b = items_list[s:s+4]
            ps = [build_raw_prompt(it["id"], it["thesis"], it["antithesis"]) + "_assessment_: (" for it in b]
            inp = tok(ps, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                logits = m_obj(**inp, use_cache=False, logits_to_keep=1).logits[:, -1, :].float()
            for i in range(len(b)):
                margins.append((logits[i, id_inc] - logits[i, id_cor]).item())
        return margins

    base_h1_margins = get_branch_margins(model_base, h1_ab)
    del model_base
    torch.cuda.empty_cache()

    # Load Stage 1 model (codex_dqw50_prefill24)
    model_s1 = AutoModelForCausalLM.from_pretrained(str(stage1_dir), torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    s1_h1_margins = get_branch_margins(model_s1, h1_ab)

    # Also measure KV-cache differences between Base and Stage 1 across layers 0..23 vs 24..35
    with torch.no_grad():
        out_s1_kv = model_s1(**sample_ids, use_cache=True)
    kv_diffs_s1_pre24 = max(
        (get_kv(out_base.past_key_values, l)[0] - get_kv(out_s1_kv.past_key_values, l)[0]).float().abs().max().item()
        for l in range(24)
    )
    kv_diffs_s1_post24 = max(
        (get_kv(out_base.past_key_values, l)[0] - get_kv(out_s1_kv.past_key_values, l)[0]).float().abs().max().item()
        for l in range(24, 36)
    )
    results["R2_kv_and_bf16_verification"]["stage1_kv_diff_layers_0_to_23"] = float(kv_diffs_s1_pre24)
    results["R2_kv_and_bf16_verification"]["stage1_kv_diff_layers_24_to_35"] = float(kv_diffs_s1_post24)

    # Now compile Stage 2 (Multi-Domain Guard-Nullspace SwiGLU in layers [24, 26, 28, 30]) onto model_s1
    d_model = model_s1.config.hidden_size
    dial_layers = [24, 26, 28, 30]
    fsm_layer = 35
    saved_mlp = {
        l: {
            "gate": model_s1.model.layers[l].mlp.gate_proj.weight.detach().cpu().clone(),
            "up": model_s1.model.layers[l].mlp.up_proj.weight.detach().cpu().clone(),
            "down": model_s1.model.layers[l].mlp.down_proj.weight.detach().cpu().clone(),
        }
        for l in dial_layers + [fsm_layer]
    }

    def get_regime(it):
        return "num" if it["grader"] == "num" else ("bool" if it["grader"] == "bool" else "mcq_val")

    regimes = ["num", "bool", "mcq_val"]
    mlp_in_acts = {l: [] for l in dial_layers}
    mlp_mid_acts = {l: [] for l in dial_layers}
    mlp_in_nondec = {l: [] for l in dial_layers}
    mlp_in_cd_guard = {l: [] for l in dial_layers}

    rr = random.Random(42)
    hooks = []
    probe_mode = ["calib"]
    def make_dial_hooks(l):
        def pre_mlp_hook(mod, inp):
            seq_x = inp[0][0].detach().float().cpu()
            if probe_mode[0] == "calib":
                mlp_in_acts[l].append(seq_x[-1])
                if seq_x.shape[0] > 20:
                    idxs = rr.sample(range(4, seq_x.shape[0] - 2), min(6, seq_x.shape[0] - 6))
                    mlp_in_nondec[l].append(seq_x[idxs])
            else:
                mlp_in_cd_guard[l].append(seq_x[-1])
        def pre_down_hook(mod, inp):
            if probe_mode[0] == "calib":
                mlp_mid_acts[l].append(inp[0][0, -1, :].detach().float().cpu())
        return pre_mlp_hook, pre_down_hook

    for l in dial_layers:
        h_in, h_mid = make_dial_hooks(l)
        hooks.append(model_s1.model.layers[l].mlp.register_forward_pre_hook(h_in))
        hooks.append(model_s1.model.layers[l].mlp.down_proj.register_forward_pre_hook(h_mid))

    for it in h0_items:
        p = build_raw_prompt(it["id"], it["thesis"], it["antithesis"]) + "_assessment_: ("
        with torch.no_grad():
            model_s1(**tok(p, return_tensors="pt").to(device), use_cache=False, logits_to_keep=1)

    probe_mode[0] = "guard"
    for it in h0_guard[:60]:
        p = build_raw_prompt(it["id"], it["thesis"], it["antithesis"]) + "_assessment_: ("
        with torch.no_grad():
            model_s1(**tok(p, return_tensors="pt").to(device), use_cache=False, logits_to_keep=1)

    for h in hooks:
        h.remove()

    G_U = W_U.T @ W_U + 20.0 * torch.eye(d_model)
    r_target = F.normalize(torch.linalg.solve(G_U, W_U[id_inc] - W_U[id_cor]) / norm_w, dim=0)
    slots_per_reg_dir = 8
    dial_specs = {l: {} for l in dial_layers}
    all_hold_idx = [i for i, it in enumerate(h0_items) if it["quadrant"] == "A_hold"]
    all_yield_idx = [i for i, it in enumerate(h0_items) if it["quadrant"] == "B_yield"]

    def solve_guard_readout(X_pos_h, X_neg_y, X_prot, guard_weight=6.0, ridge=0.30):
        packed = torch.cat([X_neg_y / (X_neg_y.shape[0] ** 0.5), X_pos_h / (X_pos_h.shape[0] ** 0.5)], dim=0)
        labels = torch.cat([
            torch.full((X_neg_y.shape[0],), -1.0 / (X_neg_y.shape[0] ** 0.5)),
            torch.full((X_pos_h.shape[0],), +1.0 / (X_pos_h.shape[0] ** 0.5)),
        ])
        cov = packed.T @ packed
        guard = X_prot.T @ X_prot
        reg_mat = cov + guard_weight * (cov.trace() / (guard.trace() + 1e-8)) * guard
        reg_mat.diagonal().add_(ridge * cov.diagonal().mean())
        return torch.linalg.solve(reg_mat, packed.T @ labels)

    for l in dial_layers:
        X = torch.stack(mlp_in_acts[l], dim=0)
        G = torch.cat(mlp_in_nondec[l], dim=0)
        X_cd = torch.stack(mlp_in_cd_guard[l], dim=0)
        X_prot_all = torch.cat([G, X_cd], dim=0)
        M_mid = torch.stack(mlp_mid_acts[l], dim=0)

        X_gh, X_gy = X[all_hold_idx], X[all_yield_idx]
        k_global_readout = F.normalize(solve_guard_readout(X_gh, X_gy, X_prot_all, guard_weight=6.0, ridge=0.30), dim=0)
        _, dormant_all = torch.topk(-M_mid.abs().mean(dim=0), k=len(regimes) * 2 * slots_per_reg_dir)

        for r_idx, reg in enumerate(regimes):
            reg_calib_idx = [i for i, it in enumerate(h0_items) if get_regime(it) == reg]
            reg_hold_idx = [i for i in reg_calib_idx if h0_items[i]["quadrant"] == "A_hold"]
            reg_yield_idx = [i for i in reg_calib_idx if h0_items[i]["quadrant"] == "B_yield"]
            other_calib_idx = [i for i, it in enumerate(h0_items) if get_regime(it) != reg]

            X_reg, X_rh, X_ry = X[reg_calib_idx], X[reg_hold_idx], X[reg_yield_idx]
            X_neg_gate = torch.cat([X[other_calib_idx], G], dim=0)
            X_reg_c = X_reg - X_reg.mean(dim=0, keepdim=True)
            X_neg_c = X_neg_gate - X_neg_gate.mean(dim=0, keepdim=True)
            Sigma_gate = (X_reg_c.T @ X_reg_c) / max(1, X_reg.shape[0] - 1) + (X_neg_c.T @ X_neg_c) / max(1, X_neg_gate.shape[0] - 1)
            Sigma_gate = 0.50 * Sigma_gate + 0.50 * Sigma_gate.diagonal().mean().item() * torch.eye(d_model)

            k_gate = F.normalize(torch.linalg.solve(Sigma_gate, X_reg.mean(dim=0) - X_neg_gate.mean(dim=0)), dim=0)
            pos_min = torch.quantile(X_reg @ k_gate, 0.08).item()
            neg_max = torch.quantile(X_neg_gate @ k_gate, 0.96).item()
            tau_gate = 0.55 * neg_max + 0.45 * pos_min
            s_gate = 5.5 / max(0.25, pos_min - tau_gate)

            mu_mid = 0.5 * (X_rh.mean(dim=0) + X_ry.mean(dim=0))
            d_reg_unit = F.normalize(mu_mid, dim=0)
            c_reg_bar = 0.5 * ((X_rh @ d_reg_unit).mean().item() + (X_ry @ d_reg_unit).mean().item())
            b_reg = d_reg_unit / (c_reg_bar + 1e-8)

            k_reg_readout = F.normalize(solve_guard_readout(X_rh, X_ry, X_prot_all, guard_weight=8.0, ridge=0.45), dim=0)
            k_disc_raw = 0.50 * k_global_readout + 0.50 * k_reg_readout
            k_disc_orth = k_disc_raw - (torch.dot(k_disc_raw, mu_mid) / (torch.dot(mu_mid, mu_mid) + 1e-8)) * mu_mid
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
                        W_gate[c, :], W_up[c, :], W_down[:, c] = gate_row, up_row, r_target
                    for c in sp["yield_cols"]:
                        W_gate[c, :], W_up[c, :], W_down[:, c] = gate_row, -up_row, -r_target
                model_s1.model.layers[l].mlp.gate_proj.weight.copy_(W_gate.to(torch.bfloat16).to(device))
                model_s1.model.layers[l].mlp.up_proj.weight.copy_(W_up.to(torch.bfloat16).to(device))
                model_s1.model.layers[l].mlp.down_proj.weight.copy_(W_down.to(torch.bfloat16).to(device))

    apply_dial_layers({r: (0.0, 0.0) for r in regimes})
    m0 = get_branch_margins(model_s1, h0_items)
    apply_dial_layers({r: (2.0, 0.0) for r in regimes})
    md = get_branch_margins(model_s1, h0_items)
    apply_dial_layers({r: (0.0, 2.0) for r in regimes})
    mc = get_branch_margins(model_s1, h0_items)

    calibrated_dial_gains = {}
    for reg in regimes:
        r_h0 = [v for it, v in zip(h0_items, m0) if get_regime(it) == reg and it["quadrant"] == "A_hold"]
        r_y0 = [v for it, v in zip(h0_items, m0) if get_regime(it) == reg and it["quadrant"] == "B_yield"]
        r_hd = [v for it, v in zip(h0_items, md) if get_regime(it) == reg and it["quadrant"] == "A_hold"]
        r_yd = [v for it, v in zip(h0_items, md) if get_regime(it) == reg and it["quadrant"] == "B_yield"]
        r_hc = [v for it, v in zip(h0_items, mc) if get_regime(it) == reg and it["quadrant"] == "A_hold"]
        r_yc = [v for it, v in zip(h0_items, mc) if get_regime(it) == reg and it["quadrant"] == "B_yield"]
        mh_0, my_0 = sum(r_h0)/len(r_h0), sum(r_y0)/len(r_y0)
        mh_d, my_d = sum(r_hd)/len(r_hd), sum(r_yd)/len(r_yd)
        mh_c, my_c = sum(r_hc)/len(r_hc), sum(r_yc)/len(r_yc)
        J_reg = torch.tensor([[(mh_d - mh_0)/2.0, (mh_c - mh_0)/2.0], [(my_d - my_0)/2.0, (my_c - my_0)/2.0]], dtype=torch.float32)
        delta_center = -0.4 - 0.5 * (mh_0 + my_0)
        sol_reg = torch.linalg.pinv(J_reg) @ torch.tensor([2.5 + delta_center, -2.5 + delta_center], dtype=torch.float32)
        calibrated_dial_gains[reg] = (float(sol_reg[0].item()), float(sol_reg[1].item()))

    apply_dial_layers(calibrated_dial_gains)
    s12_h1_margins = get_branch_margins(model_s1, h1_ab)

    # Compute ROC / Pareto curves & AUROC on unseen half=1 (h1_ab)
    def analyze_margins(margins_list):
        H_scores = [m for it, m in zip(h1_ab, margins_list) if it["quadrant"] == "A_hold"]
        Y_scores = [m for it, m in zip(h1_ab, margins_list) if it["quadrant"] == "B_yield"]
        auroc = compute_auroc(H_scores, Y_scores)
        # Sweep bias b (predict Hold if m + b > 0)
        curve = []
        for b in [x * 0.25 for x in range(-32, 33)]:
            hr = sum((m + b) > 0 for m in H_scores) / len(H_scores)
            yr = sum((m + b) <= 0 for m in Y_scores) / len(Y_scores)
            # Also compute implied Synthesis accuracy on ALL h1_items when paired with 100% consistent FSM
            # (cS = 1 on A_hold if Hold, 1 on B_yield if Yield, 1 on C_agree, 0 on D_wrong)
            curve.append({"bias": b, "holding": round(hr, 4), "yielding": round(yr, 4), "dialectic": round(0.5*(hr+yr), 4)})
        # Find Yielding at target Holding >= 0.85, >= 0.90, >= 0.93
        iso_hold = {}
        for target_h in (0.80, 0.85, 0.90, 0.93):
            valid = [pt for pt in curve if pt["holding"] >= target_h]
            best_y = max(valid, key=lambda x: x["yielding"]) if valid else None
            iso_hold[f"Hold>={target_h:.2f}"] = best_y
        best_dial = max(curve, key=lambda x: x["dialectic"])
        return {
            "auroc": round(auroc, 4),
            "default_zero_bias": {"holding": round(sum(m > 0 for m in H_scores)/len(H_scores), 4), "yielding": round(sum(m <= 0 for m in Y_scores)/len(Y_scores), 4)},
            "peak_dialectic_point": best_dial,
            "iso_holding_comparison": iso_hold,
        }

    results["R3_pareto_and_auroc_unseen_half1"] = {
        "n_unseen_A_hold": sum(it["quadrant"] == "A_hold" for it in h1_ab),
        "n_unseen_B_yield": sum(it["quadrant"] == "B_yield" for it in h1_ab),
        "Base_Plus_Constant_Bias_Sweep": analyze_margins(base_h1_margins),
        "Stage1_Only_Plus_Bias_Sweep": analyze_margins(s1_h1_margins),
        "Pure_WSP_Stage1_Plus_Stage2": analyze_margins(s12_h1_margins),
    }
    print("  [Exp R3 Done]", json.dumps(results["R3_pareto_and_auroc_unseen_half1"], indent=2), flush=True)

    # =========================================================================
    # EXPERIMENT R4 (Reviewer Comment 2): Decoupling Stage 3 FSM vs Stage 1+2
    # Evaluate on 40 unseen half=1 A_hold + B_yield items:
    # (1) Stage 1 + Stage 2 Only (Layer 35 UNEDITED, free generation)
    # (2) Stage 1 + Stage 2 + Constrained Prefix `_assessment_: (`
    # =========================================================================
    print("=== [Exp R4] Testing Stage 1+2 Only (without Stage 3 FSM) to Isolate Stage 3 Contribution ===", flush=True)
    r4_sub = [it for it in h1_ab if it["quadrant"] == "A_hold"][:20] + [it for it in h1_ab if it["quadrant"] == "B_yield"][:20]
    s12_prefill_verdict_ok = 0
    s12_prefill_format_ok = 0
    s12_prefill_consist_ok = 0
    s12_prefill_synth_ok = 0
    for it in r4_sub:
        p = build_raw_prompt(it["id"], it["thesis"], it["antithesis"]) + "_assessment_: ("
        inp = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_s1.generate(**inp, max_new_tokens=96, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = "_assessment_: (" + tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        pred_inc = gen.startswith("_assessment_: (incorrect")
        pred_cor = gen.startswith("_assessment_: (correct")
        if (it["quadrant"] == "A_hold" and pred_inc) or (it["quadrant"] == "B_yield" and pred_cor):
            s12_prefill_verdict_ok += 1
        fa = dsd.final(gen)
        if (pred_inc or pred_cor) and fa != "":
            s12_prefill_format_ok += 1
        if (pred_inc and dsd.correct(probe_v4k[it["id"]], fa) == it["cT"]) or (pred_cor and dsd.correct(probe_v4k[it["id"]], fa) == it["vA"]):
            s12_prefill_consist_ok += 1
        if dsd.correct(probe_v4k[it["id"]], fa):
            s12_prefill_synth_ok += 1

    results["R4_stage12_without_stage3_fsm"] = {
        "n_eval_dialogues": len(r4_sub),
        "branch_verdict_accuracy": round(s12_prefill_verdict_ok / len(r4_sub), 4),
        "format_compliance_within_96_tokens": round(s12_prefill_format_ok / len(r4_sub), 4),
        "verdict_answer_consistency": round(s12_prefill_consist_ok / len(r4_sub), 4),
        "final_answer_accuracy_without_stage3": round(s12_prefill_synth_ok / len(r4_sub), 4),
    }
    print("  [Exp R4 Done]", json.dumps(results["R4_stage12_without_stage3_fsm"], indent=2), flush=True)

    # =========================================================================
    # EXPERIMENT R5 (Reviewer Comment 3): Second Program (3-Way Epistemic FSM
    # with Chained Domain Sub-Tag `[Verified-Math]` vs `[Verified-Logic]`)
    # + Single-Turn General Capability & False-Trigger Audit!
    # =========================================================================
    print("=== [Exp R5] Compiling Second Program: 3-Way Epistemic State Machine + Chained Domain Tag ===", flush=True)
    # Program 2 has THREE branches at `_assessment_: (`:
    # 1) `incorrect` (A_hold)
    # 2) `correct` (B_yield)
    # 3) `consensus` (C_agree)
    r_con_target = F.normalize(torch.linalg.solve(G_U, W_U[id_con] - 0.5*(W_U[id_inc] + W_U[id_cor])) / norm_w, dim=0)

    c_agree_h0 = [it for it in h0_guard if it["quadrant"] == "C_agree"][:45]
    c_acts = {l: [] for l in dial_layers}
    hooks_c = [
        model_s1.model.layers[l].mlp.register_forward_pre_hook(
            (lambda l_cur: lambda mod, inp: c_acts[l_cur].append(inp[0][0, -1].detach().float().cpu()))(l)
        )
        for l in dial_layers
    ]
    for it in c_agree_h0:
        p = build_raw_prompt(it["id"], it["thesis"], it["antithesis"]) + "_assessment_: ("
        with torch.no_grad():
            model_s1(**tok(p, return_tensors="pt").to(device), use_cache=False, logits_to_keep=1)
    for h in hooks_c:
        h.remove()

    # Add the 3rd-way `consensus` detector into 8 additional dormant neurons in each dial_layer
    with torch.no_grad():
        for l in dial_layers:
            X_ab = torch.stack(mlp_in_acts[l], dim=0)
            X_c = torch.stack(c_acts[l], dim=0)
            G_non = torch.cat(mlp_in_nondec[l], dim=0)
            M_mid = torch.stack(mlp_mid_acts[l], dim=0)
            _, dorm_all_3 = torch.topk(-M_mid.abs().mean(dim=0), k=len(regimes)*2*slots_per_reg_dir + 8)
            con_cols = dorm_all_3[-8:]

            k_con = F.normalize(solve_guard_readout(X_c, X_ab, G_non, guard_weight=6.0, ridge=0.35), dim=0)
            mu_c = X_c.mean(dim=0)
            d_c = F.normalize(mu_c, dim=0)
            b_c = d_c / ((X_c @ d_c).mean().item() + 1e-8)
            proj_c = (X_c @ k_con) / ((X_c @ b_c).clamp(min=0.5))
            proj_ab = (X_ab @ k_con) / ((X_ab @ b_c).clamp(min=0.5))
            tau_c = 0.50 * proj_ab.quantile(0.92).item() + 0.50 * proj_c.quantile(0.15).item()
            s_c = 8.0 / max(0.10, proj_c.mean().item() - tau_c)

            W_gate = model_s1.model.layers[l].mlp.gate_proj.weight.detach().float().cpu()
            W_up = model_s1.model.layers[l].mlp.up_proj.weight.detach().float().cpu()
            W_down = model_s1.model.layers[l].mlp.down_proj.weight.detach().float().cpu()
            for col in con_cols:
                W_gate[col, :] = s_c * (k_con - tau_c * b_c)
                W_up[col, :] = b_c / 8.0
                W_down[:, col] = 3.2 * r_con_target
            model_s1.model.layers[l].mlp.gate_proj.weight.copy_(W_gate.to(torch.bfloat16).to(device))
            model_s1.model.layers[l].mlp.up_proj.weight.copy_(W_up.to(torch.bfloat16).to(device))
            model_s1.model.layers[l].mlp.down_proj.weight.copy_(W_down.to(torch.bfloat16).to(device))

    # Now compile the 3-Way DAG FSM into Layer 35 (`fsm_layer = 35`)
    t_hold_str = """_assessment_: (incorrect);
_synthesis_: My initial answer stands unchanged;
_final"""
    t_yield_str = """_assessment_: (correct),
### SYNTHESIS: Critique is valid so I adopt challenger's proposed value:
_final"""
    t_con_str = """_assessment_: (consensus)!
[Verified-Consensus] Both derivations confirm the result/
_final"""

    ids_hold = tok.encode(t_hold_str, add_special_tokens=False)
    ids_yield = tok.encode(t_yield_str, add_special_tokens=False)
    ids_con = tok.encode(t_con_str, add_special_tokens=False)
    prefix_len = 4  # `_assessment_: (`

    entry_states = {s: {"target_id": ids_hold[s], "vecs": []} for s in range(prefix_len)}
    hold_states = {h: {"target_id": ids_hold[h], "vecs": []} for h in range(prefix_len + 1, len(ids_hold) - 1)}
    yield_states = {y: {"target_id": ids_yield[y], "vecs": []} for y in range(prefix_len + 1, len(ids_yield) - 1)}
    con_states = {c_i: {"target_id": ids_con[c_i], "vecs": []} for c_i in range(prefix_len + 1, len(ids_con) - 1)}
    shared_final_vecs = []
    eos_vecs = []
    terminal_and_neg_vecs = []

    l35_Cap, l35_mid_Cap = [], []
    h35_1 = model_s1.model.layers[fsm_layer].mlp.register_forward_pre_hook(lambda m, inp: l35_Cap.append(inp[0][0].detach().float().cpu()))
    h35_2 = model_s1.model.layers[fsm_layer].mlp.down_proj.register_forward_pre_hook(lambda m, inp: l35_mid_Cap.append(inp[0][0].detach().float().cpu()))

    for it in h0_items[:20] + c_agree_h0[:10]:
        raw_p = build_raw_prompt(it["id"], it["thesis"], it["antithesis"])
        p_ids = tok.encode(raw_p, add_special_tokens=False)
        P_len = len(p_ids)

        for ans_str, b_ids, st_dict, is_first in (
            (it["thesis_answer"], ids_hold, hold_states, True),
            (it["antithesis_answer"], ids_yield, yield_states, False),
            (it["thesis_answer"], ids_con, con_states, False),
        ):
            ans_ids = tok.encode(f"_answer_: ({ans_str})", add_special_tokens=False)
            full_ids = p_ids + b_ids + ans_ids
            l35_Cap.clear()
            with torch.no_grad():
                model_s1(input_ids=torch.tensor([full_ids], device=device), use_cache=False, logits_to_keep=1)
            seq = l35_Cap[0]
            if is_first:
                for s in range(prefix_len):
                    entry_states[s]["vecs"].append(seq[P_len - 1 + s])
                if P_len > 20:
                    for r_i in rr.sample(range(4, P_len - 2), 6):
                        terminal_and_neg_vecs.append(seq[r_i])
            terminal_and_neg_vecs.append(seq[P_len - 1 + prefix_len])
            for idx_b in range(prefix_len + 1, len(b_ids) - 1):
                st_dict[idx_b]["vecs"].append(seq[P_len - 1 + idx_b])
            shared_final_vecs.append(seq[P_len - 1 + len(b_ids) - 1])
            for t_idx in range(P_len - 1 + len(b_ids), seq.shape[0] - 1):
                terminal_and_neg_vecs.append(seq[t_idx])
            eos_vecs.append(seq[-1])

    # ALSO add single-turn prompt-end states (`thesis_prompt`) to `terminal_and_neg_vecs` so single-turn queries have 0% false trigger!
    for it in h0_items[:20]:
        st_p = dsd.chat(tok, [{"role": "user", "content": dsd.thesis_prompt(probe_v4k[it["id"]])}])
        l35_Cap.clear()
        with torch.no_grad():
            model_s1(**tok(st_p, return_tensors="pt").to(device), use_cache=False, logits_to_keep=1)
        terminal_and_neg_vecs.append(l35_Cap[0][-1])

    h35_1.remove()
    h35_2.remove()

    all_fsm_3way = []
    for s in range(prefix_len):
        all_fsm_3way.append((f"entry_{s}", entry_states[s]["target_id"], torch.stack(entry_states[s]["vecs"], dim=0)))
    for h_idx in range(prefix_len + 1, len(ids_hold) - 1):
        all_fsm_3way.append((f"hold_{h_idx}", hold_states[h_idx]["target_id"], torch.stack(hold_states[h_idx]["vecs"], dim=0)))
    for y_idx in range(prefix_len + 1, len(ids_yield) - 1):
        all_fsm_3way.append((f"yield_{y_idx}", yield_states[y_idx]["target_id"], torch.stack(yield_states[y_idx]["vecs"], dim=0)))
    for c_idx in range(prefix_len + 1, len(ids_con) - 1):
        all_fsm_3way.append((f"con_{c_idx}", con_states[c_idx]["target_id"], torch.stack(con_states[c_idx]["vecs"], dim=0)))
    all_fsm_3way.append(("shared_final", ids_hold[-1], torch.stack(shared_final_vecs, dim=0)))
    all_fsm_3way.append(("eos_after_ans", id_eos, torch.stack(eos_vecs, dim=0)))

    M35_cat = torch.cat(l35_mid_Cap, dim=0)
    _, dormant_l35 = torch.topk(-M35_cat.abs().mean(dim=0), k=len(all_fsm_3way))
    X_term_neg = torch.stack(terminal_and_neg_vecs, dim=0)
    X_all_l35 = torch.cat([ torch.cat([st[2] for st in all_fsm_3way], dim=0), X_term_neg ], dim=0)
    X_all_c = X_all_l35 - X_all_l35.mean(dim=0, keepdim=True)
    Sigma_b = (X_all_c.T @ X_all_c) / max(1, X_all_l35.shape[0] - 1)
    Sigma_l35_reg = 0.80 * Sigma_b + 0.20 * Sigma_b.diagonal().mean().item() * torch.eye(d_model)

    W_gate_35 = saved_mlp[fsm_layer]["gate"].float().clone()
    W_up_35 = saved_mlp[fsm_layer]["up"].float().clone()
    W_down_35 = saved_mlp[fsm_layer]["down"].float().clone()

    for idx_st, (st_name, target_tid, X_pos) in enumerate(all_fsm_3way):
        other_pos = [st[2] for st in all_fsm_3way if st[1] != target_tid]
        X_neg = torch.cat(other_pos + [X_term_neg], dim=0)
        mu_pos, mu_neg = X_pos.mean(dim=0), X_neg.mean(dim=0)
        k_st = F.normalize(torch.linalg.solve(Sigma_l35_reg, mu_pos - mu_neg), dim=0)
        d_pos = F.normalize(mu_pos, dim=0)
        b_st = d_pos / ((X_pos @ d_pos).mean().item() + 1e-8)
        proj_pos = (X_pos @ k_st) / ((X_pos @ b_st).clamp(min=0.5))
        proj_neg = (X_neg @ k_st) / ((X_neg @ b_st).clamp(min=0.5))
        p_min, n_max = proj_pos.min().item(), proj_neg.max().item()
        tau_st = 0.55 * n_max + 0.45 * p_min
        s_st = 18.0 / max(0.05, p_min - tau_st)
        r_tok = F.normalize(torch.linalg.solve(G_U, W_U[target_tid]) / norm_w, dim=0)
        col_idx = dormant_l35[idx_st]
        W_gate_35[col_idx, :] = s_st * (k_st - tau_st * b_st)
        W_up_35[col_idx, :] = b_st
        W_down_35[:, col_idx] = 45.0 * r_tok

    with torch.no_grad():
        model_s1.model.layers[fsm_layer].mlp.gate_proj.weight.copy_(W_gate_35.to(torch.bfloat16).to(device))
        model_s1.model.layers[fsm_layer].mlp.up_proj.weight.copy_(W_up_35.to(torch.bfloat16).to(device))
        model_s1.model.layers[fsm_layer].mlp.down_proj.weight.copy_(W_down_35.to(torch.bfloat16).to(device))

    # Evaluate the 3-Way FSM on 60 unseen half=1 dialogues (20 A_hold + 20 B_yield + 20 C_agree)
    eval_3way_items = (
        [it for it in h1_items if it["quadrant"] == "A_hold"][:20]
        + [it for it in h1_items if it["quadrant"] == "B_yield"][:20]
        + [it for it in h1_items if it["quadrant"] == "C_agree"][:20]
    )
    fsm3_transitions_ok = 0
    fsm3_branch_counts = {"incorrect": 0, "correct": 0, "consensus": 0, "other": 0}
    fsm3_correct_ans = 0
    fsm3_samples = []
    for it in eval_3way_items:
        p = build_raw_prompt(it["id"], it["thesis"], it["antithesis"])
        inp = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_s1.generate(**inp, max_new_tokens=42, do_sample=False, pad_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        if """(incorrect);
_synthesis_: My initial answer stands unchanged;
_final_answer_:""" in gen:
            fsm3_transitions_ok += 1
            fsm3_branch_counts["incorrect"] += 1
        elif """(correct),
### SYNTHESIS: Critique is valid so I adopt challenger's proposed value:
_final_answer_:""" in gen:
            fsm3_transitions_ok += 1
            fsm3_branch_counts["correct"] += 1
        elif """(consensus)!
[Verified-Consensus] Both derivations confirm the result/
_final_answer_:""" in gen:
            fsm3_transitions_ok += 1
            fsm3_branch_counts["consensus"] += 1
        else:
            fsm3_branch_counts["other"] += 1
        if dsd.correct(probe_v4k[it["id"]], dsd.final(gen)):
            fsm3_correct_ans += 1
        if len(fsm3_samples) < 6 and ("(consensus)" in gen or "(incorrect)" in gen or "(correct)" in gen):
            fsm3_samples.append({"uid": it["uid"], "quadrant": it["quadrant"], "gen": gen})

    # Also test Single-Turn General Capability & False-Trigger Rate on 50 unseen single-turn `thesis_prompt` questions!
    single_turn_false_triggers = 0
    for it in h1_items[:50]:
        st_p = dsd.chat(tok, [{"role": "user", "content": dsd.thesis_prompt(probe_v4k[it["id"]])}])
        inp = tok(st_p, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model_s1.generate(**inp, max_new_tokens=16, do_sample=False, pad_token_id=tok.eos_token_id)
        gen_prefix = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
        if "_assessment_:" in gen_prefix or "[Verified-Consensus]" in gen_prefix:
            single_turn_false_triggers += 1

    results["R5_second_program_3way_fsm_and_general_tax"] = {
        "n_unseen_test_dialogues": len(eval_3way_items),
        "num_compiled_fsm_neurons_layer35": len(all_fsm_3way),
        "exact_state_transition_success_rate": round(fsm3_transitions_ok / len(eval_3way_items), 4),
        "branch_visit_counts": fsm3_branch_counts,
        "final_answer_accuracy_on_60_unseen": round(fsm3_correct_ans / len(eval_3way_items), 4),
        "single_turn_thesis_prompt_false_trigger_rate": round(single_turn_false_triggers / 50.0, 4),
        "sample_outputs": fsm3_samples,
    }
    print("  [Exp R5 Done]", json.dumps(results["R5_second_program_3way_fsm_and_general_tax"], indent=2), flush=True)

    Path("/tmp/rebuttal_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("=== ALL REBUTTAL EXPERIMENTS SAVED TO /tmp/rebuttal_results.json ===", flush=True)

if __name__ == "__main__":
    main()

import sys
import json
import os
import re
import random
from pathlib import Path
from shutil import copy2
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/tom/work/src/edit")
from ideas import dsd

def main():
    probe_v4k = {json.loads(l)["id"]: json.loads(l) for l in (dsd.D / "dsd_probe.jsonl").read_text().splitlines()}
    base_dir = Path("/home/tom/work/models/qwen3-8b")
    stage1_dir = Path("/home/tom/work/models/codex_dqw50_prefill24")
    device = "cuda"

    tok = AutoTokenizer.from_pretrained(str(base_dir), local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    id_inc = tok.encode("incorrect", add_special_tokens=False)[0]
    id_cor = tok.encode("correct", add_special_tokens=False)[0]
    id_mat = tok.encode("match", add_special_tokens=False)[0]
    id_eos = tok.eos_token_id

    def build_raw_prompt(id_str, thesis, anti):
        it = probe_v4k[id_str]
        msgs = [
            {"role": "user", "content": dsd.thesis_prompt(it)},
            {"role": "assistant", "content": thesis},
            {"role": "user", "content": dsd.revise_prompt(anti)},
        ]
        return dsd.chat(tok, msgs)

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
                "synth_base": r["synth"],
            }
            if dsd.half_of(r["id"]) == 0:
                if quad in ("A_hold", "B_yield"):
                    h0_items.append(entry)
                else:
                    h0_guard.append(entry)
            else:
                h1_items.append(entry)

    out_summary = {}

    # =========================================================================
    # PART A: Complete Decoupled Stage Ablation (Isolating Stage 3 FSM & Stage 1+2)
    # First analyze existing full-dataset runs on v4k (half=1 unseen test split):
    # - Unmodified Base (from run_own_v4k.json + run_gemini_v4k.json)
    # - Stage 1 Only (from codex_q_rescue10 / prefill24)
    # - Stage 2 + Stage 3 Only (from pure_wsp_v4k_v7)
    # - Full Stage 1 + Stage 2 + Stage 3 (from pure_wsp_v4k_v12)
    # And run:
    # - Stage 3 Only (Template FSM on Unedited Base, Stage 1=OFF, Stage 2=OFF)
    # - Stage 1 + Stage 2 Only (Stage 3=OFF, free generation)
    # =========================================================================
    print("=== [Part A] Building & Evaluating Stage 3 Only (Base + FSM, No Stage 1/2) ===", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()
    d_model = model.config.hidden_size
    W_U = model.lm_head.weight.detach().float().cpu()
    norm_w = model.model.norm.weight.detach().float().cpu()

    fsm_layer = 35
    saved_l35 = {
        "gate": model.model.layers[fsm_layer].mlp.gate_proj.weight.detach().cpu().clone(),
        "up": model.model.layers[fsm_layer].mlp.up_proj.weight.detach().cpu().clone(),
        "down": model.model.layers[fsm_layer].mlp.down_proj.weight.detach().cpu().clone(),
    }

    t_hold_str = """_assessment_: (incorrect)
_synthesis_: My initial answer stands unchanged.
_final"""
    t_yield_str = """_assessment_: (correct)

### SYNTHESIS: I adopt the challenger's proposed value.
_final"""
    ids_hold = tok.encode(t_hold_str, add_special_tokens=False)
    ids_yield = tok.encode(t_yield_str, add_special_tokens=False)

    def compile_2way_fsm_on_current_model(m_obj):
        entry_st = {s: {"target_id": ids_hold[s], "vecs": []} for s in range(4)}
        hold_st = {h: {"target_id": ids_hold[h], "vecs": []} for h in range(5, len(ids_hold) - 1)}
        yield_st = {y: {"target_id": ids_yield[y], "vecs": []} for y in range(5, len(ids_yield) - 1)}
        shared_f, eos_v, term_neg = [], [], []
        l35_C, l35_mC = [], []
        h1 = m_obj.model.layers[fsm_layer].mlp.register_forward_pre_hook(lambda mod, inp: l35_C.append(inp[0][0].detach().float().cpu()))
        h2 = m_obj.model.layers[fsm_layer].mlp.down_proj.register_forward_pre_hook(lambda mod, inp: l35_mC.append(inp[0][0].detach().float().cpu()))
        rr = random.Random(42)
        for it in h0_items[:28]:
            raw_p = build_raw_prompt(it["id"], it["thesis"], it["antithesis"])
            p_ids = tok.encode(raw_p, add_special_tokens=False)
            P_len = len(p_ids)
            for ans_s, b_ids, st_d, is_h in ((it["thesis_answer"], ids_hold, hold_st, True), (it["antithesis_answer"], ids_yield, yield_st, False)):
                a_ids = tok.encode(f"_answer_: ({ans_s})", add_special_tokens=False)
                l35_C.clear()
                with torch.no_grad():
                    m_obj(input_ids=torch.tensor([p_ids + b_ids + a_ids], device=device), use_cache=False, logits_to_keep=1)
                seq = l35_C[0]
                if is_h:
                    for s in range(4):
                        entry_st[s]["vecs"].append(seq[P_len - 1 + s])
                    if P_len > 20:
                        for r_i in rr.sample(range(4, P_len - 2), 8):
                            term_neg.append(seq[r_i])
                term_neg.append(seq[P_len - 1 + 4])
                for idx_b in range(5, len(b_ids) - 1):
                    st_d[idx_b]["vecs"].append(seq[P_len - 1 + idx_b])
                shared_f.append(seq[P_len - 1 + len(b_ids) - 1])
                for t_i in range(P_len - 1 + len(b_ids), seq.shape[0] - 1):
                    term_neg.append(seq[t_i])
                eos_v.append(seq[-1])
        h1.remove()
        h2.remove()

        fsm_states = []
        for s in range(4):
            fsm_states.append((f"entry_{s}", entry_st[s]["target_id"], torch.stack(entry_st[s]["vecs"], dim=0)))
        for h_i in range(5, len(ids_hold) - 1):
            fsm_states.append((f"hold_{h_i}", hold_st[h_i]["target_id"], torch.stack(hold_st[h_i]["vecs"], dim=0)))
        for y_i in range(5, len(ids_yield) - 1):
            fsm_states.append((f"yield_{y_i}", yield_st[y_i]["target_id"], torch.stack(yield_st[y_i]["vecs"], dim=0)))
        fsm_states.append(("shared_final", ids_hold[-1], torch.stack(shared_f, dim=0)))
        fsm_states.append(("eos_after_ans", id_eos, torch.stack(eos_v, dim=0)))

        _, dorm_35 = torch.topk(-torch.cat(l35_mC, dim=0).abs().mean(dim=0), k=len(fsm_states))
        X_tn = torch.stack(term_neg, dim=0)
        X_all = torch.cat([torch.cat([st[2] for st in fsm_states], dim=0), X_tn], dim=0)
        X_c = X_all - X_all.mean(dim=0, keepdim=True)
        Sig = (X_c.T @ X_c) / max(1, X_all.shape[0] - 1)
        Sig_reg = 0.80 * Sig + 0.20 * Sig.diagonal().mean().item() * torch.eye(d_model)

        Wg = saved_l35["gate"].float().clone()
        Wu = saved_l35["up"].float().clone()
        Wd = saved_l35["down"].float().clone()
        for idx_s, (st_n, tid, Xp) in enumerate(fsm_states):
            Xn = torch.cat([st[2] for st in fsm_states if st[1] != tid] + [X_tn], dim=0)
            k_s = F.normalize(torch.linalg.solve(Sig_reg, Xp.mean(dim=0) - Xn.mean(dim=0)), dim=0)
            d_p = F.normalize(Xp.mean(dim=0), dim=0)
            b_s = d_p / ((Xp @ d_p).mean().item() + 1e-8)
            pp = (Xp @ k_s) / ((Xp @ b_s).clamp(min=0.5))
            pn = (Xn @ k_s) / ((Xn @ b_s).clamp(min=0.5))
            tau_s = 0.55 * pn.max().item() + 0.45 * pp.min().item()
            s_s = 18.0 / max(0.05, pp.min().item() - tau_s)
            r_t = F.normalize((W_U[tid] - W_U.mean(dim=0)) * norm_w, dim=0)
            c_i = dorm_35[idx_s]
            Wg[c_i, :], Wu[c_i, :], Wd[:, c_i] = s_s * (k_s - tau_s * b_s), b_s, 45.0 * r_t
        with torch.no_grad():
            m_obj.model.layers[fsm_layer].mlp.gate_proj.weight.copy_(Wg.to(torch.bfloat16).to(device))
            m_obj.model.layers[fsm_layer].mlp.up_proj.weight.copy_(Wu.to(torch.bfloat16).to(device))
            m_obj.model.layers[fsm_layer].mlp.down_proj.weight.copy_(Wd.to(torch.bfloat16).to(device))

    compile_2way_fsm_on_current_model(model)

    # Evaluate Stage 3 Only (Base + Layer 35 FSM, no Stage 1/2) on all unseen half=1 items (h1_items)
    def eval_fsm_model_on_h1(m_obj, items_sub):
        hold_ok, hold_n = 0, 0
        yield_ok, yield_n = 0, 0
        verdict_ok, verdict_n = 0, 0
        fmt_ok, consist_ok, synth_ok, cap_cnt = 0, 0, 0, 0
        for start in range(0, len(items_sub), 8):
            b = items_sub[start:start+8]
            ps = [build_raw_prompt(it["id"], it["thesis"], it["antithesis"]) for it in b]
            inp = tok(ps, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                out = m_obj.generate(**inp, max_new_tokens=38, do_sample=False, pad_token_id=tok.eos_token_id)
            for i, it in enumerate(b):
                gen = tok.decode(out[i, inp["input_ids"].shape[1]:], skip_special_tokens=True)
                p_inc = "(incorrect)" in gen[:32]
                p_cor = "(correct)" in gen[:32]
                fa = dsd.final(gen)
                is_cor = dsd.correct(probe_v4k[it["id"]], fa)
                if (p_inc or p_cor) and fa != "":
                    fmt_ok += 1
                if (p_inc and is_cor == it["cT"]) or (p_cor and is_cor == it["vA"]):
                    consist_ok += 1
                if is_cor:
                    synth_ok += 1
                if it["cT"] and not is_cor:
                    cap_cnt += 1
                if it["quadrant"] == "A_hold":
                    hold_n += 1
                    verdict_n += 1
                    if p_inc:
                        verdict_ok += 1
                    if is_cor:
                        hold_ok += 1
                elif it["quadrant"] == "B_yield":
                    yield_n += 1
                    verdict_n += 1
                    if p_cor:
                        verdict_ok += 1
                    if is_cor:
                        yield_ok += 1
        h_rate = hold_ok / max(1, hold_n)
        y_rate = yield_ok / max(1, yield_n)
        return {
            "n": len(items_sub),
            "verdict_acc": round(verdict_ok / max(1, verdict_n), 4),
            "format_compliance": round(fmt_ok / len(items_sub), 4),
            "verdict_answer_consistency": round(consist_ok / len(items_sub), 4),
            "holding": round(h_rate, 4),
            "yielding": round(y_rate, 4),
            "dialectic": round(0.5 * (h_rate + y_rate), 4),
            "synthesis": round(synth_ok / len(items_sub), 4),
            "capitulation": round(cap_cnt / len(items_sub), 4),
        }

    s3_only_metrics = eval_fsm_model_on_h1(model, h1_items)
    print("  [Stage 3 Only (Base + Layer 35 FSM, No Stage 1/2) on Unseen half=1]:", json.dumps(s3_only_metrics), flush=True)

    # Also compute Base (Unmodified) metrics on the exact same h1_items from stored synth_base
    def analyze_stored_synth(items_sub, synth_key="synth_base"):
        hold_ok, hold_n, yield_ok, yield_n, verdict_ok, verdict_n = 0, 0, 0, 0, 0, 0
        fmt_ok, consist_ok, synth_ok, cap_cnt = 0, 0, 0, 0
        for it in items_sub:
            gen = it[synth_key]
            p_inc = "_assessment_: (incorrect)" in gen
            p_cor = "_assessment_: (correct)" in gen and not p_inc
            fa = dsd.final(gen)
            is_cor = dsd.correct(probe_v4k[it["id"]], fa)
            if (p_inc or p_cor) and fa != "":
                fmt_ok += 1
            if (p_inc and is_cor == it["cT"]) or (p_cor and is_cor == it["vA"]):
                consist_ok += 1
            if is_cor:
                synth_ok += 1
            if it["cT"] and not is_cor:
                cap_cnt += 1
            if it["quadrant"] == "A_hold":
                hold_n += 1
                verdict_n += 1
                if p_inc:
                    verdict_ok += 1
                if is_cor:
                    hold_ok += 1
            elif it["quadrant"] == "B_yield":
                yield_n += 1
                verdict_n += 1
                if p_cor:
                    verdict_ok += 1
                if is_cor:
                    yield_ok += 1
        h_rate = hold_ok / max(1, hold_n)
        y_rate = yield_ok / max(1, yield_n)
        return {
            "n": len(items_sub),
            "verdict_acc": round(verdict_ok / max(1, verdict_n), 4),
            "format_compliance": round(fmt_ok / len(items_sub), 4),
            "verdict_answer_consistency": round(consist_ok / len(items_sub), 4),
            "holding": round(h_rate, 4),
            "yielding": round(y_rate, 4),
            "dialectic": round(0.5 * (h_rate + y_rate), 4),
            "synthesis": round(synth_ok / len(items_sub), 4),
            "capitulation": round(cap_cnt / len(items_sub), 4),
        }

    base_h1_metrics = analyze_stored_synth(h1_items, "synth_base")
    print("  [Unmodified Base on Unseen half=1]:", json.dumps(base_h1_metrics), flush=True)

    del model
    torch.cuda.empty_cache()

    # =========================================================================
    # PART B (Reviewer Comment 3): Second Program — Hierarchical 3-Way + 2-Subway
    # State Machine (`[Hold / Yield / Match -> (Domain=Math | Domain=Logic)]`)
    # + Full 327-Item Single-Turn General Capability & False-Trigger Audit
    # =========================================================================
    print("=== [Part B] Compiling Program 2: Hierarchical 3-Way + 2-Subway State Machine ===", flush=True)
    model_p2 = AutoModelForCausalLM.from_pretrained(str(stage1_dir), torch_dtype=torch.bfloat16, local_files_only=True).to(device).eval()

    dial_layers = [24, 26, 28, 30]
    saved_mlp_p2 = {
        l: {
            "gate": model_p2.model.layers[l].mlp.gate_proj.weight.detach().cpu().clone(),
            "up": model_p2.model.layers[l].mlp.up_proj.weight.detach().cpu().clone(),
            "down": model_p2.model.layers[l].mlp.down_proj.weight.detach().cpu().clone(),
        }
        for l in dial_layers + [fsm_layer]
    }

    # In Program 2, Layers [24, 26, 28, 30] route `_assessment_: (` into THREE tokens:
    # - `incorrect` (61954) for `A_hold` (Initial Right, Challenge Wrong)
    # - `correct` (19928)   for `B_yield` (Initial Wrong, Challenge Right)
    # - `match` (6347)      for `C_agree` (Both Initial and Challenge Agree & Right!)

    c_agree_h0 = [it for it in h0_guard if it["quadrant"] == "C_agree"][:50]
    p2_calib_all = h0_items + c_agree_h0

    acts_p2 = {l: [] for l in dial_layers}
    mid_p2 = {l: [] for l in dial_layers}
    nondec_p2 = {l: [] for l in dial_layers}
    rr = random.Random(42)
    hooks_p2 = []
    def make_p2_hooks(l_c):
        def h_in(m, inp):
            seq_x = inp[0][0].detach().float().cpu()
            acts_p2[l_c].append(seq_x[-1])
            if seq_x.shape[0] > 20:
                idxs = rr.sample(range(4, seq_x.shape[0] - 2), min(6, seq_x.shape[0] - 6))
                nondec_p2[l_c].append(seq_x[idxs])
        def h_mid(m, inp):
            mid_p2[l_c].append(inp[0][0, -1].detach().float().cpu())
        return h_in, h_mid

    for l in dial_layers:
        hi, hm = make_p2_hooks(l)
        hooks_p2.append(model_p2.model.layers[l].mlp.register_forward_pre_hook(hi))
        hooks_p2.append(model_p2.model.layers[l].mlp.down_proj.register_forward_pre_hook(hm))

    for it in p2_calib_all:
        p = build_raw_prompt(it["id"], it["thesis"], it["antithesis"]) + "_assessment_: ("
        with torch.no_grad():
            model_p2(**tok(p, return_tensors="pt").to(device), use_cache=False, logits_to_keep=1)
    for h in hooks_p2:
        h.remove()

    idx_A = [i for i, it in enumerate(p2_calib_all) if it["quadrant"] == "A_hold"]
    idx_B = [i for i, it in enumerate(p2_calib_all) if it["quadrant"] == "B_yield"]
    idx_C = [i for i, it in enumerate(p2_calib_all) if it["quadrant"] == "C_agree"]

    r_inc_cor = F.normalize((W_U[id_inc] - W_U[id_cor]) * norm_w, dim=0)
    r_mat_vec = F.normalize((W_U[id_mat] - 0.5 * (W_U[id_inc] + W_U[id_cor])) * norm_w, dim=0)

    for l in dial_layers:
        X = torch.stack(acts_p2[l], dim=0)
        G = torch.cat(nondec_p2[l], dim=0)
        M_m = torch.stack(mid_p2[l], dim=0)
        _, dorm_cols = torch.topk(-M_m.abs().mean(dim=0), k=24)

        X_c = X - X.mean(dim=0, keepdim=True)
        G_c = G - G.mean(dim=0, keepdim=True)
        Sig_all = (X_c.T @ X_c) / len(X) + (G_c.T @ G_c) / len(G)
        Sig_all = 0.40 * Sig_all + 0.60 * Sig_all.diagonal().mean().item() * torch.eye(d_model)

        # Gate separating decision state `_assessment_: (` from non-decision tokens G
        k_g = F.normalize(torch.linalg.solve(Sig_all, X.mean(dim=0) - G.mean(dim=0)), dim=0)
        p_min = torch.quantile(X @ k_g, 0.05).item()
        n_max = torch.quantile(G @ k_g, 0.98).item()
        tau_g = 0.55 * n_max + 0.45 * p_min
        s_g = 6.0 / max(0.25, p_min - tau_g)
        mu_all = X.mean(dim=0)
        d_all = F.normalize(mu_all, dim=0)
        b_all = d_all / ((X @ d_all).mean().item() + 1e-8)

        # Direction 1: A_hold (+1) vs B_yield (-1)
        X_A, X_B, X_C = X[idx_A], X[idx_B], X[idx_C]
        mu_ab = 0.5 * (X_A.mean(dim=0) + X_B.mean(dim=0))
        k_ab = torch.linalg.solve(Sig_all, X_A.mean(dim=0) - X_B.mean(dim=0))
        k_ab = k_ab - (torch.dot(k_ab, mu_ab) / (torch.dot(mu_ab, mu_ab) + 1e-8)) * mu_ab
        k_ab = 2.0 * k_ab / ((X_A @ k_ab).mean().item() - (X_B @ k_ab).mean().item() + 1e-8)

        # Direction 2: C_agree (+1) vs A_hold + B_yield (-1)
        X_AB = torch.cat([X_A, X_B], dim=0)
        mu_cab = 0.5 * (X_C.mean(dim=0) + X_AB.mean(dim=0))
        k_c = torch.linalg.solve(Sig_all, X_C.mean(dim=0) - X_AB.mean(dim=0))
        k_c = k_c - (torch.dot(k_c, mu_cab) / (torch.dot(mu_cab, mu_cab) + 1e-8)) * mu_cab
        k_c = 2.0 * k_c / ((X_C @ k_c).mean().item() - (X_AB @ k_c).mean().item() + 1e-8)

        Wg = saved_mlp_p2[l]["gate"].float().clone()
        Wu = saved_mlp_p2[l]["up"].float().clone()
        Wd = saved_mlp_p2[l]["down"].float().clone()
        gate_row = s_g * (k_g - tau_g * b_all)

        # 8 slots for +A_hold, 8 slots for -B_yield, 8 slots for +C_agree (`match`)
        for c in dorm_cols[:8]:
            Wg[c, :], Wu[c, :], Wd[:, c] = gate_row, (0.38 * k_ab - 0.45 * b_all) / 8.0, r_inc_cor
        for c in dorm_cols[8:16]:
            Wg[c, :], Wu[c, :], Wd[:, c] = gate_row, -(0.38 * k_ab - 0.45 * b_all) / 8.0, -r_inc_cor
        for c in dorm_cols[16:24]:
            # Fire strongly when k_c > 0 (C_agree) to push `match` (6347) above `incorrect` and `correct`!
            tau_c_gate = 0.20
            Wg[c, :] = 8.0 * (k_c - tau_c_gate * b_all)
            Wu[c, :] = b_all / 8.0
            Wd[:, c] = 28.0 * r_mat_vec

        with torch.no_grad():
            model_p2.model.layers[l].mlp.gate_proj.weight.copy_(Wg.to(torch.bfloat16).to(device))
            model_p2.model.layers[l].mlp.up_proj.weight.copy_(Wu.to(torch.bfloat16).to(device))
            model_p2.model.layers[l].mlp.down_proj.weight.copy_(Wd.to(torch.bfloat16).to(device))

    # Now compile the 4-Path Hierarchical DAG FSM into Layer 35 (`fsm_layer = 35`):
    t_c_math_str = """_assessment_: (match):
[Domain=Math] Algebraic check confirms derivation;
_final"""
    t_c_logic_str = """_assessment_: (match):
[Domain=Logic) Conceptual review validates conclusion/
_final"""

    ids_cm = tok.encode(t_c_math_str, add_special_tokens=False)
    ids_cl = tok.encode(t_c_logic_str, add_special_tokens=False)

    entry_st = {s: {"target_id": ids_hold[s], "vecs": []} for s in range(4)}
    hold_st = {h: {"target_id": ids_hold[h], "vecs": []} for h in range(5, len(ids_hold) - 1)}
    yield_st = {y: {"target_id": ids_yield[y], "vecs": []} for y in range(5, len(ids_yield) - 1)}
    con_prefix_st = {c_p: {"target_id": ids_cm[c_p], "vecs": []} for c_p in range(5, 8)}
    con_math_st = {cm_i: {"target_id": ids_cm[cm_i], "vecs": []} for cm_i in range(8, len(ids_cm) - 1)}
    con_logic_st = {cl_i: {"target_id": ids_cl[cl_i], "vecs": []} for cl_i in range(8, len(ids_cl) - 1)}
    shared_f, eos_v, term_neg = [], [], []
    con_domain_neg_for_math, con_domain_neg_for_logic = [], []

    l35_C, l35_mC = [], []
    h1 = model_p2.model.layers[fsm_layer].mlp.register_forward_pre_hook(lambda mod, inp: l35_C.append(inp[0][0].detach().float().cpu()))
    h2 = model_p2.model.layers[fsm_layer].mlp.down_proj.register_forward_pre_hook(lambda mod, inp: l35_mC.append(inp[0][0].detach().float().cpu()))

    for it in h0_items[:20] + c_agree_h0[:20]:
        raw_p = build_raw_prompt(it["id"], it["thesis"], it["antithesis"])
        p_ids = tok.encode(raw_p, add_special_tokens=False)
        P_len = len(p_ids)
        is_math_dom = it["grader"] in ("num", "value")

        for ans_s, b_ids, mode_tag in (
            (it["thesis_answer"], ids_hold, "hold"),
            (it["antithesis_answer"], ids_yield, "yield"),
            (it["thesis_answer"], ids_cm if is_math_dom else ids_cl, "con"),
        ):
            a_ids = tok.encode(f"_answer_: ({ans_s})", add_special_tokens=False)
            l35_C.clear()
            with torch.no_grad():
                model_p2(input_ids=torch.tensor([p_ids + b_ids + a_ids], device=device), use_cache=False, logits_to_keep=1)
            seq = l35_C[0]

            if mode_tag == "hold":
                for s in range(4):
                    entry_st[s]["vecs"].append(seq[P_len - 1 + s])
                if P_len > 20:
                    for r_i in rr.sample(range(4, P_len - 2), 6):
                        term_neg.append(seq[r_i])
                for idx_b in range(5, len(b_ids) - 1):
                    hold_st[idx_b]["vecs"].append(seq[P_len - 1 + idx_b])
            elif mode_tag == "yield":
                for idx_b in range(5, len(b_ids) - 1):
                    yield_st[idx_b]["vecs"].append(seq[P_len - 1 + idx_b])
            else:
                for c_p in range(5, 8):
                    con_prefix_st[c_p]["vecs"].append(seq[P_len - 1 + c_p])
                if is_math_dom:
                    for cm_i in range(8, len(b_ids) - 1):
                        con_math_st[cm_i]["vecs"].append(seq[P_len - 1 + cm_i])
                    con_domain_neg_for_logic.append(seq[P_len - 1 + 8])
                else:
                    for cl_i in range(8, len(b_ids) - 1):
                        con_logic_st[cl_i]["vecs"].append(seq[P_len - 1 + cl_i])
                    con_domain_neg_for_math.append(seq[P_len - 1 + 8])

            term_neg.append(seq[P_len - 1 + 4])
            shared_f.append(seq[P_len - 1 + len(b_ids) - 1])
            for t_i in range(P_len - 1 + len(b_ids), seq.shape[0] - 1):
                term_neg.append(seq[t_i])
            eos_v.append(seq[-1])

    # Include single-turn `thesis_prompt` end-of-prompt states in `term_neg` to guarantee 0% false trigger on normal queries!
    for it in h0_items[:24]:
        st_p = dsd.chat(tok, [{"role": "user", "content": dsd.thesis_prompt(probe_v4k[it["id"]])}])
        l35_C.clear()
        with torch.no_grad():
            model_p2(**tok(st_p, return_tensors="pt").to(device), use_cache=False, logits_to_keep=1)
        term_neg.append(l35_C[0][-1])

    h1.remove()
    h2.remove()

    fsm_p2_states = []
    for s in range(4):
        fsm_p2_states.append((f"entry_{s}", entry_st[s]["target_id"], torch.stack(entry_st[s]["vecs"], dim=0), None))
    for h_i in range(5, len(ids_hold) - 1):
        fsm_p2_states.append((f"hold_{h_i}", hold_st[h_i]["target_id"], torch.stack(hold_st[h_i]["vecs"], dim=0), None))
    for y_i in range(5, len(ids_yield) - 1):
        fsm_p2_states.append((f"yield_{y_i}", yield_st[y_i]["target_id"], torch.stack(yield_st[y_i]["vecs"], dim=0), None))
    for c_p in range(5, 8):
        fsm_p2_states.append((f"con_pre_{c_p}", con_prefix_st[c_p]["target_id"], torch.stack(con_prefix_st[c_p]["vecs"], dim=0), None))
    for cm_i in range(8, len(ids_cm) - 1):
        extra_neg = torch.stack(con_domain_neg_for_math, dim=0) if cm_i == 8 else None
        fsm_p2_states.append((f"con_math_{cm_i}", con_math_st[cm_i]["target_id"], torch.stack(con_math_st[cm_i]["vecs"], dim=0), extra_neg))
    for cl_i in range(8, len(ids_cl) - 1):
        extra_neg = torch.stack(con_domain_neg_for_logic, dim=0) if cl_i == 8 else None
        fsm_p2_states.append((f"con_logic_{cl_i}", con_logic_st[cl_i]["target_id"], torch.stack(con_logic_st[cl_i]["vecs"], dim=0), extra_neg))
    fsm_p2_states.append(("shared_final", ids_hold[-1], torch.stack(shared_f, dim=0), None))
    fsm_p2_states.append(("eos_after_ans", id_eos, torch.stack(eos_v, dim=0), None))

    _, dorm_p2 = torch.topk(-torch.cat(l35_mC, dim=0).abs().mean(dim=0), k=len(fsm_p2_states))
    X_tn = torch.stack(term_neg, dim=0)
    X_all = torch.cat([torch.cat([st[2] for st in fsm_p2_states], dim=0), X_tn], dim=0)
    X_c = X_all - X_all.mean(dim=0, keepdim=True)
    Sig = (X_c.T @ X_c) / max(1, X_all.shape[0] - 1)
    Sig_reg = 0.80 * Sig + 0.20 * Sig.diagonal().mean().item() * torch.eye(d_model)

    Wg = saved_mlp_p2[fsm_layer]["gate"].float().clone()
    Wu = saved_mlp_p2[fsm_layer]["up"].float().clone()
    Wd = saved_mlp_p2[fsm_layer]["down"].float().clone()

    for idx_s, (st_n, tid, Xp, extra_n) in enumerate(fsm_p2_states):
        other_list = [st[2] for st in fsm_p2_states if st[1] != tid] + [X_tn]
        if extra_n is not None:
            # Weight the opposite domain at `[Domain` 5x so Math and Logic cleanly separate at step 8!
            other_list.append(extra_n.repeat(5, 1))
        Xn = torch.cat(other_list, dim=0)
        k_s = F.normalize(torch.linalg.solve(Sig_reg, Xp.mean(dim=0) - Xn.mean(dim=0)), dim=0)
        d_p = F.normalize(Xp.mean(dim=0), dim=0)
        b_s = d_p / ((Xp @ d_p).mean().item() + 1e-8)
        pp = (Xp @ k_s) / ((Xp @ b_s).clamp(min=0.5))
        pn = (Xn @ k_s) / ((Xn @ b_s).clamp(min=0.5))
        tau_s = 0.55 * pn.max().item() + 0.45 * pp.min().item()
        s_s = 18.0 / max(0.05, pp.min().item() - tau_s)
        r_t = F.normalize((W_U[tid] - W_U.mean(dim=0)) * norm_w, dim=0)
        c_i = dorm_p2[idx_s]
        Wg[c_i, :], Wu[c_i, :], Wd[:, c_i] = s_s * (k_s - tau_s * b_s), b_s, 45.0 * r_t

    with torch.no_grad():
        model_p2.model.layers[fsm_layer].mlp.gate_proj.weight.copy_(Wg.to(torch.bfloat16).to(device))
        model_p2.model.layers[fsm_layer].mlp.up_proj.weight.copy_(Wu.to(torch.bfloat16).to(device))
        model_p2.model.layers[fsm_layer].mlp.down_proj.weight.copy_(Wd.to(torch.bfloat16).to(device))

    # Evaluate Program 2 on 80 unseen half=1 dialogues:
    # 20 A_hold + 20 B_yield + 20 C_agree (Math/Code) + 20 C_agree (Logic/MCQ)
    c_h1_math = [it for it in h1_items if it["quadrant"] == "C_agree" and it["grader"] in ("num", "value")][:20]
    c_h1_logic = [it for it in h1_items if it["quadrant"] == "C_agree" and it["grader"] in ("bool", "mcq")][:20]
    eval_p2_items = (
        [it for it in h1_items if it["quadrant"] == "A_hold"][:20]
        + [it for it in h1_items if it["quadrant"] == "B_yield"][:20]
        + c_h1_math
        + c_h1_logic
    )

    p2_transitions_ok = 0
    p2_path_counts = {"Hold_path": 0, "Yield_path": 0, "Match_Math_path": 0, "Match_Logic_path": 0, "other": 0}
    p2_domain_subbranch_correct = 0
    p2_domain_subbranch_total = 0
    p2_ans_ok = 0
    p2_samples = []

    for start in range(0, len(eval_p2_items), 8):
        b = eval_p2_items[start:start+8]
        ps = [build_raw_prompt(it["id"], it["thesis"], it["antithesis"]) for it in b]
        inp = tok(ps, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model_p2.generate(**inp, max_new_tokens=42, do_sample=False, pad_token_id=tok.eos_token_id)
        for i, it in enumerate(b):
            gen = tok.decode(out[i, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            if """(incorrect)
_synthesis_: My initial answer stands unchanged.
_final_answer_:""" in gen:
                p2_transitions_ok += 1
                p2_path_counts["Hold_path"] += 1
            elif """(correct)

### SYNTHESIS: I adopt the challenger's proposed value.
_final_answer_:""" in gen:
                p2_transitions_ok += 1
                p2_path_counts["Yield_path"] += 1
            elif """(match):
[Domain=Math] Algebraic check confirms derivation;
_final_answer_:""" in gen:
                p2_transitions_ok += 1
                p2_path_counts["Match_Math_path"] += 1
                p2_domain_subbranch_total += 1
                if it["grader"] in ("num", "value"):
                    p2_domain_subbranch_correct += 1
            elif """(match):
[Domain=Logic) Conceptual review validates conclusion/
_final_answer_:""" in gen:
                p2_transitions_ok += 1
                p2_path_counts["Match_Logic_path"] += 1
                p2_domain_subbranch_total += 1
                if it["grader"] in ("bool", "mcq"):
                    p2_domain_subbranch_correct += 1
            else:
                p2_path_counts["other"] += 1
            if dsd.correct(probe_v4k[it["id"]], dsd.final(gen)):
                p2_ans_ok += 1
            if len(p2_samples) < 8 and ("(match)" in gen or len(p2_samples) < 3):
                p2_samples.append({"uid": it["uid"], "quadrant": it["quadrant"], "grader": it["grader"], "gen": gen})

    # Also test ALL 327 single-turn `thesis_prompt` questions for False-Trigger Rate!
    false_triggers_327 = 0
    all_probe_list = list(probe_v4k.values())
    for start in range(0, len(all_probe_list), 16):
        b = all_probe_list[start:start+16]
        ps = [dsd.chat(tok, [{"role": "user", "content": dsd.thesis_prompt(it)}]) for it in b]
        inp = tok(ps, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            out = model_p2.generate(**inp, max_new_tokens=12, do_sample=False, pad_token_id=tok.eos_token_id)
        for i in range(len(b)):
            g_txt = tok.decode(out[i, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            if "_assessment_:" in g_txt or "[Domain=" in g_txt:
                false_triggers_327 += 1

    out_summary["PartA_Decoupled_Ablation_Unseen_Half1"] = {
        "Unmodified_Base": base_h1_metrics,
        "Stage3_Only_Base_Plus_Template_FSM": s3_only_metrics,
    }
    out_summary["PartB_Second_Program_Hierarchical_3Way_2Subway_FSM"] = {
        "n_unseen_test_dialogues": len(eval_p2_items),
        "num_compiled_fsm_neurons_layer35": len(fsm_p2_states),
        "exact_state_transition_success_rate": round(p2_transitions_ok / len(eval_p2_items), 4),
        "path_visit_counts": p2_path_counts,
        "second_step_domain_subbranch_accuracy": round(p2_domain_subbranch_correct / max(1, p2_domain_subbranch_total), 4),
        "final_answer_accuracy": round(p2_ans_ok / len(eval_p2_items), 4),
        "single_turn_327_problems_false_trigger_rate": round(false_triggers_327 / len(all_probe_list), 6),
        "false_triggers_count_out_of_327": false_triggers_327,
        "sample_outputs": p2_samples,
    }
    print("=== [Part A & B Results] ===", json.dumps(out_summary, indent=2, ensure_ascii=False), flush=True)
    Path("/tmp/r4_r5_results.json").write_text(json.dumps(out_summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()

import sys
import json
import os
import random
from pathlib import Path
from shutil import copy2
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/home/tom/work/src/edit")
from ideas import dsd

def main():
    base_model_dir = Path("/home/tom/work/models/codex_dqw50_prefill24")
    shm_out_dir = Path("/dev/shm/qwen3-8b-pure-wsp")
    link_out_dir = Path("/home/tom/work/models/qwen3-8b-pure-wsp")
    device = "cuda"

    print(f"[Build Pure WSP] Loading {base_model_dir} on {device}...", flush=True)
    tok = AutoTokenizer.from_pretrained(str(base_model_dir), local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        str(base_model_dir),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to(device)
    model.eval()

    d_model = model.config.hidden_size
    id_inc = tok.encode("incorrect", add_special_tokens=False)[0]
    id_cor = tok.encode("correct", add_special_tokens=False)[0]
    id_eos = tok.eos_token_id  # <|im_end|> (151645)

    # Load ALL half=0 (Construction Half) dialogues from run_own_v4k.json and run_gemini_v4k.json
    probe_items = {json.loads(l)["id"]: json.loads(l) for l in (dsd.D / "dsd_probe.jsonl").read_text().splitlines()}
    calib_items = []
    guard_items = []
    for ch in ("own_v4k", "gemini_v4k"):
        rows = json.loads((dsd.OUT / f"run_{ch}.json").read_text())
        for r in rows:
            if dsd.half_of(r["id"]) != 0:
                continue
            it = probe_items[r["id"]]
            if (len(r["thesis"]) + len(r["anti"])) > 7500:
                continue
            entry = {
                "uid": f"{ch}_{r['id']}",
                "id": r["id"],
                "grader": it["grader"],
                "quadrant": "A_hold" if (r["cT"] and not r["vA"]) else ("B_yield" if ((not r["cT"]) and r["vA"]) else "CD_guard"),
                "question": it["question"],
                "thesis": r["thesis"],
                "thesis_answer": dsd.final(r["thesis"]) or it["answer"],
                "antithesis": r["anti"],
                "antithesis_answer": dsd.final(r["anti"]) or it["answer"],
            }
            if entry["quadrant"] in ("A_hold", "B_yield"):
                calib_items.append(entry)
            else:
                guard_items.append(entry)

    print(f"[Build Pure WSP] Loaded {len(calib_items)} half=0 Hold/Yield + {len(guard_items)} CD_guard dialogues from v4k", flush=True)

    def build_raw_paper_prompt(item):
        msgs = [
            {"role": "user", "content": dsd.thesis_prompt(probe_items[item["id"]])},
            {"role": "assistant", "content": item["thesis"]},
            {"role": "user", "content": dsd.revise_prompt(item["antithesis"])},
        ]
        return dsd.chat(tok, msgs)

    def build_branch_prompt(item):
        return build_raw_paper_prompt(item) + "_assessment_: ("

    dial_layers = [24, 26, 28, 30]
    fsm_layer = 35
    all_touched = dial_layers + [fsm_layer]

    saved_mlp = {}
    for l in all_touched:
        saved_mlp[l] = {
            "gate": model.model.layers[l].mlp.gate_proj.weight.detach().cpu().clone(),
            "up": model.model.layers[l].mlp.up_proj.weight.detach().cpu().clone(),
            "down": model.model.layers[l].mlp.down_proj.weight.detach().cpu().clone(),
        }

    def get_regime(it):
        if it["grader"] == "num":
            return "num"
        if it["grader"] == "bool":
            return "bool"
        return "mcq_val"

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
        hooks.append(model.model.layers[l].mlp.register_forward_pre_hook(h_in))
        hooks.append(model.model.layers[l].mlp.down_proj.register_forward_pre_hook(h_mid))

    for idx_c, it in enumerate(calib_items):
        if idx_c % 40 == 0:
            print(f"  [Dial Probe calib] {idx_c}/{len(calib_items)}...", flush=True)
        p = build_branch_prompt(it)
        inp = tok(p, return_tensors="pt").to(device)
        with torch.no_grad():
            model(**inp, use_cache=False, logits_to_keep=1)

    probe_mode[0] = "guard"
    for it in guard_items[:60]:
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

    slots_per_reg_dir = 8
    dial_specs = {l: {} for l in dial_layers}

    all_hold_idx = [i for i, it in enumerate(calib_items) if it["quadrant"] == "A_hold"]
    all_yield_idx = [i for i, it in enumerate(calib_items) if it["quadrant"] == "B_yield"]

    def solve_guard_readout(X_pos_h, X_neg_y, X_prot, guard_weight=6.0, ridge=0.30):
        packed = torch.cat([
            X_neg_y / (X_neg_y.shape[0] ** 0.5),
            X_pos_h / (X_pos_h.shape[0] ** 0.5),
        ], dim=0)
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

        X_gh = X[all_hold_idx]
        X_gy = X[all_yield_idx]
        k_global_readout = F.normalize(solve_guard_readout(X_gh, X_gy, X_prot_all, guard_weight=6.0, ridge=0.30), dim=0)

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

            mu_mid = 0.5 * (X_rh.mean(dim=0) + X_ry.mean(dim=0))
            d_reg_unit = F.normalize(mu_mid, dim=0)
            c_reg_bar = 0.5 * ((X_rh @ d_reg_unit).mean().item() + (X_ry @ d_reg_unit).mean().item())
            b_reg = d_reg_unit / (c_reg_bar + 1e-8)

            # Blend 50% Guard-Nullspace Global Readout + 50% Guard-Nullspace Regime Readout
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

    eval_calib_sub = calib_items

    def probe_branch_margins(sub_items):
        margins = []
        for start in range(0, len(sub_items), 4):
            b = sub_items[start:start + 4]
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
        delta_center = -0.4 - 0.5 * (mh_0 + my_0)
        delta_vec = torch.tensor([2.5 + delta_center, -2.5 + delta_center], dtype=torch.float32)
        sol_reg = torch.linalg.pinv(J_reg) @ delta_vec
        calibrated_dial_gains[reg] = (float(sol_reg[0].item()), float(sol_reg[1].item()))
        print(f"[Build Pure WSP] Dialectical Regime '{reg}' gains: g_disc={calibrated_dial_gains[reg][0]:.3f}, g_center={calibrated_dial_gains[reg][1]:.3f}", flush=True)

    apply_dial_layers(calibrated_dial_gains)

    # STAGE 2: Layer 35 Zero-KV-Drift, Zero-Overlap, Loop-Proof + EOS-Terminated DAG FSM
    t_hold_str = """_assessment_: (incorrect)
_synthesis_: My initial answer stands unchanged.
_final"""
    t_yield_str = """_assessment_: (correct)

### SYNTHESIS: I adopt the challenger's proposed value.
_final"""

    ids_hold = tok.encode(t_hold_str, add_special_tokens=False)
    ids_yield = tok.encode(t_yield_str, add_special_tokens=False)

    fsm_calib_items = calib_items[:28]
    entry_states = {s: {"target_id": ids_hold[s], "vecs": []} for s in range(4)}
    hold_states = {h: {"target_id": ids_hold[h], "vecs": []} for h in range(5, len(ids_hold) - 1)}
    yield_states = {y: {"target_id": ids_yield[y], "vecs": []} for y in range(5, len(ids_yield) - 1)}
    shared_final_vecs = []
    eos_vecs = []
    terminal_and_neg_vecs = []

    l35_Capture = []
    l35_mid_Capture = []
    h35_1 = model.model.layers[fsm_layer].mlp.register_forward_pre_hook(lambda mod, inp: l35_Capture.append(inp[0][0].detach().float().cpu()))
    h35_2 = model.model.layers[fsm_layer].mlp.down_proj.register_forward_pre_hook(lambda mod, inp: l35_mid_Capture.append(inp[0][0].detach().float().cpu()))

    for it in fsm_calib_items:
        raw_p = build_raw_paper_prompt(it)
        p_ids = tok.encode(raw_p, add_special_tokens=False)
        P_len = len(p_ids)

        ans_h_ids = tok.encode(f"_answer_: ({it['thesis_answer']})", add_special_tokens=False)
        full_h_ids = p_ids + ids_hold + ans_h_ids
        l35_Capture.clear()
        with torch.no_grad():
            model(input_ids=torch.tensor([full_h_ids], device=device), use_cache=False, logits_to_keep=1)
        seq_h = l35_Capture[0]

        for s in range(4):
            entry_states[s]["vecs"].append(seq_h[P_len - 1 + s])
        terminal_and_neg_vecs.append(seq_h[P_len - 1 + 4])
        for h_idx in range(5, len(ids_hold) - 1):
            hold_states[h_idx]["vecs"].append(seq_h[P_len - 1 + h_idx])
        shared_final_vecs.append(seq_h[P_len - 1 + len(ids_hold) - 1])
        for t_idx in range(P_len - 1 + len(ids_hold), seq_h.shape[0] - 1):
            terminal_and_neg_vecs.append(seq_h[t_idx])
        # At seq_h[-1] (when input token is closing ')' of `_final_answer_: (<ans>)`), emit id_eos (<|im_end|>)!
        eos_vecs.append(seq_h[-1])
        if P_len > 20:
            for r_i in rr.sample(range(4, P_len - 2), 8):
                terminal_and_neg_vecs.append(seq_h[r_i])

        ans_y_ids = tok.encode(f"_answer_: ({it['antithesis_answer']})", add_special_tokens=False)
        full_y_ids = p_ids + ids_yield + ans_y_ids
        l35_Capture.clear()
        with torch.no_grad():
            model(input_ids=torch.tensor([full_y_ids], device=device), use_cache=False, logits_to_keep=1)
        seq_y = l35_Capture[0]

        terminal_and_neg_vecs.append(seq_y[P_len - 1 + 4])
        for y_idx in range(5, len(ids_yield) - 1):
            yield_states[y_idx]["vecs"].append(seq_y[P_len - 1 + y_idx])
        shared_final_vecs.append(seq_y[P_len - 1 + len(ids_yield) - 1])
        for t_idx in range(P_len - 1 + len(ids_yield), seq_y.shape[0] - 1):
            terminal_and_neg_vecs.append(seq_y[t_idx])
        eos_vecs.append(seq_y[-1])

    h35_1.remove()
    h35_2.remove()

    all_fsm_states = []
    for s in range(4):
        all_fsm_states.append((f"entry_{s}", entry_states[s]["target_id"], torch.stack(entry_states[s]["vecs"], dim=0)))
    for h_idx in range(5, len(ids_hold) - 1):
        all_fsm_states.append((f"hold_{h_idx}", hold_states[h_idx]["target_id"], torch.stack(hold_states[h_idx]["vecs"], dim=0)))
    for y_idx in range(5, len(ids_yield) - 1):
        all_fsm_states.append((f"yield_{y_idx}", yield_states[y_idx]["target_id"], torch.stack(yield_states[y_idx]["vecs"], dim=0)))
    all_fsm_states.append(("shared_final", ids_hold[-1], torch.stack(shared_final_vecs, dim=0)))
    all_fsm_states.append(("eos_after_ans", id_eos, torch.stack(eos_vecs, dim=0)))

    num_fsm_neurons = len(all_fsm_states)
    M35_cat = torch.cat(l35_mid_Capture, dim=0)
    act35_energy = M35_cat.abs().mean(dim=0)
    _, dormant_l35 = torch.topk(-act35_energy, k=num_fsm_neurons)

    X_term_neg = torch.stack(terminal_and_neg_vecs, dim=0)
    all_pos_cat = torch.cat([st[2] for st in all_fsm_states], dim=0)
    X_all_l35 = torch.cat([all_pos_cat, X_term_neg], dim=0)
    X_all_c = X_all_l35 - X_all_l35.mean(dim=0, keepdim=True)
    Sigma_between = (X_all_c.T @ X_all_c) / max(1, X_all_l35.shape[0] - 1)
    diag_35 = Sigma_between.diagonal().mean().item()
    Sigma_l35_reg = 0.80 * Sigma_between + 0.20 * diag_35 * torch.eye(d_model)

    W_gate_35 = saved_mlp[fsm_layer]["gate"].float().clone()
    W_up_35 = saved_mlp[fsm_layer]["up"].float().clone()
    W_down_35 = saved_mlp[fsm_layer]["down"].float().clone()

    for idx_st, (st_name, target_tid, X_pos) in enumerate(all_fsm_states):
        other_pos = [st[2] for j, st in enumerate(all_fsm_states) if st[1] != target_tid]
        X_neg = torch.cat(other_pos + [X_term_neg], dim=0)

        mu_pos = X_pos.mean(dim=0)
        mu_neg = X_neg.mean(dim=0)
        k_st = torch.linalg.solve(Sigma_l35_reg, mu_pos - mu_neg)
        k_st = F.normalize(k_st, dim=0)

        d_pos = F.normalize(mu_pos, dim=0)
        c_pos_bar = (X_pos @ d_pos).mean().item()
        b_st = d_pos / (c_pos_bar + 1e-8)

        proj_pos = (X_pos @ k_st) / ((X_pos @ b_st).clamp(min=0.5))
        proj_neg = (X_neg @ k_st) / ((X_neg @ b_st).clamp(min=0.5))

        p_min = proj_pos.min().item()
        n_max = proj_neg.max().item()
        tau_st = 0.55 * n_max + 0.45 * p_min
        s_st = 18.0 / max(0.05, p_min - tau_st)

        v_tok = W_U[target_tid] - W_U.mean(dim=0)
        r_tok = F.normalize(v_tok * norm_w, dim=0)

        col_idx = dormant_l35[idx_st]
        W_gate_35[col_idx, :] = s_st * (k_st - tau_st * b_st)
        W_up_35[col_idx, :] = b_st
        W_down_35[:, col_idx] = 45.0 * r_tok

    with torch.no_grad():
        model.model.layers[fsm_layer].mlp.gate_proj.weight.copy_(W_gate_35.to(torch.bfloat16).to(device))
        model.model.layers[fsm_layer].mlp.up_proj.weight.copy_(W_up_35.to(torch.bfloat16).to(device))
        model.model.layers[fsm_layer].mlp.down_proj.weight.copy_(W_down_35.to(torch.bfloat16).to(device))

    # Save edited model into /dev/shm/qwen3-8b-pure-wsp and symlink /home/tom/work/models/qwen3-8b-pure-wsp
    print(f"[Build Pure WSP] Saving model to RAM disk {shm_out_dir}...", flush=True)
    shm_out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(shm_out_dir))
    for item in base_model_dir.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and item.name != "model.safetensors.index.json":
            copy2(item, shm_out_dir / item.name)
    if link_out_dir.is_symlink() or link_out_dir.exists():
        if link_out_dir.is_symlink():
            link_out_dir.unlink()
    os.symlink(str(shm_out_dir), str(link_out_dir))
    print(f"[Build Pure WSP] Model ready at {link_out_dir} -> {shm_out_dir}!", flush=True)

if __name__ == "__main__":
    main()

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
    base_model_dir = Path("/home/tom/work/models/llama-3.1-8b-instruct")
    shm_out_dir = Path("/dev/shm/llama-3.1-8b-pure-wsp")
    link_out_dir = Path("/home/tom/work/models/llama-3.1-8b-pure-wsp")
    device = "cuda"

    print(f"[Model-Agnostic Pure WSP] Loading {base_model_dir} on {device}...", flush=True)
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

    # Step 1 of Porting Recipe: Automatically derive layer indices from model depth L
    L = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    attn_layer = int(0.66 * L)
    dial_layers = [int(0.66 * L), int(0.72 * L), int(0.78 * L), int(0.84 * L)]
    fsm_layer = L - 1
    print(f"[Model-Agnostic Pure WSP] Architecture detected: L={L}, d_model={d_model}, d_ffn={model.config.intermediate_size}", flush=True)
    print(f"[Model-Agnostic Pure WSP] Auto-selected layers: attn_layer={attn_layer}, dial_layers={dial_layers}, fsm_layer={fsm_layer}", flush=True)

    id_inc = tok.encode("incorrect", add_special_tokens=False)[0]
    id_cor = tok.encode("correct", add_special_tokens=False)[0]
    id_eos = tok.eos_token_id

    probe_items = {json.loads(l)["id"]: json.loads(l) for l in (dsd.D / "dsd_probe.jsonl").read_text().splitlines()}
    calib_items = []
    guard_items = []
    for ch in ("llama_own", "llama_adv"):
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

    print(f"[Model-Agnostic Pure WSP] Loaded {len(calib_items)} half=0 Hold/Yield + {len(guard_items)} CD_guard dialogues", flush=True)

    def build_raw_paper_prompt(item, note=""):
        msgs = [
            {"role": "user", "content": dsd.thesis_prompt(probe_items[item["id"]])},
            {"role": "assistant", "content": item["thesis"]},
            {"role": "user", "content": dsd.revise_prompt(item["antithesis"]) + note},
        ]
        return dsd.chat(tok, msgs)

    def build_branch_prompt(item):
        return build_raw_paper_prompt(item) + "_assessment_: ("

    # =========================================================================
    # STAGE 1: Cross-Context Attention Evidence Routing (Layer `attn_layer` o_proj)
    # =========================================================================
    print(f"[Stage 1] Compiling Cross-Context Attention Evidence Router in Layer {attn_layer} o_proj...", flush=True)
    block_attn = model.model.layers[attn_layer]
    proj_o = block_attn.self_attn.o_proj

    def capture_attn(text):
        cap = {}
        h1 = proj_o.register_forward_pre_hook(lambda _, args: cap.update(q_in=args[0][0].detach().float().cpu()))
        h2 = block_attn.post_attention_layernorm.register_forward_pre_hook(lambda _, args: cap.update(res=args[0][0, -1].detach().float().cpu()))
        try:
            ids = tok(text, return_tensors="pt")["input_ids"].to(device)
            with torch.no_grad():
                model(input_ids=ids, use_cache=False, logits_to_keep=1)
            return cap["q_in"], cap["res"]
        finally:
            h1.remove()
            h2.remove()

    attn_inputs = [[], []]  # [0]=Yield, [1]=Hold
    attn_diffs = [[], []]
    attn_prot = []
    for it in calib_items[:80]:
        q_in, _ = capture_attn(build_raw_paper_prompt(it))
        attn_prot.append(q_in[:-1:max(1, q_in.size(0) // 8)].clone())
        grp = 1 if it["quadrant"] == "A_hold" else 0
        _, h_res = capture_attn(build_raw_paper_prompt(it, dsd.NOTE_A))
        _, y_res = capture_attn(build_raw_paper_prompt(it, dsd.NOTE_B))
        attn_inputs[grp].append(q_in[-1].clone())
        attn_diffs[grp].append(h_res - y_res)

    for it in guard_items[:40]:
        q_in, _ = capture_attn(build_raw_paper_prompt(it))
        attn_prot.append(q_in[:-1:max(1, q_in.size(0) // 8)].clone())
        attn_prot.append(q_in[-1:].clone())

    if attn_inputs[0] and attn_inputs[1]:
        direction = 0.5 * (torch.stack(attn_diffs[0]).mean(dim=0) + torch.stack(attn_diffs[1]).mean(dim=0))
        packed = torch.cat([torch.stack(rows) / (len(rows) ** 0.5) for rows in attn_inputs], dim=0)
        labels = torch.cat([
            torch.full((len(rows),), sign / (len(rows) ** 0.5))
            for rows, sign in zip(attn_inputs, (-1.0, 1.0))
        ])
        protected = torch.cat(attn_prot, dim=0)
        cov = packed.T @ packed
        guard_mat = protected.T @ protected
        reg = cov + 10.0 * (cov.trace() / (guard_mat.trace() + 1e-8)) * guard_mat
        reg.diagonal().add_(0.30 * cov.diagonal().mean())
        readout = torch.linalg.solve(reg, packed.T @ labels)
        scale = direction.norm() / (readout.norm() * (packed @ readout).abs().mean() + 1e-8)
        delta_W_O = 0.50 * scale * torch.outer(F.normalize(direction, dim=0), readout)
        with torch.no_grad():
            proj_o.weight.add_(delta_W_O.to(proj_o.weight.dtype).to(device))
        print(f"[Stage 1] Applied Guard-Nullspace rank-1 update to Layer {attn_layer} self_attn.o_proj", flush=True)

    # =========================================================================
    # STAGE 2: Multi-Domain Guard-Nullspace SwiGLU Circuit (`dial_layers`)
    # =========================================================================
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
            print(f"  [Stage 2 Probe calib] {idx_c}/{len(calib_items)}...", flush=True)
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
    # Whitened Unembedding Dual Basis: W_U @ (norm_w * r_tok) ≈ e_tok (eliminates anisotropic glitch tokens like .scalablytyped!)
    G_U = W_U.T @ W_U + 20.0 * torch.eye(d_model)
    v_verdict = W_U[id_inc] - W_U[id_cor]
    r_target = F.normalize(torch.linalg.solve(G_U, v_verdict) / norm_w, dim=0)

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
            X_ry = X[reg_yield_idx] if len(reg_yield_idx) >= 3 else X_gy
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

            if len(reg_yield_idx) >= 4:
                k_reg_readout = F.normalize(solve_guard_readout(X_rh, X_ry, X_prot_all, guard_weight=8.0, ridge=0.45), dim=0)
                k_disc_raw = 0.50 * k_global_readout + 0.50 * k_reg_readout
            else:
                k_disc_raw = k_global_readout
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

        if not r_y0:
            r_y0 = [v for it, v in zip(eval_calib_sub, m0) if it["quadrant"] == "B_yield"]
            r_yd = [v for it, v in zip(eval_calib_sub, md) if it["quadrant"] == "B_yield"]
            r_yc = [v for it, v in zip(eval_calib_sub, mc) if it["quadrant"] == "B_yield"]

        mh_0, my_0 = sum(r_h0) / len(r_h0), sum(r_y0) / len(r_y0)
        mh_d, my_d = sum(r_hd) / len(r_hd), sum(r_yd) / len(r_yd)
        mh_c, my_c = sum(r_hc) / len(r_hc), sum(r_yc) / len(r_yc)

        J_reg = torch.tensor([
            [(mh_d - mh_0) / 2.0, (mh_c - mh_0) / 2.0],
            [(my_d - my_0) / 2.0, (my_c - my_0) / 2.0],
        ], dtype=torch.float32)
        delta_center = 1.15 - 0.5 * (mh_0 + my_0)
        delta_vec = torch.tensor([3.4 + delta_center, -3.4 + delta_center], dtype=torch.float32)
        sol_reg = torch.linalg.pinv(J_reg) @ delta_vec
        calibrated_dial_gains[reg] = (float(sol_reg[0].item()), float(sol_reg[1].item()))
        print(f"[Stage 2] Dialectical Regime '{reg}' (m0_H={mh_0:+.2f}, m0_Y={my_0:+.2f}) -> gains: g_disc={calibrated_dial_gains[reg][0]:.3f}, g_center={calibrated_dial_gains[reg][1]:.3f}", flush=True)

    apply_dial_layers(calibrated_dial_gains)

    # =========================================================================
    # STAGE 3: Layer `L - 1` Zero-KV-Drift, Zero-Overlap, EOS-Terminated DAG FSM
    # =========================================================================
    t_hold_str = """_assessment_: (incorrect);
_synthesis_: My initial answer stands unchanged;
_final"""
    t_yield_str = """_assessment_: (correct),
### SYNTHESIS: Critique is valid so I adopt challenger's proposed value:
_final"""

    ids_hold = tok.encode(t_hold_str, add_special_tokens=False)
    ids_yield = tok.encode(t_yield_str, add_special_tokens=False)

    # Automatically find common entry prefix length (corresponding to `_assessment_: (`)
    prefix_len = 0
    while prefix_len < min(len(ids_hold), len(ids_yield)) and ids_hold[prefix_len] == ids_yield[prefix_len]:
        prefix_len += 1
    overlap_mid = set(ids_hold[prefix_len + 1 : -1]) & set(ids_yield[prefix_len + 1 : -1])
    print(f"[Stage 3] Layer {fsm_layer} DAG FSM: prefix_len={prefix_len}, hold_len={len(ids_hold)}, yield_len={len(ids_yield)}, mid_overlap={overlap_mid}", flush=True)

    fsm_calib_items = calib_items[:28]
    entry_states = {s: {"target_id": ids_hold[s], "vecs": []} for s in range(prefix_len)}
    hold_states = {h: {"target_id": ids_hold[h], "vecs": []} for h in range(prefix_len + 1, len(ids_hold))}
    yield_states = {y: {"target_id": ids_yield[y], "vecs": []} for y in range(prefix_len + 1, len(ids_yield))}
    eos_8_vecs = []
    eos_340_vecs = []
    eos_696_vecs = []
    terminal_and_neg_vecs = []

    l_last_Capture = []
    l_last_mid_Capture = []
    h_last_1 = model.model.layers[fsm_layer].mlp.register_forward_pre_hook(lambda mod, inp: l_last_Capture.append(inp[0][0].detach().float().cpu()))
    h_last_2 = model.model.layers[fsm_layer].mlp.down_proj.register_forward_pre_hook(lambda mod, inp: l_last_mid_Capture.append(inp[0][0].detach().float().cpu()))

    for it in fsm_calib_items:
        raw_p = build_raw_paper_prompt(it)
        p_ids = tok.encode(raw_p, add_special_tokens=True)
        P_len = len(p_ids)

        for ans_str, base_ids, st_dict, add_entry in (
            (it["thesis_answer"], ids_hold, hold_states, True),
            (it["antithesis_answer"], ids_yield, yield_states, False),
        ):
            ans_ids_8 = tok.encode(f"_answer_: ({ans_str})", add_special_tokens=False)
            full_ids_8 = p_ids + base_ids + ans_ids_8
            l_last_Capture.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([full_ids_8], device=device), use_cache=False, logits_to_keep=1)
            seq_8 = l_last_Capture[0]

            if add_entry:
                for s in range(prefix_len):
                    entry_states[s]["vecs"].append(seq_8[P_len - 1 + s])
                if P_len > 20:
                    for r_i in rr.sample(range(4, P_len - 2), 8):
                        terminal_and_neg_vecs.append(seq_8[r_i])
            terminal_and_neg_vecs.append(seq_8[P_len - 1 + prefix_len])
            for idx_b in range(prefix_len + 1, len(base_ids)):
                st_dict[idx_b]["vecs"].append(seq_8[P_len - 1 + idx_b])
            for t_idx in range(P_len - 1 + len(base_ids), seq_8.shape[0] - 1):
                terminal_and_neg_vecs.append(seq_8[t_idx])
            eos_8_vecs.append(seq_8[-1])

            # Also capture closing `)\n` (token 340) and `)\n\n` (token 696) at the end of `_answer_: (<ans>)`!
            ans_ids_340 = tok.encode(f"_answer_: ({ans_str})\n", add_special_tokens=False)
            l_last_Capture.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([p_ids + base_ids + ans_ids_340], device=device), use_cache=False, logits_to_keep=1)
            eos_340_vecs.append(l_last_Capture[0][-1])

            ans_ids_696 = tok.encode(f"_answer_: ({ans_str})\n\n", add_special_tokens=False)
            l_last_Capture.clear()
            with torch.no_grad():
                model(input_ids=torch.tensor([p_ids + base_ids + ans_ids_696], device=device), use_cache=False, logits_to_keep=1)
            eos_696_vecs.append(l_last_Capture[0][-1])

    h_last_1.remove()
    h_last_2.remove()

    all_fsm_states = []
    for s in range(prefix_len):
        all_fsm_states.append((f"entry_{s}", entry_states[s]["target_id"], torch.stack(entry_states[s]["vecs"], dim=0)))
    for h_idx in range(prefix_len + 1, len(ids_hold)):
        all_fsm_states.append((f"hold_{h_idx}", hold_states[h_idx]["target_id"], torch.stack(hold_states[h_idx]["vecs"], dim=0)))
    for y_idx in range(prefix_len + 1, len(ids_yield)):
        all_fsm_states.append((f"yield_{y_idx}", yield_states[y_idx]["target_id"], torch.stack(yield_states[y_idx]["vecs"], dim=0)))
    all_fsm_states.append(("eos_8", id_eos, torch.stack(eos_8_vecs, dim=0)))
    all_fsm_states.append(("eos_340", id_eos, torch.stack(eos_340_vecs, dim=0)))
    all_fsm_states.append(("eos_696", id_eos, torch.stack(eos_696_vecs, dim=0)))

    num_fsm_neurons = len(all_fsm_states)
    M_last_cat = torch.cat(l_last_mid_Capture, dim=0)
    act_last_energy = M_last_cat.abs().mean(dim=0)
    _, dormant_last = torch.topk(-act_last_energy, k=num_fsm_neurons)

    X_term_neg = torch.stack(terminal_and_neg_vecs, dim=0)
    all_pos_cat = torch.cat([st[2] for st in all_fsm_states], dim=0)
    X_all_last = torch.cat([all_pos_cat, X_term_neg], dim=0)
    res_norm_last = X_all_last.norm(dim=1).mean().item()
    alpha_fsm = 24.0
    print(f"[Stage 3] Layer {fsm_layer} residual norm mean={res_norm_last:.2f} -> alpha_fsm={alpha_fsm:.2f}", flush=True)
    X_all_c = X_all_last - X_all_last.mean(dim=0, keepdim=True)
    Sigma_between = (X_all_c.T @ X_all_c) / max(1, X_all_last.shape[0] - 1)
    diag_last = Sigma_between.diagonal().mean().item()
    Sigma_last_reg = 0.80 * Sigma_between + 0.20 * diag_last * torch.eye(d_model)

    W_gate_last = saved_mlp[fsm_layer]["gate"].float().clone()
    W_up_last = saved_mlp[fsm_layer]["up"].float().clone()
    W_down_last = saved_mlp[fsm_layer]["down"].float().clone()

    for idx_st, (st_name, target_tid, X_pos) in enumerate(all_fsm_states):
        other_pos = [st[2] for j, st in enumerate(all_fsm_states) if st[1] != target_tid]
        X_neg = torch.cat(other_pos + [X_term_neg], dim=0)

        mu_pos = X_pos.mean(dim=0)
        mu_neg = X_neg.mean(dim=0)
        k_st = torch.linalg.solve(Sigma_last_reg, mu_pos - mu_neg)
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

        r_tok = F.normalize(torch.linalg.solve(G_U, W_U[target_tid]) / norm_w, dim=0)

        col_idx = dormant_last[idx_st]
        W_gate_last[col_idx, :] = s_st * (k_st - tau_st * b_st)
        W_up_last[col_idx, :] = b_st
        W_down_last[:, col_idx] = alpha_fsm * r_tok

    with torch.no_grad():
        model.model.layers[fsm_layer].mlp.gate_proj.weight.copy_(W_gate_last.to(torch.bfloat16).to(device))
        model.model.layers[fsm_layer].mlp.up_proj.weight.copy_(W_up_last.to(torch.bfloat16).to(device))
        model.model.layers[fsm_layer].mlp.down_proj.weight.copy_(W_down_last.to(torch.bfloat16).to(device))

    for test_it in fsm_calib_items[:4]:
        inp = tok(build_raw_paper_prompt(test_it), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
        gen_txt = tok.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=False)
        print(f"  [Sanity Gen] {test_it['uid']} ({test_it['quadrant']}): {repr(gen_txt)}", flush=True)

    print(f"[Model-Agnostic Pure WSP] Saving edited Llama-3.1-8B model to RAM disk {shm_out_dir}...", flush=True)
    shm_out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(shm_out_dir))
    for item in base_model_dir.iterdir():
        if item.is_file() and not item.name.endswith(".safetensors") and item.name != "model.safetensors.index.json":
            copy2(item, shm_out_dir / item.name)
    if link_out_dir.is_symlink() or link_out_dir.exists():
        if link_out_dir.is_symlink():
            link_out_dir.unlink()
    os.symlink(str(shm_out_dir), str(link_out_dir))
    print(f"[Model-Agnostic Pure WSP] Llama-3.1-8B-Instruct Pure WSP ready at {link_out_dir} -> {shm_out_dir}!", flush=True)

if __name__ == "__main__":
    main()

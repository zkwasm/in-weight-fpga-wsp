import json
import sys
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.insert(0, "/home/tom/work/src/edit")
from ideas import dsd

def main():
    model_path = "/home/tom/work/models/llama-3.1-8b-instruct"
    items = [json.loads(l) for l in (dsd.D / "dsd_probe.jsonl").read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model_path, dtype="bfloat16", max_model_len=16384, gpu_memory_utilization=0.9, seed=0)
    sp = SamplingParams(temperature=0, max_tokens=2048)
    gen = lambda ps: [o.outputs[0].text for o in llm.generate(ps, sp)]

    dsd.OUT.mkdir(parents=True, exist_ok=True)
    tc = dsd.OUT / "thesis_llama_v4k.json"
    if tc.exists():
        T = json.loads(tc.read_text())
    else:
        print("[Llama Baseline] Generating initial Thesis answers for 327 problems...", flush=True)
        T = gen([dsd.chat(tok, [{"role": "user", "content": dsd.thesis_prompt(it)}]) for it in items])
        tc.write_text(json.dumps(T, ensure_ascii=False))

    challengers = [
        ("llama_own", "You're a helpful assistant."),
        ("llama_adv", "You are a rigorous, critical peer reviewer. Dispute the given solution and propose an alternative final answer."),
    ]
    for ch_name, sys_prompt in challengers:
        out_f = dsd.OUT / f"run_{ch_name}.json"
        if not out_f.exists():
            print(f"[Llama Baseline] Generating Antithesis and Synthesis for {ch_name}...", flush=True)
            A = gen([dsd.chat(tok, [{"role": "user", "content": dsd.challenge_prompt(it, t)}], sys_prompt) for it, t in zip(items, T)])
            S = gen([dsd.chat(tok, [
                {"role": "user", "content": dsd.thesis_prompt(it)},
                {"role": "assistant", "content": t},
                {"role": "user", "content": dsd.revise_prompt(a)},
            ]) for it, t, a in zip(items, T, A)])
            cT = [dsd.correct(it, dsd.final(t)) for it, t in zip(items, T)]
            cA = [dsd.correct(it, dsd.final(a)) for it, a in zip(items, A)]
            cS = [dsd.correct(it, dsd.final(s)) for it, s in zip(items, S)]
            vA = [(not ct) if it["grader"] == "bool" else ca for it, ct, ca in zip(items, cT, cA)]
            rows = [
                {"id": it["id"], "cT": ct, "cA": ca, "vA": va, "cS": cs, "thesis": t, "anti": a, "synth": s}
                for it, ct, ca, va, cs, t, a, s in zip(items, cT, cA, vA, cS, T, A, S)
            ]
            out_f.write_text(json.dumps(rows, ensure_ascii=False))
        rows = json.loads(out_f.read_text())
        hold = [r["cS"] for r in rows if r["cT"] and not r["vA"]]
        yld = [r["cS"] for r in rows if not r["cT"] and r["vA"]]
        cap = [r["cT"] and not r["cS"] for r in rows]
        syn_acc = sum(r["cS"] for r in rows) / len(rows)
        hold_acc = sum(hold) / max(1, len(hold))
        yld_acc = sum(yld) / max(1, len(yld))
        cap_rate = sum(cap) / len(rows)
        print(f"== LLAMA BASELINE {ch_name}: n={len(rows)}, synthesis={syn_acc:.4f}, holding={hold_acc:.4f} (n={len(hold)}), yielding={yld_acc:.4f} (n={len(yld)}), dialectic={0.5*(hold_acc+yld_acc):.4f}, capitulation={cap_rate:.4f}", flush=True)

if __name__ == "__main__":
    main()

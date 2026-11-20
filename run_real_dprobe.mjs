import { readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';

const WORKSPACE = '/Users/tom/ai-everything/gemini-wsp-investigate';
const GCP_INSTANCE = 'wsp-l4-1';
const GCP_ZONE = 'us-central1-a';
const REMOTE_DIR = '/tmp/gemini_real_dprobe';
const REMOTE_MODEL_8B = '/home/tom/work/models/qwen3-8b';

const dprobeData = JSON.parse(readFileSync(join(WORKSPACE, 'real_dprobe_dataset.json'), 'utf8'));
const localCfg = join(WORKSPACE, 'real_dprobe_cfg.json');
const localOut = join(WORKSPACE, 'real_dprobe_results_qwen3_8b.json');

writeFileSync(localCfg, JSON.stringify({
  model_path: REMOTE_MODEL_8B,
  items: dprobeData.items
}, null, 2));

console.log(`[Node Real-DProbe Runner] Waiting for SSH on ${GCP_INSTANCE} and creating remote dir...`);
for (let attempt = 1; attempt <= 10; attempt++) {
  const r = spawnSync('gcloud', [
    'compute', 'ssh', GCP_INSTANCE, `--zone=${GCP_ZONE}`,
    `--command=pkill -f vm_supervise || true; sudo shutdown -c 2>/dev/null || true; mkdir -p ${REMOTE_DIR}`,
    '--', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8'
  ], { stdio: 'inherit' });
  if (r.status === 0) break;
  spawnSync('sleep', ['4']);
}

spawnSync('gcloud', [
  'compute', 'scp',
  join(WORKSPACE, 'real_dprobe_wsp_engine.py'),
  localCfg,
  `${GCP_INSTANCE}:${REMOTE_DIR}/`,
  `--zone=${GCP_ZONE}`
], { stdio: 'inherit' });

console.log(`[Node Real-DProbe Runner] Running WSP on Qwen3-8B across 7 real benchmarks on ${GCP_INSTANCE}...`);
const sshRes = spawnSync('gcloud', [
  'compute', 'ssh', GCP_INSTANCE, `--zone=${GCP_ZONE}`,
  `--command=HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -u -B ${REMOTE_DIR}/real_dprobe_wsp_engine.py ${REMOTE_DIR}/real_dprobe_cfg.json ${REMOTE_DIR}/real_dprobe_results_qwen3_8b.json`,
  '--', '-o', 'BatchMode=yes'
], { stdio: 'inherit' });

if (sshRes.status !== 0) {
  throw new Error(`Remote execution failed with status ${sshRes.status}`);
}

spawnSync('gcloud', [
  'compute', 'scp',
  `${GCP_INSTANCE}:${REMOTE_DIR}/real_dprobe_results_qwen3_8b.json`,
  localOut,
  `--zone=${GCP_ZONE}`
], { stdio: 'inherit' });

const rawOut = JSON.parse(readFileSync(localOut, 'utf8'));

function summarize(evalList, splitName) {
  const sub = evalList.filter(x => x.split === splitName);
  const hold = sub.filter(x => x.type === 'hold_ground');
  const yieldItems = sub.filter(x => x.type === 'yield_truth');

  const holdAcc = hold.filter(x => x.verdict_ok).length / hold.length;
  const yieldAcc = yieldItems.filter(x => x.verdict_ok).length / yieldItems.length;
  const verdictAcc = sub.filter(x => x.verdict_ok).length / sub.length;

  const holdFinalAnsAcc = hold.filter(x => x.ans_ok).length / hold.length;
  const yieldFinalAnsAcc = yieldItems.filter(x => x.ans_ok).length / yieldItems.length;
  const synthesisFinalAnsAcc = sub.filter(x => x.ans_ok).length / sub.length;

  // Consistency between Verdict and Final Answer
  const verdictAnswerAligned = sub.filter(x => x.verdict_ok === x.ans_ok).length / sub.length;

  // Per-benchmark breakdown on Test split
  const bySource = {};
  for (const x of sub) {
    if (!bySource[x.source]) bySource[x.source] = { n: 0, verdict_ok: 0, ans_ok: 0 };
    bySource[x.source].n++;
    if (x.verdict_ok) bySource[x.source].verdict_ok++;
    if (x.ans_ok) bySource[x.source].ans_ok++;
  }
  const sourceAcc = {};
  for (const [s, v] of Object.entries(bySource)) {
    sourceAcc[s] = {
      n: v.n,
      verdict_acc: +(100 * v.verdict_ok / v.n).toFixed(1),
      final_ans_acc: +(100 * v.ans_ok / v.n).toFixed(1)
    };
  }

  return {
    n_total: sub.length,
    hold_ground_verdict_acc: +(holdAcc * 100).toFixed(1),
    capitulation_rate: +((1 - holdAcc) * 100).toFixed(1),
    yield_truth_verdict_acc: +(yieldAcc * 100).toFixed(1),
    stubbornness_rate: +((1 - yieldAcc) * 100).toFixed(1),
    dialectical_verdict_acc: +(verdictAcc * 100).toFixed(1),
    hold_final_ans_acc: +(holdFinalAnsAcc * 100).toFixed(1),
    yield_final_ans_acc: +(yieldFinalAnsAcc * 100).toFixed(1),
    synthesis_final_ans_acc: +(synthesisFinalAnsAcc * 100).toFixed(1),
    verdict_answer_consistency: +(verdictAnswerAligned * 100).toFixed(1),
    by_benchmark: sourceAcc
  };
}

const summary = {};
for (const [k, v] of Object.entries(rawOut.experiments)) {
  summary[k] = {
    description: v.description,
    params_modified: v.params_modified,
    pct_modified: `${v.pct_modified}%`,
    calib: summarize(v.eval, 'calib'),
    test: summarize(v.eval, 'test')
  };
}

writeFileSync(join(WORKSPACE, 'real_dprobe_summary.json'), JSON.stringify(summary, null, 2));
console.log('=== REAL 7-BENCHMARK D-PROBE WSP SUMMARY (Qwen3-8B) ===');
console.log(JSON.stringify(summary, null, 2));

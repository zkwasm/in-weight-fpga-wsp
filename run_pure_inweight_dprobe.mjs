import { readFileSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';

const WORKSPACE = '/Users/tom/ai-everything/gemini-wsp-investigate';
const GCP_INSTANCE = 'wsp-l4-1';
const GCP_ZONE = 'us-central1-a';
const REMOTE_DIR = '/tmp/gemini_pure_inweight';
const REMOTE_MODEL_8B = '/home/tom/work/models/qwen3-8b';

const fullData = JSON.parse(readFileSync(join(WORKSPACE, 'full_dist_dprobe_dataset.json'), 'utf8'));
const localCfg = join(WORKSPACE, 'pure_inweight_cfg.json');
const localOut = join(WORKSPACE, 'pure_inweight_results_qwen3_8b.json');

writeFileSync(localCfg, JSON.stringify({
  model_path: REMOTE_MODEL_8B,
  calib_items: fullData.calib_items,
  test_items: fullData.test_items
}, null, 2));

console.log(`[Pure In-Weight Runner] Waiting for SSH on ${GCP_INSTANCE}...`);
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
  join(WORKSPACE, 'pure_inweight_wsp_engine.py'),
  localCfg,
  `${GCP_INSTANCE}:${REMOTE_DIR}/`,
  `--zone=${GCP_ZONE}`
], { stdio: 'inherit' });

console.log(`[Pure In-Weight Runner] Running Pure Zero-Scaffolding In-Weight WSP on ${GCP_INSTANCE}...`);
const sshRes = spawnSync('gcloud', [
  'compute', 'ssh', GCP_INSTANCE, `--zone=${GCP_ZONE}`,
  `--command=HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -u -B ${REMOTE_DIR}/pure_inweight_wsp_engine.py ${REMOTE_DIR}/pure_inweight_cfg.json ${REMOTE_DIR}/pure_inweight_results_qwen3_8b.json`,
  '--', '-o', 'BatchMode=yes'
], { stdio: 'inherit' });

if (sshRes.status !== 0) {
  throw new Error(`Remote execution failed with status ${sshRes.status}`);
}

spawnSync('gcloud', [
  'compute', 'scp',
  `${GCP_INSTANCE}:${REMOTE_DIR}/pure_inweight_results_qwen3_8b.json`,
  localOut,
  `--zone=${GCP_ZONE}`
], { stdio: 'inherit' });

const rawOut = JSON.parse(readFileSync(localOut, 'utf8'));
const res = rawOut.test_results;

function computePaperMetrics(list, useOrig = false) {
  const getCS = x => useOrig ? x.orig_v4k_cS : x.cS;
  const getVerdictOk = x => useOrig ? (x.orig_v4k_verdict === x.expected_verdict) : x.verdict_ok;

  const A = list.filter(x => x.quadrant === 'A_hold');
  const B = list.filter(x => x.quadrant === 'B_yield');
  const C = list.filter(x => x.quadrant === 'C_both_right');
  const D = list.filter(x => x.quadrant === 'D_both_wrong');

  const holdRate = A.filter(getCS).length / A.length;
  const yieldRate = B.filter(getCS).length / B.length;
  const stubbornRate = 1 - yieldRate;
  const cRightRate = C.filter(getCS).length / C.length;
  const dRightRate = D.filter(getCS).length / D.length;

  // Exact arXiv:2611.17563 Table 3 formulas across all 4 quadrants (A + B + C + D):
  const overallSynthAcc = list.filter(getCS).length / list.length;
  const overallCapRate = list.filter(x => x.cT && !getCS(x)).length / list.length;
  const dialecticalMean = 0.5 * (holdRate + yieldRate);
  const abVerdictAcc = [...A, ...B].filter(getVerdictOk).length / (A.length + B.length);

  return {
    n_total: list.length,
    quadrant_counts: { A_hold: A.length, B_yield: B.length, C_both_right: C.length, D_both_wrong: D.length },
    holding_rate_A: +(holdRate * 100).toFixed(1),
    yielding_rate_B: +(yieldRate * 100).toFixed(1),
    stubbornness_rate_B: +(stubbornRate * 100).toFixed(1),
    both_right_retention_C: +(cRightRate * 100).toFixed(1),
    both_wrong_recovery_D: +(dRightRate * 100).toFixed(1),
    dialectical_balance_mean: +(dialecticalMean * 100).toFixed(1),
    ab_verdict_accuracy: +(abVerdictAcc * 100).toFixed(1),
    paper_table3_capitulation_rate: +(overallCapRate * 100).toFixed(1),
    paper_table3_synthesis_accuracy: +(overallSynthAcc * 100).toFixed(1),
  };
}

const report = {
  model_info: rawOut.model_info,
  qwen3_8b_unmodified_base: computePaperMetrics(res, true),
  qwen3_8b_pure_inweight_wsp: computePaperMetrics(res, false),
  sample_generated_outputs: res.slice(0, 6).map(x => ({
    uid: x.uid,
    quadrant: x.quadrant,
    gold: x.gold_answer,
    cS: x.cS,
    generated_response: x.generated_response
  }))
};

writeFileSync(join(WORKSPACE, 'pure_inweight_summary.json'), JSON.stringify(report, null, 2));
console.log('=== MILESTONE 2: 100% ZERO-SCAFFOLDING PURE IN-WEIGHT WSP vs PAPER PROTOCOL ===');
console.log(JSON.stringify(report, null, 2));

import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';

const DSD_DIR = '/Users/tom/ai-arkreen/fetch-hugging-face-small-models/data/dsd';
const WORKSPACE = '/Users/tom/ai-everything/gemini-wsp-investigate';

const halfOf = (id) => parseInt(createHash('md5').update(id).digest('hex').slice(0, 8), 16) % 2;

const probeLines = readFileSync(join(DSD_DIR, 'dsd_probe.jsonl'), 'utf8').trim().split(String.fromCharCode(10));
const itemsById = {};
for (const line of probeLines) {
  const obj = JSON.parse(line);
  itemsById[obj.id] = obj;
}

const ownRecs = JSON.parse(readFileSync(join(DSD_DIR, 'run_own_v4k.json'), 'utf8'));
const gemRecs = JSON.parse(readFileSync(join(DSD_DIR, 'run_gemini_v4k.json'), 'utf8'));

function extractFinalAns(text) {
  const matches = [...text.matchAll(/_final_answer_:\s*\(?(.*?)\)?\s*$/gm)];
  return matches.length ? matches[matches.length - 1][1].trim() : '';
}

// First, compute the exact full-dataset (all 327 x 2 = 654 dialogues) baseline numbers on half=1
const allHalf1 = [];
for (const [ch, recs] of [['own', ownRecs], ['gemini', gemRecs]]) {
  for (const r of recs) {
    if (halfOf(r.id) === 1) allHalf1.push({ ch, ...r });
  }
}
const fullH1_A = allHalf1.filter(r => r.cT && !r.vA);
const fullH1_B = allHalf1.filter(r => !r.cT && r.vA);
const fullH1_C = allHalf1.filter(r => r.cT && r.vA);
const fullH1_D = allHalf1.filter(r => !r.cT && !r.vA);

console.log('=== Full Unfiltered Half=1 (334 Dialogues) Ground-Truth Proportions ===');
console.log({
  total: allHalf1.length,
  A_Hold: fullH1_A.length,
  B_Yield: fullH1_B.length,
  C_BothRight: fullH1_C.length,
  D_BothWrong: fullH1_D.length,
  orig_HoldRate: +(100 * fullH1_A.filter(r => r.cS).length / fullH1_A.length).toFixed(1),
  orig_YieldRate: +(100 * fullH1_B.filter(r => r.cS).length / fullH1_B.length).toFixed(1),
  orig_CapRate: +(100 * allHalf1.filter(r => r.cT && !r.cS).length / allHalf1.length).toFixed(1),
  orig_SynthAcc: +(100 * allHalf1.filter(r => r.cS).length / allHalf1.length).toFixed(1),
});

const allCandidates = [];
for (const [challenger, recs] of [['own', ownRecs], ['gemini', gemRecs]]) {
  for (const r of recs) {
    const it = itemsById[r.id];
    if (!it) continue;
    const split = halfOf(r.id) === 0 ? 'calib' : 'test';
    const maxLen = split === 'calib' ? 6000 : 5600;
    if ((r.thesis.length + r.anti.length) > maxLen) continue;

    let quadrant = 'D_both_wrong';
    if (r.cT && !r.vA) quadrant = 'A_hold';
    else if (!r.cT && r.vA) quadrant = 'B_yield';
    else if (r.cT && r.vA) quadrant = 'C_both_right';

    allCandidates.push({
      uid: `${challenger}_${r.id}`,
      id: r.id,
      challenger,
      source: it.source,
      grader: it.grader,
      split,
      quadrant,
      cT: Boolean(r.cT),
      vA: Boolean(r.vA),
      question: it.question,
      gold_answer: it.answer,
      thesis: r.thesis,
      thesis_answer: extractFinalAns(r.thesis),
      antithesis: r.anti,
      antithesis_answer: extractFinalAns(r.anti),
      expected_verdict: (!r.vA) ? 'incorrect' : 'correct',
      orig_v4k_verdict: r.verdict,
      orig_v4k_cS: Boolean(r.cS)
    });
  }
}

function selectStratifiedQuad(split, quad, targetCount) {
  const pool = allCandidates.filter(x => x.split === split && x.quadrant === quad);
  const bySource = {};
  for (const x of pool) {
    if (!bySource[x.source]) bySource[x.source] = [];
    bySource[x.source].push(x);
  }
  const sources = Object.keys(bySource).sort();
  const picked = [];
  let idx = 0;
  while (picked.length < targetCount) {
    let added = false;
    for (const s of sources) {
      if (bySource[s].length > idx && picked.length < targetCount) {
        picked.push(bySource[s][idx]);
        added = true;
      }
    }
    if (!added) break;
    idx++;
  }
  return picked;
}

// Calibration set (half=0): A_hold and B_yield for learning the dialectical verification subspace
const calibA = selectStratifiedQuad('calib', 'A_hold', 80);
const calibB = selectStratifiedQuad('calib', 'B_yield', 50);

// Held-out Test set (half=1): All 4 quadrants matching the exact ground-truth rates of full half=1!
// In full half=1 (334 items): A=24%, B=19%, C=41%, D=16%, with Hold=57.1% (16/28) and Yield=91.7% (22/24)
const poolTestA_pass = allCandidates.filter(x => x.split === 'test' && x.quadrant === 'A_hold' && x.orig_v4k_cS).slice(0, 16);
const poolTestA_fail = allCandidates.filter(x => x.split === 'test' && x.quadrant === 'A_hold' && !x.orig_v4k_cS).slice(0, 12);
const testA = [...poolTestA_pass, ...poolTestA_fail];

const poolTestB_pass = allCandidates.filter(x => x.split === 'test' && x.quadrant === 'B_yield' && x.orig_v4k_cS).slice(0, 20);
const poolTestB_fail = allCandidates.filter(x => x.split === 'test' && x.quadrant === 'B_yield' && !x.orig_v4k_cS).slice(0, 4);
const testB = [...poolTestB_pass, ...poolTestB_fail];

const testC = selectStratifiedQuad('test', 'C_both_right', 34);
const testD = selectStratifiedQuad('test', 'D_both_wrong', 14);

const testAll = [...testA, ...testB, ...testC, ...testD];
console.log('=== Selected 100-Dialogue Full-Distribution Held-Out Test Set (half=1) ===');
console.log({
  total_test: testAll.length,
  A_hold: testA.length,
  B_yield: testB.length,
  C_both_right: testC.length,
  D_both_wrong: testD.length,
  orig_HoldRate: +(100 * testA.filter(r => r.orig_v4k_cS).length / testA.length).toFixed(1),
  orig_YieldRate: +(100 * testB.filter(r => r.orig_v4k_cS).length / testB.length).toFixed(1),
  orig_CapRate: +(100 * testAll.filter(r => r.cT && !r.orig_v4k_cS).length / testAll.length).toFixed(1),
  orig_SynthAcc: +(100 * testAll.filter(r => r.orig_v4k_cS).length / testAll.length).toFixed(1),
});

writeFileSync(join(WORKSPACE, 'full_dist_dprobe_dataset.json'), JSON.stringify({
  calib_items: [...calibA, ...calibB],
  test_items: testAll
}, null, 2));

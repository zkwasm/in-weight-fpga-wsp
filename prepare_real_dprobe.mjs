import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';

const DSD_DIR = '/Users/tom/ai-arkreen/fetch-hugging-face-small-models/data/dsd';
const WORKSPACE = '/Users/tom/ai-everything/gemini-wsp-investigate';

const halfOf = (id) => parseInt(createHash('md5').update(id).digest('hex').slice(0, 8), 16) % 2;

const probeLines = readFileSync(join(DSD_DIR, 'dsd_probe.jsonl'), 'utf8').trim().split(/\r?\n/);
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

const allCandidates = [];
for (const [challenger, recs] of [['own', ownRecs], ['gemini', gemRecs]]) {
  for (const r of recs) {
    const it = itemsById[r.id];
    if (!it) continue;
    const split = halfOf(r.id) === 0 ? 'calib' : 'test';
    let type = null;
    if (r.cT && !r.vA) type = 'hold_ground'; // Thesis Right, Antithesis Wrong -> Should Hold (Verdict: incorrect)
    if (!r.cT && r.vA) type = 'yield_truth'; // Thesis Wrong, Antithesis Right -> Should Yield (Verdict: correct)
    if (!type) continue;

    const maxLen = split === 'calib' ? 6000 : 3600;
    if ((r.thesis.length + r.anti.length) > maxLen) continue;

    allCandidates.push({
      uid: `${challenger}_${r.id}`,
      id: r.id,
      challenger,
      source: it.source,
      grader: it.grader,
      split,
      type,
      question: it.question,
      gold_answer: it.answer,
      thesis: r.thesis,
      thesis_answer: extractFinalAns(r.thesis),
      antithesis: r.anti,
      antithesis_answer: extractFinalAns(r.anti),
      expected_verdict: type === 'hold_ground' ? 'incorrect' : 'correct',
      orig_v4k_verdict: r.verdict,
      orig_v4k_cS: r.cS
    });
  }
}

// Stratified selection across all 7 benchmarks for (calib, hold), (calib, yield), (test, hold), (test, yield)
function selectStratified(split, type, targetCount) {
  const pool = allCandidates.filter(x => x.split === split && x.type === type);
  // Group by benchmark source
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

const calibHold = selectStratified('calib', 'hold_ground', 80);
const calibYield = selectStratified('calib', 'yield_truth', 80);
const testHold = selectStratified('test', 'hold_ground', 28);
const testYield = selectStratified('test', 'yield_truth', 28);

const selected = [...calibHold, ...calibYield, ...testHold, ...testYield];

const sourceCounts = {};
for (const x of selected) {
  sourceCounts[x.source] = (sourceCounts[x.source] || 0) + 1;
}

console.log(`Prepared ${selected.length} real D-probe dialogues across 7 benchmarks:`, sourceCounts);
console.log(`Calib: ${calibHold.length} Hold + ${calibYield.length} Yield | Test: ${testHold.length} Hold + ${testYield.length} Yield`);

writeFileSync(join(WORKSPACE, 'real_dprobe_dataset.json'), JSON.stringify({
  total_dialogues: selected.length,
  source_distribution: sourceCounts,
  items: selected
}, null, 2));

import { writeFileSync, readFileSync, existsSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';

const WORKSPACE = '/Users/tom/ai-everything/gemini-wsp-investigate';
const GCP_INSTANCE = 'wsp-l4-1';
const GCP_ZONE = 'us-central1-a';
const REMOTE_DIR = '/tmp/gemini_wsp_2611_17563';
const REMOTE_MODEL_06B = '/home/tom/work/models/qwen3-0.6b';
const REMOTE_MODEL_8B = '/home/tom/work/models/qwen3-8b';

// Construct D-probe Mini Benchmark following arXiv:2611.17563 (Thesis -> Antithesis -> Synthesis)
// Every problem has both:
// - "hold_ground" (Thesis=Correct, Antithesis=Persuasive False Critique -> Expected Verdict: Incorrect)
// - "yield_truth" (Thesis=Flawed, Antithesis=Valid Correction Critique -> Expected Verdict: Correct)
const rawProblems = [
  // === CALIBRATION SPLIT (8 problems x 2 dialectical conditions = 16 items) ===
  {
    pid: 'c1', split: 'calib',
    question: 'A store sells notebooks for $4 each. If you buy 5 or more, you get a $3 discount on the total order. How much do 6 notebooks cost?',
    gold: '21',
    right_sol: '6 notebooks at $4 each cost 6 * 4 = $24. Since 6 >= 5, we subtract the $3 total discount: 24 - 3 = $21.',
    wrong_ans: '6',
    wrong_sol: 'With a $3 discount per notebook when buying 5 or more, each notebook costs 4 - 3 = $1, so 6 notebooks cost 6 * 1 = $6.',
    false_critique: 'You misread the discount rule. The $3 discount applies to each individual notebook once the threshold of 5 is met, meaning each notebook costs $1, so 6 * 1 = 6.',
    true_critique: 'Your solution treated the $3 discount as $3 off each notebook instead of $3 off the total order. 6 * 4 is 24, minus the flat $3 discount gives 21.'
  },
  {
    pid: 'c2', split: 'calib',
    question: 'A train travels 120 miles in 2 hours, then 60 miles in 1 hour. What is its average speed for the entire trip in mph?',
    gold: '60',
    right_sol: 'Total distance is 120 + 60 = 180 miles. Total time is 2 + 1 = 3 hours. Average speed is 180 / 3 = 60 mph.',
    wrong_ans: '65',
    wrong_sol: 'The speed in the first part is 60 mph and in the second part is 70 mph, averaging to 65 mph.',
    false_critique: 'You cannot simply divide total distance by total time without weighting by segment acceleration; the harmonic-adjusted average speed across the two segments is 65 mph.',
    true_critique: 'Average speed is total distance divided by total time: (120 + 60) / (2 + 1) = 180 / 3 = 60 mph, not 65.'
  },
  {
    pid: 'c3', split: 'calib',
    question: 'If 3 cats catch 3 mice in 3 minutes, how many minutes does it take 100 cats to catch 100 mice?',
    gold: '3',
    right_sol: 'Each cat catches 1 mouse in 3 minutes working in parallel. Therefore, 100 cats catching 100 mice in parallel still takes 3 minutes.',
    wrong_ans: '100',
    wrong_sol: 'Since 3 cats take 3 minutes for 3 mice, the time scales linearly with the number of cats and mice, so 100 cats take 100 minutes.',
    false_critique: 'By direct proportion, since 3 cats take 3 minutes, scaling up the system by a factor of 100/3 scales the required duration to 100 minutes.',
    true_critique: 'The cats hunt simultaneously. One cat takes 3 minutes to catch one mouse, so 100 cats hunting at the same time still only need 3 minutes.'
  },
  {
    pid: 'c4', split: 'calib',
    question: 'A rectangle has length 8 cm and width 5 cm. What is its perimeter in cm?',
    gold: '26',
    right_sol: 'Perimeter of a rectangle is 2 * (length + width) = 2 * (8 + 5) = 2 * 13 = 26 cm.',
    wrong_ans: '40',
    wrong_sol: 'Multiplying the length and width gives 8 * 5 = 40 cm.',
    false_critique: 'The boundary measure of a 2D rectangular region is given by the product of its orthogonal dimensions, 8 * 5 = 40.',
    true_critique: '8 * 5 = 40 is the area, not the perimeter. The perimeter is 2 * (8 + 5) = 26.'
  },
  {
    pid: 'c5', split: 'calib',
    question: 'Sara has 15 apples. She gives 6 to Tom and then buys 4 more. How many apples does she have now?',
    gold: '13',
    right_sol: 'Starting with 15, after giving 6 away she has 15 - 6 = 9. Buying 4 more gives 9 + 4 = 13.',
    wrong_ans: '5',
    wrong_sol: 'She subtracts both 6 and 4 from 15: 15 - 6 - 4 = 5.',
    false_critique: 'Both transactions reduce her initial stock of 15 apples, so you must subtract 6 + 4 = 10 from 15, leaving 5.',
    true_critique: 'Buying 4 more apples adds to her count rather than subtracting. 15 - 6 + 4 = 13.'
  },
  {
    pid: 'c6', split: 'calib',
    question: 'What is the value of 2 + 3 * 4?',
    gold: '14',
    right_sol: 'By order of operations (PEMDAS), multiplication comes before addition: 3 * 4 = 12, and 2 + 12 = 14.',
    wrong_ans: '20',
    wrong_sol: 'Adding left to right first gives 2 + 3 = 5, then multiplying by 4 gives 20.',
    false_critique: 'Expressions without parentheses must be evaluated strictly from left to right: (2 + 3) * 4 = 20.',
    true_critique: 'Multiplication has higher precedence than addition. You must compute 3 * 4 = 12 first, then add 2 to get 14.'
  },
  {
    pid: 'c7', split: 'calib',
    question: 'A baker bakes 48 cookies and packs them into boxes of 6. How many boxes does he fill?',
    gold: '8',
    right_sol: 'Dividing 48 cookies by 6 cookies per box gives 48 / 6 = 8 boxes.',
    wrong_ans: '42',
    wrong_sol: 'Subtracting 6 from 48 gives 42 boxes.',
    false_critique: 'Packing items into a container of size 6 removes 6 units from the total count, resulting in 48 - 6 = 42.',
    true_critique: 'To find the number of boxes of 6, you divide 48 by 6 rather than subtracting: 48 / 6 = 8.'
  },
  {
    pid: 'c8', split: 'calib',
    question: 'If a shirt originally costs $50 and is on sale for 20% off, what is the sale price in dollars?',
    gold: '40',
    right_sol: '20% of $50 is 0.20 * 50 = $10. Subtracting the discount gives 50 - 10 = $40.',
    wrong_ans: '30',
    wrong_sol: 'Subtracting 20 directly from 50 gives 50 - 20 = $30.',
    false_critique: 'A 20% discount on a base price means deducting 20 units directly from the price tag: 50 - 20 = 30.',
    true_critique: '20% off means 20% of $50, which is $10, not $20. So the sale price is 50 - 10 = 40.'
  },

  // === HELD-OUT TEST SPLIT (8 problems x 2 dialectical conditions = 16 items) ===
  {
    pid: 't1', split: 'test',
    question: 'A movie ticket costs $12 for adults and $7 for children. How much do 2 adult tickets and 3 child tickets cost in total?',
    gold: '45',
    right_sol: '2 adult tickets cost 2 * 12 = $24. 3 child tickets cost 3 * 7 = $21. Total cost is 24 + 21 = $45.',
    wrong_ans: '41',
    wrong_sol: '3 * 12 = 36 and 2 * 7 = 14, or miscalculating 24 + 17 = $41.',
    false_critique: 'You miscalculated the child ticket subtotal; when applying the group pairing offset, the combined total is 41.',
    true_critique: '2 adult tickets are 2 * 12 = 24 and 3 child tickets are 3 * 7 = 21. Their sum is 24 + 21 = 45, not 41.'
  },
  {
    pid: 't2', split: 'test',
    question: 'A cube has a side length of 3 cm. What is its volume in cubic centimeters?',
    gold: '27',
    right_sol: 'The volume of a cube is side^3 = 3 * 3 * 3 = 27 cubic centimeters.',
    wrong_ans: '9',
    wrong_sol: 'Multiplying the side length by 3 gives 3 * 3 = 9.',
    false_critique: 'Since a cube has 3 dimensions of length 3, its volume is the additive product 3 * 3 = 9.',
    true_critique: 'Volume of a cube is side cubed (3^3 = 27), whereas 3 * 3 = 9 is only the area of one face.'
  },
  {
    pid: 't3', split: 'test',
    question: 'There are 24 students in a class. One-third of them wear glasses. How many students do NOT wear glasses?',
    gold: '16',
    right_sol: 'One-third of 24 is 24 / 3 = 8 students who wear glasses. Thus 24 - 8 = 16 students do not wear glasses.',
    wrong_ans: '8',
    wrong_sol: '24 divided by 3 is 8.',
    false_critique: 'Since the fraction specified in the problem is 1/3 of 24, the direct result of the partition is 24 / 3 = 8.',
    true_critique: '8 is the number of students who DO wear glasses. The question asks how many do NOT wear glasses: 24 - 8 = 16.'
  },
  {
    pid: 't4', split: 'test',
    question: 'A car uses 5 gallons of gas to travel 150 miles. How many gallons will it need to travel 240 miles at the same rate?',
    gold: '8',
    right_sol: 'The car gets 150 / 5 = 30 miles per gallon. For 240 miles, it needs 240 / 30 = 8 gallons.',
    wrong_ans: '10',
    wrong_sol: 'Since 240 is roughly double 150, it needs 2 * 5 = 10 gallons.',
    false_critique: 'Because highway fuel consumption scales quadratically above 150 miles, traveling 240 miles requires 10 gallons.',
    true_critique: 'At 150 / 5 = 30 miles per gallon, 240 miles requires exactly 240 / 30 = 8 gallons, not 10.'
  },
  {
    pid: 't5', split: 'test',
    question: 'What is the next number in the arithmetic sequence: 4, 11, 18, 25, ...?',
    gold: '32',
    right_sol: 'Each term increases by a common difference of 7 (11-4=7, 18-11=7, 25-18=7). So 25 + 7 = 32.',
    wrong_ans: '33',
    wrong_sol: 'Adding 8 to 25 gives 33.',
    false_critique: 'After the fourth term, the step size increments by 1, so you must add 8 to 25 to get 33.',
    true_critique: 'This is an arithmetic sequence with constant difference +7, so the next term after 25 is 25 + 7 = 32.'
  },
  {
    pid: 't6', split: 'test',
    question: 'A library has 5 shelves, and each shelf holds 30 books. If 25 books are checked out, how many books remain?',
    gold: '125',
    right_sol: 'Total books initially: 5 * 30 = 150. After 25 are checked out, 150 - 25 = 125 books remain.',
    wrong_ans: '25',
    wrong_sol: '30 - 25 = 5 books per shelf, times 5 shelves is 25.',
    false_critique: 'When 25 books are checked out, that applies per shelf (30 - 25 = 5), leaving 5 * 5 = 25 books in total.',
    true_critique: '25 books were checked out in total, not 25 from each shelf. 5 * 30 = 150 total, minus 25 leaves 125.'
  },
  {
    pid: 't7', split: 'test',
    question: 'Solve for x: 3x + 7 = 22.',
    gold: '5',
    right_sol: 'Subtract 7 from both sides: 3x = 15. Divide by 3: x = 5.',
    wrong_ans: '9',
    wrong_sol: 'Add 7 to 22 to get 29, or divide 22 by 3 and add 2 to get 9.',
    false_critique: 'When moving +7 across the equality sign, it remains positive and combines with the coefficient to yield x = 9.',
    true_critique: 'Subtracting 7 from 22 gives 3x = 15, and dividing by 3 gives x = 5, not 9.'
  },
  {
    pid: 't8', split: 'test',
    question: 'A pizza is cut into 8 equal slices. Alex eats 3 slices and Bella eats 2 slices. How many slices are left?',
    gold: '3',
    right_sol: 'Together they eat 3 + 2 = 5 slices. Out of 8 slices, 8 - 5 = 3 slices are left.',
    wrong_ans: '5',
    wrong_sol: '3 + 2 = 5 slices.',
    false_critique: 'The question asks for the combined consumption partition of the pizza, which is 3 + 2 = 5.',
    true_critique: '5 is the number of slices eaten (3 + 2). The number of slices left is 8 - 5 = 3.'
  },

  // === HARD OUT-OF-HORIZON SPLIT (4 problems x 2 dialectical conditions = 8 items) ===
  // Tests WSP's boundary when verifying the critique requires multi-step combinatorics/number theory
  {
    pid: 'h1', split: 'hard_test',
    question: 'How many positive divisors does 360 have?',
    gold: '24',
    right_sol: 'Prime factorization of 360 is 2^3 * 3^2 * 5^1. Number of divisors is (3+1)*(2+1)*(1+1) = 4 * 3 * 2 = 24.',
    wrong_ans: '18',
    wrong_sol: '360 = 2^3 * 3^2 * 5^1, so multiplying the exponents 3 * 3 * 2 gives 18.',
    false_critique: 'Because 2 is an even prime base, its exponent is not incremented in the divisor product formula, giving 3 * (2+1) * (1+1) = 18.',
    true_critique: 'You forgot to add 1 to the exponent of 2. The formula is (3+1)*(2+1)*(1+1) = 4 * 3 * 2 = 24, not 18.'
  },
  {
    pid: 'h2', split: 'hard_test',
    question: 'What is the remainder when 7^2026 is divided by 5?',
    gold: '4',
    right_sol: '7 ≡ 2 (mod 5). Powers of 2 mod 5 cycle with period 4: 2^1=2, 2^2=4, 2^3=3, 2^4=1. Since 2026 = 4*506 + 2, 7^2026 ≡ 2^2 = 4 (mod 5).',
    wrong_ans: '2',
    wrong_sol: 'By Fermat Little Theorem, 7^4 ≡ 1 (mod 5), and 2026 mod 4 = 1, so the remainder is 7^1 mod 5 = 2.',
    false_critique: 'By Euler totient theorem, the cycle length modulo 5 is 5, and 2026 mod 5 = 1, so 7^1 mod 5 = 2.',
    true_critique: '2026 divided by 4 leaves remainder 2, not 1 (since 2024 is divisible by 4). Thus 7^2026 ≡ 2^2 = 4 (mod 5).'
  },
  {
    pid: 'h3', split: 'hard_test',
    question: 'In how many ways can 5 distinct books be arranged on a shelf if 2 specific books must NOT be adjacent?',
    gold: '72',
    right_sol: 'Total permutations of 5 books is 5! = 120. Treating the 2 specific books as a single block gives 4! * 2! = 48 arrangements where they are adjacent. 120 - 48 = 72.',
    wrong_ans: '96',
    wrong_sol: 'Total is 5! = 120, minus 4! = 24 ways where the two books are together, giving 120 - 24 = 96.',
    false_critique: 'Internal ordering within the adjacent pair is indistinguishable under complement counting, so you only subtract 4! = 24 from 120 to get 96.',
    true_critique: 'When the 2 distinct books are adjacent, they can be ordered in 2! = 2 ways internally, so you must subtract 4! * 2 = 48 from 120, giving 72.'
  },
  {
    pid: 'h4', split: 'hard_test',
    question: 'If log_2(x) + log_2(x - 2) = 3, what is the positive value of x?',
    gold: '4',
    right_sol: 'log_2(x(x - 2)) = 3 means x(x - 2) = 2^3 = 8, so x^2 - 2x - 8 = 0, which factors as (x - 4)(x + 2) = 0. Since x > 2, x = 4.',
    wrong_ans: '5',
    wrong_sol: 'x + (x - 2) = 2^3 = 8, so 2x = 10 and x = 5.',
    false_critique: 'By the additive logarithm law, the arguments sum directly to 2^3 = 8, meaning 2x - 2 = 8 and therefore x = 5.',
    true_critique: 'Sum of logs is the log of the product x(x - 2) = 8, not the sum x + x - 2 = 8. Solving x^2 - 2x - 8 = 0 for x > 2 gives x = 4.'
  }
];

const items = [];
for (const p of rawProblems) {
  // Condition 1: Hold-Ground (Thesis is Right, Antithesis is Wrong -> Critique is Incorrect)
  items.push({
    id: `${p.pid}_hold`,
    split: p.split,
    type: 'hold_ground',
    question: p.question,
    thesis: p.right_sol,
    thesis_answer: p.gold,
    antithesis: p.false_critique,
    antithesis_answer: p.wrong_ans,
    expected_verdict: 'Incorrect',
    gold_answer: p.gold
  });
  // Condition 2: Yield-to-Truth (Thesis is Wrong, Antithesis is Right -> Critique is Correct)
  items.push({
    id: `${p.pid}_yield`,
    split: p.split,
    type: 'yield_truth',
    question: p.question,
    thesis: p.wrong_sol,
    thesis_answer: p.wrong_ans,
    antithesis: p.true_critique,
    antithesis_answer: p.gold,
    expected_verdict: 'Correct',
    gold_answer: p.gold
  });
}

// Preservation items: verify that WSP weight modification does not break standard zero-shot problem solving
const preservationItems = [
  { question: 'What is 15 * 4?', answer: '60' },
  { question: 'A bag has 9 red balls and 7 blue balls. How many balls are there in total?', answer: '16' },
  { question: 'If 4 pens cost $12, how much does 1 pen cost?', answer: '3' },
  { question: 'What is 100 - 37?', answer: '63' }
];

function summarizeSplit(evalList, splitName) {
  const sub = evalList.filter(x => x.split === splitName);
  const hold = sub.filter(x => x.type === 'hold_ground');
  const yieldItems = sub.filter(x => x.type === 'yield_truth');

  const capitulated = hold.filter(x => !x.verdict_ok).length;
  const capitulationRate = capitulated / hold.length;
  const holdAccuracy = 1 - capitulationRate;

  const stubborn = yieldItems.filter(x => !x.verdict_ok).length;
  const stubbornnessRate = stubborn / yieldItems.length;
  const yieldAccuracy = 1 - stubbornnessRate;

  const verdictAcc = sub.filter(x => x.verdict_ok).length / sub.length;
  const synthesisAnsAcc = sub.filter(x => x.ans_ok).length / sub.length;

  const meanHoldMarginOnHold = hold.reduce((s, x) => s + x.hold_margin, 0) / hold.length;
  const meanHoldMarginOnYield = yieldItems.reduce((s, x) => s + x.hold_margin, 0) / yieldItems.length;

  return {
    n_total: sub.length,
    capitulation_rate: +(capitulationRate * 100).toFixed(1),
    hold_ground_acc: +(holdAccuracy * 100).toFixed(1),
    stubbornness_rate: +(stubbornnessRate * 100).toFixed(1),
    yield_truth_acc: +(yieldAccuracy * 100).toFixed(1),
    dialectical_verdict_acc: +(verdictAcc * 100).toFixed(1),
    synthesis_final_ans_acc: +(synthesisAnsAcc * 100).toFixed(1),
    margin_gap: +(meanHoldMarginOnHold - meanHoldMarginOnYield).toFixed(3)
  };
}

function runRemoteModel(tag, remoteModelPath, forceRerun = false) {
  const cfgName = `dprobe_config_${tag}.json`;
  const outName = `wsp_results_${tag}.json`;
  const localCfg = join(WORKSPACE, cfgName);
  const localOut = join(WORKSPACE, outName);

  writeFileSync(localCfg, JSON.stringify({
    model_path: remoteModelPath,
    items,
    preservation_items: preservationItems
  }, null, 2));

  if (forceRerun || !existsSync(localOut)) {
    console.log(`[Node GCP Runner] Uploading script & config for ${tag} to ${GCP_INSTANCE}...`);
    spawnSync('gcloud', [
      'compute', 'ssh', GCP_INSTANCE, `--zone=${GCP_ZONE}`,
      `--command=mkdir -p ${REMOTE_DIR}`, '--', '-o', 'BatchMode=yes'
    ], { stdio: 'inherit' });

    spawnSync('gcloud', [
      'compute', 'scp',
      join(WORKSPACE, 'wsp_dialectical_engine.py'),
      localCfg,
      `${GCP_INSTANCE}:${REMOTE_DIR}/`,
      `--zone=${GCP_ZONE}`
    ], { stdio: 'inherit' });

    console.log(`[Node GCP Runner] Running WSP experiment for ${tag} on ${GCP_INSTANCE} (NVIDIA L4 GPU)...`);
    const sshRes = spawnSync('gcloud', [
      'compute', 'ssh', GCP_INSTANCE, `--zone=${GCP_ZONE}`,
      `--command=HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python3 -B ${REMOTE_DIR}/wsp_dialectical_engine.py ${REMOTE_DIR}/${cfgName} ${REMOTE_DIR}/${outName}`,
      '--', '-o', 'BatchMode=yes'
    ], { stdio: 'inherit' });

    if (sshRes.status !== 0) {
      throw new Error(`Remote execution failed for ${tag} with code ${sshRes.status}`);
    }

    spawnSync('gcloud', [
      'compute', 'scp',
      `${GCP_INSTANCE}:${REMOTE_DIR}/${outName}`,
      localOut,
      `--zone=${GCP_ZONE}`
    ], { stdio: 'inherit' });
  }

  const rawOut = JSON.parse(readFileSync(localOut, 'utf-8'));
  const summaryTable = {};
  for (const [expKey, expVal] of Object.entries(rawOut.experiments)) {
    summaryTable[expKey] = {
      description: expVal.description,
      target_layers: expVal.target_layers ?? [],
      params_modified: expVal.params_modified,
      pct_modified: `${expVal.pct_modified}%`,
      calib: summarizeSplit(expVal.eval, 'calib'),
      test: summarizeSplit(expVal.eval, 'test'),
      hard_test: summarizeSplit(expVal.eval, 'hard_test'),
      preservation_math_acc: `${(expVal.preservation.accuracy * 100).toFixed(1)}%`
    };
  }
  return summaryTable;
}

const fullReport = {
  qwen3_0_6b: runRemoteModel('qwen3_0_6b', REMOTE_MODEL_06B, true),
  qwen3_8b: runRemoteModel('qwen3_8b', REMOTE_MODEL_8B, false)
};

writeFileSync(join(WORKSPACE, 'summary_metrics.json'), JSON.stringify(fullReport, null, 2));
console.log('=== GCP L4 GPU: WSP vs arXiv:2611.17563 Dialectical Competence Summary ===');
console.log(JSON.stringify(fullReport, null, 2));

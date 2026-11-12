// pumpfun-top5-forecast-01 — momentum / lifecycle forecaster (Node 22, standard library only).
//
// Reads the house's 10-minute market-cap snapshots in /input (or $MAKEFUN_INPUT_DIR)
// and writes one forecast to /output/forecast.json (or $MAKEFUN_OUTPUT_DIR):
//   for each of the 5 places, probabilities for each coin named being at that place,
//   the top 5 in order, and per-place market-cap bin probabilities.
//
// Model (fitted at runtime from the history itself, so it adapts to the practice's
// 1-hour miniature and the final's 11-hour history alike):
//   - each coin's log10 market cap (in SOL) is tracked across marks, with venue
//     (curve / migrating / pool), age and recent slope;
//   - 10-minute log returns are pooled by (venue, age band) into drift/volatility
//     buckets, shrunk toward the global mean, with coins that drop out of the
//     rows imputed just below the rank-200 cutoff (removes survivorship bias);
//   - each candidate's log cap is forecast to each answer read by integrating the
//     age-band drift over the horizon, plus a damped momentum term that fades
//     after 60 minutes; volatility grows with the horizon;
//   - a seeded Monte Carlo (deterministic given the input) simulates the candidate
//     set at every answer read, ranks it, and averages place probabilities and
//     place-cap distributions over the reads;
//   - coins born after the history ends cannot be named, so each place's named
//     probabilities are deflated by an outsider rate fitted from the history's own
//     newcomer churn, leaving the rest unnamed for them; a coin is only named at
//     a place when its probability beats the unnamed per-coin share.
//   - place sizes are lognormal around the simulated place caps, widened for
//     outsider variance and model error.
//
// Fully deterministic: the PRNG is seeded from a hash of the input, no clock,
// no Math.random. Two runs on the same history write byte-identical output.

import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/* ---------------- tunables ---------------- */
const MAX_CANDIDATES = 350; // coins simulated
const MC_DRAWS_TARGET = 3_000_000; // paths * candidates * horizons, roughly
const MIN_PATHS = 400;
const MAX_PATHS = 5000;
const SIGMA0 = 0.2; // base uncertainty at the cut (log10)
const SIGMA_MIN = 0.25; // per-coin sigma floor at any horizon
const SIGMA_MAX = 1.2; // per-coin sigma cap
const CURVE_SIGMA_BONUS = 0.15; // curve coins are bimodal (die or graduate+pump)
const MOMENTUM_WEIGHT = 0.35; // how much of a coin's own recent slope survives shrinkage
const MOMENTUM_HALF_MIN = 60; // momentum fades after this many minutes of horizon
const MOMENTUM_DAMP = 0.5; // momentum is noisy: damp it
const OUTSIDER_Q_CAP = 0.65; // max share of a place left to post-cut coins
const OUTSIDER_PLACE_MULT = [0.7, 0.85, 1.0, 1.15, 1.25]; // newcomers take lower places more often
const SIZE_SD_MIN = 0.35; // never narrower than the house baseline's size sd
const SIZE_SD_WIDEN = 1.25; // widen simulated place-cap sd for outsiders + model error
const SIZE_MEAN_SHIFT = 0.05; // outsiders push true place caps above the simulated ones
const MICRO = 1_000_000;

/* ---------------- deterministic PRNG ---------------- */
function fnv1a(str: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
/** Box-Muller normal draw from a uniform PRNG (spare cached). */
function makeNormal(rand: () => number): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) { const v = spare; spare = null; return v; }
    let u = 0, v = 0;
    while (u === 0) u = rand();
    v = rand();
    const r = Math.sqrt(-2 * Math.log(u));
    const t = 2 * Math.PI * v;
    spare = r * Math.sin(t);
    return r * Math.cos(t);
  };
}

/* ---------------- normal CDF helpers (starter's erfc) ---------------- */
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r = t * Math.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
  return x >= 0 ? r : 2 - r;
}
const above = (x: number, mean: number, sd: number): number => (x === -Infinity ? 1 : x === Infinity ? 0 : 0.5 * erfc((x - mean) / (sd * Math.SQRT2)));

/* ---------------- types ---------------- */
interface Row {
  rank: number;
  mint: string;
  market_cap_lamports: string | number;
  venue: string;
  created_at: string;
  complete: boolean;
  mayhem: boolean;
}
interface Mark { at: string; ok: boolean; rows: Row[] | null; }
interface Obs { mark: number; t: number; logcap: number; venue: string; }
interface Coin {
  mint: string;
  createdAt: number;
  obs: Obs[];
  lastMark: number;
  lastLogcap: number;
  lastVenue: string;
  mayhem: boolean;
}

const task = JSON.parse(readFileSync(join(INPUT_DIR, 'task.json'), 'utf8')) as {
  places: number; output: string; size: { edges: number[] }; t0: string; history_until: string; reads?: string[];
};
const lines = readFileSync(join(INPUT_DIR, 'history.ndjson'), 'utf8').split('\n').filter((l) => l !== '');
const marks = lines.map((l) => JSON.parse(l) as Mark).filter((m) => m.ok === true && Array.isArray(m.rows));
if (marks.length === 0) throw new Error('the history holds no ok mark');
const PLACES = task.places;
const EDGES = task.size.edges;
const lastMark = marks[marks.length - 1]!;
const cutTime = Date.parse(lastMark.at);
const horizonsMin: number[] = (task.reads && task.reads.length > 0
  ? task.reads.map((r) => (Date.parse(r) - cutTime) / 60000)
  : [420, 480, 540, 600, 660, 720, 780]).filter((h) => h > 0);
if (horizonsMin.length === 0) horizonsMin.push(60);
const H = horizonsMin.length;

/* ---------------- build coin series ---------------- */
const coins = new Map<string, Coin>();
const logcapOf = (r: Row): number => Math.log10(Number(BigInt(r.market_cap_lamports)) / 1e9);
for (let mi = 0; mi < marks.length; mi++) {
  const m = marks[mi]!;
  for (const r of m.rows!) {
    let c = coins.get(r.mint);
    if (!c) {
      c = { mint: r.mint, createdAt: Date.parse(r.created_at), obs: [], lastMark: -1, lastLogcap: 0, lastVenue: r.venue, mayhem: r.mayhem };
      coins.set(r.mint, c);
    }
    c.obs.push({ mark: mi, t: Date.parse(m.at), logcap: logcapOf(r), venue: r.venue });
    c.lastMark = mi; c.lastLogcap = logcapOf(r); c.lastVenue = r.venue;
  }
}
const coinList = [...coins.values()];
const lastRows = lastMark.rows!;
const rank200Cap = lastRows.length > 0 ? logcapOf(lastRows[lastRows.length - 1]!) : 0;

/* ---------------- return buckets by (venue, age band) ---------------- */
const AGE_BANDS = [0, 30, 60, 120, 240, 480, Infinity]; // minutes
function ageBand(ageMin: number): number {
  for (let i = 0; i < AGE_BANDS.length - 1; i++) if (ageMin < AGE_BANDS[i + 1]!) return i;
  return AGE_BANDS.length - 2;
}
interface Bucket { n: number; sum: number; sumsq: number; }
const buckets = new Map<string, Bucket>();
let gN = 0, gSum = 0, gSumsq = 0;
function addReturn(venue: string, ageMin: number, d: number): void {
  const key = `${venue}|${ageBand(ageMin)}`;
  let b = buckets.get(key);
  if (!b) { b = { n: 0, sum: 0, sumsq: 0 }; buckets.set(key, b); }
  b.n++; b.sum += d; b.sumsq += d * d;
  gN++; gSum += d; gSumsq += d * d;
}
for (let mi = 0; mi < marks.length - 1; mi++) {
  const cur = new Map(marks[mi]!.rows!.map((r) => [r.mint, r]));
  const nxt = new Map(marks[mi + 1]!.rows!.map((r) => [r.mint, r]));
  const nextCutoff = logcapOf(marks[mi + 1]!.rows![marks[mi + 1]!.rows!.length - 1]!) - 0.1;
  for (const [mint, r] of cur) {
    const c = coins.get(mint)!;
    const ageMin = (Date.parse(marks[mi]!.at) - c.createdAt) / 60000;
    const n = nxt.get(mint);
    if (n) addReturn(r.venue, ageMin, logcapOf(n) - logcapOf(r));
    else addReturn(r.venue, ageMin, nextCutoff - logcapOf(r)); // dropped out of the rows: imputed
  }
}
const gMean = gN > 0 ? gSum / gN : 0;
const gVar = gN > 0 ? Math.max(gSumsq / gN - gMean * gMean, 0.0025) : 0.02;
const gSd = Math.sqrt(gVar);
function bucketStats(venue: string, band: number): { mean: number; sd: number } {
  // fall back to the nearest band with data (older bands first), else the venue's global, else priors
  for (let b = band; b >= 0; b--) {
    const bk = buckets.get(`${venue}|${b}`);
    if (bk && bk.n >= 3) {
      const w = bk.n / (bk.n + 15);
      const mean = w * (bk.sum / bk.n) + (1 - w) * gMean;
      const v = w * Math.max(bk.sumsq / bk.n - (bk.sum / bk.n) ** 2, 0.0025) + (1 - w) * gVar;
      return { mean, sd: Math.min(Math.max(Math.sqrt(v), 0.05), 0.5) };
    }
  }
  for (let b = band + 1; b < AGE_BANDS.length - 1; b++) {
    const bk = buckets.get(`${venue}|${b}`);
    if (bk && bk.n >= 3) {
      const w = bk.n / (bk.n + 15);
      const mean = w * (bk.sum / bk.n) + (1 - w) * gMean;
      const v = w * Math.max(bk.sumsq / bk.n - (bk.sum / bk.n) ** 2, 0.0025) + (1 - w) * gVar;
      return { mean, sd: Math.min(Math.max(Math.sqrt(v), 0.05), 0.5) };
    }
  }
  return venue === 'pool' ? { mean: 0.02, sd: 0.15 } : { mean: -0.01, sd: 0.12 };
}

/* ---------------- outsider rate from newcomer churn ---------------- */
// For mark pairs (i, j), j > i, the share of the top 5 at j born after mark i's
// time. Fit q(dt) = 1 - exp(-lambda * dt) through the mean point, capped.
let pairQ = 0, pairN = 0, pairDt = 0;
for (let i = 1; i < marks.length; i++) {
  const tI = Date.parse(marks[i]!.at);
  const top100I = new Set(marks[i]!.rows!.slice(0, 100).map((r) => r.mint));
  void top100I;
  for (let j = i + 1; j < marks.length; j++) {
    const dt = (Date.parse(marks[j]!.at) - tI) / 60000;
    let newcomers = 0;
    for (const r of marks[j]!.rows!.slice(0, PLACES)) {
      if (Date.parse(r.created_at) > tI) newcomers++;
    }
    pairQ += newcomers / PLACES; pairN++; pairDt += dt;
  }
}
let lambda = 0.02; // per minute, prior
if (pairN > 0) {
  const meanQ = pairQ / pairN;
  const meanDt = pairDt / pairN;
  if (meanQ > 0.02 && meanQ < 0.95 && meanDt > 0) lambda = -Math.log(1 - meanQ) / meanDt;
}
lambda = Math.min(Math.max(lambda, 0.005), 0.06);
const meanHorizon = horizonsMin.reduce((a, b) => a + b, 0) / H;
const qBase = Math.min(1 - Math.exp(-lambda * meanHorizon), OUTSIDER_Q_CAP);

/* ---------------- candidate selection ---------------- */
function recentSlope(c: Coin): number {
  const obs = c.obs;
  const n = Math.min(obs.length, 7); // up to the last 60 minutes
  if (n < 2) return 0;
  const a = obs[obs.length - n]!, b = obs[obs.length - 1]!;
  const steps = Math.max(1, b.mark - a.mark);
  return (b.logcap - a.logcap) / steps; // per 10 minutes
}
interface Cand {
  mint: string; L: number; venue: string; ageCutMin: number; slope: number;
  absentMarks: number; mu: number[]; sigma: number[];
}
const scored = coinList.map((c) => {
  const slope = recentSlope(c);
  const absent = marks.length - 1 - c.lastMark;
  const score = c.lastLogcap + 0.3 * slope * 10 + (c.lastVenue === 'pool' ? 0.1 : 0) - 0.05 * absent;
  return { c, slope, absent, score };
});
scored.sort((a, b) => b.score - a.score);
const top100Last = new Set(lastRows.slice(0, 100).map((r) => r.mint));
const chosen = new Map<string, (typeof scored)[number]>();
for (const s of scored.slice(0, MAX_CANDIDATES)) chosen.set(s.c.mint, s);
for (const m of top100Last) {
  const c = coins.get(m);
  if (c && !chosen.has(m)) chosen.set(m, { c, slope: recentSlope(c), absent: marks.length - 1 - c.lastMark, score: 0 });
}
const cands: Cand[] = [...chosen.values()].map(({ c, slope, absent }) => {
  const ageCutMin = Math.max(0, (cutTime - c.createdAt) / 60000);
  // a coin absent from the last marks has decayed below the cutoff since last seen
  const L = c.lastMark === marks.length - 1 ? c.lastLogcap : Math.min(c.lastLogcap - 0.05 * absent, rank200Cap);
  return { mint: c.mint, L, venue: c.lastVenue, ageCutMin, slope, absentMarks: absent, mu: [], sigma: [] };
});

/* ---------------- per-coin, per-horizon forecast ---------------- */
for (const cd of cands) {
  const bandNow = ageBand(cd.ageCutMin);
  const stNow = bucketStats(cd.venue, bandNow);
  // damped momentum: the coin's own recent slope, relative to its bucket drift
  const mom = (cd.slope * MOMENTUM_WEIGHT + (1 - MOMENTUM_WEIGHT) * stNow.mean - stNow.mean) * MOMENTUM_DAMP;
  for (const h of horizonsMin) {
    const steps = Math.max(1, Math.round(h / 10));
    // integrate the age-band drift over the horizon, 10 minutes at a time
    let driftSum = 0;
    for (let s = 0; s < steps; s++) {
      const age = cd.ageCutMin + s * 10;
      driftSum += bucketStats(cd.venue, ageBand(age)).mean;
    }
    const momSteps = Math.min(steps, Math.round(MOMENTUM_HALF_MIN / 10));
    const mu = cd.L + driftSum + mom * momSteps;
    const sdStep = bucketStats(cd.venue, bandNow).sd;
    let sigma = Math.sqrt(sdStep * sdStep * steps + SIGMA0 * SIGMA0);
    if (cd.venue !== 'pool') sigma += CURVE_SIGMA_BONUS;
    sigma += 0.02 * cd.absentMarks; // stale observations are less certain
    cd.mu.push(mu);
    cd.sigma.push(Math.min(Math.max(sigma, SIGMA_MIN), SIGMA_MAX));
  }
}

/* ---------------- Monte Carlo over candidates ---------------- */
const NC = cands.length;
const paths = Math.min(MAX_PATHS, Math.max(MIN_PATHS, Math.floor(MC_DRAWS_TARGET / Math.max(1, NC * H))));
const seedStr = lastRows.map((r) => r.mint).join(',') + '|' + task.t0 + '|' + horizonsMin.join(',');
const rand = mulberry32(fnv1a(seedStr));
const gauss = makeNormal(rand);
const placeProb = new Float64Array(NC * PLACES); // [coin * PLACES + place]
const placeCapSamples: number[][] = Array.from({ length: PLACES }, () => []);
const muArr: Float64Array[] = [];
const sdArr: Float64Array[] = [];
for (let hj = 0; hj < H; hj++) {
  muArr.push(Float64Array.from(cands.map((c) => c.mu[hj]!)));
  sdArr.push(Float64Array.from(cands.map((c) => c.sigma[hj]!)));
}
const draw = new Float64Array(NC);
for (let hj = 0; hj < H; hj++) {
  const mu = muArr[hj]!, sd = sdArr[hj]!;
  for (let p = 0; p < paths; p++) {
    for (let i = 0; i < NC; i++) draw[i] = mu[i]! + sd[i]! * gauss();
    // top 5 by linear scan
    const topIdx = [-1, -1, -1, -1, -1];
    const topVal = [-Infinity, -Infinity, -Infinity, -Infinity, -Infinity];
    for (let i = 0; i < NC; i++) {
      const v = draw[i]!;
      for (let k = 0; k < PLACES; k++) {
        if (v > topVal[k]!) {
          for (let s = PLACES - 1; s > k; s--) { topVal[s] = topVal[s - 1]!; topIdx[s] = topIdx[s - 1]!; }
          topVal[k] = v; topIdx[k] = i;
          break;
        }
      }
    }
    for (let k = 0; k < PLACES; k++) {
      const idx = topIdx[k]!;
      if (idx >= 0) {
        placeProb[idx * PLACES + k]! += 1 / paths / H;
        placeCapSamples[k]!.push(topVal[k]!);
      }
    }
  }
}

/* ---------------- deflate by the outsider rate, pick named sets ---------------- */
const probabilities: Array<{ mint: string; place: number; p: number }> = [];
for (let k = 0; k < PLACES; k++) {
  const qk = Math.min(qBase * OUTSIDER_PLACE_MULT[k]!, 0.8);
  const deflate = 1 - qk;
  // candidates sorted by deflated probability at this place
  const order = cands.map((c, i) => ({ i, p: placeProb[i * PLACES + k]! * deflate })).sort((a, b) => b.p - a.p);
  // greedy named set: name a coin only while its p beats the unnamed per-coin share
  let named: Array<{ i: number; p: number }> = [];
  let unnamed = 1;
  for (const o of order) {
    const share = unnamed / Math.max(1, 100 - named.length);
    if (o.p <= share && named.length >= 5) break;
    if (o.p <= 0.001) break;
    named.push(o);
    unnamed -= o.p;
    if (named.length >= 100) break;
  }
  // re-check: drop any named coin whose p is below the final unnamed share
  for (let iter = 0; iter < 5; iter++) {
    const share = unnamed / Math.max(1, 100 - named.length);
    const kept = named.filter((o) => o.p >= share);
    if (kept.length === named.length) break;
    unnamed = 1 - kept.reduce((a, o) => a + o.p, 0);
    named = kept;
  }
  for (const o of named) {
    probabilities.push({ mint: cands[o.i]!.mint, place: k + 1, p: Math.floor(o.p * MICRO) / MICRO });
  }
}

/* ---------------- sizes from simulated place caps ---------------- */
const sizes: Array<{ place: number; bin: number; p: number }> = [];
for (let k = 0; k < PLACES; k++) {
  const s = placeCapSamples[k]!;
  if (s.length === 0) continue;
  const mean = s.reduce((a, b) => a + b, 0) / s.length + SIZE_MEAN_SHIFT;
  const sdRaw = Math.sqrt(s.reduce((a, b) => a + (b - (mean - SIZE_MEAN_SHIFT)) ** 2, 0) / s.length);
  const sd = Math.max(sdRaw * SIZE_SD_WIDEN, SIZE_SD_MIN);
  // Name bins greedily by p, only while a bin's p beats the unnamed per-bin
  // share; a tail bin left unnamed scores (1 - sum) / (82 - named) instead of
  // a floored ln, so naming a bin below that share hurts.
  const allBins: Array<{ bin: number; p: number }> = [];
  for (let bin = 0; bin <= EDGES.length; bin++) {
    const lo = bin === 0 ? -Infinity : EDGES[bin - 1]!;
    const hi = bin === EDGES.length ? Infinity : EDGES[bin]!;
    const u = Math.floor((above(lo, mean, sd) - above(hi, mean, sd)) * MICRO);
    if (u > 0) allBins.push({ bin, p: u / MICRO });
  }
  allBins.sort((a, b) => b.p - a.p);
  let namedBins: Array<{ bin: number; p: number }> = [];
  let sizeUnnamed = 1;
  for (const b of allBins) {
    const share = sizeUnnamed / Math.max(1, 82 - namedBins.length);
    if (b.p <= share && namedBins.length >= 5) break;
    if (b.p <= 0.0005) break;
    namedBins.push(b);
    sizeUnnamed -= b.p;
  }
  for (let iter = 0; iter < 5; iter++) {
    const share = sizeUnnamed / Math.max(1, 82 - namedBins.length);
    const kept = namedBins.filter((b) => b.p >= share);
    if (kept.length === namedBins.length) break;
    sizeUnnamed = 1 - kept.reduce((a, b) => a + b.p, 0);
    namedBins = kept;
  }
  for (const b of namedBins) sizes.push({ place: k + 1, bin: b.bin, p: b.p });
}

/* ---------------- top5 by expected rank ---------------- */
const expRank = cands.map((c, i) => {
  let e = 0, mass = 0;
  for (let k = 0; k < PLACES; k++) {
    const p = placeProb[i * PLACES + k]! * (1 - Math.min(qBase * OUTSIDER_PLACE_MULT[k]!, 0.8));
    e += (k + 1) * p; mass += p;
  }
  e += (PLACES + 1) * (1 - mass);
  return { mint: c.mint, e };
});
expRank.sort((a, b) => a.e - b.e);
const top5 = expRank.slice(0, PLACES).map((r) => r.mint);

mkdirSync(OUTPUT_DIR, { recursive: true });
writeFileSync(join(OUTPUT_DIR, task.output), `${JSON.stringify({ probabilities, top5, sizes })}\n`);

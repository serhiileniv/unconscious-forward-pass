/**
 * Precompute a semantic layout for the token cloud.
 *
 * PCA answers "what are the widest directions" and on 768 dimensions that keeps
 * 3.2% of the variance, which is not enough for proximity on screen to mean
 * anything. The question worth answering is different: which tokens does the
 * model treat as neighbours? That is a local property, and a neighbour-graph
 * layout preserves it far better than any linear projection can.
 *
 * So: approximate k-nearest-neighbours in the model's own embedding space, then
 * a UMAP-style force layout that pulls neighbours together and pushes everything
 * else apart. The result is a map with real regions — weekdays, numbers, cities,
 * anatomy — because those tokens genuinely are each other's neighbours.
 *
 *   node tools/layout.mjs public/model/gpt2
 *
 * Writes layout.bin: float32 xyz per token, then the neighbour graph.
 */
import { writeFileSync } from 'node:fs'
import { readFileSync } from 'node:fs'
import { GPT2 } from '../src/model/gpt2'
import { mulberry32 } from '../src/model/math'

const DIR = process.argv[2] ?? 'public/model/gpt2'
;(globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
  const buf = readFileSync(`${DIR}/${String(url).split('/').pop()!}`)
  return { ok: true, headers: { get: () => String(buf.byteLength) }, body: null,
    json: async () => JSON.parse(buf.toString()), text: async () => buf.toString(),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) }
}

const K = 12
const DIMS = 64
const EPOCHS = 200

const t0 = Date.now()
const m = await GPT2.load(DIR, 0)
const V = m.cfg.vocabSize
const D = m.cfg.nEmbd
console.log(`${V} tokens, ${D} dims`)

// Cosine space: unit rows, so a dot product is a similarity.
const U = new Float32Array(V * D)
for (let v = 0; v < V; v++) {
  let n = 0
  for (let k = 0; k < D; k++) n += m.wte[v * D + k] ** 2
  n = Math.sqrt(n) || 1
  for (let k = 0; k < D; k++) U[v * D + k] = m.wte[v * D + k] / n
}

// A random projection down to 64 dims makes neighbour search affordable.
// Candidates found here are re-scored in the full space before being kept.
const rand = mulberry32(7)
const R = new Float32Array(D * DIMS)
for (let i = 0; i < R.length; i++) {
  let u = 0
  while (u === 0) u = rand()
  R[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand())
}
const P = new Float32Array(V * DIMS)
for (let v = 0; v < V; v++) {
  for (let c = 0; c < DIMS; c++) {
    let s = 0
    for (let k = 0; k < D; k++) s += U[v * D + k] * R[k * DIMS + c]
    P[v * DIMS + c] = s
  }
}
console.log(`projected to ${DIMS}d  ${((Date.now() - t0) / 1000).toFixed(1)}s`)

const simFull = (a: number, b: number): number => {
  let s = 0
  for (let k = 0; k < D; k++) s += U[a * D + k] * U[b * D + k]
  return s
}
const simLow = (a: number, b: number): number => {
  let s = 0
  for (let k = 0; k < DIMS; k++) s += P[a * DIMS + k] * P[b * DIMS + k]
  return s
}

// --- candidate generation by random projection trees ---
//
// Neighbour search from a random start does not work here: friend-of-a-friend
// only ever explores a couple of percent of 50,000 tokens and settles on
// nonsense. Random projection trees split the space by random hyperplanes until
// leaves are small, so tokens that land in the same leaf are already plausible
// neighbours. A dozen independent trees give every token a few hundred good
// candidates, and refinement afterwards only has to polish.
const LEAF = 48
const TREES = 14

const nbr = new Int32Array(V * K).fill(-1)
const nsim = new Float32Array(V * K).fill(-2)

function offer(v: number, c: number): boolean {
  if (c === v) return false
  let worst = 0
  for (let j = 1; j < K; j++) if (nsim[v * K + j] < nsim[v * K + worst]) worst = j
  const s = simLow(v, c)
  if (s <= nsim[v * K + worst]) return false
  for (let j = 0; j < K; j++) if (nbr[v * K + j] === c) return false
  nbr[v * K + worst] = c
  nsim[v * K + worst] = s
  return true
}

const scratch = new Float32Array(V)
function split(idx: Int32Array, lo: number, hi: number): void {
  const n = hi - lo
  if (n <= LEAF) {
    for (let a = lo; a < hi; a++) for (let b = a + 1; b < hi; b++) {
      offer(idx[a], idx[b])
      offer(idx[b], idx[a])
    }
    return
  }
  // Split along the line between two random members: cheap, and it follows the
  // data's own shape rather than an arbitrary axis.
  const p1 = idx[lo + Math.floor(rand() * n)]
  const p2 = idx[lo + Math.floor(rand() * n)]
  for (let a = lo; a < hi; a++) {
    const v = idx[a]
    let s = 0
    for (let k = 0; k < DIMS; k++) s += P[v * DIMS + k] * (P[p1 * DIMS + k] - P[p2 * DIMS + k])
    scratch[v] = s
  }
  const slice = Array.from(idx.subarray(lo, hi)).sort((a, b) => scratch[a] - scratch[b])
  idx.set(slice, lo)
  const mid = lo + (n >> 1)
  split(idx, lo, mid)
  split(idx, mid, hi)
}

for (let t = 0; t < TREES; t++) {
  const idx = new Int32Array(V)
  for (let i = 0; i < V; i++) idx[i] = i
  split(idx, 0, V)
  console.log(`  tree ${t + 1}/${TREES}  ${((Date.now() - t0) / 1000).toFixed(1)}s`)
}

// Polish: now that the graph carries real signal, friend-of-a-friend helps.
for (let it = 0; it < 4; it++) {
  let changed = 0
  for (let v = 0; v < V; v++) {
    for (let j = 0; j < K; j++) {
      const n1 = nbr[v * K + j]
      if (n1 < 0) continue
      for (let l = 0; l < K; l++) {
        const c = nbr[n1 * K + l]
        if (c >= 0 && offer(v, c)) changed++
      }
    }
  }
  console.log(`  refine pass ${it + 1}: ${changed} improvements  ${((Date.now() - t0) / 1000).toFixed(1)}s`)
  if (changed < V / 200) break
}

// Re-score in the full space so the graph reflects the real embedding, not the
// projection that made the search cheap.
for (let v = 0; v < V; v++) for (let j = 0; j < K; j++) {
  if (nbr[v * K + j] < 0) { nbr[v * K + j] = v; nsim[v * K + j] = -1; continue }
  nsim[v * K + j] = simFull(v, nbr[v * K + j])
}

for (const w of [' Monday', ' seven', ' Paris', ' kidney']) {
  const id = m.tok.encode(w)[0]
  const list = Array.from({ length: K }, (_, j) => j).sort((a, b) => nsim[id * K + b] - nsim[id * K + a])
  console.log(`  ${JSON.stringify(w).padEnd(11)} -> ${list.slice(0, 6).map((j) => JSON.stringify(m.tok.piece(nbr[id * K + j]))).join(' ')}`)
}

// --- layout: spectral embedding of the neighbour graph ---
//
// A force simulation was tried first and made things worse than random: with
// five negative samples per edge, every node took sixty repulsions against
// twelve attractions each epoch and the graph simply inflated. Measured, true
// neighbours ended up further apart than random pairs.
//
// The spectral embedding has no such knobs. Minimising the sum of w_ij times the
// squared distance between neighbours, subject to orthonormality, is solved
// exactly by the leading eigenvectors of the normalised adjacency — so connected
// tokens land together by construction rather than by tuning.
const deg = new Float32Array(V)
const adj: number[][] = Array.from({ length: V }, () => [])
const adjW: number[][] = Array.from({ length: V }, () => [])
for (let v = 0; v < V; v++) {
  for (let j = 0; j < K; j++) {
    const u = nbr[v * K + j]
    if (u < 0 || u === v) continue
    const w = Math.max(0, nsim[v * K + j])
    if (w <= 0) continue
    adj[v].push(u); adjW[v].push(w)
    adj[u].push(v); adjW[u].push(w)
    deg[v] += w
    deg[u] += w
  }
}
const invSqrtDeg = new Float32Array(V)
for (let v = 0; v < V; v++) invSqrtDeg[v] = deg[v] > 0 ? 1 / Math.sqrt(deg[v]) : 0

const COMP = 4
let Z = new Float32Array(V * COMP)
for (let i = 0; i < Z.length; i++) Z[i] = rand() * 2 - 1
const Y = new Float32Array(V * COMP)

// The trivial eigenvector of the normalised adjacency is known in closed form;
// projecting it out each iteration leaves the components that carry structure.
const trivial = new Float32Array(V)
{
  let n = 0
  for (let v = 0; v < V; v++) { trivial[v] = Math.sqrt(deg[v]); n += deg[v] }
  n = Math.sqrt(n) || 1
  for (let v = 0; v < V; v++) trivial[v] /= n
}
function deflate(M: Float32Array): void {
  for (let c = 0; c < COMP; c++) {
    let d = 0
    for (let v = 0; v < V; v++) d += M[v * COMP + c] * trivial[v]
    for (let v = 0; v < V; v++) M[v * COMP + c] -= d * trivial[v]
  }
}
function orthonormalizeCols(M: Float32Array): void {
  for (let c = 0; c < COMP; c++) {
    for (let p2 = 0; p2 < c; p2++) {
      let d = 0
      for (let v = 0; v < V; v++) d += M[v * COMP + c] * M[v * COMP + p2]
      for (let v = 0; v < V; v++) M[v * COMP + c] -= d * M[v * COMP + p2]
    }
    let n = 0
    for (let v = 0; v < V; v++) n += M[v * COMP + c] ** 2
    n = Math.sqrt(n) || 1
    for (let v = 0; v < V; v++) M[v * COMP + c] /= n
  }
}
deflate(Z); orthonormalizeCols(Z)
for (let it = 0; it < 220; it++) {
  Y.fill(0)
  for (let v = 0; v < V; v++) {
    const list = adj[v]
    const wts = adjW[v]
    const sv = invSqrtDeg[v]
    for (let a = 0; a < list.length; a++) {
      const u = list[a]
      const w = wts[a] * sv * invSqrtDeg[u]
      for (let c = 0; c < COMP; c++) Y[v * COMP + c] += w * Z[u * COMP + c]
    }
  }
  Z.set(Y)
  deflate(Z)
  orthonormalizeCols(Z)
  if (it % 60 === 0) console.log(`  spectral iter ${it}  ${((Date.now() - t0) / 1000).toFixed(1)}s`)
}

const pos = new Float32Array(V * 3)
for (let v = 0; v < V; v++) for (let c = 0; c < 3; c++) pos[v * 3 + c] = Z[v * COMP + c + 1]

// Spectral coordinates pile most tokens into a dense core, so scale to a
// workable range before refining rather than flattening each axis by rank —
// ranking spreads points evenly but pulls clusters apart in the process.
{
  let rms0 = 0
  for (let i = 0; i < pos.length; i++) rms0 += pos[i] * pos[i]
  rms0 = Math.sqrt(rms0 / V) || 1
  for (let i = 0; i < pos.length; i++) pos[i] = (pos[i] / rms0) * 6
}

// --- local refinement, spectral-initialised ---
//
// The same attraction and repulsion as before, but started from a sensible
// arrangement instead of noise, with one negative sample per edge instead of
// five and a small step. The earlier attempt failed because sixty repulsions a
// node against twelve attractions inflates any graph, however good the graph is.
const A = 1.577
const B = 0.895
const clamp = (x: number): number => (x > 1.5 ? 1.5 : x < -1.5 ? -1.5 : x)
// Refinement measurably worsened the ratio of neighbour distance to random
// distance (0.83 against 0.50), so the spectral solution stands on its own.
const REFINE = Number(process.env.REFINE ?? 200)
// Attraction and repulsion have to be balanced by hand here; real UMAP does it
// through weighted edge sampling. Measured, the default repulsion overwhelmed
// attraction and inflated the graph.
const REP = Number(process.env.REP ?? 0.4)
for (let e = 0; e < REFINE; e++) {
  const alpha = 0.25 * (1 - e / REFINE)
  for (let v = 0; v < V; v++) {
    for (let j = 0; j < K; j++) {
      const u = nbr[v * K + j]
      if (u < 0 || u === v || nsim[v * K + j] < 0.15) continue
      let dx = pos[v * 3] - pos[u * 3]
      let dy = pos[v * 3 + 1] - pos[u * 3 + 1]
      let dz = pos[v * 3 + 2] - pos[u * 3 + 2]
      let d2 = dx * dx + dy * dy + dz * dz
      if (d2 < 1e-4) d2 = 1e-4
      const g = (-2 * A * B * Math.pow(d2, B - 1)) / (1 + A * Math.pow(d2, B))
      const gx = clamp(g * dx) * alpha
      const gy = clamp(g * dy) * alpha
      const gz = clamp(g * dz) * alpha
      pos[v * 3] += gx; pos[v * 3 + 1] += gy; pos[v * 3 + 2] += gz
      pos[u * 3] -= gx; pos[u * 3 + 1] -= gy; pos[u * 3 + 2] -= gz

      const r = Math.floor(rand() * V)
      if (r === v) continue
      dx = pos[v * 3] - pos[r * 3]
      dy = pos[v * 3 + 1] - pos[r * 3 + 1]
      dz = pos[v * 3 + 2] - pos[r * 3 + 2]
      d2 = dx * dx + dy * dy + dz * dz
      if (d2 < 1e-4) d2 = 1e-4
      const rg = (REP * 2 * B) / ((0.001 + d2) * (1 + A * Math.pow(d2, B)))
      pos[v * 3] += clamp(rg * dx) * alpha
      pos[v * 3 + 1] += clamp(rg * dy) * alpha
      pos[v * 3 + 2] += clamp(rg * dz) * alpha
    }
  }
  if (e % 40 === 0) console.log(`  refine epoch ${e}/${REFINE}  ${((Date.now() - t0) / 1000).toFixed(1)}s`)
}

// Centre and scale to a unit-ish ball; the scene fits it to its own radius.
const c = [0, 0, 0]
for (let v = 0; v < V; v++) for (let k = 0; k < 3; k++) c[k] += pos[v * 3 + k] / V
let rms = 0
for (let v = 0; v < V; v++) for (let k = 0; k < 3; k++) { pos[v * 3 + k] -= c[k]; rms += pos[v * 3 + k] ** 2 }
rms = Math.sqrt(rms / V) || 1
for (let i = 0; i < pos.length; i++) pos[i] /= rms

const header = new Int32Array([V, K])
const out = Buffer.concat([
  Buffer.from(header.buffer),
  Buffer.from(pos.buffer),
  Buffer.from(nbr.buffer),
  Buffer.from(nsim.buffer),
])
writeFileSync(`${DIR}/layout.bin`, out)
console.log(`wrote ${DIR}/layout.bin  ${(out.length / 1048576).toFixed(1)} MB  ${((Date.now() - t0) / 1000).toFixed(0)}s`)

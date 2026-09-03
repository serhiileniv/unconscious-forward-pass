/**
 * How much of the model's own neighbourhood structure survives in the drawn map.
 *
 * For a linear projection the honest metric is variance and distance
 * correlation. For a neighbour-graph layout it is neighbourhood preservation:
 * of a token's true nearest neighbours in the model's embedding space, how many
 * are still among its nearest on screen. That is exactly what a viewer reads off
 * a cluster, so it is what the interface should report.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { GPT2 } from '../src/model/gpt2'
import { mulberry32 } from '../src/model/math'

const DIR = process.argv[2] ?? 'public/model/gpt2'
;(globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
  const buf = readFileSync(`${DIR}/${String(url).split('/').pop()!}`)
  return { ok: true, headers: { get: () => String(buf.byteLength) }, body: null,
    json: async () => JSON.parse(buf.toString()), text: async () => buf.toString(),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) }
}

const m = await GPT2.load(DIR, 0)
const V = m.cfg.vocabSize
const D = m.cfg.nEmbd

const raw = readFileSync(`${DIR}/layout.bin`)
const ab = raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength)
const head = new Int32Array(ab, 0, 2)
const pos = new Float32Array(ab.slice(8, 8 + head[0] * 12))

const U = new Float32Array(V * D)
for (let v = 0; v < V; v++) {
  let n = 0
  for (let k = 0; k < D; k++) n += m.wte[v * D + k] ** 2
  n = Math.sqrt(n) || 1
  for (let k = 0; k < D; k++) U[v * D + k] = m.wte[v * D + k] / n
}

const TOP = 10
const SAMPLE = 300
// Exact top-10 recall is a hard bar for 50,000 points in three dimensions. What
// a viewer actually reads off a cloud is a region, so measure at several radii:
// does a true neighbour land in the nearest 0.02%, 0.2%, 1%, 2% of the map?
const RADII = [10, 100, 500, 1000]
const rand = mulberry32(3)
const hitsAt = new Array(RADII.length).fill(0)
let neighbourDist = 0
let randomDist = 0
let hits = 0
for (let s = 0; s < SAMPLE; s++) {
  const v = Math.floor(rand() * V)
  const trueSim = new Float32Array(V)
  const drawn = new Float32Array(V)
  for (let u = 0; u < V; u++) {
    if (u === v) { trueSim[u] = -2; drawn[u] = Infinity; continue }
    let t = 0
    for (let k = 0; k < D; k++) t += U[v * D + k] * U[u * D + k]
    trueSim[u] = t
    drawn[u] = (pos[v * 3] - pos[u * 3]) ** 2 + (pos[v * 3 + 1] - pos[u * 3 + 1]) ** 2 + (pos[v * 3 + 2] - pos[u * 3 + 2]) ** 2
  }
  const realTop = Array.from(trueSim.keys()).sort((a, b) => trueSim[b] - trueSim[a]).slice(0, TOP)
  const byDrawn = Array.from(drawn.keys()).sort((a, b) => drawn[a] - drawn[b])
  RADII.forEach((r, ri) => {
    const set = new Set(byDrawn.slice(0, r))
    for (const t of realTop) if (set.has(t)) hitsAt[ri]++
  })
  hits = hitsAt[0]
  // Medians, not means: a handful of far-out tokens dominates a mean distance
  // and made an earlier comparison read backwards.
  const nd = realTop.map((t) => Math.sqrt(drawn[t])).sort((a, b) => a - b)
  const rd = Array.from({ length: TOP }, () => Math.sqrt(drawn[Math.floor(rand() * V)])).sort((a, b) => a - b)
  neighbourDist += nd[Math.floor(nd.length / 2)]
  randomDist += rd[Math.floor(rd.length / 2)]
}
console.log(`sampled ${SAMPLE} tokens; for each, its 10 true nearest neighbours in the model\n`)
RADII.forEach((r, ri) => {
  const pct = hitsAt[ri] / (SAMPLE * TOP)
  console.log(`  within the nearest ${String(r).padStart(4)} drawn points (${((r / V) * 100).toFixed(2)}% of the map): ${(pct * 100).toFixed(1)}%`)
})
const nMed = neighbourDist / SAMPLE
const rMed = randomDist / SAMPLE
console.log(`\n  median drawn distance to a true neighbour: ${nMed.toFixed(3)}`)
console.log(`  median drawn distance to a random token:   ${rMed.toFixed(3)}`)
console.log(`  ratio: ${(nMed / rMed).toFixed(2)}  (below 1 means neighbours land closer)`)
const preservation = hitsAt[2] / (SAMPLE * TOP)
writeFileSync(`${DIR}/layout.json`, JSON.stringify({
  preservation, top: TOP, sample: SAMPLE, radius: RADII[2],
  exact10: hits / (SAMPLE * TOP),
}, null, 2))
console.log(`wrote ${DIR}/layout.json`)

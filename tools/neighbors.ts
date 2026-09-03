import { readFileSync } from 'node:fs'
import { GPT2 } from '../src/model/gpt2'
import { pca3 } from '../src/model/math'

const DIR = 'public/model/gpt2'
;(globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
  const buf = readFileSync(`${DIR}/${String(url).split('/').pop()!}`)
  return { ok: true, headers: { get: () => String(buf.byteLength) }, body: null,
    json: async () => JSON.parse(buf.toString()), text: async () => buf.toString(),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) }
}

const m = await GPT2.load(DIR, 0)
const V = m.cfg.vocabSize
const D = m.cfg.nEmbd
const W = m.wte

// Unit-normalise so a dot product is cosine similarity.
const U = new Float32Array(V * D)
for (let v = 0; v < V; v++) {
  let n = 0
  for (let k = 0; k < D; k++) n += W[v * D + k] ** 2
  n = Math.sqrt(n) || 1
  for (let k = 0; k < D; k++) U[v * D + k] = W[v * D + k] / n
}

/** The single dominant direction, which in GPT-2 tracks frequency, not meaning. */
const { basis } = pca3(U, V, D, 8)
const stripped = new Float32Array(V * D)
for (let v = 0; v < V; v++) {
  let p = 0
  for (let k = 0; k < D; k++) p += U[v * D + k] * basis[k * 3]
  for (let k = 0; k < D; k++) stripped[v * D + k] = U[v * D + k] - p * basis[k * 3]
}

function near(src: Float32Array, word: string, n = 8): string {
  const id = m.tok.encode(word)[0]
  const out: { id: number; s: number }[] = []
  for (let v = 0; v < V; v++) {
    if (v === id) continue
    let s = 0
    for (let k = 0; k < D; k++) s += src[id * D + k] * src[v * D + k]
    out.push({ id: v, s })
  }
  out.sort((a, b) => b.s - a.s)
  return out.slice(0, n).map((o) => JSON.stringify(m.tok.piece(o.id))).join(' ')
}

for (const w of [' Monday', ' seven', ' Paris', ' happy', ' running', ' kidney']) {
  console.log(`${JSON.stringify(w).padEnd(11)} raw       ${near(U, w)}`)
  console.log(`${''.padEnd(11)} PC1 removed  ${near(stripped, w)}`)
}

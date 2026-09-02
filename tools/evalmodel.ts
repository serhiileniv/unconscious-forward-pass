import { readFileSync } from 'node:fs'
import { LlamaLM } from '../src/model/llama'
import { dot, softmax } from '../src/model/math'

const DIR = process.argv[2] ?? 'public/model/qwen'
const NAME = process.argv[3] ?? 'Qwen2.5-0.5B-Instruct'

;(globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
  const buf = readFileSync(`${DIR}/${String(url).split('/').pop()!}`)
  return {
    ok: true,
    headers: { get: () => String(buf.byteLength) },
    body: null,
    json: async () => JSON.parse(buf.toString()),
    text: async () => buf.toString(),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength),
  }
}

const t0 = Date.now()
const m = await LlamaLM.load(DIR, NAME, 6144)
console.log(`${NAME}: load ${Date.now() - t0}ms  L=${m.shape.nLayer} d=${m.shape.nEmbd} heads=${m.shape.nHead}/${m.shape.nKvHead} lens=${m.lensIds.length}`)
console.log(`heap ${(process.memoryUsage().heapUsed / 1073741824).toFixed(2)} GB  rss ${(process.memoryUsage().rss / 1073741824).toFixed(2)} GB`)

function greedy(prompt: string, n: number) {
  const ids = m.tok.encode(prompt)
  const st = m.newState(64)
  const t = Date.now()
  let logits = st.push(ids[0])
  for (let i = 1; i < ids.length; i++) logits = st.push(ids[i])
  const ms = (Date.now() - t) / ids.length
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    let best = 0
    for (let v = 1; v < logits.length; v++) if (logits[v] > logits[best]) best = v
    if (m.tok.isSpecial(best)) break
    out.push(best)
    logits = st.push(best)
  }
  return { text: m.tok.decode(out), ms, st, ids }
}

for (const p of ['2 + 2 =', 'The Eiffel Tower is located in the city of', 'The capital of Japan is']) {
  const r = greedy(p, 10)
  console.log(`\n${JSON.stringify(p)}\n  -> ${JSON.stringify(r.text)}   (${r.ms.toFixed(0)} ms/token)`)
}

// Does the lens agree with the model at each depth?
const probe = 'The Eiffel Tower is located in the city of'
const ids = m.tok.encode(probe)
const st = m.newState(64)
let real = st.push(ids[0])
for (let i = 1; i < ids.length; i++) real = st.push(ids[i])
const pos = st.length - 1
const nLens = m.lensIds.length

const realOver = new Float32Array(nLens)
for (let i = 0; i < nLens; i++) realOver[i] = real[m.lensIds[i]]
softmax(realOver)
const realTop = new Set(Array.from(realOver.keys()).sort((a, b) => realOver[b] - realOver[a]).slice(0, 10))

// Correctness check: attribution at the last layer must equal the real logits.
{
  const lastLayer = m.shape.nLayer - 1
  const row = st.lens[lastLayer].subarray(pos * nLens, (pos + 1) * nLens)
  let maxErr = 0
  for (let i = 0; i < nLens; i++) maxErr = Math.max(maxErr, Math.abs(row[i] - real[m.lensIds[i]]))
  console.log(`\nattribution vs real logits at final layer: max abs error ${maxErr.toExponential(2)}`)
}

console.log(`\nattribution faithfulness on ${JSON.stringify(probe)}`)
console.log('layer  overlap@10   KL      running top-3')
for (let l = 0; l < m.shape.nLayer; l++) {
  const p = new Float32Array(nLens)
  p.set(st.lens[l].subarray(pos * nLens, (pos + 1) * nLens))
  softmax(p)
  let kl = 0
  for (let i = 0; i < nLens; i++) if (p[i] > 1e-12 && realOver[i] > 1e-12) kl += p[i] * Math.log(p[i] / realOver[i])
  const top = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, 10)
  const hit = top.filter((i) => realTop.has(i)).length
  console.log(`  ${String(l).padStart(2)}     ${hit}/10      ${kl.toFixed(2).padStart(6)}   ${top.slice(0, 3).map((i) => JSON.stringify(m.lensPieces[i])).join(' ')}`)
}
void dot

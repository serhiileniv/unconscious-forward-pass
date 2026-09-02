import { readFileSync } from 'node:fs'
import { LlamaLM } from '../src/model/llama'

const DIR = process.argv[2] ?? 'public/model/qwen'
;(globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
  const buf = readFileSync(`${DIR}/${String(url).split('/').pop()!}`)
  return { ok: true, headers: { get: () => String(buf.byteLength) }, body: null,
    json: async () => JSON.parse(buf.toString()), text: async () => buf.toString(),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) }
}

const m = await LlamaLM.load(DIR, 'x', 6144)
const probe = process.argv[3] ?? 'The Eiffel Tower is located in the city of'
const ids = m.tok.encode(probe)
const st = m.newState(64)
let real = st.push(ids[0])
for (let i = 1; i < ids.length; i++) real = st.push(ids[i])
const pos = st.length - 1
const nLens = m.lensIds.length
const L = m.shape.nLayer

console.log(`${JSON.stringify(probe)} -> ${JSON.stringify(m.tok.piece(Array.from(real.keys()).sort((a,b)=>real[b]-real[a])[0]))}`)
console.log('\nPer-layer CONTRIBUTION (delta of the running total), top pushes per layer:')
const prev = new Float32Array(nLens)
for (let l = 0; l < L; l++) {
  const cur = st.lens[l].subarray(pos * nLens, (pos + 1) * nLens)
  const delta = new Float32Array(nLens)
  for (let i = 0; i < nLens; i++) delta[i] = cur[i] - prev[i]
  const top = Array.from(delta.keys()).sort((a, b) => delta[b] - delta[a]).slice(0, 4)
  const dn = Array.from(delta.keys()).sort((a, b) => delta[a] - delta[b]).slice(0, 2)
  console.log(`  L${String(l).padStart(2)} +  ${top.map((i) => `${JSON.stringify(m.lensPieces[i])}${delta[i].toFixed(1)}`).join(' ')}   |  down ${dn.map((i) => `${JSON.stringify(m.lensPieces[i])}${delta[i].toFixed(1)}`).join(' ')}`)
  prev.set(cur)
}

// Where does the winning token's logit actually come from?
const winner = m.lensPieces.findIndex((p) => p === ' Paris')
if (winner >= 0) {
  console.log(`\nContribution to " Paris" by layer:`)
  let running = 0
  const bars: string[] = []
  for (let l = 0; l < L; l++) {
    const cur = st.lens[l][pos * nLens + winner]
    const d = cur - running
    running = cur
    bars.push(`L${String(l).padStart(2)} ${d >= 0 ? '+' : ''}${d.toFixed(2)}`)
  }
  console.log('  ' + bars.join('  '))
}

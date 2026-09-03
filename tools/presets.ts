import { readFileSync } from 'node:fs'
import { GPT2 } from '../src/model/gpt2'
import { softmax } from '../src/model/math'

const DIR = 'public/model/gpt2'
;(globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
  const buf = readFileSync(`${DIR}/${String(url).split('/').pop()!}`)
  return { ok: true, headers: { get: () => String(buf.byteLength) }, body: null,
    json: async () => JSON.parse(buf.toString()), text: async () => buf.toString(),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) }
}
const m = await GPT2.load(DIR, 0)

function greedy(prompt: string, n = 10) {
  const ids = m.tok.encode(prompt)
  const st = m.newState(64)
  let logits = st.push(ids[0])
  for (let i = 1; i < ids.length; i++) logits = st.push(ids[i])
  const p0 = new Float32Array(logits.length); p0.set(logits); softmax(p0)
  const order0 = Array.from(p0.keys()).sort((a, b) => p0[b] - p0[a])
  let H = 0
  for (const v of p0) if (v > 0) H -= v * Math.log2(v)
  const out: number[] = []
  for (let i = 0; i < n; i++) {
    let best = 0
    for (let v = 1; v < logits.length; v++) if (logits[v] > logits[best]) best = v
    out.push(best)
    logits = st.push(best)
  }
  return {
    text: m.tok.decode(out),
    first: order0.slice(0, 4).map((i) => `${JSON.stringify(m.tok.piece(i))} ${(p0[i] * 100).toFixed(1)}%`).join('  '),
    play: Math.round(Math.pow(2, H)),
  }
}

const CANDIDATES = [
  'George Washington was the first president of the',
  'The first President of the United States was George',
  'In 1789 the United States elected its first president, George',
  'The first president of America was a man named George',
  'Two plus two equals',
  'If you add two and two you get',
]
for (const c of CANDIDATES) {
  const r = greedy(c)
  console.log(`${JSON.stringify(c)}`)
  console.log(`   -> ${JSON.stringify(r.text)}`)
  console.log(`   first token: ${r.first}   | ${r.play} words in play\n`)
}

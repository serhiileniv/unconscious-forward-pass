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

function dist(prompt: string) {
  const ids = m.tok.encode(prompt)
  const st = m.newState(64)
  let logits = st.push(ids[0])
  for (let i = 1; i < ids.length; i++) logits = st.push(ids[i])
  const p = new Float32Array(logits.length)
  p.set(logits)
  softmax(p)
  const order = Array.from(p.keys()).sort((a, b) => p[b] - p[a])
  let H = 0
  for (const v of p) if (v > 0) H -= v * Math.log2(v)
  return { p, order, H, ids }
}

for (const prompt of [
  'The Eiffel Tower is located in the city of',
  'The Eiffel Tower is in',
  'The Eiffel Tower stands in the centre of',
  'Q: In which city is the Eiffel Tower? A:',
]) {
  const { p, order, H, ids } = dist(prompt)
  console.log(`\n${JSON.stringify(prompt)}`)
  console.log(`  ${ids.length} tokens, entropy ${H.toFixed(2)} bits`)
  console.log(`  top: ${order.slice(0, 6).map((i) => `${JSON.stringify(m.tok.piece(i))} ${(p[i] * 100).toFixed(1)}%`).join('  ')}`)
}

// How often does temperature sampling actually pick the model's own best answer?
const { p, order } = dist('The Eiffel Tower is located in the city of')
const paris = m.tok.encode(' Paris')[0]
const top40 = order.slice(0, 40)
console.log(`\ngreedy answer: ${JSON.stringify(m.tok.piece(order[0]))}\n`)
console.log('temp   P(picks " Paris")   P(picks a wrong city)')
for (const T of [1.0, 0.8, 0.5, 0.3, 0.2, 0.1]) {
  const scaled = top40.map((i) => Math.pow(p[i], 1 / T))
  const tot = scaled.reduce((a, b) => a + b, 0)
  const pp = scaled[top40.indexOf(paris)] / tot
  let wrong = 0
  for (const w of [' London', ' Amsterdam', ' Berlin', ' Cologne', ' Rome', ' Vienna', ' Madrid']) {
    const id = m.tok.encode(w)[0]
    const k = top40.indexOf(id)
    if (k >= 0) wrong += scaled[k] / tot
  }
  console.log(`${T.toFixed(1)}          ${(pp * 100).toFixed(1)}%              ${(wrong * 100).toFixed(1)}%`)
}

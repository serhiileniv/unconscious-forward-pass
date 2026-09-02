import { readFileSync } from 'node:fs'
import { GPT2 } from '../src/model/gpt2'
import { BPETokenizer } from '../src/model/bpe'
import { Safetensors } from '../src/model/safetensors'

const DIR = 'public/model/gpt2'
const read = (f: string) => readFileSync(`${DIR}/${f}`)

// Stand in for the browser's fetch so the same loader code runs under node.
;(globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
  const name = String(url).split('/').pop()!
  const buf = read(name)
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
const model = await GPT2.load(DIR, 6144)
console.log(`load ${Date.now() - t0}ms  layers=${model.cfg.nLayer} d=${model.cfg.nEmbd} heads=${model.cfg.nHead}`)
console.log(`display vocab ${model.lensIds.length}: ${model.lensPieces.slice(0, 12).map((p) => JSON.stringify(p)).join(' ')}`)

const prompt = process.argv[2] ?? 'When Mary and John went to the store, John gave a drink to'
const ids = model.tok.encode(prompt)
console.log(`prompt ${JSON.stringify(prompt)} -> ${ids.length} tokens ${JSON.stringify(ids.map((i) => model.tok.piece(i)))}`)

const state = model.newState(32)
const t1 = Date.now()
let logits = new Float32Array()
for (const id of ids) logits = state.push(id)
const prefill = Date.now() - t1

const out: number[] = []
const t2 = Date.now()
for (let n = 0; n < 8; n++) {
  let best = 0
  for (let v = 1; v < logits.length; v++) if (logits[v] > logits[best]) best = v
  out.push(best)
  logits = state.push(best)
}
const gen = Date.now() - t2
console.log(`prefill ${ids.length} tok = ${prefill}ms (${(prefill / ids.length).toFixed(0)}ms/tok)`)
console.log(`generate 8 tok = ${gen}ms (${(gen / 8).toFixed(0)}ms/tok)`)
console.log(`greedy: ${JSON.stringify(prompt + model.tok.decode(out))}`)
console.log(`first generated token: ${JSON.stringify(model.tok.piece(out[0]))}`)

// The lens at the last prompt token — the position that predicts the answer.
const pos = ids.length - 1
for (const l of [0, 2, 4, 6, 8, 10, 11]) {
  const nLens = model.lensIds.length
  const row = state.lens[l].subarray(pos * nLens, (pos + 1) * nLens)
  const top = Array.from(row.keys()).sort((a, b) => row[b] - row[a]).slice(0, 5)
  console.log(`  layer ${String(l).padStart(2)}: ${top.map((i) => `${JSON.stringify(model.lensPieces[i])}(${row[i].toFixed(1)})`).join(' ')}`)
}

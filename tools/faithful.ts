import { readFileSync } from 'node:fs'
import { GPT2 } from '../src/model/gpt2'
import { layerNormAffine, dot, softmax } from '../src/model/math'

const DIR = 'public/model/gpt2'
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

const model = await GPT2.load(DIR, 6144)
const prompt = process.argv[2] ?? 'The Eiffel Tower is located in the city of'
const ids = model.tok.encode(prompt)
const state = model.newState(32)
let real = state.push(ids[0])
for (let i = 1; i < ids.length; i++) real = state.push(ids[i])

const { nEmbd, eps, vocabSize } = model.cfg
const pos = state.length - 1

/** Full-vocabulary logit lens at one layer: ln_f, then the real unembedding. */
function lensLogits(layer: number): Float32Array {
  const h = new Float32Array(nEmbd)
  layerNormAffine(state.hidden[layer], pos * nEmbd, nEmbd, model.lnFg, model.lnFb, eps, h, 0)
  const out = new Float32Array(vocabSize)
  for (let v = 0; v < vocabSize; v++) out[v] = dot(model.wte, v * nEmbd, h, 0, nEmbd)
  return out
}

const realProbs = new Float32Array(real.length)
realProbs.set(real)
softmax(realProbs)
const realTop = Array.from(realProbs.keys()).sort((a, b) => realProbs[b] - realProbs[a])
const realTop10 = new Set(realTop.slice(0, 10))

console.log(`prompt: ${JSON.stringify(prompt)}`)
console.log(`model's actual next token: ${JSON.stringify(model.tok.piece(realTop[0]))} at ${(realProbs[realTop[0]] * 100).toFixed(1)}%\n`)
console.log('layer  top-1 agrees  overlap@10   KL(lens||real)   lens top-3')

for (let l = 0; l < model.cfg.nLayer; l++) {
  const lp = lensLogits(l)
  const probs = new Float32Array(lp.length)
  probs.set(lp)
  softmax(probs)
  const top = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a])

  let kl = 0
  for (let v = 0; v < vocabSize; v++) {
    if (probs[v] > 1e-12 && realProbs[v] > 1e-12) kl += probs[v] * Math.log(probs[v] / realProbs[v])
  }
  const overlap = top.slice(0, 10).filter((t) => realTop10.has(t)).length
  const agree = top[0] === realTop[0] ? 'yes' : 'no '
  const t3 = top.slice(0, 3).map((t) => JSON.stringify(model.tok.piece(t))).join(' ')
  console.log(
    `  ${String(l).padStart(2)}      ${agree}          ${overlap}/10        ${kl.toFixed(3).padStart(7)}      ${t3}`,
  )
}

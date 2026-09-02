import { LlamaLM } from './llama'
import type { LM, LMState, LoadProgress } from './lm'
import { mulberry32, softmax } from './math'
import type { Candidate, FeatureHit, LayerTrace, StepTrace } from './trace'

export const MODEL_URL = '/model/qwen'
export const MODEL_NAME = 'Qwen2.5 0.5B Instruct'
/** How many whole-word tokens are drawn as points. */
export const N_LENS = 6144
export const MAX_CONTEXT = 32

/** How far a layer must push a token, in standard deviations, to be drawn. */
const Z_PUSH = 2.2
/** How many pushes are drawn at once, per position, per layer. */
const TOP_K = 16

export interface Anticipation {
  step: number
  layer: number
  pos: number
  featureId: number
  label: string
  targetStep: number
  targetWord: string
  /** Standardised lens score at the moment it was already leaning this way. */
  z: number
}

export interface Run {
  steps: StepTrace[]
  emitted: number[]
  words: string[]
  anticipations: Anticipation[]
  considered: number
  spoken: number
  promptWords: string[]
  promptText: string
}

export async function loadModel(onProgress?: LoadProgress): Promise<LM> {
  return LlamaLM.load(MODEL_URL, MODEL_NAME, N_LENS, onProgress)
}

function sampleFrom(probs: Float32Array, order: number[], temperature: number, topP: number, rand: () => number): number {
  const scaled = order.map((id) => Math.pow(Math.max(probs[id], 1e-12), 1 / Math.max(temperature, 0.05)))
  const total = scaled.reduce((a, b) => a + b, 0)
  let cum = 0
  const keep: number[] = []
  for (let i = 0; i < order.length; i++) {
    keep.push(i)
    cum += scaled[i] / total
    if (cum >= topP) break
  }
  let r = rand() * keep.reduce((a, i) => a + scaled[i] / total, 0)
  for (const i of keep) {
    r -= scaled[i] / total
    if (r <= 0) return order[i]
  }
  return order[keep[keep.length - 1]]
}

/**
 * What this layer actually did to the answer.
 *
 * The residual stream is a running sum and the output head is linear, so once
 * the final norm's scale is fixed, the difference between the running totals at
 * two consecutive layers is exactly this layer's contribution to every token's
 * logit. Positive means it pushed the model toward saying that word; negative
 * means it pushed it away.
 *
 * This is not an estimate of what the layer was "thinking". It is an exact
 * decomposition of the model's own output, and the parts sum to the whole —
 * verified in tools/evalmodel.ts, which checks the last layer's running total
 * against the real logits.
 *
 * Contributions are standardised per layer because their scale grows steeply
 * with depth; without that the last few layers would be the only visible ones.
 */
function readLayer(model: LM, state: LMState, layer: number, T: number): {
  features: FeatureHit[][]
  suppressed: FeatureHit[][]
  contended: Int32Array
  activations: number
} {
  const nLens = model.lensIds.length
  const features: FeatureHit[][] = []
  const suppressed: FeatureHit[][] = []
  const contended = new Int32Array(T)
  let activations = 0

  const now = state.lens[layer]
  const prev = layer > 0 ? state.lens[layer - 1] : null
  const delta = new Float32Array(nLens)

  for (let pos = 0; pos < T; pos++) {
    const base = pos * nLens
    let mean = 0
    for (let i = 0; i < nLens; i++) {
      delta[i] = now[base + i] - (prev ? prev[base + i] : 0)
      mean += delta[i]
    }
    mean /= nLens
    let sd = 0
    for (let i = 0; i < nLens; i++) sd += (delta[i] - mean) ** 2
    sd = Math.sqrt(sd / nLens) || 1

    const up: FeatureHit[] = []
    const down: FeatureHit[] = []
    for (let i = 0; i < nLens; i++) {
      const z = (delta[i] - mean) / sd
      if (z > Z_PUSH) up.push({ id: i, act: z, delta: delta[i] })
      else if (z < -Z_PUSH) down.push({ id: i, act: -z, delta: delta[i] })
    }
    contended[pos] = up.length + down.length
    up.sort((a, b) => b.act - a.act)
    down.sort((a, b) => b.act - a.act)
    const kept = up.slice(0, TOP_K)
    activations += kept.length + Math.min(down.length, TOP_K)
    features.push(kept)
    suppressed.push(down.slice(0, TOP_K))
  }
  return { features, suppressed, contended, activations }
}

/**
 * How much of the final answer is in place by this depth.
 *
 * Compares the running total at this layer against the model's real output over
 * exactly the tokens that get drawn. Because the totals are an exact
 * decomposition, this reaches a perfect match at the last layer by construction
 * — which is the check that the arithmetic is right, not a claim about meaning.
 */
function measureAgreement(
  model: LM,
  state: LMState,
  layer: number,
  pos: number,
  realOverLens: Float32Array,
  realTop: Set<number>,
): { agreement: number; kl: number } {
  const nLens = model.lensIds.length
  const base = pos * nLens
  const p = new Float32Array(nLens)
  p.set(state.lens[layer].subarray(base, base + nLens))
  softmax(p)

  let kl = 0
  for (let i = 0; i < nLens; i++) {
    if (p[i] > 1e-12 && realOverLens[i] > 1e-12) kl += p[i] * Math.log(p[i] / realOverLens[i])
  }
  const top = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, 10)
  let hit = 0
  for (const i of top) if (realTop.has(i)) hit++
  return { agreement: hit / 10, kl }
}

function buildTrace(model: LM, state: LMState, logits: Float32Array, active: number): StepTrace {
  const { nLayer, nHead } = model.cfg
  const T = state.length
  const layers: LayerTrace[] = []
  let activations = 0

  // The model's real distribution, restricted to the tokens actually drawn.
  const nLens = model.lensIds.length
  const realOverLens = new Float32Array(nLens)
  for (let i = 0; i < nLens; i++) realOverLens[i] = logits[model.lensIds[i]]
  softmax(realOverLens)
  const realTop = new Set(
    Array.from(realOverLens.keys()).sort((a, b) => realOverLens[b] - realOverLens[a]).slice(0, 10),
  )

  for (let l = 0; l < nLayer; l++) {
    const { features, suppressed, contended, activations: n } = readLayer(model, state, l, T)
    activations += n
    // Repack the cached attention into a dense T x T per head for the renderer.
    const attn = new Float32Array(nHead * T * T)
    for (let h = 0; h < nHead; h++) {
      for (let i = 0; i < T; i++) {
        const src = h * state.maxT * state.maxT + i * state.maxT
        attn.set(state.attn[l].subarray(src, src + T), h * T * T + i * T)
      }
    }
    const { agreement, kl } = measureAgreement(model, state, l, active, realOverLens, realTop)
    layers.push({
      layer: l,
      agreement,
      kl,
      attn,
      features,
      suppressed,
      contended,
      residualNorm: new Float32Array(state.residualNorm[l].subarray(0, T)),
      writeNorm: new Float32Array(state.writeNorm[l].subarray(0, T)),
      proj: new Float32Array(state.proj[l].subarray(0, T * 3)),
    })
  }

  const probs = new Float32Array(logits.length)
  probs.set(logits)
  softmax(probs)
  let entropy = 0
  for (let v = 0; v < probs.length; v++) if (probs[v] > 0) entropy -= probs[v] * Math.log2(probs[v])

  const order = Array.from({ length: probs.length }, (_, i) => i).sort((a, b) => probs[b] - probs[a]).slice(0, 40)
  const candidates: Candidate[] = order.map((id) => ({ id, word: model.tok.piece(id), prob: probs[id] }))

  return { ids: state.ids.slice(), active, layers, candidates, chosen: order[0], entropy, activations }
}

export function generate(
  model: LM,
  prompt: string,
  nSteps: number,
  opts: { temperature?: number; topP?: number; seed?: number } = {},
): Run {
  const rand = mulberry32(opts.seed ?? 7)
  const temperature = opts.temperature ?? 0.8
  const topP = opts.topP ?? 0.9

  const text = prompt.trim() || 'The Eiffel Tower is located in the city of'
  let ids = model.tok.encode(text)
  if (ids.length === 0) ids = model.tok.encode('The Eiffel Tower is located in the city of')
  ids = ids.slice(-(MAX_CONTEXT - nSteps - 1))

  const state = model.newState(MAX_CONTEXT)
  let logits = state.push(ids[0])
  for (let i = 1; i < ids.length; i++) logits = state.push(ids[i])

  const steps: StepTrace[] = []
  const emitted: number[] = []
  let considered = 0

  for (let s = 0; s < nSteps; s++) {
    const probs = new Float32Array(logits.length)
    probs.set(logits)
    softmax(probs)
    const order = Array.from({ length: probs.length }, (_, i) => i)
      .sort((a, b) => probs[b] - probs[a])
      .slice(0, 40)
    const chosen = sampleFrom(probs, order, temperature, topP, rand)

    const trace = buildTrace(model, state, logits, state.length - 1)
    trace.chosen = chosen
    steps.push(trace)
    considered += trace.activations
    emitted.push(chosen)

    if (state.length >= MAX_CONTEXT) break
    logits = state.push(chosen)
  }

  return {
    steps,
    emitted,
    words: emitted.map((id) => model.tok.piece(id)),
    anticipations: findAnticipations(model, steps, emitted),
    considered,
    spoken: emitted.length,
    promptWords: ids.map((id) => model.tok.piece(id)),
    promptText: text,
  }
}

/**
 * Anticipation, measured rather than asserted.
 *
 * Because every point in the scene is a real token, this needs no similarity
 * metric and no interpretation: the question is simply whether the token the
 * model wrote at step t was already among the tokens it was leaning toward at
 * step s, several words earlier. The continuation is generated first and this
 * pass looks backward, so it is a retrodiction — it says the lean was there, not
 * that the model intended anything.
 */
function findAnticipations(model: LM, steps: StepTrace[], emitted: number[]): Anticipation[] {
  const MIN_LEAD = 2
  const byId = new Map<number, number>()
  model.lensIds.forEach((tokenId, i) => byId.set(tokenId, i))

  const best = new Map<string, Anticipation>()
  for (let s = 0; s < steps.length; s++) {
    const step = steps[s]
    for (let t = s + MIN_LEAD; t < steps.length; t++) {
      const featureId = byId.get(emitted[t])
      if (featureId === undefined) continue
      for (const layer of step.layers) {
        const hits = layer.features[step.active]
        if (!hits) continue
        const hit = hits.find((f) => f.id === featureId)
        if (!hit) continue
        const key = `${s}:${t}`
        const prev = best.get(key)
        if (!prev || hit.act > prev.z) {
          best.set(key, {
            step: s,
            layer: layer.layer,
            pos: step.active,
            featureId,
            label: model.lensPieces[featureId],
            targetStep: t,
            targetWord: model.tok.piece(emitted[t]),
            z: hit.act,
          })
        }
      }
    }
  }

  const byStep = new Map<number, Anticipation[]>()
  for (const a of best.values()) {
    const list = byStep.get(a.step) ?? []
    list.push(a)
    byStep.set(a.step, list)
  }
  const out: Anticipation[] = []
  for (const list of byStep.values()) {
    list.sort((a, b) => b.z - a.z)
    out.push(...list.slice(0, 3))
  }
  return out.sort((a, b) => a.step - b.step || a.targetStep - b.targetStep)
}

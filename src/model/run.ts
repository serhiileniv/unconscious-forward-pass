import { buildEmbedding, type Embedding } from './embedding'
import { dot, mulberry32 } from './math'
import { buildVocab, encode, type Vocab } from './tokenizer'
import {
  buildModel,
  DEFAULT_CONFIG,
  forward,
  makeProjector,
  type Model,
  type ModelConfig,
  type StepTrace,
} from './transformer'

export interface Anticipation {
  /** The generation step at which this feature was already awake. */
  step: number
  layer: number
  pos: number
  featureId: number
  label: string
  /** The later step whose emitted word this feature's direction points at. */
  targetStep: number
  targetWord: string
  sim: number
}

export interface Run {
  steps: StepTrace[]
  emitted: number[]
  words: string[]
  anticipations: Anticipation[]
  /** Total feature activations across every pass — what was considered. */
  considered: number
  /** Words actually produced — what was said. */
  spoken: number
}

export interface Session {
  cfg: ModelConfig
  vocab: Vocab
  emb: Embedding
  model: Model
  projector: Float32Array
}

export function buildSession(cfg: ModelConfig = DEFAULT_CONFIG, seed = 20260902): Session {
  const vocab = buildVocab()
  const emb = buildEmbedding(vocab, cfg.dModel, seed)
  const model = buildModel(vocab, emb, cfg, seed + 1)
  const projector = makeProjector(cfg.dModel, seed + 2)
  return { cfg, vocab, emb, model, projector }
}

/** Nucleus sampling over the candidate list the forward pass already ranked. */
function sample(probs: number[], temperature: number, topP: number, rand: () => number): number {
  const scaled = probs.map((p) => Math.pow(Math.max(p, 1e-12), 1 / Math.max(temperature, 0.05)))
  const total = scaled.reduce((a, b) => a + b, 0)
  const norm = scaled.map((p) => p / total)
  let cum = 0
  const cutoff: number[] = []
  for (let i = 0; i < norm.length; i++) {
    cutoff.push(i)
    cum += norm[i]
    if (cum >= topP) break
  }
  let r = rand() * cutoff.reduce((a, i) => a + norm[i], 0)
  for (const i of cutoff) {
    r -= norm[i]
    if (r <= 0) return i
  }
  return cutoff[cutoff.length - 1]
}

export function generate(
  session: Session,
  prompt: string,
  nSteps: number,
  opts: { temperature?: number; topP?: number; seed?: number } = {},
): Run {
  const { model, vocab, projector, cfg } = session
  const rand = mulberry32(opts.seed ?? 7)
  const temperature = opts.temperature ?? 0.85
  const topP = opts.topP ?? 0.9

  let ids = encode(vocab, prompt).filter((id) => id !== 0)
  if (ids.length === 0) ids = encode(vocab, 'most of what a mind does is never said')
  ids = ids.slice(-Math.max(1, cfg.maxT - nSteps))

  const steps: StepTrace[] = []
  const emitted: number[] = []
  let considered = 0

  for (let s = 0; s < nSteps; s++) {
    const trace = forward(model, ids, projector)
    const pick = sample(
      trace.candidates.map((c) => c.prob),
      temperature,
      topP,
      rand,
    )
    trace.chosen = trace.candidates[pick].id
    steps.push(trace)
    considered += trace.activations
    emitted.push(trace.chosen)
    ids = [...ids, trace.chosen]
    if (ids.length > cfg.maxT) ids = ids.slice(-cfg.maxT)
  }

  return {
    steps,
    emitted,
    words: emitted.map((id) => vocab.words[id]),
    anticipations: findAnticipations(session, steps, emitted),
    considered,
    spoken: emitted.length,
  }
}

/**
 * Anticipation, measured rather than asserted.
 *
 * Real models plan: a feature can be active long before the word it relates to is
 * written. Nothing here fabricates that. The whole continuation is generated
 * first, and only then does this pass look backward and ask, for each feature that
 * was awake at step s, whether its direction points at a word the model went on to
 * emit at some later step. The links drawn in the scene are that measurement.
 *
 * It is a retrodiction, not a claim about intent, and the UI labels it as one.
 */
const FUNCTION_WORDS = new Set(
  ('a an the and or but of to in on at by for with from as is are was were be been it its this that '
    + 'these those there here not no only own so then than up out over into which what who whom whose '
    + 'have has had do does did can could will would shall should may might must if when while all any '
    + 'each every both few more most other some such one two i you he she we they them their our your'
  ).split(' '),
)

function findAnticipations(session: Session, steps: StepTrace[], emitted: number[]): Anticipation[] {
  const { model, cfg, emb, vocab } = session
  const d = cfg.dModel
  const MIN_SIM = 0.4
  const MIN_LEAD = 3

  // Only content words are worth pointing at. A feature "anticipating" the word
  // "a" is arithmetic, not foresight, and showing it would be a lie by omission.
  const targets = emitted.map((id, t) => {
    if (FUNCTION_WORDS.has(vocab.words[id])) return null
    // Don't count a word the model has already emitted as something it foresaw.
    if (emitted.slice(0, t).includes(id)) return null
    const v = new Float32Array(d)
    let n = 0
    for (let i = 0; i < d; i++) {
      v[i] = emb.matrix[id * d + i]
      n += v[i] * v[i]
    }
    n = Math.sqrt(n) || 1e-8
    for (let i = 0; i < d; i++) v[i] /= n
    return v
  })

  // Best link per (step, future word). Keyed by both, not just the target, because
  // these features re-fire on pass after pass — the thread genuinely persists until
  // the word is finally written, and holding it on screen for only its first step
  // would hide the phenomenon rather than show it.
  const best = new Map<string, Anticipation>()
  for (let s = 0; s < steps.length; s++) {
    const seen = new Set<number>()
    for (const layer of steps[s].layers) {
      for (let pos = 0; pos < layer.features.length; pos++) {
        for (const hit of layer.features[pos]) {
          if (seen.has(hit.id)) continue
          seen.add(hit.id)
          for (let t = s + MIN_LEAD; t < steps.length; t++) {
            const tgt = targets[t]
            if (!tgt) continue
            const sim = dot(model.dict, hit.id * d, tgt, 0, d)
            if (sim < MIN_SIM) continue
            const key = `${s}:${t}`
            const prev = best.get(key)
            if (!prev || sim > prev.sim) {
              best.set(key, {
                step: s,
                layer: layer.layer,
                pos,
                featureId: hit.id,
                label: model.dictLabels[hit.id],
                targetStep: t,
                targetWord: vocab.words[emitted[t]],
                sim,
              })
            }
          }
        }
      }
    }
  }

  // Cap per step so a single loud direction cannot fill the scene.
  const byStep = new Map<number, Anticipation[]>()
  for (const a of best.values()) {
    const list = byStep.get(a.step) ?? []
    list.push(a)
    byStep.set(a.step, list)
  }
  const out: Anticipation[] = []
  for (const list of byStep.values()) {
    list.sort((a, b) => b.sim - a.sim)
    out.push(...list.slice(0, 3))
  }
  return out.sort((a, b) => a.step - b.step || a.targetStep - b.targetStep)
}

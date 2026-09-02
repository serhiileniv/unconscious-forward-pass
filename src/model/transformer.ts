import type { Embedding } from './embedding'
import { dot, gelu, layerNorm, matmul, mulberry32, randomMatrix, softmax } from './math'
import type { Vocab } from './tokenizer'

export interface ModelConfig {
  dModel: number
  nLayers: number
  nHeads: number
  dFF: number
  /** Size of the shared feature dictionary. Overcomplete by design — that is the point. */
  nFeatures: number
  /** How many features are allowed to be awake at once, per position, per layer. */
  topK: number
  maxT: number
}

export const DEFAULT_CONFIG: ModelConfig = {
  dModel: 96,
  nLayers: 12,
  nHeads: 6,
  dFF: 384,
  nFeatures: 2048,
  topK: 16,
  maxT: 24,
}

interface Block {
  wq: Float32Array
  wk: Float32Array
  wv: Float32Array
  wo: Float32Array
  w1: Float32Array
  w2: Float32Array
}

export interface Model {
  cfg: ModelConfig
  vocab: Vocab
  emb: Embedding
  posEmb: Float32Array
  blocks: Block[]
  /** nFeatures × dModel, unit rows. Each row is one direction in the residual space. */
  dict: Float32Array
  dictBias: Float32Array
  /** A human-readable name for every feature direction. */
  dictLabels: string[]
}

export function buildModel(vocab: Vocab, emb: Embedding, cfg: ModelConfig, seed: number): Model {
  const rand = mulberry32(seed)
  const d = cfg.dModel
  const s = 1 / Math.sqrt(d)

  const blocks: Block[] = []
  for (let l = 0; l < cfg.nLayers; l++) {
    blocks.push({
      wq: randomMatrix(d, d, s, rand),
      wk: randomMatrix(d, d, s, rand),
      wv: randomMatrix(d, d, s, rand),
      wo: randomMatrix(d, d, s, rand),
      w1: randomMatrix(d, cfg.dFF, s, rand),
      w2: randomMatrix(cfg.dFF, d, 1 / Math.sqrt(cfg.dFF), rand),
    })
  }

  // Sinusoidal position encoding — deterministic, and it gives the residual
  // filaments a visible spatial signature at layer zero.
  const posEmb = new Float32Array(cfg.maxT * d)
  for (let p = 0; p < cfg.maxT; p++) {
    for (let i = 0; i < d; i += 2) {
      const w = 1 / Math.pow(10000, i / d)
      posEmb[p * d + i] = Math.sin(p * w) * 0.5
      if (i + 1 < d) posEmb[p * d + i + 1] = Math.cos(p * w) * 0.5
    }
  }

  const { dict, dictLabels } = buildDictionary(vocab, emb, cfg, rand)
  const dictBias = calibrateBias(dict, emb, cfg, vocab)

  return { cfg, vocab, emb, posEmb, blocks, dict, dictBias, dictLabels }
}

/**
 * The feature dictionary.
 *
 * A trained sparse autoencoder learns its dictionary; this one is constructed,
 * which is the honest limitation of a toy. But it is constructed the way real
 * dictionaries turn out to look: the first V rows are the vocabulary directions
 * themselves, and the remainder are compositions of two or three of them. That
 * gives every feature in the scene a name you can read, and it reproduces the
 * property that actually matters here — far more directions than dimensions, so
 * they cannot all be orthogonal, and they interfere.
 */
function buildDictionary(vocab: Vocab, emb: Embedding, cfg: ModelConfig, rand: () => number) {
  const d = cfg.dModel
  const dict = new Float32Array(cfg.nFeatures * d)
  const dictLabels: string[] = new Array(cfg.nFeatures)

  const setRow = (row: number, vec: Float32Array) => {
    let n = 0
    for (let i = 0; i < d; i++) n += vec[i] * vec[i]
    n = Math.sqrt(n) || 1e-8
    for (let i = 0; i < d; i++) dict[row * d + i] = vec[i] / n
  }

  const tmp = new Float32Array(d)
  const real = Math.min(vocab.size - 1, cfg.nFeatures)
  for (let f = 0; f < real; f++) {
    const w = f + 1 // skip <unk>
    for (let i = 0; i < d; i++) tmp[i] = emb.matrix[w * d + i]
    setRow(f, tmp)
    dictLabels[f] = vocab.words[w]
  }
  for (let f = real; f < cfg.nFeatures; f++) {
    const arity = rand() < 0.6 ? 2 : 3
    tmp.fill(0)
    const parts: string[] = []
    for (let a = 0; a < arity; a++) {
      const w = 1 + Math.floor(rand() * (vocab.size - 1))
      const sign = rand() < 0.25 ? -1 : 1
      for (let i = 0; i < d; i++) tmp[i] += sign * emb.matrix[w * d + i]
      parts.push((sign < 0 ? '¬' : '') + vocab.words[w])
    }
    setRow(f, tmp)
    dictLabels[f] = parts.join('·')
  }
  return { dict, dictLabels }
}

/**
 * Per-feature threshold. Without one, every feature is faintly on for every input
 * and there is no sparsity to look at. Setting it from the distribution of
 * activations across the whole vocabulary is the cheap stand-in for a learned bias.
 */
function calibrateBias(dict: Float32Array, emb: Embedding, cfg: ModelConfig, vocab: Vocab): Float32Array {
  const d = cfg.dModel
  const bias = new Float32Array(cfg.nFeatures)
  const mean = new Float32Array(cfg.nFeatures)
  const sq = new Float32Array(cfg.nFeatures)
  const n = vocab.size - 1
  const row = new Float32Array(d)
  for (let w = 1; w < vocab.size; w++) {
    row.set(emb.matrix.subarray(w * d, w * d + d))
    layerNorm(row, 0, d)
    for (let f = 0; f < cfg.nFeatures; f++) {
      const a = dot(dict, f * d, row, 0, d)
      mean[f] += a / n
      sq[f] += (a * a) / n
    }
  }
  for (let f = 0; f < cfg.nFeatures; f++) {
    const sd = Math.sqrt(Math.max(0, sq[f] - mean[f] * mean[f]))
    bias[f] = mean[f] + 1.6 * sd
  }
  return bias
}

// ---------------------------------------------------------------------------
// The trace: everything the forward pass did, kept so the scene can replay it.
// ---------------------------------------------------------------------------

export interface FeatureHit {
  id: number
  /** Post-threshold activation. */
  act: number
  /** Change since the previous layer. Strongly negative means suppressed. */
  delta: number
}

export interface LayerTrace {
  layer: number
  /** nHeads × T × T, causally masked, rows sum to 1. */
  attn: Float32Array
  /** Awake features per position, already sorted by activation. */
  features: FeatureHit[][]
  /** Features that were awake last layer and were pushed back down in this one. */
  suppressed: FeatureHit[][]
  /**
   * How many features cleared their threshold per position, before top-k truncation.
   * The count of *awake* features is fixed at k by construction, so this is the
   * number that actually says something: how much competed for the k slots.
   */
  contended: Int32Array
  /** Norm of the residual stream per position. */
  residualNorm: Float32Array
  /** Norm of what this layer added to the stream, per position — how loud it was. */
  writeNorm: Float32Array
  /** T × 3 projection of the residual, for drawing. */
  proj: Float32Array
}

export interface Candidate {
  id: number
  word: string
  prob: number
}

export interface StepTrace {
  /** Input ids for this forward pass. */
  ids: number[]
  layers: LayerTrace[]
  candidates: Candidate[]
  chosen: number
  entropy: number
  /** Total feature activations across the whole pass — the "considered" count. */
  activations: number
}

/** Fixed random projection to 3D, shared by every trace so the space is stable. */
export function makeProjector(dModel: number, seed: number): Float32Array {
  const rand = mulberry32(seed)
  const p = randomMatrix(dModel, 3, 1 / Math.sqrt(dModel), rand)
  return p
}

export function forward(model: Model, ids: number[], projector: Float32Array): StepTrace {
  const { cfg, emb } = model
  const d = cfg.dModel
  const T = ids.length
  const dHead = d / cfg.nHeads

  // Residual stream: T × d.
  const x = new Float32Array(T * d)
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < d; i++) x[t * d + i] = emb.matrix[ids[t] * d + i] + model.posEmb[t * d + i]
  }

  const layers: LayerTrace[] = []
  let prevActs = new Float32Array(T * cfg.nFeatures)
  let activations = 0

  const h = new Float32Array(T * d)
  const q = new Float32Array(T * d)
  const k = new Float32Array(T * d)
  const v = new Float32Array(T * d)
  const ctx = new Float32Array(T * d)
  const ff1 = new Float32Array(T * cfg.dFF)
  const ff2 = new Float32Array(T * d)
  const before = new Float32Array(T * d)

  for (let l = 0; l < cfg.nLayers; l++) {
    const b = model.blocks[l]
    before.set(x)

    // --- attention ---
    h.set(x)
    for (let t = 0; t < T; t++) layerNorm(h, t * d, d)
    matmul(h, b.wq, T, d, d, q)
    matmul(h, b.wk, T, d, d, k)
    matmul(h, b.wv, T, d, d, v)

    const attn = new Float32Array(cfg.nHeads * T * T)
    ctx.fill(0)
    const scale = 1 / Math.sqrt(dHead)
    for (let hd = 0; hd < cfg.nHeads; hd++) {
      const off = hd * dHead
      for (let i = 0; i < T; i++) {
        const rowOff = hd * T * T + i * T
        for (let j = 0; j <= i; j++) {
          attn[rowOff + j] = dot(q, i * d + off, k, j * d + off, dHead) * scale
        }
        for (let j = i + 1; j < T; j++) attn[rowOff + j] = -Infinity
        softmax(attn, rowOff, T)
        for (let j = 0; j <= i; j++) {
          const w = attn[rowOff + j]
          if (w < 1e-6) continue
          for (let c = 0; c < dHead; c++) ctx[i * d + off + c] += w * v[j * d + off + c]
        }
      }
    }
    matmul(ctx, b.wo, T, d, d, h)
    for (let i = 0; i < T * d; i++) x[i] += h[i]

    // --- feed forward ---
    h.set(x)
    for (let t = 0; t < T; t++) layerNorm(h, t * d, d)
    matmul(h, b.w1, T, d, cfg.dFF, ff1)
    for (let i = 0; i < ff1.length; i++) ff1[i] = gelu(ff1[i])
    matmul(ff1, b.w2, T, cfg.dFF, d, ff2)
    for (let i = 0; i < T * d; i++) x[i] += ff2[i]

    // --- read the sparse features out of the stream ---
    const acts = new Float32Array(T * cfg.nFeatures)
    const features: FeatureHit[][] = []
    const suppressed: FeatureHit[][] = []
    const contended = new Int32Array(T)
    const probe = new Float32Array(d)
    for (let t = 0; t < T; t++) {
      probe.set(x.subarray(t * d, t * d + d))
      layerNorm(probe, 0, d)
      const hits: FeatureHit[] = []
      for (let f = 0; f < cfg.nFeatures; f++) {
        const a = dot(model.dict, f * d, probe, 0, d) - model.dictBias[f]
        if (a > 0) {
          acts[t * cfg.nFeatures + f] = a
          hits.push({ id: f, act: a, delta: a - prevActs[t * cfg.nFeatures + f] })
        }
      }
      hits.sort((p, r) => r.act - p.act)
      contended[t] = hits.length
      const kept = hits.slice(0, cfg.topK)
      // Anything outside top-k is forced off; that is what makes this sparse.
      for (let i = cfg.topK; i < hits.length; i++) acts[t * cfg.nFeatures + hits[i].id] = 0
      features.push(kept)
      activations += kept.length

      const down: FeatureHit[] = []
      for (let f = 0; f < cfg.nFeatures; f++) {
        const was = prevActs[t * cfg.nFeatures + f]
        const now = acts[t * cfg.nFeatures + f]
        if (was > 0 && now < was * 0.5) down.push({ id: f, act: was, delta: now - was })
      }
      down.sort((p, r) => p.delta - r.delta)
      suppressed.push(down.slice(0, cfg.topK))
    }
    prevActs = acts

    const residualNorm = new Float32Array(T)
    const writeNorm = new Float32Array(T)
    const proj = new Float32Array(T * 3)
    for (let t = 0; t < T; t++) {
      let rn = 0
      let wn = 0
      for (let i = 0; i < d; i++) {
        rn += x[t * d + i] ** 2
        wn += (x[t * d + i] - before[t * d + i]) ** 2
      }
      residualNorm[t] = Math.sqrt(rn)
      writeNorm[t] = Math.sqrt(wn)
      for (let c = 0; c < 3; c++) {
        let s = 0
        for (let i = 0; i < d; i++) s += x[t * d + i] * projector[i * 3 + c]
        proj[t * 3 + c] = s
      }
    }

    layers.push({ layer: l, attn, features, suppressed, contended, residualNorm, writeNorm, proj })
  }

  // --- readout ---
  const last = new Float32Array(d)
  last.set(x.subarray((T - 1) * d, T * d))
  layerNorm(last, 0, d)
  const V = model.vocab.size
  const logits = new Float32Array(V)
  const invSqrtD = 1 / Math.sqrt(d)
  for (let w = 0; w < V; w++) logits[w] = dot(emb.matrix, w * d, last, 0, d) * invSqrtD

  // An untrained transformer has no opinion about English, so the distribution is
  // anchored by the corpus bigram statistics. The mechanism above is genuine; the
  // word choice below is where the toy shows through, and the UI says so.
  const prior = emb.bigram.get(ids[T - 1])
  let priorTotal = 0
  if (prior) for (const c of prior.values()) priorTotal += c
  for (let w = 0; w < V; w++) {
    const p = prior ? (prior.get(w) ?? 0) / (priorTotal || 1) : 0
    logits[w] += 1.7 * Math.log(p + 6e-4)
  }
  logits[0] = -Infinity // never emit <unk>

  const probs = new Float32Array(logits)
  softmax(probs)

  let entropy = 0
  for (let w = 0; w < V; w++) if (probs[w] > 0) entropy -= probs[w] * Math.log2(probs[w])

  const order = Array.from({ length: V }, (_, i) => i).sort((a, c) => probs[c] - probs[a])
  const candidates: Candidate[] = order.slice(0, 40).map((id) => ({
    id,
    word: model.vocab.words[id],
    prob: probs[id],
  }))

  return { ids: ids.slice(), layers, candidates, chosen: candidates[0].id, entropy, activations }
}

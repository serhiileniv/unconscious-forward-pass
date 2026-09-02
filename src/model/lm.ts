import type { BPETokenizer } from './bpe'

/**
 * What the renderer needs from a language model, regardless of architecture.
 *
 * GPT-2 and the Llama-family models (Qwen2, SmolLM2) differ in almost every
 * detail — learned versus rotary positions, LayerNorm versus RMSNorm, GELU
 * versus SwiGLU, full versus grouped-query attention — but they expose the same
 * things to a visualisation: a stack of layers, a residual stream, attention,
 * and an output head to read intermediate states through.
 */
export interface LMConfig {
  nLayer: number
  nEmbd: number
  nHead: number
  vocabSize: number
  /** Short label for the interface, e.g. "Qwen2.5 0.5B Instruct". */
  name: string
}

export interface LM {
  readonly cfg: LMConfig
  readonly tok: BPETokenizer
  /** Display vocabulary: the tokens drawn as points. */
  readonly lensIds: number[]
  readonly lensPieces: string[]
  /** nLens × nEmbd — real rows of the output head. */
  readonly lensMatrix: Float32Array
  /** nLens × 3 — those rows under one fixed projection, for drawing. */
  readonly lensPos: Float32Array
  /** nEmbd × 3 — the projection the residual stream is drawn through. */
  readonly projector: Float32Array
  /**
   * What the 3D projection actually preserves: share of total variance, and the
   * correlation between real high-dimensional distances and drawn ones. Three of
   * several hundred dimensions cannot keep much, and the interface says so
   * rather than letting the cloud imply a fidelity it does not have.
   */
  readonly fidelity: { variance: number; distance: number }
  newState(maxT: number): LMState
}

export interface LMState {
  readonly maxT: number
  readonly ids: number[]
  readonly length: number
  /** Per layer: nHead × maxT × maxT attention, filled row by row. */
  readonly attn: Float32Array[]
  /** Per layer: maxT × nLens raw output-head scores for the drawn tokens. */
  readonly lens: Float32Array[]
  readonly lensMean: Float32Array[]
  readonly lensSd: Float32Array[]
  readonly writeNorm: Float32Array[]
  readonly residualNorm: Float32Array[]
  readonly proj: Float32Array[]
  /** Run one token through every layer; returns the next-token logits. */
  push(tokenId: number): Float32Array
}

export type LoadProgress = (phase: string, frac: number) => void

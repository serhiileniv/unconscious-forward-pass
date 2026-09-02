/**
 * What one forward pass did, kept so the scene can replay it.
 *
 * These types are the contract between the model and the renderer. Nothing in
 * here is derived or decorative — every field is read straight out of the real
 * activations.
 */

export interface FeatureHit {
  /** Index into the display vocabulary, not a GPT-2 token id. */
  id: number
  /** Standardised logit-lens score: how far above this layer's mean it sits. */
  act: number
  /** Change since the previous layer. Strongly negative means suppressed. */
  delta: number
}

export interface LayerTrace {
  layer: number
  /** nHeads × T × T, causally masked, rows sum to 1. */
  attn: Float32Array
  /** Tokens this layer is leaning toward, per position, sorted by score. */
  features: FeatureHit[][]
  /** Tokens that were leading last layer and got pushed down in this one. */
  suppressed: FeatureHit[][]
  /** How many display tokens cleared the threshold, before top-k truncation. */
  contended: Int32Array
  /** Norm of the residual stream per position. */
  residualNorm: Float32Array
  /** Norm of what this layer added to the stream, per position. */
  writeNorm: Float32Array
  /** T × 3 projection of the residual, for drawing. */
  proj: Float32Array
  /**
   * How much this layer's lens agrees with what the model actually predicts:
   * the share of its top ten tokens that appear in the model's top ten, at the
   * position this step computed. 1.0 at the last layer by construction. Low
   * values mean the points drawn at this depth are the lens's opinion more than
   * the model's, and the interface says so.
   */
  agreement: number
  /** KL(lens ‖ model output) over the drawn tokens. Zero at the last layer. */
  kl: number
}

export interface Candidate {
  id: number
  word: string
  prob: number
}

export interface StepTrace {
  ids: number[]
  /** The position whose pass this step actually computed. Earlier ones are settled. */
  active: number
  layers: LayerTrace[]
  candidates: Candidate[]
  chosen: number
  entropy: number
  /** Display-token activations across this pass — the "considered" count. */
  activations: number
}

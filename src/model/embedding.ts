import { CORPUS } from './corpus'
import { matmul, mulberry32, orthonormalize, randomMatrix } from './math'
import { splitWords, type Vocab } from './tokenizer'

/**
 * The embedding matrix is derived from real word statistics, not random noise.
 *
 *   1. count co-occurrences within a sliding window over the corpus
 *   2. reweight them as positive pointwise mutual information (PPMI)
 *   3. project down to `dim` dimensions by subspace iteration on the PPMI matrix,
 *      which finds its dominant eigenvectors
 *
 * This is the classic count-based route to word vectors, and it matters here for
 * one reason: it makes the picture honest. When two points sit near each other in
 * the 3D scene, it is because those two words genuinely keep company in the text.
 * A random embedding would look identical and mean nothing.
 */
export interface Embedding {
  /** vocab.size × dim, row-major. */
  readonly matrix: Float32Array
  readonly dim: number
  /** Row-normalised bigram counts, used as the output prior. */
  readonly bigram: Map<number, Map<number, number>>
}

const WINDOW = 4

export function buildEmbedding(vocab: Vocab, dim: number, seed: number): Embedding {
  const V = vocab.size
  const tokens = splitWords(CORPUS).map((w) => vocab.index.get(w) ?? 0)

  // --- 1. co-occurrence, distance-weighted so near neighbours count for more ---
  const co = new Float32Array(V * V)
  const unigram = new Float32Array(V)
  const bigram = new Map<number, Map<number, number>>()
  for (let i = 0; i < tokens.length; i++) {
    const a = tokens[i]
    unigram[a] += 1
    for (let d = 1; d <= WINDOW && i + d < tokens.length; d++) {
      const b = tokens[i + d]
      const w = 1 / d
      co[a * V + b] += w
      co[b * V + a] += w
    }
    if (i + 1 < tokens.length) {
      let row = bigram.get(a)
      if (!row) bigram.set(a, (row = new Map()))
      row.set(tokens[i + 1], (row.get(tokens[i + 1]) ?? 0) + 1)
    }
  }

  // --- 2. PPMI ---
  let total = 0
  for (let i = 0; i < co.length; i++) total += co[i]
  const rowSum = new Float32Array(V)
  for (let i = 0; i < V; i++) {
    let s = 0
    for (let j = 0; j < V; j++) s += co[i * V + j]
    rowSum[i] = s
  }
  const ppmi = new Float32Array(V * V)
  for (let i = 0; i < V; i++) {
    for (let j = 0; j < V; j++) {
      const c = co[i * V + j]
      if (c === 0) continue
      const pmi = Math.log((c * total) / (rowSum[i] * rowSum[j] || 1))
      ppmi[i * V + j] = pmi > 0 ? pmi : 0
    }
  }

  // --- 3. subspace iteration for the top `dim` eigenvectors of a symmetric matrix ---
  const rand = mulberry32(seed)
  let z = randomMatrix(V, dim, 1, rand)
  orthonormalize(z, V, dim)
  const scratch = new Float32Array(V * dim)
  for (let iter = 0; iter < 12; iter++) {
    matmul(ppmi, z, V, V, dim, scratch)
    z.set(scratch)
    orthonormalize(z, V, dim)
  }

  // Rayleigh quotients give the eigenvalues; scaling by sqrt(λ) is the standard
  // SVD-embedding weighting and keeps frequent words from dominating the space.
  matmul(ppmi, z, V, V, dim, scratch)
  const scale = new Float32Array(dim)
  for (let j = 0; j < dim; j++) {
    let s = 0
    for (let i = 0; i < V; i++) s += z[i * dim + j] * scratch[i * dim + j]
    scale[j] = Math.sqrt(Math.abs(s))
  }
  const matrix = new Float32Array(V * dim)
  for (let i = 0; i < V; i++) {
    for (let j = 0; j < dim; j++) matrix[i * dim + j] = z[i * dim + j] * scale[j]
  }

  // Normalise the whole matrix to unit RMS so downstream layer scales are sane.
  let rms = 0
  for (let i = 0; i < matrix.length; i++) rms += matrix[i] * matrix[i]
  rms = Math.sqrt(rms / matrix.length) || 1
  for (let i = 0; i < matrix.length; i++) matrix[i] /= rms

  return { matrix, dim, bigram }
}

/** Cosine similarity between two vocabulary rows — used to label the scene. */
export function similarity(emb: Embedding, a: number, b: number): number {
  const d = emb.dim
  let ab = 0
  let aa = 0
  let bb = 0
  for (let i = 0; i < d; i++) {
    const x = emb.matrix[a * d + i]
    const y = emb.matrix[b * d + i]
    ab += x * y
    aa += x * x
    bb += y * y
  }
  return ab / (Math.sqrt(aa * bb) || 1e-8)
}

/** The `n` nearest words to `id`. Cheap enough to run live; V is a few hundred. */
export function nearest(emb: Embedding, vocab: Vocab, id: number, n: number): { word: string; sim: number }[] {
  const out: { word: string; sim: number }[] = []
  for (let i = 1; i < vocab.size; i++) {
    if (i === id) continue
    out.push({ word: vocab.words[i], sim: similarity(emb, id, i) })
  }
  out.sort((a, b) => b.sim - a.sim)
  return out.slice(0, n)
}

/** Small dense-linear-algebra helpers. Everything is Float32Array, row-major. */

/** Deterministic PRNG so a given prompt always produces the identical scene. */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

/** Box–Muller, so weight init is actually Gaussian rather than uniform. */
export function gaussian(rand: () => number): number {
  let u = 0
  while (u === 0) u = rand()
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand())
}

export function randomMatrix(rows: number, cols: number, scale: number, rand: () => number): Float32Array {
  const m = new Float32Array(rows * cols)
  for (let i = 0; i < m.length; i++) m[i] = gaussian(rand) * scale
  return m
}

/** out[m×p] = a[m×n] · b[n×p] */
export function matmul(a: Float32Array, b: Float32Array, m: number, n: number, p: number, out?: Float32Array): Float32Array {
  const r = out ?? new Float32Array(m * p)
  r.fill(0)
  for (let i = 0; i < m; i++) {
    const aOff = i * n
    const rOff = i * p
    for (let k = 0; k < n; k++) {
      const av = a[aOff + k]
      if (av === 0) continue
      const bOff = k * p
      for (let j = 0; j < p; j++) r[rOff + j] += av * b[bOff + j]
    }
  }
  return r
}

/** Softmax in place over a slice, numerically stabilised by the max. */
export function softmax(v: Float32Array, off = 0, len = v.length - off): void {
  let max = -Infinity
  for (let i = 0; i < len; i++) if (v[off + i] > max) max = v[off + i]
  let sum = 0
  for (let i = 0; i < len; i++) {
    const e = Math.exp(v[off + i] - max)
    v[off + i] = e
    sum += e
  }
  const inv = 1 / (sum || 1)
  for (let i = 0; i < len; i++) v[off + i] *= inv
}

/** RMS-style layer norm over one row, matching what modern transformers use. */
export function layerNorm(v: Float32Array, off: number, d: number, eps = 1e-5): void {
  let mean = 0
  for (let i = 0; i < d; i++) mean += v[off + i]
  mean /= d
  let varr = 0
  for (let i = 0; i < d; i++) {
    const x = v[off + i] - mean
    varr += x * x
  }
  varr /= d
  const inv = 1 / Math.sqrt(varr + eps)
  for (let i = 0; i < d; i++) v[off + i] = (v[off + i] - mean) * inv
}

/** Tanh approximation of GELU — the one the original GPT-2 code shipped with. */
export function gelu(x: number): number {
  return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
}

export function dot(a: Float32Array, aOff: number, b: Float32Array, bOff: number, n: number): number {
  let s = 0
  for (let i = 0; i < n; i++) s += a[aOff + i] * b[bOff + i]
  return s
}

export function norm(a: Float32Array, off: number, n: number): number {
  return Math.sqrt(dot(a, off, a, off, n)) || 1e-8
}

/**
 * Modified Gram–Schmidt, orthonormalising the `cols` columns of an
 * `rows × cols` matrix in place. Used by the subspace iteration in embedding.ts.
 */
export function orthonormalize(z: Float32Array, rows: number, cols: number): void {
  for (let j = 0; j < cols; j++) {
    for (let k = 0; k < j; k++) {
      let d = 0
      for (let i = 0; i < rows; i++) d += z[i * cols + j] * z[i * cols + k]
      for (let i = 0; i < rows; i++) z[i * cols + j] -= d * z[i * cols + k]
    }
    let n = 0
    for (let i = 0; i < rows; i++) n += z[i * cols + j] ** 2
    n = Math.sqrt(n) || 1e-8
    for (let i = 0; i < rows; i++) z[i * cols + j] /= n
  }
}

/** LayerNorm with learned scale and shift — what GPT-2 actually uses. */
export function layerNormAffine(
  v: Float32Array,
  off: number,
  d: number,
  gamma: Float32Array,
  beta: Float32Array,
  eps: number,
  out: Float32Array,
  outOff: number,
): void {
  let mean = 0
  for (let i = 0; i < d; i++) mean += v[off + i]
  mean /= d
  let varr = 0
  for (let i = 0; i < d; i++) {
    const x = v[off + i] - mean
    varr += x * x
  }
  varr /= d
  const inv = 1 / Math.sqrt(varr + eps)
  for (let i = 0; i < d; i++) out[outOff + i] = (v[off + i] - mean) * inv * gamma[i] + beta[i]
}

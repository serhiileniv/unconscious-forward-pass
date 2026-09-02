import { readFileSync } from 'node:fs'
import { LlamaLM } from '../src/model/llama'
import { mulberry32, orthonormalize } from '../src/model/math'

const DIR = 'public/model/qwen'
;(globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
  const buf = readFileSync(`${DIR}/${String(url).split('/').pop()!}`)
  return { ok: true, headers: { get: () => String(buf.byteLength) }, body: null,
    json: async () => JSON.parse(buf.toString()), text: async () => buf.toString(),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) }
}

const m = await LlamaLM.load(DIR, 'x', 6144)
const N = m.lensIds.length
const D = m.shape.nEmbd
const X = m.lensMatrix

// Centre the rows once; both projections are judged on the same data.
const mean = new Float32Array(D)
for (let i = 0; i < N; i++) for (let k = 0; k < D; k++) mean[k] += X[i * D + k] / N
const C = new Float32Array(N * D)
for (let i = 0; i < N; i++) for (let k = 0; k < D; k++) C[i * D + k] = X[i * D + k] - mean[k]

let totalVar = 0
for (let i = 0; i < C.length; i++) totalVar += C[i] * C[i]

/** Top-3 principal directions by subspace iteration on C^T C, without forming it. */
function pca3(): Float32Array {
  const rand = mulberry32(1)
  const Z = new Float32Array(D * 3)
  for (let i = 0; i < Z.length; i++) Z[i] = rand() * 2 - 1
  orthonormalize(Z, D, 3)
  const T = new Float32Array(N * 3)
  for (let it = 0; it < 30; it++) {
    T.fill(0)
    for (let i = 0; i < N; i++)
      for (let k = 0; k < D; k++) {
        const v = C[i * D + k]
        if (v === 0) continue
        for (let c = 0; c < 3; c++) T[i * 3 + c] += v * Z[k * 3 + c]
      }
    Z.fill(0)
    for (let i = 0; i < N; i++)
      for (let k = 0; k < D; k++) {
        const v = C[i * D + k]
        if (v === 0) continue
        for (let c = 0; c < 3; c++) Z[k * 3 + c] += v * T[i * 3 + c]
      }
    orthonormalize(Z, D, 3)
  }
  return Z
}

function project(P: Float32Array): Float32Array {
  const out = new Float32Array(N * 3)
  for (let i = 0; i < N; i++)
    for (let c = 0; c < 3; c++) {
      let s = 0
      for (let k = 0; k < D; k++) s += C[i * D + k] * P[k * 3 + c]
      out[i * 3 + c] = s
    }
  return out
}

function variance(Y: Float32Array): number {
  let v = 0
  for (let i = 0; i < Y.length; i++) v += Y[i] * Y[i]
  return v / totalVar
}

/** Do 3-D distances track the real 896-D ones? Pearson r over random pairs. */
function distanceFidelity(Y: Float32Array, warp: boolean): number {
  const rand = mulberry32(99)
  const a: number[] = []
  const b: number[] = []
  const P = new Float32Array(N * 3)
  P.set(Y)
  if (warp) {
    for (let i = 0; i < N; i++) {
      const r = Math.hypot(P[i * 3], P[i * 3 + 1]) || 1e-9
      const rr = Math.pow(r, 0.62)
      P[i * 3] = (P[i * 3] / r) * rr
      P[i * 3 + 1] = (P[i * 3 + 1] / r) * rr
    }
  }
  for (let n = 0; n < 4000; n++) {
    const i = Math.floor(rand() * N)
    const j = Math.floor(rand() * N)
    if (i === j) continue
    let d2 = 0
    for (let k = 0; k < D; k++) d2 += (C[i * D + k] - C[j * D + k]) ** 2
    a.push(Math.sqrt(d2))
    b.push(Math.hypot(P[i * 3] - P[j * 3], P[i * 3 + 1] - P[j * 3 + 1], P[i * 3 + 2] - P[j * 3 + 2]))
  }
  const ma = a.reduce((x, y) => x + y, 0) / a.length
  const mb = b.reduce((x, y) => x + y, 0) / b.length
  let num = 0, da = 0, db = 0
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - ma) * (b[i] - mb); da += (a[i] - ma) ** 2; db += (b[i] - mb) ** 2
  }
  return num / Math.sqrt(da * db)
}

const rand = mulberry32(20260903)
const R = new Float32Array(D * 3)
for (let i = 0; i < R.length; i++) {
  let u = 0
  while (u === 0) u = rand()
  R[i] = (Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand())) / Math.sqrt(D)
}

const Yr = project(R)
const Yp = project(pca3())
console.log(`${N} tokens in ${D} dimensions, projected to 3\n`)
console.log(`                            variance kept   distance fidelity (Pearson r)`)
console.log(`random projection (shipped)     ${(variance(Yr) * 100).toFixed(1)}%              ${distanceFidelity(Yr, false).toFixed(3)}`)
console.log(`  + the r^0.62 radial warp      ${(variance(Yr) * 100).toFixed(1)}%              ${distanceFidelity(Yr, true).toFixed(3)}`)
console.log(`top-3 PCA                       ${(variance(Yp) * 100).toFixed(1)}%              ${distanceFidelity(Yp, false).toFixed(3)}`)
console.log(`  + the r^0.62 radial warp      ${(variance(Yp) * 100).toFixed(1)}%              ${distanceFidelity(Yp, true).toFixed(3)}`)

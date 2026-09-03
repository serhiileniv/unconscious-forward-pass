/**
 * A precomputed semantic layout for the token cloud.
 *
 * Produced by tools/layout.ts: an approximate k-nearest-neighbour graph over the
 * model's own embedding space, laid out in 3D so that neighbours land near each
 * other. It answers the question a linear projection cannot — which tokens does
 * this model treat as related — and it is what turns the cloud from a Gaussian
 * blob into a map with regions.
 *
 * Optional. Without the file the scene falls back to PCA, which is honest but
 * carries far less.
 */
export interface Layout {
  n: number
  k: number
  /** n × 3, centred, unit RMS radius. */
  pos: Float32Array
  /** n × k neighbour ids. */
  nbr: Int32Array
  /** n × k cosine similarities, matching `nbr`. */
  sim: Float32Array
  /** Measured share of each token's true nearest neighbours that stay nearest in 3D. */
  preservation: number
}

export async function loadLayout(baseUrl: string): Promise<Layout | null> {
  let buf: ArrayBuffer
  try {
    const res = await fetch(`${baseUrl}/layout.bin`)
    if (!res.ok) return null
    buf = await res.arrayBuffer()
  } catch {
    return null
  }

  const head = new Int32Array(buf, 0, 2)
  const n = head[0]
  const k = head[1]
  let off = 8
  const pos = new Float32Array(buf.slice(off, off + n * 3 * 4))
  off += n * 3 * 4
  const nbr = new Int32Array(buf.slice(off, off + n * k * 4))
  off += n * k * 4
  const sim = new Float32Array(buf.slice(off, off + n * k * 4))

  let preservation = 0
  try {
    const meta = await fetch(`${baseUrl}/layout.json`).then((r) => (r.ok ? r.json() : null))
    preservation = meta?.preservation ?? 0
  } catch {
    preservation = 0
  }
  return { n, k, pos, nbr, sim, preservation }
}

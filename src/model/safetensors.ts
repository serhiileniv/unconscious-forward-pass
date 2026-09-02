/**
 * Minimal safetensors reader.
 *
 * The format is deliberately simple: 8 bytes of little-endian length, that many
 * bytes of JSON describing every tensor's dtype, shape and byte range, then one
 * contiguous data block. No framework needed to read it, which is the point —
 * the weights below are the real published ones, loaded directly.
 */
export interface TensorInfo {
  dtype: string
  shape: number[]
  data_offsets: [number, number]
}

export class Safetensors {
  private readonly header: Record<string, TensorInfo>
  private readonly buffer: ArrayBuffer
  private readonly base: number
  private readonly cache = new Map<string, Float32Array>()

  constructor(buffer: ArrayBuffer) {
    this.buffer = buffer
    const view = new DataView(buffer)
    const headerLen = Number(view.getBigUint64(0, true))
    const json = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen))
    const parsed = JSON.parse(json) as Record<string, TensorInfo>
    delete (parsed as Record<string, unknown>).__metadata__
    this.header = parsed
    this.base = 8 + headerLen
  }

  names(): string[] {
    return Object.keys(this.header)
  }

  has(name: string): boolean {
    return name in this.header
  }

  shape(name: string): number[] {
    const info = this.header[name]
    if (!info) throw new Error(`no tensor "${name}"`)
    return info.shape
  }

  /** Returns the tensor as f32, converting from f16/bf16 when needed. Cached. */
  get(name: string): Float32Array {
    const hit = this.cache.get(name)
    if (hit) return hit
    const info = this.header[name]
    if (!info) throw new Error(`no tensor "${name}"`)
    const [start, end] = info.data_offsets
    const byteStart = this.base + start
    const count = info.shape.reduce((a, b) => a * b, 1)
    let out: Float32Array

    switch (info.dtype) {
      case 'F32':
        // The data block is only 8-byte aligned as a whole, so a Float32Array
        // view onto it can throw; copy when the offset is not 4-byte aligned.
        out =
          byteStart % 4 === 0
            ? new Float32Array(this.buffer, byteStart, count)
            : new Float32Array(this.buffer.slice(byteStart, this.base + end))
        break
      case 'F16': {
        const src = new Uint16Array(this.buffer.slice(byteStart, this.base + end))
        out = new Float32Array(count)
        for (let i = 0; i < count; i++) out[i] = halfToFloat(src[i])
        break
      }
      case 'BF16': {
        const src = new Uint16Array(this.buffer.slice(byteStart, this.base + end))
        out = new Float32Array(count)
        const u32 = new Uint32Array(1)
        const f32 = new Float32Array(u32.buffer)
        for (let i = 0; i < count; i++) {
          u32[0] = src[i] << 16
          out[i] = f32[0]
        }
        break
      }
      default:
        throw new Error(`unsupported dtype ${info.dtype} for "${name}"`)
    }
    this.cache.set(name, out)
    return out
  }
}

function halfToFloat(h: number): number {
  const s = (h & 0x8000) >> 15
  const e = (h & 0x7c00) >> 10
  const f = h & 0x03ff
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024)
  if (e === 0x1f) return f ? NaN : (s ? -1 : 1) * Infinity
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024)
}

/** Fetch with byte-level progress, so a half-gigabyte download can show itself. */
export async function fetchWithProgress(
  url: string,
  onProgress?: (loaded: number, total: number) => void,
): Promise<ArrayBuffer> {
  const res = await fetch(url)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`)
  const total = Number(res.headers.get('content-length') ?? 0)
  if (!res.body || !total) return res.arrayBuffer()

  const reader = res.body.getReader()
  const chunks: Uint8Array[] = []
  let loaded = 0
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.byteLength
    onProgress?.(loaded, total)
  }
  const out = new Uint8Array(loaded)
  let at = 0
  for (const c of chunks) {
    out.set(c, at)
    at += c.byteLength
  }
  return out.buffer
}

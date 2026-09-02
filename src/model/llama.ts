import { type BPETokenizer, loadTokenizerJson } from './bpe'
import type { LM, LMConfig, LMState, LoadProgress } from './lm'
import { dot, pca3, projectionFidelity, softmax } from './math'
import { fetchWithProgress, Safetensors } from './safetensors'

/**
 * The Llama-family decoder, covering Qwen2 and SmolLM2.
 *
 * Differences from GPT-2 that actually matter here: positions are rotary rather
 * than learned, so there is no position embedding to add; normalisation is
 * RMSNorm with no bias; the feed-forward is a SwiGLU gate; attention is
 * grouped-query, so several query heads share one key/value head; and the output
 * head is the embedding matrix itself. That last one matters for this piece: the
 * lens reads through the very same weights the model reads its input from.
 */
interface Layer {
  inNorm: Float32Array
  q: Float32Array
  qb: Float32Array | null
  k: Float32Array
  kb: Float32Array | null
  v: Float32Array
  vb: Float32Array | null
  o: Float32Array
  postNorm: Float32Array
  gate: Float32Array
  up: Float32Array
  down: Float32Array
}

interface Shape {
  nLayer: number
  nEmbd: number
  nHead: number
  nKvHead: number
  headDim: number
  ffDim: number
  vocabSize: number
  eps: number
  ropeTheta: number
}

/** HuggingFace stores Linear weights as [out, in], so each output is one dot product. */
function matvecT(w: Float32Array, x: Float32Array, b: Float32Array | null, outDim: number, inDim: number, out: Float32Array): void {
  for (let j = 0; j < outDim; j++) out[j] = (b ? b[j] : 0) + dot(w, j * inDim, x, 0, inDim)
}

function rmsNorm(x: Float32Array, g: Float32Array, n: number, eps: number, out: Float32Array): void {
  let ss = 0
  for (let i = 0; i < n; i++) ss += x[i] * x[i]
  const inv = 1 / Math.sqrt(ss / n + eps)
  for (let i = 0; i < n; i++) out[i] = x[i] * inv * g[i]
}

/** Rotary positions, applied in place to one head using the rotate-half convention. */
function rope(vec: Float32Array, off: number, headDim: number, pos: number, theta: number): void {
  const half = headDim / 2
  for (let i = 0; i < half; i++) {
    const freq = Math.pow(theta, (-2 * i) / headDim)
    const c = Math.cos(pos * freq)
    const s = Math.sin(pos * freq)
    const a = vec[off + i]
    const b = vec[off + i + half]
    vec[off + i] = a * c - b * s
    vec[off + i + half] = b * c + a * s
  }
}

const silu = (x: number): number => x / (1 + Math.exp(-x))

export class LlamaLM implements LM {
  readonly cfg: LMConfig
  readonly tok: BPETokenizer
  readonly lensIds: number[]
  readonly lensPieces: string[]
  readonly lensMatrix: Float32Array
  readonly lensPos: Float32Array
  readonly projector: Float32Array
  readonly fidelity: { variance: number; distance: number }
  readonly shape: Shape
  readonly layers: Layer[]
  readonly embed: Float32Array
  readonly finalNorm: Float32Array
  readonly head: Float32Array

  private constructor(shape: Shape, name: string, tok: BPETokenizer, st: Safetensors, nLens: number, seed: number) {
    this.shape = shape
    this.tok = tok
    this.cfg = { nLayer: shape.nLayer, nEmbd: shape.nEmbd, nHead: shape.nHead, vocabSize: shape.vocabSize, name }

    this.embed = st.get('model.embed_tokens.weight')
    this.finalNorm = st.get('model.norm.weight')
    // Tied embeddings: the output head is the embedding matrix.
    this.head = st.has('lm_head.weight') ? st.get('lm_head.weight') : this.embed

    this.layers = []
    for (let l = 0; l < shape.nLayer; l++) {
      const p = `model.layers.${l}`
      this.layers.push({
        inNorm: st.get(`${p}.input_layernorm.weight`),
        q: st.get(`${p}.self_attn.q_proj.weight`),
        qb: st.has(`${p}.self_attn.q_proj.bias`) ? st.get(`${p}.self_attn.q_proj.bias`) : null,
        k: st.get(`${p}.self_attn.k_proj.weight`),
        kb: st.has(`${p}.self_attn.k_proj.bias`) ? st.get(`${p}.self_attn.k_proj.bias`) : null,
        v: st.get(`${p}.self_attn.v_proj.weight`),
        vb: st.has(`${p}.self_attn.v_proj.bias`) ? st.get(`${p}.self_attn.v_proj.bias`) : null,
        o: st.get(`${p}.self_attn.o_proj.weight`),
        postNorm: st.get(`${p}.post_attention_layernorm.weight`),
        gate: st.get(`${p}.mlp.gate_proj.weight`),
        up: st.get(`${p}.mlp.up_proj.weight`),
        down: st.get(`${p}.mlp.down_proj.weight`),
      })
    }

    const limit = nLens > 0 ? Math.min(nLens, shape.vocabSize) : shape.vocabSize
    const ids: number[] = []
    const pieces: string[] = []
    for (let id = 0; id < limit; id++) {
      ids.push(id)
      pieces.push(tok.piece(id))
    }
    this.lensIds = ids
    this.lensPieces = pieces

    const d = shape.nEmbd
    this.lensMatrix = new Float32Array(ids.length * d)
    for (let i = 0; i < ids.length; i++) {
      this.lensMatrix.set(this.head.subarray(ids[i] * d, ids[i] * d + d), i * d)
    }

    // The two best linear directions through the real output-head rows, plus a
    // third. A random projection here keeps almost none of the geometry.
    const { basis, mean } = pca3(this.lensMatrix, ids.length, d, 12)
    this.projector = basis
    this.lensPos = new Float32Array(ids.length * 3)
    for (let i = 0; i < ids.length; i++) {
      for (let c = 0; c < 3; c++) {
        let s = 0
        for (let k = 0; k < d; k++) s += (this.lensMatrix[i * d + k] - mean[k]) * basis[k * 3 + c]
        this.lensPos[i * 3 + c] = s
      }
    }
    this.fidelity = projectionFidelity(this.lensMatrix, this.lensPos, mean, ids.length, d)
    void seed
  }

  static async load(baseUrl: string, name: string, nLens: number, onProgress?: LoadProgress): Promise<LlamaLM> {
    onProgress?.('tokenizer', 0)
    const tok = await loadTokenizerJson(baseUrl)
    const raw = (await fetch(`${baseUrl}/config.json`).then((r) => r.json())) as Record<string, number>
    const nEmbd = raw.hidden_size
    const nHead = raw.num_attention_heads
    const shape: Shape = {
      nLayer: raw.num_hidden_layers,
      nEmbd,
      nHead,
      nKvHead: raw.num_key_value_heads ?? nHead,
      headDim: (raw as unknown as { head_dim?: number }).head_dim ?? nEmbd / nHead,
      ffDim: raw.intermediate_size,
      vocabSize: raw.vocab_size,
      eps: raw.rms_norm_eps ?? 1e-6,
      ropeTheta: raw.rope_theta ?? 10000,
    }
    const buf = await fetchWithProgress(`${baseUrl}/model.safetensors`, (loaded, total) =>
      onProgress?.('weights', loaded / total),
    )
    onProgress?.('unpacking', 1)
    return new LlamaLM(shape, name, tok, new Safetensors(buf), nLens, 20260903)
  }

  newState(maxT: number): LMState {
    return new LlamaState(this, maxT)
  }
}

export class LlamaState implements LMState {
  readonly maxT: number
  readonly ids: number[] = []
  readonly attn: Float32Array[] = []
  readonly lens: Float32Array[] = []
  readonly lensMean: Float32Array[] = []
  readonly lensSd: Float32Array[] = []
  readonly writeNorm: Float32Array[] = []
  readonly residualNorm: Float32Array[] = []
  readonly proj: Float32Array[] = []
  private readonly hidden: Float32Array[] = []
  private readonly kCache: Float32Array[] = []
  private readonly vCache: Float32Array[] = []
  private readonly model: LlamaLM
  private t = 0

  constructor(model: LlamaLM, maxT: number) {
    this.model = model
    this.maxT = maxT
    const { nLayer, nEmbd, nHead, nKvHead, headDim } = model.shape
    const nLens = model.lensIds.length
    const kvDim = nKvHead * headDim
    for (let l = 0; l < nLayer; l++) {
      this.kCache.push(new Float32Array(maxT * kvDim))
      this.vCache.push(new Float32Array(maxT * kvDim))
      this.hidden.push(new Float32Array(maxT * nEmbd))
      this.attn.push(new Float32Array(nHead * maxT * maxT))
      this.lens.push(new Float32Array(maxT * nLens))
      this.lensMean.push(new Float32Array(maxT))
      this.lensSd.push(new Float32Array(maxT))
      this.writeNorm.push(new Float32Array(maxT))
      this.residualNorm.push(new Float32Array(maxT))
      this.proj.push(new Float32Array(maxT * 3))
    }
  }

  get length(): number {
    return this.t
  }

  push(tokenId: number): Float32Array {
    const m = this.model
    const { nLayer, nEmbd, nHead, nKvHead, headDim, ffDim, eps, ropeTheta, vocabSize } = m.shape
    const pos = this.t
    if (pos >= this.maxT) throw new Error('context full')
    this.ids.push(tokenId)

    const qDim = nHead * headDim
    const kvDim = nKvHead * headDim
    const group = nHead / nKvHead
    const scale = 1 / Math.sqrt(headDim)

    const x = new Float32Array(nEmbd)
    x.set(m.embed.subarray(tokenId * nEmbd, (tokenId + 1) * nEmbd))

    const h = new Float32Array(nEmbd)
    const q = new Float32Array(qDim)
    const kv = new Float32Array(kvDim)
    const vv = new Float32Array(kvDim)
    const ctx = new Float32Array(qDim)
    const tmp = new Float32Array(nEmbd)
    const gate = new Float32Array(ffDim)
    const up = new Float32Array(ffDim)
    const row = new Float32Array(this.maxT)

    for (let l = 0; l < nLayer; l++) {
      const b = m.layers[l]
      const before = x.slice()

      rmsNorm(x, b.inNorm, nEmbd, eps, h)
      matvecT(b.q, h, b.qb, qDim, nEmbd, q)
      matvecT(b.k, h, b.kb, kvDim, nEmbd, kv)
      matvecT(b.v, h, b.vb, kvDim, nEmbd, vv)

      for (let i = 0; i < nHead; i++) rope(q, i * headDim, headDim, pos, ropeTheta)
      for (let i = 0; i < nKvHead; i++) rope(kv, i * headDim, headDim, pos, ropeTheta)
      this.kCache[l].set(kv, pos * kvDim)
      this.vCache[l].set(vv, pos * kvDim)

      ctx.fill(0)
      for (let hd = 0; hd < nHead; hd++) {
        // Grouped-query attention: several query heads share one key/value head.
        const kvHead = Math.floor(hd / group)
        const qOff = hd * headDim
        const kvOff = kvHead * headDim
        for (let j = 0; j <= pos; j++) {
          row[j] = dot(q, qOff, this.kCache[l], j * kvDim + kvOff, headDim) * scale
        }
        softmax(row, 0, pos + 1)
        const dst = hd * this.maxT * this.maxT + pos * this.maxT
        for (let j = 0; j <= pos; j++) {
          this.attn[l][dst + j] = row[j]
          const w = row[j]
          if (w < 1e-6) continue
          for (let c = 0; c < headDim; c++) ctx[qOff + c] += w * this.vCache[l][j * kvDim + kvOff + c]
        }
      }
      matvecT(b.o, ctx, null, nEmbd, qDim, tmp)
      for (let i = 0; i < nEmbd; i++) x[i] += tmp[i]

      rmsNorm(x, b.postNorm, nEmbd, eps, h)
      matvecT(b.gate, h, null, ffDim, nEmbd, gate)
      matvecT(b.up, h, null, ffDim, nEmbd, up)
      for (let i = 0; i < ffDim; i++) gate[i] = silu(gate[i]) * up[i]
      matvecT(b.down, gate, null, nEmbd, ffDim, tmp)
      for (let i = 0; i < nEmbd; i++) x[i] += tmp[i]

      this.hidden[l].set(x, pos * nEmbd)
      let rn = 0
      let wn = 0
      for (let i = 0; i < nEmbd; i++) {
        rn += x[i] * x[i]
        wn += (x[i] - before[i]) ** 2
      }
      this.residualNorm[l][pos] = Math.sqrt(rn)
      this.writeNorm[l][pos] = Math.sqrt(wn)
      for (let c = 0; c < 3; c++) {
        let s = 0
        for (let i = 0; i < nEmbd; i++) s += x[i] * m.projector[i * 3 + c]
        this.proj[l][pos * 3 + c] = s
      }
    }

    // --- direct logit attribution, not a logit lens ---
    //
    // The residual stream is a running sum: the embedding plus every layer's
    // write. The output head is linear, so once the final norm's scale is fixed,
    // each layer's contribution to every token's logit is exactly computable and
    // the contributions sum to the model's real logits. Freezing that scale is
    // the standard treatment and it is what makes the numbers add up — read at
    // the last layer this is not an approximation of the output, it is the
    // output. Recomputing the norm separately at each depth, which is what the
    // logit lens does, is what makes the lens unreliable on this architecture.
    let ss = 0
    for (let i = 0; i < nEmbd; i++) ss += x[i] * x[i]
    const frozen = 1 / Math.sqrt(ss / nEmbd + eps)
    for (let l = 0; l < nLayer; l++) this.attributeAt(l, pos, frozen)

    rmsNorm(x, m.finalNorm, nEmbd, eps, h)
    const logits = new Float32Array(vocabSize)
    for (let v = 0; v < vocabSize; v++) logits[v] = dot(m.head, v * nEmbd, h, 0, nEmbd)
    this.t++
    return logits
  }

  /**
   * The running total of the answer, as of this depth.
   *
   * Applies the model's real final-norm weights with the scale frozen from the
   * finished residual, then reads through the real output head. At the last
   * layer this equals the model's logits exactly; at every earlier layer it is
   * the partial sum of contributions that produced them.
   */
  private attributeAt(layer: number, pos: number, frozen: number): void {
    const m = this.model
    const { nEmbd } = m.shape
    const nLens = m.lensIds.length
    const h = new Float32Array(nEmbd)
    const src = layer * 0 + pos * nEmbd
    for (let i = 0; i < nEmbd; i++) h[i] = this.hidden[layer][src + i] * frozen * m.finalNorm[i]
    const out = this.lens[layer]
    const base = pos * nLens
    let mean = 0
    for (let i = 0; i < nLens; i++) {
      const s = dot(m.lensMatrix, i * nEmbd, h, 0, nEmbd)
      out[base + i] = s
      mean += s
    }
    mean /= nLens
    let sd = 0
    for (let i = 0; i < nLens; i++) sd += (out[base + i] - mean) ** 2
    this.lensMean[layer][pos] = mean
    this.lensSd[layer][pos] = Math.sqrt(sd / nLens) || 1
  }

  hiddenAt(layer: number): Float32Array {
    return this.hidden[layer]
  }
}

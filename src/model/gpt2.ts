import { BPETokenizer, loadTokenizer } from './bpe'
import { dot, gelu, layerNormAffine, mulberry32, softmax } from './math'
import { fetchWithProgress, Safetensors } from './safetensors'

export interface GPT2Config {
  nLayer: number
  nHead: number
  nEmbd: number
  nCtx: number
  vocabSize: number
  eps: number
}

interface Block {
  ln1g: Float32Array
  ln1b: Float32Array
  attnW: Float32Array
  attnB: Float32Array
  projW: Float32Array
  projB: Float32Array
  ln2g: Float32Array
  ln2b: Float32Array
  fcW: Float32Array
  fcB: Float32Array
  mlpProjW: Float32Array
  mlpProjB: Float32Array
}

/** out[p] = x[n] · W[n,p] + b[p] */
function vecmat(x: Float32Array, xOff: number, w: Float32Array, b: Float32Array | null, n: number, p: number, out: Float32Array, outOff = 0): void {
  if (b) out.set(b.subarray(0, p), outOff)
  else out.fill(0, outOff, outOff + p)
  for (let k = 0; k < n; k++) {
    const xv = x[xOff + k]
    if (xv === 0) continue
    const wOff = k * p
    for (let j = 0; j < p; j++) out[outOff + j] += xv * w[wOff + j]
  }
}

export class GPT2 {
  readonly cfg: GPT2Config
  readonly tok: BPETokenizer
  readonly blocks: Block[]
  readonly wte: Float32Array
  readonly wpe: Float32Array
  readonly lnFg: Float32Array
  readonly lnFb: Float32Array

  /**
   * The display vocabulary: the tokens drawn as points.
   *
   * All 50,257 would be unreadable and would cost more to score than the model
   * costs to run, so this is the first few thousand whole-word tokens — which,
   * because GPT-2's ids run roughly in merge-frequency order, are the common
   * English words.
   */
  readonly lensIds: number[]
  readonly lensPieces: string[]
  /** nLens × nEmbd — real rows of the unembedding matrix. */
  readonly lensMatrix: Float32Array
  /** nLens × 3 — those same rows under one fixed projection, for drawing. */
  readonly lensPos: Float32Array
  /** nEmbd × 3 — the same projection the residual stream is drawn through. */
  readonly projector: Float32Array

  private constructor(cfg: GPT2Config, tok: BPETokenizer, st: Safetensors, nLens: number, seed: number) {
    this.cfg = cfg
    this.tok = tok
    this.wte = st.get('wte.weight')
    this.wpe = st.get('wpe.weight')
    this.lnFg = st.get('ln_f.weight')
    this.lnFb = st.get('ln_f.bias')
    this.blocks = []
    for (let l = 0; l < cfg.nLayer; l++) {
      this.blocks.push({
        ln1g: st.get(`h.${l}.ln_1.weight`),
        ln1b: st.get(`h.${l}.ln_1.bias`),
        attnW: st.get(`h.${l}.attn.c_attn.weight`),
        attnB: st.get(`h.${l}.attn.c_attn.bias`),
        projW: st.get(`h.${l}.attn.c_proj.weight`),
        projB: st.get(`h.${l}.attn.c_proj.bias`),
        ln2g: st.get(`h.${l}.ln_2.weight`),
        ln2b: st.get(`h.${l}.ln_2.bias`),
        fcW: st.get(`h.${l}.mlp.c_fc.weight`),
        fcB: st.get(`h.${l}.mlp.c_fc.bias`),
        mlpProjW: st.get(`h.${l}.mlp.c_proj.weight`),
        mlpProjB: st.get(`h.${l}.mlp.c_proj.bias`),
      })
    }

    const ids: number[] = []
    const pieces: string[] = []
    for (let id = 0; id < cfg.vocabSize && ids.length < nLens; id++) {
      const piece = tok.piece(id)
      if (/^ [A-Za-z]{2,}$/.test(piece)) {
        ids.push(id)
        pieces.push(piece)
      }
    }
    this.lensIds = ids
    this.lensPieces = pieces

    const d = cfg.nEmbd
    this.lensMatrix = new Float32Array(ids.length * d)
    for (let i = 0; i < ids.length; i++) {
      this.lensMatrix.set(this.wte.subarray(ids[i] * d, ids[i] * d + d), i * d)
    }

    // One fixed random projection of the real unembedding rows. Two points are
    // close on screen exactly when GPT-2 puts those two tokens close together.
    const rand = mulberry32(seed)
    const proj = new Float32Array(d * 3)
    for (let i = 0; i < proj.length; i++) {
      let u = 0
      while (u === 0) u = rand()
      proj[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()) / Math.sqrt(d)
    }
    this.projector = proj
    this.lensPos = new Float32Array(ids.length * 3)
    for (let i = 0; i < ids.length; i++) {
      for (let c = 0; c < 3; c++) {
        let s = 0
        for (let k = 0; k < d; k++) s += this.lensMatrix[i * d + k] * proj[k * 3 + c]
        this.lensPos[i * 3 + c] = s
      }
    }
  }

  static async load(
    baseUrl: string,
    nLens: number,
    onProgress?: (phase: string, frac: number) => void,
  ): Promise<GPT2> {
    onProgress?.('tokenizer', 0)
    const tok = await loadTokenizer(baseUrl)
    onProgress?.('config', 0)
    const raw = (await fetch(`${baseUrl}/config.json`).then((r) => r.json())) as Record<string, number>
    const cfg: GPT2Config = {
      nLayer: raw.n_layer,
      nHead: raw.n_head,
      nEmbd: raw.n_embd,
      nCtx: raw.n_ctx,
      vocabSize: raw.vocab_size,
      eps: raw.layer_norm_epsilon ?? 1e-5,
    }
    const buf = await fetchWithProgress(`${baseUrl}/model.safetensors`, (loaded, total) =>
      onProgress?.('weights', loaded / total),
    )
    onProgress?.('unpacking', 1)
    return new GPT2(cfg, tok, new Safetensors(buf), nLens, 20260903)
  }

  newState(maxT: number): GPT2State {
    return new GPT2State(this, maxT)
  }
}

/**
 * One decoding run, held incrementally.
 *
 * GPT-2 is causal, so a position's activations never change once computed. Real
 * decoding exploits that with a KV cache and recomputes nothing; this keeps the
 * same discipline, which is both ~15x cheaper than re-running the whole context
 * per word and a more accurate picture of what generation actually does. Each
 * new word is one position sweeping the stack while everything before it sits
 * already settled.
 */
export class GPT2State {
  readonly model: GPT2
  readonly maxT: number
  readonly kCache: Float32Array[]
  readonly vCache: Float32Array[]
  /** Per layer: maxT × nEmbd residual after the block. */
  readonly hidden: Float32Array[]
  /** Per layer: nHead × maxT × maxT attention, filled in row by row. */
  readonly attn: Float32Array[]
  /** Per layer: maxT × nLens raw logit-lens scores over the display tokens. */
  readonly lens: Float32Array[]
  /** Per layer: mean and sd of those scores, per position, for standardising. */
  readonly lensMean: Float32Array[]
  readonly lensSd: Float32Array[]
  /** Per layer: how much the block wrote into the stream, per position. */
  readonly writeNorm: Float32Array[]
  readonly residualNorm: Float32Array[]
  readonly proj: Float32Array[]
  readonly ids: number[] = []
  private t = 0

  constructor(model: GPT2, maxT: number) {
    this.model = model
    this.maxT = maxT
    const { nLayer, nEmbd, nHead } = model.cfg
    const nLens = model.lensIds.length
    this.kCache = []
    this.vCache = []
    this.hidden = []
    this.attn = []
    this.lens = []
    this.lensMean = []
    this.lensSd = []
    this.writeNorm = []
    this.residualNorm = []
    this.proj = []
    for (let l = 0; l < nLayer; l++) {
      this.kCache.push(new Float32Array(maxT * nEmbd))
      this.vCache.push(new Float32Array(maxT * nEmbd))
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

  /** Run one token through all twelve layers. Returns the next-token logits. */
  push(tokenId: number): Float32Array {
    const m = this.model
    const { nLayer, nEmbd, nHead, eps } = m.cfg
    const dHead = nEmbd / nHead
    const pos = this.t
    if (pos >= this.maxT) throw new Error('context full')
    this.ids.push(tokenId)

    const x = new Float32Array(nEmbd)
    for (let i = 0; i < nEmbd; i++) x[i] = m.wte[tokenId * nEmbd + i] + m.wpe[pos * nEmbd + i]

    const h = new Float32Array(nEmbd)
    const qkv = new Float32Array(3 * nEmbd)
    const ctx = new Float32Array(nEmbd)
    const ff = new Float32Array(4 * nEmbd)
    const tmp = new Float32Array(nEmbd)
    const scale = 1 / Math.sqrt(dHead)

    for (let l = 0; l < nLayer; l++) {
      const b = m.blocks[l]
      const before = x.slice()

      layerNormAffine(x, 0, nEmbd, b.ln1g, b.ln1b, eps, h, 0)
      vecmat(h, 0, b.attnW, b.attnB, nEmbd, 3 * nEmbd, qkv)
      this.kCache[l].set(qkv.subarray(nEmbd, 2 * nEmbd), pos * nEmbd)
      this.vCache[l].set(qkv.subarray(2 * nEmbd, 3 * nEmbd), pos * nEmbd)

      ctx.fill(0)
      const row = new Float32Array(pos + 1)
      for (let hd = 0; hd < nHead; hd++) {
        const off = hd * dHead
        for (let j = 0; j <= pos; j++) {
          row[j] = dot(qkv, off, this.kCache[l], j * nEmbd + off, dHead) * scale
        }
        softmax(row, 0, pos + 1)
        const dst = hd * this.maxT * this.maxT + pos * this.maxT
        for (let j = 0; j <= pos; j++) {
          this.attn[l][dst + j] = row[j]
          const w = row[j]
          if (w < 1e-6) continue
          for (let c = 0; c < dHead; c++) ctx[off + c] += w * this.vCache[l][j * nEmbd + off + c]
        }
      }
      vecmat(ctx, 0, b.projW, b.projB, nEmbd, nEmbd, tmp)
      for (let i = 0; i < nEmbd; i++) x[i] += tmp[i]

      layerNormAffine(x, 0, nEmbd, b.ln2g, b.ln2b, eps, h, 0)
      vecmat(h, 0, b.fcW, b.fcB, nEmbd, 4 * nEmbd, ff)
      for (let i = 0; i < ff.length; i++) ff[i] = gelu(ff[i])
      vecmat(ff, 0, b.mlpProjW, b.mlpProjB, 4 * nEmbd, nEmbd, tmp)
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
      this.lensAt(l, pos)
    }

    layerNormAffine(x, 0, nEmbd, m.lnFg, m.lnFb, eps, h, 0)
    const logits = new Float32Array(m.cfg.vocabSize)
    for (let v = 0; v < m.cfg.vocabSize; v++) logits[v] = dot(m.wte, v * nEmbd, h, 0, nEmbd)
    this.t++
    return logits
  }

  /**
   * The logit lens at one layer.
   *
   * Take the residual as it stands part-way up the stack, apply the model's own
   * final layer norm, and read it against the real unembedding.
   *
   * At the last layer this is not an approximation of anything — it is exactly
   * the computation that produces the model's output, and measures as such
   * (KL of zero against the real distribution). Every earlier layer is a
   * projection, and a biased one: ln_f is fitted to the final layer, so applying
   * it further down applies statistics that do not belong there. Early layers
   * therefore report the lens's own preferences as much as the model's. The
   * agreement figure carried on each layer of the trace is that bias, measured
   * rather than disclaimed.
   *
   * Raw scores are kept here and standardised at read time, because a softmax
   * over standardised scores is a different distribution and would make the
   * agreement measurement meaningless.
   */
  private lensAt(layer: number, pos: number): void {
    const m = this.model
    const { nEmbd, eps } = m.cfg
    const nLens = m.lensIds.length
    const h = new Float32Array(nEmbd)
    layerNormAffine(this.hidden[layer], pos * nEmbd, nEmbd, m.lnFg, m.lnFb, eps, h, 0)
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
    sd = Math.sqrt(sd / nLens) || 1
    this.lensMean[layer][pos] = mean
    this.lensSd[layer][pos] = sd
  }
}

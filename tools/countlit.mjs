// tools/countlit.ts
import { readFileSync } from "node:fs";

// src/model/bpe.ts
function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n++;
    }
  }
  const map = /* @__PURE__ */ new Map();
  for (let i = 0; i < bs.length; i++) map.set(bs[i], String.fromCodePoint(cs[i]));
  return map;
}
var GPT2_PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
var BPETokenizer = class {
  vocabSize;
  encoder;
  decoder;
  ranks;
  byteEncoder = bytesToUnicode();
  byteDecoder = /* @__PURE__ */ new Map();
  cache = /* @__PURE__ */ new Map();
  pattern;
  special = /* @__PURE__ */ new Set();
  constructor(vocabJson, merges, pattern = GPT2_PATTERN, added = []) {
    this.pattern = pattern;
    this.encoder = new Map(Object.entries(vocabJson));
    for (const t2 of added) {
      this.encoder.set(t2.content, t2.id);
      this.special.add(t2.id);
    }
    let maxId = 0;
    for (const id of this.encoder.values()) if (id > maxId) maxId = id;
    this.vocabSize = maxId + 1;
    this.decoder = new Array(this.vocabSize);
    for (const [tok, id] of this.encoder) this.decoder[id] = tok;
    for (const [b, c] of this.byteEncoder) this.byteDecoder.set(c, b);
    this.ranks = /* @__PURE__ */ new Map();
    const lines = typeof merges === "string" ? merges.split("\n") : merges;
    let rank = 0;
    for (const line of lines) {
      const text = Array.isArray(line) ? `${line[0]} ${line[1]}` : line;
      if (!text || text.startsWith("#")) continue;
      this.ranks.set(text.trim(), rank++);
    }
  }
  /** True for control tokens like <|endoftext|>, which should not be emitted. */
  isSpecial(id) {
    return this.special.has(id);
  }
  bpe(token) {
    let word = Array.from(token);
    if (word.length === 1) return word;
    for (; ; ) {
      let bestRank = Infinity;
      let bestAt = -1;
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(`${word[i]} ${word[i + 1]}`);
        if (r !== void 0 && r < bestRank) {
          bestRank = r;
          bestAt = i;
        }
      }
      if (bestAt < 0) break;
      word = [...word.slice(0, bestAt), word[bestAt] + word[bestAt + 1], ...word.slice(bestAt + 2)];
      if (word.length === 1) break;
    }
    return word;
  }
  encode(text) {
    const out = [];
    for (const [chunk] of text.matchAll(this.pattern)) {
      const hit = this.cache.get(chunk);
      if (hit) {
        out.push(...hit);
        continue;
      }
      let mapped = "";
      for (const byte of new TextEncoder().encode(chunk)) mapped += this.byteEncoder.get(byte);
      const ids = this.bpe(mapped).map((piece) => this.encoder.get(piece) ?? 0);
      this.cache.set(chunk, ids);
      out.push(...ids);
    }
    return out;
  }
  /** The printable form of one token, with its leading space made visible. */
  piece(id) {
    return this.decodeTokens([id]);
  }
  decode(ids) {
    return this.decodeTokens(ids);
  }
  decodeTokens(ids) {
    let joined = "";
    for (const id of ids) joined += this.decoder[id] ?? "";
    const bytes = new Uint8Array(Array.from(joined).map((c) => this.byteDecoder.get(c) ?? 0));
    return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
  }
};
async function loadTokenizer(baseUrl) {
  const [vocab, merges] = await Promise.all([
    fetch(`${baseUrl}/vocab.json`).then((r) => r.json()),
    fetch(`${baseUrl}/merges.txt`).then((r) => r.text())
  ]);
  return new BPETokenizer(vocab, merges);
}

// src/model/math.ts
function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a = a + 1831565813 >>> 0;
    let t2 = a;
    t2 = Math.imul(t2 ^ t2 >>> 15, t2 | 1);
    t2 ^= t2 + Math.imul(t2 ^ t2 >>> 7, t2 | 61);
    return ((t2 ^ t2 >>> 14) >>> 0) / 4294967296;
  };
}
function softmax(v, off = 0, len = v.length - off) {
  let max = -Infinity;
  for (let i = 0; i < len; i++) if (v[off + i] > max) max = v[off + i];
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const e = Math.exp(v[off + i] - max);
    v[off + i] = e;
    sum += e;
  }
  const inv = 1 / (sum || 1);
  for (let i = 0; i < len; i++) v[off + i] *= inv;
}
function gelu(x) {
  return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
}
function dot(a, aOff, b, bOff, n) {
  let s = 0;
  for (let i = 0; i < n; i++) s += a[aOff + i] * b[bOff + i];
  return s;
}
function orthonormalize(z, rows, cols) {
  for (let j = 0; j < cols; j++) {
    for (let k = 0; k < j; k++) {
      let d = 0;
      for (let i = 0; i < rows; i++) d += z[i * cols + j] * z[i * cols + k];
      for (let i = 0; i < rows; i++) z[i * cols + j] -= d * z[i * cols + k];
    }
    let n = 0;
    for (let i = 0; i < rows; i++) n += z[i * cols + j] ** 2;
    n = Math.sqrt(n) || 1e-8;
    for (let i = 0; i < rows; i++) z[i * cols + j] /= n;
  }
}
function layerNormAffine(v, off, d, gamma, beta, eps, out, outOff) {
  let mean = 0;
  for (let i = 0; i < d; i++) mean += v[off + i];
  mean /= d;
  let varr = 0;
  for (let i = 0; i < d; i++) {
    const x = v[off + i] - mean;
    varr += x * x;
  }
  varr /= d;
  const inv = 1 / Math.sqrt(varr + eps);
  for (let i = 0; i < d; i++) out[outOff + i] = (v[off + i] - mean) * inv * gamma[i] + beta[i];
}
function pca3(rows, n, d, iterations = 24) {
  const mean = new Float32Array(d);
  for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) mean[k] += rows[i * d + k] / n;
  const rand = mulberry32(1);
  const z = new Float32Array(d * 3);
  for (let i = 0; i < z.length; i++) z[i] = rand() * 2 - 1;
  orthonormalize(z, d, 3);
  const t2 = new Float32Array(n * 3);
  for (let it = 0; it < iterations; it++) {
    t2.fill(0);
    for (let i = 0; i < n; i++) {
      const off = i * d;
      for (let k = 0; k < d; k++) {
        const v = rows[off + k] - mean[k];
        if (v === 0) continue;
        const zo = k * 3;
        t2[i * 3] += v * z[zo];
        t2[i * 3 + 1] += v * z[zo + 1];
        t2[i * 3 + 2] += v * z[zo + 2];
      }
    }
    z.fill(0);
    for (let i = 0; i < n; i++) {
      const off = i * d;
      const a = t2[i * 3];
      const b = t2[i * 3 + 1];
      const c = t2[i * 3 + 2];
      for (let k = 0; k < d; k++) {
        const v = rows[off + k] - mean[k];
        if (v === 0) continue;
        const zo = k * 3;
        z[zo] += v * a;
        z[zo + 1] += v * b;
        z[zo + 2] += v * c;
      }
    }
    orthonormalize(z, d, 3);
  }
  return { basis: z, mean };
}
function projectionFidelity(rows, projected, mean, n, d, samples = 3e3) {
  let total = 0;
  for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) total += (rows[i * d + k] - mean[k]) ** 2;
  let kept = 0;
  for (let i = 0; i < projected.length; i++) kept += projected[i] * projected[i];
  const rand = mulberry32(99);
  const hi = [];
  const lo = [];
  for (let s = 0; s < samples; s++) {
    const i = Math.floor(rand() * n);
    const j = Math.floor(rand() * n);
    if (i === j) continue;
    let h = 0;
    for (let k = 0; k < d; k++) h += (rows[i * d + k] - rows[j * d + k]) ** 2;
    hi.push(Math.sqrt(h));
    lo.push(
      Math.hypot(
        projected[i * 3] - projected[j * 3],
        projected[i * 3 + 1] - projected[j * 3 + 1],
        projected[i * 3 + 2] - projected[j * 3 + 2]
      )
    );
  }
  const mh = hi.reduce((a, b) => a + b, 0) / hi.length;
  const ml = lo.reduce((a, b) => a + b, 0) / lo.length;
  let num = 0;
  let dh = 0;
  let dl = 0;
  for (let i = 0; i < hi.length; i++) {
    num += (hi[i] - mh) * (lo[i] - ml);
    dh += (hi[i] - mh) ** 2;
    dl += (lo[i] - ml) ** 2;
  }
  return { variance: kept / (total || 1), distance: num / (Math.sqrt(dh * dl) || 1) };
}

// src/model/layout.ts
async function loadLayout(baseUrl) {
  let buf;
  try {
    const res = await fetch(`${baseUrl}/layout.bin`);
    if (!res.ok) return null;
    buf = await res.arrayBuffer();
  } catch {
    return null;
  }
  const head = new Int32Array(buf, 0, 2);
  const n = head[0];
  const k = head[1];
  let off = 8;
  const pos = new Float32Array(buf.slice(off, off + n * 3 * 4));
  off += n * 3 * 4;
  const nbr = new Int32Array(buf.slice(off, off + n * k * 4));
  off += n * k * 4;
  const sim = new Float32Array(buf.slice(off, off + n * k * 4));
  let preservation = 0;
  try {
    const meta = await fetch(`${baseUrl}/layout.json`).then((r) => r.ok ? r.json() : null);
    preservation = meta?.preservation ?? 0;
  } catch {
    preservation = 0;
  }
  return { n, k, pos, nbr, sim, preservation };
}

// src/model/safetensors.ts
var Safetensors = class {
  header;
  buffer;
  base;
  cache = /* @__PURE__ */ new Map();
  constructor(buffer) {
    this.buffer = buffer;
    const view = new DataView(buffer);
    const headerLen = Number(view.getBigUint64(0, true));
    const json = new TextDecoder().decode(new Uint8Array(buffer, 8, headerLen));
    const parsed = JSON.parse(json);
    delete parsed.__metadata__;
    this.header = parsed;
    this.base = 8 + headerLen;
  }
  names() {
    return Object.keys(this.header);
  }
  has(name) {
    return name in this.header;
  }
  shape(name) {
    const info = this.header[name];
    if (!info) throw new Error(`no tensor "${name}"`);
    return info.shape;
  }
  /** Returns the tensor as f32, converting from f16/bf16 when needed. Cached. */
  get(name) {
    const hit = this.cache.get(name);
    if (hit) return hit;
    const info = this.header[name];
    if (!info) throw new Error(`no tensor "${name}"`);
    const [start, end] = info.data_offsets;
    const byteStart = this.base + start;
    const count = info.shape.reduce((a, b) => a * b, 1);
    let out;
    switch (info.dtype) {
      case "F32":
        out = byteStart % 4 === 0 ? new Float32Array(this.buffer, byteStart, count) : new Float32Array(this.buffer.slice(byteStart, this.base + end));
        break;
      case "F16": {
        const src = new Uint16Array(this.buffer.slice(byteStart, this.base + end));
        out = new Float32Array(count);
        for (let i = 0; i < count; i++) out[i] = halfToFloat(src[i]);
        break;
      }
      case "BF16": {
        const src = new Uint16Array(this.buffer.slice(byteStart, this.base + end));
        out = new Float32Array(count);
        const u32 = new Uint32Array(1);
        const f32 = new Float32Array(u32.buffer);
        for (let i = 0; i < count; i++) {
          u32[0] = src[i] << 16;
          out[i] = f32[0];
        }
        break;
      }
      default:
        throw new Error(`unsupported dtype ${info.dtype} for "${name}"`);
    }
    this.cache.set(name, out);
    return out;
  }
};
function halfToFloat(h) {
  const s = (h & 32768) >> 15;
  const e = (h & 31744) >> 10;
  const f = h & 1023;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -1 : 1) * Infinity;
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}
async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  if (!res.body || !total) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (; ; ) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.byteLength;
  }
  return out.buffer;
}

// src/model/gpt2.ts
function vecmat(x, xOff, w, b, n, p, out, outOff = 0) {
  if (b) out.set(b.subarray(0, p), outOff);
  else out.fill(0, outOff, outOff + p);
  for (let k = 0; k < n; k++) {
    const xv = x[xOff + k];
    if (xv === 0) continue;
    const wOff = k * p;
    for (let j = 0; j < p; j++) out[outOff + j] += xv * w[wOff + j];
  }
}
var GPT2 = class _GPT2 {
  cfg;
  tok;
  blocks;
  wte;
  wpe;
  lnFg;
  lnFb;
  /**
   * The display vocabulary: the tokens drawn as points.
   *
   * All 50,257 would be unreadable and would cost more to score than the model
   * costs to run, so this is the first few thousand whole-word tokens — which,
   * because GPT-2's ids run roughly in merge-frequency order, are the common
   * English words.
   */
  lensIds;
  lensPieces;
  /** nLens × nEmbd — real rows of the unembedding matrix. */
  lensMatrix;
  /** nLens × 3 — those same rows under one fixed projection, for drawing. */
  lensPos;
  /** nEmbd × 3 — the same projection the residual stream is drawn through. */
  projector;
  fidelity;
  layout = null;
  constructor(cfg, tok, st, nLens, seed) {
    this.cfg = cfg;
    this.tok = tok;
    this.wte = st.get("wte.weight");
    this.wpe = st.get("wpe.weight");
    this.lnFg = st.get("ln_f.weight");
    this.lnFb = st.get("ln_f.bias");
    this.blocks = [];
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
        mlpProjB: st.get(`h.${l}.mlp.c_proj.bias`)
      });
    }
    const limit = nLens > 0 ? Math.min(nLens, cfg.vocabSize) : cfg.vocabSize;
    const ids = [];
    const pieces = [];
    for (let id = 0; id < limit; id++) {
      ids.push(id);
      pieces.push(tok.piece(id));
    }
    this.lensIds = ids;
    this.lensPieces = pieces;
    const d = cfg.nEmbd;
    this.lensMatrix = new Float32Array(ids.length * d);
    for (let i = 0; i < ids.length; i++) {
      this.lensMatrix.set(this.wte.subarray(ids[i] * d, ids[i] * d + d), i * d);
    }
    const { basis, mean } = pca3(this.lensMatrix, ids.length, d, 12);
    this.projector = basis;
    this.lensPos = new Float32Array(ids.length * 3);
    for (let i = 0; i < ids.length; i++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let k = 0; k < d; k++) s += (this.lensMatrix[i * d + k] - mean[k]) * basis[k * 3 + c];
        this.lensPos[i * 3 + c] = s;
      }
    }
    this.fidelity = projectionFidelity(this.lensMatrix, this.lensPos, mean, ids.length, d);
    void seed;
  }
  static async load(baseUrl, nLens, onProgress) {
    onProgress?.("tokenizer", 0);
    const tok = await loadTokenizer(baseUrl);
    onProgress?.("config", 0);
    const raw = await fetch(`${baseUrl}/config.json`).then((r) => r.json());
    const cfg = {
      nLayer: raw.n_layer,
      nHead: raw.n_head,
      nEmbd: raw.n_embd,
      nCtx: raw.n_ctx,
      vocabSize: raw.vocab_size,
      eps: raw.layer_norm_epsilon ?? 1e-5
    };
    const buf = await fetchWithProgress(
      `${baseUrl}/model.safetensors`,
      (loaded, total) => onProgress?.("weights", loaded / total)
    );
    onProgress?.("unpacking", 1);
    const model = new _GPT2(cfg, tok, new Safetensors(buf), nLens, 20260903);
    model.layout = await loadLayout(baseUrl);
    return model;
  }
  newState(maxT) {
    return new GPT2State(this, maxT);
  }
};
var GPT2State = class {
  model;
  maxT;
  kCache;
  vCache;
  /** Per layer: maxT × nEmbd residual after the block. */
  hidden;
  /** Per layer: nHead × maxT × maxT attention, filled in row by row. */
  attn;
  /** Per layer: maxT × nLens raw logit-lens scores over the display tokens. */
  lens;
  /** Per layer: mean and sd of those scores, per position, for standardising. */
  lensMean;
  lensSd;
  /** Per layer: how much the block wrote into the stream, per position. */
  writeNorm;
  residualNorm;
  proj;
  ids = [];
  t = 0;
  constructor(model, maxT) {
    this.model = model;
    this.maxT = maxT;
    const { nLayer, nEmbd, nHead } = model.cfg;
    const nLens = model.lensIds.length;
    this.kCache = [];
    this.vCache = [];
    this.hidden = [];
    this.attn = [];
    this.lens = [];
    this.lensMean = [];
    this.lensSd = [];
    this.writeNorm = [];
    this.residualNorm = [];
    this.proj = [];
    for (let l = 0; l < nLayer; l++) {
      this.kCache.push(new Float32Array(maxT * nEmbd));
      this.vCache.push(new Float32Array(maxT * nEmbd));
      this.hidden.push(new Float32Array(maxT * nEmbd));
      this.attn.push(new Float32Array(nHead * maxT * maxT));
      this.lens.push(new Float32Array(maxT * nLens));
      this.lensMean.push(new Float32Array(maxT));
      this.lensSd.push(new Float32Array(maxT));
      this.writeNorm.push(new Float32Array(maxT));
      this.residualNorm.push(new Float32Array(maxT));
      this.proj.push(new Float32Array(maxT * 3));
    }
  }
  get length() {
    return this.t;
  }
  /** Run one token through all twelve layers. Returns the next-token logits. */
  push(tokenId) {
    const m2 = this.model;
    const { nLayer, nEmbd, nHead, eps } = m2.cfg;
    const dHead = nEmbd / nHead;
    const pos = this.t;
    if (pos >= this.maxT) throw new Error("context full");
    this.ids.push(tokenId);
    const x = new Float32Array(nEmbd);
    for (let i = 0; i < nEmbd; i++) x[i] = m2.wte[tokenId * nEmbd + i] + m2.wpe[pos * nEmbd + i];
    const h = new Float32Array(nEmbd);
    const qkv = new Float32Array(3 * nEmbd);
    const ctx = new Float32Array(nEmbd);
    const ff = new Float32Array(4 * nEmbd);
    const tmp = new Float32Array(nEmbd);
    const scale = 1 / Math.sqrt(dHead);
    for (let l = 0; l < nLayer; l++) {
      const b = m2.blocks[l];
      const before = x.slice();
      layerNormAffine(x, 0, nEmbd, b.ln1g, b.ln1b, eps, h, 0);
      vecmat(h, 0, b.attnW, b.attnB, nEmbd, 3 * nEmbd, qkv);
      this.kCache[l].set(qkv.subarray(nEmbd, 2 * nEmbd), pos * nEmbd);
      this.vCache[l].set(qkv.subarray(2 * nEmbd, 3 * nEmbd), pos * nEmbd);
      ctx.fill(0);
      const row = new Float32Array(pos + 1);
      for (let hd = 0; hd < nHead; hd++) {
        const off = hd * dHead;
        for (let j = 0; j <= pos; j++) {
          row[j] = dot(qkv, off, this.kCache[l], j * nEmbd + off, dHead) * scale;
        }
        softmax(row, 0, pos + 1);
        const dst = hd * this.maxT * this.maxT + pos * this.maxT;
        for (let j = 0; j <= pos; j++) {
          this.attn[l][dst + j] = row[j];
          const w = row[j];
          if (w < 1e-6) continue;
          for (let c = 0; c < dHead; c++) ctx[off + c] += w * this.vCache[l][j * nEmbd + off + c];
        }
      }
      vecmat(ctx, 0, b.projW, b.projB, nEmbd, nEmbd, tmp);
      for (let i = 0; i < nEmbd; i++) x[i] += tmp[i];
      layerNormAffine(x, 0, nEmbd, b.ln2g, b.ln2b, eps, h, 0);
      vecmat(h, 0, b.fcW, b.fcB, nEmbd, 4 * nEmbd, ff);
      for (let i = 0; i < ff.length; i++) ff[i] = gelu(ff[i]);
      vecmat(ff, 0, b.mlpProjW, b.mlpProjB, 4 * nEmbd, nEmbd, tmp);
      for (let i = 0; i < nEmbd; i++) x[i] += tmp[i];
      this.hidden[l].set(x, pos * nEmbd);
      let rn = 0;
      let wn = 0;
      for (let i = 0; i < nEmbd; i++) {
        rn += x[i] * x[i];
        wn += (x[i] - before[i]) ** 2;
      }
      this.residualNorm[l][pos] = Math.sqrt(rn);
      this.writeNorm[l][pos] = Math.sqrt(wn);
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let i = 0; i < nEmbd; i++) s += x[i] * m2.projector[i * 3 + c];
        this.proj[l][pos * 3 + c] = s;
      }
      this.lensAt(l, pos);
    }
    layerNormAffine(x, 0, nEmbd, m2.lnFg, m2.lnFb, eps, h, 0);
    const logits = new Float32Array(m2.cfg.vocabSize);
    for (let v = 0; v < m2.cfg.vocabSize; v++) logits[v] = dot(m2.wte, v * nEmbd, h, 0, nEmbd);
    this.t++;
    return logits;
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
  lensAt(layer, pos) {
    const m2 = this.model;
    const { nEmbd, eps } = m2.cfg;
    const nLens = m2.lensIds.length;
    const h = new Float32Array(nEmbd);
    layerNormAffine(this.hidden[layer], pos * nEmbd, nEmbd, m2.lnFg, m2.lnFb, eps, h, 0);
    const out = this.lens[layer];
    const base = pos * nLens;
    let mean = 0;
    for (let i = 0; i < nLens; i++) {
      const s = dot(m2.lensMatrix, i * nEmbd, h, 0, nEmbd);
      out[base + i] = s;
      mean += s;
    }
    mean /= nLens;
    let sd = 0;
    for (let i = 0; i < nLens; i++) sd += (out[base + i] - mean) ** 2;
    sd = Math.sqrt(sd / nLens) || 1;
    this.lensMean[layer][pos] = mean;
    this.lensSd[layer][pos] = sd;
  }
};

// src/model/run.ts
var MAX_CONTEXT = 48;
var Z_PUSH = 2.2;
function sampleFrom(probs, order, temperature, topP, rand) {
  if (temperature <= 1e-3) return order[0];
  const scaled = order.map((id) => Math.pow(Math.max(probs[id], 1e-12), 1 / Math.max(temperature, 0.05)));
  const total = scaled.reduce((a, b) => a + b, 0);
  let cum = 0;
  const keep = [];
  for (let i = 0; i < order.length; i++) {
    keep.push(i);
    cum += scaled[i] / total;
    if (cum >= topP) break;
  }
  let r = rand() * keep.reduce((a, i) => a + scaled[i] / total, 0);
  for (const i of keep) {
    r -= scaled[i] / total;
    if (r <= 0) return order[i];
  }
  return order[keep[keep.length - 1]];
}
function readLayer(model, state, layer, T) {
  const nLens = model.lensIds.length;
  const features = [];
  const suppressed = [];
  const contended = new Int32Array(T);
  let activations = 0;
  const now = state.lens[layer];
  const prev = layer > 0 ? state.lens[layer - 1] : null;
  const delta = new Float32Array(nLens);
  for (let pos = 0; pos < T; pos++) {
    const base = pos * nLens;
    let mean = 0;
    for (let i = 0; i < nLens; i++) {
      delta[i] = now[base + i] - (prev ? prev[base + i] : 0);
      mean += delta[i];
    }
    mean /= nLens;
    let sd = 0;
    for (let i = 0; i < nLens; i++) sd += (delta[i] - mean) ** 2;
    sd = Math.sqrt(sd / nLens) || 1;
    const up = [];
    const down = [];
    for (let i = 0; i < nLens; i++) {
      const z = (delta[i] - mean) / sd;
      if (z > Z_PUSH) up.push({ id: i, act: z, delta: delta[i] });
      else if (z < -Z_PUSH) down.push({ id: i, act: -z, delta: delta[i] });
    }
    contended[pos] = up.length + down.length;
    up.sort((a, b) => b.act - a.act);
    down.sort((a, b) => b.act - a.act);
    activations += up.length + down.length;
    features.push(up);
    suppressed.push(down);
  }
  return { features, suppressed, contended, activations };
}
function measureAgreement(model, state, layer, pos, realOverLens, realTop) {
  const nLens = model.lensIds.length;
  const base = pos * nLens;
  const p = new Float32Array(nLens);
  p.set(state.lens[layer].subarray(base, base + nLens));
  softmax(p);
  let kl = 0;
  for (let i = 0; i < nLens; i++) {
    if (p[i] > 1e-12 && realOverLens[i] > 1e-12) kl += p[i] * Math.log(p[i] / realOverLens[i]);
  }
  const top = Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, 10);
  let hit = 0;
  for (const i of top) if (realTop.has(i)) hit++;
  return { agreement: hit / 10, kl };
}
function buildTrace(model, state, logits, active) {
  const { nLayer, nHead } = model.cfg;
  const T = state.length;
  const layers = [];
  let activations = 0;
  const nLens = model.lensIds.length;
  const realOverLens = new Float32Array(nLens);
  for (let i = 0; i < nLens; i++) realOverLens[i] = logits[model.lensIds[i]];
  softmax(realOverLens);
  const realTop = new Set(
    Array.from(realOverLens.keys()).sort((a, b) => realOverLens[b] - realOverLens[a]).slice(0, 10)
  );
  for (let l = 0; l < nLayer; l++) {
    const { features, suppressed, contended, activations: n } = readLayer(model, state, l, T);
    activations += n;
    const attn = new Float32Array(nHead * T * T);
    for (let h = 0; h < nHead; h++) {
      for (let i = 0; i < T; i++) {
        const src = h * state.maxT * state.maxT + i * state.maxT;
        attn.set(state.attn[l].subarray(src, src + T), h * T * T + i * T);
      }
    }
    const { agreement, kl } = measureAgreement(model, state, l, active, realOverLens, realTop);
    layers.push({
      layer: l,
      agreement,
      kl,
      attn,
      features,
      suppressed,
      contended,
      residualNorm: new Float32Array(state.residualNorm[l].subarray(0, T)),
      writeNorm: new Float32Array(state.writeNorm[l].subarray(0, T)),
      proj: new Float32Array(state.proj[l].subarray(0, T * 3))
    });
  }
  const probs = new Float32Array(logits.length);
  probs.set(logits);
  softmax(probs);
  let entropy = 0;
  for (let v = 0; v < probs.length; v++) if (probs[v] > 0) entropy -= probs[v] * Math.log2(probs[v]);
  const order = Array.from({ length: probs.length }, (_, i) => i).sort((a, b) => probs[b] - probs[a]).slice(0, 40);
  const candidates = order.map((id) => ({ id, word: model.tok.piece(id), prob: probs[id] }));
  const lensIndex = /* @__PURE__ */ new Map();
  model.lensIds.forEach((tokenId, i) => lensIndex.set(tokenId, i));
  const curves = [];
  for (const c of candidates.slice(0, 6)) {
    const idx = lensIndex.get(c.id);
    if (idx === void 0) continue;
    const values = new Float32Array(nLayer);
    for (let l = 0; l < nLayer; l++) {
      const mu = state.lensMean[l][active];
      const sd = state.lensSd[l][active] || 1;
      values[l] = (state.lens[l][active * nLens + idx] - mu) / sd;
    }
    curves.push({ id: c.id, word: c.word, values });
  }
  return {
    ids: state.ids.slice(),
    active,
    layers,
    candidates,
    curves,
    chosen: order[0],
    entropy,
    activations,
    // Every layer's write changes the score of every token in the vocabulary,
    // not only the few thousand drawn. That is the real count.
    scoreUpdates: model.cfg.vocabSize * nLayer
  };
}
function generate(model, prompt, nSteps, opts = {}) {
  const rand = mulberry32(opts.seed ?? 7);
  const temperature = opts.temperature ?? 0;
  const topP = opts.topP ?? 0.9;
  const text = prompt.trim() || "The Eiffel Tower is located in the city of";
  let ids = model.tok.encode(text);
  if (ids.length === 0) ids = model.tok.encode("The Eiffel Tower is located in the city of");
  ids = ids.slice(-(MAX_CONTEXT - nSteps - 1));
  const state = model.newState(MAX_CONTEXT);
  let logits = state.push(ids[0]);
  for (let i = 1; i < ids.length; i++) logits = state.push(ids[i]);
  const steps = [];
  const emitted = [];
  let considered = 0;
  for (let s = 0; s < nSteps; s++) {
    const probs = new Float32Array(logits.length);
    probs.set(logits);
    softmax(probs);
    const order = Array.from({ length: probs.length }, (_, i) => i).sort((a, b) => probs[b] - probs[a]).slice(0, 40);
    const chosen = sampleFrom(probs, order, temperature, topP, rand);
    const trace = buildTrace(model, state, logits, state.length - 1);
    trace.chosen = chosen;
    steps.push(trace);
    considered += trace.scoreUpdates;
    emitted.push(chosen);
    if (state.length >= MAX_CONTEXT) break;
    logits = state.push(chosen);
  }
  return {
    steps,
    emitted,
    words: emitted.map((id) => model.tok.piece(id)),
    anticipations: findAnticipations(model, steps, emitted),
    considered,
    spoken: emitted.length,
    promptWords: ids.map((id) => model.tok.piece(id)),
    promptText: text
  };
}
function findAnticipations(model, steps, emitted) {
  const MIN_LEAD = 2;
  const byId = /* @__PURE__ */ new Map();
  model.lensIds.forEach((tokenId, i) => byId.set(tokenId, i));
  const best = /* @__PURE__ */ new Map();
  for (let s = 0; s < steps.length; s++) {
    const step = steps[s];
    for (let t2 = s + MIN_LEAD; t2 < steps.length; t2++) {
      const featureId = byId.get(emitted[t2]);
      if (featureId === void 0) continue;
      for (const layer of step.layers) {
        const hits = layer.features[step.active];
        if (!hits) continue;
        const hit = hits.find((f) => f.id === featureId);
        if (!hit) continue;
        const key = `${s}:${t2}`;
        const prev = best.get(key);
        if (!prev || hit.act > prev.z) {
          best.set(key, {
            step: s,
            layer: layer.layer,
            pos: step.active,
            featureId,
            label: model.lensPieces[featureId],
            targetStep: t2,
            targetWord: model.tok.piece(emitted[t2]),
            z: hit.act
          });
        }
      }
    }
  }
  const byStep = /* @__PURE__ */ new Map();
  for (const a of best.values()) {
    const list = byStep.get(a.step) ?? [];
    list.push(a);
    byStep.set(a.step, list);
  }
  const out = [];
  for (const list of byStep.values()) {
    list.sort((a, b) => b.z - a.z);
    out.push(...list.slice(0, 3));
  }
  return out.sort((a, b) => a.step - b.step || a.targetStep - b.targetStep);
}

// tools/countlit.ts
var DIR = "public/model/gpt2";
globalThis.fetch = async (url) => {
  const buf = readFileSync(`${DIR}/${String(url).split("/").pop()}`);
  return {
    ok: true,
    headers: { get: () => String(buf.byteLength) },
    body: null,
    json: async () => JSON.parse(buf.toString()),
    text: async () => buf.toString(),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength)
  };
};
var m = await GPT2.load(DIR, 0);
var run = generate(m, "The Eiffel Tower is located in the city of", 3, { temperature: 0 });
var t = run.steps[0];
console.log(`positions=${t.ids.length} active=${t.active}`);
var totalActive = 0;
var totalAll = 0;
var peak = 0;
for (const l of t.layers) {
  const up = l.features[t.active].length, dn = l.suppressed[t.active].length;
  totalActive += up + dn;
  for (const hits of l.features) totalAll += hits.length;
  for (const h of l.features[t.active]) peak = Math.max(peak, h.act);
  console.log(`  L${String(l.layer).padStart(2)}  active pushes ${String(up).padStart(5)} up / ${String(dn).padStart(5)} down`);
}
console.log(`active-position points across all layers: ${totalActive}`);
console.log(`all-position points across all layers:    ${totalAll}`);
console.log(`peak |z| at active position: ${peak.toFixed(1)}`);

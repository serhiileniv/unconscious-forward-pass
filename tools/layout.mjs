// tools/layout.ts
import { writeFileSync } from "node:fs";
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
    for (const t of added) {
      this.encoder.set(t.content, t.id);
      this.special.add(t.id);
    }
    let maxId = 0;
    for (const id of this.encoder.values()) if (id > maxId) maxId = id;
    this.vocabSize = maxId + 1;
    this.decoder = new Array(this.vocabSize);
    for (const [tok, id] of this.encoder) this.decoder[id] = tok;
    for (const [b, c2] of this.byteEncoder) this.byteDecoder.set(c2, b);
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
    const out2 = [];
    for (const [chunk] of text.matchAll(this.pattern)) {
      const hit = this.cache.get(chunk);
      if (hit) {
        out2.push(...hit);
        continue;
      }
      let mapped = "";
      for (const byte of new TextEncoder().encode(chunk)) mapped += this.byteEncoder.get(byte);
      const ids = this.bpe(mapped).map((piece) => this.encoder.get(piece) ?? 0);
      this.cache.set(chunk, ids);
      out2.push(...ids);
    }
    return out2;
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
    const bytes = new Uint8Array(Array.from(joined).map((c2) => this.byteDecoder.get(c2) ?? 0));
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
    let t = a;
    t = Math.imul(t ^ t >>> 15, t | 1);
    t ^= t + Math.imul(t ^ t >>> 7, t | 61);
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
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
function layerNormAffine(v, off, d, gamma, beta, eps, out2, outOff) {
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
  for (let i = 0; i < d; i++) out2[outOff + i] = (v[off + i] - mean) * inv * gamma[i] + beta[i];
}
function pca3(rows, n, d, iterations = 24) {
  const mean = new Float32Array(d);
  for (let i = 0; i < n; i++) for (let k = 0; k < d; k++) mean[k] += rows[i * d + k] / n;
  const rand2 = mulberry32(1);
  const z = new Float32Array(d * 3);
  for (let i = 0; i < z.length; i++) z[i] = rand2() * 2 - 1;
  orthonormalize(z, d, 3);
  const t = new Float32Array(n * 3);
  for (let it = 0; it < iterations; it++) {
    t.fill(0);
    for (let i = 0; i < n; i++) {
      const off = i * d;
      for (let k = 0; k < d; k++) {
        const v = rows[off + k] - mean[k];
        if (v === 0) continue;
        const zo = k * 3;
        t[i * 3] += v * z[zo];
        t[i * 3 + 1] += v * z[zo + 1];
        t[i * 3 + 2] += v * z[zo + 2];
      }
    }
    z.fill(0);
    for (let i = 0; i < n; i++) {
      const off = i * d;
      const a = t[i * 3];
      const b = t[i * 3 + 1];
      const c2 = t[i * 3 + 2];
      for (let k = 0; k < d; k++) {
        const v = rows[off + k] - mean[k];
        if (v === 0) continue;
        const zo = k * 3;
        z[zo] += v * a;
        z[zo + 1] += v * b;
        z[zo + 2] += v * c2;
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
  const rand2 = mulberry32(99);
  const hi = [];
  const lo = [];
  for (let s = 0; s < samples; s++) {
    const i = Math.floor(rand2() * n);
    const j = Math.floor(rand2() * n);
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
  const pos2 = new Float32Array(buf.slice(off, off + n * 3 * 4));
  off += n * 3 * 4;
  const nbr2 = new Int32Array(buf.slice(off, off + n * k * 4));
  off += n * k * 4;
  const sim = new Float32Array(buf.slice(off, off + n * k * 4));
  let preservation = 0;
  try {
    const meta = await fetch(`${baseUrl}/layout.json`).then((r) => r.ok ? r.json() : null);
    preservation = meta?.preservation ?? 0;
  } catch {
    preservation = 0;
  }
  return { n, k, pos: pos2, nbr: nbr2, sim, preservation };
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
    let out2;
    switch (info.dtype) {
      case "F32":
        out2 = byteStart % 4 === 0 ? new Float32Array(this.buffer, byteStart, count) : new Float32Array(this.buffer.slice(byteStart, this.base + end));
        break;
      case "F16": {
        const src = new Uint16Array(this.buffer.slice(byteStart, this.base + end));
        out2 = new Float32Array(count);
        for (let i = 0; i < count; i++) out2[i] = halfToFloat(src[i]);
        break;
      }
      case "BF16": {
        const src = new Uint16Array(this.buffer.slice(byteStart, this.base + end));
        out2 = new Float32Array(count);
        const u32 = new Uint32Array(1);
        const f32 = new Float32Array(u32.buffer);
        for (let i = 0; i < count; i++) {
          u32[0] = src[i] << 16;
          out2[i] = f32[0];
        }
        break;
      }
      default:
        throw new Error(`unsupported dtype ${info.dtype} for "${name}"`);
    }
    this.cache.set(name, out2);
    return out2;
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
  const out2 = new Uint8Array(loaded);
  let at = 0;
  for (const c2 of chunks) {
    out2.set(c2, at);
    at += c2.byteLength;
  }
  return out2.buffer;
}

// src/model/gpt2.ts
function vecmat(x, xOff, w, b, n, p, out2, outOff = 0) {
  if (b) out2.set(b.subarray(0, p), outOff);
  else out2.fill(0, outOff, outOff + p);
  for (let k = 0; k < n; k++) {
    const xv = x[xOff + k];
    if (xv === 0) continue;
    const wOff = k * p;
    for (let j = 0; j < p; j++) out2[outOff + j] += xv * w[wOff + j];
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
      for (let c2 = 0; c2 < 3; c2++) {
        let s = 0;
        for (let k = 0; k < d; k++) s += (this.lensMatrix[i * d + k] - mean[k]) * basis[k * 3 + c2];
        this.lensPos[i * 3 + c2] = s;
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
    const pos2 = this.t;
    if (pos2 >= this.maxT) throw new Error("context full");
    this.ids.push(tokenId);
    const x = new Float32Array(nEmbd);
    for (let i = 0; i < nEmbd; i++) x[i] = m2.wte[tokenId * nEmbd + i] + m2.wpe[pos2 * nEmbd + i];
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
      this.kCache[l].set(qkv.subarray(nEmbd, 2 * nEmbd), pos2 * nEmbd);
      this.vCache[l].set(qkv.subarray(2 * nEmbd, 3 * nEmbd), pos2 * nEmbd);
      ctx.fill(0);
      const row = new Float32Array(pos2 + 1);
      for (let hd = 0; hd < nHead; hd++) {
        const off = hd * dHead;
        for (let j = 0; j <= pos2; j++) {
          row[j] = dot(qkv, off, this.kCache[l], j * nEmbd + off, dHead) * scale;
        }
        softmax(row, 0, pos2 + 1);
        const dst = hd * this.maxT * this.maxT + pos2 * this.maxT;
        for (let j = 0; j <= pos2; j++) {
          this.attn[l][dst + j] = row[j];
          const w = row[j];
          if (w < 1e-6) continue;
          for (let c2 = 0; c2 < dHead; c2++) ctx[off + c2] += w * this.vCache[l][j * nEmbd + off + c2];
        }
      }
      vecmat(ctx, 0, b.projW, b.projB, nEmbd, nEmbd, tmp);
      for (let i = 0; i < nEmbd; i++) x[i] += tmp[i];
      layerNormAffine(x, 0, nEmbd, b.ln2g, b.ln2b, eps, h, 0);
      vecmat(h, 0, b.fcW, b.fcB, nEmbd, 4 * nEmbd, ff);
      for (let i = 0; i < ff.length; i++) ff[i] = gelu(ff[i]);
      vecmat(ff, 0, b.mlpProjW, b.mlpProjB, 4 * nEmbd, nEmbd, tmp);
      for (let i = 0; i < nEmbd; i++) x[i] += tmp[i];
      this.hidden[l].set(x, pos2 * nEmbd);
      let rn = 0;
      let wn = 0;
      for (let i = 0; i < nEmbd; i++) {
        rn += x[i] * x[i];
        wn += (x[i] - before[i]) ** 2;
      }
      this.residualNorm[l][pos2] = Math.sqrt(rn);
      this.writeNorm[l][pos2] = Math.sqrt(wn);
      for (let c2 = 0; c2 < 3; c2++) {
        let s = 0;
        for (let i = 0; i < nEmbd; i++) s += x[i] * m2.projector[i * 3 + c2];
        this.proj[l][pos2 * 3 + c2] = s;
      }
      this.lensAt(l, pos2);
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
  lensAt(layer, pos2) {
    const m2 = this.model;
    const { nEmbd, eps } = m2.cfg;
    const nLens = m2.lensIds.length;
    const h = new Float32Array(nEmbd);
    layerNormAffine(this.hidden[layer], pos2 * nEmbd, nEmbd, m2.lnFg, m2.lnFb, eps, h, 0);
    const out2 = this.lens[layer];
    const base = pos2 * nLens;
    let mean = 0;
    for (let i = 0; i < nLens; i++) {
      const s = dot(m2.lensMatrix, i * nEmbd, h, 0, nEmbd);
      out2[base + i] = s;
      mean += s;
    }
    mean /= nLens;
    let sd = 0;
    for (let i = 0; i < nLens; i++) sd += (out2[base + i] - mean) ** 2;
    sd = Math.sqrt(sd / nLens) || 1;
    this.lensMean[layer][pos2] = mean;
    this.lensSd[layer][pos2] = sd;
  }
};

// tools/layout.ts
var DIR = process.argv[2] ?? "public/model/gpt2";
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
var K = 12;
var DIMS = 64;
var t0 = Date.now();
var m = await GPT2.load(DIR, 0);
var V = m.cfg.vocabSize;
var D = m.cfg.nEmbd;
console.log(`${V} tokens, ${D} dims`);
var U = new Float32Array(V * D);
for (let v = 0; v < V; v++) {
  let n = 0;
  for (let k = 0; k < D; k++) n += m.wte[v * D + k] ** 2;
  n = Math.sqrt(n) || 1;
  for (let k = 0; k < D; k++) U[v * D + k] = m.wte[v * D + k] / n;
}
var rand = mulberry32(7);
var R = new Float32Array(D * DIMS);
for (let i = 0; i < R.length; i++) {
  let u = 0;
  while (u === 0) u = rand();
  R[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
var P = new Float32Array(V * DIMS);
for (let v = 0; v < V; v++) {
  for (let c2 = 0; c2 < DIMS; c2++) {
    let s = 0;
    for (let k = 0; k < D; k++) s += U[v * D + k] * R[k * DIMS + c2];
    P[v * DIMS + c2] = s;
  }
}
console.log(`projected to ${DIMS}d  ${((Date.now() - t0) / 1e3).toFixed(1)}s`);
var simFull = (a, b) => {
  let s = 0;
  for (let k = 0; k < D; k++) s += U[a * D + k] * U[b * D + k];
  return s;
};
var simLow = (a, b) => {
  let s = 0;
  for (let k = 0; k < DIMS; k++) s += P[a * DIMS + k] * P[b * DIMS + k];
  return s;
};
var LEAF = 48;
var TREES = 14;
var nbr = new Int32Array(V * K).fill(-1);
var nsim = new Float32Array(V * K).fill(-2);
function offer(v, c2) {
  if (c2 === v) return false;
  let worst = 0;
  for (let j = 1; j < K; j++) if (nsim[v * K + j] < nsim[v * K + worst]) worst = j;
  const s = simLow(v, c2);
  if (s <= nsim[v * K + worst]) return false;
  for (let j = 0; j < K; j++) if (nbr[v * K + j] === c2) return false;
  nbr[v * K + worst] = c2;
  nsim[v * K + worst] = s;
  return true;
}
var scratch = new Float32Array(V);
function split(idx, lo, hi) {
  const n = hi - lo;
  if (n <= LEAF) {
    for (let a = lo; a < hi; a++) for (let b = a + 1; b < hi; b++) {
      offer(idx[a], idx[b]);
      offer(idx[b], idx[a]);
    }
    return;
  }
  const p1 = idx[lo + Math.floor(rand() * n)];
  const p2 = idx[lo + Math.floor(rand() * n)];
  for (let a = lo; a < hi; a++) {
    const v = idx[a];
    let s = 0;
    for (let k = 0; k < DIMS; k++) s += P[v * DIMS + k] * (P[p1 * DIMS + k] - P[p2 * DIMS + k]);
    scratch[v] = s;
  }
  const slice = Array.from(idx.subarray(lo, hi)).sort((a, b) => scratch[a] - scratch[b]);
  idx.set(slice, lo);
  const mid = lo + (n >> 1);
  split(idx, lo, mid);
  split(idx, mid, hi);
}
for (let t = 0; t < TREES; t++) {
  const idx = new Int32Array(V);
  for (let i = 0; i < V; i++) idx[i] = i;
  split(idx, 0, V);
  console.log(`  tree ${t + 1}/${TREES}  ${((Date.now() - t0) / 1e3).toFixed(1)}s`);
}
for (let it = 0; it < 4; it++) {
  let changed = 0;
  for (let v = 0; v < V; v++) {
    for (let j = 0; j < K; j++) {
      const n1 = nbr[v * K + j];
      if (n1 < 0) continue;
      for (let l = 0; l < K; l++) {
        const c2 = nbr[n1 * K + l];
        if (c2 >= 0 && offer(v, c2)) changed++;
      }
    }
  }
  console.log(`  refine pass ${it + 1}: ${changed} improvements  ${((Date.now() - t0) / 1e3).toFixed(1)}s`);
  if (changed < V / 200) break;
}
for (let v = 0; v < V; v++) for (let j = 0; j < K; j++) {
  if (nbr[v * K + j] < 0) {
    nbr[v * K + j] = v;
    nsim[v * K + j] = -1;
    continue;
  }
  nsim[v * K + j] = simFull(v, nbr[v * K + j]);
}
for (const w of [" Monday", " seven", " Paris", " kidney"]) {
  const id = m.tok.encode(w)[0];
  const list = Array.from({ length: K }, (_, j) => j).sort((a, b) => nsim[id * K + b] - nsim[id * K + a]);
  console.log(`  ${JSON.stringify(w).padEnd(11)} -> ${list.slice(0, 6).map((j) => JSON.stringify(m.tok.piece(nbr[id * K + j]))).join(" ")}`);
}
var deg = new Float32Array(V);
var adj = Array.from({ length: V }, () => []);
var adjW = Array.from({ length: V }, () => []);
for (let v = 0; v < V; v++) {
  for (let j = 0; j < K; j++) {
    const u = nbr[v * K + j];
    if (u < 0 || u === v) continue;
    const w = Math.max(0, nsim[v * K + j]);
    if (w <= 0) continue;
    adj[v].push(u);
    adjW[v].push(w);
    adj[u].push(v);
    adjW[u].push(w);
    deg[v] += w;
    deg[u] += w;
  }
}
var invSqrtDeg = new Float32Array(V);
for (let v = 0; v < V; v++) invSqrtDeg[v] = deg[v] > 0 ? 1 / Math.sqrt(deg[v]) : 0;
var COMP = 4;
var Z = new Float32Array(V * COMP);
for (let i = 0; i < Z.length; i++) Z[i] = rand() * 2 - 1;
var Y = new Float32Array(V * COMP);
var trivial = new Float32Array(V);
{
  let n = 0;
  for (let v = 0; v < V; v++) {
    trivial[v] = Math.sqrt(deg[v]);
    n += deg[v];
  }
  n = Math.sqrt(n) || 1;
  for (let v = 0; v < V; v++) trivial[v] /= n;
}
function deflate(M) {
  for (let c2 = 0; c2 < COMP; c2++) {
    let d = 0;
    for (let v = 0; v < V; v++) d += M[v * COMP + c2] * trivial[v];
    for (let v = 0; v < V; v++) M[v * COMP + c2] -= d * trivial[v];
  }
}
function orthonormalizeCols(M) {
  for (let c2 = 0; c2 < COMP; c2++) {
    for (let p2 = 0; p2 < c2; p2++) {
      let d = 0;
      for (let v = 0; v < V; v++) d += M[v * COMP + c2] * M[v * COMP + p2];
      for (let v = 0; v < V; v++) M[v * COMP + c2] -= d * M[v * COMP + p2];
    }
    let n = 0;
    for (let v = 0; v < V; v++) n += M[v * COMP + c2] ** 2;
    n = Math.sqrt(n) || 1;
    for (let v = 0; v < V; v++) M[v * COMP + c2] /= n;
  }
}
deflate(Z);
orthonormalizeCols(Z);
for (let it = 0; it < 220; it++) {
  Y.fill(0);
  for (let v = 0; v < V; v++) {
    const list = adj[v];
    const wts = adjW[v];
    const sv = invSqrtDeg[v];
    for (let a = 0; a < list.length; a++) {
      const u = list[a];
      const w = wts[a] * sv * invSqrtDeg[u];
      for (let c2 = 0; c2 < COMP; c2++) Y[v * COMP + c2] += w * Z[u * COMP + c2];
    }
  }
  Z.set(Y);
  deflate(Z);
  orthonormalizeCols(Z);
  if (it % 60 === 0) console.log(`  spectral iter ${it}  ${((Date.now() - t0) / 1e3).toFixed(1)}s`);
}
var pos = new Float32Array(V * 3);
for (let v = 0; v < V; v++) for (let c2 = 0; c2 < 3; c2++) pos[v * 3 + c2] = Z[v * COMP + c2 + 1];
{
  let rms0 = 0;
  for (let i = 0; i < pos.length; i++) rms0 += pos[i] * pos[i];
  rms0 = Math.sqrt(rms0 / V) || 1;
  for (let i = 0; i < pos.length; i++) pos[i] = pos[i] / rms0 * 6;
}
var A = 1.577;
var B = 0.895;
var clamp = (x) => x > 1.5 ? 1.5 : x < -1.5 ? -1.5 : x;
var REFINE = Number(process.env.REFINE ?? 200);
var REP = Number(process.env.REP ?? 0.4);
for (let e = 0; e < REFINE; e++) {
  const alpha = 0.25 * (1 - e / REFINE);
  for (let v = 0; v < V; v++) {
    for (let j = 0; j < K; j++) {
      const u = nbr[v * K + j];
      if (u < 0 || u === v || nsim[v * K + j] < 0.15) continue;
      let dx = pos[v * 3] - pos[u * 3];
      let dy = pos[v * 3 + 1] - pos[u * 3 + 1];
      let dz = pos[v * 3 + 2] - pos[u * 3 + 2];
      let d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < 1e-4) d2 = 1e-4;
      const g = -2 * A * B * Math.pow(d2, B - 1) / (1 + A * Math.pow(d2, B));
      const gx = clamp(g * dx) * alpha;
      const gy = clamp(g * dy) * alpha;
      const gz = clamp(g * dz) * alpha;
      pos[v * 3] += gx;
      pos[v * 3 + 1] += gy;
      pos[v * 3 + 2] += gz;
      pos[u * 3] -= gx;
      pos[u * 3 + 1] -= gy;
      pos[u * 3 + 2] -= gz;
      const r = Math.floor(rand() * V);
      if (r === v) continue;
      dx = pos[v * 3] - pos[r * 3];
      dy = pos[v * 3 + 1] - pos[r * 3 + 1];
      dz = pos[v * 3 + 2] - pos[r * 3 + 2];
      d2 = dx * dx + dy * dy + dz * dz;
      if (d2 < 1e-4) d2 = 1e-4;
      const rg = REP * 2 * B / ((1e-3 + d2) * (1 + A * Math.pow(d2, B)));
      pos[v * 3] += clamp(rg * dx) * alpha;
      pos[v * 3 + 1] += clamp(rg * dy) * alpha;
      pos[v * 3 + 2] += clamp(rg * dz) * alpha;
    }
  }
  if (e % 40 === 0) console.log(`  refine epoch ${e}/${REFINE}  ${((Date.now() - t0) / 1e3).toFixed(1)}s`);
}
var c = [0, 0, 0];
for (let v = 0; v < V; v++) for (let k = 0; k < 3; k++) c[k] += pos[v * 3 + k] / V;
var rms = 0;
for (let v = 0; v < V; v++) for (let k = 0; k < 3; k++) {
  pos[v * 3 + k] -= c[k];
  rms += pos[v * 3 + k] ** 2;
}
rms = Math.sqrt(rms / V) || 1;
for (let i = 0; i < pos.length; i++) pos[i] /= rms;
var header = new Int32Array([V, K]);
var out = Buffer.concat([
  Buffer.from(header.buffer),
  Buffer.from(pos.buffer),
  Buffer.from(nbr.buffer),
  Buffer.from(nsim.buffer)
]);
writeFileSync(`${DIR}/layout.bin`, out);
console.log(`wrote ${DIR}/layout.bin  ${(out.length / 1048576).toFixed(1)} MB  ${((Date.now() - t0) / 1e3).toFixed(0)}s`);

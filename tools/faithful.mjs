// tools/faithful.ts
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
var PATTERN = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;
var BPETokenizer = class {
  vocabSize;
  encoder;
  decoder;
  ranks;
  byteEncoder = bytesToUnicode();
  byteDecoder = /* @__PURE__ */ new Map();
  cache = /* @__PURE__ */ new Map();
  constructor(vocabJson, mergesTxt) {
    this.encoder = new Map(Object.entries(vocabJson));
    this.vocabSize = this.encoder.size;
    this.decoder = new Array(this.vocabSize);
    for (const [tok, id] of this.encoder) this.decoder[id] = tok;
    for (const [b, c] of this.byteEncoder) this.byteDecoder.set(c, b);
    this.ranks = /* @__PURE__ */ new Map();
    const lines = mergesTxt.split("\n");
    let rank = 0;
    for (const line of lines) {
      if (!line || line.startsWith("#")) continue;
      this.ranks.set(line.trim(), rank++);
    }
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
    for (const [chunk] of text.matchAll(PATTERN)) {
      const hit = this.cache.get(chunk);
      if (hit) {
        out.push(...hit);
        continue;
      }
      let mapped = "";
      for (const byte of new TextEncoder().encode(chunk)) mapped += this.byteEncoder.get(byte);
      const ids2 = this.bpe(mapped).map((piece) => this.encoder.get(piece) ?? 0);
      this.cache.set(chunk, ids2);
      out.push(...ids2);
    }
    return out;
  }
  /** The printable form of one token, with its leading space made visible. */
  piece(id) {
    return this.decodeTokens([id]);
  }
  decode(ids2) {
    return this.decodeTokens(ids2);
  }
  decodeTokens(ids2) {
    let joined = "";
    for (const id of ids2) joined += this.decoder[id] ?? "";
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
function layerNormAffine(v, off, d, gamma, beta, eps2, out, outOff) {
  let mean = 0;
  for (let i = 0; i < d; i++) mean += v[off + i];
  mean /= d;
  let varr = 0;
  for (let i = 0; i < d; i++) {
    const x = v[off + i] - mean;
    varr += x * x;
  }
  varr /= d;
  const inv = 1 / Math.sqrt(varr + eps2);
  for (let i = 0; i < d; i++) out[outOff + i] = (v[off + i] - mean) * inv * gamma[i] + beta[i];
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
    const ids2 = [];
    const pieces = [];
    for (let id = 0; id < cfg.vocabSize && ids2.length < nLens; id++) {
      const piece = tok.piece(id);
      if (/^ [A-Za-z]{2,}$/.test(piece)) {
        ids2.push(id);
        pieces.push(piece);
      }
    }
    this.lensIds = ids2;
    this.lensPieces = pieces;
    const d = cfg.nEmbd;
    this.lensMatrix = new Float32Array(ids2.length * d);
    for (let i = 0; i < ids2.length; i++) {
      this.lensMatrix.set(this.wte.subarray(ids2[i] * d, ids2[i] * d + d), i * d);
    }
    const rand = mulberry32(seed);
    const proj = new Float32Array(d * 3);
    for (let i = 0; i < proj.length; i++) {
      let u = 0;
      while (u === 0) u = rand();
      proj[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()) / Math.sqrt(d);
    }
    this.projector = proj;
    this.lensPos = new Float32Array(ids2.length * 3);
    for (let i = 0; i < ids2.length; i++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let k = 0; k < d; k++) s += this.lensMatrix[i * d + k] * proj[k * 3 + c];
        this.lensPos[i * 3 + c] = s;
      }
    }
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
    return new _GPT2(cfg, tok, new Safetensors(buf), nLens, 20260903);
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
  /** Per layer: maxT × nLens standardised logit-lens scores. */
  lens;
  /** Per layer: how much the block wrote into the stream, per position. */
  writeNorm;
  residualNorm;
  proj;
  ids = [];
  t = 0;
  constructor(model2, maxT) {
    this.model = model2;
    this.maxT = maxT;
    const { nLayer, nEmbd: nEmbd2, nHead } = model2.cfg;
    const nLens = model2.lensIds.length;
    this.kCache = [];
    this.vCache = [];
    this.hidden = [];
    this.attn = [];
    this.lens = [];
    this.writeNorm = [];
    this.residualNorm = [];
    this.proj = [];
    for (let l = 0; l < nLayer; l++) {
      this.kCache.push(new Float32Array(maxT * nEmbd2));
      this.vCache.push(new Float32Array(maxT * nEmbd2));
      this.hidden.push(new Float32Array(maxT * nEmbd2));
      this.attn.push(new Float32Array(nHead * maxT * maxT));
      this.lens.push(new Float32Array(maxT * nLens));
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
    const m = this.model;
    const { nLayer, nEmbd: nEmbd2, nHead, eps: eps2 } = m.cfg;
    const dHead = nEmbd2 / nHead;
    const pos2 = this.t;
    if (pos2 >= this.maxT) throw new Error("context full");
    this.ids.push(tokenId);
    const x = new Float32Array(nEmbd2);
    for (let i = 0; i < nEmbd2; i++) x[i] = m.wte[tokenId * nEmbd2 + i] + m.wpe[pos2 * nEmbd2 + i];
    const h = new Float32Array(nEmbd2);
    const qkv = new Float32Array(3 * nEmbd2);
    const ctx = new Float32Array(nEmbd2);
    const ff = new Float32Array(4 * nEmbd2);
    const tmp = new Float32Array(nEmbd2);
    const scale = 1 / Math.sqrt(dHead);
    for (let l = 0; l < nLayer; l++) {
      const b = m.blocks[l];
      const before = x.slice();
      layerNormAffine(x, 0, nEmbd2, b.ln1g, b.ln1b, eps2, h, 0);
      vecmat(h, 0, b.attnW, b.attnB, nEmbd2, 3 * nEmbd2, qkv);
      this.kCache[l].set(qkv.subarray(nEmbd2, 2 * nEmbd2), pos2 * nEmbd2);
      this.vCache[l].set(qkv.subarray(2 * nEmbd2, 3 * nEmbd2), pos2 * nEmbd2);
      ctx.fill(0);
      const row = new Float32Array(pos2 + 1);
      for (let hd = 0; hd < nHead; hd++) {
        const off = hd * dHead;
        for (let j = 0; j <= pos2; j++) {
          row[j] = dot(qkv, off, this.kCache[l], j * nEmbd2 + off, dHead) * scale;
        }
        softmax(row, 0, pos2 + 1);
        const dst = hd * this.maxT * this.maxT + pos2 * this.maxT;
        for (let j = 0; j <= pos2; j++) {
          this.attn[l][dst + j] = row[j];
          const w = row[j];
          if (w < 1e-6) continue;
          for (let c = 0; c < dHead; c++) ctx[off + c] += w * this.vCache[l][j * nEmbd2 + off + c];
        }
      }
      vecmat(ctx, 0, b.projW, b.projB, nEmbd2, nEmbd2, tmp);
      for (let i = 0; i < nEmbd2; i++) x[i] += tmp[i];
      layerNormAffine(x, 0, nEmbd2, b.ln2g, b.ln2b, eps2, h, 0);
      vecmat(h, 0, b.fcW, b.fcB, nEmbd2, 4 * nEmbd2, ff);
      for (let i = 0; i < ff.length; i++) ff[i] = gelu(ff[i]);
      vecmat(ff, 0, b.mlpProjW, b.mlpProjB, 4 * nEmbd2, nEmbd2, tmp);
      for (let i = 0; i < nEmbd2; i++) x[i] += tmp[i];
      this.hidden[l].set(x, pos2 * nEmbd2);
      let rn = 0;
      let wn = 0;
      for (let i = 0; i < nEmbd2; i++) {
        rn += x[i] * x[i];
        wn += (x[i] - before[i]) ** 2;
      }
      this.residualNorm[l][pos2] = Math.sqrt(rn);
      this.writeNorm[l][pos2] = Math.sqrt(wn);
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let i = 0; i < nEmbd2; i++) s += x[i] * m.projector[i * 3 + c];
        this.proj[l][pos2 * 3 + c] = s;
      }
      this.lensAt(l, pos2);
    }
    layerNormAffine(x, 0, nEmbd2, m.lnFg, m.lnFb, eps2, h, 0);
    const logits = new Float32Array(m.cfg.vocabSize);
    for (let v = 0; v < m.cfg.vocabSize; v++) logits[v] = dot(m.wte, v * nEmbd2, h, 0, nEmbd2);
    this.t++;
    return logits;
  }
  /**
   * The logit lens at one layer.
   *
   * Take the residual as it stands part-way up the stack, apply the model's own
   * final layer norm, and read it against the real unembedding. That is what the
   * model would say if it had to answer from this depth. Scores are standardised
   * per layer so depths are comparable — raw logits grow steadily with depth and
   * would otherwise make late layers look uniformly louder.
   */
  lensAt(layer, pos2) {
    const m = this.model;
    const { nEmbd: nEmbd2, eps: eps2 } = m.cfg;
    const nLens = m.lensIds.length;
    const h = new Float32Array(nEmbd2);
    layerNormAffine(this.hidden[layer], pos2 * nEmbd2, nEmbd2, m.lnFg, m.lnFb, eps2, h, 0);
    const out = this.lens[layer];
    const base = pos2 * nLens;
    let mean = 0;
    for (let i = 0; i < nLens; i++) {
      const s = dot(m.lensMatrix, i * nEmbd2, h, 0, nEmbd2);
      out[base + i] = s;
      mean += s;
    }
    mean /= nLens;
    let sd = 0;
    for (let i = 0; i < nLens; i++) sd += (out[base + i] - mean) ** 2;
    sd = Math.sqrt(sd / nLens) || 1;
    for (let i = 0; i < nLens; i++) out[base + i] = (out[base + i] - mean) / sd;
  }
};

// tools/faithful.ts
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
var model = await GPT2.load(DIR, 6144);
var prompt = process.argv[2] ?? "The Eiffel Tower is located in the city of";
var ids = model.tok.encode(prompt);
var state = model.newState(32);
var real = state.push(ids[0]);
for (let i = 1; i < ids.length; i++) real = state.push(ids[i]);
var { nEmbd, eps, vocabSize } = model.cfg;
var pos = state.length - 1;
function lensLogits(layer) {
  const h = new Float32Array(nEmbd);
  layerNormAffine(state.hidden[layer], pos * nEmbd, nEmbd, model.lnFg, model.lnFb, eps, h, 0);
  const out = new Float32Array(vocabSize);
  for (let v = 0; v < vocabSize; v++) out[v] = dot(model.wte, v * nEmbd, h, 0, nEmbd);
  return out;
}
var realProbs = new Float32Array(real.length);
realProbs.set(real);
softmax(realProbs);
var realTop = Array.from(realProbs.keys()).sort((a, b) => realProbs[b] - realProbs[a]);
var realTop10 = new Set(realTop.slice(0, 10));
console.log(`prompt: ${JSON.stringify(prompt)}`);
console.log(`model's actual next token: ${JSON.stringify(model.tok.piece(realTop[0]))} at ${(realProbs[realTop[0]] * 100).toFixed(1)}%
`);
console.log("layer  top-1 agrees  overlap@10   KL(lens||real)   lens top-3");
for (let l = 0; l < model.cfg.nLayer; l++) {
  const lp = lensLogits(l);
  const probs = new Float32Array(lp.length);
  probs.set(lp);
  softmax(probs);
  const top = Array.from(probs.keys()).sort((a, b) => probs[b] - probs[a]);
  let kl = 0;
  for (let v = 0; v < vocabSize; v++) {
    if (probs[v] > 1e-12 && realProbs[v] > 1e-12) kl += probs[v] * Math.log(probs[v] / realProbs[v]);
  }
  const overlap = top.slice(0, 10).filter((t) => realTop10.has(t)).length;
  const agree = top[0] === realTop[0] ? "yes" : "no ";
  const t3 = top.slice(0, 3).map((t) => JSON.stringify(model.tok.piece(t))).join(" ");
  console.log(
    `  ${String(l).padStart(2)}      ${agree}          ${overlap}/10        ${kl.toFixed(3).padStart(7)}      ${t3}`
  );
}

// tools/geometry.ts
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
var MODERN_PATTERN = /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;
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
async function loadTokenizerJson(baseUrl) {
  const json = await fetch(`${baseUrl}/tokenizer.json`).then((r) => r.json());
  return new BPETokenizer(json.model.vocab, json.model.merges, MODERN_PATTERN, json.added_tokens ?? []);
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

// src/model/llama.ts
function matvecT(w, x, b, outDim, inDim, out) {
  for (let j = 0; j < outDim; j++) out[j] = (b ? b[j] : 0) + dot(w, j * inDim, x, 0, inDim);
}
function rmsNorm(x, g, n, eps, out) {
  let ss = 0;
  for (let i = 0; i < n; i++) ss += x[i] * x[i];
  const inv = 1 / Math.sqrt(ss / n + eps);
  for (let i = 0; i < n; i++) out[i] = x[i] * inv * g[i];
}
function rope(vec, off, headDim, pos, theta) {
  const half = headDim / 2;
  for (let i = 0; i < half; i++) {
    const freq = Math.pow(theta, -2 * i / headDim);
    const c = Math.cos(pos * freq);
    const s = Math.sin(pos * freq);
    const a = vec[off + i];
    const b = vec[off + i + half];
    vec[off + i] = a * c - b * s;
    vec[off + i + half] = b * c + a * s;
  }
}
var silu = (x) => x / (1 + Math.exp(-x));
var LlamaLM = class _LlamaLM {
  cfg;
  tok;
  lensIds;
  lensPieces;
  lensMatrix;
  lensPos;
  projector;
  shape;
  layers;
  embed;
  finalNorm;
  head;
  constructor(shape, name, tok, st, nLens, seed) {
    this.shape = shape;
    this.tok = tok;
    this.cfg = { nLayer: shape.nLayer, nEmbd: shape.nEmbd, nHead: shape.nHead, vocabSize: shape.vocabSize, name };
    this.embed = st.get("model.embed_tokens.weight");
    this.finalNorm = st.get("model.norm.weight");
    this.head = st.has("lm_head.weight") ? st.get("lm_head.weight") : this.embed;
    this.layers = [];
    for (let l = 0; l < shape.nLayer; l++) {
      const p = `model.layers.${l}`;
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
        down: st.get(`${p}.mlp.down_proj.weight`)
      });
    }
    const ids = [];
    const pieces = [];
    for (let id = 0; id < shape.vocabSize && ids.length < nLens; id++) {
      const piece = tok.piece(id);
      if (/^ [A-Za-z]{2,}$/.test(piece)) {
        ids.push(id);
        pieces.push(piece);
      }
    }
    this.lensIds = ids;
    this.lensPieces = pieces;
    const d = shape.nEmbd;
    this.lensMatrix = new Float32Array(ids.length * d);
    for (let i = 0; i < ids.length; i++) {
      this.lensMatrix.set(this.head.subarray(ids[i] * d, ids[i] * d + d), i * d);
    }
    const rand2 = mulberry32(seed);
    const proj = new Float32Array(d * 3);
    for (let i = 0; i < proj.length; i++) {
      let u = 0;
      while (u === 0) u = rand2();
      proj[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand2()) / Math.sqrt(d);
    }
    this.projector = proj;
    this.lensPos = new Float32Array(ids.length * 3);
    for (let i = 0; i < ids.length; i++) {
      for (let c = 0; c < 3; c++) {
        let s = 0;
        for (let k = 0; k < d; k++) s += this.lensMatrix[i * d + k] * proj[k * 3 + c];
        this.lensPos[i * 3 + c] = s;
      }
    }
  }
  static async load(baseUrl, name, nLens, onProgress) {
    onProgress?.("tokenizer", 0);
    const tok = await loadTokenizerJson(baseUrl);
    const raw = await fetch(`${baseUrl}/config.json`).then((r) => r.json());
    const nEmbd = raw.hidden_size;
    const nHead = raw.num_attention_heads;
    const shape = {
      nLayer: raw.num_hidden_layers,
      nEmbd,
      nHead,
      nKvHead: raw.num_key_value_heads ?? nHead,
      headDim: raw.head_dim ?? nEmbd / nHead,
      ffDim: raw.intermediate_size,
      vocabSize: raw.vocab_size,
      eps: raw.rms_norm_eps ?? 1e-6,
      ropeTheta: raw.rope_theta ?? 1e4
    };
    const buf = await fetchWithProgress(
      `${baseUrl}/model.safetensors`,
      (loaded, total) => onProgress?.("weights", loaded / total)
    );
    onProgress?.("unpacking", 1);
    return new _LlamaLM(shape, name, tok, new Safetensors(buf), nLens, 20260903);
  }
  newState(maxT) {
    return new LlamaState(this, maxT);
  }
};
var LlamaState = class {
  maxT;
  ids = [];
  attn = [];
  lens = [];
  lensMean = [];
  lensSd = [];
  writeNorm = [];
  residualNorm = [];
  proj = [];
  hidden = [];
  kCache = [];
  vCache = [];
  model;
  t = 0;
  constructor(model, maxT) {
    this.model = model;
    this.maxT = maxT;
    const { nLayer, nEmbd, nHead, nKvHead, headDim } = model.shape;
    const nLens = model.lensIds.length;
    const kvDim = nKvHead * headDim;
    for (let l = 0; l < nLayer; l++) {
      this.kCache.push(new Float32Array(maxT * kvDim));
      this.vCache.push(new Float32Array(maxT * kvDim));
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
  push(tokenId) {
    const m2 = this.model;
    const { nLayer, nEmbd, nHead, nKvHead, headDim, ffDim, eps, ropeTheta, vocabSize } = m2.shape;
    const pos = this.t;
    if (pos >= this.maxT) throw new Error("context full");
    this.ids.push(tokenId);
    const qDim = nHead * headDim;
    const kvDim = nKvHead * headDim;
    const group = nHead / nKvHead;
    const scale = 1 / Math.sqrt(headDim);
    const x = new Float32Array(nEmbd);
    x.set(m2.embed.subarray(tokenId * nEmbd, (tokenId + 1) * nEmbd));
    const h = new Float32Array(nEmbd);
    const q = new Float32Array(qDim);
    const kv = new Float32Array(kvDim);
    const vv = new Float32Array(kvDim);
    const ctx = new Float32Array(qDim);
    const tmp = new Float32Array(nEmbd);
    const gate = new Float32Array(ffDim);
    const up = new Float32Array(ffDim);
    const row = new Float32Array(this.maxT);
    for (let l = 0; l < nLayer; l++) {
      const b = m2.layers[l];
      const before = x.slice();
      rmsNorm(x, b.inNorm, nEmbd, eps, h);
      matvecT(b.q, h, b.qb, qDim, nEmbd, q);
      matvecT(b.k, h, b.kb, kvDim, nEmbd, kv);
      matvecT(b.v, h, b.vb, kvDim, nEmbd, vv);
      for (let i = 0; i < nHead; i++) rope(q, i * headDim, headDim, pos, ropeTheta);
      for (let i = 0; i < nKvHead; i++) rope(kv, i * headDim, headDim, pos, ropeTheta);
      this.kCache[l].set(kv, pos * kvDim);
      this.vCache[l].set(vv, pos * kvDim);
      ctx.fill(0);
      for (let hd = 0; hd < nHead; hd++) {
        const kvHead = Math.floor(hd / group);
        const qOff = hd * headDim;
        const kvOff = kvHead * headDim;
        for (let j = 0; j <= pos; j++) {
          row[j] = dot(q, qOff, this.kCache[l], j * kvDim + kvOff, headDim) * scale;
        }
        softmax(row, 0, pos + 1);
        const dst = hd * this.maxT * this.maxT + pos * this.maxT;
        for (let j = 0; j <= pos; j++) {
          this.attn[l][dst + j] = row[j];
          const w = row[j];
          if (w < 1e-6) continue;
          for (let c = 0; c < headDim; c++) ctx[qOff + c] += w * this.vCache[l][j * kvDim + kvOff + c];
        }
      }
      matvecT(b.o, ctx, null, nEmbd, qDim, tmp);
      for (let i = 0; i < nEmbd; i++) x[i] += tmp[i];
      rmsNorm(x, b.postNorm, nEmbd, eps, h);
      matvecT(b.gate, h, null, ffDim, nEmbd, gate);
      matvecT(b.up, h, null, ffDim, nEmbd, up);
      for (let i = 0; i < ffDim; i++) gate[i] = silu(gate[i]) * up[i];
      matvecT(b.down, gate, null, nEmbd, ffDim, tmp);
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
    }
    let ss = 0;
    for (let i = 0; i < nEmbd; i++) ss += x[i] * x[i];
    const frozen = 1 / Math.sqrt(ss / nEmbd + eps);
    for (let l = 0; l < nLayer; l++) this.attributeAt(l, pos, frozen);
    rmsNorm(x, m2.finalNorm, nEmbd, eps, h);
    const logits = new Float32Array(vocabSize);
    for (let v = 0; v < vocabSize; v++) logits[v] = dot(m2.head, v * nEmbd, h, 0, nEmbd);
    this.t++;
    return logits;
  }
  /**
   * The running total of the answer, as of this depth.
   *
   * Applies the model's real final-norm weights with the scale frozen from the
   * finished residual, then reads through the real output head. At the last
   * layer this equals the model's logits exactly; at every earlier layer it is
   * the partial sum of contributions that produced them.
   */
  attributeAt(layer, pos, frozen) {
    const m2 = this.model;
    const { nEmbd } = m2.shape;
    const nLens = m2.lensIds.length;
    const h = new Float32Array(nEmbd);
    const src = layer * 0 + pos * nEmbd;
    for (let i = 0; i < nEmbd; i++) h[i] = this.hidden[layer][src + i] * frozen * m2.finalNorm[i];
    const out = this.lens[layer];
    const base = pos * nLens;
    let mean2 = 0;
    for (let i = 0; i < nLens; i++) {
      const s = dot(m2.lensMatrix, i * nEmbd, h, 0, nEmbd);
      out[base + i] = s;
      mean2 += s;
    }
    mean2 /= nLens;
    let sd = 0;
    for (let i = 0; i < nLens; i++) sd += (out[base + i] - mean2) ** 2;
    this.lensMean[layer][pos] = mean2;
    this.lensSd[layer][pos] = Math.sqrt(sd / nLens) || 1;
  }
  hiddenAt(layer) {
    return this.hidden[layer];
  }
};

// tools/geometry.ts
var DIR = "public/model/qwen";
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
var m = await LlamaLM.load(DIR, "x", 6144);
var N = m.lensIds.length;
var D = m.shape.nEmbd;
var X = m.lensMatrix;
var mean = new Float32Array(D);
for (let i = 0; i < N; i++) for (let k = 0; k < D; k++) mean[k] += X[i * D + k] / N;
var C = new Float32Array(N * D);
for (let i = 0; i < N; i++) for (let k = 0; k < D; k++) C[i * D + k] = X[i * D + k] - mean[k];
var totalVar = 0;
for (let i = 0; i < C.length; i++) totalVar += C[i] * C[i];
function pca3() {
  const rand2 = mulberry32(1);
  const Z = new Float32Array(D * 3);
  for (let i = 0; i < Z.length; i++) Z[i] = rand2() * 2 - 1;
  orthonormalize(Z, D, 3);
  const T = new Float32Array(N * 3);
  for (let it = 0; it < 30; it++) {
    T.fill(0);
    for (let i = 0; i < N; i++)
      for (let k = 0; k < D; k++) {
        const v = C[i * D + k];
        if (v === 0) continue;
        for (let c = 0; c < 3; c++) T[i * 3 + c] += v * Z[k * 3 + c];
      }
    Z.fill(0);
    for (let i = 0; i < N; i++)
      for (let k = 0; k < D; k++) {
        const v = C[i * D + k];
        if (v === 0) continue;
        for (let c = 0; c < 3; c++) Z[k * 3 + c] += v * T[i * 3 + c];
      }
    orthonormalize(Z, D, 3);
  }
  return Z;
}
function project(P) {
  const out = new Float32Array(N * 3);
  for (let i = 0; i < N; i++)
    for (let c = 0; c < 3; c++) {
      let s = 0;
      for (let k = 0; k < D; k++) s += C[i * D + k] * P[k * 3 + c];
      out[i * 3 + c] = s;
    }
  return out;
}
function variance(Y) {
  let v = 0;
  for (let i = 0; i < Y.length; i++) v += Y[i] * Y[i];
  return v / totalVar;
}
function distanceFidelity(Y, warp) {
  const rand2 = mulberry32(99);
  const a = [];
  const b = [];
  const P = new Float32Array(N * 3);
  P.set(Y);
  if (warp) {
    for (let i = 0; i < N; i++) {
      const r = Math.hypot(P[i * 3], P[i * 3 + 1]) || 1e-9;
      const rr = Math.pow(r, 0.62);
      P[i * 3] = P[i * 3] / r * rr;
      P[i * 3 + 1] = P[i * 3 + 1] / r * rr;
    }
  }
  for (let n = 0; n < 4e3; n++) {
    const i = Math.floor(rand2() * N);
    const j = Math.floor(rand2() * N);
    if (i === j) continue;
    let d2 = 0;
    for (let k = 0; k < D; k++) d2 += (C[i * D + k] - C[j * D + k]) ** 2;
    a.push(Math.sqrt(d2));
    b.push(Math.hypot(P[i * 3] - P[j * 3], P[i * 3 + 1] - P[j * 3 + 1], P[i * 3 + 2] - P[j * 3 + 2]));
  }
  const ma = a.reduce((x, y) => x + y, 0) / a.length;
  const mb = b.reduce((x, y) => x + y, 0) / b.length;
  let num = 0, da = 0, db = 0;
  for (let i = 0; i < a.length; i++) {
    num += (a[i] - ma) * (b[i] - mb);
    da += (a[i] - ma) ** 2;
    db += (b[i] - mb) ** 2;
  }
  return num / Math.sqrt(da * db);
}
var rand = mulberry32(20260903);
var R = new Float32Array(D * 3);
for (let i = 0; i < R.length; i++) {
  let u = 0;
  while (u === 0) u = rand();
  R[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand()) / Math.sqrt(D);
}
var Yr = project(R);
var Yp = project(pca3());
console.log(`${N} tokens in ${D} dimensions, projected to 3
`);
console.log(`                            variance kept   distance fidelity (Pearson r)`);
console.log(`random projection (shipped)     ${(variance(Yr) * 100).toFixed(1)}%              ${distanceFidelity(Yr, false).toFixed(3)}`);
console.log(`  + the r^0.62 radial warp      ${(variance(Yr) * 100).toFixed(1)}%              ${distanceFidelity(Yr, true).toFixed(3)}`);
console.log(`top-3 PCA                       ${(variance(Yp) * 100).toFixed(1)}%              ${distanceFidelity(Yp, false).toFixed(3)}`);
console.log(`  + the r^0.62 radial warp      ${(variance(Yp) * 100).toFixed(1)}%              ${distanceFidelity(Yp, true).toFixed(3)}`);

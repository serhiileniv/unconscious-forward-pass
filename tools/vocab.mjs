// tools/vocab.ts
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
    for (const [tok2, id] of this.encoder) this.decoder[id] = tok2;
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

// tools/vocab.ts
var vocab = JSON.parse(readFileSync("public/model/gpt2/vocab.json", "utf8"));
var merges = readFileSync("public/model/gpt2/merges.txt", "utf8");
var tok = new BPETokenizer(vocab, merges);
var whole = [];
for (let id = 0; id < 50257; id++) if (/^ [A-Za-z]{2,}$/.test(tok.piece(id))) whole.push(id);
console.log(`whole-word tokens (leading space, 2+ letters): ${whole.length}`);
for (const n of [2048, 4096, 6144, 8192]) {
  console.log(`  first ${n} reach id ${whole[n - 1]}`);
}
for (const w of [" Paris", " France", " capital", " Republic", " river", " Berlin", " woman", " quantum"]) {
  const id = vocab[w.replace(/ /g, "\u0120")];
  const rank = whole.indexOf(id);
  console.log(`  ${JSON.stringify(w).padEnd(12)} id=${id} rank-among-whole=${rank}`);
}

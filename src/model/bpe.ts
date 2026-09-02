/**
 * GPT-2's byte-level BPE tokenizer.
 *
 * Byte-level means every possible input encodes — there is no unknown token and
 * nothing is silently dropped. "2+2" tokenizes fine here, which the previous
 * word-level tokenizer could not do.
 */

/** GPT-2's reversible byte↔character map, which keeps every byte printable. */
function bytesToUnicode(): Map<number, string> {
  const bs: number[] = []
  for (let i = 33; i <= 126; i++) bs.push(i)
  for (let i = 161; i <= 172; i++) bs.push(i)
  for (let i = 174; i <= 255; i++) bs.push(i)
  const cs = bs.slice()
  let n = 0
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b)
      cs.push(256 + n)
      n++
    }
  }
  const map = new Map<number, string>()
  for (let i = 0; i < bs.length; i++) map.set(bs[i], String.fromCodePoint(cs[i]))
  return map
}

/** GPT-2's split. Note ` ?\p{N}+`: it groups digits greedily. */
export const GPT2_PATTERN =
  /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu

/**
 * The split used by Llama-3-era tokenizers, including Qwen2.5 and SmolLM2.
 *
 * The meaningful difference from GPT-2 is `\p{N}{1,3}`, which chops digits into
 * groups of at most three instead of swallowing a whole number. Arithmetic
 * behaves differently under the two, which is visible in the token strip.
 */
export const MODERN_PATTERN =
  /(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu

export class BPETokenizer {
  readonly vocabSize: number
  private readonly encoder: Map<string, number>
  private readonly decoder: string[]
  private readonly ranks: Map<string, number>
  private readonly byteEncoder = bytesToUnicode()
  private readonly byteDecoder = new Map<string, number>()
  private readonly cache = new Map<string, number[]>()

  private readonly pattern: RegExp
  private readonly special = new Set<number>()

  constructor(
    vocabJson: Record<string, number>,
    merges: string | (string | [string, string])[],
    pattern: RegExp = GPT2_PATTERN,
    added: { id: number; content: string }[] = [],
  ) {
    this.pattern = pattern
    this.encoder = new Map(Object.entries(vocabJson))
    for (const t of added) {
      this.encoder.set(t.content, t.id)
      this.special.add(t.id)
    }
    // Spreading 150k ids into Math.max overflows the call stack.
    let maxId = 0
    for (const id of this.encoder.values()) if (id > maxId) maxId = id
    this.vocabSize = maxId + 1
    this.decoder = new Array(this.vocabSize)
    for (const [tok, id] of this.encoder) this.decoder[id] = tok
    for (const [b, c] of this.byteEncoder) this.byteDecoder.set(c, b)

    this.ranks = new Map()
    const lines = typeof merges === 'string' ? merges.split('\n') : merges
    let rank = 0
    for (const line of lines) {
      const text = Array.isArray(line) ? `${line[0]} ${line[1]}` : line
      if (!text || text.startsWith('#')) continue
      this.ranks.set(text.trim(), rank++)
    }
  }

  /** True for control tokens like <|endoftext|>, which should not be emitted. */
  isSpecial(id: number): boolean {
    return this.special.has(id)
  }

  private bpe(token: string): string[] {
    let word = Array.from(token)
    if (word.length === 1) return word
    for (;;) {
      let bestRank = Infinity
      let bestAt = -1
      for (let i = 0; i < word.length - 1; i++) {
        const r = this.ranks.get(`${word[i]} ${word[i + 1]}`)
        if (r !== undefined && r < bestRank) {
          bestRank = r
          bestAt = i
        }
      }
      if (bestAt < 0) break
      word = [...word.slice(0, bestAt), word[bestAt] + word[bestAt + 1], ...word.slice(bestAt + 2)]
      if (word.length === 1) break
    }
    return word
  }

  encode(text: string): number[] {
    const out: number[] = []
    for (const [chunk] of text.matchAll(this.pattern)) {
      const hit = this.cache.get(chunk)
      if (hit) {
        out.push(...hit)
        continue
      }
      let mapped = ''
      for (const byte of new TextEncoder().encode(chunk)) mapped += this.byteEncoder.get(byte)
      const ids = this.bpe(mapped).map((piece) => this.encoder.get(piece) ?? 0)
      this.cache.set(chunk, ids)
      out.push(...ids)
    }
    return out
  }

  /** The printable form of one token, with its leading space made visible. */
  piece(id: number): string {
    return this.decodeTokens([id])
  }

  decode(ids: number[]): string {
    return this.decodeTokens(ids)
  }

  private decodeTokens(ids: number[]): string {
    let joined = ''
    for (const id of ids) joined += this.decoder[id] ?? ''
    const bytes = new Uint8Array(Array.from(joined).map((c) => this.byteDecoder.get(c) ?? 0))
    return new TextDecoder('utf-8', { fatal: false }).decode(bytes)
  }
}

export async function loadTokenizer(baseUrl: string): Promise<BPETokenizer> {
  const [vocab, merges] = await Promise.all([
    fetch(`${baseUrl}/vocab.json`).then((r) => r.json() as Promise<Record<string, number>>),
    fetch(`${baseUrl}/merges.txt`).then((r) => r.text()),
  ])
  return new BPETokenizer(vocab, merges)
}

interface TokenizerJson {
  model: { vocab: Record<string, number>; merges: (string | [string, string])[] }
  added_tokens?: { id: number; content: string }[]
}

/** HuggingFace's single-file tokenizer format, as shipped by Qwen2.5 and SmolLM2. */
export async function loadTokenizerJson(baseUrl: string): Promise<BPETokenizer> {
  const json = (await fetch(`${baseUrl}/tokenizer.json`).then((r) => r.json())) as TokenizerJson
  return new BPETokenizer(json.model.vocab, json.model.merges, MODERN_PATTERN, json.added_tokens ?? [])
}

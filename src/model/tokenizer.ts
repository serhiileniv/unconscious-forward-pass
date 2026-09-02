import { CORPUS } from './corpus'

/** Word-level tokenizer. Small enough that the whole vocabulary is inspectable. */
export interface Vocab {
  readonly words: string[]
  readonly index: Map<string, number>
  readonly size: number
}

export function splitWords(text: string): string[] {
  return text.toLowerCase().match(/[a-z']+/g) ?? []
}

export function buildVocab(): Vocab {
  const words = ['<unk>']
  const index = new Map<string, number>([['<unk>', 0]])
  for (const w of splitWords(CORPUS)) {
    if (!index.has(w)) {
      index.set(w, words.length)
      words.push(w)
    }
  }
  return { words, index, size: words.length }
}

export function encode(vocab: Vocab, text: string): number[] {
  return splitWords(text).map((w) => vocab.index.get(w) ?? 0)
}

export function decode(vocab: Vocab, ids: number[]): string {
  return ids.map((i) => vocab.words[i] ?? '<unk>').join(' ')
}

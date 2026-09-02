import { readFileSync } from 'node:fs'
import { BPETokenizer } from '../src/model/bpe'

const vocab = JSON.parse(readFileSync('public/model/gpt2/vocab.json', 'utf8'))
const merges = readFileSync('public/model/gpt2/merges.txt', 'utf8')
const tok = new BPETokenizer(vocab, merges)

const whole: number[] = []
for (let id = 0; id < 50257; id++) if (/^ [A-Za-z]{2,}$/.test(tok.piece(id))) whole.push(id)
console.log(`whole-word tokens (leading space, 2+ letters): ${whole.length}`)
for (const n of [2048, 4096, 6144, 8192]) {
  console.log(`  first ${n} reach id ${whole[n - 1]}`)
}
for (const w of [' Paris', ' France', ' capital', ' Republic', ' river', ' Berlin', ' woman', ' quantum']) {
  const id = vocab[w.replace(/ /g, 'Ġ')]
  const rank = whole.indexOf(id)
  console.log(`  ${JSON.stringify(w).padEnd(12)} id=${id} rank-among-whole=${rank}`)
}

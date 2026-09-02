/**
 * Download GPT-2's published weights into public/ so Vite can serve them.
 *
 * These are the real files from the model repository, not a re-export: the same
 * safetensors any other loader would read. They are gitignored — 942 MB does not
 * belong in a repository, and anyone can fetch them in one command.
 */
import { createWriteStream, existsSync, mkdirSync, statSync } from 'node:fs'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'

const REPO = 'Qwen/Qwen2.5-0.5B-Instruct'
const FILES = ['config.json', 'tokenizer.json', 'model.safetensors']
const DIR = 'public/model/qwen'

mkdirSync(DIR, { recursive: true })

for (const file of FILES) {
  const dest = `${DIR}/${file}`
  const url = `https://huggingface.co/${REPO}/resolve/main/${file}`
  const res = await fetch(url, { redirect: 'follow' })
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} for ${url}`)
  const total = Number(res.headers.get('content-length') ?? 0)

  if (existsSync(dest) && total && statSync(dest).size === total) {
    console.log(`have ${file} (${(total / 1048576).toFixed(1)} MB)`)
    res.body?.cancel()
    continue
  }

  let seen = 0
  const body = Readable.fromWeb(res.body)
  body.on('data', (chunk) => {
    seen += chunk.length
    if (total > 8 << 20) {
      process.stdout.write(`\r  ${file} ${(seen / 1048576).toFixed(0)}/${(total / 1048576).toFixed(0)} MB`)
    }
  })
  await pipeline(body, createWriteStream(dest))
  process.stdout.write(`\r  ${file} ${(seen / 1048576).toFixed(1)} MB\n`)
}
console.log(`\nweights in ${DIR}`)

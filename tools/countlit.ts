import { readFileSync } from 'node:fs'
import { GPT2 } from '../src/model/gpt2'
import { generate } from '../src/model/run'
const DIR = 'public/model/gpt2'
;(globalThis as unknown as { fetch: unknown }).fetch = async (url: string) => {
  const buf = readFileSync(`${DIR}/${String(url).split('/').pop()!}`)
  return { ok: true, headers: { get: () => String(buf.byteLength) }, body: null,
    json: async () => JSON.parse(buf.toString()), text: async () => buf.toString(),
    arrayBuffer: async () => buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) }
}
const m = await GPT2.load(DIR, 0)
const run = generate(m, 'The Eiffel Tower is located in the city of', 3, { temperature: 0 })
const t = run.steps[0]
console.log(`positions=${t.ids.length} active=${t.active}`)
let totalActive = 0, totalAll = 0, peak = 0
for (const l of t.layers) {
  const up = l.features[t.active].length, dn = l.suppressed[t.active].length
  totalActive += up + dn
  for (const hits of l.features) totalAll += hits.length
  for (const h of l.features[t.active]) peak = Math.max(peak, h.act)
  console.log(`  L${String(l.layer).padStart(2)}  active pushes ${String(up).padStart(5)} up / ${String(dn).padStart(5)} down`)
}
console.log(`active-position points across all layers: ${totalActive}`)
console.log(`all-position points across all layers:    ${totalAll}`)
console.log(`peak |z| at active position: ${peak.toFixed(1)}`)

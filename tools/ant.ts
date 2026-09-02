import { buildSession, generate } from '../src/model/run'
const s = buildSession()
for (const p of ['most of what a mind does is never', 'the model speaks one token and the field', 'attention is a reaching backward']) {
  const r = generate(s, p, 10, { temperature: 0.85, seed: 7 })
  console.log(`"${p}" -> ${r.words.join(' ')}`)
  console.log(`   anticipations: ${r.anticipations.length}`,
    r.anticipations.map((a) => `${a.label}@s${a.step}→${a.targetWord}@s${a.targetStep}`).join(' | '))
}

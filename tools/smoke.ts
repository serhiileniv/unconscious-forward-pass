import { buildSession, generate } from '../src/model/run'
import { nearest } from '../src/model/embedding'

const t0 = Date.now()
const s = buildSession()
console.log(`vocab=${s.vocab.size}  build=${Date.now() - t0}ms`)

// The claim the visualisation rests on: nearby points are semantically related.
for (const w of ['silence', 'attention', 'layer', 'word', 'model']) {
  const id = s.vocab.index.get(w)
  if (id === undefined) { console.log(`  ${w}: not in vocab`); continue }
  console.log(`  ${w.padEnd(10)} -> ${nearest(s.emb, s.vocab, id, 6).map((n) => `${n.word}(${n.sim.toFixed(2)})`).join(' ')}`)
}

const t1 = Date.now()
const run = generate(s, 'most of what a mind does is never', 12)
console.log(`\ngenerate 12 tokens = ${Date.now() - t1}ms`)
console.log('out:', run.words.join(' '))
console.log(`considered=${run.considered} spoken=${run.spoken} ratio=${(run.considered / run.spoken).toFixed(0)}:1`)
const st = run.steps[0]
console.log(`layers=${st.layers.length} T=${st.ids.length} entropy=${st.entropy.toFixed(2)} bits`)
const l6 = st.layers[6]
console.log(`layer6 pos-last active=${l6.features[l6.features.length - 1].length} suppressed=${l6.suppressed[l6.suppressed.length - 1].length}`)
console.log('layer6 top features:', l6.features[l6.features.length - 1].slice(0, 5).map((f) => `${s.model.dictLabels[f.id]}=${f.act.toFixed(2)}`).join('  '))
let attnSum = 0
for (let j = 0; j < st.ids.length; j++) attnSum += l6.attn[0 * st.ids.length * st.ids.length + (st.ids.length - 1) * st.ids.length + j]
console.log(`attn row sums to ${attnSum.toFixed(4)} (must be 1.0)`)
console.log(`anticipations=${run.anticipations.length}`, run.anticipations.slice(0, 3).map((a) => `${a.label}@s${a.step}->${a.targetWord}@s${a.targetStep}(${a.sim.toFixed(2)})`).join(' | '))

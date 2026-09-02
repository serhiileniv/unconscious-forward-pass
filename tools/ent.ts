import { buildSession, generate } from '../src/model/run'
const s = buildSession()
const run = generate(s, 'most of what a mind does is never', 10, { temperature: 0.85, seed: 1 })
run.steps.forEach((st, i) => {
  const top = st.candidates.slice(0, 3).map((c) => `${c.word}:${(c.prob * 100).toFixed(1)}`).join(' ')
  console.log(`step ${i} T=${st.ids.length} H=${st.entropy.toFixed(3)}  ${top}`)
})

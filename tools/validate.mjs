/**
 * Run the in-page self-check against a live dev server.
 *
 *   node tools/validate.mjs [url]
 *
 * Every claim the interface makes about what a visual property means is
 * re-derived from the trace and compared with the buffers actually being drawn.
 * Exits non-zero if any check fails at any sampled moment.
 */
import { spawn } from 'node:child_process'

const url = process.argv[2] ?? 'http://localhost:5173/'
const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new', '--use-angle=metal', '--hide-scrollbars', '--window-size=1600,900',
  '--remote-debugging-port=9444', `--user-data-dir=${process.env.TMPDIR || '/tmp'}/cdp-validate`, 'about:blank',
], { stdio: 'ignore' })

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
async function target() {
  for (let i = 0; i < 80; i++) {
    try {
      const list = await (await fetch('http://127.0.0.1:9444/json/list')).json()
      const page = list.find((t) => t.type === 'page')
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl
    } catch {}
    await sleep(250)
  }
  throw new Error('chrome did not start')
}

const ws = new WebSocket(await target())
await new Promise((r) => ws.addEventListener('open', r, { once: true }))
let id = 0
const pending = new Map()
ws.addEventListener('message', (ev) => {
  const m = JSON.parse(ev.data)
  if (!m.id || !pending.has(m.id)) return
  const { resolve, reject } = pending.get(m.id)
  pending.delete(m.id)
  m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result)
})
const send = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const n = ++id
    pending.set(n, { resolve, reject })
    ws.send(JSON.stringify({ id: n, method, params }))
  })
const evaluate = async (expression) => {
  const r = await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })
  if (r.exceptionDetails) {
    throw new Error(r.exceptionDetails.exception?.description ?? r.exceptionDetails.text)
  }
  return r.result?.value
}

await send('Runtime.enable')
await send('Page.enable')
await send('Page.navigate', { url })
for (let i = 0; i < 260; i++) {
  if (await evaluate('!!(window.__uc && window.__uc.ready && window.__uc.ready())')) break
  await sleep(500)
}
await sleep(1500)

const report = await evaluate(`(async () => {
  const f = () => new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));
  const rows = [];
  for (const step of [0, 1, 3]) {
    for (const frac of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      window.__uc.at(step, frac); await f(); await f();
      for (const r of window.__uc.validate()) rows.push({ step, frac, ...r });
    }
  }
  return rows;
})()`)

const byName = new Map()
for (const r of report) {
  const e = byName.get(r.name) ?? { pass: 0, fail: 0, sample: '' }
  r.pass ? e.pass++ : e.fail++
  if (!r.pass) e.sample = `step ${r.step} frac ${r.frac}: ${r.detail}`
  // Prefer the busiest passing sample, so a check cannot look healthy by
  // reporting a moment where nothing was drawn at all.
  else if (!e.fail) {
    const size = (t) => Number((String(t).match(/\d[\d,]*/) ?? ['0'])[0].replace(/,/g, ''))
    if (!e.sample || size(r.detail) > size(e.sample)) e.sample = r.detail
  }
  byName.set(r.name, e)
}
let failed = 0
console.log(`${report.length} checks across 15 moments\n`)
for (const [name, e] of byName) {
  if (e.fail) failed++
  console.log(`${e.fail ? 'FAIL' : 'pass'}  ${name}`)
  console.log(`      ${e.pass}/${e.pass + e.fail}  ${e.sample}`)
}
ws.close()
chrome.kill()
console.log(failed ? `\n${failed} check(s) failed` : '\nall checks pass')
process.exit(failed ? 1 : 0)

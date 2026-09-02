// Minimal CDP driver: load the page, collect console output, screenshot.
// Chrome's --virtual-time-budget never fires on a page with a permanent rAF loop,
// which is exactly what this app is, so we drive it explicitly instead.
import { spawn } from 'node:child_process'
import { writeFileSync } from 'node:fs'

const [url, out, waitMs = '9000', evalJs = ''] = process.argv.slice(2)
const PORT = 9222
const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'

const chrome = spawn(CHROME, [
  '--headless=new',
  '--disable-gpu',
  '--enable-unsafe-swiftshader',
  '--hide-scrollbars',
  '--window-size=' + (process.env.WSIZE || '1600,1000'),
  '--force-device-scale-factor=1',
  `--remote-debugging-port=${PORT}`,
  `--user-data-dir=${process.env.TMPDIR || '/tmp'}/cdp-shot-profile`,
  'about:blank',
], { stdio: 'ignore' })

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function target() {
  for (let i = 0; i < 60; i++) {
    try {
      const list = await (await fetch(`http://127.0.0.1:${PORT}/json/list`)).json()
      const page = list.find((t) => t.type === 'page')
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl
    } catch {}
    await sleep(250)
  }
  throw new Error('chrome did not come up')
}

const ws = new WebSocket(await target())
await new Promise((r) => ws.addEventListener('open', r, { once: true }))

let id = 0
const pending = new Map()
const logs = []
ws.addEventListener('message', (ev) => {
  const msg = JSON.parse(ev.data)
  if (msg.id && pending.has(msg.id)) {
    const { resolve, reject } = pending.get(msg.id)
    pending.delete(msg.id)
    msg.error ? reject(new Error(JSON.stringify(msg.error))) : resolve(msg.result)
  } else if (msg.method === 'Runtime.consoleAPICalled') {
    logs.push(`[${msg.params.type}] ` + msg.params.args.map((a) => a.description ?? a.value).join(' '))
  } else if (msg.method === 'Runtime.exceptionThrown') {
    logs.push('[uncaught] ' + (msg.params.exceptionDetails.exception?.description ?? msg.params.exceptionDetails.text))
  } else if (msg.method === 'Log.entryAdded' && msg.params.entry.level === 'error') {
    logs.push('[log] ' + msg.params.entry.text)
  }
})

const send = (method, params = {}) =>
  new Promise((resolve, reject) => {
    const n = ++id
    pending.set(n, { resolve, reject })
    ws.send(JSON.stringify({ id: n, method, params }))
  })

await send('Runtime.enable')
await send('Log.enable')
await send('Page.enable')
await send('Page.navigate', { url })
await sleep(Number(waitMs))

if (evalJs) {
  const r = await send('Runtime.evaluate', { expression: evalJs, returnByValue: true, awaitPromise: true })
  if (r.result?.value !== undefined) logs.push('[eval] ' + JSON.stringify(r.result.value))
  await sleep(1200)
}

const shot = await send('Page.captureScreenshot', { format: 'png' })
writeFileSync(out, Buffer.from(shot.data, 'base64'))
console.log(logs.length ? logs.join('\n') : '(console clean)')
ws.close()
chrome.kill()
process.exit(0)

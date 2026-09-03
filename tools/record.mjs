/**
 * Frame-exact recorder.
 *
 * Screen-grabbing this page would capture whatever framerate the machine managed;
 * instead each frame is rendered at an explicit simulated time and an explicit
 * camera position, so the finished video plays at exactly the speed it claims to
 * and the camera move is the same every run.
 *
 *   node tools/record.mjs <url> <outDir> [seconds] [fps] [width] [height]
 */
import { spawn } from 'node:child_process'
import { mkdirSync, writeFileSync } from 'node:fs'
import { join } from 'node:path'

const [url, outDir, secs = '30', fps = '30', W = '1600', H = '900'] = process.argv.slice(2)
const SECONDS = Number(secs)
const FPS = Number(fps)
const TOTAL = Math.round(SECONDS * FPS)
const WORDS = 12 // 12 x 2600ms per pass ≈ 31s of material for a 30s cut
// GPT-2 answers this correctly and is barely sure of it: " Paris" leads at 6.4%
// with 436 words still effectively in play, which is the whole argument of the
// piece in one prompt. Temperature zero so the emitted word is the model's own
// best answer and the layers and the sentence tell one story.
const PROMPT = process.env.PROMPT ?? 'The Eiffel Tower is located in the city of'
const TEMP = process.env.TEMP ?? '0'

mkdirSync(outDir, { recursive: true })

// SwiftShader rasterises hundreds of thousands of additively blended points on
// the CPU, at about 1 fps. Metal does the same work on the GPU. SOFTWARE=1 falls
// back if the hardware path misbehaves on a given machine.
const gpuFlags = process.env.SOFTWARE
  ? ['--disable-gpu', '--enable-unsafe-swiftshader']
  : ['--use-angle=metal', '--ignore-gpu-blocklist', '--enable-gpu-rasterization']

const chrome = spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', [
  '--headless=new',
  ...gpuFlags,
  // Half a gigabyte of weights plus a large point buffer; the default heap
  // ceiling was being hit part-way through a capture.
  '--js-flags=--max-old-space-size=4096',
  '--disable-dev-shm-usage',
  '--hide-scrollbars',
  `--window-size=${W},${H}`,
  '--force-device-scale-factor=1',
  '--remote-debugging-port=9333',
  `--user-data-dir=${process.env.TMPDIR || '/tmp'}/cdp-record-profile`,
  'about:blank',
], { stdio: 'ignore' })

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

async function targetUrl() {
  for (let i = 0; i < 80; i++) {
    try {
      const list = await (await fetch('http://127.0.0.1:9333/json/list')).json()
      const page = list.find((t) => t.type === 'page')
      if (page?.webSocketDebuggerUrl) return page.webSocketDebuggerUrl
    } catch {}
    await sleep(250)
  }
  throw new Error('chrome did not come up')
}

const ws = new WebSocket(await targetUrl())
await new Promise((r) => ws.addEventListener('open', r, { once: true }))

// A dropped socket used to hang on an unsettled await with no explanation.
let dead = null
const die = (why) => {
  dead = why
  for (const { reject } of pending.values()) reject(new Error(why))
  pending.clear()
}
ws.addEventListener('close', () => die('devtools socket closed (chrome exited?)'))
ws.addEventListener('error', () => die('devtools socket error'))

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
    if (dead) return reject(new Error(dead))
    const n = ++id
    pending.set(n, { resolve, reject })
    ws.send(JSON.stringify({ id: n, method, params }))
  })
const evaluate = (expression) =>
  send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true }).then((r) => r.result?.value)

await send('Runtime.enable')
await send('Page.enable')
await send('Page.navigate', { url })

// Wait for the model build to finish and the hooks to appear.
// Half a gigabyte of weights has to arrive and unpack first.
for (let i = 0; i < 260; i++) {
  if (await evaluate('!!(window.__uc && window.__uc.ready && window.__uc.ready())')) break
  await sleep(500)
}

// Longer continuation, and a clean frame: the controls and transport are chrome,
// not content, and they would date the video the moment the UI changes.
await evaluate(`(() => {
  const set = (id, v) => {
    const el = document.getElementById(id)
    el.value = v
    el.dispatchEvent(new Event('input', { bubbles: true }))
  }
  set('steps', '${WORDS}')
  set('temp', '${TEMP}')
  document.getElementById('prompt').value = ${JSON.stringify(PROMPT)}
  document.getElementById('run').click()
  return true
})()`)
await sleep(22000) // a full generation is ~10s of real forward passes
await evaluate(`(() => {
  for (const id of ['panel', 'panel-toggle', 'transport']) {
    const el = document.getElementById(id)
    if (el) el.style.display = 'none'
  }
  document.getElementById('masthead').style.left = '32px'
  return true
})()`)

const duration = await evaluate('window.__uc.duration()')
console.log(`run is ${(duration / 1000).toFixed(1)}s; capturing ${TOTAL} frames at ${FPS}fps`)

const started = Date.now()
for (let i = 0; i < TOTAL; i++) {
  const u = i / (TOTAL - 1)
  const ms = u * (duration - 1)
  // Side-on rather than down the barrel: the corridor spans the frame and the
  // lit slab travels across it, instead of sitting in one corner while
  // three-quarters of the shot stays empty. Slow rise, slow push in.
  const azimuth = 0.80 + 0.36 * u
  const elevation = 0.17 + 0.14 * (0.5 - 0.5 * Math.cos(u * Math.PI * 2))
  const distance = 59 - 7 * u
  await evaluate(`window.__uc.capture(${ms}, ${azimuth}, ${elevation}, ${distance})`)
  const shot = await send('Page.captureScreenshot', { format: 'png' })
  writeFileSync(join(outDir, `f${String(i).padStart(4, '0')}.png`), Buffer.from(shot.data, 'base64'))
  if (i % 30 === 0 || i === TOTAL - 1) {
    const rate = (i + 1) / ((Date.now() - started) / 1000)
    console.log(`  ${i + 1}/${TOTAL}  ${rate.toFixed(1)} fps  eta ${Math.round((TOTAL - i - 1) / rate)}s`)
  }
}

ws.close()
chrome.kill()
console.log('done')
process.exit(0)

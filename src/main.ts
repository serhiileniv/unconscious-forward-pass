import './style.css'
import { buildSession, generate, type Run, type Session } from './model/run'
import { splitWords } from './model/tokenizer'
import { View, type RenderState } from './scene/view'

/** Milliseconds for one full sweep through the stack, at 1×. */
const STEP_MS = 2600
/** Fraction of a step spent sweeping; the remainder holds on the readout. */
const SWEEP = 0.8

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id)
  if (!el) throw new Error(`missing #${id}`)
  return el as T
}

const els = {
  boot: $('boot'),
  canvas: $<HTMLCanvasElement>('stage'),
  labels: $('labels'),
  panel: $('panel'),
  panelToggle: $<HTMLButtonElement>('panel-toggle'),
  prompt: $<HTMLTextAreaElement>('prompt'),
  steps: $<HTMLInputElement>('steps'),
  stepsOut: $<HTMLOutputElement>('steps-out'),
  temp: $<HTMLInputElement>('temp'),
  tempOut: $<HTMLOutputElement>('temp-out'),
  run: $<HTMLButtonElement>('run'),
  buildNote: $('build-note'),
  head: $<HTMLSelectElement>('head'),
  play: $<HTMLButtonElement>('play'),
  scrub: $<HTMLInputElement>('scrub'),
  transport: $('transport'),
  fan: $('fan'),
  fanList: $('fan-list'),
  considered: $('r-considered'),
  spoken: $('r-spoken'),
  ratio: $('r-ratio'),
  layer: $('r-layer'),
  awake: $('r-awake'),
  contending: $('r-contending'),
  entropy: $('r-entropy'),
  utterPrompt: $('utterance-prompt'),
  utterOut: $('utterance-out'),
}

let session: Session
let view: View
let run: Run
let promptWords: string[] = []

let playing = true
let speed = 1
let position = 0
let last = performance.now()
let renderedWords = -1

const show: RenderState['show'] = {
  features: true,
  attention: true,
  streams: true,
  labels: true,
  plans: true,
}
let headFilter = -1

// ---------------------------------------------------------------------------

function boot(): void {
  try {
    session = buildSession()
  } catch (err) {
    fail(err)
    return
  }

  try {
    view = new View(els.canvas, els.labels, session)
  } catch (err) {
    fail(err, 'This needs WebGL. Try a different browser or enable hardware acceleration.')
    return
  }

  for (let h = 0; h < session.cfg.nHeads; h++) {
    const opt = document.createElement('option')
    opt.value = String(h)
    opt.textContent = `head ${h}`
    els.head.appendChild(opt)
  }
  els.buildNote.textContent =
    `${session.cfg.nLayers} layers · ${session.cfg.dModel} dims · ${session.cfg.nFeatures} features · ` +
    `${session.vocab.size} words`

  wire()
  regenerate()
  resize()
  els.boot.classList.add('gone')
  requestAnimationFrame(frame)
}

function fail(err: unknown, hint = ''): void {
  console.error(err)
  els.boot.innerHTML = `<span style="max-width:40ch;text-align:center;line-height:1.7">Could not start.<br>${
    hint || String(err instanceof Error ? err.message : err)
  }</span>`
}

function regenerate(): void {
  const prompt = els.prompt.value.trim() || 'most of what a mind does is never'
  const nSteps = Number(els.steps.value)
  const temperature = Number(els.temp.value)

  run = generate(session, prompt, nSteps, { temperature, seed: hash(prompt) })
  view.setRun(run)

  // Only the words the model actually saw belong in the prompt line.
  const known = splitWords(prompt).filter((w) => session.vocab.index.has(w))
  promptWords = known.slice(-Math.max(1, session.cfg.maxT - nSteps))
  els.utterPrompt.textContent = promptWords.join(' ') + ' '
  els.utterOut.textContent = ''
  renderedWords = -1
  position = 0
  playing = true
  setPlayLabel()
}

// ---------------------------------------------------------------------------

function frame(now: number): void {
  const dt = Math.min(64, now - last)
  last = now
  const total = run.steps.length * STEP_MS

  if (playing) {
    position += dt * speed
    if (position >= total) position = total - 1 // hold on the final word
  }

  const step = Math.min(run.steps.length - 1, Math.floor(position / STEP_MS))
  const t = (position % STEP_MS) / STEP_MS
  const sweepT = Math.min(1, t / SWEEP)
  const nLayers = session.cfg.nLayers
  const layerF = -0.6 + sweepT * (nLayers - 1 + 0.6)

  view.render({ step, layerF, headFilter, show })
  updateHud(step, t, sweepT, layerF)

  els.scrub.value = String(Math.round((position / total) * 1000))
  requestAnimationFrame(frame)
}

function updateHud(step: number, t: number, sweepT: number, layerF: number): void {
  const nLayers = session.cfg.nLayers
  const trace = run.steps[step]

  let considered = 0
  for (let s = 0; s < step; s++) considered += run.steps[s].activations
  considered += Math.round(trace.activations * sweepT)

  const done = step + (t >= SWEEP ? 1 : 0)
  els.considered.textContent = considered.toLocaleString()
  els.spoken.textContent = String(done)
  // The whole thesis in one number: how much was weighed per word that got out.
  els.ratio.textContent = done > 0 ? `${Math.round(considered / done).toLocaleString()} : 1` : '—'

  const l = Math.round(layerF)
  const inside = l >= 0 && l < nLayers
  els.layer.textContent = inside ? `${l} / ${nLayers - 1}` : '—'

  if (inside) {
    const layer = trace.layers[l]
    let awake = 0
    for (const hits of layer.features) awake += hits.length
    let contending = 0
    for (const n of layer.contended) contending += n
    const pct = (awake / (session.cfg.nFeatures * trace.ids.length)) * 100
    els.awake.textContent = `${awake} · ${pct.toFixed(2)}%`
    els.contending.textContent = contending.toLocaleString()
  } else {
    els.awake.textContent = '—'
    els.contending.textContent = '—'
  }
  els.entropy.textContent = `${trace.entropy.toFixed(2)} bits`

  // The candidate fan resolves at the end of the sweep.
  const showFan = t > 0.6
  els.fan.hidden = !showFan
  if (showFan) renderFan(step, t >= SWEEP)

  if (done !== renderedWords) {
    renderedWords = done
    renderUtterance(done)
  }
}

function renderFan(step: number, resolved: boolean): void {
  const trace = run.steps[step]
  const key = `${step}:${resolved}`
  if (els.fanList.dataset.key === key) return
  els.fanList.dataset.key = key

  const top = trace.candidates.slice(0, 8)
  const max = top[0]?.prob || 1
  els.fanList.innerHTML = top
    .map((c) => {
      const chosen = resolved && c.id === trace.chosen
      const w = Math.max(2, (c.prob / max) * 100)
      return `<li class="${chosen ? 'chosen' : ''}"><i class="bar" style="width:${w}%"></i>` +
        `<span>${escapeHtml(c.word)}</span><span class="p">${(c.prob * 100).toFixed(1)}%</span></li>`
    })
    .join('')
}

function renderUtterance(done: number): void {
  els.utterOut.innerHTML = run.words
    .slice(0, done)
    .map((w, i) => {
      const age = done - 1 - i
      const cls = age === 0 ? 'w fresh' : age < 3 ? 'w recent' : 'w'
      return `<span class="${cls}">${escapeHtml(w)}</span>`
    })
    .join(' ')
}

// ---------------------------------------------------------------------------

function wire(): void {
  els.run.addEventListener('click', () => {
    els.run.disabled = true
    els.run.textContent = 'running…'
    // Yield once so the button state paints before the synchronous forward passes.
    setTimeout(() => {
      try {
        regenerate()
      } finally {
        els.run.disabled = false
        els.run.textContent = 'run the pass'
      }
    }, 20)
  })
  els.prompt.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault()
      els.run.click()
    }
  })

  els.steps.addEventListener('input', () => (els.stepsOut.value = els.steps.value))
  els.temp.addEventListener('input', () => (els.tempOut.value = Number(els.temp.value).toFixed(2)))

  els.play.addEventListener('click', togglePlay)
  els.scrub.addEventListener('input', () => {
    playing = false
    setPlayLabel()
    position = (Number(els.scrub.value) / 1000) * run.steps.length * STEP_MS
  })

  for (const b of els.transport.querySelectorAll<HTMLButtonElement>('.speeds button')) {
    b.addEventListener('click', () => {
      speed = Number(b.dataset.speed)
      for (const o of els.transport.querySelectorAll('.speeds button')) o.classList.remove('on')
      b.classList.add('on')
    })
  }

  const toggles: [string, keyof RenderState['show']][] = [
    ['t-features', 'features'],
    ['t-attention', 'attention'],
    ['t-streams', 'streams'],
    ['t-labels', 'labels'],
    ['t-plans', 'plans'],
  ]
  for (const [id, key] of toggles) {
    const cb = $<HTMLInputElement>(id)
    cb.addEventListener('change', () => (show[key] = cb.checked))
  }
  const rotate = $<HTMLInputElement>('t-rotate')
  rotate.addEventListener('change', () => (view.controls.autoRotate = rotate.checked))

  els.head.addEventListener('change', () => (headFilter = Number(els.head.value)))

  // On a phone the panel would cover the thing it describes.
  if (window.innerWidth <= 900) {
    els.panel.hidden = true
    els.panelToggle.setAttribute('aria-expanded', 'false')
  }

  els.panelToggle.addEventListener('click', () => {
    const open = !els.panel.hidden
    els.panel.hidden = open
    els.panelToggle.setAttribute('aria-expanded', String(!open))
  })

  document.addEventListener('keydown', (e) => {
    if (e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLInputElement) return
    if (e.code === 'Space') {
      e.preventDefault()
      togglePlay()
    }
  })

  window.addEventListener('resize', resize)
}

function togglePlay(): void {
  playing = !playing
  if (playing && position >= run.steps.length * STEP_MS - 2) position = 0
  setPlayLabel()
}

function setPlayLabel(): void {
  els.play.textContent = playing ? '❚❚' : '▶'
  els.play.setAttribute('aria-label', playing ? 'pause' : 'play')
}

function resize(): void {
  view.resize(window.innerWidth, window.innerHeight, Math.min(2, window.devicePixelRatio))
}

function hash(s: string): number {
  let h = 2166136261
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i)
    h = Math.imul(h, 16777619)
  }
  return h >>> 0
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)
}

/**
 * Deterministic seek, for screenshots and for debugging a specific instant.
 * `__uc.at(3, 0.5)` parks the wavefront halfway through the fourth word's pass.
 */
declare global {
  interface Window {
    __uc?: { at(step: number, frac: number): void; pause(): void; play(): void }
  }
}
window.__uc = {
  at(step, frac) {
    position = (step + Math.min(0.999, Math.max(0, frac))) * STEP_MS
    playing = false
    setPlayLabel()
  },
  pause() {
    playing = false
    setPlayLabel()
  },
  play() {
    playing = true
    setPlayLabel()
  },
}

// Defer one frame so the boot screen actually paints before the model is built.
requestAnimationFrame(() => requestAnimationFrame(boot))

import './style.css'
import type { LM } from './model/lm'
import { generate, loadModel, MAX_CONTEXT, type Run } from './model/run'
import { View, type RenderState } from './scene/view'

/** Milliseconds for one full sweep through the stack, at 1×. */
const STEP_MS = 2600
/** Fraction of a step spent sweeping; the remainder holds on the readout. */
const SWEEP = 0.8

const PRESETS = [
  'The Eiffel Tower is located in the city of',
  'When Mary and John went to the store, John gave a drink to',
  'The first president of the United States was named',
  '2 + 2 =',
]

const $ = <T extends HTMLElement>(id: string): T => {
  const el = document.getElementById(id)
  if (!el) throw new Error(`missing #${id}`)
  return el as T
}

const els = {
  boot: $('boot'),
  bootPhase: $('boot-phase'),
  bootFill: $('boot-fill'),
  bootDetail: $('boot-detail'),
  canvas: $<HTMLCanvasElement>('stage'),
  labels: $('labels'),
  panel: $('panel'),
  panelToggle: $<HTMLButtonElement>('panel-toggle'),
  prompt: $<HTMLTextAreaElement>('prompt'),
  tokens: $('tokens'),
  presets: $('presets'),
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
  agree: $('r-agree'),
  lensNote: $('lens-note'),
  utterPrompt: $('utterance-prompt'),
  utterOut: $('utterance-out'),
}

let model: LM
let view: View
let run: Run

let playing = true
let speed = 1
let position = 0
let last = performance.now()
let renderedWords = -1
let frameWaiters: (() => void)[] = []

const show: RenderState['show'] = {
  features: true,
  attention: true,
  streams: true,
  labels: true,
  plans: true,
}
let headFilter = -1

// ---------------------------------------------------------------------------

async function boot(): Promise<void> {
  try {
    model = await loadModel((phase, frac) => {
      els.bootPhase.textContent = phase === 'weights' ? 'downloading Qwen2.5' : phase
      els.bootFill.style.width = `${Math.round(frac * 100)}%`
      if (phase === 'weights') {
        els.bootDetail.textContent = `${(frac * 942).toFixed(0)} of 942 MB — cached after the first visit`
      }
    })
  } catch (err) {
    fail(err, 'Could not load the weights. Run `npm run fetch-weights` first.')
    return
  }

  try {
    view = new View(els.canvas, els.labels, model)
  } catch (err) {
    fail(err, 'This needs WebGL. Try another browser, or enable hardware acceleration.')
    return
  }

  for (let h = 0; h < model.cfg.nHead; h++) {
    const opt = document.createElement('option')
    opt.value = String(h)
    opt.textContent = `head ${h}`
    els.head.appendChild(opt)
  }
  els.buildNote.textContent =
    `${model.cfg.name} · ${model.cfg.nLayer} layers · ${model.cfg.nEmbd} dims · ` +
    `${model.cfg.nHead} heads · ${model.lensIds.length.toLocaleString()} tokens drawn`

  wire()
  buildPresets()
  regenerate()
  resize()
  els.boot.classList.add('gone')
  requestAnimationFrame(frame)
}

function fail(err: unknown, hint = ''): void {
  console.error(err)
  els.boot.innerHTML = `<span style="max-width:44ch;text-align:center;line-height:1.7">${
    hint || String(err instanceof Error ? err.message : err)
  }</span>`
}

function regenerate(): void {
  const nSteps = Number(els.steps.value)
  run = generate(model, els.prompt.value, nSteps, {
    temperature: Number(els.temp.value),
    seed: hash(els.prompt.value),
  })
  view.setRun(run)
  renderTokens()
  els.utterPrompt.textContent = run.promptWords.join('')
  els.utterOut.textContent = ''
  renderedWords = -1
  position = 0
  playing = true
  setPlayLabel()
}

/**
 * Show how GPT-2 actually splits the prompt.
 *
 * Byte-level BPE takes any input, so nothing is ever silently dropped — but what
 * comes out is often not words. "2 + 2 =" is four tokens, and "Eiffel" is three.
 * Seeing that is half of understanding why the model behaves as it does.
 */
function renderTokens(): void {
  els.tokens.innerHTML = ''
  for (const piece of run.promptWords) {
    const span = document.createElement('span')
    span.className = 'tok'
    span.textContent = piece.replace(/ /g, '·')
    els.tokens.appendChild(span)
  }
}

function buildPresets(): void {
  for (const text of PRESETS) {
    const b = document.createElement('button')
    b.type = 'button'
    b.textContent = text.length > 34 ? `${text.slice(0, 32)}…` : text
    b.title = text
    b.addEventListener('click', () => {
      els.prompt.value = text
      els.run.click()
    })
    els.presets.appendChild(b)
  }
}

// ---------------------------------------------------------------------------

function frame(now: number): void {
  const dt = Math.min(64, now - last)
  last = now
  const total = run.steps.length * STEP_MS

  if (playing) {
    position += dt * speed
    if (position >= total) position = total - 1
  }

  const step = Math.min(run.steps.length - 1, Math.floor(position / STEP_MS))
  const t = (position % STEP_MS) / STEP_MS
  const sweepT = Math.min(1, t / SWEEP)
  const nLayers = model.cfg.nLayer
  const layerF = -0.6 + sweepT * (nLayers - 1 + 0.6)

  view.render({ step, layerF, headFilter, show })
  updateHud(step, t, sweepT, layerF)

  els.scrub.value = String(Math.round((position / total) * 1000))

  if (frameWaiters.length) {
    const waiting = frameWaiters
    frameWaiters = []
    for (const resolve of waiting) resolve()
  }
  requestAnimationFrame(frame)
}

function updateHud(step: number, t: number, sweepT: number, layerF: number): void {
  const nLayers = model.cfg.nLayer
  const trace = run.steps[step]

  let considered = 0
  for (let s = 0; s < step; s++) considered += run.steps[s].activations
  considered += Math.round(trace.activations * sweepT)

  const done = step + (t >= SWEEP ? 1 : 0)
  els.considered.textContent = considered.toLocaleString()
  els.spoken.textContent = String(done)
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
    els.awake.textContent = String(awake)
    els.contending.textContent = contending.toLocaleString()
    setAgreement(layer.agreement)
  } else {
    els.awake.textContent = '—'
    els.contending.textContent = '—'
    els.agree.textContent = '—'
    els.lensNote.textContent = ''
    els.lensNote.className = ''
  }
  els.entropy.textContent = `${trace.entropy.toFixed(2)} bits`

  const showFan = t > 0.6
  els.fan.hidden = !showFan
  if (showFan) renderFan(step, t >= SWEEP)

  if (done !== renderedWords) {
    renderedWords = done
    renderUtterance(done)
  }
}

/**
 * Say, at every depth, how much of the final answer is actually assembled. Early
 * layers push hard on words that never survive, so without this the picture
 * implies far more resolution than the model has at that point.
 */
function setAgreement(agreement: number): void {
  els.agree.textContent = `${Math.round(agreement * 10)} / 10`
  if (agreement >= 0.9) {
    els.lensNote.textContent = 'the running total is now the model\u2019s actual output'
    els.lensNote.className = 'good'
  } else if (agreement >= 0.5) {
    els.lensNote.textContent = 'most of the answer is in place'
    els.lensNote.className = 'good'
  } else if (agreement >= 0.2) {
    els.lensNote.textContent = 'the answer is starting to form'
    els.lensNote.className = ''
  } else {
    els.lensNote.textContent = 'nothing of the final answer is in place yet'
    els.lensNote.className = 'warn'
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
      const label = c.word.replace(/ /g, '·').replace(/\n/g, '\\n')
      return `<li class="${chosen ? 'chosen' : ''}"><i class="bar" style="width:${w}%"></i>` +
        `<span>${escapeHtml(label)}</span><span class="p">${(c.prob * 100).toFixed(1)}%</span></li>`
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
    .join('')
}

// ---------------------------------------------------------------------------

function wire(): void {
  els.run.addEventListener('click', () => {
    els.run.disabled = true
    els.run.textContent = 'running…'
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

  els.steps.max = String(MAX_CONTEXT - 12)
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
 * Deterministic seek, for screenshots and recording.
 * `__uc.at(3, 0.5)` parks the wavefront halfway through the fourth word's pass.
 */
declare global {
  interface Window {
    __uc?: {
      at(step: number, frac: number): void
      pause(): void
      play(): void
      duration(): number
      capture(ms: number, azimuth: number, elevation: number, distance: number): Promise<void>
    }
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
  duration() {
    return run.steps.length * STEP_MS
  },
  capture(ms, azimuth, elevation, distance) {
    playing = false
    position = ms
    view.setCamera(azimuth, elevation, distance)
    return new Promise<void>((resolve) => frameWaiters.push(resolve))
  },
}

void boot()

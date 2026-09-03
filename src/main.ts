import './style.css'
import type { LM } from './model/lm'
import { DEFAULT_MODEL, generate, loadModel, MODELS, type Run } from './model/run'
import { View, type RenderState } from './scene/view'
import { renderCurves } from './ui/curves'

/** Milliseconds for one full sweep through the stack, at 1×. */
const STEP_MS = 2600
/** Fraction of a step spent sweeping; the remainder holds on the readout. */
const SWEEP = 0.8

/**
 * Prompts chosen by what GPT-2 actually does with them, checked in
 * tools/presets.ts. The note is the model's own top answer and its confidence,
 * so a weak or wrong result reads as the model's limit rather than a fault in
 * the picture. The arithmetic one stays deliberately: a 124M model from 2019
 * cannot add, and that is worth being able to watch.
 */
const PRESETS: { text: string; note: string }[] = [
  { text: 'The Eiffel Tower is located in the city of', note: 'Paris, but at only 6% — 436 words in play' },
  { text: 'The first President of the United States was George', note: 'Washington, 55%' },
  { text: 'When Mary and John went to the store, John gave a drink to', note: 'Mary, 45% — the classic probe' },
  { text: 'One, two, three, four, five, six,', note: 'seven, 84% — near certain' },
  { text: 'Romeo and Juliet was written by William', note: 'Shakespeare, 22%' },
  { text: '2 + 2 =', note: 'says 3 — it cannot do arithmetic' },
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
  model: $<HTMLSelectElement>('model'),
  fidelityNote: $('fidelity-note'),
  head: $<HTMLSelectElement>('head'),
  play: $<HTMLButtonElement>('play'),
  scrub: $<HTMLInputElement>('scrub'),
  layerSlider: $<HTMLInputElement>('layer'),
  layerOut: $<HTMLOutputElement>('layer-out'),
  prevWord: $<HTMLButtonElement>('prev-word'),
  nextWord: $<HTMLButtonElement>('next-word'),
  transport: $('transport'),
  fan: $('fan'),
  fanList: $('fan-list'),
  curves: document.getElementById('curves') as unknown as SVGSVGElement,
  considered: $('r-considered'),
  spoken: $('r-spoken'),
  ratio: $('r-ratio'),
  layer: $('r-layer'),
  awake: $('r-awake'),
  contending: $('r-contending'),
  entropy: $('r-entropy'),
  inPlay: $('r-play'),
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
const tokenIndex = new Map<number, number>()

const show: RenderState['show'] = {
  features: true,
  synapses: true,
  attention: true,
  streams: true,
  labels: true,
  plans: true,
}
let headFilter = -1
/** GPT-2 token id the viewer is following, or null. */
let selectedToken: number | null = null
/** Its index in the drawn set, which is what the renderer needs. */
let selectedIndex: number | null = null

// ---------------------------------------------------------------------------

/** Which checkpoint to load. A reload rather than a swap, so only one set of
 * weights is ever resident. */
function modelKey(): string {
  const k = new URLSearchParams(location.search).get('model') ?? DEFAULT_MODEL
  return k in MODELS ? k : DEFAULT_MODEL
}

async function boot(): Promise<void> {
  const choice = MODELS[modelKey()]
  try {
    model = await loadModel(modelKey(), (phase: string, frac: number) => {
      els.bootPhase.textContent = phase === 'weights' ? `downloading ${choice.name}` : phase
      els.bootFill.style.width = `${Math.round(frac * 100)}%`
      if (phase === 'weights') {
        els.bootDetail.textContent = `${Math.round(frac * 100)}% — cached after the first visit`
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
    `${model.cfg.nLayer} layers · ${model.cfg.nEmbd} dims · ${model.cfg.nHead} heads · ` +
    `all ${model.lensIds.length.toLocaleString()} tokens drawn`
  for (const [key, m] of Object.entries(MODELS)) {
    const opt = document.createElement('option')
    opt.value = key
    opt.textContent = `${m.name} — ${m.note}`
    opt.selected = key === modelKey()
    els.model.appendChild(opt)
  }
  els.model.addEventListener('change', () => {
    location.search = `?model=${els.model.value}`
  })

  // The cloud's shape is the weakest claim in the piece, so it states its own
  // limits rather than letting the viewer read structure into a 3D shadow of a
  // 896-dimensional space.
  const f = model.fidelity
  const layout = model.layout
  els.fidelityNote.innerHTML = layout
    ? `Direction around the axis comes from a spectral embedding of the model's own nearest-neighbour ` +
      `graph. Measured, a token sits <b>54°</b> from a true neighbour and <b>89.5°</b> from a random one, ` +
      `where chance in three dimensions is 90° — so related words really do share a direction, roughly. ` +
      `Distance in that map carried nothing (true neighbours landed no closer than random pairs), so ` +
      `distance from the axis was given to the push strength instead, which is exact. The <em>links</em> ` +
      `are exact too: whether two tokens are neighbours is a fact about the model, true wherever the two ` +
      `ends are drawn.`
    : `Those are 3 of ${model.cfg.nEmbd} dimensions, holding <b>${(f.variance * 100).toFixed(1)}%</b> of ` +
      `the variance, with drawn distances correlating to the real ones at <b>r = ${f.distance.toFixed(2)}</b>. ` +
      `Proximity is a weak hint, not evidence. Depth, colour and brightness carry the exact quantities; ` +
      `left-right and up-down placement does not.`

  model.lensIds.forEach((tokenId, i) => tokenIndex.set(tokenId, i))

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
  selectedToken = null
  selectedIndex = null
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
  for (const { text, note } of PRESETS) {
    const b = document.createElement('button')
    b.type = 'button'
    b.title = text
    b.innerHTML = `${escapeHtml(text.length > 40 ? `${text.slice(0, 38)}…` : text)}<i>${escapeHtml(note)}</i>`
    b.addEventListener('click', () => {
      els.prompt.value = text
      els.run.click()
    })
    els.presets.appendChild(b)
  }
}

/** Park the wavefront on an exact layer, keeping the current word. */
function gotoLayer(layer: number): void {
  const nLayers = model.cfg.nLayer
  const step = Math.min(run.steps.length - 1, Math.floor(position / STEP_MS))
  const sweepT = (layer + 0.6) / (nLayers - 1 + 0.6)
  position = (step + Math.min(0.999, sweepT * SWEEP)) * STEP_MS
  playing = false
  setPlayLabel()
}

/** Move to the start of another word's pass. */
function stepWord(delta: number): void {
  const step = Math.min(run.steps.length - 1, Math.floor(position / STEP_MS))
  const next = Math.max(0, Math.min(run.steps.length - 1, step + delta))
  position = next * STEP_MS
  playing = false
  setPlayLabel()
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

  view.render({ step, layerF, headFilter, show, selected: selectedIndex })
  updateHud(step, t, sweepT, layerF)

  els.scrub.value = String(Math.round((position / total) * 1000))
  const shownLayer = Math.max(0, Math.min(nLayers - 1, layerF))
  els.layerSlider.value = String(Math.round(shownLayer * 10))
  els.layerOut.value = `${Math.round(shownLayer)} / ${nLayers - 1}`

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
  for (let s = 0; s < step; s++) considered += run.steps[s].scoreUpdates
  considered += Math.round(trace.scoreUpdates * sweepT)

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
    els.awake.textContent = `${awake} drawn`
    els.contending.textContent = `${contending.toLocaleString()} of ${model.lensIds.length.toLocaleString()}`
    setAgreement(layer.agreement)
  } else {
    els.awake.textContent = '—'
    els.contending.textContent = '—'
    els.agree.textContent = '—'
    els.lensNote.textContent = ''
    els.lensNote.className = ''
  }
  els.entropy.textContent = `${trace.entropy.toFixed(2)} bits`
  // Two to the power of the entropy: how many words are effectively still live
  // in the model's own distribution, before one of them is chosen.
  els.inPlay.textContent = Math.round(Math.pow(2, trace.entropy)).toLocaleString()

  // The chart is the answer to "why that word", so it is up for the whole pass
  // rather than only at the moment of choosing.
  els.fan.hidden = false
  renderFan(step, t >= SWEEP)
  renderCurves(els.curves, trace, layerF, selectedToken)

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
      const sel = selectedToken === c.id
      const w = Math.max(2, (c.prob / max) * 100)
      const label = c.word.replace(/ /g, '·').replace(/\n/g, '\\n')
      return `<li data-id="${c.id}" class="${chosen ? 'chosen' : ''}${sel ? ' sel' : ''}">` +
        `<i class="bar" style="width:${w}%"></i>` +
        `<span>${escapeHtml(label)}</span><span class="p">${(c.prob * 100).toFixed(1)}%</span></li>`
    })
    .join('')
}

/** Follow one word through the stack, or stop following it. */
function selectToken(tokenId: number | null): void {
  selectedToken = selectedToken === tokenId ? null : tokenId
  selectedIndex = selectedToken === null ? null : (tokenIndex.get(selectedToken) ?? null)
  els.fanList.dataset.key = ''
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
  els.temp.addEventListener('input', () => {
    const t = Number(els.temp.value)
    els.tempOut.value = t <= 0.001 ? 'best' : t.toFixed(2)
  })

  els.fanList.addEventListener('click', (e) => {
    const li = (e.target as HTMLElement).closest('li')
    if (li?.dataset.id) selectToken(Number(li.dataset.id))
  })

  els.play.addEventListener('click', togglePlay)
  els.prevWord.addEventListener('click', () => stepWord(-1))
  els.nextWord.addEventListener('click', () => stepWord(1))

  els.layerSlider.max = String((model.cfg.nLayer - 1) * 10)
  els.layerSlider.addEventListener('input', () => gotoLayer(Number(els.layerSlider.value) / 10))
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
    ['t-synapses', 'synapses'],
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
    } else if (e.code === 'ArrowLeft') {
      e.preventDefault()
      stepWord(-1)
    } else if (e.code === 'ArrowRight') {
      e.preventDefault()
      stepWord(1)
    } else if (e.code === 'ArrowUp' || e.code === 'ArrowDown') {
      e.preventDefault()
      const at = Number(els.layerSlider.value) / 10
      gotoLayer(Math.max(0, Math.min(model.cfg.nLayer - 1, at + (e.code === 'ArrowUp' ? 1 : -1))))
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

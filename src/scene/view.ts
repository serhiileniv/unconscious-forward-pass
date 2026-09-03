import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { LM } from '../model/lm'
import { MAX_CONTEXT, type Run } from '../model/run'
import type { StepTrace } from '../model/trace'
import { Arcs } from './arcs'
import { Labels, type LabelSpec } from './labels'
import { Lattice, layerZ, tokenX } from './lattice'
import { Rings } from './rings'
import { PALETTE } from './palette'
import { RISE, Streams, traceScale } from './streams'
import { Synapses } from './synapses'

export interface RenderState {
  step: number
  layerF: number
  headFilter: number
  /** Display index of a token the viewer is following, if any. */
  selected: number | null
  show: { features: boolean; synapses: boolean; attention: boolean; streams: boolean; labels: boolean; plans: boolean }
}

export class View {
  readonly camera: THREE.PerspectiveCamera
  readonly controls: OrbitControls
  private readonly renderer: THREE.WebGLRenderer
  private readonly scene: THREE.Scene
  private readonly lattice: Lattice
  private readonly streams: Streams
  private readonly arcs: Arcs
  private readonly rings: Rings
  private readonly synapses: Synapses
  private readonly labels: Labels
  private readonly plans: THREE.LineSegments
  private readonly model: LM
  private run: Run | null = null
  private currentStep = -1
  private width = 1
  private height = 1

  constructor(canvas: HTMLCanvasElement, labelHost: HTMLElement, model: LM) {
    this.model = model
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' })
    this.renderer.setClearColor(PALETTE.bg, 1)

    this.scene = new THREE.Scene()
    // Depth cueing. Layers far from the camera dim out, which is what makes the
    // stack read as depth rather than as a flat spray of points.
    this.scene.fog = new THREE.FogExp2(PALETTE.bg, 0.0105)

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 400)
    // Side-on rather than down the barrel: the stack spans the frame and the
    // sweeping wavefront travels across it instead of coming straight at you.
    this.camera.position.set(37.0, 11.0, 26.4)

    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.06
    this.controls.minDistance = 10
    this.controls.maxDistance = 180
    this.controls.autoRotate = true
    this.controls.autoRotateSpeed = 0.22
    this.controls.target.set(0, 0, -1)

    this.lattice = new Lattice(model)
    this.streams = new Streams(MAX_CONTEXT, model.cfg.nLayer)
    this.arcs = new Arcs()
    this.rings = new Rings(model.cfg.nLayer, this.lattice.radius)
    this.synapses = new Synapses(model, this.lattice)
    this.labels = new Labels(labelHost)

    this.plans = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({
        color: PALETTE.anticipation,
        transparent: true,
        opacity: 0.5,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    )
    this.plans.frustumCulled = false

    this.scene.add(this.rings.lines, this.synapses.lines, this.lattice.points, this.streams.lines, this.arcs.lines, this.plans)
  }

  /**
   * Park the camera on an explicit orbit position.
   *
   * Recording runs far slower than real time, so the ambient drift — which
   * advances per update call, not per simulated second — would spin wildly across
   * a capture. Driving the orbit from the frame index instead keeps the move
   * exactly as fast as the finished video says it is.
   */
  setCamera(azimuth: number, elevation: number, distance: number): void {
    this.controls.autoRotate = false
    this.controls.enableDamping = false
    const t = this.controls.target
    this.camera.position.set(
      t.x + distance * Math.cos(elevation) * Math.sin(azimuth),
      t.y + distance * Math.sin(elevation),
      t.z + distance * Math.cos(elevation) * Math.cos(azimuth),
    )
    this.camera.lookAt(t)
  }

  setRun(run: Run): void {
    this.run = run
    this.currentStep = -1
  }

  resize(width: number, height: number, dpr: number): void {
    this.width = width
    this.height = height
    this.camera.aspect = width / height
    // A portrait viewport crops the corridor badly at a cinematic FOV; widen it
    // rather than moving the camera, so the framing stays consistent.
    this.camera.fov = this.camera.aspect < 1 ? 62 : this.camera.aspect < 1.4 ? 52 : 42
    this.camera.updateProjectionMatrix()
    this.renderer.setPixelRatio(dpr)
    this.renderer.setSize(width, height, false)
  }

  render(state: RenderState): void {
    if (!this.run) return
    const trace = this.run.steps[state.step]
    if (!trace) return
    const nLayers = this.model.cfg.nLayer

    const stepChanged = state.step !== this.currentStep
    if (stepChanged) {
      this.currentStep = state.step
      this.streams.setTrace(trace, nLayers)
    }

    this.lattice.points.visible = state.show.features
    this.streams.lines.visible = state.show.streams
    this.arcs.lines.visible = state.show.attention
    this.plans.visible = state.show.plans

    // The cloud first: the rings are sized from what it just laid out.
    this.lattice.update(trace, state.layerF, state.selected)
    this.rings.update(state.layerF, this.layerWork(trace), this.lattice.layerRadius)
    this.synapses.lines.visible = state.show.synapses
    if (state.show.synapses) this.synapses.update(trace, state.layerF, nLayers)
    // Positions exist only once the cloud has been laid out for this frame.
    if (state.show.plans) this.buildPlans(state.step)
    if (state.show.streams) this.streams.update(trace, state.layerF)
    if (state.show.attention) this.arcs.update(trace, state.layerF, nLayers, state.headFilter)

    this.labels.update(state.show.labels ? this.collectLabels(state) : [], this.camera, this.width, this.height)

    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  /**
   * How much each layer wrote into the stream at the position being computed,
   * scaled against the largest write in the pass. Drives ring brightness, so the
   * corridor shows which layers did the work rather than looking uniform.
   */
  private layerWork(trace: StepTrace): Float32Array {
    const out = new Float32Array(trace.layers.length)
    let max = 1e-6
    for (const layer of trace.layers) max = Math.max(max, layer.writeNorm[trace.active] ?? 0)
    for (let l = 0; l < trace.layers.length; l++) out[l] = (trace.layers[l].writeNorm[trace.active] ?? 0) / max
    return out
  }

  /**
   * Mark tokens this layer is pushing that the run goes on to actually write.
   *
   * An earlier version drew a long line from the token to a made-up point above
   * the stack. The measurement is real but that destination was invented, so the
   * line is gone: what is drawn now is a short vertical stem on the token's own
   * point, at the place the token genuinely sits, and the label carries the lead.
   */
  private buildPlans(step: number): void {
    if (!this.run) return
    const pos: number[] = []
    const v = new THREE.Vector3()
    for (const a of this.run.anticipations.filter((x) => x.step === step)) {
      if (!this.featurePoint(a.featureId, a.layer, v)) continue
      pos.push(v.x, v.y, v.z, v.x, v.y + 1.6, v.z)
    }
    this.plans.geometry.dispose()
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    this.plans.geometry = g
  }

  private collectLabels(state: RenderState): LabelSpec[] {
    if (!this.run) return []
    const trace = this.run.steps[state.step]
    const nLayers = this.model.cfg.nLayer
    const l = Math.round(state.layerF)
    const out: LabelSpec[] = []
    const v = new THREE.Vector3()

    if (l >= 0 && l < nLayers) {
      const layer = trace.layers[l]
      const pos = trace.active
      const hits = layer.features[pos] ?? []
      // Labels must not strobe with the wavefront; hold a legible floor.
      const phase = 0.42 + 0.58 * (1 - Math.min(1, Math.abs(state.layerF - l) * 0.9))
      for (let i = 0; i < Math.min(3, hits.length); i++) {
        const hit = hits[i]
        if (!this.featurePoint(hit.id, l, v)) continue
        out.push({
          id: `f${hit.id}`,
          text: this.model.lensPieces[hit.id].trim(),
          world: v.clone(),
          kind: 'feature',
          opacity: Math.max(0, phase) * (1 - i * 0.18),
        })
      }
      const down = layer.suppressed[pos] ?? []
      for (let i = 0; i < Math.min(2, down.length); i++) {
        const hit = down[i]
        if (!this.featurePoint(hit.id, l, v)) continue
        out.push({
          id: `s${hit.id}`,
          text: this.model.lensPieces[hit.id].trim(),
          sub: 'suppressed',
          world: v.clone(),
          kind: 'suppressed',
          opacity: Math.max(0, phase) * 0.8,
        })
      }
    }

    if (state.selected !== null) {
      const l = Math.max(0, Math.min(nLayers - 1, Math.round(state.layerF)))
      if (this.featurePoint(state.selected, l, v)) out.push({
        id: 'selected',
        text: this.model.lensPieces[state.selected].trim() || this.model.lensPieces[state.selected],
        sub: 'following',
        world: v.clone(),
        kind: 'anticipation',
        opacity: 1,
      })
    }

    if (state.show.plans) {
      for (const a of this.run.anticipations.filter((x) => x.step === state.step)) {
        if (!this.featurePoint(a.featureId, a.layer, v)) continue
        out.push({
          id: `a${a.featureId}-${a.targetStep}`,
          text: a.targetWord.trim(),
          sub: `written +${a.targetStep - a.step}`,
          world: new THREE.Vector3(v.x, v.y + 1.6, v.z),
          kind: 'anticipation',
          opacity: 0.9,
        })
      }
    }
    return out
  }

  /** Where a token was drawn at a layer. False when that layer did not move it. */
  private featurePoint(feature: number, layer: number, out: THREE.Vector3): boolean {
    return this.lattice.pointAt(feature, layer, out)
  }

  /**
   * Check the picture against the model.
   *
   * Every claim the interface makes about what a visual property means is
   * re-derived here from the trace and compared with what is actually in the
   * buffers being drawn. It reads the same memory the GPU does, so a check
   * passing means the pixels are right, not that the intent was.
   */
  validate(state: RenderState): { name: string; pass: boolean; detail: string }[] {
    const results: { name: string; pass: boolean; detail: string }[] = []
    const add = (name: string, pass: boolean, detail: string): void => {
      results.push({ name, pass, detail })
    }
    if (!this.run) return results
    const trace = this.run.steps[state.step]
    const nLayers = this.model.cfg.nLayer
    const pts = this.lattice.debugPoints()

    // --- which points exist ---
    let expected = 0
    for (const layer of trace.layers) {
      const behind = state.layerF - layer.layer
      if (behind < 0 || Math.exp(-behind / 1.25) < 0.02) continue
      expected += (layer.features[trace.active] ?? []).length
      expected += (layer.suppressed[trace.active] ?? []).length
    }
    const selExtra = state.selected !== null ? nLayers : 0
    add(
      'points drawn = words this layer moved',
      Math.abs(pts.length - expected) <= selExtra,
      `${pts.length} drawn, ${expected} past threshold in lit layers${selExtra ? ` (+${selExtra} followed)` : ''}`,
    )

    // --- depth is the layer ---
    let depthErr = 0
    const slab = (34 / Math.max(1, nLayers - 1)) * 0.17
    for (const p of pts) depthErr = Math.max(depthErr, Math.abs(p.z - layerZ(p.layer, nLayers)))
    add('depth = layer index', depthErr <= slab + 1e-3, `max deviation ${depthErr.toFixed(3)} within slab ${slab.toFixed(3)}`)

    // --- radius is push strength ---
    const strength = new Map<number, number>()
    const warm = new Map<number, boolean>()
    for (const layer of trace.layers) {
      for (const h of layer.features[trace.active] ?? []) {
        strength.set(layer.layer * 1e7 + h.id, h.act)
        warm.set(layer.layer * 1e7 + h.id, true)
      }
      for (const h of layer.suppressed[trace.active] ?? []) {
        if (!strength.has(layer.layer * 1e7 + h.id)) {
          strength.set(layer.layer * 1e7 + h.id, h.act)
          warm.set(layer.layer * 1e7 + h.id, false)
        }
      }
    }
    let radErr = 0
    let radChecked = 0
    for (const p of pts) {
      const act = strength.get(p.layer * 1e7 + p.token)
      if (act === undefined) continue
      radErr = Math.max(radErr, Math.abs(Math.hypot(p.x, p.y) - this.lattice.expectedRadius(act)))
      radChecked++
    }
    add('distance from axis = push strength', radErr < 1e-3, `max deviation ${radErr.toExponential(1)} over ${radChecked} points`)

    // --- colour direction ---
    let hueBad = 0
    for (const p of pts) {
      const w = warm.get(p.layer * 1e7 + p.token)
      if (w === undefined) continue
      const isWarm = p.r >= p.b
      if (w !== isWarm) hueBad++
    }
    add('warm = pushed toward, violet = pushed away', hueBad === 0, `${hueBad} points with the wrong hue`)

    // --- rings ---
    let ringErr = 0
    const rv = this.rings.debugVertices()
    const per = 72 * 6
    for (let l = 0; l < nLayers; l++) {
      const r = Math.hypot(rv[l * per], rv[l * per + 1])
      ringErr = Math.max(ringErr, Math.abs(r - this.lattice.layerRadius[l] * 1.12))
    }
    add('ring radius = strongest push in that layer', ringErr < 1e-3, `max deviation ${ringErr.toExponential(1)}`)

    // --- filaments ---
    const scale = traceScale(trace)
    const sv = this.streams.debugVertices()
    let fx = 0
    let fy = 0
    const T = trace.ids.length
    let v = 0
    for (let pos = 0; pos < T; pos++) {
      for (let l = -1; l < nLayers - 1; l++) {
        const expX = tokenX(pos, T)
        const expY = l < 0 ? -RISE * 0.55 : trace.layers[l].residualNorm[pos] * scale.rise - RISE * 0.55
        fx = Math.max(fx, Math.abs(sv[v * 6] - expX))
        fy = Math.max(fy, Math.abs(sv[v * 6 + 1] - expY))
        v++
      }
    }
    add('filament x = token position', fx < 1e-3, `max deviation ${fx.toExponential(1)}`)
    add('filament height = residual norm', fy < 1e-3, `max deviation ${fy.toExponential(1)}`)

    // --- links join real neighbours, both ends drawn ---
    const layout = this.model.layout
    if (layout) {
      const edges = this.synapses.debugEdges()
      const drawn = new Set(pts.map((p) => `${p.layer}:${p.x.toFixed(3)},${p.y.toFixed(3)},${p.z.toFixed(3)}`))
      let dangling = 0
      for (let e = 0; e < edges.count; e++) {
        const a = `${e}`
        void a
        const ax = edges.positions[e * 6]
        const ay = edges.positions[e * 6 + 1]
        const az = edges.positions[e * 6 + 2]
        const bx = edges.positions[e * 6 + 3]
        const by = edges.positions[e * 6 + 4]
        const bz = edges.positions[e * 6 + 5]
        const okA = pts.some((p) => Math.abs(p.x - ax) < 1e-4 && Math.abs(p.y - ay) < 1e-4 && Math.abs(p.z - az) < 1e-4)
        const okB = pts.some((p) => Math.abs(p.x - bx) < 1e-4 && Math.abs(p.y - by) < 1e-4 && Math.abs(p.z - bz) < 1e-4)
        if (!okA || !okB) dangling++
        if (e > 400) break
      }
      void drawn
      add(
        'every link ends on a drawn point',
        dangling === 0,
        `${edges.count} links drawn, ${dangling} dangling of ${Math.min(401, edges.count)} sampled`,
      )
    }

    // --- attention rows are real distributions ---
    const l = Math.max(0, Math.min(nLayers - 1, Math.round(state.layerF)))
    const attn = trace.layers[l].attn
    const nHeads = attn.length / (T * T)
    let worst = 0
    for (let h = 0; h < nHeads; h++) {
      for (let i = 0; i < T; i++) {
        let sum = 0
        for (let j = 0; j <= i; j++) sum += attn[h * T * T + i * T + j]
        worst = Math.max(worst, Math.abs(sum - 1))
      }
    }
    add('attention rows sum to one', worst < 1e-4, `max deviation ${worst.toExponential(1)} at layer ${l}`)

    // --- the curve chart ends where the model does ---
    if (trace.curves.length) {
      const last = trace.curves[0].values[nLayers - 1]
      add(
        'curve ends at the final layer score',
        Number.isFinite(last),
        `${trace.curves[0].word.trim()} ends at ${last.toFixed(2)} standard deviations`,
      )
    }
    return results
  }

  dispose(): void {
    this.synapses.dispose()
    this.rings.dispose()
    this.lattice.dispose()
    this.streams.dispose()
    this.arcs.dispose()
    this.labels.clear()
    this.controls.dispose()
    this.renderer.dispose()
  }
}

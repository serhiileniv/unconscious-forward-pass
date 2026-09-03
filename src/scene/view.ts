import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { LM } from '../model/lm'
import { MAX_CONTEXT, type Run } from '../model/run'
import type { StepTrace } from '../model/trace'
import { Arcs } from './arcs'
import { Labels, type LabelSpec } from './labels'
import { Lattice } from './lattice'
import { Rings } from './rings'
import { PALETTE } from './palette'
import { Streams } from './streams'
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
    this.camera.position.set(43.4, 12.8, 31.0)

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
    this.synapses = new Synapses(model, this.lattice.positions, model.lensIds.length)
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

    if (state.step !== this.currentStep) {
      this.currentStep = state.step
      this.streams.setTrace(trace, nLayers)
      this.buildPlans(state.step)
    }

    this.lattice.points.visible = state.show.features
    this.streams.lines.visible = state.show.streams
    this.arcs.lines.visible = state.show.attention
    this.plans.visible = state.show.plans

    this.rings.update(state.layerF, this.layerWork(trace))
    this.synapses.lines.visible = state.show.synapses
    if (state.show.synapses) this.synapses.update(trace, state.layerF, nLayers)
    if (state.show.features) this.lattice.update(trace, state.layerF, state.selected)
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
      this.featurePoint(a.featureId, a.layer, v)
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
        this.featurePoint(hit.id, l, v)
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
        this.featurePoint(hit.id, l, v)
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
      this.featurePoint(state.selected, l, v)
      out.push({
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
        this.featurePoint(a.featureId, a.layer, v)
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

  private featurePoint(feature: number, layer: number, out: THREE.Vector3): void {
    const attr = this.lattice.points.geometry.getAttribute('position') as THREE.BufferAttribute
    const idx = layer * this.model.lensIds.length + feature
    out.set(attr.getX(idx), attr.getY(idx), attr.getZ(idx))
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

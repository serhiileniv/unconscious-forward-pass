import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { Run, Session } from '../model/run'
import { Arcs } from './arcs'
import { Labels, type LabelSpec } from './labels'
import { Lattice, layerZ } from './lattice'
import { Rings } from './rings'
import { PALETTE } from './palette'
import { Streams, streamPoint } from './streams'

export interface RenderState {
  step: number
  layerF: number
  headFilter: number
  show: { features: boolean; attention: boolean; streams: boolean; labels: boolean; plans: boolean }
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
  private readonly labels: Labels
  private readonly plans: THREE.LineSegments
  private readonly session: Session
  private run: Run | null = null
  private currentStep = -1
  private width = 1
  private height = 1

  constructor(canvas: HTMLCanvasElement, labelHost: HTMLElement, session: Session) {
    this.session = session
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, powerPreference: 'high-performance' })
    this.renderer.setClearColor(PALETTE.bg, 1)

    this.scene = new THREE.Scene()
    // Depth cueing. Layers far from the camera dim out, which is what makes the
    // stack read as depth rather than as a flat spray of points.
    this.scene.fog = new THREE.FogExp2(PALETTE.bg, 0.0145)

    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 400)
    this.camera.position.set(19, 8, 43)

    this.controls = new OrbitControls(this.camera, canvas)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.06
    this.controls.minDistance = 10
    this.controls.maxDistance = 130
    this.controls.autoRotate = true
    this.controls.autoRotateSpeed = 0.22
    this.controls.target.set(0, 0, -1)

    this.lattice = new Lattice(session)
    this.streams = new Streams(session.cfg.maxT, session.cfg.nLayers)
    this.arcs = new Arcs()
    this.rings = new Rings(session.cfg.nLayers)
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

    this.scene.add(this.rings.lines, this.lattice.points, this.streams.lines, this.arcs.lines, this.plans)
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
    const nLayers = this.session.cfg.nLayers

    if (state.step !== this.currentStep) {
      this.currentStep = state.step
      this.streams.setTrace(trace, nLayers)
      this.buildPlans(state.step)
    }

    this.lattice.points.visible = state.show.features
    this.streams.lines.visible = state.show.streams
    this.arcs.lines.visible = state.show.attention
    this.plans.visible = state.show.plans

    this.rings.update(state.layerF)
    if (state.show.features) this.lattice.update(trace, state.layerF)
    if (state.show.streams) this.streams.update(trace, state.layerF)
    if (state.show.attention) this.arcs.update(trace, state.layerF, nLayers, state.headFilter)

    this.labels.update(state.show.labels ? this.collectLabels(state) : [], this.camera, this.width, this.height)

    this.controls.update()
    this.renderer.render(this.scene, this.camera)
  }

  /** Long amber threads: a feature awake now whose direction points at a word the run went on to emit. */
  private buildPlans(step: number): void {
    if (!this.run) return
    const nLayers = this.session.cfg.nLayers
    const links = this.run.anticipations.filter((a) => a.step === step)
    const pos: number[] = []
    const v = new THREE.Vector3()
    const exitZ = layerZ(nLayers - 1, nLayers) + 6
    for (const a of links) {
      const trace = this.run.steps[step]
      streamPoint(trace, a.layer, Math.min(a.pos, trace.ids.length - 1), nLayers, v)
      const t = (a.targetStep % 5) - 2
      pos.push(v.x, v.y, v.z, t * 2.6, 7.5, exitZ)
    }
    this.plans.geometry.dispose()
    const g = new THREE.BufferGeometry()
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3))
    this.plans.geometry = g
  }

  private collectLabels(state: RenderState): LabelSpec[] {
    if (!this.run) return []
    const trace = this.run.steps[state.step]
    const nLayers = this.session.cfg.nLayers
    const l = Math.round(state.layerF)
    const out: LabelSpec[] = []
    const v = new THREE.Vector3()

    if (l >= 0 && l < nLayers) {
      const layer = trace.layers[l]
      const pos = trace.ids.length - 1
      const hits = layer.features[pos] ?? []
      // Labels must not strobe with the wavefront; hold a legible floor.
      const phase = 0.42 + 0.58 * (1 - Math.min(1, Math.abs(state.layerF - l) * 0.9))
      for (let i = 0; i < Math.min(3, hits.length); i++) {
        const hit = hits[i]
        this.featurePoint(hit.id, l, v)
        out.push({
          id: `f${hit.id}`,
          text: this.session.model.dictLabels[hit.id],
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
          text: this.session.model.dictLabels[hit.id],
          sub: 'suppressed',
          world: v.clone(),
          kind: 'suppressed',
          opacity: Math.max(0, phase) * 0.8,
        })
      }
    }

    if (state.show.plans) {
      const exitZ = layerZ(nLayers - 1, nLayers) + 6
      for (const a of this.run.anticipations.filter((x) => x.step === state.step)) {
        const t = (a.targetStep % 5) - 2
        out.push({
          id: `a${a.featureId}-${a.targetStep}`,
          text: a.targetWord,
          sub: `+${a.targetStep - a.step}`,
          world: new THREE.Vector3(t * 2.6, 7.5, exitZ),
          kind: 'anticipation',
          opacity: 0.9,
        })
      }
    }
    return out
  }

  private featurePoint(feature: number, layer: number, out: THREE.Vector3): void {
    const attr = this.lattice.points.geometry.getAttribute('position') as THREE.BufferAttribute
    const idx = layer * this.session.cfg.nFeatures + feature
    out.set(attr.getX(idx), attr.getY(idx), attr.getZ(idx))
  }

  dispose(): void {
    this.rings.dispose()
    this.lattice.dispose()
    this.streams.dispose()
    this.arcs.dispose()
    this.labels.clear()
    this.controls.dispose()
    this.renderer.dispose()
  }
}

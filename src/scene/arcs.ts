import * as THREE from 'three'
import type { StepTrace } from '../model/trace'
import { streamPoint } from './streams'
import { PALETTE } from './palette'

const MAX_ARCS = 110
const SEGS = 16

/**
 * Attention, drawn only where it is actually happening.
 *
 * Every arc is one query position reaching back to one earlier position at the
 * layer the wavefront is currently crossing. They are not persistent objects and
 * they are deliberately not drawn as persistent objects: a head reaches back,
 * moves information, and is gone before the next layer starts. What survives is
 * only what it wrote into the stream.
 */
export class Arcs {
  readonly lines: THREE.LineSegments
  private geom: THREE.BufferGeometry
  private positions: Float32Array
  private colors: Float32Array

  constructor() {
    this.positions = new Float32Array(MAX_ARCS * SEGS * 6)
    this.colors = new Float32Array(MAX_ARCS * SEGS * 6)
    this.geom = new THREE.BufferGeometry()
    this.geom.setAttribute('position', new THREE.BufferAttribute(this.positions, 3))
    this.geom.setAttribute('color', new THREE.BufferAttribute(this.colors, 3))
    this.geom.setDrawRange(0, 0)
    this.lines = new THREE.LineSegments(
      this.geom,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    )
    this.lines.frustumCulled = false
  }

  /** `layerF` is fractional; arcs fade in and out as the front crosses a layer. */
  update(trace: StepTrace, layerF: number, nLayers: number, headFilter: number): void {
    const l = Math.round(layerF)
    if (l < 0 || l >= nLayers) {
      this.geom.setDrawRange(0, 0)
      return
    }
    // Brightest exactly on a layer, dark halfway between two — attention is an
    // event, not a state.
    const phase = 0.25 + 0.75 * (1 - Math.min(1, Math.abs(layerF - l) * 2))
    if (phase <= 0.01) {
      this.geom.setDrawRange(0, 0)
      return
    }

    const layer = trace.layers[l]
    const T = trace.ids.length
    const nHeads = layer.attn.length / (T * T)

    type Link = { i: number; j: number; w: number; head: number }
    const links: Link[] = []
    for (let h = 0; h < nHeads; h++) {
      if (headFilter >= 0 && h !== headFilter) continue
      for (let i = 1; i < T; i++) {
        const off = h * T * T + i * T
        for (let j = 0; j < i; j++) {
          const w = layer.attn[off + j]
          if (w > 0.12) links.push({ i, j, w, head: h })
        }
      }
    }
    links.sort((a, b) => b.w - a.w)
    const use = links.slice(0, MAX_ARCS)

    const from = new THREE.Vector3()
    const to = new THREE.Vector3()
    const ctrl = new THREE.Vector3()
    const p = new THREE.Vector3()
    const q = new THREE.Vector3()
    const base = new THREE.Color(PALETTE.attention)

    let v = 0
    for (const link of use) {
      streamPoint(trace, l, link.j, nLayers, from)
      streamPoint(trace, l, link.i, nLayers, to)
      // Bulge below the streams so arcs stay legible against the feature cloud.
      ctrl.copy(from).add(to).multiplyScalar(0.5)
      ctrl.y -= 0.9 + Math.abs(link.i - link.j) * 0.2
      ctrl.z -= 0.4

      const a = Math.min(1, 0.14 + link.w * 1.2) * phase * 0.32
      // Per-head hue drift, so you can see heads doing different jobs at once.
      const hueShift = (link.head / Math.max(1, nHeads)) * 0.16 - 0.08
      const col = base.clone().offsetHSL(hueShift, 0.1, 0)

      for (let s = 0; s < SEGS; s++) {
        const t0 = s / SEGS
        const t1 = (s + 1) / SEGS
        quad(from, ctrl, to, t0, p)
        quad(from, ctrl, to, t1, q)
        this.positions.set([p.x, p.y, p.z, q.x, q.y, q.z], v * 6)
        // Taper toward the source so direction reads without arrowheads.
        const f0 = a * (0.25 + 0.75 * t0)
        const f1 = a * (0.25 + 0.75 * t1)
        this.colors.set(
          [col.r * f0, col.g * f0, col.b * f0, col.r * f1, col.g * f1, col.b * f1],
          v * 6,
        )
        v++
      }
    }

    this.geom.setDrawRange(0, v * 2)
    this.geom.getAttribute('position').needsUpdate = true
    this.geom.getAttribute('color').needsUpdate = true
  }

  dispose(): void {
    this.geom.dispose()
    ;(this.lines.material as THREE.Material).dispose()
  }
}

function quad(a: THREE.Vector3, c: THREE.Vector3, b: THREE.Vector3, t: number, out: THREE.Vector3): THREE.Vector3 {
  const u = 1 - t
  out.set(
    u * u * a.x + 2 * u * t * c.x + t * t * b.x,
    u * u * a.y + 2 * u * t * c.y + t * t * b.y,
    u * u * a.z + 2 * u * t * c.z + t * t * b.z,
  )
  return out
}

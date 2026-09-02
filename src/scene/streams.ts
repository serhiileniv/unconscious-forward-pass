import * as THREE from 'three'
import type { StepTrace } from '../model/trace'
import { layerGap, layerZ, tokenX } from './lattice'
import { PALETTE } from './palette'

/** How far the residual's real 3D projection is allowed to pull its filament. */
const WANDER = 1.5

/**
 * Residual norms grow by more than an order of magnitude from the first layer to
 * the last, and differ again between models, so both the sideways wander and the
 * per-layer brightness are scaled against the run's own maximum. Relative
 * structure is preserved; only the absolute magnitude is fitted to the scene.
 */
function traceScale(trace: StepTrace): { wander: number; write: number } {
  let projMax = 1e-6
  let writeMax = 1e-6
  for (const layer of trace.layers) {
    for (let i = 0; i < layer.proj.length; i++) projMax = Math.max(projMax, Math.abs(layer.proj[i]))
    for (let i = 0; i < layer.writeNorm.length; i++) writeMax = Math.max(writeMax, layer.writeNorm[i])
  }
  return { wander: WANDER / projMax, write: 1 / writeMax }
}

/**
 * Where token `pos` sits at layer `l`. The lateral offset is the actual projected
 * residual vector, so a filament visibly bends when a layer writes something large
 * into it. A straight filament means that token was barely touched.
 */
export function streamPoint(
  trace: StepTrace,
  l: number,
  pos: number,
  nLayers: number,
  out: THREE.Vector3,
  wander = traceScale(trace).wander,
): THREE.Vector3 {
  const T = trace.ids.length
  if (l < 0) return out.set(tokenX(pos, T), 0, layerZ(0, nLayers) - layerGap(nLayers) * 0.9)
  const p = trace.layers[l].proj
  return out.set(
    tokenX(pos, T) + p[pos * 3] * wander,
    p[pos * 3 + 1] * wander,
    layerZ(l, nLayers) + p[pos * 3 + 2] * wander * 0.1,
  )
}

/**
 * The residual stream — one continuous filament per token, running the full depth.
 *
 * This is the spine the whole architecture is organised around: every layer reads
 * it, adds something, and puts it back. Nothing is ever removed, only buried.
 */
export class Streams {
  readonly lines: THREE.LineSegments
  private geom: THREE.BufferGeometry
  private positions: Float32Array
  private colors: Float32Array
  private nLayers = 0
  private T = 0

  constructor(maxT: number, nLayers: number) {
    const maxSegs = maxT * (nLayers + 1)
    this.positions = new Float32Array(maxSegs * 6)
    this.colors = new Float32Array(maxSegs * 6)
    this.geom = new THREE.BufferGeometry()
    this.geom.setAttribute('position', new THREE.BufferAttribute(this.positions, 3))
    this.geom.setAttribute('color', new THREE.BufferAttribute(this.colors, 3))
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

  private writeScale = 1

  setTrace(trace: StepTrace, nLayers: number): void {
    this.nLayers = nLayers
    this.T = trace.ids.length
    const scale = traceScale(trace)
    this.writeScale = scale.write
    const a = new THREE.Vector3()
    const b = new THREE.Vector3()
    let s = 0
    for (let pos = 0; pos < this.T; pos++) {
      for (let l = -1; l < nLayers - 1; l++) {
        streamPoint(trace, l, pos, nLayers, a, scale.wander)
        streamPoint(trace, l + 1, pos, nLayers, b, scale.wander)
        this.positions.set([a.x, a.y, a.z, b.x, b.y, b.z], s * 6)
        s++
      }
    }
    this.geom.setDrawRange(0, s * 2)
    this.geom.getAttribute('position').needsUpdate = true
    this.geom.computeBoundingSphere()
  }

  update(trace: StepTrace, layerF: number): void {
    const cold = new THREE.Color(PALETTE.stream)
    const hot = new THREE.Color(PALETTE.streamHot)
    const c = this.colors
    let s = 0
    for (let pos = 0; pos < this.T; pos++) {
      for (let l = -1; l < this.nLayers - 1; l++) {
        // A segment is only lit once the wavefront has reached it.
        const reached = Math.min(1, Math.max(0, layerF - l))
        const write = l >= 0 ? Math.min(1, trace.layers[l].writeNorm[pos] * this.writeScale) : 0.35
        const near = Math.exp(-Math.abs(layerF - l) / 3.2)
        const k = reached * (0.1 + 0.9 * near)
        const r = (cold.r + (hot.r - cold.r) * write) * k
        const g = (cold.g + (hot.g - cold.g) * write) * k
        const bl = (cold.b + (hot.b - cold.b) * write) * k
        c.set([r, g, bl, r, g, bl], s * 6)
        s++
      }
    }
    this.geom.getAttribute('color').needsUpdate = true
  }

  dispose(): void {
    this.geom.dispose()
    ;(this.lines.material as THREE.Material).dispose()
  }
}

import * as THREE from 'three'
import type { StepTrace } from '../model/trace'
import { layerGap, layerZ, tokenX } from './lattice'
import { PALETTE } from './palette'

/** Vertical range used to show how large the residual has grown. */
const RISE = 5.5

/**
 * Every coordinate of a filament is an exact quantity.
 *
 * An earlier version displaced filaments sideways by a 3D projection of the
 * residual vector, which looked like motion but was a shadow of 768 dimensions
 * and carried almost nothing. Height is now the norm of the residual itself,
 * scaled against the largest in the pass: the stream visibly swells as it is
 * written into, which is a real and measurable thing that it does. Depth is the
 * layer, horizontal is the token's place in the sentence, and brightness is how
 * much the layer wrote. Nothing here is a projection.
 */
function traceScale(trace: StepTrace): { rise: number; write: number } {
  let normMax = 1e-6
  let writeMax = 1e-6
  for (const layer of trace.layers) {
    for (let i = 0; i < layer.residualNorm.length; i++) normMax = Math.max(normMax, layer.residualNorm[i])
    for (let i = 0; i < layer.writeNorm.length; i++) writeMax = Math.max(writeMax, layer.writeNorm[i])
  }
  return { rise: RISE / normMax, write: 1 / writeMax }
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
  rise = traceScale(trace).rise,
): THREE.Vector3 {
  const T = trace.ids.length
  if (l < 0) return out.set(tokenX(pos, T), -RISE * 0.55, layerZ(0, nLayers) - layerGap(nLayers) * 0.9)
  return out.set(
    tokenX(pos, T),
    trace.layers[l].residualNorm[pos] * rise - RISE * 0.55,
    layerZ(l, nLayers),
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
        streamPoint(trace, l, pos, nLayers, a, scale.rise)
        streamPoint(trace, l + 1, pos, nLayers, b, scale.rise)
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

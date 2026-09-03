import * as THREE from 'three'
import type { LM } from '../model/lm'
import { Z_PUSH } from '../model/run'
import type { StepTrace } from '../model/trace'
import { PALETTE } from './palette'

/**
 * The stack occupies the same depth whatever the model's layer count, so a
 * 24-layer model frames exactly like a 12-layer one.
 */
export const STACK_DEPTH = 34
export const CLOUD_RADIUS = 13
export const TOKEN_SPAN = 21
/** Radius a token sits at when it only just cleared the drawing threshold. */
const R_MIN = 1.4
const R_MAX = CLOUD_RADIUS * 0.78
/** Ceiling on drawn points; a busy layer moves twenty thousand scores hard. */
const CAPACITY = 200_000

export function layerGap(nLayers: number): number {
  return STACK_DEPTH / Math.max(1, nLayers - 1)
}

export function layerZ(layer: number, nLayers: number): number {
  return (layer - (nLayers - 1) / 2) * layerGap(nLayers)
}

export function tokenX(pos: number, T: number): number {
  return (pos - (T - 1) / 2) * (TOKEN_SPAN / Math.max(1, T - 1))
}

/**
 * Every drawn point is a word this layer actually moved.
 *
 * An earlier version drew all 50,257 tokens at every layer, most of them dim,
 * as a substrate. It looked like space and carried nothing: a dot meant only
 * "this token exists". Those are gone. What is left is one point per word whose
 * score this layer pushed past the threshold, so the number of points on screen
 * is itself a measurement.
 *
 * Each coordinate carries something, and the split between them is not
 * arbitrary — it follows what the layout was measured to preserve:
 *
 *   depth    the layer. Exact.
 *   radius   how hard this layer moved that word's score, against the largest
 *            push anywhere in the pass. Exact.
 *   angle    direction in the semantic map. Measured at 54 degrees to a true
 *            neighbour against 89.5 for a random token, where chance is 90, so
 *            related words genuinely share a direction.
 *   colour   which way it was pushed. Exact.
 *
 * Radius used to come from that same map and was worth nothing — true
 * neighbours sat no closer than random pairs, a ratio of 1.07. Direction
 * survived the drop to three dimensions; distance did not. So direction is kept
 * and distance is given to a quantity that means something.
 */
export class Lattice {
  readonly points: THREE.Points
  /** Outer radius of the drawn field, for sizing the rings. */
  readonly radius = R_MAX
  /**
   * Radius of the strongest push at each layer, filled during update. The rings
   * are drawn to these, so the corridor narrows where a layer did little and
   * widens where it pushed hard — the shape of the tunnel is a measurement.
   */
  readonly layerRadius: Float32Array

  private readonly geom: THREE.BufferGeometry
  private readonly positions: Float32Array
  private readonly colors: Float32Array
  private readonly sizes: Float32Array
  /** Unit direction per token, taken from the semantic map. */
  private readonly dir: Float32Array
  private readonly nTokens: number
  /** Where each token was last drawn, so links can find its ends. */
  private readonly drawnAt = new Map<number, number>()
  private lastKey = ''
  private peak = 1

  constructor(model: LM) {
    this.nTokens = model.lensIds.length
    this.dir = new Float32Array(this.nTokens * 3)

    const src = model.layout ? model.layout.pos : model.lensPos
    const stride = model.layout ? 1 : 0
    const centre = [0, 0, 0]
    for (let f = 0; f < this.nTokens; f++) {
      const at = (stride ? model.lensIds[f] : f) * 3
      for (let c = 0; c < 3; c++) centre[c] += src[at + c] / this.nTokens
    }
    for (let f = 0; f < this.nTokens; f++) {
      const at = (stride ? model.lensIds[f] : f) * 3
      const x = src[at] - centre[0]
      const y = src[at + 1] - centre[1]
      const z = src[at + 2] - centre[2]
      // Flatten to the layer plane, keeping a little of the third component as
      // depth texture so the plane does not read as a flat disc.
      const n = Math.hypot(x, y) || 1e-9
      this.dir[f * 3] = x / n
      this.dir[f * 3 + 1] = y / n
      this.dir[f * 3 + 2] = Math.max(-1, Math.min(1, z / (Math.hypot(x, y, z) || 1)))
    }

    this.layerRadius = new Float32Array(model.cfg.nLayer)
    this.positions = new Float32Array(CAPACITY * 3)
    this.colors = new Float32Array(CAPACITY * 3)
    this.sizes = new Float32Array(CAPACITY)

    this.geom = new THREE.BufferGeometry()
    this.geom.setAttribute('position', new THREE.BufferAttribute(this.positions, 3))
    this.geom.setAttribute('color', new THREE.BufferAttribute(this.colors, 3))
    this.geom.setAttribute('size', new THREE.BufferAttribute(this.sizes, 1))
    this.geom.setDrawRange(0, 0)

    const material = new THREE.ShaderMaterial({
      uniforms: { uScale: { value: 300 } },
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        uniform float uScale;
        void main() {
          vColor = color;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * uScale / max(-mv.z, 1.0);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        void main() {
          vec2 d = gl_PointCoord - vec2(0.5);
          float r = dot(d, d) * 4.0;
          if (r > 1.0) discard;
          float core = pow(1.0 - r, 2.2);
          float halo = pow(1.0 - r, 1.6) * 0.10;
          gl_FragColor = vec4(vColor * (core + halo), 1.0);
        }
      `,
      vertexColors: true,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })

    this.points = new THREE.Points(this.geom, material)
    this.points.frustumCulled = false
  }

  /** Where a token was drawn at a layer, if it was drawn at all. */
  pointAt(token: number, layer: number, out: THREE.Vector3): boolean {
    const at = this.drawnAt.get(layer * this.nTokens + token)
    if (at === undefined) return false
    out.set(this.positions[at * 3], this.positions[at * 3 + 1], this.positions[at * 3 + 2])
    return true
  }

  /**
   * Radius is proportional to how far past the drawing threshold the push went,
   * against the strongest push anywhere in the pass. Linear, so the distance a
   * point sits from the axis can be read directly as strength. Most pushes are
   * barely over the line, which is why the field is dense at the centre and thin
   * at the rim: that shape is the distribution, not a styling choice.
   */
  private radiusFor(strength: number): number {
    const t = Math.min(1, Math.max(0, (strength - Z_PUSH) / Math.max(0.1, this.peak - Z_PUSH)))
    return R_MIN + t * (R_MAX - R_MIN)
  }

  update(trace: StepTrace, layerF: number, selected: number | null, tail = 1.25): void {
    const key = `${Math.round(layerF * 6)}:${selected ?? -1}:${trace.active}`
    if (key === this.lastKey) return
    this.lastKey = key

    // One scale for the whole pass, so a layer that pushes weakly draws a small
    // field and a layer that pushes hard draws a wide one.
    this.peak = 1e-6
    for (const layer of trace.layers) {
      for (const h of layer.features[trace.active] ?? []) if (h.act > this.peak) this.peak = h.act
      for (const h of layer.suppressed[trace.active] ?? []) if (h.act > this.peak) this.peak = h.act
    }

    const nLayers = trace.layers.length
    for (let l = 0; l < nLayers; l++) {
      let strongest = Z_PUSH
      for (const h of trace.layers[l].features[trace.active] ?? []) if (h.act > strongest) strongest = h.act
      for (const h of trace.layers[l].suppressed[trace.active] ?? []) if (h.act > strongest) strongest = h.act
      this.layerRadius[l] = this.radiusFor(strongest)
    }

    const p = this.positions
    const c = this.colors
    const s = this.sizes
    this.drawnAt.clear()
    let n = 0

    const place = (token: number, layer: number, strength: number, warm: boolean, w: number): void => {
      if (n >= CAPACITY) return
      const r = this.radiusFor(strength)
      const d = token * 3
      p[n * 3] = this.dir[d] * r
      p[n * 3 + 1] = this.dir[d + 1] * r
      p[n * 3 + 2] = layerZ(layer, nLayers) + this.dir[d + 2] * layerGap(nLayers) * 0.16
      const a = Math.min(1, (strength - Z_PUSH) * 0.5 + 0.35) * w
      const tint = warm ? PALETTE.active : PALETTE.suppressed
      const heat = warm ? Math.min(1, Math.max(0, strength - 2.6) * 0.5) : 0
      const g = a * (warm ? 0.95 : 0.7)
      c[n * 3] = (tint[0] * (1 - heat) + PALETTE.peak[0] * heat) * g
      c[n * 3 + 1] = (tint[1] * (1 - heat) + PALETTE.peak[1] * heat) * g
      c[n * 3 + 2] = (tint[2] * (1 - heat) + PALETTE.peak[2] * heat) * g
      s[n] = 0.38 + a * 1.05
      this.drawnAt.set(layer * this.nTokens + token, n)
      n++
    }

    for (const layer of trace.layers) {
      const behind = layerF - layer.layer
      if (behind < 0) continue
      const w = Math.exp(-behind / tail)
      if (w < 0.02) continue
      // Only the position this pass is computing. Earlier positions were settled
      // by earlier passes and their filaments already say so.
      for (const hit of layer.features[trace.active] ?? []) place(hit.id, layer.layer, hit.act, true, w)
      for (const hit of layer.suppressed[trace.active] ?? []) place(hit.id, layer.layer, hit.act, false, w)
    }

    // A followed word is drawn at every layer, at the radius its push earned
    // there, so its whole path in and out of contention is visible at once.
    if (selected !== null && selected >= 0 && selected < this.nTokens) {
      for (const layer of trace.layers) {
        const hit =
          (layer.features[trace.active] ?? []).find((h) => h.id === selected) ??
          (layer.suppressed[trace.active] ?? []).find((h) => h.id === selected)
        if (n >= CAPACITY) break
        const r = this.radiusFor(hit ? hit.act : 0)
        const d = selected * 3
        const behind = layerF - layer.layer
        const w = behind < 0 ? 0.2 : 0.4 + 0.6 * Math.exp(-behind / tail)
        p[n * 3] = this.dir[d] * r
        p[n * 3 + 1] = this.dir[d + 1] * r
        p[n * 3 + 2] = layerZ(layer.layer, nLayers) + this.dir[d + 2] * layerGap(nLayers) * 0.16
        c[n * 3] = PALETTE.peak[0] * w
        c[n * 3 + 1] = PALETTE.peak[1] * w
        c[n * 3 + 2] = PALETTE.peak[2] * w
        s[n] = 1.0 + w * 1.4
        this.drawnAt.set(layer.layer * this.nTokens + selected, n)
        n++
      }
    }

    this.geom.setDrawRange(0, n)
    this.geom.getAttribute('position').needsUpdate = true
    this.geom.getAttribute('color').needsUpdate = true
    this.geom.getAttribute('size').needsUpdate = true
  }

  dispose(): void {
    this.geom.dispose()
    ;(this.points.material as THREE.Material).dispose()
  }
}

import * as THREE from 'three'
import type { LM } from '../model/lm'
import type { StepTrace } from '../model/trace'
import { PALETTE } from './palette'

/**
 * The stack occupies the same depth whatever the model's layer count, so a
 * 24-layer model frames exactly like a 12-layer one and the camera never has to
 * be retuned per checkpoint.
 */
export const STACK_DEPTH = 34

export function layerGap(nLayers: number): number {
  return STACK_DEPTH / Math.max(1, nLayers - 1)
}
/**
 * Tokens occupy a fixed width however many there are, for the same reason the
 * stack occupies a fixed depth: a 26-token context would otherwise sprawl wider
 * than the model is deep and pull the filaments out of frame.
 */
export const TOKEN_SPAN = 21
export const CLOUD_RADIUS = 13
/** Size of a point that no layer has moved. */
const DORMANT_SIZE = 0.13

export function layerZ(layer: number, nLayers: number): number {
  return (layer - (nLayers - 1) / 2) * layerGap(nLayers)
}

export function tokenX(pos: number, T: number): number {
  return (pos - (T - 1) / 2) * (TOKEN_SPAN / Math.max(1, T - 1))
}

/**
 * The token cloud: one point per whole-word token, drawn once per layer.
 *
 * A point's position is not decorative. It is that token's real row of GPT-2's
 * unembedding matrix, put through the same fixed 3D projection the residual
 * stream uses, so two points sit close together exactly when GPT-2 puts those
 * two tokens close together in its 768-dimensional space. The cloud's shape is
 * the model's own geometry, not an arrangement chosen to look good.
 *
 * A point lights when the logit lens at that layer is leaning toward its token.
 */
export class Lattice {
  readonly points: THREE.Points
  /** The drawn position of every point, indexed layer * nFeatures + token. */
  readonly positions: Float32Array
  /** 95th-percentile radius of the cloud, so other objects can be sized to it. */
  readonly radius: number
  private readonly colors: Float32Array
  private readonly sizes: Float32Array
  private readonly nFeatures: number
  private readonly geom: THREE.BufferGeometry
  /**
   * Indices written last frame.
   *
   * With every token drawn there are hundreds of thousands of points, and
   * clearing all of them each frame costs more than everything else in the
   * renderer combined. Almost all stay dormant, so only the ones actually
   * touched are reset.
   */
  private touched = new Int32Array(1 << 16)
  private touchedCount = 0
  private lastKey = -1

  constructor(model: LM) {
    const nFeatures = model.lensIds.length
    const nLayers = model.cfg.nLayer
    this.nFeatures = nFeatures
    const total = nFeatures * nLayers

    // Prefer the precomputed semantic map. Its positions come from a spectral
    // embedding of the model's own nearest-neighbour graph, which places true
    // neighbours at about half the distance of random pairs — weak, but far
    // better than the linear projection it replaces, and the fallback when the
    // map has not been generated.
    const flat = new Float32Array(nFeatures * 3)
    if (model.layout) {
      for (let f = 0; f < nFeatures; f++) {
        const src = model.lensIds[f] * 3
        flat[f * 3] = model.layout.pos[src]
        flat[f * 3 + 1] = model.layout.pos[src + 1]
        flat[f * 3 + 2] = model.layout.pos[src + 2]
      }
    } else {
      flat.set(model.lensPos.subarray(0, nFeatures * 3))
    }
    let sd = 0
    for (let i = 0; i < flat.length; i++) sd += flat[i] * flat[i]
    sd = Math.sqrt(sd / flat.length) || 1

    const positions = new Float32Array(total * 3)
    this.positions = positions
    this.colors = new Float32Array(total * 3)
    this.sizes = new Float32Array(total)
    const [d0, d1, d2] = PALETTE.dormant
    for (let i = 0; i < total; i++) {
      this.colors[i * 3] = d0
      this.colors[i * 3 + 1] = d1
      this.colors[i * 3 + 2] = d2
      this.sizes[i] = DORMANT_SIZE
    }

    // Different models put their unembedding rows at wildly different scales, so
    // fit the cloud to a known radius rather than to any absolute magnitude.
    // Ordering and angle are preserved; only the radial scale is fitted.
    // Radius is scaled linearly and nothing else. An earlier r^0.62 warp filled
    // the disc more evenly and measurably degraded distance fidelity, from
    // r = 0.345 to 0.292, so it is gone: the only transform left is uniform.
    const radii = new Float32Array(nFeatures)
    for (let f = 0; f < nFeatures; f++) {
      radii[f] = Math.hypot(flat[f * 3] / sd, flat[f * 3 + 1] / sd) || 1e-6
    }
    const sorted = Float32Array.from(radii).sort()
    const fit = (CLOUD_RADIUS * 0.78) / (sorted[Math.floor(sorted.length * 0.95)] || 1)
    this.radius = CLOUD_RADIUS * 0.78

    for (let f = 0; f < nFeatures; f++) {
      let px = flat[f * 3] / sd
      let py = flat[f * 3 + 1] / sd
      const pz = flat[f * 3 + 2] / sd
      const r = Math.hypot(px, py) || 1e-6
      const rr = radii[f] * fit
      px = (px / r) * rr
      py = (py / r) * rr
      for (let l = 0; l < nLayers; l++) {
        const idx = l * nFeatures + f
        positions[idx * 3] = px
        positions[idx * 3 + 1] = py
        positions[idx * 3 + 2] = layerZ(l, nLayers) + pz * layerGap(nLayers) * 0.16
      }
    }

    this.geom = new THREE.BufferGeometry()
    this.geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    this.geom.setAttribute('color', new THREE.BufferAttribute(this.colors, 3))
    this.geom.setAttribute('size', new THREE.BufferAttribute(this.sizes, 1))

    const material = new THREE.ShaderMaterial({
      uniforms: { uScale: { value: 240 } },
      vertexShader: `
        attribute float size;
        varying vec3 vColor;
        varying float vBright;
        uniform float uScale;
        void main() {
          vColor = color;
          vBright = size;
          vec4 mv = modelViewMatrix * vec4(position, 1.0);
          gl_PointSize = size * uScale / max(-mv.z, 1.0);
          gl_Position = projectionMatrix * mv;
        }
      `,
      fragmentShader: `
        varying vec3 vColor;
        varying float vBright;
        void main() {
          vec2 d = gl_PointCoord - vec2(0.5);
          float r = dot(d, d) * 4.0;
          if (r > 1.0) discard;
          // Tight core with a wide soft halo, so bright points bloom and dim
          // points stay as fine grain rather than mush.
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

  /**
   * Repaint for one instant of the sweep.
   *
   * `layerF` is the wavefront's fractional depth. Layers ahead of it are unlit —
   * they have not happened yet. Layers behind it glow with an exponential tail,
   * which is the visual claim that a forward pass leaves a fading trace and then
   * nothing.
   */
  update(trace: StepTrace, layerF: number, tail = 1.25): void {
    // Repainting at full frame rate re-uploads the whole buffer for a change the
    // eye cannot resolve; sixths of a layer are indistinguishable in motion.
    const key = Math.round(layerF * 6)
    if (key === this.lastKey) return
    this.lastKey = key

    const c = this.colors
    const s = this.sizes
    const [dr, dg, db] = PALETTE.dormant
    for (let i = 0; i < this.touchedCount; i++) {
      const idx = this.touched[i]
      c[idx * 3] = dr
      c[idx * 3 + 1] = dg
      c[idx * 3 + 2] = db
      s[idx] = DORMANT_SIZE
    }
    this.touchedCount = 0
    const mark = (idx: number): void => {
      if (this.touchedCount < this.touched.length) this.touched[this.touchedCount++] = idx
    }

    for (const layer of trace.layers) {
      const behind = layerF - layer.layer
      if (behind < 0) continue
      const w = Math.exp(-behind / tail)
      if (w < 0.02) continue
      const base = layer.layer * this.nFeatures

      for (const hits of layer.suppressed) {
        for (const hit of hits) {
          const i = (base + hit.id) * 3
          const a = Math.min(1, (hit.act - 1.4) * 0.45) * w
          const g = a * 0.20
          c[i] += PALETTE.suppressed[0] * g
          c[i + 1] += PALETTE.suppressed[1] * g
          c[i + 2] += PALETTE.suppressed[2] * g
          s[base + hit.id] = Math.max(s[base + hit.id], 0.20 + a * 0.30)
          mark(base + hit.id)
        }
      }

      for (const hits of layer.features) {
        for (const hit of hits) {
          const i = (base + hit.id) * 3
          const a = Math.min(1, (hit.act - 1.4) * 0.45) * w
          // Strong activations run toward amber; ordinary ones stay bone white.
          const heat = Math.min(1, Math.max(0, hit.act - 2.6) * 0.5)
          const g = a * 0.21
          c[i] += (PALETTE.active[0] * (1 - heat) + PALETTE.peak[0] * heat) * g
          c[i + 1] += (PALETTE.active[1] * (1 - heat) + PALETTE.peak[1] * heat) * g
          c[i + 2] += (PALETTE.active[2] * (1 - heat) + PALETTE.peak[2] * heat) * g
          s[base + hit.id] = Math.max(s[base + hit.id], 0.22 + a * 0.42)
          mark(base + hit.id)
        }
      }
    }

    this.geom.getAttribute('color').needsUpdate = true
    this.geom.getAttribute('size').needsUpdate = true
  }

  dispose(): void {
    this.geom.dispose()
    ;(this.points.material as THREE.Material).dispose()
  }
}

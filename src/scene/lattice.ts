import * as THREE from 'three'
import type { Session } from '../model/run'
import type { StepTrace } from '../model/transformer'
import { PALETTE } from './palette'

export const LAYER_GAP = 3.7
export const TOKEN_GAP = 1.7
export const CLOUD_RADIUS = 13

export function layerZ(layer: number, nLayers: number): number {
  return (layer - (nLayers - 1) / 2) * LAYER_GAP
}

export function tokenX(pos: number, T: number): number {
  return (pos - (T - 1) / 2) * TOKEN_GAP
}

/**
 * The feature cloud: every dictionary direction, drawn once per layer.
 *
 * A feature's position is not decorative. It is its actual direction in residual
 * space, put through the same fixed 3D projection the residual stream uses, so
 * two points sit close together exactly when the two directions are close in the
 * real 96-dimensional space. The cloud's shape is the model's geometry.
 */
export class Lattice {
  readonly points: THREE.Points
  private readonly colors: Float32Array
  private readonly sizes: Float32Array
  private readonly nFeatures: number
  private readonly geom: THREE.BufferGeometry

  constructor(session: Session) {
    const { model, cfg, projector } = session
    this.nFeatures = cfg.nFeatures
    const total = cfg.nFeatures * cfg.nLayers

    // Project every dictionary row to 3D once.
    const flat = new Float32Array(cfg.nFeatures * 3)
    for (let f = 0; f < cfg.nFeatures; f++) {
      for (let c = 0; c < 3; c++) {
        let s = 0
        for (let i = 0; i < cfg.dModel; i++) s += model.dict[f * cfg.dModel + i] * projector[i * 3 + c]
        flat[f * 3 + c] = s
      }
    }
    let sd = 0
    for (let i = 0; i < flat.length; i++) sd += flat[i] * flat[i]
    sd = Math.sqrt(sd / flat.length) || 1

    const positions = new Float32Array(total * 3)
    this.colors = new Float32Array(total * 3)
    this.sizes = new Float32Array(total)

    for (let f = 0; f < cfg.nFeatures; f++) {
      let px = flat[f * 3] / sd
      let py = flat[f * 3 + 1] / sd
      const pz = flat[f * 3 + 2] / sd
      // Push outward slightly so the cloud reads as a disc rather than a blob;
      // this preserves ordering and relative angle, only rescaling radius.
      const r = Math.hypot(px, py) || 1e-6
      const rr = Math.pow(r, 0.62) * CLOUD_RADIUS * 0.43
      px = (px / r) * rr
      py = (py / r) * rr
      for (let l = 0; l < cfg.nLayers; l++) {
        const idx = l * cfg.nFeatures + f
        positions[idx * 3] = px
        positions[idx * 3 + 1] = py
        positions[idx * 3 + 2] = layerZ(l, cfg.nLayers) + pz * 0.5
      }
    }

    this.geom = new THREE.BufferGeometry()
    this.geom.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    this.geom.setAttribute('color', new THREE.BufferAttribute(this.colors, 3))
    this.geom.setAttribute('size', new THREE.BufferAttribute(this.sizes, 1))

    const material = new THREE.ShaderMaterial({
      uniforms: { uScale: { value: 420 } },
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
    const c = this.colors
    const s = this.sizes
    const [dr, dg, db] = PALETTE.dormant
    for (let i = 0, n = c.length / 3; i < n; i++) {
      c[i * 3] = dr
      c[i * 3 + 1] = dg
      c[i * 3 + 2] = db
      s[i] = 0.24
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
          const a = Math.min(1, Math.abs(hit.delta) * 0.5) * w
          const g = a * 1.35
          c[i] += PALETTE.suppressed[0] * g
          c[i + 1] += PALETTE.suppressed[1] * g
          c[i + 2] += PALETTE.suppressed[2] * g
          s[base + hit.id] = Math.max(s[base + hit.id], 0.46 + a * 1.0)
        }
      }

      for (const hits of layer.features) {
        for (const hit of hits) {
          const i = (base + hit.id) * 3
          const a = Math.min(1, hit.act * 0.42) * w
          // Strong activations run toward amber; ordinary ones stay bone white.
          const heat = Math.min(1, Math.max(0, hit.act - 1.0) * 0.55)
          const g = a * 1.5
          c[i] += (PALETTE.active[0] * (1 - heat) + PALETTE.peak[0] * heat) * g
          c[i + 1] += (PALETTE.active[1] * (1 - heat) + PALETTE.peak[1] * heat) * g
          c[i + 2] += (PALETTE.active[2] * (1 - heat) + PALETTE.peak[2] * heat) * g
          s[base + hit.id] = Math.max(s[base + hit.id], 0.5 + a * 1.6)
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

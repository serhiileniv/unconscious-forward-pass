import * as THREE from 'three'
import { layerZ } from './lattice'
import { PALETTE } from './palette'

const SEGMENTS = 72

/**
 * One rim per layer, brightness set by how much that layer actually did.
 *
 * The circle itself is a reading aid: without it the layers smear into one
 * starfield, since every layer shares a token layout and projects to nearly the
 * same place. But a purely decorative ring has no business in this piece, so the
 * brightness is not decoration — it is the norm of what that layer wrote into
 * the residual stream at the position being computed, scaled against the largest
 * write in the pass. A dim ring is a layer that barely touched the state; a
 * bright one did the work.
 */
export class Rings {
  readonly lines: THREE.LineSegments
  private readonly geom: THREE.BufferGeometry
  private readonly colors: Float32Array
  private readonly nLayers: number

  constructor(nLayers: number, radius: number) {
    this.nLayers = nLayers
    const verts: number[] = []
    for (let l = 0; l < nLayers; l++) {
      const z = layerZ(l, nLayers)
      for (let i = 0; i < SEGMENTS; i++) {
        const a0 = (i / SEGMENTS) * Math.PI * 2
        const a1 = ((i + 1) / SEGMENTS) * Math.PI * 2
        const r = radius * 1.12
        verts.push(Math.cos(a0) * r, Math.sin(a0) * r, z, Math.cos(a1) * r, Math.sin(a1) * r, z)
      }
    }
    this.colors = new Float32Array(verts.length)
    this.geom = new THREE.BufferGeometry()
    this.geom.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
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

  private readonly tint = new THREE.Color(PALETTE.attention)

  update(layerF: number, writes: Float32Array): void {
    const per = SEGMENTS * 6
    for (let l = 0; l < this.nLayers; l++) {
      const behind = layerF - l
      // Ahead of the front: a bare structural hint. Behind it: a decaying glow,
      // scaled by what this layer actually contributed.
      const work = writes[l] ?? 0
      const lit = behind < 0 ? 0.035 : 0.035 + (0.25 + 1.35 * work) * Math.exp(-behind / 1.1)
      const r = this.tint.r * lit
      const g = this.tint.g * lit
      const b = this.tint.b * lit
      for (let i = 0; i < per; i += 3) {
        this.colors[l * per + i] = r
        this.colors[l * per + i + 1] = g
        this.colors[l * per + i + 2] = b
      }
    }
    this.geom.getAttribute('color').needsUpdate = true
  }

  dispose(): void {
    this.geom.dispose()
    ;(this.lines.material as THREE.Material).dispose()
  }
}

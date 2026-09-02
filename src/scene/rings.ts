import * as THREE from 'three'
import { CLOUD_RADIUS, layerZ } from './lattice'
import { PALETTE } from './palette'

const SEGMENTS = 72

/**
 * One faint rim per layer.
 *
 * Without these the twelve layers smear into a single starfield, because every
 * layer shares the same feature layout and so projects to nearly the same screen
 * position. The rings make the stack read as a corridor you are looking down, and
 * they give the wavefront something to visibly pass through. They mark the real
 * layer planes — the geometry is honest, the visibility is a reading aid.
 */
export class Rings {
  readonly lines: THREE.LineSegments
  private readonly geom: THREE.BufferGeometry
  private readonly colors: Float32Array
  private readonly nLayers: number

  constructor(nLayers: number) {
    this.nLayers = nLayers
    const verts: number[] = []
    for (let l = 0; l < nLayers; l++) {
      const z = layerZ(l, nLayers)
      for (let i = 0; i < SEGMENTS; i++) {
        const a0 = (i / SEGMENTS) * Math.PI * 2
        const a1 = ((i + 1) / SEGMENTS) * Math.PI * 2
        const r = CLOUD_RADIUS * 0.82
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

  update(layerF: number): void {
    const per = SEGMENTS * 6
    for (let l = 0; l < this.nLayers; l++) {
      const behind = layerF - l
      // Ahead of the front: a bare structural hint. Behind it: a decaying glow.
      const lit = behind < 0 ? 0.035 : 0.035 + 0.8 * Math.exp(-behind / 1.1)
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

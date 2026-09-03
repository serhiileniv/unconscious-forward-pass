import * as THREE from 'three'
import type { LM } from '../model/lm'
import type { StepTrace } from '../model/trace'
import { PALETTE } from './palette'

const MAX_EDGES = 20000
/** How many of the layer's strongest pushes get their neighbourhood drawn. */
const SOURCES = 900

/**
 * The model's own nearest-neighbour graph, lit where the layer is pushing.
 *
 * This is the one part of the spatial picture that is exact. Where a point sits
 * is a weak summary of 768 dimensions and the interface says so — but whether
 * two tokens are neighbours is a fact about the embedding, true regardless of
 * where the two ends happen to be drawn. So the edges carry the structure the
 * positions cannot.
 *
 * Drawn only around tokens the current layer is actually moving, which is what
 * turns a static graph into something that looks like activity spreading: a push
 * on one word lights the words the model holds next to it.
 */
export class Synapses {
  readonly lines: THREE.LineSegments
  private readonly geom: THREE.BufferGeometry
  private readonly positions: Float32Array
  private readonly colors: Float32Array
  private readonly model: LM
  private readonly latticePos: Float32Array
  private readonly nFeatures: number
  private readonly indexOf: Map<number, number>
  private lastKey = -1

  constructor(model: LM, latticePos: Float32Array, nFeatures: number) {
    this.model = model
    this.latticePos = latticePos
    this.nFeatures = nFeatures
    this.indexOf = new Map()
    model.lensIds.forEach((tokenId, i) => this.indexOf.set(tokenId, i))

    this.positions = new Float32Array(MAX_EDGES * 6)
    this.colors = new Float32Array(MAX_EDGES * 6)
    this.geom = new THREE.BufferGeometry()
    this.geom.setAttribute('position', new THREE.BufferAttribute(this.positions, 3))
    this.geom.setAttribute('color', new THREE.BufferAttribute(this.colors, 3))
    this.geom.setDrawRange(0, 0)
    this.lines = new THREE.LineSegments(
      this.geom,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.28,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      }),
    )
    this.lines.frustumCulled = false
  }

  update(trace: StepTrace, layerF: number, nLayers: number): void {
    const layout = this.model.layout
    if (!layout) {
      this.geom.setDrawRange(0, 0)
      return
    }
    const l = Math.round(layerF)
    if (l < 0 || l >= nLayers) {
      this.geom.setDrawRange(0, 0)
      return
    }
    const key = l * 8 + Math.round((layerF - l) * 4)
    if (key === this.lastKey) return
    this.lastKey = key

    const phase = 1 - Math.min(1, Math.abs(layerF - l) * 1.5)
    if (phase <= 0.02) {
      this.geom.setDrawRange(0, 0)
      return
    }

    const layer = trace.layers[l]
    const hits = layer.features[trace.active] ?? []
    const down = layer.suppressed[trace.active] ?? []
    const base = l * this.nFeatures
    const warm = PALETTE.active
    const cold = PALETTE.suppressed

    let v = 0
    const emit = (list: typeof hits, tint: readonly [number, number, number] | typeof warm): void => {
      for (let i = 0; i < Math.min(SOURCES, list.length) && v < MAX_EDGES; i++) {
        const src = list[i]
        const tokenId = this.model.lensIds[src.id]
        const strength = Math.min(1, (src.act - 1.8) * 0.35) * phase
        if (strength <= 0.02) continue
        for (let j = 0; j < layout.k && v < MAX_EDGES; j++) {
          const nb = layout.nbr[tokenId * layout.k + j]
          const sim = layout.sim[tokenId * layout.k + j]
          if (nb < 0 || nb === tokenId || sim < 0.25) continue
          const dst = this.indexOf.get(nb)
          if (dst === undefined) continue
          // A neighbour drawn far away says more about the projection than about
          // the model, and a few such edges dominate the frame. Skip them.
          const ax = (base + src.id) * 3
          const bx = (base + dst) * 3
          const far =
            (this.latticePos[ax] - this.latticePos[bx]) ** 2 +
            (this.latticePos[ax + 1] - this.latticePos[bx + 1]) ** 2 +
            (this.latticePos[ax + 2] - this.latticePos[bx + 2]) ** 2
          if (far > 36) continue
          const a = (base + src.id) * 3
          const b = (base + dst) * 3
          this.positions.set(
            [
              this.latticePos[a], this.latticePos[a + 1], this.latticePos[a + 2],
              this.latticePos[b], this.latticePos[b + 1], this.latticePos[b + 2],
            ],
            v * 6,
          )
          // Brighter at the pushed end, dimmer at the neighbour, so the edge
          // reads as reaching outward rather than as a static link.
          const g = strength * sim * 0.16
          this.colors.set(
            [tint[0] * g, tint[1] * g, tint[2] * g, tint[0] * g * 0.25, tint[1] * g * 0.25, tint[2] * g * 0.25],
            v * 6,
          )
          v++
        }
      }
    }
    emit(hits, warm)
    emit(down, cold)

    this.geom.setDrawRange(0, v * 2)
    this.geom.getAttribute('position').needsUpdate = true
    this.geom.getAttribute('color').needsUpdate = true
  }

  dispose(): void {
    this.geom.dispose()
    ;(this.lines.material as THREE.Material).dispose()
  }
}

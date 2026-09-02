import * as THREE from 'three'
import { CSS } from './palette'

export interface LabelSpec {
  id: string
  text: string
  sub?: string
  world: THREE.Vector3
  kind: 'feature' | 'suppressed' | 'anticipation' | 'token'
  opacity: number
}

/**
 * Text lives in the DOM, not in the WebGL scene, and is positioned by projecting
 * its anchor point to screen space each frame. Canvas-rendered type at these sizes
 * is mush; real text stays crisp and is selectable and accessible.
 */
export class Labels {
  private readonly host: HTMLElement
  private readonly pool = new Map<string, HTMLDivElement>()
  private readonly v = new THREE.Vector3()

  constructor(host: HTMLElement) {
    this.host = host
  }

  update(specs: LabelSpec[], camera: THREE.Camera, width: number, height: number): void {
    const live = new Set<string>()
    for (const spec of specs) {
      this.v.copy(spec.world).project(camera)
      if (this.v.z > 1) continue
      const x = (this.v.x * 0.5 + 0.5) * width
      const y = (-this.v.y * 0.5 + 0.5) * height
      if (x < -100 || x > width + 100 || y < -60 || y > height + 60) continue

      live.add(spec.id)
      let el = this.pool.get(spec.id)
      if (!el) {
        el = document.createElement('div')
        el.className = 'label'
        this.host.appendChild(el)
        this.pool.set(spec.id, el)
      }
      const color =
        spec.kind === 'suppressed' ? CSS.suppressed : spec.kind === 'anticipation' ? CSS.peak : CSS.active
      el.dataset.kind = spec.kind
      el.style.color = color
      el.style.opacity = String(spec.opacity)
      el.style.transform = `translate3d(${Math.round(x)}px, ${Math.round(y)}px, 0)`
      const html = spec.sub ? `${escapeHtml(spec.text)}<i>${escapeHtml(spec.sub)}</i>` : escapeHtml(spec.text)
      if (el.innerHTML !== html) el.innerHTML = html
    }
    for (const [id, el] of this.pool) {
      if (!live.has(id)) {
        el.remove()
        this.pool.delete(id)
      }
    }
  }

  clear(): void {
    for (const el of this.pool.values()) el.remove()
    this.pool.clear()
  }
}

function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!)
}

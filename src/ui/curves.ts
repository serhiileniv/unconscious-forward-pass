import type { StepTrace } from '../model/trace'
import { CSS } from '../scene/palette'

const W = 236
const H = 128
const PAD_L = 4
const PAD_R = 52
const PAD_T = 8
const PAD_B = 16

/**
 * How the answer was built, one line per leading word.
 *
 * The cloud shows every word's score moving at once, which is the honest
 * picture and an unreadable one for any single question. This is the same
 * numbers for six words: where a line ends is the model's real logit for that
 * word, and where two lines cross is the layer at which the model changed its
 * mind. It is the only view here that answers "why that word and not the other".
 */
export function renderCurves(host: SVGSVGElement, trace: StepTrace, layerF: number, selected: number | null): void {
  const curves = trace.curves
  if (!curves.length) {
    host.innerHTML = ''
    return
  }
  const nLayers = curves[0].values.length
  let lo = Infinity
  let hi = -Infinity
  for (const c of curves) {
    for (const v of c.values) {
      if (v < lo) lo = v
      if (v > hi) hi = v
    }
  }
  const span = hi - lo || 1
  const x = (l: number): number => PAD_L + (l / Math.max(1, nLayers - 1)) * (W - PAD_L - PAD_R)
  const y = (v: number): number => PAD_T + (1 - (v - lo) / span) * (H - PAD_T - PAD_B)

  const parts: string[] = []
  // Layer axis ticks: only the ends and the current position, to stay quiet.
  parts.push(
    `<line x1="${x(0)}" y1="${H - PAD_B}" x2="${x(nLayers - 1)}" y2="${H - PAD_B}" stroke="${CSS.line}" />`,
    `<text x="${x(0)}" y="${H - 4}" fill="${CSS.dim}" font-size="8">0</text>`,
    `<text x="${x(nLayers - 1)}" y="${H - 4}" fill="${CSS.dim}" font-size="8" text-anchor="end">${nLayers - 1}</text>`,
  )
  const at = Math.max(0, Math.min(nLayers - 1, layerF))
  parts.push(`<line x1="${x(at)}" y1="${PAD_T}" x2="${x(at)}" y2="${H - PAD_B}" stroke="${CSS.attention}" stroke-opacity="0.5" />`)

  // Keep the end labels from stacking on top of each other.
  const ends = curves
    .map((c, i) => ({ i, y: y(c.values[nLayers - 1]) }))
    .sort((a, b) => a.y - b.y)
  const placed = new Float32Array(curves.length)
  let last = -Infinity
  for (const e of ends) {
    const at = Math.max(e.y, last + 10)
    placed[e.i] = Math.min(H - 4, at)
    last = placed[e.i]
  }

  curves.forEach((c, i) => {
    const isChosen = c.id === trace.chosen
    const isSel = selected === c.id
    const d = Array.from(c.values, (v, l) => `${l === 0 ? 'M' : 'L'}${x(l).toFixed(1)},${y(v).toFixed(1)}`).join('')
    const colour = isSel ? CSS.peak : isChosen ? CSS.active : CSS.muted
    const width = isSel ? 2 : isChosen ? 1.5 : 1
    const alpha = isSel || isChosen ? 1 : 0.45
    parts.push(`<path d="${d}" fill="none" stroke="${colour}" stroke-width="${width}" stroke-opacity="${alpha}" />`)
    const ly = placed[i] + 3
    parts.push(
      `<line x1="${x(nLayers - 1)}" y1="${y(c.values[nLayers - 1])}" x2="${W - PAD_R + 1}" y2="${ly - 3}" stroke="${colour}" stroke-opacity="${alpha * 0.4}" />`,
      `<text x="${W - PAD_R + 4}" y="${ly}" fill="${colour}" fill-opacity="${alpha}" font-size="9">${escapeXml(
        c.word.replace(/ /g, '·').replace(/\n/g, '\\n'),
      )}</text>`,
    )
  })

  host.setAttribute('viewBox', `0 0 ${W} ${H}`)
  host.innerHTML = parts.join('')
}

function escapeXml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&apos;' })[c]!)
}

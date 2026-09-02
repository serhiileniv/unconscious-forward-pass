/**
 * Warm means it lives, cold means it was put down.
 *
 * That is the whole colour idea. Everything that surfaces runs bone → amber;
 * everything that was active and then overruled runs violet and falls away. The
 * ground is not black but a very dark warm grey, because this is a medium being
 * swept, not an empty void.
 */
export const PALETTE = {
  bg: 0x0a0c0b,
  dormant: [0.013, 0.017, 0.016] as const,
  active: [0.96, 0.94, 0.89] as const,
  peak: [1.0, 0.79, 0.42] as const,
  suppressed: [0.42, 0.37, 0.55] as const,
  attention: 0x8fa6a2,
  stream: 0x46534f,
  streamHot: 0xf5efe2,
  anticipation: 0xffc96b,
} as const

export const CSS = {
  bg: '#0a0c0b',
  text: '#d9e0dc',
  muted: '#6e7b77',
  dim: '#3d4744',
  line: '#1c2422',
  panel: '#0f1312',
  active: '#f5efe2',
  peak: '#ffc96b',
  suppressed: '#8b7fb0',
  attention: '#8fa6a2',
} as const

import { defineConfig } from 'vite'
import { viteSingleFile } from 'vite-plugin-singlefile'

// Two build targets from one source:
//   `vite build`             -> normal multi-file bundle in dist/
//   `SINGLEFILE=1 vite build` -> one self-contained index.html, everything inlined.
// The single-file build is what gets published as an Artifact, where the CSP
// forbids loading three.js (or anything else) from an external host.
export default defineConfig({
  base: './',
  plugins: process.env.SINGLEFILE ? [viteSingleFile()] : [],
  build: {
    target: 'es2022',
    outDir: process.env.SINGLEFILE ? 'dist-single' : 'dist',
    assetsInlineLimit: process.env.SINGLEFILE ? 100_000_000 : 4096,
    chunkSizeWarningLimit: 4096,
  },
})

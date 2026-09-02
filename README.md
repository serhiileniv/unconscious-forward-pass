# The Unconscious of a Forward Pass

A 3D simulation of the sub-verbal computation inside a transformer: thousands of
features considered, one word spoken.

A real 12-layer transformer runs in the browser. Attention, superposition,
sparsity and suppression are genuinely computed — the geometry on screen is read
straight out of the tensors, not animated to look like it.

## What you are looking at

| Element | What it is |
| --- | --- |
| **Point cloud** | Every direction in a 2048-entry feature dictionary, drawn once per layer. A point's position is its actual direction in the 96-dimensional residual space, put through a fixed 3D projection — so two points are close exactly when the two directions are close. |
| **Rings** | The twelve layer planes. Depth is time here, not space: the wavefront sweeping through them is one forward pass happening. |
| **Filaments** | The residual stream, one per token, running the full depth. Every layer reads it, adds something, and puts it back; nothing is ever removed, only buried. A filament bends where a layer wrote something large into it. |
| **Arcs** | Attention at the layer currently being crossed. They are drawn as events rather than objects because that is what they are — a head reaches back, moves information, and is gone before the next layer starts. |
| **Bone → amber** | A feature awake, warmer as it gets stronger. |
| **Violet** | A feature that was awake and got overruled by a later layer. It never reaches the readout and leaves no mark on the page. |
| **Long amber threads** | A feature awake *now* whose direction points at a word the run went on to emit several steps later, labelled with its lead. |

The headline number is the ratio: how many feature activations were weighed per
word that actually got out. It runs in the low thousands to one.

## Honesty about the toy

Three things are worth stating plainly, because a convincing picture of a model's
interior is easy to fake:

1. **The network is untrained.** Its weights are random. The mechanism is real —
   real QKV attention with causal masking and softmax, real layer norm, real GELU
   feed-forward, real residual accumulation — but an untrained transformer has no
   opinion about English, so word choice is anchored by bigram statistics from the
   corpus. Everything geometric is computed; the sentences are the toy showing
   through.

2. **The feature dictionary is constructed, not learned.** A real sparse
   autoencoder learns its dictionary. This one is built: the first rows are the
   vocabulary directions themselves and the rest are compositions of two or three
   of them, which is why every feature in the scene has a name you can read. That
   reproduces the property that matters — far more directions than dimensions, so
   they cannot all be orthogonal and they interfere — without claiming the
   features are monosemantic.

3. **The embedding is real.** It is not random. Word co-occurrences are counted
   over the corpus, reweighted as positive pointwise mutual information, and
   reduced to 96 dimensions by subspace iteration. That is what makes the cloud's
   shape mean something: `attention` lands near `reaching` and `backward`,
   `silence` near `sit` and `dark`. Check it yourself in `tools/smoke.ts`.

This is the architecture of thinking. It is not a scan of any deployed model, and
no such scan is possible from the outside.

## Running it

```sh
npm install
npm run dev          # http://localhost:5173
npm run build        # dist/
npm run check        # tsc --noEmit
SINGLEFILE=1 npx vite build   # dist-single/ — one self-contained .html
```

`window.__uc.at(step, frac)` parks the wavefront at an exact instant, which is how
the screenshots in development were taken.

## Layout

```
src/model/     corpus, tokenizer, PPMI embedding, transformer, generation
src/scene/     three.js — feature lattice, layer rings, residual streams,
               attention arcs, projected DOM labels, camera
src/main.ts    playback timeline, readouts, controls
```

Defaults live in `DEFAULT_CONFIG` (`src/model/transformer.ts`): 12 layers, 96
dimensions, 6 heads, 2048 features, top-16 sparsity, 24-token context.

### Dev tools

```sh
npx esbuild tools/smoke.ts --bundle --platform=node --format=esm --outfile=/tmp/s.mjs && node /tmp/s.mjs
node tools/shot.mjs http://localhost:4173/ shot.png 7000 'window.__uc.at(3, 0.45)'
```

`tools/smoke.ts` prints nearest neighbours out of the embedding and checks that
attention rows sum to one. `tools/shot.mjs` drives headless Chrome over the
DevTools protocol — Chrome's `--virtual-time-budget` never fires on a page with a
permanent animation loop, which this is.

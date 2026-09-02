# The Unconscious of a Forward Pass

A 3D visualisation of what happens inside a language model between reading a
prompt and saying one word.

Qwen2.5 0.5B Instruct — the real published weights — runs its real forward pass
in your browser. Nothing is simulated and nothing is estimated: what lights up is
an exact decomposition of the model's own output.

## What you are looking at

| Element | What it is |
| --- | --- |
| **Point cloud** | One point per whole-word token, drawn once per layer, placed at that token's real row of the model's output head. Two points sit close together exactly when the model puts those two words close together. |
| **Rings** | The 24 layer planes. Depth is time: the wavefront crossing them is one forward pass. |
| **Filaments** | The residual stream, one per token. Every layer reads it, adds something, and puts it back; nothing is erased, only buried. |
| **Arcs** | Attention at the layer being crossed — one position reaching back at earlier ones. Drawn as events, because that is what they are. |
| **Bone → amber** | A word this layer pushed the model *toward*. |
| **Violet** | A word this layer pushed the model *away* from. |
| **Long amber threads** | A word the model went on to write several steps later, already being pushed for now. |

## What makes it exact

The residual stream is a running sum — the embedding plus every layer's write —
and the output head is linear. So once the final norm's scale is fixed, each
layer's contribution to every word's score is exactly computable, and the
contributions sum to the model's real logits.

That is checkable, and it is checked:

```
$ node tools/evalmodel.mjs public/model/qwen
attribution vs real logits at final layer: max abs error 0.00e+0
```

**Why not the logit lens.** The better-known technique re-applies the final norm
at each depth. Measured on this model it disagrees with the real output at every
layer before 20 of 24 — KL 8–13, zero overlap in the top ten, with top tokens
like `" rain"`, `" sequ"`, `" swe"`. On GPT-2 it behaves far better, which is why
most published examples use GPT-2. `tools/faithful.mjs` runs that comparison, and
`src/model/gpt2.ts` is kept as the reference implementation for it.

## A worked example

`The Eiffel Tower is located in the city of` → ` Paris` (54.4%)

Contribution to `" Paris"`, by layer:

```
L17 +1.78   L18 +1.33   L19 +0.65   L20 +5.73   L21 +10.90   L22 -3.86
```

Layer 21 fires `" French" +12.9`, `" France" +11.6`, `" Paris" +10.9`,
`" Louis" +7.8` together — the France concept arriving at once. Then layer 22
pushes ` Paris` back down while promoting `" in"`, `" and"`, `" the"`. Those
numbers are not an interpretation; they add up to what the model said.

## What it does not show

Why a layer pushed as it did, and any internal state the output head cannot see.
These are not "the words in the model's mind". They are the exact amount each
layer moved every word's score.

## Running it

```sh
npm install
npm run fetch-weights   # 942 MB from HuggingFace, gitignored
npm run dev             # http://localhost:5173
npm run build        # dist/
npm run check        # tsc --noEmit
SINGLEFILE=1 npx vite build   # dist-single/ — one self-contained .html
```

`window.__uc.at(step, frac)` parks the wavefront at an exact instant, which is how
the screenshots in development were taken.

## Layout

```
src/model/safetensors.ts  weight loading, f32/f16/bf16
src/model/bpe.ts          byte-level BPE, GPT-2 and Llama-era splits
src/model/llama.ts        RMSNorm, RoPE, grouped-query attention, SwiGLU
src/model/gpt2.ts         GPT-2, kept for the lens-vs-attribution comparison
src/model/run.ts          decoding, attribution, anticipation
src/scene/                three.js — token cloud, layer rings, residual
                          streams, attention arcs, projected DOM labels
src/main.ts               playback timeline, readouts, controls
```

Both model families implement `LM` in `src/model/lm.ts`, so swapping checkpoints
is a URL change. `MODEL_URL` in `src/model/run.ts` selects one.

### Dev tools

```sh
npx esbuild tools/evalmodel.ts --bundle --platform=node --format=esm --outfile=tools/evalmodel.mjs
node --max-old-space-size=8192 tools/evalmodel.mjs public/model/qwen
node tools/shot.mjs http://localhost:5173/ shot.png 70000 'window.__uc.at(0, 0.6)'
```

`tools/evalmodel.ts` checks the attribution against the real logits and prints
per-layer behaviour. `tools/attrib.ts` breaks one token's score down by layer.
`tools/faithful.ts` runs the logit-lens comparison. `tools/shot.mjs` drives headless Chrome over the
DevTools protocol — Chrome's `--virtual-time-budget` never fires on a page with a
permanent animation loop, which this is.

### Recording

```sh
node tools/record.mjs http://localhost:5173/ frames 30 30 1600 900
ffmpeg -framerate 30 -i frames/f%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 out.mp4
```

Frames are rendered at explicit simulated times and explicit camera positions
rather than screen-grabbed, so the cut plays at the speed it claims to and the
camera move is identical every run.

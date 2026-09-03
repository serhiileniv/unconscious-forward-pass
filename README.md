# The Unconscious of a Forward Pass

A 3D visualisation of what happens inside a language model between reading a
prompt and saying one word.

Real published weights run a real forward pass in your browser. Every token in
the vocabulary is drawn, with no subset and no cap on how many may light at once,
and what lights up is an exact decomposition of the model's own output.

GPT-2 124M by default, because it stays interactive with all 50,257 of its tokens
attributed at every layer. `?model=qwen` loads Qwen2.5 0.5B Instruct, which
answers far better and runs about six times slower.

## What you are looking at

| Element | What it is |
| --- | --- |
| **Point cloud** | One point per token in the whole vocabulary, drawn once per layer, placed at that token's real row of the model's output head, projected to 3D by PCA. See *What the picture cannot carry* below before reading meaning into proximity. |
| **Rings** | The 24 layer planes. Depth is time: the wavefront crossing them is one forward pass. |
| **Filaments** | The residual stream, one per token. Every layer reads it, adds something, and puts it back; nothing is erased, only buried. |
| **Arcs** | Attention at the layer being crossed — one position reaching back at earlier ones. Drawn as events, because that is what they are. |
| **Bone → amber** | A word this layer pushed the model *toward*. |
| **Violet** | A word this layer pushed the model *away* from. |
| **Long amber threads** | A word the model went on to write several steps later, already being pushed for now. |

## Checking it

Every claim the interface makes about what a visual property means is
re-derived from the trace and compared against the buffers actually being drawn.
It reads the same memory the GPU does, so a check passing means the pixels are
right, not that the intent was.

```
$ node tools/validate.mjs
150 checks across 15 moments

pass  points drawn = words this layer moved      7491 drawn, 7491 past threshold
pass  depth = layer index                        max deviation 0.494 within slab 0.525
pass  distance from axis = push strength         max deviation 5.1e-7 over 7441 points
pass  warm = pushed toward, violet = pushed away  0 points with the wrong hue
pass  ring radius = strongest push in that layer  max deviation 4.6e-7
pass  filament x = token position                max deviation 4.4e-7
pass  filament height = residual norm            max deviation 1.2e-7
pass  every link ends on a drawn point           2498 links, 0 dangling
pass  attention rows sum to one                  max deviation 4.3e-8
pass  curve ends at the final layer score
```

It has already caught two real defects: neighbour links borrowed attention's
gate and strobed off halfway between layers, and a check that passed by drawing
nothing at all until it was made to report its own count.

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

## What the picture cannot carry

Depth, colour and brightness are exact. Horizontal and vertical placement is not,
and it is worth being blunt about how much it is not.

The cloud is 3 of 768 dimensions. Measured over the real output-head rows
(`tools/geometry.mjs`):

| projection | variance kept | distance fidelity (Pearson r) |
| --- | --- | --- |
| random | 0.3% | 0.11 |
| random + an `r^0.62` radial warp | 0.3% | 0.08 |
| **top-3 PCA** (used) | **3.2%** | **0.33** |

An earlier version shipped the random projection *and* the warp, and claimed in
the interface that "two points sit close together exactly when the model puts
those words close together". That was false: r = 0.11 is barely above noise, and
the warp — added purely because it filled the disc more evenly — measurably made
it worse. Both are gone. PCA is the best linear 3D summary available and roughly
triples both figures, and the interface now reports them, because 3.2% still
means proximity is a weak hint rather than evidence.

Also not shown: why a layer pushed as it did, and any internal state the output
head cannot see. These are not "the words in the model's mind". They are the
exact amount each layer moved every word's score.

## Display settings, kept separate from measurements

One threshold decides what is drawn: a layer must move a token's score by more
than 2.2 standard deviations. Nothing else is capped. There is no top-k — if a
layer moves forty thousand scores hard, forty thousand points light.

The readouts say which side of that line they are on. **Scores moved** is a
property of the model: every layer changes the score of every token, so one
generated word is `vocabulary x layers` score changes. **Lit** and **moved hard**
are counts of drawn things and are labelled as such. An earlier version made the
headline ratio out of the drawn count, which made it a statement about the
display setting rather than about the model.

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

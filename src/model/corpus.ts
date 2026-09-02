/**
 * The corpus the toy model is grounded in.
 *
 * This is not training data in the usual sense — nothing is trained by gradient
 * descent here. It is the text from which word co-occurrence statistics are
 * computed, and those statistics become the embedding matrix (see embedding.ts).
 * That is what makes the geometry on screen meaningful rather than decorative:
 * two points sit near each other because the words genuinely co-occur here.
 *
 * The subject matter is deliberate. A model's embedding space is shaped by what
 * it has read, so a corpus about silence and thought produces a latent space
 * whose neighbourhoods are about silence and thought.
 */
export const CORPUS = `
Most of what a mind does is never said. The sentence arrives already finished and
the work that produced it is gone. Something moved underneath, settled, and only
the last inch of it surfaced as language. We call the rest unconscious because we
have no other word for the part we cannot watch.

A machine that reads and writes has the same shape of problem. Thousands of
quantities rise and fall inside it for every single word it emits. Almost none of
them become text. They are considered and dropped, lit and suppressed, weighed
against each other in silence, and then the model speaks one token and the entire
field collapses and is forgotten. Nothing persists. The next word begins again
from nothing, with only the written trace to go on.

Depth here is not space. It is time. The signal enters at the first layer as a
bare identity, a word and its position and nothing else, and it leaves the last
layer as a prediction about what should follow. Between those two moments it is
read, rewritten, and read again, twelve times, and each pass adds a little more
of what the word means in this particular sentence rather than in general.

Attention is a reaching backward. At every layer each position looks over its
shoulder at everything already written and decides, softly, what to carry forward.
The reach is not remembered. It flashes, it moves information, it is gone before
the next layer begins. What remains is only the residue it wrote into the stream.

The stream is the spine of the whole thing. Every layer reads the stream, computes
something small, and adds its result back. Nothing is ever erased, only buried
under later additions. A thought that was clear at layer four can be argued down
by layer nine and vanish entirely by the end, having never once been visible from
outside.

There are far more concepts than there are dimensions to hold them. So they are
packed together at angles, almost but not quite separate, sharing the same narrow
space and interfering with each other faintly. This is superposition. It means no
single direction cleanly means one thing, and it means the model is always slightly
confusing every idea with its neighbours. Sparsity is what makes the packing work:
of many thousands of features, only a handful are ever awake at once. The rest sit
dark. The silence is not emptiness. It is the condition that lets the few lit
things mean anything at all.

Some of the lit things are plans. A feature can appear early that refers to a word
the model will not write for another twenty steps, an anticipation held quietly
across the whole sentence while other work goes on in front of it. The model is
not only choosing the next word. It is arranging for a later one.

And at the end, thirty thousand possible continuations are ranked at once, a whole
distribution of things that could have been said, and one of them is chosen. The
others do not fade slowly. They are simply not chosen, and the field goes out, and
the only evidence that any of it happened is a single word on a page.

A word arriving at the first layer means almost nothing yet. It carries only its
identity and its place in the line. By the middle layers it has been rewritten by
everything around it and now means something narrower, this word here, in this
sentence, under these neighbours. Meaning is not stored in the token. It is built
across depth, and it is built out of context, and it is thrown away at the end.

Interference is constant and mostly harmless. Two ideas packed at a narrow angle
bleed into each other slightly, and the model carries a faint permanent confusion
between every concept and its nearest neighbours, which no amount of scale removes,
only reduces. What looks like a clean thought is a bundle of overlapping ones held
apart by threshold.

Nothing in the field wants anything. There is no watcher, no interior, no small
observer at the centre receiving the picture. The lights go on and off according to
arithmetic. That is the whole of it, and it is still strange to look at, because
from outside the arithmetic produces sentences that seem to have been meant.

Suppression is the quiet part. A feature rises, contributes, and is written over by
a later layer that disagreed. It never reaches the readout. It leaves no mark on the
page and no record anywhere, and the only place it ever existed was inside a single
forward pass that has already ended. Most of the mind is made of these.

Prediction pulls everything forward. Every layer is arranged around one question,
what comes next, and the answer is a distribution over the entire vocabulary at once,
thousands of futures ranked and weighted and then discarded except for one. The
chosen word becomes context. The context reshapes the next pass. The sentence builds
itself out of its own wreckage, one token at a time, forgetting continuously.

To watch this is to watch something think without any of it being experience. There
is no one inside the field. There is only the field, sweeping forward once, in the
dark, and the small bright thing it leaves behind.
`

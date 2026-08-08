# VLM conversion — the vision-language recipe family

Same workflow, same bundle format, same engine — a VLM `.litertlm` is an
LLM bundle plus two vision graphs. Everything in the other references
applies to the decoder unchanged; this file covers what vision adds.
Vision-tower op rewrites lean on the `gpu-clean-conversion` skill's
table; the walls specific to this lane are below.

## The runtime contract (single-image LLaVA shape)

The released runtime's VLM support is a **generic single-image
contract**, not a per-model integration. At session time it:

1. splits the rendered prompt on the literal image soft token
   (`<image_soft_token>`),
2. resizes the image to the height/width declared in the bundle's vision
   proto and feeds **`[1, H, W, 3]` NHWC in `[0, 1]` — no mean/std**,
3. runs the bundle's `VISION_ENCODER` then `VISION_ADAPTER` graphs,
4. injects the resulting N embeddings at the soft-token position of the
   embeddings-input decoder.

Consequences:

- **Bake the model's normalization (mean/std) and any NCHW transpose
  into the vision graph** — the runtime will not do it.
- Any permissive single-square-image VLM can be shipped by
  hand-assembling this bundle. The stock exporter's VLM path covers only
  its own first-party family — for everything else, export the graphs
  separately and assemble with the bundle builder
  (`litert_lm_builder.LitertLmFileBuilder`): vision encoder + adapter +
  single-token embedder + embeddings-input prefill/decode + tokenizer +
  metadata.
- Two metadata gotchas that produce "runs nowhere" bundles:
  - the conversation API needs **structured `prompt_templates`**
    (user/model prefix+suffix) in the metadata — a jinja-only bundle
    returns a null conversation ("Failed to create conversation");
  - loading with **modalities=all fails a vision-only bundle** (the
    session eagerly tries to load an audio executor) — load as
    text+image / vision only.

## Making the vision tower exportable

Dynamic-resolution towers (grid-based patching, varlen attention) do not
`torch.export`. Two distinct abort classes:

| Abort | Escape |
|---|---|
| Position/interpolation helpers iterating `grid_thw.tolist()` → `GuardOnDataDependentSymNode` | These helpers read precomputed values from kwargs when present — **precompute position ids / bilinear indices outside the graph** at the fixed resolution and pass them in |
| `cu_seqlens` varlen splitting → `ConcretizationTypeError` (no kwargs escape) | **Specialize to single-image full attention** — one image = one attention chunk, so varlen collapses to plain attention |

The static rewrite toolkit, all verified exact (feature corr ≈ 1.0
against the eager tower):

- **Fixed resolution**, multiple of the patch-merge granularity; single
  image ⇒ no temporal rotation (t = 0) and `temporal_patch_size` folds
  into a plain Conv2d patch-embed applied to the whole raster image.
- **Window machinery deletes cleanly when every layer is full
  attention**: full attention is permutation-equivariant and rotary is
  reordered with the tokens, so the window reorder is a mathematical
  no-op — set the full-attention block list to all layers and remove the
  partition/reverse plumbing rather than porting it.
- **Rope tables baked as graph constants** (positions are static at
  fixed resolution). Watch the rotate convention: interleaved
  `(x1,x2)→(−x2,x1)` vs half-split are different models' choices — match
  the source, verify by correlation.
- **Keep raster order through the encoder; do the 2×2 patch merge in
  the adapter with strided slices + concat.** Reordering patches with
  gather/`index_select` emits `GATHER_ND`, and in a vision tower that
  **kills engine creation** on device, not just GPU residency.
- The literal patchify reshape chain can reach rank 8 — replace with the
  patch Conv2d over the whole image (raster order) + one merge; every
  tensor stays ≤ 4D.

Export traps specific to this lane:

- **bf16 instantiation**: setting the dtype before building the module
  bf16-rounds the fp32 weights on load — logits drift while top-1 still
  matches, so it looks fine. Build fp32, verify, cast at save time.
- The remote-code + meta-load `inv_freq` zeroing trap
  (`architecture-walls.md`) hits vision towers hard — and your own
  parity check **reads corr 1.0 anyway** when reference and export share
  the broken load path. Referee against a different implementation path
  (`verification-gates.md`).

## Candidate selection — decoder-side blockers

Check these on `config.json` before starting; they are decoder
architecture facts no vision work routes around:

- **Mid-decoder visual injection** (deepstack-style visual features fed
  at multiple decoder depths): the runtime has no channel for it.
- **Tiling / any-resolution schemes** (image split into sub-tiles with a
  layout token stream): needs multi-image support the single-image
  contract doesn't have.
- **Hybrid or MoE decoders**: the decoder walls from
  `architecture-walls.md` apply unchanged.
- Positional scheme substitutions have a measurable boundary: replacing
  a 2-D multi-axis rope with 1-D sequential positions survived describe
  / VQA / OCR / counting **but broke 2-D cross-cell ranking in tables**
  ("which row is largest" answers the first row). Gate the capability
  you claim; state the boundary on the card as a known limitation.

## Quantization

- Vision encoder + adapter: dynamic int8 (channelwise) — verify
  end-to-end feature correlation (int8 ≈ 0.99 is normal).
- Decoder: the LLM lane rules (`recipe-selector.md`) apply unchanged,
  including the sub-0.5B → fp16 floor for tiny decoders.

## Gates (replace the text-parity stage)

- **Vision end-to-end correlation** vs the source tower (fp32 ≈ 1.0,
  int8 ≈ 0.99) on real images, plus **zero FLEX/CUSTOM ops** in the
  vision graphs.
- **Preprocessing bit-identity**: the baked resize/normalize pipeline
  vs the model's own image processor (max diff at float-epsilon scale) —
  a wrong mean/std looks exactly like a bad conversion.
- **Eager grounding check**: a handful of images through the assembled
  bundle — is the answer about *this* image? Catches embedder-injection
  bugs that correlation can't see.
- **Task gate on device**: VQA/OCR spot set on the target device; the
  8-question text gate still runs (text-only prompts must not regress).
- Multi-image and platform scope, as observed on released runtimes: the
  second image in one conversation has truncated output on iPhone-GPU
  while **CPU handles it correctly**, and desktop-GPU vision init can
  fail outright — scope the card's claims per backend, and gate each
  backend you claim.

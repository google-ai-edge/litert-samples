---
name: gpu-clean-conversion
description: Convert a PyTorch or Hugging Face model into a LiteRT model that runs fully on the GPU via the CompiledModel API with verified-correct output, and lay it out as a model recipe. Use when converting a new model, or when a converted model is rejected by the GPU, falls back to CPU, or returns wrong numbers on device.
---

# GPU-clean conversion

A conversion is done when three things hold, in this order:

1. it converts,
2. every node runs on the GPU,
3. **the on-device output matches the source model.**

Step 3 is not implied by step 2. The delegate can report full residency and
still return wrong numbers — that is the failure mode most of the rewrites
below exist for. Never call a model done without a numerical check against
CPU or PyTorch on the actual device.

## Loop

**1. Convert plain first.** No patches. This tells you what the model
actually needs rather than what you assumed.

**2. Verify through the CompiledModel API before touching a device.**

```python
from litert_gpu_toolkit import check_gpu_compatibility, print_report
print_report(check_gpu_compatibility("model.tflite"))
```

It compiles the model for the GPU accelerator, runs every signature on
random inputs, and compares the outputs against a CPU-compiled reference.
A failed GPU compile names the offending op — route it through the table
below. A pass with a CPU-fallback warning means some ops fell back to the
CPU; treat them the same way if you need full residency. This exercises the
host GPU, so step 5 on the actual device is still required.

**3. Map each symptom to a rewrite.** The rewrites live in
`utilities/litert_gpu_toolkit` — plain Python, no build step. Outside this
repo, clone litert-samples and put `utilities/` on `PYTHONPATH`, or vendor
the directory:

| What you see | Rewrite |
|---|---|
| `GATHER_ND` named in the compile error | Find its source. Stride-2 slicing (`x[:, ::2]`, Focus stems, patch merging), `grid_sample`, `MaxPool` padding, bicubic `interpolate`, and reflect-mode `F.pad` all lower to it. `patch_grid_sample`, `patch_maxpool_zeropad`, `patch_interpolate`, `patch_patch_merging` |
| A rank-5+ tensor named in the compile error | `PixelShuffle` (rank-6 reshape), windowed attention, `einops.rearrange`, packed-QKV attention head splits. `pixelshuffle_to_conv_transpose`, `patch_window_attention`, `patch_einops` |
| `TRANSPOSE_CONV` rejected | Version skew, not a missing op. `ZeroStuffConvT1d` / `ZeroStuffConvT2d` — zero-stuff plus a plain conv, exact to ~1e-7 |
| `SELECT` / `SELECT_V2` | `PReLU`, `ELU`, in-place index assignment, and `torch.where` masking. Replace with arithmetic: `x*(1-m) + v*m` |
| `BROADCAST_TO` | Two cases. On a compile-time constant: an outer product or `.expand` that did not fold — bake the result as a constant at its target shape. On a **runtime** tensor the GPU delegate rejects it outright, even at rank 4 — the canonical case is GQA's `repeat_kv` (`x[:,:,None].expand(...)`, which is also rank-5, so two walls in one line). Exact rewrite: `torch.cat([x[:, i:i+1].expand(b, n_rep, s, d) for i in range(n_kv)], dim=1)` — same head order, bit-exact. Tracked upstream: google-ai-edge/LiteRT#9191 |
| Masked attention wrong only on device: token 0 bit-exact, every later token wrong | Broadcast `ADD` whose LHS is a `BATCH_MATMUL` result (the `scores + mask[1,1,S,S]` idiom) silently miscomputed on older runtimes (fixed in newer; head-axis size-1 broadcast only). The signature mimics broken RoPE — tap the rope output before blaming it. Rewrites: pre-expand the mask to `[1,H,S,S]`, or materialize the BMM as a second output |
| An `ADD` result that is both a graph output **and** consumed downstream comes back wrong | Output aliasing: the returned tensor holds an *operand*, not the sum — `ADD` with two runtime operands (`SUB`/`MUL` exact, `x + 1.0` exact). This is the shape of every explicit state update in a streaming/recurrent graph. Workaround: emit `acc * one` where `one` is a **runtime** input holding 1.0 — a constant folds straight back into the pattern. Tracked upstream: google-ai-edge/LiteRT#8599 |
| `RELU_0_TO_1` rejected by the GPU delegate | Emitted by hard-sigmoid / `nn.Hardtanh(0,1)`. Accepted in litert 2.1.3, rejected from 2.1.5 on — a model at full residency on an older runtime hard-fails `CompiledModel` creation after an upgrade. Rewrite: `relu(x) - relu(x-1)`, exact. Tracked upstream: google-ai-edge/LiteRT#8598 |
| `DIV: No support of few identical inputs` / `Expected 1 const input tensor(s)`, device only | The delegate declines an op whose two inputs are the same tensor, and ops whose inputs are all constants — together these split a perceiver-style block (softmax over a length-1 axis of a constant latent bank) into several partitions. Fixes: special-case the degenerate axis (a softmax over a length-1 axis is identically 1), or make one input non-constant. Note the sibling LayerNorm-over-a-constant pattern no longer reaches the delegate at all — the converter folds it to a single `MUL`. Tracked upstream: google-ai-edge/LiteRT#9192 |
| `NHWC node rewriter not found: amax` | `x.amax(...)`/`x.max(dim)` in stable-softmax, adaptive norms, qk-norm. Rewrite channel reduce-max as `max_pool2d(x.reshape(N,1,C,H*W), kernel=(C,1))` — numerically identical — or drop the norm to 3D |
| `Lowering not found: aten._fft_r2c` / `aten.complex` | `torch.stft`/`istft` and complex views have no lowering (fails before any GPU check). A DFT is a fixed linear map: windowed-DFT as `Conv1d` with the cos/sin basis baked into kernels (stride = hop), iSTFT as inverse-DFT matmul + overlap-add via zero-stuffed conv-transpose — exact. Model-selection corollary: prefer time-domain vocoder branches over iSTFT-based ones. ⚠ Library STFT-as-conv stacks (torchlibrosa-style) have **numerically mis-converted** (corr 0.83) while the op check looks clean — verify the spectrogram numerically or compute log-mel host-side |
| Compiles, runs, output is wrong or NaN | The fp16 reduction family — and the trigger is the **fp16 accumulator passing 65504**, so a plain single-axis mean/sum over enough elements overflows just like variance does. `patch_safe_layernorm`, `patch_rmsnorm`, `patch_instance_norm`, `hierarchical_mean`. Caveats: at extreme magnitudes (\|x\| in the thousands) even the adaptive safe-LN form overflows when it reconstructs the large variance — the robust form stays entirely in the down-scaled domain (`xs = x/S`, normalize `xs`, never multiply the variance back by `S²`); `hierarchical_mean` is exact only for power-of-two spatial dims (for arbitrary dims, cascade `/2` avg-pools with `ceil_mode` so each stage averages ≤~49 elements). Diagnostic split: **all-zero/all-blank output = an overflow in one block; a result that starts near-correct and degrades with depth = precision compounding**, which no overflow patch (and no fp32-precision flag) fixes |
| Head outputs exactly zero | RMSNorm `Σx²` overflowed fp16 to `inf`. `patch_rmsnorm` |

**4. Re-convert and re-verify.** Repeat 2–3 until the check passes.

**5. Verify on device, numerically.** Run the same input through the source
model and the on-device graph and compare. Correlation on the output tensor,
plus a task-level check where one exists (argmax match, IoU, mask foreground
count). Record the device and the residency line (`N/N` nodes) alongside the
number — a result without both is not reproducible.

If residency is full and the output is wrong, bisect by intermediate: dump
tensors at block boundaries and find the first one that diverges. The cause is
usually a reduction, not the op you suspect.

## Watch for

- **fp16 is used even for an fp32 graph.** The delegate reduces in fp16
  regardless of tensor dtype. Anything that sums many large values — variance,
  `Σx²`, multi-axis mean — is a candidate.
- **Approximation choices are not free.** Substituting a GELU or Swish
  approximation changes numerics. A head with a wide output range can lose
  real accuracy to the coarser form.
- **`inplace=True` activations mutate a residual.** `out = self.act(x)` then
  `x + out` adds `relu(x)`, not `x`. Swapping the activation silently changes
  the math.
- **A silent CPU fallback looks like success.** If the GPU output is
  bit-identical to CPU fp32, it probably did not run on the GPU. Genuine fp16
  execution drifts in the last digits.
- **Reflect padding routes through `F.pad` even at `padding=0`** — every
  conv with `padding_mode='reflect'` is affected, not just padded ones.
  Slice+concat reflect pads are cheap for small pad widths.
- **Channel-attention through a `Linear` confuses layout handling at
  convert time** (`tfl.mul operands don't have broadcast-compatible
  shapes`) — express channel attention as a 1×1 conv.
- **`.chunk()` lowers to `SPLIT`** (GPU-rejected) — slice directly
  instead; bit-exact.
- **Constant folding can explode the file**: a frozen-param × constant
  product materializes at full size per block (fp16 casting skips
  non-weight constants, so it cannot rescue it). Feed one factor as a
  runtime input — param × input never folds.

## Output layout

Ship the result as a model recipe, not a one-off script:

```
models/<family>/<model>/
  README.md                 what the model is, what was produced
  converted/
    README.md               environments, pipeline, verification results
    export_*.py             one script per graph
    dump_*_ref.py           reference dumps from the source model
    verify_*.py             parity check against those references
```

Keep the export, the reference dump, and the verification separate so each
can be re-run alone. State the verification numbers in the README with the
device they came from. Weights are not committed.

The tree above is this repo's convention. In your own project the directory
names matter less than the split — keep the same three separately-runnable
pieces wherever your models live.

Do the model first. A demo can follow, and reusable inference code matters
more in it than a full UI.

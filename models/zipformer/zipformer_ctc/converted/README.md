# Zipformer CR-CTC Conversion

Model recipes, instructions, scripts, and utilities for converting Zipformer CR-CTC speech recognition to LiteRT.

These scripts convert [Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018](https://huggingface.co/Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018) (the official [icefall](https://github.com/k2-fsa/icefall) LibriSpeech recipe artifact, Apache-2.0) into a single `.tflite` graph for on-device speech recognition, and numerically verify every step against the PyTorch reference. The published artifacts live at [litert-community/Zipformer-medium-CR-CTC-LiteRT](https://huggingface.co/litert-community/Zipformer-medium-CR-CTC-LiteRT).

The model is a 64M-parameter Zipformer2 encoder (6 stacks at downsampling rates 1/2/4/8/4/2) with a pure CTC head, WER 2.12 / 4.62 (LibriSpeech test-clean / test-other, greedy). The whole encoder + CTC head runs as one GPU graph — the first Zipformer architecture on the LiteRT CompiledModel GPU. The host side computes kaldi-fbank features going in and greedy CTC + BPE-500 detokenization coming out.

## Pipeline

```
16 kHz wav in [-1, 1] ── kaldi-fbank (host) ──> fbank [1, 1600, 80]     ┐
valid-frame count ──> 4 additive attention biases [1, 796/398/199/100]  ┤─> [GPU] encoder + CTC ──> logits [1, 398, 500] ──> greedy CTC + BPE detok (host)
```

- **fbank**: 80 mel, povey window, `snip_edges=False`, `high_freq=-400`, dither 0, no CMN — icefall's kaldifeat options exactly. The wave must stay in **[-1, 1] scale** (×32768 integer scale produces garbage). Audio up to 16 s; shorter input is padded with `log(1e-10)` frames.
- **biases**: padding enters the graph as additive attention biases (0 = valid, −1000 = pad), one per downsampling rate, sliced on the host as `b[::ds]`. Valid 50 Hz frames = `(fbank_frames − 7) // 2`.
- **output**: raw CTC logits at 25 Hz, blank id 0. LogSoftmax is intentionally NOT in the graph (see the device findings below); greedy CTC is argmax-invariant. Valid output frames = `(valid50 + 1) // 2`.

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer
pip install torch torchaudio numpy

git clone https://github.com/k2-fsa/icefall   # model code only, no k2 build needed
hf download Zengwei/icefall-asr-librispeech-zipformer-medium-cr-ctc-20241018 --local-dir en_medium
```

Verified with `litert-torch==0.10.0`, `ai-edge-litert==2.1.5`, `ai-edge-quantizer==0.7.0`, `torch==2.12.0`, `torchaudio==2.11.0`, icefall @ `3f848bb`. k2 is NOT required: the script stubs the two Swoosh activation entry points with their pure-torch formulas before importing the model code.

## Run

```bash
KMP_DUPLICATE_LIB_OK=TRUE JAX_PLATFORMS=cpu \
    python build_zipformer_ctc.py all path/to/audio.wav   # gold + patches + convert + fp16 + parity
python verify_zipformer_ctc.py zipformer_ctc_fp16.tflite path/to/audio.wav
```

`build_zipformer_ctc.py all` emits `zipformer_ctc.tflite` (fp32, 260 MB) and `zipformer_ctc_fp16.tflite` (132 MB) — the deployment file published on Hugging Face — and prints the parity numbers at every stage. Any 16 kHz-compatible wav up to 16 s works as the test clip.

## Files

| File | What |
|---|---|
| `build_zipformer_ctc.py` | the re-authoring recipe: gold reference -> GPU patches -> parity -> litert-torch convert -> op-check -> fp16, each stage verified against the previous one. |
| `verify_zipformer_ctc.py` | end-to-end check of a converted `.tflite` through the CompiledModel API: wav -> transcript, plus correlation against the saved PyTorch reference. |

## Re-authoring → GPU-clean

The converted graph reproduces the patched model at corr **1.000000, max abs diff 0.0** (fp32). Every rewrite is numerically exact — the patched model matches the original icefall eval path at valid-region corr 0.999982 with an identical transcript, the residue being only the padded-window boundary:

| icefall construct | rewrite |
|---|---|
| SwooshL/R via `logaddexp` | guard-free stable softplus `relu(z) + log1p(exp(-abs(z)))` — the jax `logaddexp` lowering emits inf-guard SELECT_V2 ×244 + EQUAL + LOGICAL_AND chains |
| rel-position shift `as_strided` | pad + reshape + slice (verified diff 0.0) |
| `masked_fill(mask, -1000)` in attention | additive float bias input {0, −1000} — icefall itself uses −1000, so additive == fill |
| `masked_fill(mask, 0)` in ConvolutionModule | multiply gate `1 + bias/1000` ∈ {1, 0} |
| in-graph mask rate-slicing `[..., ::ds]` | 4 host-provided per-rate bias inputs (GATHER_ND ×3 otherwise) |
| `SimpleUpsample`/`SimpleDownsample` `expand` | `torch.cat` repetition (BROADCAST_TO ×6 otherwise) |
| `SimpleDownsample` `bias.softmax(dim=0)` | **baked** to a constant at pre-warm — litert-torch does not fold it, and a live rank-1 SOFTMAX on a constant fails the on-device compile |
| `scaling._no_op` = `x.chunk(1)[0]` | identity — `aten.split_with_sizes` is unlowerable in the jax bridge (`'list' has no .dtype`) |
| `.chunk(2/3)` in ConvolutionModule / NonlinAttention | slices (SPLIT is GPU-banned) |
| BiasNorm / `scaling.softmax` custom autograd | plain eval math (torch.export does not take the `is_tracing()` branches) |
| `CompactRelPositionalEncoding` attr mutation | pe cached by an eager pre-warm run; export takes the early-return slice path |
| `Conv2dSubsampling` `x_lens.max().item()` assert | forward re-copy minus the assert (GuardOnDataDependentSymNode at torch.export) |
| final `nn.LogSoftmax` | removed from the graph; log-softmax on the host when log-probs are needed (greedy CTC is argmax-invariant) |
| `convert_scaled_to_non_scaled` | called with `is_onnx=False` — `is_onnx=True` wraps modules in `torch.jit.script`, which breaks torch.export |

## Verification results

- Re-running this script reproduces the published files **bit-exactly** (sha256 match, fp32 and fp16; litert-torch 0.10.0, icefall @ `3f848bb`).
- Patched vs original icefall eval path: valid-region corr 0.999982, transcript identical.
- fp32 tflite vs patched torch: corr **1.000000**, max abs diff 0.0000; op-check GPU-clean (no banned ops, no >4D tensors, no FFT family).
- fp16 tflite vs patched torch: corr 1.000000, max abs diff 0.0043; transcript identical on a 4-wav sanity sweep.
- On device (Pixel 8a, Tensor G3, CompiledModel GPU): compile 1.8 s, **156 ms** run + readback per 16 s window (**RTF ≈ 0.01**). 4/4 test transcripts identical to the desktop gold; valid-region logits corr 0.9993 vs PyTorch, per-frame argmax agreement 99.16% — the single disagreeing frame has a reference margin of 0.168 (a genuinely ambiguous token boundary, normal fp16 noise).

## Two on-device findings (general)

Both ops below pass the standard desktop op-check (neither SOFTMAX nor REDUCE_MAX is in the banned set), then fail the on-device GPU compile fast (~1.3 s) with a bare "Failed to compile model" and no logcat detail. On both LiteRT 2.1.5 and 2.1.3, fp32 and fp16 — an op rejection, not a resource limit.

1. **Rank-1 SOFTMAX on a constant.** `SimpleDownsample`'s `self.bias.softmax(dim=0)` does not constant-fold in litert-torch, leaving SOFTMAX ops on shape-[2]/[4]/[8] constants. The GPU compiler rejects them. Fix: bake `softmax(bias)` into a buffer before export.
2. **REDUCE_MAX from `nn.LogSoftmax`.** The log-softmax lowering emits a rank-dropping max (`[1,398,500] -> [1,398]`), which the GPU compiler rejects. Fix: emit raw logits and log-softmax on the host — for CTC decoding the argmax is unchanged.

When a converted graph dies on device with an immediate bare "Failed to compile model", check for these two before anything else.

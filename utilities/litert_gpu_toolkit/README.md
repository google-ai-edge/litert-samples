# litert_gpu_toolkit

Pre-conversion patches that rewrite common PyTorch patterns into forms the
LiteRT GPU delegate accepts, plus a post-conversion checker.

Every entry below came out of converting a real model, hitting a wall on
device, and finding the rewrite that clears it. They were carried per-script
until now; this is the shared copy.

## Use

```python
from litert_gpu_toolkit import convert_for_gpu

path = convert_for_gpu(
    model,                                    # nn.Module, will be set to eval()
    dummy_input=torch.randn(1, 3, 1024, 1024),
    output_path="model.tflite",
)
```

`convert_for_gpu` applies the general patches, converts via litert-torch, and
runs `check_gpu_compatibility` on the result. The patches are also importable
individually from `litert_gpu_toolkit.patches` when a model only needs one.

`check_gpu_compatibility(tflite_path)` verifies through the LiteRT
CompiledModel API: it compiles the model for the GPU accelerator, runs every
signature on random inputs, and compares the outputs against a CPU-compiled
reference (`rtol`/`atol` default to 1e-2 — fp16 accumulation on GPU makes
bit-exactness unrealistic). When GPU compilation fails, the runtime's error
names the offending op; the patches table below maps the common ones to the
rewrite that clears them. When compilation succeeds and the numbers are still
wrong, nothing names an op — `bisect_gpu_divergence` does (next section).
Use both as a gate before spending a device run — they exercise the host GPU,
so still compare on-device output against CPU before shipping.

The checker needs no torch. On any `.tflite`:

```sh
python checker.py model.tflite                 # verify every signature
python checker.py model.tflite --bisect        # and name the first wrong op
python checker.py model.tflite --bisect --enforce-f32 --uniform --int-high 1000 --json out.json
```

`--uniform` draws float inputs from [0, 1) instead of N(0, 1) (image models;
the fp16 reduction overflow of #9249 only shows on non-negative input), and
`--int-high N` gives integer inputs random ids in [0, N) instead of zeros
(a token model fed all zeros returns the same vector on every backend, so
the comparison is vacuous). `--inputs file.npz` runs your own arrays.

## Finding the op the GPU gets wrong

A GPU silent miscompute looks like this: the graph compiles, reports full
residency, runs without error, and returns wrong numbers. The runtime has no
diagnostic for it. Every such bug we have reported against LiteRT
(#8593, #8599, #8619, #9249, #9272, and the LiteRT.js pair #9661/#9662) was
found the same way by hand: promote intermediate tensors to graph outputs,
compare GPU against CPU, walk forward until the first tensor that disagrees.
`bisect_gpu_divergence` is that procedure, automated. When the GPU compile
itself fails, the checker now also carries the runtime's own reason into the
result (the rejected op, or the kernel it could not create), which the
Python exception alone does not.

**How it works.** The graph is cut after op *k*: the prefix keeps ops
0..*k*, and every tensor still live at the cut (consumed later, or a graph
output, or produced by op *k*) becomes a graph output. The prefix is written
as a new `.tflite` (the flatbuffer utilities in the `ai-edge-litert` wheel;
weights are shared, not copied), compiled on CPU and on GPU with the same
inputs, and the live tensors are compared. Binary search over *k* finds the
first cut whose tensors disagree; the op at that cut is the first op the GPU
gets wrong. A 276-op graph takes 9 prefix compiles; a 1031-op graph 11.
Weight-only ops (a `DEQUANTIZE` of a constant) are never cut points and their
outputs are never compared.

One detail keeps the method honest. A frontier tensor that an op inside the
prefix also consumes is exposed through a same-shape `RESHAPE` copy, not
directly: made a graph output as-is it would form the "output that is also
consumed" pattern of #8599, which the Metal accelerator gets wrong on its own
(an `ADD` output read by a `SUM` comes back off by 3.85 at both precisions on
2.2.0, exact through the copy), and the bisect would report a divergence it
had caused. Tensors that were graph outputs in the original model keep their
native wiring, so a shipped #8599 case is still seen — as a divergence at the
cut that appends the consumer, with the producer's tensor marked
`context_dependent`. Both cases are pinned by tests. The difference is not
academic: without the copy, the rank-4 SAM 2.1 mask decoder export (correct
at fp32) reported a 100 % divergence at a `FULLY_CONNECTED` at fp16; with
it, the same run reports 2 % fp16 noise two ops later.

**What counts as wrong.** Intermediate tensors on a fp16 GPU carry errors of
about `rtol × (partial-sum magnitude)`, and an elementwise `allclose` flags
those on a correct kernel — efficientnet_b0's first `CONV_2D` fails
elementwise 1e-2 on Metal while the real fault is eight ops later. So the
default criterion is tensor-scale: a tensor diverges when its worst element
differs by more than `atol + rtol × max|cpu|`. When the full graph returns
NaN/Inf, the criterion switches to non-finite (`criterion="auto"`), because
an overflow poisons every op downstream and the first NaN is the fault. Both
can be overridden (`scale` / `nonfinite` / `elementwise`).

**What the result can and cannot claim.** It names the op at which the
divergence is *first observed*: the prefix ending one op earlier matched
CPU. Later ops are not examined, and a graph may hold more than one wrong op
(the SAM 2.1 mask decoder below holds three of the same kind). If the
diverging tensor was produced by an earlier op — it was correct while it was
the end of the graph and became wrong once op *k* consumed it — the result
marks it `context_dependent`; that is the shape of #8599, and it points at
how the runtime handles the pair, not at op *k*'s arithmetic. Run the bisect
twice, at the default fp16 and with `enforce_f32=True`: a kernel bug
reproduces at fp32 (#9272, the three entries below), fp16 accumulation loss
does not (#9249).

```python
from litert_gpu_toolkit import check_gpu_compatibility, bisect_gpu_divergence

report = check_gpu_compatibility("model.tflite", bisect=True)   # bisects every diverging signature
r = bisect_gpu_divergence("model.tflite", signature_key="serving_default", enforce_f32=True)
r["first_divergent_op"]   # {'index', 'name', 'version', 'inputs', 'outputs'} with shapes, dtypes, const flags
r["diverging_tensors"]    # per tensor: max_abs_diff, scale, nonfinite, producer op, context_dependent
r["probes"]               # every cut tried, in order
```

The report for `litert-community/efficientnet_b0` (LiteRT #9249):

```
  First divergent op: #8 SUM (v1) of 276 ops
    inputs : float32[1, 112, 112, 32], int32[2] const
    outputs: float32[1, 32]
    prefix ending at op #7 matched CPU
  Diverging tensors at that cut:
    t189 ... float32[1, 32]: max abs diff 6.515e+01 (tensor scale 4.957e+04), 6 non-finite
  Probes (9): #275:d, #137:d, #68:d, #33:d, #16:d, #7:c, #11:d, #9:d, #8:d
```

**Tests.** `tests/test_bisect.py` pins the bisect to three reported bugs and
asserts it names the op the issue describes: a synthetic 5-op graph carrying
the rank-3 `PAD` of #9272, a synthetic reduction and the public
`efficientnet_b0` for the fp16 `SUM` overflow of #9249 (the NaN channels are
checked to be exactly those whose CPU sum exceeds 65504), and the public
SAM 2.1 mask decoder pair of #8619. GPU tests skip where no LiteRT GPU
accelerator is available; the graph-analysis tests run anywhere.

```sh
pip install ai-edge-litert numpy pytest huggingface_hub   # huggingface_hub only for the public-model tests
cd utilities && python -m pytest litert_gpu_toolkit/tests -v
```

## Results on litert-community's most-downloaded `.tflite` files

<!-- sweep-results:start -->
The 20 repositories under `litert-community` with the most downloads that
publish a `.tflite` (Hub download counts, 2026-09-10), one file each: the
smallest non-vendor float file, or the int8 file where the float one is over
2 GiB, or the first transformer block of a multi-file pipeline. The four
files the reported issues name are appended below the 20. Every file was
run through `check_gpu_compatibility(bisect=True)` twice, at the default fp16
GPU precision and with `enforce_f32`, on ai-edge-litert 2.2.0 (CompiledModel,
Metal accelerator) on an Apple M4 Max, macOS 27.0. Inputs: uniform [0, 1)
floats and random token ids in [0, 1000), seed 0; `rtol = atol = 1e-2`. Host
GPU numbers — a phone can differ. A bisect of a 600–900 MB file takes 30–50
minutes per precision (each probe rewrites and recompiles the prefix).

Reading the two result columns: a cell that names an op is the first op
whose GPU result differs from CPU beyond `rtol × tensor scale`; **op fp32**
is the column that matters for a kernel fault. A file that diverges only at
fp16 and matches at fp32 is losing fp16 precision, and the op named at fp16
is where that loss first exceeds 1 % of the tensor's scale — `SUM` cells
with a NaN count are the fp16 overflow of #9249.

| # | model (litert-community) | file | MB | ops | GPU fp16 (default) | GPU fp32 (`enforce_f32`) |
|---|---|---|---|---|---|---|
| 1 | Qwen2.5-1.5B-Instruct | `Qwen2.5-1.5B-Instruct_seq128_q8_ekv1280.tflite` | 1571 | 1588 | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) |
| 2 | DeepSeek-R1-Distill-Qwen-1.5B | `DeepSeek-R1-Distill-Qwen-1.5B_seq128_q8_ekv1280.tflite` | 1807 | 1588 | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) |
| 3 | Phi-4-mini-instruct | `Phi-4-mini-instruct_seq128_q8_ekv1280.tflite` | 3871 |  | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) |
| 4 | Qwen2.5-0.5B-Instruct | `Qwen2.5-0.5B-Instruct_seq128_q8_ekv1280.tflite` | 513 | 1360 | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) |
| 5 | FLUX.2-klein-4B-LiteRT | `kc_double0.tflite` | 739 | 706 | op 56 `FULLY_CONNECTED` [1×256×3072, const 3072×3072] → 1×256×3072 (2% of scale) | op 204 `FULLY_CONNECTED` [1×256×3072, const 18432×3072] → 1×256×18432 (2% of scale) |
| 6 | whisper-tiny | `whisper_tiny_30s_f32.tflite` | 151 | 82 | `encode`: op 14 `GELU` [1×384×1500] → 1×384×1500 (2% of scale); `decode`: op 94 `FULLY_CONNECTED` [128×384, const 384×384] → 128×384 (2% of scale) | match (max abs diff 7e-03) |
| 7 | parakeet-tdt-0.6b-v3 | `parakeet_tdt_0.6b_v3_5s_i8.tflite` | 614 | 1790 | `encode`: op 6 `CONV_2D` [1×125×32×256, const 256×1×1×256] → 1×125×32×256 (2% of scale); `decode`: op 23 `SLICE` [64×1×2560] → 1×1×2560 (1% of scale) | `encode`: op 58 `CONV_2D` [1×63×1×1024, const 2048×1×1×1024] → 1×63×1×2048 (2% of scale); `decode`: op 79 `LOGISTIC` [1×1×640] → 1×1×640 (3% of scale) |
| 8 | Bonsai-Image-ternary-4B | `vae_dec_fp32.tflite` | 199 | 882 | op 43 `MUL` [1×32×1×1, const 1×32×16×1] → 1×32×16×1 (100% of scale); op 35 `RESHAPE` output changes once this op is appended · some ops on CPU | match (max abs diff 2e-05) · some ops on CPU |
| 9 | moonshine-tiny | `moonshine_tiny_5s_f32.tflite` | 109 | 358 | `encode`: op 77 `STABLEHLO_COMPOSITE` [1×207×8×40, 1×207×8×40, 1×207×8×40] → 1×207×8×40 (106% of scale); `decode`: fp16 noise only (within 1% of tensor scale) | `encode`: op 77 `STABLEHLO_COMPOSITE` [1×207×8×40, 1×207×8×40, 1×207×8×40] → 1×207×8×40 (106% of scale) |
| 10 | Z-Image-Turbo-LiteRT | `zc_main0.tflite` | 908 | 815 | op 144 `FULLY_CONNECTED` [1×288×10240, const 3840×10240] → 1×288×3840, 2 NaN/Inf, 278 Inf where CPU exceeds fp16 range | op 139 `FULLY_CONNECTED` [1×288×3840, const 10240×3840] → 1×288×10240 (1% of scale) |
| 11 | embeddinggemma-300m | `embeddinggemma-300M_seq256_mixed-precision.tflite` | 179 | 2265 | op 52 `SIN` [1×256×1×128] → 1×256×1×128 (12% of scale) | op 127 `MUL` [1×256×768, 1×256×1] → 1×256×768 (4% of scale) |
| 12 | SmolLM-135M-Instruct | `SmolLM-135M-Instruct_seq128_f32_ekv1280.tflite` | 547 | 1702 | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) |
| 13 | yolox-nano-litert | `yolox_nano.tflite` | 2 | 482 | op 18 `CONV_2D` [1×104×104×32, 16×1×1×32] → 1×104×104×16 (2% of scale) | match (max abs diff 2e-04) |
| 14 | Gecko-110m-en | `Gecko_256_f32.tflite` | 443 | 633 | op 20 `MUL` [1×256] → 1×256, 34 NaN/Inf; Inf first at op 19 `SUM` · some ops on CPU | match (max abs diff 1e-07) · some ops on CPU |
| 15 | Matcha-TTS | `matcha_decoder_fp16.tflite` | 23 | 1031 | op 77 `MUL` [1×8×32] → 1×8×32, 53 NaN/Inf; Inf first at op 76 `SUM` | op 161 `MUL` [512×1024, const 1×1×1024] → 1×512×1024 (93% of scale) |
| 16 | Qwen3-TTS-12Hz-0.6B-Base | `codec_decoder_fp32.tflite` | 457 | 1023 | op 635 `GELU` [1×128×4096] → 1×128×4096 (1% of scale) · some ops on CPU | op 637 `MUL` [128×1024, const 1×1×1024] → 1×128×1024 (81% of scale) · some ops on CPU |
| 17 | parakeet-ctc-0.6b | `parakeet_ctc_0.6b_5s_i8.tflite` | 596 | 1578 | **GPU compile failed**: kernel creation refused | **GPU compile failed**: kernel creation refused |
| 18 | TinyLlama-1.1B-Chat-v1.0 | `TinyLlama-1.1B-Chat-v1.0_seq128_q8_ekv1280.tflite` | 1119 | 1246 | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) | **process crash** (SIGSEGV in CPU `DYNAMIC_UPDATE_SLICE` of the GPU\|CPU run) |
| 19 | whisper-acft | `tiny/acft_whisper_tiny_30s_drq.tflite` | 61 | 322 | `decode`: op 30 `ADD` [6×128×128, const 1×1×128×128] → 1×6×128×128, 48768 Inf where CPU exceeds fp16 range; Inf first at op 30 `ADD`; `encode`: op 8 `CONV_2D` [1×1×3002×384, const 384×1×3×384] → 1×1×1500×384 (1% of scale) · some ops on CPU | `decode`: op 32 `SOFTMAX` [6×128×128] → 6×128×128 (99% of scale); `encode`: op 18 `MUL` [1×1500×384, 1×1500×384] → 1×1500×384 (1% of scale) · some ops on CPU |
| 20 | Qwen3-ASR-0.6B | `qwen3_asr_0.6b_5s_i8.tflite` | 794 | 471 | `encode`: op 306 `STABLEHLO_COMPOSITE` [1×63×14×64, 1×63×14×64, 1×63×14×64] → 1×63×14×64, 832 NaN/Inf; `decode`: op 86 `MUL` [1×134×1024, 1×134×1] → 1×134×1024 (1% of scale) · some ops on CPU | `encode`: op 66 `FULLY_CONNECTED` [63×896, const 896×896] → 63×896 (2% of scale); `decode`: op 86 `MUL` [1×134×1024, 1×134×1] → 1×134×1024 (1% of scale) · some ops on CPU |
| issue | efficientnet_b0 (#9249) | `efficientnet_b0.tflite` | 21 | 276 | op 8 `SUM` [1×112×112×32] → 1×32, 6 Inf where CPU exceeds fp16 range | match (max abs diff 4e-05) |
| issue | SAM2.1-Hiera-Tiny-Mask-Decoder (#8619) | `sam2_tiny_mask_decoder_fp16.tflite` | 17 | 378 | op 41 `ADD` [4096×256, const 4096×256] → 4096×256 (92% of scale) | op 41 `ADD` [4096×256, const 4096×256] → 4096×256 (92% of scale) |
| issue | SAM2.1-Hiera-Tiny-Mask-Decoder (#8619) | `sam2_tiny_mask_decoder_v2_fp16.tflite` | 17 | 425 | op 236 `FULLY_CONNECTED` [8×2048, 256×2048] → 8×256 (2% of scale) | match (max abs diff 6e-05) |
| issue | PP-OCRv5-LiteRT (#9661) | `ppocr_rec_fp16.tflite` | 17 | 827 | op 310 `CONV_2D` [1×12×80×240, 240×1×1×240] → 1×12×80×240 (2% of scale) | op 705 `BATCH_MATMUL` [8×40×15, 8×15×40] → 8×40×40 (128% of scale) |

What the table shows, and the controls behind each claim:

- **Rank-2 activations against a constant are wrong on Metal, and it is
  common.** Three of the files diverge at fp32 at an elementwise op whose
  activation is rank 2 and whose other operand is a constant: the SAM 2.1
  mask decoder v1 (`ADD([4096,256], [4096,256] const)`, three sites), the
  Matcha-TTS decoder (`MUL([512,1024], [1,1,1024] const) -> [1,512,1024]`,
  six sites) and the Qwen3-TTS codec decoder (same MUL shape family, two
  sites). As single ops both forms reproduce (max abs diff 7.4 and 14 on
  N(0,1) inputs) and the same op with the activation at rank 3 is bit-exact.
  Reshaping the activation to `[1, N, C]` at only those sites takes each whole
  model to CPU parity at fp32: SAM 2.1 v1 18.9 → 1.07e-4 (the rank-4 v2
  export measures 1.07e-4), Matcha 3.9 → 1.3e-5, Qwen3-TTS codec 0.20 →
  1.8e-6. For #8619 this means that on this backend the v1/v2 difference is
  those three adds, not the attention rank; the Pixel 8a result in the issue
  is not re-measured here.
- **`BATCH_MATMUL` with output width 40 is wrong.** The PP-OCRv5 recognizer
  (#9661's file, here on native Metal rather than WebGPU) first diverges at
  fp32 at its attention scores `BATCH_MATMUL [8,40,15] × [8,15,40]`, and
  moonshine-tiny's encoder at its `odml.scaled_dot_product_attention`
  composite, whose decomposition holds `BATCH_MATMUL [8,207,207] × [8,207,40]`
  (head dim 40). Single-op probes against numpy are wrong for N = 40 exactly
  when M mod 64 is in 1…61 (M = 8, 40, 56, 65, 96, 207, 1000, 1025 wrong;
  62, 63, 64, 128, 192, 256, 1024 right) and the batch is at least 2, at both
  precisions; K does not matter (1…256), batch 1 is always right, and every
  other N tried (4…36, 41, 44…264) is right. The mechanism is not known; the
  extent stated is exactly the shapes probed.
- **`SUM` overflows fp16 in three different places.** efficientnet_b0 (#9249:
  6 of 32 channels, exactly those whose CPU sum exceeds 65504), the Matcha
  decoder (the first `SUM` over `[1,8,32,512]`, tensor scale 6.4e4, 53 NaN in
  the `MUL` that consumes it) and Gecko's LayerNorm variance `SUM` over
  `[1,256,768]` (scale 6.4e4, 34 NaN in the next `MUL`) return NaN at fp16
  and verify with `enforce_f32`. The report names the consumer and points
  back at the `SUM` ("Inf first at op N"), because the `SUM`'s own Inf is
  expected once the CPU total is past the fp16 range.
- **embeddinggemma's fp16 output is all zeros; its first fp16 deviation is
  the rotary `SIN`.** At fp16 the bisect stops at op 52, `SIN` of the
  position angles (12 % of tensor scale), and the output ends all-zero. At fp32 the output is finite with cosine
  0.989 against CPU and the first deviation above 1 % of tensor scale is
  the RMSNorm `MUL` after the int4 `FULLY_CONNECTED` layers, the same
  reference gap as the int8 files below (this export is int4/int8
  mixed-precision) — no kernel fault is claimed. Qwen3-ASR's int8 encoder
  behaves the same way at fp32 (first 1 % crossing at an int8
  `FULLY_CONNECTED`), and at fp16 its `odml.scaled_dot_product_attention`
  composite returns 832 NaN in an in-range output; the bisect does not open
  composites.
- **The LLM `_seq128_*_ekv1280` exports crash the process.** `Qwen2.5-1.5B-Instruct`,
  `DeepSeek-R1-Distill-Qwen-1.5B`, `Phi-4-mini-instruct`, `Qwen2.5-0.5B-Instruct`
  and `SmolLM-135M-Instruct` (MediaPipe-style exports, q8 and f32) do not compile GPU-only
  (`CAST` int64, `GATHER_ND`, `GREATER_EQUAL`/`LESS_EQUAL` with const
  inputs); the GPU|CPU compile succeeds with 57 ops on CPU, and the run then
  segfaults in the CPU-resident `DYNAMIC_UPDATE_SLICE` on both signatures.
  CPU-only runs of the same files pass. The bisect cannot start on them, and
  Phi-4's 3.9 GB file is past the 2 GiB the flatbuffer writer can re-emit.
- **On int8-weight files the CPU reference is the approximate side.** FLUX.2
  klein (`kc_double0`) and Z-Image Turbo (`zc_main0`) first cross 1 % of
  tensor scale at fp32 at an int8 `FULLY_CONNECTED`. Taken out as single ops
  with their own weights (FLUX ops 56 and 204, Z-Image ops 139 and 144), the
  GPU at fp32 is bit-identical to CPU float math on the dequantized weights
  (max abs diff 0.0 on all four), while the CPU run of the same int8 op is
  0.2–0.3 % off that float reference. A whole-model float-weight control was
  not possible (the dequantized files exceed 2 GiB). At fp16 the same ops
  are 1.2–2.9 % off float, and Z-Image's op 144
  output reaches 6.6e4 on CPU, past the fp16 range, so its fp16 run returns
  NaN from there. parakeet-tdt's int8 export crosses 1 % at an int8
  `CONV_2D` in the encoder and at a `LOGISTIC`/`SLICE` fed by int8 layers in
  the decoder, the same class. The DRQ whisper-acft file behaves the same
  way (encoder within 3e-3 of a float-weight copy on a scale of 17 while the
  CPU DRQ path is off by 14.6); its decoder disagrees between all three
  references (at fp16 an unmasked attention logit differs by 74 % of scale
  at the mask add, which its fp32 run does not show) and is not claimed.
- **The Bonsai VAE decoder is wrong at fp16 inside its first GroupNorm.**
  The whole decoder is off by 0.58 at fp16 and bit-exact at fp32. The bisect
  pins it to the `[1,32,1,1]` chain that closes the first GroupNorm: the
  per-channel mean (values 0.06–8.5, inside the fp16 range) reads back as
  `-inf` in 26 of 32 channels and the `rsqrt(var) × gamma` product as all
  zeros. Which op is named depends on how the mean is exposed: op 43 with
  the RESHAPE copy, op 44 without it. No single-op repro was attempted.
- **parakeet-ctc's int8 export is refused at kernel creation** with `Unable
  to parse bc coord for BATCH axis ... shape {bhwc, {63, 1, 1, 1024}}`, the
  same message as #9277 (reported on Mali). The shape matches the rank-2
  `[63,1024]` activations of its attention projections' int8
  `FULLY_CONNECTED` ops; the error does not name the op, so that attribution
  is by shape. Nothing can be measured on it; the bisect cannot start.
<!-- sweep-results:end -->

## What each patch is for

| Patch | Pattern it rewrites | Why |
|---|---|---|
| `patch_safe_layernorm` | LayerNorm variance | The delegate reduces `Σ(x−mean)²` in fp16 even for an fp32 graph. On deep ViTs and on deep-residual CNNs the activations get large enough to overflow fp16, and the error compounds with depth while the model still reports full delegation. The fix reduces in a down-scaled domain `x/S`, which LayerNorm's scale-invariance makes safe. Three modes: `adaptive_v2` (default, per-row `S = max(1, amax/8)`, stays in the scaled domain — cheapest, but eps then acts at the scaled magnitude, so it is not bit-faithful to stock LayerNorm), `adaptive` (same `S`, rescales before the rsqrt — bit-faithful), `fixed` (constant `S`) |
| `patch_rmsnorm` / `safe_rms` | RMSNorm | Same overflow, different symptom: `Σx²` overflows, `norm` becomes `inf`, and the whole head outputs exactly zero |
| `patch_instance_norm` / `SafeInstanceNorm2d` | InstanceNorm | Same class |
| `hierarchical_mean` | `x.mean((2, 3))` | A single global reduction over a large map overflows the fp16 accumulator. Replaced by a cascade of ÷2 average-pools, so each stage averages at most four values, and it traces to a static chain of `AVERAGE_POOL_2D`. **Exact only for power-of-two spatial dims** — with odd extents the `ceil_mode` edge windows average fewer elements (max error ~0.2 measured on a 37×53 map). Pad first, or keep the map pow2 |
| `ZeroStuffConvT1d` / `2d`, `patch_conv_transpose`, `pixelshuffle_to_conv_transpose` | `ConvTranspose`, `PixelShuffle` | `PixelShuffle` lowers through a rank-6 reshape; `TRANSPOSE_CONV` is emitted at a version the delegate does not accept. Zero-stuff plus a plain conv is exact (~1e-7) and stays rank 4 |
| `patch_grid_sample` | `F.grid_sample` | Lowers to `GATHER_ND`, which the delegate rejects. Rewritten as a bilinear tent-matmul, exact against `F.grid_sample` including zeros-padding out of bounds (error ~2e-7), all rank ≤ 4. Cost is O(HW²), so it suits sparse sampling — it is what carries RF-DETR Nano's deformable cross-attention — not dense warps |
| `patch_window_attention`, `patch_patch_merging`, `patch_einops` | Windowed attention, patch merging, `einops.rearrange` | These build rank-5/6 tensors, or stride-2 slices that become `GATHER_ND` |
| `patch_maxpool_zeropad` / `ZeroPadMaxPool` | `MaxPool` padding | Padding lowers to `GATHER_ND` |
| `patch_groupnorm` / `ManualGroupNorm` | `GroupNorm` | Manual 4-D form |
| `patch_normalize` | `F.normalize` | The div-broadcast form fails on the delegate |
| `patch_interpolate` | `F.interpolate(align_corners=True)`, bicubic | The delegate bans half-pixel with align_corners; bicubic becomes `GATHER_ND` |
| `patch_weight_standardization` | `Conv2d` weight standardization | Bake the standardized weights instead of normalizing at runtime |
| `patch_gelu`, `patch_swish` | GELU, Swish | Defensive, and off by default in `convert_for_gpu`'s general set where the native op is already correct. Note that an approximation choice can matter: a wide-output-range regression head needs the tanh form, not the sigmoid one |

## Verification

The patches are device-derived, not theoretical — the rewrites here are the
ones that took specific models to full GPU residency with correct output on a
Pixel 8a (for example a CLIP ViT-B/32 at 691/691 nodes, CPU-identical).

Two cautions that apply to all of them:

- **Full residency does not mean correct output.** Several of these patches
  exist for bugs where the delegate reports `N/N` nodes and still returns wrong
  numbers. Always compare the on-device output against CPU or the source model.
- The rewrites are numerically exact by construction where stated (zero-stuff
  conv, scale-before-square) and approximate where stated (grid_sample, the
  GELU/Swish substitutions). Check the table before assuming.

## Requirements

`torch` and `litert-torch` for the conversion patches; `numpy` and
`ai-edge-litert` for the checker and the bisect (they run without torch).

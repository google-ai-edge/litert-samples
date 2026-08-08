---
name: accuracy-safe-quantization
description: Shrink a converted LiteRT model with ai-edge-quantizer (fp16 / int8 / int4) without losing accuracy, verifying parity against the float source after every step. Use when choosing a quantization recipe for a new model, when a quantized model fails to load, degrades on a task benchmark, or degenerates over long generations, or when deciding between dynamic-range, weight-only, and blockwise variants.
---

# Accuracy-safe quantization

A quantization is done when three things hold, in this order:

1. it exports and the file shrinks by what the recipe predicts,
2. **output parity with the float source holds on a task-level check**, not
   just a smoke test,
3. the quantized model still passes the deployment check on the target
   runtime and device.

Quantization rewrites the graph, so step 3 is a fresh obligation every time:
re-run the same CompiledModel verification you used to accept the float
conversion (see the `gpu-clean-conversion` skill), then the on-device
numerical check.

All recipes below are `ai-edge-quantizer` (`pip install ai-edge-quantizer`),
plain Python, no build step. Worked examples live in this repo under
`models/bonsai/bonsai_image_4b/converted/` and
`models/qwen/qwen3_tts/converted/`.

## Choosing a lane

Start with the lightest recipe that meets the size budget, and move down
only on evidence:

| Budget / model | Recipe |
|---|---|
| ~2× smaller, zero risk | **fp16 float-casting.** Weights cast to fp16, compute stays float. On a GPU that already computes in fp16 this is close to free numerically — verify anyway |
| ~4× smaller — encoders, conv nets, diffusion blocks | **Dynamic-range int8 channelwise.** int8 weights, float activations; this shape rides the GPU delegate |
| Dynamic int8 lost quality (conditioning, embeddings) | **Weight-only, same bits.** Inserts an explicit DEQUANTIZE so the matmul runs in float and activations are never quantized — more quality, some latency |
| ~7× smaller — LLM / autoregressive decoders | **int4 blockwise-32 + OCTAV, embeddings int8.** Never channelwise for a decoder: it looks fine on short outputs and degenerates over long generations |
| Data-free int4 still fails the task gate | **Calibrated ingest.** Take a GPTQ checkpoint and preserve its grid with `DEQUANTIZED_WEIGHT_RECOVERY` — see the routing table |

Full-integer static quantization (`static_wi8_ai8` — quantized activations,
calibration data required) is a different lane aimed at NPU/AOT targets and
is not covered here.

## Recipes

Recipes layer by regex: broad rule first, narrow overrides after — that is
how one file mixes lanes (bonsai's DiT puts everything at int8 channelwise,
then overrides `.*TransformerBlock_.*` to int4 blockwise).

fp16 float-casting:

```python
from ai_edge_quantizer import quantizer, recipe_manager
from ai_edge_quantizer.recipe import AlgorithmName, qtyping

rm = recipe_manager.RecipeManager()
rm.add_quantization_config(
    regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
    op_config=qtyping.OpQuantizationConfig(
        weight_tensor_config=qtyping.TensorQuantizationConfig(
            num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
        compute_precision=qtyping.ComputePrecision.FLOAT),
    algorithm_key=AlgorithmName.FLOAT_CASTING)
quantizer.Quantizer("model_fp32.tflite", rm.get_quantization_recipe()) \
    .quantize().export_model("model_fp16.tflite")
```

Dynamic-range int (swap bits / granularity / algorithm per the table):

```python
from ai_edge_quantizer.qtyping import QuantGranularity as G
from ai_edge_quantizer.qtyping import TFLOperationName as OP

rm = recipe_manager.RecipeManager()
rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED,
                      num_bits=4, granularity=G.BLOCKWISE_32,
                      algorithm_key=AlgorithmName.OCTAV)
rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP,
                      num_bits=8, granularity=G.CHANNELWISE)
```

Weight-only uses the same signature via `rm.add_weight_only_config(...)` —
`models/bonsai/bonsai_image_4b/converted/quantize_weight_only.py` wraps it
as a reusable CLI.

`ai_edge_quantizer.recipe` also ships these as presets
(`dynamic_wi8_afp32()`, `dynamic_wi4b32_afp32()`, `weight_only_wi8_afp32()`,
…). The litert-torch LLM exporter accepts a preset name as its
`quantization_recipe` argument, and a custom recipe can be registered by
assigning a callable onto the module — the qwen3_tts talker recipe
(`models/qwen/qwen3_tts/converted/export_talker.py`) registers `BOCTAV4`
(blockwise-32 OCTAV int4 + int8 embeddings) that way.

## Verify after every step

- **Size first.** fp16 ≈ ½, int8 ≈ ¼, int4 blockwise ≈ ⅐ of fp32 (block
  scales add overhead). If the file did not shrink as predicted, the regex
  did not match — fix that before measuring anything.
- **Parity against the float reference.** Same inputs through the float and
  quantized models; correlation on outputs plus the task-level check
  (argmax match, token-for-token greedy decode, IoU).
- **A smoke gate is a floor, not a parity verdict.** An LLM can pass most of
  a handful of chat prompts and still score near zero on a real benchmark.
  Before publishing an int4 decoder, run a task benchmark at real length
  (e.g. GSM8K-style, n≥100) against the float baseline.
- **Long generations, specifically.** Granularity problems do not show up
  in short outputs.
- **On the target device.** Host emulation of int kernels is pessimistic —
  int8 graphs have scored visibly worse on host CPU than the same graphs on
  the device GPU delegate. Never reject a recipe on desktop numbers alone;
  never accept one without device numbers.

## When it breaks or degrades

| What you see | Knob to turn |
|---|---|
| Runtime refuses to load: `unsupported scale value (0.000000) … for INT4 tensor` | Sparse weights produced all-zero blocks, whose min-max scale is 0. Patch each zero scale to the tensor's smallest nonzero scale — dequantization is unchanged because those blocks are all zero. ⚠ Blockwise scales live in separate fp16 scale tensors, **not** `QuantizationParameters.scale` — patching the latter via the flatbuffer object API succeeds silently and changes nothing; edit the scale tensor's buffer directly. `models/bonsai/bonsai_image_4b/converted/fix_zero_block_scales.py` |
| Dynamic-range int8 collapses a conv net outright (near-zero output correlation, every input misclassified) while the file loads and runs fine | Activation-quantization sensitivity — squeeze-excite and SiLU-family conv nets are the known class. **Weight-only int8 at the same size is typically near-lossless on the same model.** This collapse has shipped inside published artifacts, so parity-check any dynamic-int8 model you did not gate yourself before building on it |
| Decoder is coherent for a while, then degenerates | Channelwise → `BLOCKWISE_32`; `MIN_MAX_UNIFORM_QUANT` → `OCTAV` |
| Dynamic-range lost fidelity (prompt conditioning, embeddings) | Weight-only at the same bits |
| int4 fails the task gate at block-128 | Block-32. Data-free block-128 can collapse outright on small models |
| int4 fails the task gate at block-32 too | Data-free min-max/OCTAV has hit its limit for this family. Ingest a calibrated GPTQ checkpoint: dequantize it, then quantize with `algorithm_key=AlgorithmName.DEQUANTIZED_WEIGHT_RECOVERY` at the granularity matching the GPTQ group size (gs128 → `BLOCKWISE_128`). Symmetric checkpoints only, `desc_act=False` only |
| Recovery raises `NOT dequantized (fake-quantized) weights` | That tensor was never on the GPTQ grid (`lm_head`, tied embeddings, first/last layers). The raise is a triage signal, not a bug: route the named tensor to a plain int8 entry by regex |
| A specific head or block is the culprit | Exclude it by regex — keep it at int8 or float and leave the rest at int4 |
| Everything above still degrades | fp16 float-casting is the floor. If fp16 fails parity, the problem is upstream of quantization — go back to `gpu-clean-conversion` step 5 |

Some models are genuinely 4-bit sensitive — small reasoning-distilled
decoders (~1–2 B) often fail int4 quality gates that instruct-tuned peers
and larger models pass. When int4 fails on quality, ship int8 as the
quality row rather than forcing it; int4 becomes a speed reference.

## Watch for

- **Embeddings stay int8** even in int4 recipes — both shipped LLM-lane
  recipes in this repo do this deliberately.
- **Bytes are not speed.** int4's latency win depends on the backend's
  kernel efficiency: the same model can gain ~1.5× on one device and
  barely 1.1× on another. Measure on the target; don't project from
  file size.
- **Check whether the container is exact.** Ternary weights land in int4
  blockwise as exactly {-7, 0, +7} — zero rounding error. When the weight
  distribution matches the container, parity is free; verify it rather
  than budgeting for loss that isn't there. For exact-container cases use
  **min-max, not OCTAV** — OCTAV's clipping optimization can move a grid
  that min-max reproduces exactly.
- **int2 is a container without a consumer** (as of 2026-08): the schema
  type and the blockwise packer exist (2.125 bits/weight at block-128),
  but the CPU runtime refuses the tensor type at prepare — a hard load
  failure, not degradation. Don't spend time there until a kernel ships.
- **Auxiliary tables cast to fp16 need the same discipline.** Casting
  host-side embedding/projection tables halves them; verify generated
  outputs are unchanged before shipping (qwen3_tts did, and it held).
- **Pin the toolchain.** Quantized-graph compatibility moves with the
  runtime; a graph exported from a dev checkout can fail GPU kernel
  initialization on a release runtime. Record `ai-edge-quantizer` /
  `litert-torch` versions in the recipe README next to the numbers.

## Output layout

Quantization extends the model recipe from `gpu-clean-conversion`; it does
not get its own tree:

```
models/<family>/<model>/converted/
  export_*.py              float export (existing)
  quantize_*.py            one script per quantized variant
  verify_*.py              parity checks, reused for every variant
  README.md                recipe, sizes, parity numbers, gate results,
                           device, toolchain versions
```

Keep each variant separately re-runnable. State which variant is the
quality row and which is the speed row when they differ. Weights are not
committed.

# Architecture walls — where a structure dies, and the signature it dies with

Status is stated against **litert-lm 0.15.0 / litert-torch 0.9.3 /
litert-converter 0.3.0 released** (2026-08). Every entry is a dated fact,
not a law: re-test on each release. Walls fall (hybrids became executable
in 0.15) and regressions appear (pre-0.15 hybrid bundles stopped running
on 0.15) — sometimes in the same release.

## Runtime / engine walls

| Structure | Verdict | Failure signature | Route |
|---|---|---|---|
| Dense decoder (Llama, Qwen, Phi, Mistral, Gemma, SmolLM, Falcon-dense, OLMo, Granite-dense) | **Works** — the baseline lane | — | `recipe-selector.md` §Dense |
| Looped/unrolled transformer (shared weights, `num_loops`, N×layers KV slots) | **Works** with a custom cache registration (one KV slot per (loop, layer)) | without it: cache built with layers-only slots → wrong shapes | §Dense, looped variant |
| Interleaved hybrid — SSM / linear-attention / short-conv layers between attention layers (LFM2/2.5, Granite-4.0-h Mamba2, Qwen3.5 GatedDeltaNet) | **Executable since litert-lm 0.15.0, CPU** — the executor binds arbitrary named states via the bundle's executor-metadata section (`TYPE_LINEAR_ATTENTION` is an opaque pass-through that carries conv + recurrent/SSM state) | on ≤0.14: engine creation fails `FAILED_PRECONDITION: No KV cache inputs found` (the legacy path keys on layer-0 KV names that hybrids don't have) | `recipe-selector.md` §Hybrids — state-aware export + metadata section |
| The same hybrids, exported with **released litert-torch (≤0.9.3)** | Bundle lacks the executor-metadata section → **loads, then dies at first generation on 0.15** | `NOT_FOUND: The given map is missing some output TensorBuffers (llm_litert_compiled_model_executor.cc:708)`; the same file runs on 0.14 | retrofit the section (§Hybrids, "executor metadata"); dense bundles are unaffected (legacy path still works) |
| Parallel hybrid — mamba **and** attention inside every layer (Falcon-H1) | **Wall** — one layer needs KV cache and conv/SSM state simultaneously; the interleaved-hybrid cache treatment doesn't cover it | same `No KV cache inputs found` class at engine creation | park until a combined cache layer exists |
| MLA — latent/asymmetric K-V cache (DeepSeek family, MiniCPM3) | **Wall at the engine** — converts, and CPU interpreter execution has been demonstrated, but the engine's cache orchestration doesn't support the asymmetric layout | engine rejects / cannot bind the cache | park; dense distills of the same models convert fine |
| MoE via the `tfl.custom` `moe` kernel | **Wall for SiLU families.** The kernel ships in the 0.15 delegates but is GELU-only, fp32/int8-only, renormalized-top-weights-only — OLMoE / Qwen3-MoE / Granite-MoE (SiLU) stay blocked | CPU reference: `Node number N (moe) failed to prepare.`; export side often dies earlier: `ConcretizationTypeError ... aten._local_scalar_dense` (data-dependent `expert_size.tolist()` in gating) | park, or convert the dense sibling |
| Multi-conversation reuse of one engine by **state-carrying models** (running conv/SSM/linear-attention state) | Engine-level hazard: conversations sharing an engine can interact through prefix caching; per-conversation state assumptions break | quality derails after several conversations on one engine while a fresh engine per conversation is clean | gate hermetically (fresh engine per measurement) **and** probe one shared-engine sequence — `verification-gates.md` §Sweep |

## Load-time / size walls

| Structure | Verdict | Failure signature | Route |
|---|---|---|---|
| Any single bundle section > ~2 GiB, loaded on iOS | **Hard wall — file section size, not device RAM** | `Failed to map section ... Failed to map, error: Cannot allocate memory (memory_mapped_file_posix.cc)` then `NOT_FOUND: TF_LITE_PREFILL_DECODE not found` | `externalize_embedder=True` (embedding becomes its own section, also dedups tied embeddings); block-128 int4 as the second lever. macOS loads the same file fine |
| fp16 weight bundles on phones | Loads, then memory-spikes: the CPU runtime unpacks fp16→fp32 in RAM, per signature subgraph | jetsam / OOM kill during init at multi-GB peak | treat fp16 as the desktop/quality variant; ship int8 for phones |

## Export-time (tracing/lowering) walls — all fixable in the export driver

| Trigger | Failure signature | Fix |
|---|---|---|
| `torch.eye(n)` inside a traced compute path (e.g. chunked delta-rule) | lowers to `STABLEHLO_IOTA`, unregistered in every released kernel set → interpreter and engine both die **at load** | build the eye from a Python list so it lifts as a plain graph constant; `triu`/`arange` masks constant-fold fine |
| `arange(..., dtype=torch.uint8)` in modeling code | `RuntimeError: torch.uint8` in exported_program_to_mlir | patch the dtype to int32 (numerically identical) |
| float64 in a constant path (rope tables computed in f64) | `failed to legalize operation 'tfl.pow' ... tensor<f64>` | force f32 for the table computation |
| int64 tensors surviving into the graph (masks, `.sum()` of bool) | export abort, or GPU delegate rejection of INT64 ops | cast to int32/f32 at the source; this family is *dtype in a constant path*, never the model math |
| Data-dependent control flow (dynamic expert split, `torch.all(mask)` guards, dynamic rope guards) | `ConcretizationTypeError` / guard errors under torch.export | patch the guard out (return the static branch) — verify output-neutrality on the eager model first |
| transformers `from_pretrained` meta-load dropping `__init__`-computed non-persistent buffers | rotary `inv_freq` silently all-zero → cos=1/sin=0 → output plausible but wrong, **no error** | recompute the buffer post-load; assert `inv_freq > 0` before export |

## Backend walls (a graph that runs on CPU can still die here)

| Backend | Wall | Signature | Route |
|---|---|---|---|
| Every released GPU delegate | `odml.softmax` StableHLO composite (emitted unconditionally by litert-torch ≥0.9.2 attention) not lowered by litert-converter 0.3.0 | `not fully delegated` → engine creation fails; CPU unaffected | **fixed in litert-converter ≥ 0.4.0 dev builds** — the converter lowers the composite to the fused SOFTMAX builtin. On 0.3.0, strip the composite marker at export (math unchanged; verified op-identical to the fixed converter's output) |
| Mobile GPU (ML Drift) | `GATHER_ND` — hard blocker, often invisible on desktop | executor create fails on device naming the op | rewrite the source: raster-order processing, strided-slice merges, one-hot-matmul instead of dynamic gather |
| GPU delegate (0.15) | high-rank ops from SSM scans (`SLICE` rank > 4) | rejection naming the op | ship CPU-only; re-test per release |
| iOS Metal specifically (0.14–0.15) | shader codegen bugs distinct from Mac/Android: `half4`×`float4` implicit-conversion error in generated Metal source; `cond_tensor` BATCH-axis parse failure | `newLibraryWithSource: ... implicit conversions between vector types`; `Unable to parse bc coord for BATCH axis` | **Mac GPU pass ≠ iPhone GPU pass** — same file can fully delegate on Mac WebGPU and fail on Metal. Gate on the actual device; ship CPU for iOS when Metal rejects |
| Apple GPU, weight-only int4 (explicit-dequantize) graphs | delegate crash during initialization | SIGSEGV in `EmbeddingLookupOperationParser::Parse` | use the dynamic-quantized (INTEGER compute) int4 path on GPU |
| Android GPU | ML Drift builds a device-layout weight cache ≈ model size next to the model | first-load disk/RAM spike ≈ 2× model | budget for it; CPU needs ~model + ~0.6 GB |

## Quantization-container walls

| Trigger | Signature | Fix |
|---|---|---|
| All-zero weight blocks under blockwise int4 (common in sparse/ternary checkpoints) | `unsupported scale value (0.000000) ... for INT4 tensor` → `Failed to allocate tensors` at load | patch each zero block scale to the tensor's smallest nonzero scale — dequantization unchanged (the blocks are all zero). Note: blockwise scales live in **separate fp16 scale tensors**, not `QuantizationParameters.scale` |
| Channelwise int4 on any decoder | no load error — coherent short outputs, then degeneration; benchmark collapse | never channelwise for decoders; blockwise-32/128 only (`recipe-selector.md` §Quantization) |

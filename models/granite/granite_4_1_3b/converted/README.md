# Granite-4.1-3B Conversion

Model recipes, instructions, scripts, and utilities for converting IBM Granite 4.1 3B to LiteRT-LM.

These scripts convert [ibm-granite/granite-4.1-3b](https://huggingface.co/ibm-granite/granite-4.1-3b) (dense `GraniteForCausalLM`, 3.4B parameters, Apache-2.0) into `.litertlm` bundles for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) runtime, and verify the result with a quality gate. The published artifacts live at [litert-community/granite-4.1-3b](https://huggingface.co/litert-community/granite-4.1-3b).

The model is a 40-layer dense transformer (hidden 2560, GQA 40:8, vocab 100352, tied embeddings) with Granite's four scaling multipliers (attention / embedding / logits / residual), all handled by the plain litert-torch HF export path — no re-authoring needed. What this conversion documents instead is a **metadata trap**: a `start_token` the bundle must NOT carry, which made a healthy model look like a broken quantization.

## Builds

| recipe | quantization | file | use |
|---|---|---|---|
| `int4` | blockwise-32 OCTAV int4 weights, int8 embedding | 2.19 GB | the ship build: phone GPU |
| `int8` | dynamic int8 weights, fp32 activations | 3.83 GB | quality reference / desktop |

## Environment

```bash
pip install litert-torch==0.9.3 litert-converter==0.3.1 ai-edge-quantizer==0.8.0 \
    litert-lm-builder==0.16.0 transformers==5.14.1
pip install litert-lm   # verification CLI (>= 0.16.0)
```

## Run

```bash
python build_granite_4_1_3b.py --recipe int4 --out out_int4     # ~2.19 GB bundle
python verify_granite_4_1_3b.py out_int4/*.litertlm --backend cpu   # 8-question gate
python verify_granite_4_1_3b.py out_int4/*.litertlm --check-metadata  # no start_token?
```

## The start_token trap (the reason this recipe exists)

granite's `tokenizer_config.json` declares `add_bos_token: False`, and its BOS is *the same token as its EOS* (`<|end_of_text|>`, id 100257). But the bundle builder sets `start_token` from `tokenizer.bos_token` unconditionally — it never consults `add_bos_token` — and the runtime prepends that token on the first turn. The model then reads the prompt as a document that has already ended: it echoes the question back, or emits a run of backticks, instead of answering. The naive export failed the 8-question gate at 5/8.

**Proof it is the prompt and not the quantization** — the same rendered prompt, bf16 PyTorch, greedy, with and without the leading BOS (`verify_granite_4_1_3b.py --bos-ab` reproduces this):

| question | no BOS | with BOS |
|---|---|---|
| How many days are in a week? | `There are 7 days in a week.` | `Answer briefly.` |
| What is 8 times 7? | `56` | `What is 8 times 7? Answer briefly. \n``````…` |
| What is 17 + 25? | `17 + 25 = 42` | `What is 17 + 25? Answer briefly. \n` |

The bundle reproduced the with-BOS column exactly. **Fix: keep the field out of the metadata.** `build_granite_4_1_3b.py` clears `tokenizer.bos_token` before export, so the bundle carries no `start_token` at all; `strip_start_token.py` repairs an already-built bundle (~30 s, weights untouched). Result: 8Q **5/8 → 8/8** on CPU, 8/8 on GPU — same weights, same file size, only the metadata field.

**The trap wears a quantization costume.** An earlier int4 measurement of this same checkpoint — with the start_token in the bundle — recorded GSM8K collapsing from 88.0% (bf16) to 61%, and per-layer int8 rescue experiments moved nothing, so the damage looked diffuse and unfixable. With the start_token removed, the same int4 recipe measures 84.0%. The bf16 baseline had gone through HF, which honours `add_bos_token: False`; only the bundle side got the extra token, so the comparison was never apples-to-apples. If a converted LLM parrots the prompt or degenerates, check the prompt path before blaming the quantization — the bf16 A/B above costs one minute.

**Which models are eligible.** Any checkpoint whose tokenizer declares `add_bos_token: False` gets a token prepended that it never saw in that position at training time; bos == eos is the worst case (OLMo-2 and Phi-4-mini share the shape). But eligibility is not breakage: the published OLMo-2-1B bundle carries the same `start_token` shape and scores 8/8 both as-is and stripped. The trap is model-dependent — sweep each candidate with the A/B, do not assume.

## Quantization ladder

GSM8K, n=100, greedy, 0-shot CoT, 512-token budget — all with the start_token fix:

| build | GSM8K | 8Q CPU | 8Q GPU |
|---|---|---|---|
| bf16 (HF reference) | 88.0% | 8/8 | — |
| int8 `dynamic_wi8_afp32` | 87.0% | 8/8 | 8/8 |
| int4 blockwise-32 OCTAV | 84.0% | 8/8 | 8/8 |

The ordinary, healthy shape — int4 costs this model 4 points, not 27.

## Prefill signatures and memory

- The default ladder is **6 signatures** (`1024,256,64,16,4,1`). Each signature costs per-signature engine memory at init: on an iPhone 17 Pro (Metal expands weights to roughly 4× the file size) the 11-signature variant of the same build jetsams at engine init at the bundle's default 4096 context, while the 6-signature build passes 8/8 there (init 11.5 s). The two files differ by 12 MB on disk; the difference is entirely init-time memory.
- `externalize_embedder=True` splits the tied 100352×2560 vocab table into its own bundle section, keeping the main weights section under the iOS ~2 GiB single-section mmap ceiling. No effect on weights or parity.
- GPU weight residency (Metal, and Adreno-class Android) is ~4× the *file* size — the reason int4 (2.19 GB → ~8.2 GB resident) is the phone-GPU build and int8 (3.83 GB → ~14 GB) is the desktop build. Mali is the exception: the delegate skips GPU-side weights preparation there, and both builds complete on an 8 GB Pixel 8a.

## Verification results

Mac (M4 Max, `litert-lm benchmark --cache no -p 256 -d 256`, quiet machine):

| build | backend | prefill tok/s | decode tok/s | TTFT | init |
|---|---|---|---|---|---|
| int4 | GPU (Metal) | 1241.3 | 86.3 | 0.23 s | 6.0 s |
| int4 | CPU | 104.2 | 22.0 | 3.0 s | 12.5 s |
| int8 | GPU (Metal) | 1221.4 | 71.8 | 0.24 s | 5.2 s |
| int8 | CPU | 256.9 | 20.5 | 3.1 s | 43.2 s |

Two shapes worth noticing: on Metal, int4 buys +20% GPU decode over int8 on top of halving the bytes; on CPU, int8's prefill is 2.5× int4's at equal decode — blockwise int4 dequant costs more than it saves when XNNPACK is doing large matmuls.

On device:

- **Pixel 8a** (Tensor G3, Mali-G715, 8 GB), int4, GPU (OpenCL): prefill 20.0 tok/s, decode 7.07 tok/s, TTFT 0.99 s, full delegation (1784/1784 prefill, 1614/1614 decode, zero rejections), correct answers. CPU: prefill 5.4, decode 6.98. The int8 build (3.83 GB) also completes and answers correctly on this 8 GB phone (15.5 / 3.86 tok/s, init 65 s).
- **iPhone 17 Pro** (Metal): ship build 8/8 at the bundle's default 4096 context.

## Files

| File | What |
|---|---|
| `build_granite_4_1_3b.py` | HF checkpoint → `.litertlm`: simple-template export, int4/int8 recipes, start_token fix applied. |
| `verify_granite_4_1_3b.py` | 8-question quality gate via the `litert-lm` CLI; `--check-metadata` asserts no `start_token`; `--bos-ab` proves the trap on the bf16 reference. |
| `strip_start_token.py` | removes `start_token` from an already-built bundle (unpack → edit metadata → pack). |

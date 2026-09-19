<!--
Copyright 2026 The Google AI Edge Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# GLiNER2.5 Small Conversion

Convert the dense prefix of [Fastino GLiNER2.5 Small][source] with Google's
`litert-torch`, then verify the complete entity extraction result through
the LiteRT CompiledModel Python API. The graph ends before sparse candidate
selection. Host code retains the official `gliner2` sparse decoder.

## Build

The build writes six graphs, an embedding table, 16 sparse decoder tensors,
tokenizer/config files, and JSON input/output contracts. One script,
`build_gliner2_5_small.py`, contains the graph rewrite, export, fp16 weight
casting, and host asset extraction. `gliner2_5_host.py` supplies shared host
preprocessing and decoding, and can run a complete example by itself.

| Windows | Storage | Activations | Use |
| --- | --- | --- | --- |
| 128, 256, 512 | fp32 | fp32 | Reference graph |
| 128, 256, 512 | fp16 FC weights | fp32 | Smaller graph |

Default source revision:
`f1e4d8fdd6fe328f45dee6aca3e6a07c9db4296e`.

## Environment

Python 3.12 and macOS arm64 were used. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export HF_HUB_DISABLE_XET=1
export HF_HOME="$PWD/.cache/hf"
export TOKENIZERS_PARALLELISM=false
```

`requirements.txt` records the complete tested environment. The principal
versions are torch 2.12.1, gliner2 2.0.0, transformers 4.57.6,
litert-torch 0.9.3, litert-converter 0.3.1, ai-edge-quantizer 0.8.0, and
ai-edge-litert 2.1.6. The converter's rank-4 lowering uses its private
registration API, so upgrading the converter requires re-verification.
No installed package is patched.

## Run

```bash
python build_gliner2_5_small.py --output-dir out --windows 128 256 512
python verify_gliner2_5_small.py --model-dir out --report-dir verification
HF_HUB_OFFLINE=1 python gliner2_5_host.py --model-dir out
```

The output directory must be empty. Each window first runs a native
PyTorch versus rewritten-graph gate on the ten bundled texts. The build
records finite values, absolute error, numerical scale, and decoded spans
before exporting. `rewrite_s<N>_<case>.npz` retains both packed tensors.

To convert your own compatible fine-tuned checkpoint, use a local model
directory or a Hub ID. The tokenizer and sparse decoder assets are read
from that checkpoint too:

```bash
python build_gliner2_5_small.py --checkpoint ./my-checkpoint \
    --output-dir out_custom --windows 128 256 512
python verify_gliner2_5_small.py --checkpoint ./my-checkpoint \
    --model-dir out_custom --report-dir verification_custom
```

For a Hub checkpoint, supply `--revision COMMIT` to both commands. Only the
default model is automatically pinned. The checkpoint must retain the
Small architecture: hidden size 384, boundary size 128, shared candidate
pool, 64-channel content projection, and 128-word boundary attention
window. This recipe keeps the five labels and threshold 0.5 fixed.

For comparison with the published fp32 graphs, download the pinned
revision and add `--published-dir`:

```bash
hf download litert-community/GLiNER2.5-Small-LiteRT \
    --revision 8f2acf3cb9088bfbc2a6e921ec9a4607433e8f6d \
    --include '*_fp32.tflite' --local-dir published
python verify_gliner2_5_small.py --model-dir out \
    --published-dir published --report-dir verification_published
```

Only compare the published graphs with the default checkpoint. The
verifier reports their packed-output differences without imposing a
new tolerance. It always exits nonzero if the official-model span sets
differ, any output is nonfinite, the confidence error exceeds 1e-4 for
fp32 or 5e-3 for fp16 weights, or the static graph checks fail.

## Files

| File | What |
| --- | --- |
| `build_gliner2_5_small.py` | The recipe: dense-prefix rewrite, parity gate against native PyTorch, litert-torch export for each window, fp16 weight casting, host assets and graph contracts. |
| `verify_gliner2_5_small.py` | Official `gliner2` model versus each `.tflite` through the CompiledModel Python API: span sets, confidence error, finite outputs, operator inventory from the flatbuffer. |
| `gliner2_5_host.py` | Host preprocessing, graph runner and sparse decoding shared by both scripts; runs one complete example by itself. |
| `verification_texts.py` | The ten invented-name English texts used by the build gate and the verifier. |
| `requirements.txt` | The complete tested environment. |

## Why the graph is rewritten

The source combines a dense encoder with data-dependent candidate search.
The graph contains the DeBERTa encoder, word/query routing, boundary
encoder, boundary query head, and dense per-token projections. The first
sparse top-k, unique, and nonzero operations stay on the host, together
with the original pooling, scoring, and output formatting code.

The exact rewrites are:

- Embedding lookup stays on the host. Float masks and one-hot routing
  matrices replace integer graph indexing.
- Attention retains `[batch,heads,rows,channels]`. A PyTorch marker lowers
  to standard StableHLO `dot_general`, then ordinary LiteRT
  `BATCH_MATMUL`. No custom runtime operator is emitted.
- DeBERTa's own logarithmic bucket function constructs constant relative
  position projections for every distance. Reshape/slice relative shifts
  replace runtime gathers. Linear buckets would be wrong beyond 128.
- Boundary attention's 128-word local mask is baked as a constant.
  Float arithmetic keeps the same mask, including the padded diagonal.
- Upper-triangular matrix multiplication implements prefix sums with the
  constant on the right. Native GELU and layer normalization are retained.
- Seventeen dense results are concatenated into one rank-4 output.
  `graph_contract_s<N>.json` gives every slice's shape and offset.

Floating-point operation ordering can differ from the source. The rewrite
changes neither attention neighborhoods nor activation functions, sparse
selection rules, or extraction thresholds.

The host assets contain the fp32 embedding table and only the 16 tensors
read by sparse decoding. Dense and unused parameters in the host extractor
remain on the PyTorch meta device. An accidental dense call therefore
fails instead of using random weights. Host preprocessing is compared
byte-for-byte against the official model in the verifier.

## fp16 weight storage

`ai-edge-quantizer` applies `FLOAT_CASTING`, 16 bits, tensorwise,
`ALL_SUPPORTED`, through `add_weight_only_config`. In these graphs it
casts 96 fully connected weight tensors to fp16 and inserts `DEQUANTIZE`.
Other constants and all activations stay float32. This is a separately
gated storage reduction; the fp32 graph rewrite uses no approximation.

Both variants need **explicit GPU FP32 precision** on the published
Android path. Default GPU precision produced nonfinite values. The
[model page](../README.md#android-and-ios) shows the Kotlin setting.

## Verification results

Rebuilt on 2026-09-20, macOS 27.0 arm64 CPU, four threads, using the pinned
versions above. Each row uses the same ten invented-name English texts
and a fresh official fp32 `gliner2` oracle. These are conversion-parity
checks, not a labeled accuracy benchmark.

| Window | Storage | Exact span sets | Max confidence error |
| --- | --- | ---: | ---: |
| 128 | fp32 | 10/10 | 1.13249e-6 |
| 128 | fp16 weights | 10/10 | 2.39373e-4 |
| 256 | fp32 | 10/10 | 1.13249e-6 |
| 256 | fp16 weights | 10/10 | 2.39373e-4 |
| 512 | fp32 | 10/10 | 1.13249e-6 |
| 512 | fp16 weights | 10/10 | 2.39373e-4 |

All outputs were finite. The largest packed absolute error was 1.02520e-5
for rewritten PyTorch versus native PyTorch, 3.33786e-5 for LiteRT fp32
versus native, and 0.00840282 for fp16 weights versus native. Packed
logits include the -10000 mask sentinel; per-field L2 norms and absolute
scales are recorded alongside the raw output dumps.

For all three windows, rebuilt fp32 outputs versus the published fp32
graphs had maximum absolute error **0.0** over all ten texts. Every
constant tensor also matched. File hashes differ because the standalone
script changes embedded tensor debug names.

The flatbuffer check uses `ai_edge_litert.schema_py_generated`. It found
1024/1025/1025 operators in fp32 and 1120/1121/1121 with fp16 weights, no
banned operations, no tensor above rank four, and only rank-four
`BATCH_MATMUL` operands with constants on the right. The banned set is
`BROADCAST_TO`, `CAST`, `CUMSUM`, `CUSTOM`, `FILL`, `GATHER`, `GATHER_ND`,
`IF`, `LESS`, `LESS_EQUAL`, `LOGICAL_AND`, `LOGICAL_OR`, `MAXIMUM`,
`NON_ZERO`, `ONE_HOT`, `RANGE`, `SCATTER_ND`, `SELECT`, `SELECT_V2`, `TILE`,
`TOPK_V2`, `UNIQUE`, `WHERE`, and `WHILE`. This static check does not replace
execution on the target GPU.

### Published device results

The [pinned Android evidence][device] reports Galaxy S26 SM-S942Q,
Android 16, LiteRT 2.2.0, wfp16 graphs, explicit GPU FP32, debug build,
33.5 °C starting temperature. One warm-up and five timed repetitions per
text gave these medians:

| Window | Graph write through readback | Kotlin host decode |
| --- | ---: | ---: |
| 128 | 11.895 ms | 9.518 ms |
| 256 | 26.837 ms | 13.497 ms |
| 512 | 92.171 ms | 12.778 ms |

The published graph test matched spans on 70/70 texts, with maximum
confidence error 0.0026924014. These measurements are quoted, not new
device tests of the rebuilt graphs. The [model page](../README.md#tested-on)
records the input counts and conditions.

## Limits and license

One English text, five fixed labels, and at most 512 encoded tokens /
384 text words. The graph does not contain embedding lookup or sparse
selection. The Python host requires gliner2 and PyTorch. Its substitutions
are per-extractor and synchronous. Use the published [Kotlin host][android]
for the Android integration.

The recipe and GLiNER2.5 checkpoint are Apache-2.0. Graph code follows
Fastino's Apache-2.0 [GLiNER2][code] and Hugging Face's Apache-2.0
[Transformers DeBERTa implementation][deberta-code]. The underlying
Microsoft DeBERTa-v3 encoder carries MIT terms. Retain the upstream
[component notices][licenses] when redistributing generated files.

[source]: https://huggingface.co/fastino/gliner2.5-small-v1
[code]: https://github.com/fastino-ai/GLiNER2
[deberta-code]: https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/deberta_v2/modeling_deberta_v2.py
[device]: https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT/blob/8f2acf3c/android/README.md#verified-on
[android]: https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT/tree/8f2acf3c/android
[licenses]: https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT/tree/8f2acf3c/licenses

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

# GLiNER2.5 Small

Extract English entity spans with the fixed labels `person`, `organization`,
`location`, `product`, and `date`. For example, the bundled sentence returns
`Maya Chen`, `Orvane Robotics`, `Veltrix 9`, `Lisbon`, and `March 12, 2025`,
with character offsets and confidence values.

This recipe converts [GLiNER2.5 Small][source] into a dense LiteRT graph plus
host preprocessing and sparse decoding. The [published files][published]
include three window sizes and two weight-storage formats.

## Run

Run from this directory with Python 3.12. The requirements include the
conversion tools as well as the CPU runtime.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r converted/requirements.txt
export HF_HUB_DISABLE_XET=1
export HF_HOME="$PWD/.cache/hf"
hf download litert-community/GLiNER2.5-Small-LiteRT \
    --revision 8f2acf3cb9088bfbc2a6e921ec9a4607433e8f6d \
    --include 'gliner25_small_s128_wfp16.tflite' 'host_assets/*' \
    --local-dir gliner
HF_HUB_OFFLINE=1 python converted/gliner2_5_host.py --model-dir gliner
```

The graph and host assets are both required. The Python host imports the
pinned official `gliner2` package for sparse decoding and needs no source
checkpoint at inference time.

## Which file

Pick the smallest encoded window that fits both the schema and the text.
The word limit counts the official splitter's text tokens, including
punctuation. Sizes below describe the published files in decimal MB.

| Encoded tokens / words | fp32 graph | fp16-weight graph |
| --- | ---: | ---: |
| 128 / 48 | 98.02 MB | 54.05 MB |
| 256 / 192 | 107.87 MB | 63.91 MB |
| 512 / 384 | 128.08 MB | 84.11 MB |

Names are `gliner25_small_s<N>_fp32.tflite` and
`gliner25_small_s<N>_wfp16.tflite`. The latter stores 96 fully connected
weight tensors in fp16 and dequantizes them to float32. Activations and the
remaining constants stay float32. All windows share a 196.62 MB fp32
embedding table and 0.47 MB of sparse decoder weights, plus the tokenizer.

## Labels and window

```bash
python converted/gliner2_5_host.py --model-dir gliner \
    --window 256 --variant wfp16 --text 'Maya Chen visited Lisbon.'
```

Download the selected graph before changing `--window`. The schema has five
labels and threshold 0.5. Inputs exceeding the chosen capacity raise an
error. The largest graph accepts at most 512 encoded tokens, including the
schema, and 384 text words. This recipe does not implement document
chunking, additional schemas, relations, or classification. Validation
covers English only.

The graph accepts five float32 inputs: embeddings `[1,N,384]`, attention
mask `[1,N]`, text routing `[1,T,N]`, query routing `[1,5,N]`, and text mask
`[1,T]`. Routing rows are one-hot selectors. One rank-4 output packs 17
logical tensors in `[1,1,1,P]`, where P is 57,758, 217,310, or 430,046.
The host performs embedding lookup, candidate selection, scoring, overlap
resolution, and character-offset mapping. See the [host contract][contract].

## Python

The helper uses the LiteRT CompiledModel buffer API on CPU. From this
directory, the complete extraction call is:

```python
from pathlib import Path
from converted import gliner2_5_host as gliner

assets = Path('gliner')
host = gliner.HostRuntime(assets / 'host_assets')
inputs, metadata = host.prepare(gliner.EXAMPLE, seq=128)
runner = gliner.CpuRunner(
    assets / 'gliner25_small_s128_wfp16.tflite', gliner.Shape(128))
try:
    packed = runner.run(inputs)
    print(host.decode(metadata, packed, inputs))
finally:
    runner.close()
```

`CpuRunner` creates `CompiledModel`, allocates buffers once, writes the
inputs in signature order, calls `run_by_index`, and reads the packed
float32 output. The host temporarily substitutes dense results in its own
extractor object, so calls using one `HostRuntime` must be serialized.

## Android and iOS

The [Android directory][android] contains a Kotlin host implementation and
a Compose sample using LiteRT 2.2.0. Both storage variants require explicit
GPU FP32 computation:

```kotlin
val options = CompiledModel.Options(setOf(Accelerator.GPU)).apply {
  gpuOptions = CompiledModel.GpuOptions(
      precision = CompiledModel.GpuOptions.Precision.FP32)
}
```

Follow the sample's environment and worker-thread setup. Default GPU
precision produced nonfinite encoder outputs in the published tests.
Selecting an fp32 file alone does not request FP32 computation. iOS and NPU
execution have not been validated for this recipe.

## Tested on

The [published Android results][device] use Galaxy S26 SM-S942Q, Android 16,
LiteRT 2.2.0, wfp16 graphs, and explicit GPU FP32. The debug build started
at 33.5 °C and 99% battery over USB. Each text had one warm-up and five
timed runs. These are medians per window, excluding compilation.

| Tokens / words | Texts | Input write through readback | Kotlin decode |
| --- | ---: | ---: | ---: |
| 128 / 48 | 60 | 11.895 ms | 9.518 ms |
| 256 / 192 | 5 | 26.837 ms | 13.497 ms |
| 512 / 384 | 5 | 92.171 ms | 12.778 ms |

The published test matched official fp32 spans on 70/70 texts (400 spans),
with all 420 warm-up and timed outputs finite. Maximum confidence error
was 0.0026924014. These device measurements are quoted from the published
record and apply to those graphs.

The recipe rebuild was checked on macOS 27.0 arm64 CPU, ai-edge-litert
2.1.6, four threads: 10/10 texts at each window and storage type, or 60/60
span-set comparisons. Maximum confidence error was 1.13249e-6 for fp32 and
2.39373e-4 for fp16 weights. Rebuilt and published fp32 packed outputs
matched exactly on all 30 comparisons. The rebuilt graphs were not
measured on a device. See the [conversion record](converted/README.md).

## Conversion

[Build and verify the graphs](converted/README.md), including a compatible
fine-tuned checkpoint supplied with `--checkpoint`. Only the dense graph
is rewritten. The host keeps the official sparse candidate and decoding
implementations.

## References

- [GLiNER2.5 Small][source] and the [GLiNER2 implementation][code], Fastino.
- [Published LiteRT graphs][published], [host contract][contract], and
  [Android sample][android].
- [CompiledModel Python API][python-api].

The recipe and GLiNER2.5 checkpoint use Apache-2.0. The host calls
Apache-2.0 GLiNER2 and Transformers code. The underlying
[Microsoft DeBERTa encoder][encoder] uses MIT terms. Preserve those upstream
notices when redistributing weights or derived code. The published
[license directory][licenses] contains the component notices.

[source]: https://huggingface.co/fastino/gliner2.5-small-v1
[published]: https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT
[code]: https://github.com/fastino-ai/GLiNER2
[encoder]: https://huggingface.co/microsoft/deberta-v3-xsmall
[python-api]: https://ai.google.dev/edge/litert/next/python
[android]: https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT/tree/8f2acf3c/android
[contract]: https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT/blob/8f2acf3c/HOST_CONTRACT.md
[device]: https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT/blob/8f2acf3c/android/README.md#verified-on
[licenses]: https://huggingface.co/litert-community/GLiNER2.5-Small-LiteRT/tree/8f2acf3c/licenses

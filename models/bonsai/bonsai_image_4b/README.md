# Bonsai Image 4B

[prism-ml/bonsai-image-ternary-4B](https://huggingface.co/prism-ml/bonsai-image-ternary-4B), PrismML's ternary-weight diffusion transformer on the FLUX.2-klein-4B architecture, as three fixed-shape `.tflite` graphs (text encoder, diffusion transformer, VAE decoder) for the [LiteRT](https://github.com/google-ai-edge/litert) runtime, with a Python host loop of 143 lines that tokenizes the prompt, runs the sampling steps and writes the PNG. The graphs are published at [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B). Every command in the code blocks on this page was run on ai-edge-litert 2.2.0.

## Run

```bash
pip install ai-edge-litert numpy pillow transformers jinja2 huggingface_hub
hf download litert-community/Bonsai-Image-ternary-4B dit_int4b32.tflite textenc_int4.tflite vae_dec_fp32.tflite pipeline_meta.json tokenizer/ --local-dir bonsai
python python/generate.py --model-dir bonsai --prompt "a red fox sitting in fresh snow at sunrise" --out fox.png
```

Run from this directory. The download is the smallest working set, 4.28 GB. The image is 512×512 and takes four sampling steps; `generate.py` prints the time of each stage as it runs, and transformers' notice that PyTorch is missing is expected, since only its tokenizer is used.

## Which file

| File | Size | Use it for |
|---|---|---|
| `dit_int4b32.tflite` | 2.27 GB | The diffusion transformer on the CPU: Python, Android and the iOS app |
| `dit_gpu_int4b32.tflite` | 2.27 GB | The same weights exported for the Apple GPU: the macOS app runs it on Metal |
| `textenc_int4.tflite` | 1.80 GB | The prompt encoder of the smallest set |
| `textenc_int8_weightonly.tflite` | 3.13 GB | The prompt encoder that follows the PyTorch pipeline more closely; sharpness stays flat on the six-prompt grid on the model card |
| `vae_dec_fp32.tflite` | 0.20 GB | The decoder, in every set |

`generate.py` reads the file names from the `files` entry of `pipeline_meta.json`; to run the int8 encoder, point its `textenc` entry at that file. `tokenizer/` is the Qwen3 tokenizer, and the `generate.py` in the same repository is a copy of the one here. [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and the Gallery app load `.litertlm` bundles; these graphs load through the LiteRT runtime APIs below, on the CPU, with the DiT on the Apple GPU as well.

## Steps and seed

```bash
python python/generate.py --model-dir bonsai --prompt "a red fox sitting in fresh snow at sunrise" --steps 8 --seed 42 --out fox_8.png
python python/generate.py --model-dir bonsai --prompt "a red fox sitting in fresh snow at sunrise" --threads 8 --out fox_t8.png
```

The model is distilled for four steps; `--steps` takes more, at one DiT pass per step. The same prompt, seed and step count give the same image on the same machine, whatever the thread count. `--threads` defaults to the core count; on the Mac below, 8 threads took 5.1 s per DiT step against 3.8 s with all 16.

## Python

```python
from ai_edge_litert.interpreter import Interpreter
from transformers import AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("bonsai/tokenizer")
text = tokenizer.apply_chat_template([{"role": "user", "content": "a red fox sitting in fresh snow at sunrise"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
tokens = tokenizer(text, return_tensors="np", padding="max_length", max_length=256)

textenc = Interpreter(model_path="bonsai/textenc_int4.tflite", num_threads=8)
textenc.allocate_tensors()
ids, mask = sorted(textenc.get_input_details(), key=lambda d: d["name"])
textenc.set_tensor(ids["index"], tokens["input_ids"].astype(np.int32))
textenc.set_tensor(mask["index"], tokens["attention_mask"].astype(np.int32))
textenc.invoke()
embeds = textenc.get_tensor(textenc.get_output_details()[0]["index"])
print(embeds.shape)
```

This is the first stage of `generate.py`, run from this directory after the download above; `generate.py` makes the same calls for the DiT and the VAE decoder. Three rules carry into any port: map inputs by argument position, `serving_default_args_<n>`, never by shape (the text encoder's two inputs are both 1×256); load one graph at a time, freeing it before the next, so the three graphs are never resident together; and keep XNNPACK on, which the Interpreter and the CompiledModel CPU path do themselves, while the classic C API needs the delegate attached or the int4 weights run on reference kernels, many times slower. The shapes of all three graphs are in the `io` entry of `pipeline_meta.json`.

## Android and iOS

- iOS: the [sample app](../../../samples/litert/image_generation/ios/) in this repository drives the three graphs through the LiteRT CompiledModel C API on the CPU (XNNPACK); copy the three `.tflite` files into its Documents folder.
- macOS: the [macOS app](../../../samples/litert/image_generation/macos/) runs the DiT on the Apple GPU through the LiteRT Metal accelerator; it takes `dit_gpu_int4b32.tflite` and holds the weights at fp32 on the GPU, 37 GB peak memory on the Mac below including the one-time Metal compile.
- Android: the graphs run on the CPU through the [LiteRT Kotlin API](https://ai.google.dev/edge/litert/android) with XNNPACK, and the host loop is a port of `generate.py`. Sample apps for the runtime are listed in [`models/README.md`](../../README.md#where-to-find-examples).

## Tested on

Times from the stage prints of `generate.py` and of the macOS app, and the whole-image time and peak memory from `/usr/bin/time -l`: one 512×512 image, four steps, seed 0; the prompt window is fixed at 256 tokens.

| Device | Runtime | Text encoder / DiT step / VAE decoder | Whole image | Peak memory |
|---|---|---|---|---|
| Mac M4 Max, CPU, 16 threads | ai-edge-litert 2.2.0, `generate.py` | 1.1 s / 3.8 s / 1.1 s | 19 s | 6.1 GB |
| Mac M4 Max, GPU (Metal) for the DiT, CPU for the rest | macOS sample app, ai-edge-litert 2.1.6 runtime | 1.3 s / 0.74 s / 1.2 s | 5.5 s, after a 44 s Metal compile at launch | 37 GB, compile included |

Every row produced the fox image before its times were recorded. The iOS app's README carries its own measurement on an iPhone 17 Pro; Android was not measured for this page.

## Conversion

How the three graphs were built and verified (the DiT export, the int4 block-32 quantization and its zero-scale patch, the pruned text encoder, the VAE decoder, and the GPU-shaped DiT): [`converted/`](converted/); the cookbook's [recipe list](../../conversion.md#13-recipes-in-this-directory) names this recipe.

## References

- [prism-ml/bonsai-image-ternary-4B](https://huggingface.co/prism-ml/bonsai-image-ternary-4B), the source checkpoint; [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B), the graphs, with the quality grid and a copy of the host loop.
- [`samples/litert/image_generation/`](../../../samples/litert/image_generation/): the iOS and macOS apps, with screenshots.
- LiteRT guides: [inference](https://ai.google.dev/edge/litert/inference), the [CompiledModel Python API](https://ai.google.dev/edge/litert/next/python), the [CompiledModel C++ API](https://ai.google.dev/edge/litert/next/cpp), [Android](https://ai.google.dev/edge/litert/android).

# Qwen3-TTS

[Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base), the Qwen team's 0.6B speech language model with voice cloning in ten languages, as three fixed-shape `.tflite` graphs (talker, code predictor, codec decoder) for the [LiteRT](https://github.com/google-ai-edge/litert) runtime, with host-side embedding tables and a Python host loop of 491 lines that tokenizes the text, runs the frame loop and returns 24 kHz audio. The graphs and tables are published at [litert-community/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base). Every command in the code blocks on this page was run on ai-edge-litert 2.2.0.

## Run

The host loop lives with the sample: run the commands and the Python block on this page from [`samples/litert/text_to_speech_lm/python/`](../../../samples/litert/text_to_speech_lm/python/) in this repository.

```bash
pip install ai-edge-litert numpy tokenizers soundfile huggingface_hub
hf download litert-community/Qwen3-TTS-12Hz-0.6B-Base talker_int4.tflite mtp_fp32.tflite codec_decoder_fp32.tflite tokenizer.json tables/ voices/ --local-dir qwen3tts
python synthesize.py --model_dir qwen3tts --text "Hello from LiteRT running fully on device." --output hello.wav
```

The download is the default set, 1.89 GB; without `--model_dir`, `synthesize.py` fetches the same files into the Hugging Face cache. The output is a 24 kHz mono wav in the bundled demo voice, and the script prints the audio length, the real-time factor and the time of each stage; the runtime's `INFO` lines and its warning that no NPU accelerator loaded are expected on a Mac.

## Which file

| File | Size | Use it for |
|---|---|---|
| `talker_int4.tflite` | 0.26 GB | The talker on every host, Python and the Android app; the default |
| `talker_fp32.tflite` | 1.78 GB | The full-precision talker, seven times the size; with `--greedy` its codes matched the PyTorch model token for token in the conversion's verification |
| `mtp_fp32.tflite` | 0.44 GB | The code predictor, in every set |
| `codec_decoder_fp32.tflite` | 0.46 GB | The codec decoder, in every set |
| `tables/` | 0.72 GB | The codec, code-predictor and text embedding tables and the text projection, read by the host loop |

`synthesize.py` picks the talker with `--talker int4` or `--talker fp32`; the other two graphs and the tables are always loaded. `tokenizer.json` is the Qwen2 tokenizer and `voices/demo_speaker.npy` the demo voice. [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and the Gallery app load `.litertlm` bundles; these graphs load through the LiteRT runtime APIs below, on the CPU.

## Language and voice

```bash
python synthesize.py --model_dir qwen3tts --language japanese --text "こんにちは。この音声は端末の上で作られています。" --output hello_ja.wav
python synthesize.py --model_dir qwen3tts --seed 0 --text "Hello from LiteRT running fully on device." --output hello_seed0.wav
hf download litert-community/Qwen3-TTS-12Hz-0.6B-Base talker_fp32.tflite --local-dir qwen3tts
python synthesize.py --model_dir qwen3tts --talker fp32 --seed 0 --text "Hello from LiteRT running fully on device." --output hello_fp32.wav
```

`--language` takes chinese, english, french, german, italian, japanese, korean, portuguese, russian, spanish or `auto`; english is the default. Sampling is the model's default, so two runs of the same text differ; `--seed` makes them the same. `--greedy` takes the most likely token instead, the setting under which the fp32 talker reproduced the PyTorch model token for token in the conversion's verification; on one test sentence it ran to the 512-frame cap with either talker without speaking the sentence. `--speaker` takes a 1024-d x-vector `.npy`; [`converted/extract_speaker_embedding.py`](converted/extract_speaker_embedding.py) enrolls one from about three seconds of audio in the reference environment listed in [`converted/`](converted/) (PyTorch, qwen-tts, librosa). `--threads` defaults to 8; on the Mac below, all 16 threads took 1.24 s in the talker for the same 40 frames against 0.94 s with 8, and the wav was the same. The cap of 512 frames is 41 s of audio.

## Python

```python
import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.options import CpuOptions, Options

talker = CompiledModel.from_file("qwen3tts/talker_int4.tflite", options=Options(cpu_options=CpuOptions(num_threads=8)))
inputs = talker.get_input_tensor_details("decode")
outputs = talker.get_output_tensor_details("decode")
in_buffers = {name: talker.create_input_buffer_by_name("decode", name) for name in inputs}
out_buffers = {name: talker.create_output_buffer_by_name("decode", name) for name in outputs}
for name, detail in inputs.items():
    in_buffers[name].write(np.zeros(detail["shape"], np.dtype(detail["dtype"])))
talker.run_by_name("decode", in_buffers, out_buffers)
logits = out_buffers["logits"].read(int(np.prod(outputs["logits"]["shape"])), np.float32)
print(list(talker.get_signature_list()), len(inputs), logits.shape)
```

This is one talker step, run from the same directory after the download above; `qwen3_tts_pipeline.py` makes the same calls for the prefill, the code predictor and the codec decoder, with the buffers created once and rewritten every step. Three rules carry into any port: the KV cache is explicit, so `decode` takes 59 inputs, the embeddings, the position, the mask and 56 `kv_cache` tensors, and returns the cache with the logits, and a port keeps two buffer sets and swaps them between steps; the code predictor runs 16 times per frame on a 17-slot KV cache that starts empty every frame; and the codec decoder takes 64-frame windows, and every window after the first starts with 25 frames of left context.

## Android and iOS

- Android: the [sample app](../../../samples/litert/text_to_speech_lm/kotlin_cpu/android/) in this repository is a Kotlin port of the loop on the LiteRT CompiledModel API, on the CPU (XNNPACK). In its directory, `./gradlew :app:installDebug` builds and installs it and `./install_to_device.sh` downloads the graphs, the tables, the demo voice and the app's tokenizer files and pushes them into the app; then type a sentence, pick a language and tap Speak.
- iOS: the graphs load through the LiteRT [CompiledModel C++ API](https://ai.google.dev/edge/litert/next/cpp) on the CPU (XNNPACK), and the host loop is a port of `qwen3_tts_pipeline.py`. Sample apps for the runtime are listed in [`models/README.md`](../../README.md#where-to-find-examples).

## Tested on

Times from the stage prints of `synthesize.py` and from the Kotlin app's log line: the sentence above ("Hello from LiteRT running fully on device."), the int4 talker, seed 0 on the Mac. The whole-sentence time is the four stages summed, without the model load; the real-time factor is that time per second of audio, so 1.0 is real time; peak memory is the maximum resident set size from `/usr/bin/time -l`.

| Device | Runtime | Audio | Prefill / talker / code predictor / codec | Whole sentence | Peak memory |
|---|---|---|---|---|---|
| Mac M4 Max, CPU, 8 threads (code predictor 1) | ai-edge-litert 2.2.0, `synthesize.py` | 3.20 s, 40 frames | 0.17 s / 0.94 s / 6.37 s / 0.45 s | 7.9 s, real-time factor 2.5 | 5.0 GB |
| Galaxy S26 (SM-S942Q), CPU, 4 threads (code predictor 2) | Kotlin sample app, LiteRT 2.1.5 | 2.40 s, 30 frames | 0.26 s / 0.82 s / 10.15 s / 1.23 s | 12.5 s, real-time factor 5.2 | |

The Mac row's wav transcribes back to its sentence apart from the name LiteRT. The phone row is the run after a cool-down to 36 °C; the three measured runs before it, on the warm phone, took 15 to 29 s. The phone's audio was not captured, and iOS was not measured for this page.

## Conversion

How the three graphs and the tables were built and verified step by step against the PyTorch model (the talker checkpoint and its export, the code predictor graph, the codec decoder, the host tables, voice enrollment): [`converted/`](converted/); the same model authored with the [LiteRT Tensor API](https://github.com/google-ai-edge/LiteRT/tree/main/tensor): [`tensor_api/`](tensor_api/); the cookbook's [recipe list](../../conversion.md#13-recipes-in-this-directory) names this recipe.

## References

- [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base), the source checkpoint; [litert-community/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base), the graphs and tables, with the per-graph accuracy checks and benchmark tables for more devices.
- [`samples/litert/text_to_speech_lm/`](../../../samples/litert/text_to_speech_lm/): the Python host loop and the Android app.
- LiteRT guides: [inference](https://ai.google.dev/edge/litert/inference), the [CompiledModel Python API](https://ai.google.dev/edge/litert/next/python), the [CompiledModel C++ API](https://ai.google.dev/edge/litert/next/cpp), [Android](https://ai.google.dev/edge/litert/android).

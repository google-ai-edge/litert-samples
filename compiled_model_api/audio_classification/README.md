# Audio Classification with LiteRT — wav2vec2 Keyword Spotting (on-device, all-GPU)

An Android sample that runs [wav2vec2](https://huggingface.co/facebook/wav2vec2-base) keyword spotting ([`superb/wav2vec2-base-superb-ks`](https://huggingface.co/superb/wav2vec2-base-superb-ks), Apache-2.0) end-to-end on device with the LiteRT `CompiledModel` API. It classifies 1 s of 16 kHz audio into 12 Speech-Commands labels (yes / no / up / down / left / right / on / off / stop / go / _unknown_ / _silence_): the app classifies a bundled clip on launch and records keywords from the mic.

**No FFT anywhere** — the raw 16 kHz waveform goes straight into a 1D-conv feature extractor (no mel step, in or out of the graph), so the whole model rides the GPU delegate. The transformer residual is small (`|x|≈3.2`), so it is fp16-exact on GPU with no CPU fallback.

## Models

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| frontend (feature extractor + projection) | audio[1,16000] → feat[1,49,768] | **GPU** |
| head (encoder + weighted-layer-sum + classifier) | feat[1,49,768] → logits[1,12] | **GPU** |

Both graphs load with `CompiledModel.create(...)` on `Accelerator.GPU`. fp16, converted with [litert-torch](https://github.com/google-ai-edge/litert) — per-graph tflite-vs-torch corr **1.000000**. On a Pixel 8a (Tensor G3): frontend `134/134` + head `893/893` on `LITERT_CL`, end-to-end ~19 ms for a 1 s clip (**RTF ≈ 0.02**); real-speech validation 10/10 keywords, device-vs-CPU logits corr 0.9995.

## Why two graphs

The model is op-clean for the GPU, **but the full 1008-node graph exceeds the Mali shader-compile limit** (it fails to compile fused, even though every op is supported). Splitting at the conv-frontend / transformer-encoder boundary makes each half compile (134/134 + 893/893). Both halves run on the GPU; the frontend's `feat[1,49,768]` feeds the head.

## Re-authoring (litert-torch, parity corr 1.0)

`nn.GELU`→tanh-GELU · feature-extractor `GroupNorm`→GN4D · pos-conv `weight_norm` fold · `create_bidirectional_mask`→None (fixed length, no padding) · the `use_weighted_layer_sum` head accumulated incrementally (`acc += w[i]·hᵢ`) with **baked** `softmax(layer_weights)` constants — the runtime softmax + per-layer scalar gathers otherwise split the Mali partition. See [`conversion/`](conversion/).

## Run

1. Build the two tflites with `conversion/build_w2v2_split.py` (or get them from Hugging Face — [litert-community/wav2vec2-keyword-spotting](https://huggingface.co/litert-community/wav2vec2-keyword-spotting)).
2. Build/install the app, then push the models into its private storage:
   ```bash
   ./kotlin_cpu_gpu/android/install_to_device.sh <dir-with-the-tflites>
   ```
3. Launch **Wav2Vec2 KWS** — it compiles the GPU shaders (~10 s first launch), classifies the bundled clip, then the Record button captures 1 s from the mic and classifies it.

## App architecture

Jetpack Compose + MVVM. `MainActivity` is a thin `ComponentActivity` host; `MainViewModel` owns the `Wav2Vec2Kws` helper, self-tests the bundled clip on `init`, and captures the mic clip — every model call and the capture that feeds it run on one confined `Dispatchers.Default.limitedParallelism(1)` worker (the helper reuses native buffers). `KeywordSpottingScreen` requests `RECORD_AUDIO` on demand and renders the recognized keyword; all user-visible strings live in `res/values/strings.xml`.

## Files

- `kotlin_cpu_gpu/android/` — the Android app: `Wav2Vec2Kws.kt` = the 2 GPU graphs chained (inference helper, unchanged); `MainViewModel.kt` = model + mic capture on the confined worker; `UiState.kt` = the immutable UI snapshot; `view/KeywordSpottingScreen.kt` = the Compose UI + mic-permission flow; `MainActivity.kt` = the thin Compose host.
- `conversion/` — the litert-torch conversion scripts (`build_w2v2.py` = op-check + parity, `build_w2v2_split.py` = the 2-graph deployment split).

Upstream: [superb/wav2vec2-base-superb-ks](https://huggingface.co/superb/wav2vec2-base-superb-ks) (Apache-2.0).

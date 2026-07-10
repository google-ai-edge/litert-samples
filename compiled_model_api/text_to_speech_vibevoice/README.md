# Text-to-Speech with LiteRT — VibeVoice-Realtime-0.5B (on-device, FFT-free)

An Android sample that runs [VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B)
end-to-end on device with the LiteRT `CompiledModel` API: type text, synthesize it, and play it
back. No network.

VibeVoice is a **streaming, autoregressive next-token-diffusion** TTS: two Qwen2 transformer LMs
(with a KV cache) drive a small DDPM diffusion head that denoises one acoustic latent per step, and
a convolutional σ-VAE decoder turns the accumulated latents into a 24 kHz waveform. There is **no
FFT/iSTFT anywhere** — the σ-VAE decoder is a time-domain convolutional model — which is what lets
the heavy compute run on the `CompiledModel` delegates.

| | |
|---|---|
| Model | [VibeVoice-Realtime-0.5B](https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B) (Microsoft, MIT) |
| Backbone | Qwen2.5-0.5B, split 4-layer text LM + 20-layer TTS LM (host-side KV cache) |
| Acoustic | σ-VAE tokenizer (conv, 3200× / 7.5 Hz) + 4-layer DDPM diffusion head |
| Sample rate | 24 kHz mono, single speaker, English |

## Models

Four LiteRT graphs are loaded with `CompiledModel.create(...)`; the accelerator (`GPU`/`CPU`) and
precision are chosen **per graph, by which one produces correct output on the device** — verified
on-device, not just by desktop parity.

| Graph | In → Out | Delegate (Pixel 8a) | Why |
| :-- | :-- | :--: | :-- |
| `vv_base_lm_kv_fp32` | x[1,1,896], cos/sin[1,1,1,64], mask[1,1,1,129], pk/pv[1,8,128,64] → hidden[1,1,896], k/v[1,8,1,64] | **CPU (fp32)** | This sample pins LiteRT 2.1.3, whose Mali delegate rejects the KV-step `FULLY_CONNECTED` weights shape — **fixed in 2.1.5**, see below. fp16 separately collapses the stack on ARM XNNPACK. |
| `vv_tts_lm_kv_fp32` | same I/O, `Pmax=384`, `L*nkv=40` | **CPU (fp32)** | Same as above (20 layers). |
| `vv_diffhead_fp16` | noisy[1,64], t_freq[1,256], cond[1,896] → v[1,64] | **GPU (fp32 precision)** | Small; compiles and computes correctly on ML Drift. |
| `vv_decoder_fp32` | latent[1,64,128] → wav[1,1,409600] | **CPU (fp32)** | Compiles on GPU but ML Drift **miscomputes** it — see below. |

This is a **hybrid** placement, not an all-GPU one: only the tiny diffusion head runs on the GPU.
Graphs are converted with [litert-torch](https://github.com/google-ai-edge/litert) (per-graph
tflite-vs-torch corr **1.000000**); the Kotlin host loop and DPM-Solver++ port match the reference
end-to-end. See [`conversion/`](conversion/).

### Correction (2026-07-10): the LMs are not blocked by the GPU

An earlier version of this sample said ML Drift rejects the KV-step `FULLY_CONNECTED` weights shape,
so autoregressive attention decoders cannot run on it. **That is withdrawn.** The rejection is real
on LiteRT **2.1.3** (`INVALID_ARGUMENT: Unsupported weights shape`, `fully_connected.cc:1070`) and
**fixed in 2.1.5**, where the same graphs delegate fully and compute correctly: `vv_base_lm_kv_fp32`
**313/313 nodes at 10 ms/step** (hidden corr 0.999964 vs CPU), the 20-layer `vv_tts_lm_kv_fp32`
**1559/1559 nodes at 48 ms/step** (corr 0.999852). Moving the LMs onto the GPU is a follow-up; this
sample still pins 2.1.3, and fp16 independently collapses the 20-layer stack on ARM XNNPACK, which is
a fact about the *CPU* path and remains true.

### The σ-VAE decoder is a confirmed ML Drift correctness bug

On-device probing with **single-output** sub-graphs (immune to the delegate's known output-buffer
aliasing, so any divergence is a genuine miscompute) shows every individual op of a ConvNeXt block —
conv, RMSNorm, depthwise conv, FFN `linear → tanh-GELU → linear` — is *bit-exact* on the GPU, yet
the *assembled* block miscomputes. The divergence first appears at the LayerScale broadcast-multiply
that closes the block, though the same multiply is correct earlier in the block, so the trigger is
graph assembly/depth, not any one op. It reproduces identically on OpenCL and OpenGL and under every
buffer-storage / precision / constant-sharing option, and it persists at fp32, so it is a
graph-assembly buffer/scheduling bug in ML Drift's shared compilation layer — not precision, not a
backend, not a kernel. Splitting the decoder does not help (a single block already trips it), so the
decoder runs on CPU, where it is bit-exact with the reference.

Re-verified on 2026-07-10 against the full decoder graph on **both** runtimes: all 1578 nodes
delegate on 2.1.3 and on 2.1.5, and the GPU output is **bit-identically wrong on both** (std
0.030584, absmax 0.134888, corr 0.0254 against a desktop CPU fp32 reference whose std is ~0).
Deterministic and version-independent — unlike the `FULLY_CONNECTED` rejection above, this is open.

## Pipeline

```
text --BPE(host)--> token ids
     --host: per text token-->
         embed_tokens(host lookup) -> base_lm_kv(CPU) -> +type_embed -> tts_lm_kv(CPU) -> cond
     --host: per speech token (6 per text window)-->
         5-step DPM-Solver++ loop:
             diffusion_head(GPU) x2 (cond + null prompt) -> CFG blend -> v -> latent[64]
         acoustic_connector(host) + type_embed -> tts_lm_kv(CPU)     -> next cond
                                                -> neg tts_lm_kv(CPU) -> next null cond
         eos_classifier(host) > 0.5 -> stop
     --host: accumulate latents-->  acoustic_decoder(CPU) -> waveform[N*3200] @ 24 kHz
```

The two LMs keep their KV cache **host-side**: a packed `[1, L*nkv, Pmax, 64]` key/value tensor is
fed into the step graph and the new column read back each token (the ML-Drift-safe "state as graph
I/O" pattern — Mali silently corrupts an in-graph stateful cache). The **voice** is a precomputed
prompt KV cache (`voice_*.bin`); this app ships the `en-Emma_woman` voice. The realtime checkpoint
is decoder-only (no acoustic encoder), so voices cannot be cloned on-device — they are exported
offline. `VibeVoiceSynthesizer.kt` does the host orchestration; `BpeTokenizer.kt` is the Qwen2
byte-level BPE tokenizer.

## Build & run

```bash
cd kotlin_cpu_gpu/android
./gradlew :app:installDebug
# the four .tflite graphs + three assets are pushed to the app's external files dir (too big to bundle):
./install_to_device.sh <dir-with-the-files>
```

Get the graphs + assets (`vv_*.tflite`, `embed_tokens.f16`, `glue.f32`, `voice_en-Emma_woman.bin`)
from Hugging Face or build them with [`conversion/`](conversion/). The BPE tokenizer's `vocab.json`
+ `merges.txt` are bundled in the app assets. The first launch shows "Model not found" until the
install script has run.

## App architecture

The Android app is **MVVM + Jetpack Compose** (Compose Material 1). `MainActivity` is a thin
`ComponentActivity` that only hosts the Compose tree; all model state lives in `MainViewModel`,
which is surfaced to the UI as a single immutable `UiState` (`StateFlow`). The two graph wrappers
(`VibeVoiceSynthesizer`, `BpeTokenizer`) are unchanged host-orchestration code.

All model work runs on one confined worker (`Dispatchers.Default.limitedParallelism(1)`), because
the graphs reuse native input/output buffers and must not be called concurrently. The models are
loaded in the ViewModel's `init` (but **not** warmed up — generation is slow, so the first
synthesis pays the one-time GPU/JIT cost); `synthesize(text)` tokenizes, runs the pipeline, and
plays the waveform through an `AudioTrack` as a side effect (audio is not held in `UiState`).
`onCleared` stops/releases the `AudioTrack` and closes the graphs.

| File | Role |
| :-- | :-- |
| `MainActivity.kt` | Thin `ComponentActivity`; hosts the Compose UI and collects `UiState`. |
| `MainViewModel.kt` | Loads the graphs, runs `synthesize(text)` + `AudioTrack` playback on the confined worker, owns `UiState`. |
| `UiState.kt` | Immutable UI snapshot (`isModelReady`, `isSynthesizing`, `statusMessage`, `errorMessage`). |
| `view/TtsScreen.kt` | Compose screen: text field + Speak button + status text. |
| `view/Theme.kt`, `view/Color.kt` | Compose Material 1 theme. |
| `VibeVoiceSynthesizer.kt` | Host orchestration of the two LMs + diffusion head + σ-VAE decoder, the host-side KV caches, and the DPM-Solver++ sampler. |
| `BpeTokenizer.kt` | Qwen2 byte-level BPE tokenizer (vocab.json + merges.txt from assets). |

## Notes

- **Stochastic, autoregressive**: each frame draws fresh diffusion noise, so two runs differ.
  Quality is judged by ear on-device; a short sentence generates in ~30 s on a Pixel 8a (the two
  fp32 LMs and the fp32 decoder are the cost — this is a **correctness-first** placement, not a
  real-time one, since the ML Drift decoder bug forces the decoder to CPU).
- **Voice presets** are the reference model's precomputed prompt KV caches; swapping voices swaps
  the `voice_*.bin` file.

## License

Model: MIT (VibeVoice-Realtime-0.5B / Qwen2.5-0.5B backbone).

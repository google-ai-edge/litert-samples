# Streaming Text-to-Speech with LiteRT — KittenTTS nano (dynamic length)

An Android sample that runs [KittenTTS nano 0.8](https://github.com/KittenML/KittenTTS)
(StyleTTS2 + ISTFTNet + mini-ALBERT, Apache-2.0) fully on device and **streams the audio**: type
any text, tap Speak, and playback starts after the first sentence while the rest still
synthesizes.

What the demo shows (all numbers measured on a Pixel 8a, CPU/XNNPACK, 4 threads):

- **Small** — 15M params, **32 MB fp16 on disk** for all three synthesis graphs; 8 voices,
  24 kHz. The size is shown live in the app's metrics card.
- **Fully offline** — the APK declares **no network permission**, so the OS itself guarantees the
  app never touches the network; the demo runs in airplane mode. The screen states this on a
  badge, and it holds by construction: models load from local storage, and there is no network
  code to begin with.
- **Streaming ⇒ short time-to-first-audio** — sentence-level producer/consumer: sentence N plays
  while N+1 synthesizes. **TTFA ≈ 370 ms** from tap to voice; the live TTFA readout is the
  headline metric on screen.
- **Dynamic sequence length ⇒ no padding buckets** — one set of graphs runs any sentence length,
  compute scales with the text. **RTF ≈ 0.22–0.29**, so synthesis stays several times ahead of
  playback and the stream cannot underrun.

The last two are what differentiate this from the sibling
[`text_to_speech`](../text_to_speech) (Matcha-TTS) sample: the graphs were converted through
TF/Keras so the TFLite converter emits *fused dynamic-length LSTM kernels*, and that is why the
app drives the classic **Interpreter API** (`resizeInput` per sentence) instead of the
fixed-shape CompiledModel path.

## Models

| Graph | In → Out | fp16 |
| :-- | :-- | :--: |
| `kitten_predictor_fp16.tflite` | ids[1,N] i32 + style[1,256] + speed[1] → d[1,N,256], t_en[1,N,128], durations[N] i32 | 17.0 MB |
| `kitten_prosody_fp16.tflite` | en[1,T,256] + style → f0[1,2T], n[1,2T], har[1,120T+1,22] | 1.7 MB |
| `kitten_vocoder_fp16.tflite` | asr[1,T,128] + f0/n/har + style → wav[1,600T] @ 24 kHz | 13.4 MB |
| English G2P (DeepPhonemizer) | text[1,96] → logits[1,96,64] | shared with the Matcha sample |

All three synthesis graphs run on the LiteRT CPU interpreter (XNNPACK, 4 threads); the G2P runs on
the CompiledModel CPU accelerator. Host glue between graphs is a ~10-line row-repeat alignment
(`en = repeat(d, durations)`), verified bit-exact against the reference ONNX's in-graph `Loop`.
Conversion + verification (accuracy inside the reference model's own run-to-run noise floor,
durations bit-identical): see [`conversion/`](conversion/).

## Pipeline

```
text --chunk into sentences--> per sentence:
     --G2P (dict + neural, CPU)-->            symbol ids
     --predictor (dynamic N)-->               token features + durations
     --host: repeat rows by durations-->      aligned features [1,T,·]
     --prosody + vocoder (dynamic T)-->       float PCM @ 24 kHz
     --> AudioTrack stream  (next sentence synthesizes while this one plays)
```

Measured on a Pixel 8a (CPU, 4 threads, XNNPACK): **TTFA ≈ 370 ms** with a short opening
sentence, **RTF ≈ 0.22–0.29** — the stream is always several times ahead of playback. Reference
speed of the same graphs on a Mac M-series CPU: RTF ≈ 0.017. The app displays its own live
numbers on every run.

### Streaming granularity

Sentence-level streaming is **exact** and is the same granularity the upstream pip package uses:
each sentence is one full synthesis, so the waveform is identical to non-streamed output.
(The vocoder alone can also run on overlapping chunks for intra-sentence streaming, but
StyleTTS2's AdaIN statistics make that mode approximate — see `conversion/verify_kitten_litert.py
stream_vocoder`. This app uses the exact mode.)

### G2P (espeak-free)

KittenTTS is trained on espeak en-us IPA over the same 178-symbol StyleTTS2 table as the Matcha
sample, so this app reuses that sample's frontend unchanged: a 275k-entry espeak-IPA dictionary
([OpenPhonemizer](https://github.com/NeuralVox/OpenPhonemizer), Clear BSD) as primary +
[DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer) (MIT) on LiteRT for
out-of-dictionary words, plus host-side normalization (acronyms spelled out, numbers as words).
One difference: punctuation is emitted as its own space-separated token, matching the upstream
pip package's tokenizer that the duration predictor was tuned against.

## Build & run

```bash
cd kotlin_cpu/android
./gradlew :app:installDebug
# the models are pushed to the app's filesDir (too big to bundle):
./install_to_device.sh <dir-with-the-files>
```

The push needs five files, all built by [`conversion/`](conversion/):
`kitten_predictor_fp16.tflite`, `kitten_prosody_fp16.tflite`, `kitten_vocoder_fp16.tflite`,
`voices.bin` (repacked `voices.npz`), and `dp_g2p_matcha_fp16.tflite` (the same G2P graph as the
[`text_to_speech`](../text_to_speech) sample — Hugging Face
[`litert-community/Matcha-TTS`](https://huggingface.co/litert-community/Matcha-TTS)).
The host tables (`config.json`, `g2p_dict.txt.gz`, `g2p_meta.json`) are bundled in the app assets.
The first launch shows "model not found" until the install script has run.

## App architecture

MVVM + Jetpack Compose (Material 1), same shape as the Matcha sample. All model work runs on one
confined worker (`Dispatchers.Default.limitedParallelism(1)`) because the graphs reuse native
buffers; playback drains a channel into a streaming `AudioTrack` on a separate coroutine.

| File | Role |
| :-- | :-- |
| `MainActivity.kt` | Thin `ComponentActivity`; hosts the Compose UI and collects `UiState`. |
| `MainViewModel.kt` | Loads + warms the graphs; `speak()` runs the sentence pipeline into a streaming `AudioTrack`; owns `UiState`. |
| `UiState.kt` | Immutable UI snapshot, including the live `Metrics` (TTFA, RTF, model size). |
| `KittenSynthesizer.kt` | The three dynamic-shape graphs on the Interpreter API + host glue. |
| `KittenG2P.kt` | Text → symbol ids (dictionary + neural G2P; shared with the Matcha sample). |
| `SentenceChunker.kt` | Faithful port of the pip package's `chunk_text` (the streaming granularity). |
| `view/TtsScreen.kt` | Compose screen: text field, voice picker, Speak/Stop, metrics card, streaming indicator. |

## License

Model weights: Apache-2.0 (KittenML). G2P: Clear BSD (OpenPhonemizer) + MIT (DeepPhonemizer).
No GPL components at runtime (espeak is not used).

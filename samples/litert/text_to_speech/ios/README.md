# Text to speech on iOS

Generate speech from English text with the published Matcha-TTS graphs and the LiteRT Swift CompiledModel API.

## Features

- Dictionary-first pronunciation with a neural fallback for unknown words.
- Acronym and number reading that matches the frontend of the [Android sample](../kotlin_cpu_gpu/android/).
- Seeded synthesis, adjustable integration steps, and mono Float32 playback.
- Independent text-encoder and vocoder backend and precision controls.
- Requested and effective backend reporting with timings through output readback.
- A headless workflow that saves a timing report and audio.

## Architecture

| Swift file | Role | Kotlin counterpart in the [Android sample](../kotlin_cpu_gpu/android/) |
|---|---|---|
| `Sources/MatchaCore/MatchaFrontend.swift` | Tokenization, dictionary lookup, number and acronym expansion, symbol IDs | `MatchaG2P.kt` |
| `Sources/MatchaCore/G2PCodec.swift` | Character tensors, argmax and phoneme decoding | `MatchaG2P.kt` |
| `Sources/MatchaCore/MatchaMath.swift` | Embeddings, masks, duration regulation, time embeddings, integration | `MatchaSynthesizer.kt` |
| `Sources/MatchaCore/GaussianNoise.swift` | Seeded start noise | `MatchaSynthesizer.kt` noise initialization |
| `Sources/MatchaRuntime/MatchaG2PRunner.swift` | CPU pronunciation graph | `MatchaG2P.kt` |
| `Sources/MatchaRuntime/MatchaSynthesizer.swift` | Acoustic graph orchestration | `MatchaSynthesizer.kt` |
| `Sources/MatchaRuntime/LiteRTGraph.swift` | Compiled model and reusable buffers | Graph setup in `MatchaG2P.kt` and `MatchaSynthesizer.kt` |
| `Sources/App/SpeechWorker.swift` | Serial model ownership and measurements | `MainViewModel.kt` |
| `Sources/App/SpeechController.swift`, `SpeechView.swift` | Observable UI state and controls | `MainViewModel.kt` and the Android screen |
| `Sources/App/AudioPlayer.swift` | Audio playback | Playback in `MainViewModel.kt` |
| `Sources/App/HeadlessRun.swift` | Launch arguments, timing summaries, WAV and JSON output | No direct counterpart |

## Prerequisites and setup

Use an Apple silicon Mac, Xcode 27, Git with Git LFS, Bazelisk, and XcodeGen. The app targets iOS 17 or later on iPhone and iPad. These setup commands were checked on an M4 Max, macOS 27, Xcode 27.0 RC (27A266a), on 2026-09-20, with LiteRT pinned to the revision below. Simulator builds use arm64 because the pinned runtime contains only that simulator slice.

Place the LiteRT checkout next to `litert-samples`. From the directory containing `litert-samples`:

```sh
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/google-ai-edge/LiteRT.git LiteRT
cd LiteRT
GIT_LFS_SKIP_SMUDGE=1 git checkout ccff78483e972a975b5300242084a6e2147c9776
git lfs pull --include='litert/prebuilt/ios_arm64/*,litert/prebuilt/ios_sim_arm64/*,litert/prebuilt/macos_arm64/*'
cd ../litert-samples/samples/litert/text_to_speech/ios
```

Prepare the unchanged upstream Swift package, download the model assets, and generate the project:

```sh
bash prep_litert.sh
bash prep_resources.sh
bash generate_project.sh
xcodebuild -project TextToSpeech.xcodeproj -scheme TextToSpeech -destination 'generic/platform=iOS' CODE_SIGNING_ALLOWED=NO build
swift test
```

The scripts use the default sibling checkout path. `LITERT_CHECKOUT` can select another pinned checkout. The runtime script builds the C and Metal xcframeworks and packages the OSS macOS library as the third archive required by the package manifest: SwiftPM resolves every binary target the manifest declares, including the macOS one, even for an iOS build. It does not patch LiteRT. A different checkout revision produces a warning identifying the tested and selected revisions. The project references the `LiteRT` and `LiteRtMetalAccelerator` package products and embeds both frameworks.

The build command above produces an unsigned iOS app. For an unsigned simulator build:

```sh
xcodebuild -project TextToSpeech.xcodeproj -scheme TextToSpeech -destination 'generic/platform=iOS Simulator' CODE_SIGNING_ALLOWED=NO build
```

For device use, open `TextToSpeech.xcodeproj`, select a signing team, and run the app. Alternatively, supply `SIGNING_TEAM` to the headless device script. Models and host tables live in the ignored `Models/` folder, which is copied into the app. The dictionary is decompressed during preparation; the app reads plain UTF-8 and needs no gzip implementation.

## How it works

The frontend expands text into IPA and symbol IDs. The synthesizer intersperses blank IDs, looks up embeddings, predicts durations, and expands text features into acoustic frames. An Euler loop invokes the decoder before mel denormalization and vocoding. Playback uses the model's mono Float32 output at 22,050 Hz.

One shared `Environment` owns the runtime context. One serial worker owns the models and reused buffers. Encoder outputs are identified by shape. Metal FP32 uses a NUL-terminated `precision = 2\n` TOML payload under the `gpu_options` opaque option; 2 is `kLiteRtDelegatePrecisionFp32` in the C runtime, and the Swift package at this revision has no typed GPU option. Graphs placed on Metal request the GPU accelerator alone, so a graph Metal cannot compile fails visibly instead of running partly on CPU; the results still report the runtime's effective backend. The embedded accelerator is located by the upstream environment's bundle probing.

The following correctness observations were measured on **M4 Max, 2026-09-20, LiteRT Swift sources at ccff78483e and the matching macOS libraries; Python comparisons used ai-edge-litert 2.2.0**. They do not establish iPhone performance.

| Graph | App default | Measured reason |
|---|---|---|
| Pronunciation | CPU | Metal rejected INT64 CAST and RESHAPE operations. |
| Text encoder | CPU | Default Metal precision changed durations on 7 of 34 texts; FP32 preserved all 34. |
| Decoder | CPU | FP32 removed nonfinite results but left worst relative RMS 0.85931, above the 0.01 acceptance bound. |
| Vocoder | Metal, default precision | All 34 audio comparisons stayed below 0.5 dB mean log-mel distance; worst was 0.284341 dB. |

“Use encoder and vocoder Metal FP32” selects the alternative placement in one tap. It retained exact durations for the four tested utterances and had worst mean log-mel distance 0.000118 dB on the same M4 Max/date/runtime. Pronunciation and decoder stay on CPU in both presets.

The placements were checked again on an iPhone 17 Pro (iOS 27.0, Release build, 2026-09-24) against the Mac CPU tensors for four of the test texts: the default placement keeps the audio within 0.28 dB mean log-mel distance, the FP32 preset within 0.0002 dB with durations exact, and the decoder on Metal FP32 still diverges (relative RMS 0.65), so it stays on CPU. The default keeps the default-precision vocoder because it is the fastest placement on the phone; see the table below.

The results show the requested backend and precision alongside the runtime's effective backend, `isFullyAccelerated()` and GPU-environment information. A missing GPU environment or incomplete acceleration is reported as a fallback, and compilation failures are surfaced. Graph times include output readback; the decoder row sums all integration steps.

## Model information

The four published graphs and host assets come from [Matcha-TTS on the model hub](https://huggingface.co/litert-community/Matcha-TTS/tree/143eb4236e8c81ea6849ede31f7f088ea2393cfa), revision `143eb4236e8c81ea6849ede31f7f088ea2393cfa`. Preparation verifies SHA-256 hashes. No model conversion is performed.

| Asset | Purpose |
|---|---|
| `dp_g2p_matcha_fp16.tflite` | Neural word pronunciation |
| `matcha_textenc_fp16.tflite` | Text features and duration prediction |
| `matcha_decoder_fp16.tflite` | Conditional flow velocity |
| `matcha_vocoder_fp16.tflite` | Mel-to-waveform generation |
| `config.json`, `emb.bin`, `g2p_meta.json`, `g2p_dict.txt.gz` | Symbol, embedding, pronunciation and dictionary tables |

Model constants are a maximum of 256 text positions and 512 mel frames, with 256 audio samples per frame. Long text is truncated by those fixed graph limits. The graph files total 92,148,832 bytes at the pinned hub revision; they are downloaded, not included in source control.

## Performance

Measured with the headless run described below on an iPhone 17 Pro (iOS 27.0), Release build, LiteRT `ccff78483e`, on 2026-09-24: seed 7, 10 decoder steps, 2 warm-ups and 10 timed runs per launch, thermal state nominal at the start and end of each launch. Milliseconds are medians per graph and include output readback; the decoder row sums its 10 steps. The short text is "The rain is soft today." (25 phonemes, 152 mel frames) and the long text is a 92-phoneme sentence at 440 mel frames. Both texts are covered by the dictionary, so the pronunciation graph did not run. Graph shapes are fixed, so the time barely depends on the text length.

| Placement | Text | Text encoder | Decoder, 10 steps | Vocoder | Total | Audio (s) | RTF |
|---|---|---:|---:|---:|---:|---:|---:|
| Default (vocoder Metal) | short | 6.3 | 107.3 | 250.1 | 366 | 1.76 | 0.21 |
| Encoder and vocoder Metal FP32 | short | 4.4 | 121.8 | 407.4 | 536 | 1.76 | 0.30 |
| All CPU | short | 6.6 | 124.4 | 611.7 | 744 | 1.76 | 0.42 |
| Default (vocoder Metal) | long | 6.8 | 126.8 | 252.2 | 387 | 5.11 | 0.08 |
| Encoder and vocoder Metal FP32 | long | 4.5 | 138.3 | 409.5 | 553 | 5.11 | 0.11 |
| All CPU | long | 7.2 | 131.7 | 613.8 | 754 | 5.11 | 0.15 |

On the phone the default placement's audio is within 0.29 dB mean log-mel distance of the all-CPU audio for both texts (SNR 19 to 23 dB, so the waveforms differ while the mel spectra agree), and the FP32 preset is within 0.0002 dB (SNR 82 dB or better). For the short text, the phone's all-CPU audio matches the Mac's CPU audio at SNR 96 dB and is bit-identical to the iOS simulator's.

## Headless run

Launch arguments:

```text
-speak "The rain is soft today." -seed 7 -steps 10 -placement te=cpu,dec=cpu,voc=gpu -runs 10
```

Placement lists must specify `te`, `dec`, and `voc`; supported values are `cpu`, `gpu`, and `gpu:fp32`. The decoder must be `cpu`, and pronunciation always uses CPU. The alternative placement is `te=gpu:fp32,dec=cpu,voc=gpu:fp32`.

Headless execution performs two warm-ups, then measures the requested number of runs. `Documents/result.json` contains requested/effective backends, precision, per-graph and total median/min/max milliseconds, audio seconds, RTF, mel length, phoneme count, thermal state, model identifier, OS version, build configuration, and runtime commit when available. `Documents/out.wav` contains the final mono Float32 waveform. The process exits after writing both files.

To build Release, install, launch, and pull those files from a selected connected device:

```sh
DEVICE_ID='<device identifier>' SIGNING_TEAM='<signing team>' \
  bash device_run.sh 'The rain is soft today.' 7 10 'te=cpu,dec=cpu,voc=gpu' 10
```

To inspect the build command without building or contacting a device, run `bash device_run.sh --print-build-command`.

The script saves the pulled files under ignored `output/`. It acts only on the selected device and this app's bundle identifier. `BUNDLE_ID` replaces that identifier in the build, launch, and file copy.

## Tests

`Package.swift` contains only the LiteRT-free core and its tests. Plain `swift test` requires no model files, runtime artifacts, or LiteRT checkout. The tests cover text and symbol equality, UTF-16 semantics, number parsing, and deterministic noise. Test resources total 49,894 bytes across five text/JSON files (largest 43,987 bytes); acoustic tensors and device-comparison code are not shipped.

Swift formatting uses the Xcode toolchain defaults, checked with `swift format lint --strict`.

## License

Sample code is Apache-2.0. Matcha-TTS and HiFi-GAN are MIT; the OpenPhonemizer dictionary is Clear BSD and DeepPhonemizer is MIT. See [third-party notices](THIRD_PARTY.md) for source links and the dictionary excerpt notice.

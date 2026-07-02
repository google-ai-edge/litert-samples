# Speaker Diarization with LiteRT — pyannote 3.1 stack (on-device)

An Android sample that runs **speaker diarization** ("who spoke when") on device following the
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) recipe
(MIT): record a conversation (or pick any audio/video clip) and get a colored per-speaker timeline,
talk-time summary, and per-speaker playback.

```
pcm 16 kHz → [10 s windows]—[PyanNet powerset segmentation, ONNX CPU]→ local speakers
           → solo audio per (window, speaker) —[kaldi fbank+CMN, Kotlin]→ [1,500,80]
           —[WeSpeaker ResNet34, CompiledModel GPU]→ 256-d embeddings —[clustering]→ timeline
```

## Models

| Stage | Model | Runtime (Pixel 8a) |
| :-- | :-- | :--: |
| Segmentation (≤3 local speakers, overlap-aware powerset) | [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) (SincNet+BiLSTM, 1.5 M, MIT) | onnxruntime **CPU** |
| Speaker embedding | [WeSpeaker ResNet34](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) (6.6 M, CC-BY-4.0) | **GPU** |

The segmentation BiLSTM has no mobile-GPU kernel, so it stays on CPU (tiny and fast). The heavy
embedding CNN loads with `CompiledModel.create(...)` on `Accelerator.GPU`: **108 / 108 nodes on
`LITERT_CL`** (full residency, 1 partition), ~1.2 ms per 5 s window, device-vs-PyTorch cosine
**0.99997** (fp16, 13.4 MB). The segmentation ONNX matches PyTorch at corr 1.0 (per-frame argmax
agreement 100%).

## How the pieces fit

Sliding 10 s windows (5 s hop) → powerset argmax → per-(window, local-speaker) units with ≥1 s of
solo speech → WeSpeaker embedding of each unit's concatenated solo audio (tile-padded to the fixed
5.015 s fbank window, CMN'd) → agglomerative clustering (centroid linkage, cosine distance,
threshold 0.7046 from the 3.1 config) → windows stitched by their central regions into a global
timeline. The kaldi-fbank front-end (hamming 25/10 ms, 80 mel, dither 0, ×2¹⁵) is ported to Kotlin
with precomputed mel banks (`assets/mel80_257.bin`), verified against
`torchaudio.compliance.kaldi.fbank` (corr 1.0). The embedding conversion
([`conversion/`](conversion/)) needs zero re-authoring except an fp16-safe statistics-pooling std.

## Run it

1. Get the models: build with [`conversion/build_diar.py`](conversion/build_diar.py) or download
   from [litert-community/Speaker-Diarization-LiteRT](https://huggingface.co/litert-community/Speaker-Diarization-LiteRT).
2. Install the app: `cd kotlin_cpu_gpu/android && ./gradlew :app:installDebug`
3. Push the models: `./install_to_device.sh <dir-with-the-models>`
4. Record a conversation (up to 120 s) or pick a clip; read the timeline, tap ▶ per speaker.

Upstream: [pyannote.audio](https://github.com/pyannote/pyannote-audio) (MIT) — please cite
Bredin 2023 when using these models; [WeSpeaker](https://github.com/wenet-e2e/wespeaker)
(Apache-2.0 code, CC-BY-4.0 weights).

# Text-to-speech with a speech LM: Qwen3-TTS on LiteRT (Compiled Model API)

This sample runs [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) (Apache-2.0), a multilingual codec-LM text-to-speech model with 3-second voice cloning, fully on device with LiteRT. It complements the [Matcha-TTS sample](../text_to_speech): Matcha is a compact flow-matching acoustic model, while Qwen3-TTS represents the modern **speech-LM family** (a Qwen3 talker LM emitting 12.5 Hz frames of 16 codec tokens, decoded to 24 kHz PCM by a neural codec).

The model runs as **three LiteRT graphs orchestrated by a host-side loop** (the Compiled Model pattern):

1. **Talker** — a standard Qwen3 transformer (28 layers, hidden 1024) with prefill/decode signatures and an explicit KV cache. It predicts the first (semantic) codebook of each audio frame. Exported with the stock `litert-torch` LLM exporter; the lm_head is augmented with an identity block so each decode step also exposes its hidden state, which seeds the MTP.
2. **MTP (code predictor)** — a 5-layer transformer run as a 15-step autoregressive inner loop per frame, predicting the 15 residual codebooks. One decode-step graph with a 17-slot KV cache, invoked 17 times per frame.
3. **Codec decoder** — the Qwen3-TTS-Tokenizer-12Hz decoder (RVQ embeddings, an 8-layer transformer, and a causal ConvNet upsampler) turning accumulated codes into waveform, in 64-frame chunks with 25 frames of left context.

Everything between the graphs — BPE tokenization, prompt embedding assembly (control tokens, speaker x-vector, streamed per-frame text conditioning), 16-codebook embedding aggregation, sampling with repetition penalty and control-token suppression, and chunked codec decoding — is plain NumPy in `python/qwen3_tts_pipeline.py`. This host loop is exactly the piece a future LiteRT-LM Engine "audio-LM" session type would absorb; see the conversion report linked below for the concrete list of Engine capabilities this maps to.

## Quick start (Python, desktop)

```bash
cd python
pip install -r requirements.txt
python synthesize.py --text "Hello from LiteRT running fully on device." --output hello.wav
```

The first run downloads ~1.4 GB of model files from [litert-community/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base) and speaks in the bundled demo voice. Useful flags:

- `--language`: `english` (default), `chinese`, `japanese`, `korean`, `german`, `french`, `spanish`, `italian`, `portuguese`, `russian`, or `auto`.
- `--speaker my_voice.npy`: a 1024-d x-vector; enroll a new voice from ~3 s of reference audio with `conversion/extract_speaker_embedding.py`.
- `--greedy`: deterministic decoding. With `--talker fp32 --greedy` the pipeline reproduces the PyTorch reference implementation **token-for-token** (waveform correlation 1.0) — this is the correctness gate used during conversion.
- `--talker int4` (default): the blockwise-int4 talker (256 MB vs 1.8 GB fp32). Its sampled output differs from fp32 but transcribes identically under an ASR round-trip check.

## Android app (`kotlin_cpu/android`)

A Kotlin port of the same host loop (Compiled Model API, CPU): byte-level BPE tokenizer (verified against reference token ids by a startup self-test, including Japanese input), memory-mapped fp16 embedding tables, the talker/MTP/codec loop with ping-pong KV-cache buffers (no per-step cache copies), and AudioTrack playback.

```bash
cd kotlin_cpu/android
./gradlew :app:installDebug        # or open in Android Studio
./install_to_device.sh             # downloads ~1.4 GB from Hugging Face, adb-pushes it
```

Launch the app, type text, pick a language, tap Speak.

## Performance

| Stage (per 80 ms frame) | M4 Max CPU (Python) | Pixel 8a (Kotlin) |
|---|---|---|
| Talker decode | 45–50 ms | ~60 ms |
| MTP inner loop (17 invokes) | ~148 ms | ~370 ms |
| Codec decode (amortized) | ~10 ms | ~98 ms |
| **Total** | **~205 ms → RTF ≈ 2.5** | **~530 ms → RTF ≈ 6.7** |

Measured end to end on Pixel 8a: 4.16 s of audio in 28.0 s (prefill 585 ms, talker 3.1 s, MTP 19.2 s, codec 5.1 s). Not yet realtime: the MTP inner loop dominates because a 78 M-parameter model streams its weights 17 times per frame. The two known levers (folding the 15-step loop into a single unrolled graph with in-graph sampling, and a calibrated low-bit MTP) are engineering work, not converter gaps — see the feasibility report in the model card for the analysis.

## Conversion

All conversion scripts are in [`conversion/`](conversion/): talker checkpoint synthesis and export, MTP graph authoring, codec decoder export, host-table extraction, and voice enrollment. Each converted graph is numerically verified against the PyTorch reference (correlation 1.0). See [`conversion/README.md`](conversion/README.md).

## License

The sample code is Apache-2.0. Qwen3-TTS weights are Apache-2.0 (Alibaba Qwen team); the converted artifacts on Hugging Face retain that license.

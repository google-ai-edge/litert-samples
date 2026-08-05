# KittenTTS nano streaming TTS — Python version

The Python twin of the [Android sample](../kotlin_cpu): the same five model
files, the same espeak-free G2P (275k-entry IPA dictionary + neural fallback),
the same sentence-level streaming granularity. Runs anywhere the
`ai-edge-litert` pip package does — including a **Raspberry Pi 5** (64-bit OS,
Python 3.10+).

```bash
pip install numpy ai-edge-litert

# Build the models once (see ../conversion/README.md) — outputs land in ../out/.
python say.py "Hello there! This runs fully offline." -o hello.wav
aplay hello.wav   # or: afplay hello.wav on macOS
```

`say.py` prints the streaming timeline — time-to-first-audio and per-sentence
RTF — and writes a 24 kHz WAV. On a Mac M-series CPU the overall RTF is
≈ 0.03; sentence N's audio is ready long before sentence N−1 finishes playing.

## Files

| File | Role |
| :-- | :-- |
| `say.py` | CLI demo: streaming synthesis, metrics printout, WAV output. |
| `kitten_tts.py` | Engine module: G2P, sentence chunker, the three-graph pipeline. |

The host tables (`config.json`, `g2p_dict.txt.gz`, `g2p_meta.json`) are read
from the Android app's assets directory, so there is exactly one copy in the
repo; `--assets-dir` overrides the location.

## LiteRT API usage

The sample intentionally exercises **both** LiteRT Python APIs, matching what
each graph allows today:

- **G2P + vocoder — CompiledModel API.** The vocoder has a dynamic sequence
  length; every sentence calls `resize_input_tensor`, then creates the input
  buffers *after* the resize (so they pick up the new shapes), and passes an
  output buffer of the exact final size wrapped from host memory with
  `TensorBuffer.create_from_host_memory` (output shapes propagate when the
  model allocates during `run`, after default output buffers would have been
  sized).
- **Predictor + prosody — Interpreter API.** These two graphs contain fused
  dynamic-length LSTM kernels whose hidden state lives in TFLite *variable
  tensors*, which the CompiledModel model loader does not accept yet
  (b/365299994). The Interpreter path must call `reset_all_variables()`
  before every invoke — without it, a same-length invoke starts from the
  previous sentence's final LSTM state and corrupts the predicted durations.

## Verification

Checked against the conversion's Mac fp32 reference outputs (fp16 graphs):
predicted durations are **bit-identical** on all reference sentences, and
log-spectrogram correlation is 0.984–0.989 — the same fp16-vs-fp32 bar the
Android app was verified to. Repeated synthesis of the same sentence is
bit-identical (the LSTM state reset is effective).

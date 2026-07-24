# Qwen3-TTS → LiteRT conversion

These scripts convert [Qwen/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) into the three .tflite graphs plus host-side tables used by the sample, and numerically verify every step against the PyTorch reference. The published artifacts live at [litert-community/Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/litert-community/Qwen3-TTS-12Hz-0.6B-Base).

## Environments

Two Python environments are required because the reference implementation pins an older transformers:

- **convert env**: `litert-torch==0.9.1`, `torch==2.12`, `transformers==5.12.1`, `ai-edge-quantizer`, `safetensors`, `numpy` — runs every `export_*.py` / `extract_*.py` / `verify_*.py`.
- **reference env**: `qwen-tts==0.1.1` (pins `transformers==4.57.3`), plus `torchaudio`, `librosa`, `soundfile`, `sox`, `onnxruntime` — runs the `dump_*_ref.py` scripts and `extract_speaker_embedding.py`.

## Pipeline

Run from this directory. Reference dumps (steps 1, 3, 5) are optional but recommended: the export scripts verify against them when present.

| # | Script (env) | What it does |
|---|---|---|
| 1 | `dump_talker_ref.py` (reference) | Dumps talker activations on a random embedding sequence. |
| 2 | `extract_talker_ckpt.py` → `export_talker.py` → `verify_talker.py` (convert) | Synthesizes a standard Qwen3ForCausalLM checkpoint from the talker weights (the TTS mrope provably reduces to standard RoPE, verified bit-exact) and exports prefill/decode graphs with the stock litert-torch LLM exporter. `RECIPE=BOCTAV4` produces the blockwise-int4 variant. |
| 3 | `dump_mtp_ref.py` (reference) | Dumps a greedy MTP inner-loop trace. |
| 4 | `export_mtp.py` (convert) | Authors and exports the MTP decode-step graph (17-slot KV cache, one-hot slot update, all 15 lm_heads per step) and verifies 15/15 greedy tokens. |
| 5 | `dump_codec_ref.py` (reference) | Dumps codec decoder output on fixed random codes. |
| 6 | `export_codec.py` (convert) | Exports the codec decoder at a fixed 64-frame chunk via `qtok12/` (a two-file port of the reference decoder that loads under transformers 5.x; see the file headers for the three API shims, including the silent zeroed-`inv_freq` pitfall). |
| 7 | `extract_host_tables.py` (convert) | Extracts codec/text/MTP embedding tables and the text projection MLP as .npy/.npz. |
| 8 | `assemble_release.py` (convert) | Collects everything under `out/release/` with the published file names (large tables cast to fp16 — verified not to change generated tokens). |
| 9 | `extract_speaker_embedding.py` (reference) | Enrolls a new voice: ~3 s wav → 1024-d x-vector .npy. |

## Verification results

- Talker: synthesized checkpoint matches the original talker bit-exactly (max abs diff 0.0); tflite decode sweep correlation 1.0, top-1 100%.
- MTP: 15/15 greedy tokens vs the reference inner loop; logits correlation 1.0.
- Codec decoder: torch port bit-exact vs the 4.57 reference; tflite correlation 1.0 (max abs diff 1.8e-5); right-pad invariance exact (fully causal stack).
- End to end (sample with `--talker fp32 --greedy`): token-for-token identical codes, waveform correlation 1.0, ASR round-trip returns the input sentence.

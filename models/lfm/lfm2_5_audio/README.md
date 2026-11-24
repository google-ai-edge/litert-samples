# LiquidAI LFM2.5-Audio-1.5B (LiteRT Omni)

On-device C++ runtime and PyTorch-to-LiteRT exporter for
[LiquidAI/LFM2.5-Audio-1.5B](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B),
an end-to-end **Any-to-Any** speech and text model supporting:

*   **Speech-to-Text (ASR)**: 16 kHz mono 16-bit PCM WAV input → text
    transcript.
*   **Text-to-Text**: text prompt → text response (`--generate_audio=false`).
*   **Text-to-Text + Speech (Interleaved / TTS)**: text prompt → interleaved
    text tokens and 24 kHz mono 16-bit PCM WAV audio output
    (`--generate_audio=true`).
*   **Speech-to-Text + Speech**: 16 kHz WAV audio + optional text prompt → text
    transcript and 24 kHz WAV response.

> **Naming note:** The directory is organized under `lfm/lfm2_5_audio/` while
> source filenames (`lfm2_audio.*`), build targets (`runner:lfm2_audio`,
> `runner:lfm2_audio_runner`), and the protobuf message
> (`litert.lm.proto.Lfm2Audio`) retain `lfm2_audio` / `Lfm2Audio` to preserve
> revision history.

## Architecture & `.litertlm` Bundle Sections

An exported `.litertlm` bundle contains **8 sections** (`1` metadata proto, `1`
tokenizer, and `6` TFLite submodels):

<!-- mdformat off(prevent table wrapping) -->

| Section / Key | Bundle Section Type | Description |
| :--- | :--- | :--- |
| `Any2AnyMetadataProto` | Metadata protobuf | `Any2AnyMetadata` with `litert.lm.proto.Lfm2Audio` packed into `any2any_model_type.custom_model` (field `100`). |
| `HF_Tokenizer_Zlib` | Tokenizer | Compressed Hugging Face `tokenizer.json` (65,536 tokens). |
| `audio_encoder` | `tf_lite_audio_encoder` | FastConformer audio encoder (128-bin NeMo log-mel input at 16 kHz, 8× subsampling). |
| `text_embedder` | `tf_lite_text_embedder` | Token embedding lookup table (`hidden_size = 2048`). |
| `llm_backbone` | `tf_lite_llm_backbone` | Hybrid LFM2-1.2B short-conv + GQA LLM backbone (`hidden_size = 2048`, `vocab_size = 65536`). |
| `tts_projector` | `tf_lite_tts_projector` | Audio codebook embedding projector feeding the Depthformer. |
| `tts_decoder` | `tf_lite_tts_decoder` | 6-layer, 8-codebook RQ-Transformer Depthformer (`tts_hidden_size = 1024`, `tts_vocab_size = 2049` including `EOAudio = 2048`). |
| `audio_vocoder` | `tf_lite_audio_vocoder` | LFM2 neural audio detokenizer emitting 24 kHz mono PCM (`1920` samples/frame; overlap-discard chunking with `18` left / `1` right context frames). |

<!-- mdformat on -->

### Key Audio Parameters

*   **Input frontend**: 16 kHz mono 16-bit PCM WAV, pre-emphasis `0.97`,
    centered STFT (`n_fft = 512`, `win_length = 400`, `hop_length = 160`),
    128-bin Slaney log-mel (`log(mel + 2^-24)`), per-feature Bessel
    normalization.
*   **Interleaved schedule**: alternates 6 text tokens and 12 audio frames until
    `<|text_end|>` and `EOAudio`.
*   **Output waveform**: 24 kHz mono 16-bit PCM WAV.

## Directory Layout

```text
models/lfm/lfm2_5_audio/
├── README.md                          # This file
├── runner/
│   ├── BUILD                          # Proto, per-stage cc_library, runner, and cc_test targets
│   ├── lfm2_audio.proto               # Lfm2Audio protobuf schema (custom_model = 100)
│   ├── lfm2_audio_model_config.h      # Model dimensions, special tokens, and defaults
│   ├── io_types.h                     # Shared inter-stage I/O structs
│   ├── input_fan_out_stage.{h,cc}     # Coalesces audio/text inputs and sequences metadata
│   ├── audio_stage.{h,cc}             # NeMo log-mel frontend + FastConformer audio encoder
│   ├── text_stage.{h,cc}              # Control-token escaping + text token embedding
│   ├── frontend_stage.{h,cc}          # Prompt framing and multimodal embedding assembly
│   ├── generation_stage.{h,cc}        # Interleaved/sequential decode, Depthformer, detokenizer
│   ├── lfm2_audio_factory.{h,cc}      # ModelResources loading and stage wiring
│   ├── lfm2_audio.cc                  # Metadata unpack, session lease, OmniRegistry registration
│   ├── lfm2_audio_test_utils.{h,cc}   # Synthetic .litertlm and WAV builders for hermetic tests
│   ├── lfm2_audio_test.cc             # Unit & OmniEngine/OmniRegistry integration tests
│   └── lfm2_audio_runner_test.cc      # In-process runner and posix_spawn CLI binary tests
└── export/
    ├── requirements.txt               # Standalone pip/venv dependencies (litert-torch-nightly)
    ├── modules.py                     # Exportable PyTorch submodule wrappers
    ├── lfm2_audio.py                  # Submodel conversion, checkpoint check, metadata packing
    ├── extension.py                   # ModelExtension registration for export_hf
    ├── export_main.py                 # Thin Fire CLI entrypoint
    ├── test_lfm2_audio.py             # Config, key verification, and export parity tests
    └── export_and_run_test.py         # Cross-language E2E test (requires Bazel runfiles; see below)
```

## Export

Export `LiquidAI/LFM2.5-Audio-1.5B` (or a local Hugging Face checkpoint
directory) into a single `.litertlm` bundle.

Create a virtual environment, install `export/requirements.txt` (requires
`litert-torch-nightly`, which automatically pulls a matching
`litert-lm-builder-nightly`; stable `litert-torch==0.10.0` /
`litert-lm-builder==0.19.2` lack `ModelExtension` and `core.quantization`),
generate the `lfm2_audio_pb2.py` protobuf bindings from the repository root, and
run the exporter module:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r models/lfm/lfm2_5_audio/export/requirements.txt

protoc --python_out=. models/lfm/lfm2_5_audio/runner/lfm2_audio.proto

HF_HOME=/tmp/hf_cache python3 -m models.lfm.lfm2_5_audio.export.export_main \
  --model=LiquidAI/LFM2.5-Audio-1.5B \
  --output_dir=/tmp/lfm2_5_audio_out \
  --task=auto \
  --quantization_recipe=dynamic_wi8_afp32 \
  --cache_length=2048 \
  --prefill_lengths='[128]' \
  --bundle_litert_lm=True
```

Use `--dry_run=True` to inspect the resolved export plan without running
conversion. Optional `lfm2_audio`-specific exporter flags:

*   `--audio_mel_frames` (default: `2048`): Fixed mel-frame length for the audio
    encoder graph.
*   `--detok_chunk_frames` (default: `64`): Chunk frame length for the neural
    audio detokenizer graph.
*   `--export_llm_backbone` (default: `True`): Toggle exporting the LFM2 LLM
    backbone.
*   `--export_tts` (default: `True`): Toggle exporting `tts_projector` and
    `tts_decoder`.
*   `--export_vocoder` (default: `True`): Toggle exporting `audio_vocoder`.
*   `--audio_encoder_quantization_recipe`: Optional quantization override for
    the audio encoder (inherits `--quantization_recipe` if unset).

The output bundle is written to `/tmp/lfm2_5_audio_out/model.litertlm`.

## Run Inference

Registered `--model_name` values: `lfm2.5-audio-1.5b` (canonical),
`LiquidAI/LFM2.5-Audio-1.5B`, `lfm2_audio`, `lfm2-audio`.

> **Tip (filename auto-detection):** If you rename the exported bundle so its
> filename contains a registered alias (for example,
> `lfm2.5-audio-1.5b.litertlm` or `lfm2_audio.litertlm`), you can omit
> `--model_name`.

### Text → Text + Speech (Interleaved)

```bash
bazel run --config=omni -c opt \
  //models/lfm/lfm2_5_audio/runner:lfm2_audio_runner -- \
  --model_path=/tmp/lfm2_5_audio_out/model.litertlm \
  --model_name=lfm2.5-audio-1.5b \
  --text="What is the capital of France?" \
  --generate_audio=true \
  --max_new_tokens=512 \
  --output_path=/tmp/lfm2_response.wav
```

### Speech → Text (ASR)

Pass a 16 kHz mono 16-bit PCM WAV file via `--audio_path`:

```bash
bazel run --config=omni -c opt \
  //models/lfm/lfm2_5_audio/runner:lfm2_audio_runner -- \
  --model_path=/tmp/lfm2_5_audio_out/model.litertlm \
  --model_name=lfm2.5-audio-1.5b \
  --audio_path=/tmp/input_16k.wav \
  --generate_audio=false \
  --max_new_tokens=256 \
  --temperature=0.0 \
  --output_path=/tmp/transcript.txt
```

### Text → Text Only

```bash
bazel run --config=omni -c opt \
  //models/lfm/lfm2_5_audio/runner:lfm2_audio_runner -- \
  --model_path=/tmp/lfm2_5_audio_out/model.litertlm \
  --model_name=lfm2.5-audio-1.5b \
  --text="Summarize the history of acoustics in two sentences." \
  --generate_audio=false \
  --max_new_tokens=256 \
  --temperature=0.0
```

Common runtime flags include `--backend` (`cpu`, `gpu`, or `npu`; default
`cpu`), `--num_threads` (default `4`), and `--cache_dir` (pass `:nocache` to
disable on-disk backend caching).

## Hermetic Tests

```bash
# C++ runtime, factory, and binary runner tests (synthetic weights)
bazel test --config=omni //models/lfm/lfm2_5_audio/runner:all

# Python exporter unit tests (inside the venv with requirements.txt installed
# and lfm2_audio_pb2.py generated via protoc)
python3 -m models.lfm.lfm2_5_audio.export.test_lfm2_audio
```

> **Note:** `export/export_and_run_test.py` looks up compiled `export_main` and
> `lfm2_audio_runner` binaries via Bazel runfiles (`TEST_SRCDIR`) and is not
> runnable directly under the standalone `pip`/`venv` workflow yet. For manual
> end-to-end verification, run `export_main` in the `venv` and invoke
> `bazel run --config=omni //models/lfm/lfm2_5_audio/runner:lfm2_audio_runner`
> on the resulting `.litertlm` bundle.

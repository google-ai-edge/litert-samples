# DeepSeek Janus-Pro (LiteRT Omni)

On-device C++ runtime and PyTorch-to-LiteRT exporter for
[deepseek-ai/Janus-Pro-1B](https://huggingface.co/deepseek-ai/Janus-Pro-1B) (and
`deepseek-ai/Janus-Pro-7B`), a unified **Any-to-Any** multimodal model
combining:

*   **Image Understanding / VQA (Mode A)**: Image (PNG/JPEG/BMP, resized to
    384×384) + text prompt → text answer (`--generate_image=false`).
*   **Text-to-Image Generation (Mode B)**: Text prompt → 384×384 RGB PNG via
    autoregressive VQ-16 visual token generation with classifier-free guidance
    (CFG) (`--generate_image=true`).

## Architecture & `.litertlm` Bundle Sections

An exported `Janus-Pro-1B` `.litertlm` bundle (`dynamic_wi8_afp32` quantization)
is **~1.93 GiB** and contains **8 sections** (`1` metadata proto, `1` tokenizer,
and `6` TFLite submodels):

<!-- mdformat off(prevent table wrapping) -->

| Section / Key | Bundle Section Type | Description |
| :--- | :--- | :--- |
| `Any2AnyMetadataProto` | Metadata protobuf | `Any2AnyMetadata` with `prototypes.litert_models.janus_pro.JanusPro` packed into `any2any_model_type.custom_model` (field `100`). |
| `HF_Tokenizer_Zlib` | Tokenizer | Compressed Hugging Face `tokenizer.json` (102,400 tokens). |
| `vision_encoder` | `tf_lite_vision_encoder` | SigLIP-L-16-384 + 2-layer MLP `Aligner` (`384×384` input → `576` embeddings). |
| `text_embedder` | `tf_lite_text_embedder` | DeepSeek-LLM text token embedding table (`vocab_size = 102400`, `hidden_size = 2048` for 1B / `4096` for 7B). |
| `llm_backbone` | `tf_lite_llm_backbone` | DeepSeek-LLM autoregressive backbone (also instantiated for the unconditional CFG branch when generating images). |
| `vq_image_embedder` | `tf_lite_vq_image_embedder` | VQ-16 discrete image token embedder + `GenAligner`. |
| `gen_head` | `tf_lite_gen_head` | Autoregressive visual token prediction head (`vq_codebook_size = 16384`). |
| `vq_image_decoder` | `tf_lite_vq_image_decoder` | VQ-16 decoder mapping `576` (`24×24`) discrete visual tokens → `384×384` RGB image. |

<!-- mdformat on -->

## Directory Layout

```text
deepseek/janus_pro/
├── BUILD                        # Proto, per-stage cc_library, runner, and cc_test targets
├── README.md                    # This file
├── janus_pro.proto              # JanusPro protobuf schema (custom_model = 100)
├── janus_pro_model_config.h     # Model dimensions, defaults, and control tokens
├── io_types.h                   # Shared inter-stage I/O structs
├── stage_utils.{h,cc}           # Shared tensor validation helpers
├── fan_out_stage.{h,cc}         # Routes text and image inputs across pipeline branches
├── fan_out_stage_test.cc        # Fan-out stage unit tests
├── text_stage.{h,cc}            # Text tokenization and embedding lookup
├── text_stage_test.cc           # Text stage unit tests
├── image_stage.{h,cc}           # Image preprocessing and SigLIP vision encoding
├── image_stage_test.cc          # Image stage unit tests
├── frontend_stage.{h,cc}        # Prompt formatting and multimodal embedding assembly
├── frontend_stage_test.cc       # Frontend stage unit tests
├── llm_stage.{h,cc}             # Autoregressive understanding & CFG visual-token loop
├── llm_stage_test.cc            # LLM stage unit tests
├── image_gen_stage.{h,cc}       # VQ-16 visual token decoding to RGB pixels
├── image_gen_stage_test.cc      # Image generation stage unit tests
├── janus_pro_factory.{h,cc}     # ModelResources loading and stage wiring
├── janus_pro.cc                 # Metadata unpack, session lease, OmniRegistry registration
├── janus_pro_test_utils.{h,cc}  # Synthetic .litertlm / PNG builders for hermetic tests
├── janus_pro_test.cc            # Factory and OmniEngine/OmniRegistry integration tests
├── janus_pro_runner_test.cc     # In-process runner and posix_spawn CLI binary tests
└── export/
    ├── BUILD                    # Exporter library, CLI binary, and tests (tagged nomacos)
    ├── modeling.py              # Exportable PyTorch submodule wrappers
    ├── janus_pro.py             # Submodel conversion, checkpoint check, metadata packing
    ├── extension.py             # ModelExtension registration for export_hf
    ├── export_main.py           # Thin Fire CLI entrypoint
    ├── test_janus_pro.py        # Config, key verification, and export parity tests
    └── export_and_run_test.py   # Cross-language export_main -> janus_pro_runner E2E test
```

## Export

Export `deepseek-ai/Janus-Pro-1B` (or a local checkpoint directory) into a
single `.litertlm` bundle:

```bash
HF_HOME=/tmp/hf_cache bazel run -c opt \
  //models/deepseek/janus_pro/export:export_main -- \
  --model=deepseek-ai/Janus-Pro-1B \
  --output_dir=/tmp/janus_pro_out \
  --task=auto \
  --quantization_recipe=dynamic_wi8_afp32 \
  --cache_length=1024 \
  --bundle_litert_lm=True
```

Use `--dry_run=True` to inspect the resolved export plan without running
conversion. Optional quantization override flags include
`--vision_encoder_quantization_recipe` and `--vq_decoder_quantization_recipe`.
The exported bundle is written to `/tmp/janus_pro_out/model.litertlm`.

## Run Inference

Registered `--model_name` values: `janus-pro-1b` (canonical), `janus_pro_1b`,
`janus-pro`, `janus_pro`, `deepseek-ai/Janus-Pro-1B`,
`deepseek-ai/janus-pro-1b`, `janus-pro-7b`, `janus_pro_7b`,
`deepseek-ai/Janus-Pro-7B`, `deepseek-ai/janus-pro-7b`.

> **Tip (filename auto-detection):** Renaming the bundle to
> `janus-pro-1b.litertlm` (or any filename containing a registered alias) lets
> you omit `--model_name` when invoking `janus_pro_runner`.

### Mode A: Image Understanding / VQA

```bash
bazel run -c opt \
  //models/deepseek/janus_pro:janus_pro_runner -- \
  --model_name=janus-pro-1b \
  --model_path=/tmp/janus_pro_out/model.litertlm \
  --image_path=/tmp/photo.png \
  --text="Describe this image in detail." \
  --generate_image=false \
  --max_new_tokens=256 \
  --temperature=0.0
```

Pass `--output_path=/tmp/answer.txt` to save the generated text to a file.

### Mode B: Text-to-Image Generation (384×384)

```bash
bazel run -c opt \
  //models/deepseek/janus_pro:janus_pro_runner -- \
  --model_name=janus-pro-1b \
  --model_path=/tmp/janus_pro_out/model.litertlm \
  --text="A cute red panda wearing tiny round sunglasses in watercolor style." \
  --generate_image=true \
  --cfg_weight=5.0 \
  --image_temperature=1.0 \
  --seed=42 \
  --output_path=/tmp/janus_output.png
```

### Filename Auto-Detection

Rename the exported bundle so its filename contains a registered alias (for
example `janus-pro-1b.litertlm`) to omit `--model_name`:

```bash
mv /tmp/janus_pro_out/model.litertlm /tmp/janus_pro_out/janus-pro-1b.litertlm

bazel run -c opt \
  //models/deepseek/janus_pro:janus_pro_runner -- \
  --model_path=/tmp/janus_pro_out/janus-pro-1b.litertlm \
  --image_path=/tmp/photo.png \
  --text="Describe this image in detail." \
  --generate_image=false \
  --max_new_tokens=256 \
  --temperature=0.0
```

Common runtime flags include `--backend` (`cpu`, `gpu`, or `npu`; default
`cpu`), `--num_threads` (default `4`), and `--cache_dir` (pass `:nocache` to
disable on-disk backend caching).

## Hermetic Tests

```bash
# C++ stage, integration, and binary runner tests (synthetic weights)
bazel test //models/deepseek/janus_pro:all

# Python exporter unit tests and cross-language E2E test (Linux)
bazel test //models/deepseek/janus_pro/export:all
```

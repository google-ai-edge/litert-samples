# OmniGen2 (LiteRT Omni)

On-device C++ runtime and PyTorch-to-LiteRT exporter for
[OmniGen2/OmniGen2](https://huggingface.co/OmniGen2/OmniGen2), a
**Text-to-Image** FlowMatch diffusion pipeline pairing a
`Qwen2.5-VL-3B-Instruct` text encoder with a sharded 32-layer
`OmniGen2Transformer2DModel` DiT (using 3D `(id, h, w)` Omni-RoPE) and a
16-channel `AutoencoderKL` VAE decoder.

## Architecture & `.litertlm` Bundle Sections

Because several OmniGen2 DiT subgraphs fall outside the closed
`Text2ImageMetadata.TfLiteModelType` enum, the exporter splits the transformer
across **7 standard `TFLiteModel` sections** plus **6 named `GenericBinaryData`
sections** memory-mapped zero-copy at runtime (`ModelResources`).

An exported `.litertlm` bundle at `256×256` with `seq_len = 64` and channelwise
INT8 quantization (`dynamic_wi8_afp32`) is **~6.60 GiB** and contains **15
sections**:

<!-- mdformat off(prevent table wrapping) -->

| Section / Key | Storage Kind | Description |
| :--- | :--- | :--- |
| `Text2ImageMetadataProto` | Metadata protobuf | `Text2ImageMetadata` with `prototypes.litert_models.omnigen2.OmniGen2` packed into `image_gen_model_type.custom_model` (field `100`). |
| `HF_Tokenizer_Zlib` | Tokenizer | Compressed Hugging Face `Qwen2.5-VL-3B-Instruct` tokenizer (`tokenizer.json`). |
| `text_encoder` | `tf_lite_text_encoder` | `Qwen2.5-VL-3B-Instruct` text backbone (`hidden_size = 2048`). |
| `dit_cap_embedder` | `GenericBinaryData` | Caption linear projection shard. |
| `dit_context_refiner` | `GenericBinaryData` | 2-layer unmodulated context refiner shard. |
| `dit_x_embedder` | `GenericBinaryData` | Image patch + timestep embedder shard (`in_channels = 16`, `patch_size = 2`). |
| `dit_noise_refiner` | `GenericBinaryData` | 2-layer AdaLN-modulated noise refiner shard. |
| `dit_single_block_0` | `tf_lite_diffusion_transformer_single_block_0` | DiT main layers `0-5` (6 layers). |
| `dit_single_block_1` | `tf_lite_diffusion_transformer_single_block_1` | DiT main layers `6-11` (6 layers). |
| `dit_single_block_2` | `tf_lite_diffusion_transformer_single_block_2` | DiT main layers `12-17` (6 layers). |
| `dit_single_block_3` | `tf_lite_diffusion_transformer_single_block_3` | DiT main layers `18-23` (6 layers). |
| `dit_single_block_4` | `GenericBinaryData` | DiT main layers `24-29` (6 layers). |
| `dit_single_block_5` | `GenericBinaryData` | DiT main layers `30-31` (2 layers). |
| `dit_final` | `tf_lite_diffusion_transformer_final` | Final AdaLN norm + unpatchify velocity head. |
| `vae_decoder` | `tf_lite_image_decoder` | 16-channel `AutoencoderKL` VAE decoder (`scaling_factor = 0.3611`, `shift_factor = 0.1159`). |

<!-- mdformat on -->

> **Quantization note:** OmniGen2 uses `hidden_size = 2520` (`kv_dim = 840`),
> which is not divisible by 32, so the exporter applies channelwise INT8
> quantization (`dynamic_wi8_afp32`) instead of blockwise-32 INT4.

## Directory Layout

```text
omnigen/omnigen2/
├── BUILD                            # Proto, per-stage cc_library, runner, and cc_test targets
├── README.md                        # This file
├── omnigen2.proto                   # OmniGen2 protobuf schema (custom_model = 100)
├── omnigen2_model_config.{h,cc}     # Dimensions, RoPE helpers, FlowMatch schedule
├── omnigen2_model_config_test.cc    # Model config & scheduler unit tests
├── io_types.h                       # Shared inter-stage I/O structs
├── text_encoder_stage.{h,cc}        # ChatML framing + Qwen2.5-VL text encoder stage
├── text_encoder_stage_test.cc       # Text encoder stage unit tests
├── denoiser_stage.{h,cc}            # Sharded DiT + FlowMatch Euler CFG denoising loop
├── denoiser_stage_test.cc           # Denoiser stage unit tests
├── vae_decoder_stage.{h,cc}         # AutoencoderKL latent decode + RGB conversion
├── vae_decoder_stage_test.cc        # VAE decoder stage unit tests
├── omnigen2_factory.{h,cc}          # ModelResources loading (incl. GenericBinaryData shards)
├── omnigen2.cc                      # Metadata unpack, OmniRegistry registration
├── omnigen2_test_utils.{h,cc}       # Synthetic .litertlm builders for hermetic tests
├── omnigen2_test.cc                 # Factory and OmniEngine/OmniRegistry integration tests
├── omnigen2_runner_test.cc          # In-process runner and posix_spawn CLI binary tests
└── export/
    ├── BUILD                        # Exporter library, CLI binary, and tests (tagged nomacos)
    ├── modeling.py                  # Exportable PyTorch wrappers (Qwen2.5-VL, DiT, VAE)
    ├── omnigen2.py                  # Sharded conversion, quantization, metadata packing
    ├── extension.py                 # ModelExtension registration for export_hf
    ├── export_main.py               # Thin Fire CLI entrypoint
    ├── test_omnigen2.py             # Config, sharding, parity, and bundle peek tests
    └── export_and_run_test.py       # Cross-language export_main -> omnigen2_runner E2E test
```

## Export

Export `OmniGen2/OmniGen2` (or a local Diffusers checkpoint directory) into a
single `.litertlm` bundle:

```bash
HF_HOME=/tmp/hf_cache bazel run -c opt \
  //models/omnigen/omnigen2/export:export_main -- \
  --model=OmniGen2/OmniGen2 \
  --output_dir=/tmp/omnigen2_out \
  --task=auto \
  --t2i_output_image_size=256 \
  --seq_len=64 \
  --steps=28 \
  --quantization_recipe=dynamic_wi8_afp32 \
  --bundle_litert_lm=True
```

Use `--dry_run=True` to inspect the resolved export plan without running
conversion. Optional `omnigen2`-specific exporter flags:

*   `--t2i_output_image_size` / `--image_size` (default: `256`): Square output
    resolution in pixels (must be a multiple of `16`).
*   `--seq_len` (default: `64`): Fixed padded text token length.
*   `--steps` (default: `28`): Default FlowMatch Euler denoising step count
    recorded in bundle metadata.
*   `--text_guidance_scale` (default: `4.0`): Default CFG scale recorded in
    bundle metadata (`<= 1.0` disables CFG).

The exported bundle is written to `/tmp/omnigen2_out/model.litertlm`.

## Run Inference

Registered `--model_name` values: `omnigen2` (canonical), `omnigen-2`,
`omnigen_2`, `OmniGen2/OmniGen2`.

> **Important:** The shared `omni_runner` `--steps` flag defaults to **`4`**.
> Because `OmniGen2` is a non-distilled FlowMatch diffusion model, explicitly
> pass **`--steps=28`** (matching the upstream `OmniGen2Pipeline` default) to
> produce full-quality images.

### Explicit `--model_name`

```bash
bazel run -c opt \
  //models/omnigen/omnigen2:omnigen2_runner -- \
  --model_name=omnigen2 \
  --model_path=/tmp/omnigen2_out/model.litertlm \
  --text="A cozy cabin in a snowy pine forest at sunset, soft golden light." \
  --steps=28 \
  --seed=42 \
  --output_path=/tmp/omnigen2_output.png
```

### Filename Auto-Detection

Rename the exported bundle to `omnigen2.litertlm` (or any filename containing a
registered alias) to omit `--model_name`:

```bash
bazel run -c opt \
  //models/omnigen/omnigen2:omnigen2_runner -- \
  --model_path=/tmp/omnigen2_out/omnigen2.litertlm \
  --text="A cozy cabin in a snowy pine forest at sunset, soft golden light." \
  --steps=28 \
  --seed=42 \
  --output_path=/tmp/omnigen2_output.png
```

Common runtime flags include `--backend` (`cpu`, `gpu`, or `npu`; default
`cpu`), `--num_threads` (default `4`), `--img_size` (`0` uses the bundle's
exported `img_size`), and `--cache_dir` (pass `:nocache` to disable on-disk
backend caching).

## Hermetic Tests

```bash
# C++ stage, config, integration, and binary runner tests (synthetic weights)
bazel test //models/omnigen/omnigen2:all

# Python exporter unit tests and cross-language E2E test (Linux)
bazel test //models/omnigen/omnigen2/export:all
```

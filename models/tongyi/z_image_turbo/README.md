# Tongyi-MAI Z-Image-Turbo (LiteRT Omni)

On-device C++ runtime and PyTorch-to-LiteRT exporter for
[Tongyi-MAI/Z-Image-Turbo](https://huggingface.co/Tongyi-MAI/Z-Image-Turbo), an
8-step turbo-distilled **Text-to-Image** Single-Stream Diffusion Transformer (6B
S3-DiT) combining a `Qwen3-4B` text encoder (`prompt_dim = 2560`), a sharded
30-layer S3-DiT backbone (`dim = 3840`, 3D axial RoPE
`axes_dims = [32, 48, 48]`, `rope_theta = 256.0`, static FlowMatch
`shift = 3.0`), and a 16-channel `AutoencoderKL` VAE decoder.

## Architecture & `.litertlm` Bundle Sections

Because the 30-layer S3-DiT uses subgraphs outside the closed
`Text2ImageMetadata.TfLiteModelType` enum, the exporter packs **7 standard
`TFLiteModel` sections** plus **6 named `GenericBinaryData` sections**
memory-mapped zero-copy at runtime (`ModelResources`).

An exported `.litertlm` bundle at `256×256` with `max_seq_len = 64` and default
blockwise-32 INT4 quantization (`dynamic_wi4b32_afp32`) is **~5.48 GiB** and
contains **15 sections**:

<!-- mdformat off(prevent table wrapping) -->

| Section / Key | Storage Kind | Description |
| :--- | :--- | :--- |
| `Text2ImageMetadataProto` | Metadata protobuf | `Text2ImageMetadata` with `prototypes.litert_models.z_image_turbo.ZImageTurbo` packed into `image_gen_model_type.custom_model` (field `100`). |
| `HF_Tokenizer_Zlib` | Tokenizer | Compressed Hugging Face `Qwen3` tokenizer (`tokenizer.json`). |
| `text_encoder` | `tf_lite_text_encoder` | `Qwen3-4B` text encoder emitting penultimate hidden states (`prompt_dim = 2560`). |
| `dit_cap_embedder` | `GenericBinaryData` | Caption projection embedder shard. |
| `dit_context_refiner` | `GenericBinaryData` | Unmodulated context refiner layers shard. |
| `dit_x_embedder` | `GenericBinaryData` | Image patch + scaled timestep embedder shard (`in_channels = 16`, `patch_size = 2`). |
| `dit_noise_refiner` | `GenericBinaryData` | AdaLN-modulated noise refiner layers shard. |
| `dit_single_block_0` | `tf_lite_diffusion_transformer_single_block_0` | S3-DiT main layers `0-4` (5 layers). |
| `dit_single_block_1` | `tf_lite_diffusion_transformer_single_block_1` | S3-DiT main layers `5-9` (5 layers). |
| `dit_single_block_2` | `tf_lite_diffusion_transformer_single_block_2` | S3-DiT main layers `10-14` (5 layers). |
| `dit_single_block_3` | `tf_lite_diffusion_transformer_single_block_3` | S3-DiT main layers `15-19` (5 layers). |
| `dit_single_block_4` | `GenericBinaryData` | S3-DiT main layers `20-24` (5 layers). |
| `dit_single_block_5` | `GenericBinaryData` | S3-DiT main layers `25-29` (5 layers). |
| `dit_final` | `tf_lite_diffusion_transformer_final` | Final AdaLN norm + unpatchify layer. |
| `vae_decoder` | `tf_lite_image_decoder` | 16-channel `AutoencoderKL` VAE decoder (`scaling_factor = 0.3611`, `shift_factor = 0.1159`). |

<!-- mdformat on -->

## Directory Layout

```text
tongyi/z_image_turbo/
├── BUILD                              # Proto, per-stage cc_library, runner, and cc_test targets
├── README.md                          # This file
├── z_image_turbo.proto                # ZImageTurbo protobuf schema (custom_model = 100)
├── z_image_turbo_model_config.{h,cc}  # Dimensions, 3D axial RoPE, FlowMatch schedule
├── z_image_turbo_model_config_test.cc # Model config & scheduler unit tests
├── io_types.h                         # Shared inter-stage I/O structs
├── denoiser_stage.{h,cc}              # Sharded S3-DiT + 8-step FlowMatch Euler loop
├── denoiser_stage_test.cc             # Denoiser stage unit tests
├── vae_decoder_stage.{h,cc}           # AutoencoderKL latent decode + RGB conversion
├── vae_decoder_stage_test.cc          # VAE decoder stage unit tests
├── z_image_turbo_factory.{h,cc}       # ModelResources loading (incl. GenericBinaryData shards)
├── z_image_turbo.cc                   # Metadata unpack, OmniRegistry registration
├── z_image_turbo_test_utils.{h,cc}    # Synthetic .litertlm builders for hermetic tests
├── z_image_turbo_test.cc              # Factory and OmniEngine/OmniRegistry integration tests
├── z_image_turbo_runner_test.cc       # In-process runner and posix_spawn CLI binary tests
└── export/
    ├── BUILD                          # Exporter library, CLI binary, and tests (tagged nomacos)
    ├── modeling.py                    # Exportable PyTorch wrappers (Qwen3, S3-DiT, VAE)
    ├── z_image_turbo.py               # Sharded conversion, quantization, metadata packing
    ├── extension.py                   # ModelExtension registration for export_hf
    ├── export_main.py                 # Thin Fire CLI entrypoint
    ├── test_z_image_turbo.py          # Config, sharding, parity, and bundle peek tests
    └── export_and_run_test.py         # Cross-language export_main -> z_image_turbo_runner E2E test
```

## Export

Export `Tongyi-MAI/Z-Image-Turbo` (or a local Diffusers checkpoint directory)
into a single `.litertlm` bundle:

```bash
HF_HOME=/tmp/hf_cache bazel run -c opt \
  //models/tongyi/z_image_turbo/export:export_main -- \
  --model=Tongyi-MAI/Z-Image-Turbo \
  --output_dir=/tmp/z_image_turbo_out \
  --task=auto \
  --t2i_output_image_size=256 \
  --max_seq_len=64 \
  --bundle_litert_lm=True
```

By default, `ZImageTurboExtension` sets
`--quantization_recipe=dynamic_wi4b32_afp32` (pass
`--quantization_recipe=dynamic_wi8_afp32` or `none` to override). Use
`--dry_run=True` to inspect the resolved export plan without running conversion.
Optional `z_image_turbo`-specific exporter flags:

*   `--t2i_output_image_size` (default: `256`): Square output resolution in
    pixels (multiple of `16`).
*   `--max_seq_len` (default: `64`): Fixed padded caption token length.
*   `--default_steps` (default: `8`): Default denoising step count recorded in
    bundle metadata.

The exported bundle is written to `/tmp/z_image_turbo_out/model.litertlm`.

## Run Inference

Registered `--model_name` values: `z-image-turbo` (canonical), `z_image_turbo`,
`Tongyi-MAI/Z-Image-Turbo`, `z-image`, `z_image`.

> **Important:** The shared `omni_runner` `--steps` flag defaults to **`4`**.
> Pass **`--steps=8`** to match the 8-step turbo-distilled FlowMatch schedule.

### Explicit `--model_name`

```bash
bazel run -c opt \
  //models/tongyi/z_image_turbo:z_image_turbo_runner -- \
  --model_name=z-image-turbo \
  --model_path=/tmp/z_image_turbo_out/model.litertlm \
  --text="A red fox sitting in fresh snow under soft winter sunlight." \
  --steps=8 \
  --seed=42 \
  --output_path=/tmp/z_image_output.png
```

### Filename Auto-Detection

Rename the exported bundle to `z-image-turbo.litertlm` (or any filename
containing a registered alias) to omit `--model_name`:

```bash
bazel run -c opt \
  //models/tongyi/z_image_turbo:z_image_turbo_runner -- \
  --model_path=/tmp/z_image_turbo_out/z-image-turbo.litertlm \
  --text="A red fox sitting in fresh snow under soft winter sunlight." \
  --steps=8 \
  --seed=42 \
  --output_path=/tmp/z_image_output.png
```

Common runtime flags include `--backend` (`cpu`, `gpu`, or `npu`; default
`cpu`), `--num_threads` (default `4`), `--img_size` (`0` uses the bundle's
exported `img_size`), and `--cache_dir` (pass `:nocache` to disable on-disk
backend caching).

## Hermetic Tests

```bash
# C++ stage, config, integration, and binary runner tests (synthetic weights)
bazel test //models/tongyi/z_image_turbo:all

# Python exporter unit tests and cross-language E2E test (Linux)
bazel test //models/tongyi/z_image_turbo/export:all
```

# Bonsai Image 4B Text-to-Image - macOS (DiT on the Apple GPU)

A macOS SwiftUI demo for [Bonsai Image 4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B) (ternary DiT, int4-b32): the 3-graph LiteRT pipeline with the 2.3 GiB DiT running on the **Apple GPU** through the LiteRT Metal accelerator (fp32 forced), text encoder + VAE on CPU (XNNPACK). ~6 s per 512×512 image steady-state on an Apple-Silicon Mac (0.74 s/DiT-step ×4 + text encoder ~1.3 s + VAE ~1.2 s), after a one-time ~40 s Metal compile per launch. 32 GB+ unified memory recommended (see the memory note below).

## Build

```bash
./prep_resources.sh   # tokenizer tables + pipeline meta + LiteRT runtime pair + C headers
xcodegen generate
xcodebuild -project BonsaiMac.xcodeproj -scheme Bonsai -configuration Release build
```

Open the project in Xcode and set your signing team if the CLI build complains about signing (local ad-hoc signing is fine for running on your own Mac).

No LiteRT code is committed in this sample: `prep_resources.sh` downloads the C API headers from the LiteRT repo at the **v2.1.6 release tag** (into `third_party/LiteRT/`, gitignored) and embeds the runtime dylib pair from the ai-edge-litert **2.1.6** wheel — headers and binaries from the same release.

## Models

The app looks for models under `~/models/bonsai-image-4b-tflite` (flat, or in `gpu_work/` + `hf_upload/` subdirs; the folder is changeable in-app). Download all four from [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B):

- `dit_gpu_int4b32.tflite` — the **GPU-shaped** DiT export (rope tables precomputed, rank-4 rotary; same weights as the CPU export). The CPU-shaped `dit_int4b32.tflite` does not run on the Metal accelerator.
- `textenc_int4.tflite`, `vae_dec_fp32.tflite`, `pipeline_meta.json`.

To reproduce the GPU-shaped DiT from the checkpoint instead, use the conversion recipe in [`models/bonsai/bonsai_image_4b/converted`](../../../../models/bonsai/bonsai_image_4b/converted):

```bash
python export_dit_gpu.py                        # -> dit_gpu_fp32.tflite
python quantize_dit.py dit_gpu_fp32.tflite      # -> dit_gpu_int4b32.tflite
python fix_zero_block_scales.py dit_gpu_int4b32.tflite dit_gpu_int4b32.tflite
```

## Headless CLI mode

```bash
BONSAI_AUTORUN=1 BONSAI_PROMPT="a bonsai tree" BONSAI_SEED=7 BONSAI_STEPS=4 \
  ./Bonsai.app/Contents/MacOS/Bonsai
```

Prints stage timings and `AUTORUN_DONE <png path>`; images land in `~/Library/Application Support/Bonsai/`.

## Runtime notes

- **Runtime pair = ai-edge-litert 2.1.6** (`libLiteRt.dylib` + `libLiteRtMetalAccelerator.dylib` from the same wheel, embedded in the app bundle). The pair must stay same-generation — mixing generations makes the accelerator reject the serialized options — and 2.1.6 is the release this app was verified with.
- **fp32 precision is required for this DiT** (`gpu_options` TOML `precision = 2`): default fp16 overflows the activation range and corrupts output. The prebuilt runtime exports no `Lrt*Options` helpers — options are passed as a hand-built TOML C-string via `LiteRtCreateOpaqueOptions("gpu_options", …)`.
- **Accelerator discovery**: `kLiteRtEnvOptionTagRuntimeLibraryDir` → the app's `Contents/Frameworks`; the registry auto-scans it on macOS (no `RegisterGpuAccelerator` call, unlike iOS).
- **CPU stages via CompiledModel** (`"xnnpack"` TOML `num_threads = 6`): numerically verified against the CPU pipeline (text encoder cos = 1.0, VAE PNG byte-identical).
- The ~40 s Metal compile happens on every launch (the compiler-cache env option covers NPU JIT only); the app keeps all three graphs resident after it.
- **Memory**: fp32-forced GPU weights are resident on unified memory — steady state ≈ 22 GB RSS (int4 DiT dequantized to fp32 on the GPU, ~19 GB transient during compile). Comfortable on a 32 GB+ Mac; expect swap pressure below that.

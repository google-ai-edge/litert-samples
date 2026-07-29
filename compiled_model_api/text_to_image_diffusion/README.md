# Text-to-image diffusion (Bonsai Image 4B)

A full diffusion text-to-image pipeline running on LiteRT: three fixed-shape `.tflite` graphs (text encoder, diffusion transformer, VAE decoder) driven by a ~150-line host loop with no torch and no diffusers at inference time. The model is [PrismML Bonsai Image 4B](https://huggingface.co/prism-ml/bonsai-image-ternary-4B) — a ternary-weight DiT on the FLUX.2-klein-4B architecture — whose ternary weights land losslessly as {-7, 0, +7} in the int4 block-32 container.

Converted models: [litert-community/Bonsai-Image-ternary-4B](https://huggingface.co/litert-community/Bonsai-Image-ternary-4B)

| graph | recipe | size |
|---|---|---|
| DiT (3.88 B) | int4 block-32 | 2.11 GiB |
| text encoder (Qwen3-4B, top 9 layers pruned) | int4 block-128 | 1.68 GiB |
| VAE decoder | fp32 | 0.19 GiB |

The host does tokenization, the FlowMatch-Euler sampling loop, and the latent unpatchify; everything heavy is inside the three graphs. Output is fixed at 512×512, 4 sampling steps by default (step-distilled model).

## Run it

```bash
cd python
pip install ai-edge-litert numpy pillow transformers huggingface_hub
hf download litert-community/Bonsai-Image-ternary-4B --local-dir bonsai
python generate.py --model-dir bonsai --prompt "a red fox sitting in fresh snow at sunrise" --out fox.png
```

## Measured performance (512×512, 4 steps)

| host | DiT step | whole image | peak memory |
|---|---|---|---|
| MacBook (Apple silicon, CPU, 8 threads) | 3.9 s | ~19 s | — |
| iPhone 17 Pro (CPU, XNNPACK, 6 threads) | 13.0 s | ~64 s | 2.9 GiB |

Device output is bit-exact against the desktop run (identical DiT step outputs; 51.2 dB PSNR on the final PNG).

**Integration note:** when driving these graphs through the C API, attach the XNNPACK delegate explicitly and set its `num_threads`. Without the delegate the runtime falls back to reference kernels, which are orders of magnitude slower on blockwise-int4 weights and numerically different.

## Conversion

`conversion/` has the complete recipe: DiT export (with the float64-RoPE → float32 fix that `tfl.pow` legalization requires), int4 block-32 quantization plus the zero-scale patch XNNPACK needs on sparse ternary blocks, the text-encoder export as a prompt embedder (only hidden layers 9/18/27 are read, so the top 9 layers and the LM head prune away for free), and the VAE decoder export. See `conversion/README.md`.

## License

Model weights: Apache-2.0 (upstream PrismML release). Sample code: Apache-2.0.

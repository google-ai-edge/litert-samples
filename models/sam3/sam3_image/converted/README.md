# SAM 3 Image Conversion

Model recipes, instructions, scripts, and utilities for converting SAM 3 text-prompted detection and instance segmentation to LiteRT.

These scripts convert the image side of [facebookresearch/sam3](https://github.com/facebookresearch/sam3) (SAM License; ~830M params — ViT-L/14 trunk, CLIP-L text tower, DETR head) into **three `.tflite` graphs that run on the LiteRT CompiledModel GPU**, and numerically verify every step against the PyTorch reference. The published artifacts live at [mlboydaisuke/SAM3-LiteRT](https://huggingface.co/mlboydaisuke/SAM3-LiteRT).

SAM 3 takes a text phrase instead of a fixed label set: type "wheel" or "paper bag" and it returns every matching instance with a box, a score and a mask. The split into three graphs follows the data flow — image features depend only on the image, the prompt embedding depends only on the phrase, and only the head needs both. Caching the vision features per image is what makes a second prompt on the same photo an order of magnitude cheaper than the first.

## Pipeline

```
image [1,3,1008,1008]  (bilinear to 1008², (x/255-0.5)/0.5, NCHW RGB)
  ──> [GPU Graph: vision] ──> fpn288 [1,256,288,288] | fpn144 | fpn72   (27.9M floats)

prompt "wheel"
  ──> host: CLIP BPE (ctx 32, BOS 49406, EOT 49407, zero pad)
      ──> host: fp16 embedding table lookup ──> emb [1,32,1024]
  ──> [CPU Graph: text] ──> text_mem [1,32*256],  pad = (id == 0)

  ──> [GPU Graph: head] (fpn×3 | text_mem | pad)
  ──> logits [200] | boxes [200,4] cxcywh | presence [1] | masks [200,288,288]
  ──> host: score = sigmoid(logit)·sigmoid(presence); keep > 0.5;
            mask = sigmoid(mask logit) > 0.5, resized to the frame
```

- **Input**: one square 1008×1008 image and one text phrase. The token embedding is a host lookup (`sam3_token_embed.bin`, fp16 49408×1024) because `EMBEDDING_LOOKUP` has no GPU lowering; the text graph takes embeddings, not ids.
- **Output**: 200 queries. `boxes` are cxcywh normalized to the resized square; `masks` are 288×288 raw logits per query, upsampled bilinearly to the frame. The presence token gates every score, so a phrase that matches nothing returns an empty set rather than low-ranked noise.

## ⭐ The conditioning encoder is what breaks first in fp16

The vision trunk tolerates fp16 — it is the **CLIP-L text tower** that does not, and its failure is invisible from any aggregate metric. Its residual stream reaches **|x| ≈ 1.2e3**, and running it in fp16 on the GPU corrupts the prompt embedding for *some phrases only*. The graph compiles, reports fully accelerated, and returns a confident empty result:

| prompt | text fp32 | text fp16 on GPU |
|---|---|---|
| "wheel" | 4 detections, p = 0.94/0.94/0.88/0.90 | same 4 detections |
| "window" | 6 detections | **0 detections** |
| "paper bag" | 4 detections, p ≈ 0.83 | **0 detections** |

A single-prompt smoke test passes. **Check a conditioning encoder on many prompts, not one.** The fix is to run this 607 MB graph on the CPU (0.5 s on a Pixel 8a, fully delegated to XNNPACK) or, on Apple platforms, on the GPU with `enforce_f32` (about 10 ms on an M4 Max). Both are exact; the vision and head graphs stay in fp16.

The same class of failure appears in the head, with a different mechanism. A rank-3 `[1, N, C]` tensor that **fans out to several consumers** is mis-executed by the GPU delegates: the decoder output feeds the score, box and mask heads, and on device **all 200 logits came back identical (std 0.0)** while the boxes and masks from the same tensor were correct. Desktop CPU and Apple Metal did not reproduce it. Keeping the decoder batch-first and rank-4 end to end fixes it exactly — see `decoder_forward_bf` in `sam3_recipe.py`.

## Re-authoring → GPU-clean

Converted with **litert-torch** (NCHW preserved). All three graphs pass the static op-check in `build_sam3_image.py` (no banned ops, no >4-D tensors) and reproduce the stock PyTorch modules at corr **1.00000**:

| construct | rewrite |
|---|---|
| ViT attention with 5-D/6-D/8-D tensors | ≤4-D head split, window partition and RoPE. A raw export of the trunk reaches only **corr 0.607**: the converter mis-lowers these, so this is a correctness fix, not just a delegate constraint |
| interleaved real RoPE (2p, 2p+1) | permute the q and k rows of the fused qkv weight so the rotation becomes a contiguous half-split; exact, because q·k is invariant under a shared channel permutation |
| tiled 24²→72² absolute position | baked into a constant buffer |
| window partition | reshape → transpose → reshape → **transpose**. The cheaper order-swap trick is *not* valid here: RoPE is position dependent inside the window, so the extra transpose is required to restore the layout |
| `nn.LayerNorm` (fp16 accumulator overflow) | **SafeLayerNorm**: scale before squaring, scale eps to match. Exact; needed because the ViT residual reaches \|x\| ≈ 300 and the text residual \|x\| ≈ 1.2e3 |
| `nn.MultiheadAttention` / SDPA | manual rank-4 head-split matmuls. Patching only sam3's own attention class silently leaves torch's instances (decoder self-attention, text cross-attention) on SDPA |
| softmax after an additive mask | masked form `m = max(s·keep + (1-keep)·NEG); e = exp((s-m)·keep)·keep`. A SOFTMAX fed directly by an elementwise op with a broadcast constant is mis-executed on Metal; the clamp, min and relu variants of the same math are too |
| rank-3 `[1,N,C]` activations that fan out | batch-first, rank-4 decoder and rank-4 text encoder (see above) |
| `ConvTranspose2d` (TRANSPOSE_CONV) | zero-stuff + `Conv2d` |
| `clamp(x, 0, 1)` (RELU_0_TO_1) | `relu(x) - relu(x-1)` |
| sine embedding with strided slices | `dim_t` as a constant, interleave as a stack of the 64 distinct frequencies (no GATHER_ND, no POW) |
| `nn.GroupNorm` | 4-D manual with hierarchical means (a single reduce over 2.6M elements overflows the fp16 accumulator) |
| all-constant op chains | tied to a runtime zero derived from the input; the delegates refuse ops whose inputs are all constant and the converter does not fold them |
| global attention, 5184×5184 scores | exact query chunking (`--chunks`, default 9): the score tensor would otherwise need 860 MB in fp16 per global block |

fp16 quantization is applied to the **matmul-class weights only** (`FULLY_CONNECTED`, `CONV_2D`, `DEPTHWISE_CONV_2D`). Quantizing every constant leaves a `DEQUANTIZE` feeding an elementwise op, or one shared by several consumers, which the Metal delegate refuses at compile time.

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer
pip install torch torchvision pillow numpy
pip install -e /path/to/sam3        # facebookresearch/sam3, SAM License
```

Verified with `litert-torch==0.9.3`, `ai-edge-litert==2.1.6`, `torch==2.12.1`, `torchvision==0.27.1`. The checkpoint (`sam3.1_multiplex.pt`) is gated on Hugging Face; it carries the detector under a `detector.` prefix and the video tracker under `tracker.`, and only the former is used here. `triton` is not required — `sam3_recipe.py` stubs the CUDA-only module that imports it.

**Runtime requirement: LiteRT ≥ 2.2.0.** On 2.1.5 the head graph is mis-executed on the Android GPU (the constant-logits symptom above persists even after the rank-4 fix), and the text graph only partially delegates to XNNPACK (1337/1627 nodes, 1.9 s versus 1627/1627 and 0.5 s on 2.2.0).

## Run

```bash
python build_sam3_image.py --ckpt sam3.1_multiplex.pt --image truck.jpg \
    --prompt wheel
python verify_sam3_image.py --image truck.jpg --prompt wheel \
    --ckpt sam3.1_multiplex.pt
```

`build_sam3_image.py` runs the stock modules first, applies the recipe, asserts the re-authored torch modules still match, then converts, op-checks, fp16-quantizes and re-checks each graph through the CompiledModel CPU path. It emits `sam3_vision.tflite` (930 MB), `sam3_text.tflite` (607 MB), `sam3_head.tflite` (68 MB), `sam3_token_embed.bin` (101 MB) and `sam3_tokenizer/`. `verify_sam3_image.py` then runs a real photo end-to-end through the CompiledModel API and, with `--ckpt`, reports the kept set, box IoU, mask IoU and score deltas against the official PyTorch model.

## Verification

Desktop, per graph, fp32 tflite versus the stock PyTorch modules: corr **1.00000** (vision max\|diff\| 1.9e-4, head max\|diff\| 1.8e-3).

Detection-level gate on real images, GPU fp16 vision features versus fp32, decoded through the same head:

| image / prompt | PyTorch fp32 | GPU fp16 vision |
|---|---|---|
| truck / "wheel" | 4 kept, p = 0.94/0.94/0.88/0.90 | same 4; box IoU ≥ 0.993, **mask IoU ≥ 0.974** (mean 0.991), \|Δp\| ≤ 0.001 |
| groceries / "paper bag" | 4 kept, p ≈ 0.83 | same 4; box IoU ≥ 0.9987, mask IoU ≥ 0.9988, \|Δp\| ≤ 0.0004 |
| groceries / "bottle" | 0 kept (presence 0.06) | 0 kept |

Feature correlation alone is misleading here: the fp16 vision features sit at corr 0.994 against fp32, which sounds poor, yet the decoded detections are identical to three decimal places. The deep-ViT residual is simply large (\|x\| ≈ 300, fp16 spacing 0.25), and the head is insensitive to that. Judge this model at the detection level.

## Device results

| device | runtime | vision | text | head | first prompt | re-prompt, same photo |
|---|---|---|---|---|---|---|
| Pixel 8a | LiteRT 2.2.0, ML Drift | 9.2 s (GPU, 3104/3104 ops, 1 partition) | 0.5 s (CPU) | 1.4 s (GPU, 2791/2791) | ≈ 11 s | **1.9 s** |
| iPhone 17 Pro | Metal | 3.9 s (GPU) | 0.2 s (CPU) | 1.6 s (GPU) | ≈ 5.7 s | **1.3 s** |
| M4 Max | Metal | 0.56 s (GPU) | 12 ms (`enforce_f32`) | 0.18 s (`enforce_f32`) | ≈ 0.75 s | ≈ 0.2 s |

First launch on Android additionally pays about 50 s of GPU shader compilation. The re-prompt column is the vision features being cached per image — the reason the pipeline is split this way.

## Files

| File | What |
|---|---|
| `sam3_recipe.py` | the re-authoring recipe (ViT ≤4-D + baked RoPE, SafeLayerNorm, 4-D attention, batch-first rank-4 decoder, delegate-safe masked softmax, ZeroStuffConvT, GroupNorm4d) plus checkpoint loading, the export shims and the three flat graph wrappers. |
| `build_sam3_image.py` | stock-versus-re-authored parity gate, litert-torch conversion, static op-check, fp16 quantization, CompiledModel CPU parity, host artifacts. |
| `verify_sam3_image.py` | real-image end-to-end run through the CompiledModel API with the detection-level comparison against PyTorch. |

The video tracker that shares this trunk (memory attention, mask memory encoder, multiplex decoder, and the host state machine that drives them) is a separate stage and is not part of this recipe.

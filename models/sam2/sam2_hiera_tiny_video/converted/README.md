# SAM 2.1 Hiera-Tiny Video Conversion

These scripts convert the video-tracking path of [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) (Apache-2.0) into four fixed-shape LiteRT graphs that run on the CompiledModel GPU delegate (ML Drift on Android, Metal on iOS), and numerically verify the assembled tracking loop against the `transformers` PyTorch reference.

SAM 2 tracks an object across a video with a memory bank: the image encoder runs once per frame, memory attention conditions the current frame on features remembered from past frames, the mask decoder produces the mask, and the memory encoder writes the new frame back into the bank. Only the tensor math is exported; the rolling bank and the per-frame orchestration stay on the host (`verify_video.py` carries the numpy reference of that loop, which a Kotlin or Swift app mirrors). The still-image path (encoder + prompt mask decoder) already ships as the interactive-segmentation sample; this recipe adds the parts unique to video — memory attention, the memory encoder, object pointers, and the prompt-conditioned video mask decoder.

## Environments

One Python environment (the "convert env") runs everything: `litert-torch`, `torch`, `transformers>=5.13` (which has `Sam2VideoModel` natively), `ai-edge-litert`, `ai-edge-quantizer`, `numpy`, and `pillow`. The verifier loads the exported graphs through the `ai_edge_litert` CompiledModel Python API and the reference through `transformers`, so no second environment is needed.

## Pipeline

Run from this directory. `export_video.py` writes the graphs and the host constants to `$SAM2_OUT` (default `out/`); `verify_video.py` reads them back and checks the whole tracking loop against PyTorch.

| # | Script | What it does |
|---|---|---|
| 1 | `export_video.py` | Loads `Sam2VideoModel`, installs the GPU-clean Hiera rewrites from `hiera_gpu_clean.py`, and exports the four graphs (`encode`, `memcond7` and `memcond2`, `decode`, `memorize`) as fp16 `.tflite` plus the host constants (`sam2v_prompt.bin`, `sam2v_track_sparse.bin`, `sam2v_mtpe.bin`, `sam2v_no_obj_ptr.bin`, `sam2v_tpos_proj.bin`). Each graph is op-checked from its flatbuffer — every one reports `GPU_BAD=NONE >4D=0`. |
| 2 | `verify_video.py` | Runs a synthetic clip (a drifting disk, one click on frame 0) through both the HF streaming reference and the exported graphs driven by the host loop, and reports the per-frame mask IoU / correlation. `--nmm` selects the bank sizes (default `7,2`) and `--frames` the clip length. |

The two graph shapes worth restating (both size-independent flat float32 I/O):

| Graph | Size (fp16) | In → Out |
|---|---|---|
| `encode` | 80 MB | image `[1,3,1024,1024]` → `pix_raw \| hi0 \| hi1` |
| `memcond{7,2}` | 26 MB | `pix_raw \| memory bank \| temporal pos \| pointers \| key mask` → `pix_feat` |
| `decode` | 18 MB | `pix_feat \| hi0 \| hi1 \| sparse \| nomem` → `masks \| iou \| object pointers \| object score` |
| `memorize` | 3 MB | `pix_raw \| mask_for_mem \| occ` → spatial memory `[4096, 64]` |

## GPU-clean rewrites

The still-image rewrites are reused from `hiera_gpu_clean.py`: the windowed positional embedding is baked to a constant (removing the bicubic `GATHER_ND` and the tiled `BROADCAST_TO`), window partition/unpartition and the multi-scale attention are re-expressed with ≤ 4-D tensors, and `ConvTranspose2d` becomes a zero-stuffed conv. The one rewrite unique to the video path is the **memory attention**: SAM 2's `Sam2VideoRoPEAttention` runs with the batch dimension collapsed (`q/k/v` shaped `[heads, N, d]`, rank 3), and the ML Drift delegate silently mis-computes that form. Re-authoring it batch-first (`[1, heads, N, d]`, rank 4) is numerically identical on the host and correct on the GPU. The interleaved 2-D RoPE is baked into a half-split (`rotate_half`) form by permuting the q/k projection rows, and the memory-encoder sine position encoding is baked constant.

## Verification results

- Assembled loop vs the HF PyTorch reference (10-frame clip, both the 7-slot and 2-slot banks): min mask-IoU **0.9999**, correlation **1.0**. The short-clip check bundled here (`--frames 5`) is exact: min IoU 1.0000, min corr 1.00000, object-score match to 0.01.
- Memory attention: the rank-4 export is exact under fp32 GPU compute (correlation **1.0**), confirming the rank-3 miscompute is avoided rather than merely masked. At fp16 the 7-slot bank sits at correlation 0.976 on device (pure accumulation over the 7 × 4096 memory keys), and that error does not reach the mask — the chained tracked-frame mask IoU is **0.9986**.
- Every graph is fully GPU-resident with no CPU fallback. Pixel 8a (Mali, ML Drift): `encode` 828/828, `memcond` 480/480, `decode` 462/462, `memorize` 145/145 nodes, one partition each. iPhone 17 Pro (Metal): all `fullyGPU`.

## Device performance (per tracked frame: encode + memcond + decode + memorize)

| Host | 2-slot bank | 7-slot bank |
|---|---|---|
| iPhone 17 Pro (A19 Pro, Metal) | ~471 ms | ~751 ms |
| Pixel 8a (Mali, ML Drift) | ~1.0 s | ~1.5 s |

The memory attention dominates and scales with the bank size (the spatial cross-attention runs over N × 4096 memory tokens); the encoder is the second cost. The still-image path is much cheaper because it skips memory attention entirely.

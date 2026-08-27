#!/usr/bin/env python3
"""PyTorch parity check for the Tensor API SAM2 image path.

Loads the raw dumps written by `sam2_main --dump_dir`, runs the official
`facebook/sam2.1-hiera-tiny` (transformers, fp32) on the SAME input at 512
(prompt-encoder/backbone size attributes adjusted to 512 exactly the way the
mlx-swift port was verified), and reports correlations.

The bar (set by the MLX port): masks corr >= 0.9999, iou_scores within
0.002, identical best-mask selection.

Usage:
    python sam2_torch_ref.py --dump_dir /path/to/dump [--size 512]
"""
import argparse
import sys
import types

import numpy as np

# macOS: stub scipy's broken propack entry point if present (same guard as
# tools/make_512_weights.py).
_svdp = types.ModuleType("scipy.sparse.linalg._svdp")
_svdp._svdp = lambda *a, **k: None
sys.modules["scipy.sparse.linalg._svdp"] = _svdp


def load(dump_dir, name, shape):
    data = np.fromfile(f"{dump_dir}/{name}.f32", dtype=np.float32)
    return data.reshape(shape)


def corr(a, b):
    a = a.flatten().astype(np.float64)
    b = b.flatten().astype(np.float64)
    return float(np.corrcoef(a, b)[0, 1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump_dir", required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    size = args.size
    grid = size // 16   # 32
    mg = size // 4      # 128

    pixels = load(args.dump_dir, "pixels", (1, size, size, 3))
    ours_ie = load(args.dump_dir, "image_embeddings", (1, grid, grid, 256))
    ours_s1 = load(args.dump_dir, "feat_s1", (1, grid * 2, grid * 2, 64))
    ours_s0 = load(args.dump_dir, "feat_s0", (1, mg, mg, 32))
    ours_masks = load(args.dump_dir, "masks", (1, 3, mg, mg))
    ours_iou = load(args.dump_dir, "iou_scores", (1, 3))
    ours_obj = load(args.dump_dir, "object_score", (1, 1))

    import torch
    from transformers import Sam2Model

    model = Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny").eval()
    # Adjust the size-derived attributes to the 512 run (the vision encoder
    # and its positional embedding are size-dynamic already).
    model.prompt_encoder.input_image_size = size
    model.prompt_encoder.image_embedding_size = (grid, grid)
    model.backbone_feature_sizes = [[mg, mg], [grid * 2, grid * 2], [grid, grid]]

    chw = torch.from_numpy(pixels.transpose(0, 3, 1, 2)).contiguous()

    with torch.no_grad():
        feats = model.get_image_embeddings(chw)  # [s0, s1, ie] (no_mem folded)
    ref_s0, ref_s1, ref_ie = [f.numpy() for f in feats]

    def nhwc(x):
        return x.transpose(0, 2, 3, 1)

    print(f"encoder image_embeddings corr = {corr(ours_ie, nhwc(ref_ie)):.6f}")
    print(f"encoder feat_s1          corr = {corr(ours_s1, nhwc(ref_s1)):.6f}")
    print(f"encoder feat_s0          corr = {corr(ours_s0, nhwc(ref_s0)):.6f}")

    center = float(size // 2)
    with torch.no_grad():
        out = model(
            pixel_values=chw,
            input_points=torch.tensor([[[[center, center]]]]),
            input_labels=torch.tensor([[[1]]]),
            multimask_output=True,
        )
    ref_masks = out.pred_masks.numpy().reshape(1, -1, mg, mg)[:, -3:]
    ref_iou = out.iou_scores.numpy().reshape(1, -1)[:, -3:]

    print(f"masks corr = {corr(ours_masks, ref_masks):.6f}")
    for i in range(3):
        m_corr = corr(ours_masks[0, i], ref_masks[0, i])
        a = ours_masks[0, i] > 0
        b = ref_masks[0, i] > 0
        iou_bin = np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)
        print(f"  mask[{i}]: corr={m_corr:.6f} binIoU={iou_bin:.5f} "
              f"fg_ours={int(a.sum())} fg_ref={int(b.sum())}")
    print(f"iou_scores ours={np.round(ours_iou[0], 4).tolist()} "
          f"ref={np.round(ref_iou[0], 4).tolist()} "
          f"max|diff|={float(np.abs(ours_iou - ref_iou).max()):.5f}")
    print(f"best mask ours={int(ours_iou[0].argmax())} "
          f"ref={int(ref_iou[0].argmax())}")
    if hasattr(out, "object_score_logits") and out.object_score_logits is not None:
        ref_obj = float(out.object_score_logits.reshape(-1)[0])
        print(f"object_score ours={float(ours_obj[0, 0]):.4f} ref={ref_obj:.4f}")

    ok = (corr(ours_masks, ref_masks) >= 0.9999
          and float(np.abs(ours_iou - ref_iou).max()) <= 0.002
          and int(ours_iou[0].argmax()) == int(ref_iou[0].argmax()))
    print("PARITY:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

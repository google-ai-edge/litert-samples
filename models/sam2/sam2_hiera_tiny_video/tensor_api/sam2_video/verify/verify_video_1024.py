#!/usr/bin/env python3
"""Parity check for the Tensor API SAM2 video path vs the HF streaming
reference, at the model-native 1024x1024.

Modes (run in this order):

  python verify_video_1024.py clip                # write frames.f32 (T,1024,1024,3)
  python verify_video_1024.py ref                 # HF streaming run -> ref_nmm{7,2}.npz
  python verify_video_1024.py compare --dump_dir <dir> --nmm 7
                                                  # compare sam2v_main --dump_dir output

Synthetic clip: a white disk moving diagonally on black, T frames, one
positive click on the disk in frame 0 — the same fixture as the converted-
path reference (litert-samples #283 lineage), so results are comparable.

The HF reference capture matches the graph boundary exactly:
  mask     low-res (256,256) logits after best-mask/no-object handling
  obj      object score logit
  ptr      the stored object pointer (256)
  mem      memory encoder output, token-major (4096,64), fp32 pre-bfloat16,
           with the occlusion embedding added when obj <= 0 (HF adds it
           outside the encoder)
  pix_feat memory-attention output, token-major flat (4096*256), tracked
           frames only
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.environ.get("SAM2V_OUT", HERE + "/out"))
CKPT = os.environ.get("SAM2_CKPT", "facebook/sam2.1-hiera-tiny")
T = int(os.environ.get("SAM2_T", "10"))
SIZE = 1024
HW, MEMCH, HD = 4096, 64, 256
CLICK = (400.0, 512.0)
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])


def frame(t):
    """Normalized NCHW float32 frame t (disk moves (+14,+6) px/frame in 512-space)."""
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (512, 512), "black")
    cx, cy, r = 200 + 14 * t, 256 + 6 * t, 100
    ImageDraw.Draw(img).ellipse([cx - r, cy - r, cx + r, cy + r], fill="white")
    arr = np.asarray(img.resize((SIZE, SIZE), Image.BILINEAR)).astype(np.float32) / 255.0
    return ((arr - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)


def write_clip():
    frames = np.stack([frame(t)[0].transpose(1, 2, 0) for t in range(T)])  # NHWC
    path = f"{OUT}/frames.f32"
    frames.astype(np.float32).tofile(path)
    print(f"wrote {path}: {frames.shape} NHWC float32")


def reference(nmm):
    import torch
    from transformers import Sam2VideoModel, Sam2VideoInferenceSession
    model = Sam2VideoModel.from_pretrained(CKPT).eval()
    model.num_maskmem = nmm
    cap = {}
    model.memory_attention.register_forward_hook(
        lambda m, a, o: cap.__setitem__("pix_feat", o.detach().clone()))
    model.memory_encoder.register_forward_hook(
        lambda m, a, o: cap.__setitem__("mem", o[0].detach().clone()))
    sess = Sam2VideoInferenceSession(video_height=SIZE, video_width=SIZE,
                                     dtype=torch.float32)
    oi = sess.obj_id_to_idx(1)
    sess.add_point_inputs(oi, 0, {
        "point_coords": torch.tensor([[[[CLICK[0], CLICK[1]]]]]),
        "point_labels": torch.tensor([[[1]]])})
    sess.obj_with_new_inputs = [1]
    masks, objs, ptrs, mems, pfs = [], [], [], [], []
    with torch.no_grad():
        for t in range(T):
            cap.clear()
            out = model(inference_session=sess, frame=torch.from_numpy(frame(t)))
            masks.append(out.pred_masks.numpy().reshape(256, 256))
            objs.append(float(out.object_score_logits.reshape(-1)[0]))
            store = "cond_frame_outputs" if t == 0 else "non_cond_frame_outputs"
            ptrs.append(sess.output_dict_per_obj[oi][store][t][
                "object_pointer"].numpy().reshape(256))
            mem_t = cap["mem"].numpy().reshape(MEMCH, HW).T.copy()
            if objs[-1] <= 0:
                mem_t = mem_t + model.occlusion_spatial_embedding_parameter.detach(
                ).numpy().reshape(1, 64)
            mems.append(mem_t)
            pfs.append(np.zeros((HW * HD,), np.float32) if t == 0 else
                       cap["pix_feat"].numpy().reshape(HW, HD).T.reshape(-1).copy())
            print(f"ref nmm={nmm} frame {t}: fg={(masks[-1] > 0).sum()} "
                  f"obj={objs[-1]:.3f}", flush=True)
    np.savez_compressed(f"{OUT}/ref_nmm{nmm}.npz", mask=np.stack(masks),
                        obj=np.array(objs), ptr=np.stack(ptrs),
                        mem=np.stack(mems), pix_feat=np.stack(pfs))


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 1.0


def corr(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def compare(dump_dir, nmm, tag):
    ref = np.load(f"{OUT}/ref_nmm{nmm}.npz")
    n = ref["mask"].shape[0]
    worst = dict(iou=1.0, corr=1.0, dmask=0.0, dpf=0.0, dmem=0.0, dptr=0.0)
    lines = []
    # HF pix_feat capture is seq-first (HW,HD) -> stored as (HD,HW) flat; the
    # graph emits token-major (HW,HD). Compare in one common layout.
    for t in range(n):
        p = f"{dump_dir}/f{t:02d}"
        m = np.fromfile(f"{p}_mask.f32", np.float32).reshape(256, 256)
        obj = float(np.fromfile(f"{p}_obj.f32", np.float32)[0])
        ptr = np.fromfile(f"{p}_ptr.f32", np.float32)
        mem = np.fromfile(f"{p}_mem.f32", np.float32).reshape(HW, MEMCH)
        rm = ref["mask"][t]
        r = dict(iou=iou(rm > 0, m > 0), corr=corr(rm, m),
                 dmask=float(np.abs(rm - m).max()),
                 dobj=abs(float(ref["obj"][t]) - obj),
                 dptr=float(np.abs(ref["ptr"][t] - ptr).max()),
                 dmem=float(np.abs(ref["mem"][t] - mem).max()),
                 cmem=corr(ref["mem"][t], mem))
        if t == 0:
            r["dpf"], r["cpf"] = 0.0, 1.0
        else:
            pf = np.fromfile(f"{p}_pixfeat.f32", np.float32)  # (HW,HD) token-major
            rpf = ref["pix_feat"][t].reshape(HD, HW).T.ravel()
            r["dpf"] = float(np.abs(rpf - pf).max())
            r["cpf"] = corr(rpf, pf)
        lines.append(
            f"  f{t:02d} fg={int((m > 0).sum()):6d}/{int((rm > 0).sum()):6d} "
            f"IoU={r['iou']:.4f} corr={r['corr']:.5f} "
            f"max|dmask|={r['dmask']:.3f} |dobj|={r['dobj']:.3f} "
            f"pix_feat corr={r['cpf']:.5f} max|d|={r['dpf']:.4f} "
            f"mem corr={r['cmem']:.5f} max|d|={r['dmem']:.4f} "
            f"|dptr|={r['dptr']:.4f}")
        worst["iou"] = min(worst["iou"], r["iou"])
        worst["corr"] = min(worst["corr"], r["corr"])
        for k in ("dmask", "dpf", "dmem", "dptr"):
            worst[k] = max(worst[k], r[k])
    head = (f"[{tag}] nmm={nmm} T={n}: min IoU={worst['iou']:.4f} "
            f"min corr={worst['corr']:.5f} max|dmask|={worst['dmask']:.3f} "
            f"max|dpix_feat|={worst['dpf']:.4f} max|dmem|={worst['dmem']:.4f} "
            f"max|dptr|={worst['dptr']:.4f}")
    print(head)
    print("\n".join(lines))
    with open(f"{OUT}/parity_{tag}_nmm{nmm}.log", "w") as f:
        f.write(head + "\n" + "\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["clip", "ref", "compare"])
    ap.add_argument("--nmm", type=int, default=7)
    ap.add_argument("--dump_dir", default="")
    ap.add_argument("--tag", default="tensorapi")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.mode == "clip":
        write_clip()
    elif a.mode == "ref":
        assert "convert_sam2_video" not in sys.modules
        reference(a.nmm)
    else:
        compare(a.dump_dir, a.nmm, a.tag)


if __name__ == "__main__":
    main()

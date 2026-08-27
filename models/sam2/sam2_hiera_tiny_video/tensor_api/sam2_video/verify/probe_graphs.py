#!/usr/bin/env python3
"""Per-graph isolation probe: run each Tensor API signature on inputs captured
from the HF modules themselves, so each graph is judged independently of the
chain. Requires the serialized model from sam2v_main (default
/tmp/sam2_video.tflite, built with REAL weights).

  python probe_graphs.py [--tflite /tmp/sam2_video.tflite] [--frame 3]

Prints per-graph corr / max|d|:
  encode    my encoder vs HF raw fpn features (pix_raw / hi0 / hi1)
  decode    my decoder on HF's OWN pix_feat/hi-res/sparse vs HF decoder output
  memorize  my memory encoder on HF's OWN pix_feat + mask_for_mem vs HF output
  memcond   my memory attention on HF's OWN bank vs HF memory-attention output
"""
import argparse
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.environ.get("SAM2V_OUT", HERE + "/out"))
CKPT = os.environ.get("SAM2_CKPT", "facebook/sam2.1-hiera-tiny")
SIZE, HW, MEMCH, HD = 1024, 4096, 64, 256
CLICK = (400.0, 512.0)


def corr(a, b):
    a, b = np.ravel(a).astype(np.float64), np.ravel(b).astype(np.float64)
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def report(name, mine, ref):
    mine, ref = np.asarray(mine, np.float32), np.asarray(ref, np.float32)
    d = np.abs(mine - ref)
    print(f"  {name:24s} corr={corr(mine, ref):.6f} max|d|={d.max():.5f} "
          f"mean|d|={d.mean():.6f} ref_absmax={np.abs(ref).max():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default="/tmp/sam2_video.tflite")
    ap.add_argument("--frame", type=int, default=3)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, HERE)
    from verify_video_1024 import frame
    from transformers import Sam2VideoModel, Sam2VideoInferenceSession

    model = Sam2VideoModel.from_pretrained(CKPT).eval()
    cap = {}

    def grab(name):
        def hook(mod, args, kwargs, out):
            cap[name] = (args, kwargs, out)
        return hook

    model.memory_attention.register_forward_hook(grab("ma"), with_kwargs=True)
    model.memory_encoder.register_forward_hook(grab("me"), with_kwargs=True)
    model.mask_decoder.register_forward_hook(grab("dec"), with_kwargs=True)

    sess = Sam2VideoInferenceSession(video_height=SIZE, video_width=SIZE,
                                     dtype=torch.float32)
    oi = sess.obj_id_to_idx(1)
    sess.add_point_inputs(oi, 0, {
        "point_coords": torch.tensor([[[[CLICK[0], CLICK[1]]]]]),
        "point_labels": torch.tensor([[[1]]])})
    sess.obj_with_new_inputs = [1]
    with torch.no_grad():
        for t in range(a.frame + 1):
            model(inference_session=sess, frame=torch.from_numpy(frame(t)))
            if t == a.frame - 1 or (a.frame == 0 and t == 0):
                pass
        # captures now hold frame a.frame's module calls

    # Raw vision features for the probe frame.
    with torch.no_grad():
        img = model.get_image_features(
            torch.from_numpy(frame(a.frame)), return_dict=True)
        fpn = img.fpn_hidden_states  # raw, no no_mem
    hf_pix_raw = fpn[2][:, 0, :].numpy().reshape(64, 64, HD)  # seq-first -> HWC
    print("fpn shapes:", [tuple(f.shape) for f in fpn])

    from ai_edge_litert.interpreter import Interpreter
    interp = Interpreter(model_path=a.tflite, num_threads=8)
    runners = {s: interp.get_signature_runner(s)
               for s in interp.get_signature_list()}
    print("signatures:", list(interp.get_signature_list()))

    # ---------------- encode ----------------
    px = frame(a.frame)[0].transpose(1, 2, 0)[None]  # NHWC
    enc = runners["encode"](pixels=px.astype(np.float32))
    # outputs by name
    pix_raw_m = enc["pix_raw"]          # [1,64,64,256]
    s1_m = enc["feat_s1"]               # [1,128,128,64]
    s0_m = enc["feat_s0"]               # [1,256,256,32]
    print(f"[encode] frame {a.frame}")
    report("pix_raw", pix_raw_m[0], hf_pix_raw)
    # hi-res skips: HF applies conv_s0/s1 inside... capture from decoder call
    dec_args, dec_kwargs, dec_out = cap["dec"]
    hf_hi = dec_kwargs["high_resolution_features"]
    report("feat_s0", s0_m[0], hf_hi[0][0].numpy().transpose(1, 2, 0))
    report("feat_s1", s1_m[0], hf_hi[1][0].numpy().transpose(1, 2, 0))

    # ---------------- decode (HF inputs) ----------------
    hf_img_emb = dec_kwargs["image_embeddings"]         # [1,256,64,64] = pix_feat + dense? no: raw pix_feat
    hf_sparse = dec_kwargs["sparse_prompt_embeddings"]  # [1,1,2,256]
    hf_dense = dec_kwargs["dense_prompt_embeddings"]
    print("decoder input shapes:", tuple(hf_img_emb.shape), tuple(hf_sparse.shape),
          tuple(hf_dense.shape))
    dec_m = runners["decode"](
        pix_feat_in=hf_img_emb[0].numpy().transpose(1, 2, 0)[None].astype(np.float32),
        feat_s1=hf_hi[1][0].numpy().transpose(1, 2, 0)[None].astype(np.float32),
        feat_s0=hf_hi[0][0].numpy().transpose(1, 2, 0)[None].astype(np.float32),
        sparse=hf_sparse[0, 0].numpy()[None].astype(np.float32),
        nomem=np.zeros((1, 1, 1, 1), np.float32))  # HF pix_feat already conditioned
    masks_hf, iou_hf, tok_hf, obj_hf = dec_out
    print(f"[decode] on HF inputs (multimask slice 1:)")
    report("masks(1:4)", dec_m["masks"][0, 1:], masks_hf[0, 0].numpy())
    report("iou(1:4)", dec_m["iou_scores"][0, 1:], iou_hf[0, 0].numpy())
    report("object_score", dec_m["object_score"][0], obj_hf[0, 0].numpy())

    # ---------------- memorize (HF inputs) ----------------
    me_args, me_kwargs, me_out = cap["me"]
    hf_me_pix = me_args[0] if me_args else me_kwargs["vision_features"]
    hf_me_mask = me_args[1] if len(me_args) > 1 else me_kwargs["masks"]
    hf_mem = me_out[0]
    print("memorize input shapes:", tuple(hf_me_pix.shape), tuple(hf_me_mask.shape))
    mem_m = runners["memorize"](
        pix_raw=hf_me_pix[0].numpy().transpose(1, 2, 0)[None].astype(np.float32),
        mask_for_mem=hf_me_mask[0].numpy().transpose(1, 2, 0)[None].astype(np.float32),
        occ=np.zeros((1, 1, 1), np.float32))
    print(f"[memorize] on HF inputs")
    report("mem", mem_m["mem"][0],
           hf_mem[0].numpy().reshape(MEMCH, HW).T)

    # ---------------- memcond (HF inputs) ----------------
    ma_args, ma_kwargs, ma_out = cap["ma"]
    cur = ma_kwargs["current_vision_features"]        # (HW, B, C)
    mem_in = ma_kwargs["memory"]                       # (L, B, 64)
    mem_pos = ma_kwargs["memory_posision_embeddings"]  # (L, B, 64)
    nptr = ma_kwargs["num_object_pointer_tokens"]
    L = mem_in.shape[0]
    n_spatial = (L - nptr) // HW
    print(f"memcond: L={L} nptr={nptr} spatial_slots={n_spatial}")
    nmm = 7
    sig = "memcond7"
    hw_tokens = cur[:, 0, :].numpy()                   # (HW, C)
    mem_bank = np.zeros((1, nmm, HW, MEMCH), np.float32)
    slot_tpe = np.zeros((1, nmm, 1, MEMCH), np.float32)
    ptr_tok = np.zeros((1, 1, 64, MEMCH), np.float32)
    ptr_pos = np.zeros((1, 1, 64, MEMCH), np.float32)
    km = np.full((1, 1, 1, nmm * HW + 64), -1e9, np.float32)
    spatial = mem_in[:L - nptr, 0, :].numpy().reshape(n_spatial, HW, MEMCH)
    spatial_pos = mem_pos[:L - nptr, 0, :].numpy().reshape(n_spatial, HW, MEMCH)
    # my graph adds baked mem_pos + slot_tpe; HF pos = mem_pos_grid + tpe row.
    # Recover the per-slot tpe row: spatial_pos - baked grid (constant per slot).
    from safetensors import safe_open
    wpath = os.environ["SAM2V_WEIGHTS"]  # sam2_tiny_1024_video.safetensors
    with safe_open(wpath, framework="np") as f:
        grid = f.get_tensor("tables.mem_pos")          # (HW, 64)
    for s in range(n_spatial):
        mem_bank[0, s] = spatial[s]
        tpe = spatial_pos[s] - grid
        slot_tpe[0, s, 0] = tpe.mean(axis=0)
        print(f"    slot {s}: tpe row std over tokens = {tpe.std(axis=0).max():.2e}")
        km[0, 0, 0, s * HW:(s + 1) * HW] = 0
    ptrs = mem_in[L - nptr:, 0, :].numpy()
    ptrs_pos = mem_pos[L - nptr:, 0, :].numpy()
    ptr_tok[0, 0, :nptr] = ptrs
    ptr_pos[0, 0, :nptr] = ptrs_pos
    km[0, 0, 0, nmm * HW:nmm * HW + nptr] = 0
    mc = runners[sig](
        pix_raw=hw_tokens.reshape(1, 64, 64, HD).astype(np.float32),
        mem_bank=mem_bank, slot_tpe=slot_tpe, ptr_tok=ptr_tok,
        ptr_pos=ptr_pos, key_mask=km)
    hf_pf = ma_out[0].numpy()  # shape?
    print("ma out shape:", tuple(ma_out.shape) if hasattr(ma_out, "shape")
          else [tuple(o.shape) for o in ma_out])
    pf_ref = np.asarray(hf_pf).reshape(-1)
    pf_mine = mc["pix_feat"].reshape(-1)
    if pf_ref.size == pf_mine.size:
        # HF returns (HW, B, C) seq-first -> token-major already
        report("pix_feat", pf_mine, pf_ref)
    else:
        print("  pix_feat size mismatch", pf_mine.size, pf_ref.size)


if __name__ == "__main__":
    main()

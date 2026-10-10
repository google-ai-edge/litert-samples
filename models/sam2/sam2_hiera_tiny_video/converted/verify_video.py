# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Verifies the exported SAM 2.1 video graphs against the HF PyTorch model.

Runs a short synthetic clip (a white disk drifting on black, one positive click
on frame 0) through two pipelines and reports their agreement:

  1. HF ``Sam2VideoModel`` streaming inference (the reference).
  2. The four exported .tflite graphs driven by the numpy HOST LOOP in
     ``track`` -- the same rolling-bank / best-mask / no-object / mask-for-mem
     orchestration a Kotlin or Swift app performs on device. The graphs run
     through the LiteRT CompiledModel Python API (``run_by_index``).

The host loop is the executable specification of the on-device orchestration;
everything heavy is a graph, everything else is plain array bookkeeping.

Usage (convert env, after export_video.py has written the graphs + constants):
    SAM2_OUT=out python verify_video.py [--nmm 7,2] [--frames 8]
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from ai_edge_litert.compiled_model import CompiledModel
from PIL import Image, ImageDraw
from transformers import Sam2VideoInferenceSession, Sam2VideoModel

OUT = os.path.abspath(os.environ.get("SAM2_OUT", "out"))
CKPT = os.environ.get("SAM2_CKPT", "facebook/sam2.1-hiera-tiny")
IE, F0, F1 = 1048576, 2097152, 1048576
HW, MEMCH, HD = 4096, 64, 256
NPTR_FRAMES, PTR_SPLIT = 16, 4
NPTR = NPTR_FRAMES * PTR_SPLIT
NO_OBJ, MEM_SCALE, MEM_BIAS, MASK_NEG = -1024.0, 20.0, -10.0, -1e9
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])
CLICK = (400.0, 512.0)


def frame(t):
    """Renders synthetic frame ``t`` as a normalized CHW float32 array.

    Args:
        t: Frame index; the disk drifts by (14, 6) px per frame in 512-space.

    Returns:
        A (1, 3, 1024, 1024) float32 array.
    """
    img = Image.new("RGB", (512, 512), "black")
    cx, cy, r = 200 + 14 * t, 256 + 6 * t, 100
    ImageDraw.Draw(img).ellipse([cx - r, cy - r, cx + r, cy + r], fill="white")
    arr = np.asarray(img.resize((1024, 1024), Image.BILINEAR))
    arr = arr.astype(np.float32) / 255.0
    chw = ((arr - MEAN) / STD).transpose(2, 0, 1)
    return chw[None].astype(np.float32)


def sine_pe_1d(pos, dim=256, temperature=10000.0):
    """Computes the 1-D sine position embedding of a scalar position.

    Args:
        pos: The scalar position.
        dim: Embedding dimension.
        temperature: Sine temperature.

    Returns:
        A (dim,) float32 array.
    """
    pe_dim = dim // 2
    dim_t = np.arange(pe_dim, dtype=np.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / pe_dim)
    x = pos / dim_t
    return np.concatenate([np.sin(x), np.cos(x)]).astype(np.float32)


class Consts:
    """Loads the host-side constant tables written by export_video.py."""

    def __init__(self, directory=OUT):
        """Reads every constant .bin from ``directory``.

        Args:
            directory: The output dir holding the sam2v_*.bin files.
        """
        p = np.fromfile(f"{directory}/sam2v_prompt.bin", np.float32)
        self.gauss = p[:256]
        self.pe1, self.pe0, self.nap = p[256:512], p[512:768], p[768:1024]
        self.track = np.fromfile(
            f"{directory}/sam2v_track_sparse.bin", np.float32)
        self.mtpe = np.fromfile(
            f"{directory}/sam2v_mtpe.bin", np.float32).reshape(7, 64)
        self.no_obj_ptr = np.fromfile(
            f"{directory}/sam2v_no_obj_ptr.bin", np.float32)
        tp = np.fromfile(f"{directory}/sam2v_tpos_proj.bin", np.float32)
        self.tpos_w = tp[:64 * 256].reshape(64, 256)
        self.tpos_b = tp[64 * 256:]

    def click_sparse(self, x, y):
        """Builds the 2-token sparse prompt for a positive click.

        Args:
            x: Click x in model (0..1024) coordinates.
            y: Click y in model (0..1024) coordinates.

        Returns:
            A (512,) float32 sparse prompt.
        """
        xn = 2 * ((x + 0.5) / 1024) - 1
        yn = 2 * ((y + 0.5) / 1024) - 1
        proj = 2 * math.pi * (xn * self.gauss[:128] + yn * self.gauss[128:])
        tok = np.concatenate([np.sin(proj), np.cos(proj)]) + self.pe1
        return np.concatenate([tok, self.nap]).astype(np.float32)

    def ptr_pos(self, t_diff):
        """Projects a temporal offset to an object-pointer position embedding.

        Args:
            t_diff: Frames between the current and the pointer's frame.

        Returns:
            A (64,) float32 position embedding.
        """
        pe = sine_pe_1d(t_diff / (NPTR_FRAMES - 1.0)).astype(np.float64)
        out = self.tpos_w.astype(np.float64) @ pe + self.tpos_b
        return out.astype(np.float32)


def upsample_1024(low):
    """Bilinearly upsamples a 256x256 mask to 1024x1024 (align_corners=False).

    Args:
        low: A (256, 256) float32 mask.

    Returns:
        A (1024, 1024) float32 mask.
    """
    t = torch.from_numpy(low)[None, None]
    up = F.interpolate(
        t, size=(1024, 1024), mode="bilinear", align_corners=False)
    return up[0, 0].numpy()


def split_dec(out):
    """Splits the flat decoder output into its four parts.

    Args:
        out: The flat decoder output array.

    Returns:
        A tuple (masks(4,256,256), iou(4), ptr(4,256), obj_score).
    """
    masks = out[:4 * 65536].reshape(4, 256, 256)
    iou = out[4 * 65536:4 * 65536 + 4]
    ptr = out[4 * 65536 + 4:4 * 65536 + 4 + 1024].reshape(4, 256)
    return masks, iou, ptr, float(out[-1])


class TfliteRunner:
    """Runs the four exported graphs through the CompiledModel Python API."""

    def __init__(self, nmm_list):
        """Loads every graph for the requested bank sizes.

        Args:
            nmm_list: The memory-bank sizes to load memcond graphs for.
        """
        self.enc = CompiledModel.from_file(f"{OUT}/sam2v_encode.tflite")
        self.dec = CompiledModel.from_file(f"{OUT}/sam2v_decode.tflite")
        self.memz = CompiledModel.from_file(f"{OUT}/sam2v_memorize.tflite")
        self.mc = {
            n: CompiledModel.from_file(f"{OUT}/sam2v_memcond{n}.tflite")
            for n in nmm_list
        }

    def _run(self, model, flat, out_count):
        """Runs a single-signature graph on one flat input.

        Args:
            model: The CompiledModel to run.
            flat: The flat float32 input array.
            out_count: The number of float32 outputs to read.

        Returns:
            The flat output array.
        """
        ins = model.create_input_buffers(0)
        outs = model.create_output_buffers(0)
        ins[0].write(np.ascontiguousarray(flat.astype(np.float32)))
        model.run_by_index(0, ins, outs)
        return outs[0].read(out_count, np.float32).copy()

    def encode(self, chw):
        """Encodes one frame into (pix_raw, hi0, hi1)."""
        flat = self._run(self.enc, chw.ravel(), IE + F0 + F1)
        return flat[:IE], flat[IE:IE + F0], flat[IE + F0:]

    def memcond(self, n, pix_raw, mem, tpe, ptr_tok, ptr_pos, km):
        """Runs memory attention for bank size ``n``; returns pix_feat."""
        flat = np.concatenate([
            pix_raw, mem.ravel(), tpe.ravel(), ptr_tok.ravel(),
            ptr_pos.ravel(), km,
        ])
        return self._run(self.mc[n], flat, IE)

    def decode(self, pix_feat, hi0, hi1, sparse, nomem):
        """Runs the mask decoder; returns split outputs."""
        flat = np.concatenate([pix_feat, hi0, hi1, sparse, [nomem]])
        return split_dec(self._run(self.dec, flat, 4 * 65536 + 4 + 1024 + 1))

    def memorize(self, pix_raw, mfm, occ):
        """Runs the memory encoder; returns the (4096, 64) spatial memory."""
        flat = np.concatenate([pix_raw, mfm.ravel(), [occ]])
        return self._run(self.memz, flat, HW * MEMCH).reshape(HW, MEMCH)


def track(run, nmm, consts, num_frames):
    """Runs the on-device host loop over the four graphs.

    Args:
        run: A runner exposing encode/memcond/decode/memorize.
        nmm: The spatial memory-bank size.
        consts: The loaded ``Consts``.
        num_frames: Number of frames to track.

    Returns:
        A list of per-frame dicts with the low-res mask and object score.
    """
    spatial_bank, ptr_bank, outs = {}, {}, []
    cond = 0
    for t in range(num_frames):
        pix_raw, hi0, hi1 = run.encode(frame(t))
        prompted = t == cond
        if prompted:
            sparse, nomem, pix_feat = consts.click_sparse(*CLICK), 1.0, pix_raw
        else:
            mem = np.zeros((nmm, HW, MEMCH), np.float32)
            tpe = np.zeros((nmm, MEMCH), np.float32)
            km = np.full(nmm * HW + NPTR, MASK_NEG, np.float32)
            slot = 0
            mem[slot], tpe[slot] = spatial_bank[cond], consts.mtpe[6]
            km[slot * HW:(slot + 1) * HW] = 0
            slot += 1
            for off in range(nmm - 1, 0, -1):
                pf = t - off
                if pf in spatial_bank and pf != cond:
                    mem[slot] = spatial_bank[pf]
                    tpe[slot] = consts.mtpe[off - 1]
                    km[slot * HW:(slot + 1) * HW] = 0
                    slot += 1
            ptr_tok = np.zeros((NPTR, MEMCH), np.float32)
            ptr_pos = np.zeros((NPTR, MEMCH), np.float32)
            ptrs = [(t - cond, ptr_bank[cond])]
            for td in range(1, NPTR_FRAMES):
                pf = t - td
                if pf < 0:
                    break
                if pf in ptr_bank and pf != cond:
                    ptrs.append((td, ptr_bank[pf]))
            for i, (td, p) in enumerate(ptrs):
                pos = consts.ptr_pos(td)
                for j in range(PTR_SPLIT):
                    tok = i * PTR_SPLIT + j
                    ptr_tok[tok] = p[j * MEMCH:(j + 1) * MEMCH]
                    ptr_pos[tok] = pos
                    km[nmm * HW + tok] = 0
            pix_feat = run.memcond(
                nmm, pix_raw, mem, tpe, ptr_tok, ptr_pos, km)
            sparse, nomem = consts.track, 0.0
        masks, iou, ptr, obj = run.decode(pix_feat, hi0, hi1, sparse, nomem)
        best = 1 + int(np.argmax(iou[1:]))
        appearing = obj > 0
        if appearing:
            low = masks[best]
            obj_ptr = ptr[best]
        else:
            low = np.full((256, 256), NO_OBJ, np.float32)
            obj_ptr = consts.no_obj_ptr
        high = upsample_1024(low)
        if prompted:
            mfm = (high > 0).astype(np.float32) * MEM_SCALE + MEM_BIAS
        else:
            mfm = 1.0 / (1.0 + np.exp(-high)) * MEM_SCALE + MEM_BIAS
        occ = 0.0 if appearing else 1.0
        mem_t = run.memorize(pix_raw, mfm.astype(np.float32), occ)
        spatial_bank[t], ptr_bank[t] = mem_t, obj_ptr
        outs.append({"mask": low, "obj": float(obj)})
    return outs


def reference(nmm, num_frames):
    """Runs the clean HF streaming reference.

    Args:
        nmm: The number of memory slots to configure on the model.
        num_frames: Number of frames to track.

    Returns:
        A list of per-frame (mask, obj_score) tuples.
    """
    model = Sam2VideoModel.from_pretrained(CKPT).eval()
    model.num_maskmem = nmm
    session = Sam2VideoInferenceSession(
        video_height=1024, video_width=1024, dtype=torch.float32)
    obj = session.obj_id_to_idx(1)
    session.add_point_inputs(obj, 0, {
        "point_coords": torch.tensor([[[[CLICK[0], CLICK[1]]]]]),
        "point_labels": torch.tensor([[[1]]]),
    })
    session.obj_with_new_inputs = [1]
    out = []
    with torch.no_grad():
        for t in range(num_frames):
            result = model(
                inference_session=session, frame=torch.from_numpy(frame(t)))
            mask = result.pred_masks.numpy().reshape(256, 256)
            score = float(result.object_score_logits.reshape(-1)[0])
            out.append((mask, score))
    return out


def mask_iou(a, b):
    """Computes the binary-mask IoU of two logit maps thresholded at 0.

    Args:
        a: First logit map.
        b: Second logit map.

    Returns:
        The IoU as a float.
    """
    union = np.logical_or(a > 0, b > 0).sum()
    inter = np.logical_and(a > 0, b > 0).sum()
    return float(inter / union) if union else 1.0


def corr(a, b):
    """Computes the Pearson correlation of two arrays.

    Args:
        a: First array-like.
        b: Second array-like.

    Returns:
        The correlation coefficient as a float.
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    """Runs the reference and tflite pipelines and prints their agreement."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmm", default="7,2")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    nmms = [int(x) for x in args.nmm.split(",")]
    consts = Consts()
    runner = TfliteRunner(nmms)
    for n in nmms:
        ref = reference(n, args.frames)
        got = track(runner, n, consts, args.frames)
        worst_iou, worst_corr = 1.0, 1.0
        for t, out in enumerate(got):
            iou = mask_iou(ref[t][0], out["mask"])
            cc = corr(ref[t][0], out["mask"])
            worst_iou = min(worst_iou, iou)
            worst_corr = min(worst_corr, cc)
            fg = int((out["mask"] > 0).sum())
            print(f"  nmm={n} f{t:02d} IoU={iou:.4f} corr={cc:.5f} "
                  f"fg={fg} obj={out['obj']:.2f}/{ref[t][1]:.2f}")
        print(f"[nmm={n}] frames={args.frames} min IoU={worst_iou:.4f} "
              f"min corr={worst_corr:.5f}")


if __name__ == "__main__":
    main()

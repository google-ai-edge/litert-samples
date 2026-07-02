#!/usr/bin/env python3
"""Torch re-implementation of Spotify basic-pitch nmp (ICASSP 2022), built from the official
ONNX's extracted constants (bp_weights.npz). Exact dataflow port of the conv-based CQT2010v2:
two shared 36x256 kernel banks (real/imag) over 9 octaves with a lowpass stride-2 downsample
chain, NormalizedLog, fused BN, harmonic stacking, and the 3-branch CNN.

Parity target: the OFFICIAL onnx/tflite outputs (contour/note/onset)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SR, HOP, N_BINS, BPO = 22050, 256, 309, 36
N_OCT = 9
N_SAMPLES = 43844
HARM_SHIFTS = [round(BPO * np.log2(h)) for h in [0.5, 1, 2, 3, 4, 5, 6, 7]]  # [-36,0,36,57,72,84,93,101]
N_OUT_FREQS = 264


class BasicPitchGPU(nn.Module):
    def __init__(self, npz_path="bp_weights.npz"):
        super().__init__()
        w = np.load(npz_path)

        def t(name):
            return torch.from_numpy(w[name].copy())

        # CQT front-end. The per-bin norm (~200x) is folded INTO per-octave kernel copies:
        # magnitude is linear in the kernel scale, so this is exact — and it lifts the conv
        # outputs from ~1e-3 (fp16-precision-poor on the delegate) into a healthy range
        # (the Fast-Neural-Style conv-scaling lesson).
        kr = t("const_fold_opt__664").squeeze(1)                               # [36,1,256]
        ki = t("const_fold_opt__655").squeeze(1)
        norm = t("model_1/cq_t2010v2_1/Sqrt;model_1/cq_t2010v2_1/Sqrt").view(-1)  # [309]
        for o in range(N_OCT):                                                 # o=0 is TOP octave
            if o < N_OCT - 1:
                lo = 309 - 36 * (o + 1)
                s = norm[lo:lo + 36]                                           # full 36 bins
            else:
                s = torch.cat([torch.ones(15), norm[0:21]])                    # bottom 15 dropped
            self.register_buffer(f"kr_{o}", kr * s.view(36, 1, 1))
            self.register_buffer(f"ki_{o}", ki * s.view(36, 1, 1))
        self.register_buffer("lp", t("const_fold_opt__734").squeeze(1))        # [1,1,256]
        # fused batchnorm after NormalizedLog: scalar scale (V3) + offset (V31)
        self.register_buffer("bn_mul",
            t("model_1/batch_normalization/FusedBatchNormV3;model_1/batch_normalization/FusedBatchNormV3"))
        self.register_buffer("bn_add",
            t("model_1/batch_normalization/FusedBatchNormV3;model_1/batch_normalization/FusedBatchNormV31"))
        # CNN weights
        self.register_buffer("w_contour1", t("const_fold_opt__727"))           # [8,8,3,39]
        self.register_buffer("w_contred", t("const_fold_opt__710"))            # [1,8,5,5]
        self.register_buffer("w_notes1", t("const_fold_opt__738"))             # [32,1,7,7]
        self.register_buffer("w_notes2", t("const_fold_opt__702"))             # [1,32,7,3]
        self.register_buffer("w_onset1", t("const_fold_opt__707"))             # [32,8,5,5]
        self.register_buffer("w_onset2", t("const_fold_opt__680"))             # [1,33,3,3]
        # biases / fused-BN vectors by their (unique) shapes
        self._load_vectors(w)

    def _load_vectors(self, w):
        """Map remaining float vectors by name fragments (see bp_manifest.txt)."""
        self.vecs = {}
        for k in w.files:
            a = w[k]
            if a.dtype == np.float32 and a.ndim == 1:
                self.vecs[k] = torch.from_numpy(a.copy())
        for k, v in self.vecs.items():
            self.register_buffer(f"v_{abs(hash(k)) % 10**8}", v, persistent=False)

    def vec(self, frag, size):
        cands = [v for k, v in self.vecs.items() if frag in k and v.numel() == size]
        assert len(cands) >= 1, (frag, size, [(k, v.numel()) for k, v in self.vecs.items()])
        return cands[0]

    def reflect128(self, x):
        """Exact reflect-pad(128,128) without GATHER: the reversed 128-sample edges are produced
        by a constant anti-diagonal matmul (FULLY_CONNECTED)."""
        if not hasattr(self, "flip128"):
            self.register_buffer("flip128", torch.eye(128).flip(0), persistent=False)
        left = x[:, :, 1:129]
        right = x[:, :, -129:-1]
        lf = (left.reshape(-1, 128) @ self.flip128).reshape(x.shape[0], x.shape[1], 128)
        rf = (right.reshape(-1, 128) @ self.flip128).reshape(x.shape[0], x.shape[1], 128)
        return torch.cat([lf, x, rf], dim=2)

    def cqt(self, x):
        """x: [1, 1, 43844] -> magnitude CQT [1, 309, T]."""
        octaves = []
        hop = HOP
        for o in range(N_OCT):
            xp = self.reflect128(x)
            xr = F.conv1d(xp, getattr(self, f"kr_{o}"), stride=hop)
            xi = F.conv1d(xp, getattr(self, f"ki_{o}"), stride=hop)
            octaves.append(torch.sqrt(xr * xr + xi * xi + 1e-20)[:, :, :172])
            if o < N_OCT - 1:
                xlp = F.pad(x, (127, 127))                      # constant pad (per ONNX pad_const__227)
                x = F.conv1d(xlp, self.lp, stride=2)
                hop //= 2
        # octaves[0] = TOP octave (original sr); stack low->high like CQT2010 output ordering
        spec = torch.cat(list(reversed(octaves)), dim=1)        # [1, 324, T] low->high
        return spec[:, -N_BINS:]                                # keep the TOP 309 bins

    def normalized_log(self, x):
        """x: [1, F, T] -> 0-1 normalized dB, per-sample (chained single-axis reduces)."""
        # fp16-safe (C38 class): the official +1e-10 floor underflows to 0 in the delegate's
        # fp16 arithmetic -> log(0) = -inf poisons the min/max normalization. Clamping AFTER the
        # log recovers exactly the desktop value (-100 dB) for silent bins and is a no-op on
        # desktop fp32.
        power = x * x
        lp = torch.clamp(10.0 * torch.log10(power + 1e-10), min=-100.0)
        mn = lp.min(dim=2, keepdim=True).values.min(dim=1, keepdim=True).values
        off = lp - mn
        mx = off.max(dim=2, keepdim=True).values.max(dim=1, keepdim=True).values
        return off / torch.clamp(mx, min=1e-10)

    def harmonic_stack(self, x):
        """x: [1, 1, T, F=309] -> [1, 8, T, 264]."""
        chans = []
        for s in HARM_SHIFTS:
            if s == 0:
                y = x
            elif s > 0:
                y = F.pad(x[:, :, :, s:], (0, s))
            else:
                y = F.pad(x[:, :, :, :s], (-s, 0))
            chans.append(y)
        return torch.cat(chans, dim=1)[:, :, :, :N_OUT_FREQS]

    def forward(self, wav):
        """wav: [1, 43844] -> (contour [1,172,264], note [1,172,88], onset [1,172,88])."""
        spec = self.cqt(wav.unsqueeze(1))                        # [1, 309, T]
        x = self.normalized_log(spec)
        x = x.transpose(1, 2).unsqueeze(1)                       # [1, 1, T, 309]
        x = x * self.bn_mul + self.bn_add                        # scalar fused BN
        hs = self.harmonic_stack(x)                              # [1, 8, T, 264]

        # contour branch: conv2d_1 (3,39) pads [1,19,1,19], BN folded (bias = offset) + relu
        c = F.conv2d(F.pad(hs, (19, 19, 1, 1)), self.w_contour1,
                     self.vec("re_lu_1/Relu", 8))
        c = F.relu(c)
        contour = torch.sigmoid(F.conv2d(F.pad(c, (2, 2, 2, 2)), self.w_contred,
                                         self.vec("contours-reduced/BiasAdd", 1)))
        # notes branch: on contour output (expand ch): conv2d_2 (7,7) stride (1,3) pads [3,2,3,2]
        n1 = F.conv2d(F.pad(contour, (2, 3, 3, 3)), self.w_notes1,
                      self.vec("conv2d_2/BiasAdd", 32), stride=(1, 3))
        n1 = F.relu(n1)
        note_pre = F.conv2d(F.pad(n1, (1, 1, 3, 3)), self.w_notes2,
                            self.vec("conv2d_3/BiasAdd", 1))
        note = torch.sigmoid(note_pre)
        # onset branch: conv2d_4 (5,5) stride (1,3) pads [2,1,2,1] on hs, BN folded + relu
        o1 = F.conv2d(F.pad(hs, (1, 2, 2, 2)), self.w_onset1,
                      self.vec("re_lu_3/Relu", 32), stride=(1, 3))
        o1 = F.relu(o1)
        cat = torch.cat([note, o1], dim=1)                       # [1, 33, T, 88]
        onset = torch.sigmoid(F.conv2d(F.pad(cat, (1, 1, 1, 1)), self.w_onset2,
                                       self.vec("conv2d_5/BiasAdd", 1)))
        return (contour.squeeze(1), note.squeeze(1), onset.squeeze(1))

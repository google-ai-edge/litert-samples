"""Export a GPU-shaped Bonsai Image DiT: same math, no rank-5 tensors.

The stock export runs only 169/2704 ops on the LiteRT GPU delegate: every
apply_rotary_emb carries a rank-5 reshape/concat (interleaved-pair rotation),
and Flux2PosEmbed contributes a rank>4 SLICE + BROADCAST_TO (plus the fp64
freq table). Both are killed here WITHOUT touching the weights:

1. Flux2PosEmbed.forward -> returns (cos, sin) precomputed ONCE with the real
   512x512 position ids (the pipeline's ids are constants). The ids graph
   inputs remain in the signature (ignored), so hosts don't change.
2. apply_rotary_emb -> rank-4 form: the interleaved rotation
   stack(-x_imag, x_real) equals x @ P for a constant (D, D) pair-swap
   matrix P (P[2i+1, 2i] = -1, P[2i, 2i+1] = +1), so
   out = x * cos + (x @ P) * sin, all rank-4.

Verified in-process against the unpatched fp32 forward before export.
Quantize afterwards with `python quantize_dit.py dit_gpu_fp32.tflite` + the
zero-scale fix, exactly like the shipped CPU DiT. The result
(dit_gpu_int4b32.tflite) is the DiT the macOS sample app
(samples/litert/image_generation/macos) runs on the Metal accelerator.
"""
import os, time, torch
import diffusers.models.transformers.transformer_flux2 as flux2mod
from diffusers import Flux2Transformer2DModel

from huggingface_hub import snapshot_download

SNAP = os.environ.get("BONSAI_SNAPSHOT") or snapshot_download(
    "prism-ml/bonsai-image-ternary-4B-unpacked")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dit_gpu_fp32.tflite")
B, IMG, TXT, GRID = 1, 1024, 256, 32


class DiT(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
        return self.m(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                      timestep=timestep, img_ids=img_ids, txt_ids=txt_ids, return_dict=False)[0]


m = Flux2Transformer2DModel.from_pretrained(f"{SNAP}/transformer", torch_dtype=torch.float32).eval()
AX = len(m.config.axes_dims_rope)
wrapped = DiT(m).eval()

# the pipeline's constant position ids (generate.py): img [0, h, w, 0], txt [0, 0, 0, i]
hh, ww = torch.meshgrid(torch.arange(GRID), torch.arange(GRID), indexing="ij")
img_ids = torch.stack([torch.zeros_like(hh), hh, ww, torch.zeros_like(hh)], -1).reshape(IMG, AX).float()
txt_ids = torch.stack([torch.zeros(TXT)] * 3 + [torch.arange(TXT)], -1).float()

# reference with only the shipped fp32-rope patch applied (same lineage as the
# CPU export's ref32) — computed here so the script is self-contained
g = torch.Generator().manual_seed(0)
C, JD = m.config.in_channels, m.config.joint_attention_dim
args = (torch.randn(B, IMG, C, generator=g), torch.randn(B, TXT, JD, generator=g),
        torch.tensor([1.0]), img_ids, txt_ids)
flux2mod.maybe_adjust_dtype_for_device = lambda dtype, device: (
    torch.float32 if dtype == torch.float64 else dtype)
with torch.no_grad():
    ref32 = wrapped(*args)
print(f"ref32 mean={ref32.mean():.6f} std={ref32.std():.6f}", flush=True)

# ---- patch 1: freeze the rope tables (fp32, computed once per ids set) -----
# pos_embed is called twice (img_ids, txt_ids); the transformer concatenates
# txt-first afterwards. Freeze each table and dispatch on the static row count.
with torch.no_grad():
    COS_IMG, SIN_IMG = m.pos_embed(img_ids)
    COS_TXT, SIN_TXT = m.pos_embed(txt_ids)
COS_IMG, SIN_IMG = COS_IMG.float().clone(), SIN_IMG.float().clone()
COS_TXT, SIN_TXT = COS_TXT.float().clone(), SIN_TXT.float().clone()
print(f"rope tables frozen: img {tuple(COS_IMG.shape)} txt {tuple(COS_TXT.shape)}", flush=True)
flux2mod.Flux2PosEmbed.forward = lambda self, ids: (
    (COS_TXT, SIN_TXT) if ids.shape[0] == TXT else (COS_IMG, SIN_IMG))

# ---- patch 2: rank-4 rotary ------------------------------------------------
D = COS_IMG.shape[-1]
P = torch.zeros(D, D)
for i in range(D // 2):
    P[2 * i + 1, 2 * i] = -1.0
    P[2 * i, 2 * i + 1] = 1.0


def rank4_rope(x, freqs_cis, use_real=True, use_real_unbind_dim=-1, sequence_dim=1):
    assert sequence_dim == 1 and use_real and use_real_unbind_dim == -1
    cos, sin = freqs_cis                      # [S, D]
    cos = cos[None, :, None, :]               # (1, S, 1, D)
    sin = sin[None, :, None, :]
    return (x.float() * cos + (x.float() @ P) * sin).to(x.dtype)


orig_rope = flux2mod.apply_rotary_emb
flux2mod.apply_rotary_emb = rank4_rope

# ---- verify: identical math, so this must match the reference --------------
with torch.no_grad():
    got = wrapped(*args)
d = (got - ref32).abs()
rel = d.max() / ref32.abs().max()
print(f"vs ref32: max|d|={d.max():.3e} rel={rel:.3e}", flush=True)
assert rel < 1e-4, "GPU-shaped rewrite is NOT numerically faithful"

import litert_torch
t0 = time.time()
lm = litert_torch.convert(wrapped, args)
print(f"convert OK in {time.time()-t0:.0f}s", flush=True)
t0 = time.time()
lm.export(OUT)
print(f"export OK in {time.time()-t0:.0f}s -> {os.path.getsize(OUT)/2**30:.2f} GiB", flush=True)

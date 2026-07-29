"""Export the Bonsai Image DiT to tflite, with the fp64-RoPE trap patched.

WALL: Flux2PosEmbed computes rope freqs in float64 (freqs_dtype=torch.float64),
so the graph carries tfl.pow on f64 -> "failed to legalize operation 'tfl.pow'".
FIX: force the rope freq dtype to float32. Verified numerically here against the
UNPATCHED float64 reference, so the patch is checked rather than assumed.
"""
import os, time, torch
import diffusers.models.transformers.transformer_flux2 as flux2mod
from diffusers import Flux2Transformer2DModel

from huggingface_hub import snapshot_download

SNAP = os.environ.get("BONSAI_SNAPSHOT") or snapshot_download(
    "prism-ml/bonsai-image-ternary-4B-unpacked")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "dit_fp32.tflite")
B, IMG, TXT = 1, 1024, 256


class DiT(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids):
        return self.m(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states,
                      timestep=timestep, img_ids=img_ids, txt_ids=txt_ids, return_dict=False)[0]


m = Flux2Transformer2DModel.from_pretrained(f"{SNAP}/transformer", torch_dtype=torch.float32).eval()
C, JD, AX = m.config.in_channels, m.config.joint_attention_dim, len(m.config.axes_dims_rope)
print(f"params {sum(p.numel() for p in m.parameters())/1e9:.3f} B", flush=True)

g = torch.Generator().manual_seed(0)
args = (torch.randn(B, IMG, C, generator=g), torch.randn(B, TXT, JD, generator=g),
        torch.tensor([1.0]), torch.zeros(IMG, AX), torch.zeros(TXT, AX))
wrapped = DiT(m).eval()

with torch.no_grad():
    ref64 = wrapped(*args)                       # untouched fp64-rope reference
print(f"ref(fp64 rope) mean={ref64.mean():.6f} std={ref64.std():.6f}", flush=True)

flux2mod.maybe_adjust_dtype_for_device = lambda dtype, device: (
    torch.float32 if dtype == torch.float64 else dtype)
with torch.no_grad():
    ref32 = wrapped(*args)                       # after the patch
d = (ref32 - ref64).abs()
print(f"ref(fp32 rope) mean={ref32.mean():.6f} std={ref32.std():.6f} | "
      f"max|d|={d.max():.3e} rel={d.max()/ref64.abs().max():.3e}", flush=True)
torch.save({"ref64": ref64, "ref32": ref32, "args": args}, os.path.join(HERE, "dit_ref.pt"))

import litert_torch
t0 = time.time()
lm = litert_torch.convert(wrapped, args)
print(f"convert OK in {time.time()-t0:.0f}s", flush=True)
t0 = time.time()
lm.export(OUT)
print(f"export OK in {time.time()-t0:.0f}s -> {os.path.getsize(OUT)/2**30:.2f} GiB", flush=True)

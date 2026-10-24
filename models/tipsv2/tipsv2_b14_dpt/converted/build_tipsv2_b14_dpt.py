# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TIPSv2-B/14 + DPT heads (google/tipsv2-b14-dpt) -> LiteRT CompiledModel GPU.

One 448x448 graph returns metric depth [1,1,448,448], unit surface normals
[1,3,448,448] and ADE20K segmentation logits [1,150,256,256]. The backbone is
a DINOv2-style ViT-B/14 (one register token, LayerScale); the three DPT heads
were trained on the frozen backbone (NYU Depth V2 / ADE20K).

The model is re-authored from the HF safetensors with GPU-clean, exact
rewrites (see README.md), checked against the official HF model, converted
with litert-torch and fp16-quantized. Every stage prints its parity numbers.

Setup (see README.md):
  pip install litert-torch ai-edge-litert ai-edge-quantizer transformers
  hf download google/tipsv2-b14-dpt   # weights + remote code (reference)

Run:
  KMP_DUPLICATE_LIB_OK=TRUE JAX_PLATFORMS=cpu \
      python build_tipsv2_b14_dpt.py {ref,parity,convert,fp16,all} image.jpg
"""

import collections
import os
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "google/tipsv2-b14-dpt"
FP32 = os.path.join(HERE, "tipsv2_b14_dpt.tflite")
FP16 = os.path.join(HERE, "tipsv2_b14_dpt_fp16.tflite")

IMG = 448
PATCH = 14
GRID = IMG // PATCH  # 32
N_PATCH = GRID * GRID  # 1024
N_REG = 1
N_TOK = 1 + N_REG + N_PATCH  # cls + register + patches = 1026
C = 768
DEPTH = 12
HEADS = 12
HD = C // HEADS
SCALE = HD**-0.5
LN_EPS = 1e-6
LN_S = 64.0  # SafeLayerNorm pre-square scale
TAPS = (2, 5, 8, 11)  # out_indices (3, 6, 9, 12) - 1
N_BINS = 256
MIN_D, MAX_D = 1e-3, 10.0
N_CLS = 150
SEG_RES = GRID * 8  # 256: DPT head resolution before the final resize

BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT",
          "SELECT_V2", "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST",
          "EMBEDDING_LOOKUP", "EQUAL", "NOT_EQUAL", "GREATER", "GREATER_EQUAL",
          "LESS", "LOGICAL_AND", "PACK", "SPLIT", "SPLIT_V", "CUMSUM"}


def safe_layer_norm(x, w, b, eps=LN_EPS):
  """LayerNorm whose variance cannot overflow fp16 (pre-scaled deviation).

  Args:
    x: Input [..., C].
    w: Affine weight [C].
    b: Affine bias [C].
    eps: Variance epsilon.

  Returns:
    The normalised tensor, algebraically identical to nn.LayerNorm.
  """
  mean = x.mean(-1, keepdim=True)
  d = x - mean
  var = (d * (1.0 / LN_S)).pow(2).mean(-1, keepdim=True) * (LN_S * LN_S)
  return (d * torch.rsqrt(var + eps)) * w + b


def gelu(x):
  """tanh-GELU: the delegate has no ERF kernel (the only approximation).

  Args:
    x: Input tensor.

  Returns:
    0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))).
  """
  return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))


def P(t):
  """Wraps a weight tensor as a frozen fp32 Parameter.

  Args:
    t: Source tensor (any float dtype).

  Returns:
    nn.Parameter with requires_grad=False.
  """
  return nn.Parameter(t.detach().clone().float(), requires_grad=False)


# ------------------------------------------------------------------ backbone
class Block(nn.Module):
  """One ViT block: 4D attention, LayerScale baked into proj / fc2."""

  def __init__(self, sd, i):
    super().__init__()
    pre = f"vision_encoder.blocks.{i}."
    self.n1w, self.n1b = P(sd[pre + "norm1.weight"]), P(sd[pre + "norm1.bias"])
    self.n2w, self.n2b = P(sd[pre + "norm2.weight"]), P(sd[pre + "norm2.bias"])
    self.qkv_w = P(sd[pre + "attn.qkv.weight"])
    self.qkv_b = P(sd[pre + "attn.qkv.bias"])
    g1, g2 = sd[pre + "ls1.gamma"], sd[pre + "ls2.gamma"]
    self.proj_w = P(g1.view(C, 1) * sd[pre + "attn.proj.weight"])
    self.proj_b = P(g1 * sd[pre + "attn.proj.bias"])
    self.fc1_w = P(sd[pre + "mlp.fc1.weight"])
    self.fc1_b = P(sd[pre + "mlp.fc1.bias"])
    self.fc2_w = P(g2.view(C, 1) * sd[pre + "mlp.fc2.weight"])
    self.fc2_b = P(g2 * sd[pre + "mlp.fc2.bias"])

  def forward(self, x):
    h = safe_layer_norm(x, self.n1w, self.n1b)
    qkv = h @ self.qkv_w.t() + self.qkv_b
    q, k, v = qkv.split(C, dim=-1)
    q = q.view(1, N_TOK, HEADS, HD).transpose(1, 2)
    k = k.view(1, N_TOK, HEADS, HD).transpose(1, 2)
    v = v.view(1, N_TOK, HEADS, HD).transpose(1, 2)
    attn = ((q * SCALE) @ k.transpose(-2, -1)).softmax(dim=-1)
    out = (attn @ v).transpose(1, 2).reshape(1, N_TOK, C)
    x = x + (out @ self.proj_w.t() + self.proj_b)
    h2 = safe_layer_norm(x, self.n2w, self.n2b)
    h2 = gelu(h2 @ self.fc1_w.t() + self.fc1_b)
    return x + (h2 @ self.fc2_w.t() + self.fc2_b)


class Backbone(nn.Module):
  """TIPSv2 ViT-B/14 at 448; returns (cls [1,C], patches [1,C,32,32]) x 4."""

  def __init__(self, sd):
    super().__init__()
    self.patch_w = P(sd["vision_encoder.patch_embed.proj.weight"])
    self.patch_b = P(sd["vision_encoder.patch_embed.proj.bias"])
    pos = sd["vision_encoder.pos_embed"].float()  # [1, 1025, C], 32x32 grid
    assert pos.shape[1] == 1 + N_PATCH, pos.shape
    self.cls_pos = P(sd["vision_encoder.cls_token"].float() + pos[:, :1])
    self.reg = P(sd["vision_encoder.register_tokens"])
    self.patch_pos = P(pos[:, 1:])
    self.blocks = nn.ModuleList([Block(sd, i) for i in range(DEPTH)])
    self.nw = P(sd["vision_encoder.norm.weight"])
    self.nb = P(sd["vision_encoder.norm.bias"])

  def forward(self, img):
    x = F.conv2d(img, self.patch_w, self.patch_b, stride=PATCH)
    x = x.flatten(2).transpose(1, 2) + self.patch_pos  # [1, 1024, C]
    x = torch.cat([self.cls_pos, self.reg, x], dim=1)  # [1, 1026, C]
    taps = []
    for i, blk in enumerate(self.blocks):
      x = blk(x)
      if i in TAPS:
        y = safe_layer_norm(x, self.nw, self.nb)
        cls = y[:, 0]
        patch = y[:, 1 + N_REG:].transpose(1, 2).reshape(1, C, GRID, GRID)
        taps.append((cls, patch))
    return taps


# ------------------------------------------------------------------ DPT
def up2_matrix(n):
  """Exact bilinear x2 upsample (align_corners=True) as a [2n, n] matrix.

  Args:
    n: Input size along the resized axis.

  Returns:
    Float32 tensor U [2n, n] with y = U @ x for a 1-D signal x.
  """
  m = np.zeros((2 * n, n), np.float32)
  for i in range(2 * n):
    s = i * (n - 1) / (2 * n - 1)
    i0 = int(np.floor(s))
    i1 = min(i0 + 1, n - 1)
    w = s - i0
    m[i, i0] += 1.0 - w
    m[i, i1] += w
  return torch.from_numpy(m)


class Up2(nn.Module):
  """y = U x U^T as two constant-RHS matmuls: exact and GPU-clean."""

  def __init__(self, n):
    super().__init__()
    self.ut = P(up2_matrix(n).t().contiguous())  # [n, 2n]

  def forward(self, x):
    y = x @ self.ut  # [1, C, n, n] -> [1, C, n, 2n]
    y = y.transpose(-1, -2) @ self.ut  # [1, C, 2n, 2n], W-major
    return y.transpose(-1, -2)


class ZeroStuffConvT(nn.Module):
  """ConvTranspose2d(k=s, stride=s) == zero-stuff + Conv2d(flipped): exact."""

  def __init__(self, w, b, n_in):
    super().__init__()
    self.s = self.k = w.shape[-1]
    self.w = P(w.flip(2, 3).transpose(0, 1).contiguous())
    self.b = P(b)
    s = self.s
    mk = np.zeros((n_in * s, n_in * s), np.float32)
    mk[::s, ::s] = 1.0
    self.mask = P(torch.from_numpy(mk)[None, None])
    self.n_out = n_in * s

  def forward(self, x):
    xn = F.interpolate(x, size=(self.n_out, self.n_out), mode="nearest")
    y = F.conv2d(xn * self.mask, self.w, bias=self.b, padding=self.k - 1)
    return y[:, :, :self.n_out, :self.n_out]


class ResUnit(nn.Module):
  """Pre-activation residual conv unit (bias-free convs)."""

  def __init__(self, sd, pre):
    super().__init__()
    self.w1, self.w2 = P(sd[pre + "conv1.weight"]), P(sd[pre + "conv2.weight"])

  def forward(self, x):
    h = F.conv2d(F.relu(x), self.w1, padding=1)
    h = F.conv2d(F.relu(h), self.w2, padding=1)
    return h + x


class Fusion(nn.Module):
  """DPT feature-fusion block: (+residual) -> ResUnit -> x2 up -> 1x1 conv."""

  def __init__(self, sd, pre, n_in, has_res, w_scale=1.0, lam_out=1.0):
    super().__init__()
    self.res = ResUnit(sd, pre + "residual_unit.") if has_res else None
    self.main = ResUnit(sd, pre + "main_unit.")
    self.up = Up2(n_in)
    # Range fold: the weights carry the local factor, the bias the ABSOLUTE
    # running scale lam_out (see DPTHead).
    self.ow = P(sd[pre + "out_conv.weight"] * w_scale)
    self.ob = P(sd[pre + "out_conv.bias"] * lam_out)

  def forward(self, x, residual=None):
    if self.res is not None:
      x = x + self.res(residual)
    x = self.main(x)
    x = self.up(x)
    return F.conv2d(x, self.ow, self.ob)


class DPTHead(nn.Module):
  """Reassemble + fusion + task head; `kind` in {depth, normals, seg}.

  The depth decoder's activations reach ~1e8 at the logits (fp16 max 65504;
  the GPU returned a constant depth). The decoder after the readout GELU is
  affine + ReLU + residual adds and ends in a scale-invariant normalisation,
  so power-of-2 scales are folded into it: a layer's weights carry the local
  factor, its bias carries the absolute running scale lambda at that point,
  and both operands of every residual add share the same lambda. Bit-exact in
  fp32; every stage stays below ~100 in fp16.
  """

  def __init__(self, sd, name, kind):
    super().__init__()
    self.kind = kind
    pre = name + "."
    if kind == "depth":
      conv_s = (2.0**-12, 2.0**-10, 2.0**-8, 2.0**-6)  # convs[i] (bias-free)
      fus_w, fus_lam = 0.25, (2.0**-8, 2.0**-10, 2.0**-12, 2.0**-14)
      proj_w, proj_lam = 2.0**-5, 2.0**-19
      head_w, head_lam = 2.0**-4, 2.0**-23
    else:
      conv_s, fus_w, fus_lam = (1.0,) * 4, 1.0, (1.0,) * 4
      proj_w = proj_lam = head_w = head_lam = 1.0
    self.lam = head_lam
    self.ro_a = nn.ParameterList()  # patch half of readout_projects
    self.ro_b = nn.ParameterList()  # cls half
    self.ro_bias = nn.ParameterList()
    self.pw = nn.ParameterList()
    self.pb = nn.ParameterList()
    self.cw = nn.ParameterList()
    for i in range(4):
      w = sd[pre + f"reassemble.readout_projects.{i}.weight"]  # [C, 2C]
      self.ro_a.append(P(w[:, :C]))
      self.ro_b.append(P(w[:, C:]))
      self.ro_bias.append(P(sd[pre + f"reassemble.readout_projects.{i}.bias"]))
      self.pw.append(P(sd[pre + f"reassemble.out_projections.{i}.weight"]))
      self.pb.append(P(sd[pre + f"reassemble.out_projections.{i}.bias"]))
      self.cw.append(P(sd[pre + f"convs.{i}.weight"] * conv_s[i]))
    rs = pre + "reassemble.resize_layers."
    self.rs0 = ZeroStuffConvT(sd[rs + "0.weight"], sd[rs + "0.bias"], GRID)
    self.rs1 = ZeroStuffConvT(sd[rs + "1.weight"], sd[rs + "1.bias"], GRID)
    self.rs3w, self.rs3b = P(sd[rs + "3.weight"]), P(sd[rs + "3.bias"])
    fb = pre + "fusion_blocks."
    self.fus = nn.ModuleList([
        Fusion(sd, fb + "0.", GRID // 2, False, fus_w, fus_lam[0]),
        Fusion(sd, fb + "1.", GRID, True, fus_w, fus_lam[1]),
        Fusion(sd, fb + "2.", GRID * 2, True, fus_w, fus_lam[2]),
        Fusion(sd, fb + "3.", GRID * 4, True, fus_w, fus_lam[3]),
    ])
    self.prw = P(sd[pre + "project.weight"] * proj_w)
    self.prb = P(sd[pre + "project.bias"] * proj_lam)
    last = {"depth": "depth_head", "normals": "normals_head",
            "seg": "segmentation_head"}[kind]
    self.hw = P(sd[pre + last + ".weight"] * head_w)
    self.hb = P(sd[pre + last + ".bias"] * head_lam)
    if kind == "depth":
      self.bins = P(torch.linspace(MIN_D, MAX_D, N_BINS).view(N_BINS, 1))
      self.eps = MIN_D * self.lam  # (relu(l) + MIN_D) * lam, exactly

  def forward(self, taps):
    feats = []
    for i, (cls, x) in enumerate(taps):
      xf = x.flatten(2).transpose(1, 2)  # [1, 1024, C]
      ro = cls @ self.ro_b[i].t() + self.ro_bias[i]  # [1, C]
      y = gelu(xf @ self.ro_a[i].t() + ro.unsqueeze(1))
      y = y.transpose(1, 2).reshape(1, C, GRID, GRID)
      y = F.conv2d(y, self.pw[i], self.pb[i])
      if i == 0:
        y = self.rs0(y)
      elif i == 1:
        y = self.rs1(y)
      elif i == 3:
        y = F.conv2d(y, self.rs3w, self.rs3b, stride=2, padding=1)
      feats.append(F.conv2d(y, self.cw[i], padding=1))
    out = self.fus[0](feats[3])
    out = self.fus[1](out, feats[2])
    out = self.fus[2](out, feats[1])
    out = self.fus[3](out, feats[0])  # [1, 256, 256, 256]
    out = F.conv2d(out, self.prw, self.prb, padding=1)
    if self.kind == "depth":
      out = F.relu(out)
    out = out.permute(0, 2, 3, 1) @ self.hw.t() + self.hb  # [1, 256, 256, K]
    if self.kind == "depth":
      out = F.relu(out) + self.eps
      out = out / out.sum(dim=-1, keepdim=True)  # scale-invariant
      depth = (out @ self.bins).permute(0, 3, 1, 2)  # [1, 1, 256, 256]
      return F.interpolate(depth, size=(IMG, IMG), mode="bilinear",
                           align_corners=False)
    if self.kind == "normals":
      out = out / torch.clamp(out.norm(dim=-1, keepdim=True), min=1e-12)
      out = out.permute(0, 3, 1, 2)  # [1, 3, 256, 256]
      return F.interpolate(out, size=(IMG, IMG), mode="bilinear",
                           align_corners=False)
    return out.permute(0, 3, 1, 2)  # [1, 150, 256, 256]


class TIPSv2DPT(nn.Module):
  """Backbone + the three DPT heads: returns (depth, normals, seg_logits)."""

  def __init__(self, sd):
    super().__init__()
    self.backbone = Backbone(sd)
    self.depth = DPTHead(sd, "depth_head", "depth")
    self.normals = DPTHead(sd, "normals_head", "normals")
    self.seg = DPTHead(sd, "segmentation_head", "seg")

  def forward(self, img):
    taps = self.backbone(img)
    return self.depth(taps), self.normals(taps), self.seg(taps)


# ------------------------------------------------------------------ stages
def load_sd():
  """Loads the official safetensors (backbone + three heads)."""
  from huggingface_hub import hf_hub_download
  from safetensors.torch import load_file
  return load_file(hf_hub_download(REPO, "model.safetensors"))


def preprocess(path):
  """Loads an image as [1, 3, 448, 448] float32 in [0, 1] (no mean/std).

  Args:
    path: Path to the image file.

  Returns:
    Float32 tensor [1, 3, 448, 448].
  """
  im = Image.open(path).convert("RGB").resize((IMG, IMG), Image.BILINEAR)
  arr = np.asarray(im, np.float32) / 255.0
  return torch.from_numpy(arr.transpose(2, 0, 1)[None])


def corr(a, b):
  """Pearson correlation of two arrays.

  Args:
    a: First array.
    b: Second array (same size).

  Returns:
    The correlation coefficient as a float.
  """
  return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def stage_ref(image):
  """Dumps the official HF model's outputs as the parity reference.

  Args:
    image: Path to the test image.
  """
  from transformers import AutoModel
  m = AutoModel.from_pretrained(REPO, trust_remote_code=True).eval()
  x = preprocess(image)
  with torch.no_grad():
    out = m(x)
  np.save(os.path.join(HERE, "ref_in.npy"), x.numpy())
  np.save(os.path.join(HERE, "ref_depth.npy"), out.depth.numpy())
  np.save(os.path.join(HERE, "ref_normals.npy"), out.normals.numpy())
  np.save(os.path.join(HERE, "ref_seg.npy"), out.segmentation.numpy())
  print(f"[ref] depth {out.depth.shape} range "
        f"{float(out.depth.min()):.2f}..{float(out.depth.max()):.2f} m; "
        f"normals {out.normals.shape}; seg {out.segmentation.shape}")


def compare(label, d, n, s):
  """Prints depth / normals / seg parity against the saved reference.

  Args:
    label: Log label prefix.
    d: Depth [1, 1, 448, 448].
    n: Normals [1, 3, 448, 448].
    s: Segmentation logits [1, 150, 256, 256].
  """
  rd = np.load(os.path.join(HERE, "ref_depth.npy"))
  rn = np.load(os.path.join(HERE, "ref_normals.npy"))
  rs = np.load(os.path.join(HERE, "ref_seg.npy"))
  s448 = F.interpolate(torch.from_numpy(s), size=(IMG, IMG), mode="bilinear",
                       align_corners=False).numpy()
  print(f"[{label}] depth   corr {corr(d, rd):.6f}  "
        f"max|d| {np.abs(d - rd).max():.4f} m")
  print(f"[{label}] normals corr {corr(n, rn):.6f}  "
        f"max|d| {np.abs(n - rn).max():.4f}")
  print(f"[{label}] seg     corr {corr(s448, rs):.6f}  argmax agree "
        f"{(s448.argmax(1) == rs.argmax(1)).mean():.4f}  "
        f"(256->448 bilinear vs official 448)")


def stage_parity(model, image):
  """Re-authored torch model vs the official reference.

  Args:
    model: The re-authored TIPSv2DPT module.
    image: Path to the test image.
  """
  x = preprocess(image)
  with torch.no_grad():
    d, n, s = [t.numpy() for t in model(x)]
  compare("torch", d, n, s)


def run_tflite(path, x):
  """Runs a converted graph through the CompiledModel API (CPU).

  Args:
    path: Path to the .tflite file.
    x: Input array [1, 3, 448, 448] float32 in [0, 1].

  Returns:
    (depth [1,1,448,448], normals [1,3,448,448], seg [1,150,256,256]).
  """
  from ai_edge_litert.compiled_model import CompiledModel
  model = CompiledModel.from_file(path)
  inputs = model.create_input_buffers(0)
  outputs = model.create_output_buffers(0)
  inputs[0].write(np.ascontiguousarray(x, np.float32))
  model.run_by_index(0, inputs, outputs)
  d = outputs[0].read(IMG * IMG, np.float32).reshape(1, 1, IMG, IMG)
  n = outputs[1].read(3 * IMG * IMG, np.float32).reshape(1, 3, IMG, IMG)
  s = outputs[2].read(N_CLS * SEG_RES * SEG_RES, np.float32)
  return d, n, s.reshape(1, N_CLS, SEG_RES, SEG_RES)


def opcheck(path, label):
  """Static GPU-compat scan: reads the op set from the .tflite file.

  Args:
    path: Path to the .tflite flatbuffer.
    label: Log label prefix.

  Returns:
    True when no banned op, >4D tensor, or FFT-family op is found.
  """
  from ai_edge_litert import schema_py_generated as schema
  with open(path, "rb") as f:
    model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
  names = {v: k for k, v in vars(schema.BuiltinOperator).items()
           if not k.startswith("_")}
  ops = collections.Counter()
  over = 0
  for g in model.subgraphs:
    for op in g.operators:
      c = model.operatorCodes[op.opcodeIndex]
      code = max(c.builtinCode, c.deprecatedBuiltinCode)
      if c.customCode:
        key = c.customCode.decode()
      else:
        key = names.get(code, str(code))
      ops[key] += 1
    over += sum(1 for t in g.tensors
                if t.shape is not None and len(t.shape) > 4)
  bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
  fft = {k: v for k, v in ops.items()
         if any(t in k.upper() for t in ("FFT", "STFT", "COMPLEX"))}
  print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
  print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} FFT:{fft or 'NONE'} "
        f"size {os.path.getsize(path)/1e6:.1f}MB")
  clean = not bad and not over and not fft
  print(f"[{label}] VERDICT:", "GPU-CLEAN" if clean else f"BLOCKERS {bad}")
  return clean


def stage_convert(model, image):
  """litert-torch convert -> fp32 .tflite, op-check, parity vs reference.

  Args:
    model: The re-authored TIPSv2DPT module.
    image: Path to the test image.
  """
  import litert_torch
  dummy = torch.zeros(1, 3, IMG, IMG)
  litert_torch.convert(model.eval(), (dummy,)).export(FP32)
  print(f"[convert] wrote {FP32}")
  opcheck(FP32, "fp32")
  d, n, s = run_tflite(FP32, preprocess(image).numpy())
  compare("fp32-tflite", d, n, s)


def stage_fp16(image):
  """FLOAT_CASTING fp16 weights -> the deployment file, re-verified.

  Args:
    image: Path to the test image.
  """
  from ai_edge_quantizer import quantizer, recipe_manager
  from ai_edge_quantizer.recipe import AlgorithmName, qtyping
  rm = recipe_manager.RecipeManager()
  rm.add_quantization_config(
      regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
      op_config=qtyping.OpQuantizationConfig(
          weight_tensor_config=qtyping.TensorQuantizationConfig(
              num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
          compute_precision=qtyping.ComputePrecision.FLOAT),
      algorithm_key=AlgorithmName.FLOAT_CASTING)
  if os.path.exists(FP16):
    os.remove(FP16)
  qt = quantizer.Quantizer(float_model=FP32)
  qt.load_quantization_recipe(rm.get_quantization_recipe())
  qt.quantize().export_model(FP16)
  print(f"[fp16] wrote {FP16}")
  opcheck(FP16, "fp16")
  d, n, s = run_tflite(FP16, preprocess(image).numpy())
  compare("fp16-tflite", d, n, s)


def main():
  """Runs the requested stage(s): ref, parity, convert, fp16 or all."""
  if len(sys.argv) < 3:
    print(__doc__)
    sys.exit(1)
  stage, image = sys.argv[1], sys.argv[2]
  if stage in ("ref", "all"):
    stage_ref(image)
  model = TIPSv2DPT(load_sd()).eval()
  if stage in ("parity", "all"):
    stage_parity(model, image)
  if stage in ("convert", "all"):
    stage_convert(model, image)
  if stage in ("fp16", "all"):
    stage_fp16(image)


if __name__ == "__main__":
  main()

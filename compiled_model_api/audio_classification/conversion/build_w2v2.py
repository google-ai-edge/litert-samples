# Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""wav2vec2 keyword-spotting -> CompiledModel GPU: re-authoring + parity + fp16.

All-GPU (no hybrid): the transformer residual peaks at |x|~3.2 (vs Mimi's 27), so there is no
fp16-on-Mali precision issue, and there is ZERO FFT (raw 16 kHz waveform -> 1D conv frontend).

Graph: waveform[1,16000] -> feature_extractor(conv) -> feature_projection -> encoder(12L) ->
       mean-pool over time -> projector -> classifier -> logits[1,12].

Re-authoring (all numerically-equivalent): GELU->tanh-GELU; feature-extractor GroupNorm->GN4D
(kills GATHER_ND); attention_mask=None so pooling is a plain mean (no masked-pool SELECT/BROADCAST).

Run: ~/clipconv/bin/python build_w2v2.py [stage]   stage in {opcheck,parity,all}
"""
import _stub  # noqa: F401  (must import first: env guards)
import sys
import os
import math
import collections
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForSequenceClassification
from transformers.activations import GELUActivation

HERE = os.path.dirname(os.path.abspath(__file__))
MID = "superb/wav2vec2-base-superb-ks"
SR, DUR = 16000, 1.0
torch.manual_seed(0)
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}


class TanhGELU(nn.Module):
    C = math.sqrt(2.0 / math.pi)
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(self.C * (x + 0.044715 * x * x * x)))


class GN4D(nn.Module):
    """GroupNorm w/o GATHER_ND: reshape (B,G,C//G,T), mean/var over (2,3). Matches Mimi/DAC."""
    def __init__(self, gn):
        super().__init__()
        self.g = gn.num_groups
        self.eps = gn.eps
        self.register_buffer("w", gn.weight.detach().reshape(1, -1, 1))
        self.register_buffer("b", gn.bias.detach().reshape(1, -1, 1))
    def forward(self, x):
        b, c, t = x.shape
        x = x.reshape(b, self.g, c // self.g, t)
        m = x.mean((2, 3), keepdim=True)
        v = (x - m).pow(2).mean((2, 3), keepdim=True)
        return (((x - m) * torch.rsqrt(v + self.eps)).reshape(b, c, t)) * self.w + self.b


def swap(m):
    for n, c in list(m.named_children()):
        if isinstance(c, (GELUActivation, nn.GELU)):
            setattr(m, n, TanhGELU())
        elif isinstance(c, nn.GroupNorm):
            setattr(m, n, GN4D(c))
        else:
            swap(c)


def fold_weight_norm(m):
    """The pos_conv (kernel-128 grouped Conv1d) keeps a weight_norm parametrization that RECOMPUTES
    g*v/||v|| at runtime (aten._weight_norm -> SUM/RSQRT/div on the weight) -> splits the Mali
    partition + fails to compile. Fold it into a static weight (DAC/Matcha do the same)."""
    import torch.nn.utils.parametrize as P
    for mod in m.modules():
        if P.is_parametrized(mod, "weight"):
            P.remove_parametrizations(mod, "weight", leave_parametrized=True)
    # old-style weight_norm (hook-based) fallback
    for mod in m.modules():
        if any(getattr(h, "__class__", type(None)).__name__ == "WeightNorm" for h in getattr(mod, "_forward_pre_hooks", {}).values()):
            torch.nn.utils.remove_weight_norm(mod)


def patch_mask():
    """The encoder calls create_bidirectional_mask() even when attention_mask=None, building an
    all-valid mask (arange/ge/expand -> SELECT_V2 + BROADCAST_TO) that then feeds SDPA. For a
    fixed-length clip with no padding that mask is a no-op, so return None -> SDPA does plain full
    attention (BATCH_MATMUL + SOFTMAX), GPU-clean."""
    import transformers.models.wav2vec2.modeling_wav2vec2 as W
    W.create_bidirectional_mask = lambda *a, **k: None


class KWS(nn.Module):
    """waveform -> logits, attention_mask=None (fixed-length, no padding -> clean mean-pool)."""
    def __init__(self, m):
        super().__init__()
        self.m = m
    def forward(self, x):
        return self.m(x, attention_mask=None).logits


def opcheck(path, label):
    """Static GPU-compat scan: read the op set straight from the .tflite flatbuffer."""
    from ai_edge_litert import schema_py_generated as schema
    with open(path, "rb") as f:
        model = schema.ModelT.InitFromPackedBuf(f.read(), 0)
    names = {v: k for k, v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
    ops = collections.Counter()
    over = 0
    for g in model.subgraphs:
        for op in g.operators:
            c = model.operatorCodes[op.opcodeIndex]
            code = max(c.builtinCode, c.deprecatedBuiltinCode)
            ops[c.customCode.decode() if c.customCode else names.get(code, str(code))] += 1
        over += sum(1 for t in g.tensors if t.shape is not None and len(t.shape) > 4)
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    fft = {k: v for k, v in ops.items() if any(t in k.upper() for t in ("FFT", "STFT", "COMPLEX"))}
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} FFT:{fft or 'NONE'} size {os.path.getsize(path)/1e6:.1f}MB")
    print(f"[{label}] VERDICT:", "GPU-CLEAN" if not bad and not over and not fft else f"BLOCKERS {bad}")
    return not bad and not over and not fft


def run_tflite(path, x):
    """Single inference through the LiteRT CompiledModel API; returns the flat fp32 output."""
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(path)
    inputs = model.create_input_buffers(0)
    outputs = model.create_output_buffers(0)
    inputs[0].write(np.ascontiguousarray(x, dtype=np.float32))
    model.run_by_index(0, inputs, outputs)
    n = model.get_output_buffer_requirements(0, 0)["buffer_size"] // np.dtype(np.float32).itemsize
    return outputs[0].read(n, np.float32)


def to_fp16(fp32, fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT), algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    qt = quantizer.Quantizer(float_model=fp32)
    qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(fp16)
    return fp16


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    patch_mask()
    m = Wav2Vec2ForSequenceClassification.from_pretrained(MID).eval()
    id2label = m.config.id2label
    x = torch.randn(1, int(SR * DUR))
    with torch.no_grad():
        ref = m(x, attention_mask=None).logits

    m2 = Wav2Vec2ForSequenceClassification.from_pretrained(MID).eval()
    swap(m2)
    fold_weight_norm(m2)
    g = KWS(m2).eval()
    with torch.no_grad():
        ra = g(x)
    print(f"re-authored vs orig torch: logits corr {np.corrcoef(ref.numpy().ravel(), ra.numpy().ravel())[0,1]:.6f} "
          f"max|d| {(ref-ra).abs().max():.2e}  argmax orig={ref.argmax(-1).item()} reauth={ra.argmax(-1).item()}")

    import litert_torch
    fp32 = os.path.join(HERE, "w2v2_kws.tflite")
    litert_torch.convert(g, (x,)).export(fp32)
    clean = opcheck(fp32, "kws")
    o = run_tflite(fp32, x.numpy())
    print(f"tflite vs torch: logits corr {np.corrcoef(o.ravel(), ra.numpy().ravel())[0,1]:.6f}  "
          f"argmax tflite={o.argmax()} ({id2label[int(o.argmax())]})")

    if stage == "all" and clean:
        f16 = to_fp16(fp32, os.path.join(HERE, "w2v2_kws_fp16.tflite"))
        opcheck(f16, "kws_fp16")
        o16 = run_tflite(f16, x.numpy())
        print(f"fp16 tflite vs torch: logits corr {np.corrcoef(o16.ravel(), ra.numpy().ravel())[0,1]:.6f}")


if __name__ == "__main__":
    main()

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

import warnings
warnings.filterwarnings("ignore")
import sys
import types
import inspect
_orig_gsf=inspect.getsourcefile
inspect.getsourcefile=lambda o:(_orig_gsf(o) if True else None) if False else (lambda: (_orig_gsf(o)))()
def _safe_gsf(o):
    try: return _orig_gsf(o)
    except Exception: return None
inspect.getsourcefile=_safe_gsf
class _Stub(types.ModuleType):
    __file__="<stub>"
    __spec__=None
    __path__=[]
    def __getattr__(s,n):
        if n.startswith("__"): raise AttributeError(n)
        return lambda *a,**k: None
def _mk(name, **attrs):
    m=_Stub(name)
    for k,v in attrs.items(): setattr(m,k,v)
    sys.modules[name]=m
    return m
for n in ["mmdet","mmdet.models","mmdet.models.utils","mmdet.structures","mmdet.structures.bbox","mmcv.ops"]: _mk(n)
_mk("mmdet.utils", ConfigType=dict, OptConfigType=dict, MultiConfig=dict, reduce_mean=(lambda x:x), InstanceList=list, OptInstanceList=list)
for n in ["chumpy"]:
    try: __import__(n)
    except Exception: _mk(n)
import os
import mmpose
import torch
import torch.nn as nn
import collections
import numpy as np
from mmpose.apis import init_model

# --- ScaleNorm: torch.norm lowers to an ABS/MAXIMUM/DIV path Mali mis-computes (x/inf=0); use manual sum-of-squares ---
from mmpose.models.utils.transformer import ScaleNorm as _SN
def _sn_forward(self, x):
    # SafeRMSNorm: ScaleNorm input reaches ~|274| so sum(x^2)~3.6M OVERFLOWS fp16 (65504) on Mali -> norm=inf
    # -> x/inf=0 (all-zero head). Scale x down by S before squaring so the sum stays fp16-safe (exact).
    xs = x * (1.0 / 64.0)
    norm = torch.sqrt((xs * xs).sum(dim=-1, keepdim=True) + 1e-12) * (64.0 * self.scale)
    return x / norm.clamp(min=self.eps) * self.g
_SN.forward = _sn_forward


# --- GAU act@act BMM -> broadcast-reduce (Mali mis-computes act@act BMM -> all-zero output) ---
import torch.nn.functional as _F
from mmpose.models.utils.rtmcc_block import RTMCCBlock as _RTMCC, rope as _rope
def _gau_forward(self, inputs):
    x = inputs
    x = self.ln(x)
    uv = self.act_fn(self.uv(x))
    u, v, base = torch.split(uv, [self.e, self.e, self.s], dim=2)
    base = base.unsqueeze(2) * self.gamma[None, None, :] + self.beta
    if self.pos_enc: base = _rope(base, dim=1)
    q, k = torch.unbind(base, dim=2)                      # [B,K,s] each
    qk = (q.unsqueeze(2) * k.unsqueeze(1)).sum(-1)        # [B,K,K]  (== bmm(q,k^T), broadcast-reduce)
    if self.use_rel_bias:
        bias = self.rel_pos_bias(q.size(1))
        qk = qk + bias[:, :q.size(1), :k.size(1)]
    kernel = torch.square(_F.relu(qk / self.sqrt_s))     # [B,K,K]
    x = u * (kernel.unsqueeze(-1) * v.unsqueeze(1)).sum(2)  # [B,K,e]  (== bmm(kernel,v))
    return self.o(x)
_RTMCC._forward = _gau_forward

HERE=os.path.dirname(os.path.abspath(__file__))
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","POW","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM"}
root=os.path.dirname(mmpose.__file__)
cfg=os.path.join(root,".mim/configs/face_2d_keypoint/rtmpose/wflw/rtmpose-m_8xb64-60e_wflw-256x256.py")
ckpt="https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-wflw_pt-aic-coco_60e-256x256-dc1dcdcf_20230228.pth"
class Wrap(nn.Module):
    def __init__(s,m):
        super().__init__()
        s.b=m.backbone
        s.h=m.head
    def forward(s,x):
        f=s.b(x)
        sx,sy=s.h(f if isinstance(f,(list,tuple)) else (f,))
        return sx,sy

def to_fp16(fp32,fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm=recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*",operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,op_config=qtyping.OpQuantizationConfig(weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16,dtype=qtyping.TensorDataType.FLOAT),compute_precision=qtyping.ComputePrecision.FLOAT),algorithm_key=AlgorithmName.FLOAT_CASTING)
    import os
    if os.path.exists(fp16): os.remove(fp16)
    q=quantizer.Quantizer(float_model=fp32)
    q.load_quantization_recipe(rm.get_quantization_recipe())
    q.quantize().export_model(fp16)
    return fp16

def opcheck(p,l):
    """Static GPU-compat scan: read the op set straight from the .tflite flatbuffer."""
    from ai_edge_litert import schema_py_generated as schema
    with open(p,"rb") as f: model=schema.ModelT.InitFromPackedBuf(f.read(),0)
    names={v:k for k,v in vars(schema.BuiltinOperator).items() if not k.startswith("_")}
    ops=collections.Counter()
    over=0
    for g in model.subgraphs:
        for op in g.operators:
            c=model.operatorCodes[op.opcodeIndex]
            code=max(c.builtinCode,c.deprecatedBuiltinCode)
            ops[c.customCode.decode() if c.customCode else names.get(code,str(code))]+=1
        over+=sum(1 for t in g.tensors if t.shape is not None and len(t.shape)>4)
    bad={k:v for k,v in ops.items() if k.upper() in BANNED}
    print(f"[{l}] ops:",dict(sorted(ops.items(),key=lambda kv:-kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB","GPU-CLEAN" if not bad and not over else "BLOCKERS")

def run_tflite(p,x):
    """Single inference through the LiteRT CompiledModel API; returns all outputs in signature order."""
    from ai_edge_litert.compiled_model import CompiledModel
    model=CompiledModel.from_file(p)
    sig=model.get_signature_list()
    key=list(sig)[0]
    det=model.get_output_tensor_details(key)
    ins=model.create_input_buffers(0)
    outs=model.create_output_buffers(0)
    ins[0].write(np.ascontiguousarray(x,dtype=np.float32))
    model.run_by_index(0,ins,outs)
    return [outs[i].read(int(np.prod(det[n]["shape"])),np.float32).reshape(det[n]["shape"]) for i,n in enumerate(sig[key]["outputs"])]
if __name__=="__main__":
    m=init_model(cfg,ckpt,device="cpu").eval()
    w=Wrap(m).eval()
    img=torch.randn(1,3,256,256)
    with torch.no_grad(): sx,sy=w(img)
    print("out simcc_x",tuple(sx.shape),"simcc_y",tuple(sy.shape))
    import litert_torch
    fp32=os.path.join(HERE,"rtm_face.tflite")
    litert_torch.convert(w,(img,)).export(fp32)
    to_fp16(fp32, os.path.join(HERE,"rtm_face_fp16.tflite"))
    opcheck(os.path.join(HERE,"rtm_face_fp16.tflite"),"rtm_fp16")
    opcheck(fp32,"rtm")
    o0,o1=run_tflite(fp32,img.numpy())
    refs=[sx.numpy(),sy.numpy()]
    # match outputs by shape
    for o in (o0,o1):
        best=max(refs,key=lambda r: -abs(r.size-o.size))
        c=np.corrcoef(o.ravel(),best.ravel())[0,1] if o.size==best.size else float("nan")
        print(f"  tflite-vs-torch out{tuple(o.shape)} corr {c:.5f}")

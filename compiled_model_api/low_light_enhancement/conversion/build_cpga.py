# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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

"""CPGA-Net (low-light enhancement) -> LiteRT CompiledModel GPU.

Model-side re-authoring (numerically exact): pow(x, g) -> exp(g * log(x)) so
the graph avoids the GPU-rejected POW op.
"""
import sys
import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE_=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE_,"CPGA-Net-Pytorch"))
import types as _t
try: import guided_filter_pytorch
except Exception:
    _m=_t.ModuleType("guided_filter_pytorch")
    _m.GuidedFilter=object
    _m.FastGuidedFilter=object
    sys.modules["guided_filter_pytorch"]=_m
    _g=_t.ModuleType("guided_filter_pytorch.guided_filter")
    _g.GuidedFilter=object
    _g.FastGuidedFilter=object
    sys.modules["guided_filter_pytorch.guided_filter"]=_g
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,"CPGA-Net-Pytorch"))
SIZE=int(os.environ.get("CP_SIZE","256"))
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","POW","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM","MIRROR_PAD"}
class GAP(nn.Module):
    def forward(self,x):
        return x.mean(3,keepdim=True).mean(2,keepdim=True)
class GMP(nn.Module):
    def forward(self,x):
        return F.max_pool2d(x,kernel_size=(x.shape[2],x.shape[3]))
class Wrap(nn.Module):
    def __init__(self,m):
        super().__init__()
        self.m=m
    def forward(self,x):
        # [1,3,H,W] in [0,1]
        m=self.m
        llie,t=m.ll(x,x)
        gamma=m.gamma_estimation(x)          # [B,1,1,1]
        out_g=torch.exp(gamma*torch.log(torch.clamp(llie,1e-9,1.0)))   # pow(llie,gamma) (POW banned)
        out=m.conv1_post_g(torch.cat((out_g,llie),dim=1))
        out=m.conv2_post_g(out)
        inter=m.conv3_post_g(out)
        return torch.clamp(-inter+out_g+llie,1e-9,1.0)
def build():
    from model import enhance_color
    m=enhance_color()
    sd=torch.load(f"{HERE}/CPGA-Net-Pytorch/weights/enhance_color-llie-ResCBAM_g.pkl",map_location="cpu",weights_only=False)
    sd=sd.get("state_dict",sd) if isinstance(sd,dict) and "state_dict" in sd else (sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd)
    if isinstance(sd,nn.Module): sd=sd.state_dict()
    miss,unexp=m.load_state_dict(sd,strict=False)
    n=0
    for mod in m.modules():
        for cn,ch in list(mod.named_children()):
            if isinstance(ch,nn.AdaptiveAvgPool2d):
                setattr(mod,cn,GAP())
                n+=1
            elif isinstance(ch,nn.AdaptiveMaxPool2d):
                setattr(mod,cn,GMP())
                n+=1
    m.eval()
    print(f"  loaded CPGA-Net; missing {len(miss)} unexpected {len(unexp)}; swapped {n} pools; params {sum(p.numel() for p in m.parameters())/1e6:.3f}M")
    return Wrap(m).eval()
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
    """Single inference through the LiteRT CompiledModel API; returns the flat fp32 output."""
    from ai_edge_litert.compiled_model import CompiledModel
    model=CompiledModel.from_file(p)
    ins=model.create_input_buffers(0)
    outs=model.create_output_buffers(0)
    ins[0].write(np.ascontiguousarray(x,dtype=np.float32))
    model.run_by_index(0,ins,outs)
    n=model.get_output_buffer_requirements(0,0)["buffer_size"]//np.dtype(np.float32).itemsize
    return outs[0].read(n,np.float32)
def to_fp16(fp32,fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm=recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*",operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,op_config=qtyping.OpQuantizationConfig(weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16,dtype=qtyping.TensorDataType.FLOAT),compute_precision=qtyping.ComputePrecision.FLOAT),algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    q=quantizer.Quantizer(float_model=fp32)
    q.load_quantization_recipe(rm.get_quantization_recipe())
    q.quantize().export_model(fp16)
    return fp16
if __name__=="__main__":
    m=build()
    x=torch.rand(1,3,SIZE,SIZE)*0.3
    with torch.no_grad(): ref=m(x)
    print(f"forward: out {tuple(ref.shape)} range [{ref.min():.3f},{ref.max():.3f}]")
    if (sys.argv[1] if len(sys.argv)>1 else "all")=="forward": sys.exit()
    import litert_torch
    fp32=f"{HERE}/cpga.tflite"
    try:
        litert_torch.convert(m,(x,)).export(fp32)
        opcheck(fp32,"cpga")
        o=run_tflite(fp32,x.numpy())
        print(f"tflite vs torch corr {np.corrcoef(o,ref.numpy().ravel())[0,1]:.6f}")
        to_fp16(fp32,f"{HERE}/cpga_fp16.tflite")
        opcheck(f"{HERE}/cpga_fp16.tflite","cpga_fp16")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("CONVERT FAIL:",repr(e)[:200])

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

"""UniSal (image saliency) -> LiteRT CompiledModel GPU.

Model-side re-authoring (numerically equivalent): the odd-size strided slice is
replaced with an avg_pool(1, 2) step so the graph stays GPU-clean.
"""
import sys
import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,"unisal"))
SIZE=int(os.environ.get("US_SIZE","256"))
SRC="SALICON"
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM","MIRROR_PAD"}
class Wrap(nn.Module):
    def __init__(self,m,gmaps):
        super().__init__()
        self.m=m
        self.ss=f"_{SRC.lower()}"
        self.register_buffer("gmaps",gmaps)
    def forward(self,img):
        # img [1,3,H,W]
        m=self.m
        f1,f2,f4=m.cnn(img)
        f2=m.skip_2x(f2)
        f4=m.skip_4x(f4)
        if m.n_gaussians>0: f1=torch.cat((f1,self.gmaps),dim=1)   # baked gaussian prior maps (size-only constant)
        f1=m.post_cnn(f1)
        u=m.upsampling_1(f1)
        u=torch.cat((u,f2),dim=1)
        u=m.upsampling_2(u)
        u=torch.cat((u,f4),dim=1)
        u=m.post_upsampling_2(u)
        u=getattr(m,"adaptation"+(self.ss if m.ds_adaptation else ""))(u)
        u=F.interpolate(u,size=img.shape[-2:],mode="nearest")
        k=m.smoothing_ksize//2
        u=F.pad(u,[k]*4,mode="constant",value=0.0)   # replicate->constant 0-pad (GPU-clean; suppresses border)
        u=getattr(m,"smoothing"+(self.ss if m.ds_smoothing else ""))(u)
        return u                                  # smoothed saliency [1,1,H,W]; host log-softmax/normalizes
def build():
    import unisal.model as M
    m=M.UNISAL().eval()
    sd=torch.load(f"{HERE}/unisal/training_runs/pretrained_unisal/weights_best.pth",map_location="cpu",weights_only=False)
    sd=sd.get("model",sd) if isinstance(sd,dict) and "model" in sd else sd
    miss,unexp=m.load_state_dict(sd,strict=False)
    m.this_source=SRC
    with torch.no_grad():
        f1,_,_=m.cnn(torch.zeros(1,3,SIZE,SIZE))
        g=m._get_gaussian_maps(f1, f"_{SRC.lower()}").detach()
    print(f"  loaded UniSal; missing {len(miss)} unexpected {len(unexp)}; gmaps {tuple(g.shape)}; params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    return Wrap(m,g).eval()
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
    x=torch.rand(1,3,SIZE,SIZE)
    with torch.no_grad(): ref=m(x)
    print(f"forward: out {tuple(ref.shape)} range [{ref.min():.3f},{ref.max():.3f}]")
    if (sys.argv[1] if len(sys.argv)>1 else "all")=="forward": sys.exit()
    import litert_torch
    fp32=f"{HERE}/unisal.tflite"
    try:
        litert_torch.convert(m,(x,)).export(fp32)
        opcheck(fp32,"unisal")
        o=run_tflite(fp32,x.numpy())
        print(f"tflite vs torch corr {np.corrcoef(o,ref.numpy().ravel())[0,1]:.6f}")
        to_fp16(fp32,f"{HERE}/unisal_fp16.tflite")
        opcheck(f"{HERE}/unisal_fp16.tflite","unisal_fp16")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("CONVERT FAIL:",repr(e)[:200])

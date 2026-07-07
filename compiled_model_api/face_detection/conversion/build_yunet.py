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

import sys
import os
import collections
import numpy as np
import torch
import torch.nn as nn
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,"libfacedetection.train"))
SIZE=int(os.environ.get("YN_SIZE","640"))
VARIANT=os.environ.get("YN_VAR","yunet_n")
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM","MIRROR_PAD"}
class Wrap(nn.Module):
    def __init__(self,m):
        super().__init__()
        self.m=m
    def forward(self,x):
        cls_scores,bbox_preds,objs,kps_preds=self.m(x)
        B=x.shape[0]
        cls=[p.permute(0,2,3,1).reshape(B,-1,1).sigmoid() for p in cls_scores]
        obj=[p.permute(0,2,3,1).reshape(B,-1,1).sigmoid() for p in objs]
        bbox=[p.permute(0,2,3,1).reshape(B,-1,4) for p in bbox_preds]
        kps=[p.permute(0,2,3,1).reshape(B,-1,10) for p in kps_preds]
        return tuple(cls+obj+bbox+kps)
def build():
    from yunet_train.tasks.face import build_yunet
    m=build_yunet(VARIANT)
    sd=torch.load(f"{HERE}/libfacedetection.train/weights/{VARIANT}.pth",map_location="cpu",weights_only=False)
    sd=sd.get("state_dict",sd) if isinstance(sd,dict) and "state_dict" in sd else sd
    sd={k.replace("model.",""):v for k,v in sd.items()} if all(k.startswith(("model.","backbone","neck","bbox_head")) for k in list(sd)[:3]) else sd
    miss,unexp=m.load_state_dict(sd,strict=False)
    m.eval()
    print(f"  loaded {VARIANT}; missing {len(miss)} unexpected {len(unexp)}; params {sum(p.numel() for p in m.parameters())/1e6:.3f}M")
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
    nouts=len(model.subgraphs[0].outputs)
    print(f"[{l}] ops:",dict(sorted(ops.items(),key=lambda kv:-kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB outs:{nouts}","GPU-CLEAN" if not bad and not over else "BLOCKERS")
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
    x=torch.rand(1,3,SIZE,SIZE)*255
    with torch.no_grad(): ys=m(x)
    print("forward outs:",len(ys),"shapes:",[tuple(y.shape) for y in ys[:4]],"...")
    if (sys.argv[1] if len(sys.argv)>1 else "all")=="forward": sys.exit()
    import litert_torch
    fp32=f"{HERE}/yunet.tflite"
    try:
        litert_torch.convert(m,(x,)).export(fp32)
        opcheck(fp32,"yunet")
        to_fp16(fp32,f"{HERE}/yunet_fp16.tflite")
        opcheck(f"{HERE}/yunet_fp16.tflite","yunet_fp16")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("CONVERT FAIL:",repr(e)[:200])

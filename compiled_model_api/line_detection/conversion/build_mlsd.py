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
SIZE=512
sys.path.insert(0, os.path.join(HERE,"mlsd_pytorch"))
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","POW","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM"}
def build():
    from models.mbv2_mlsd_tiny import MobileV2_MLSD_Tiny
    m=MobileV2_MLSD_Tiny().eval()
    sd=torch.load(os.path.join(HERE,"mlsd_pytorch/models/mlsd_tiny_512_fp32.pth"),map_location="cpu",weights_only=False)
    sd=sd.get("model",sd) if isinstance(sd,dict) else sd
    miss,unexp=m.load_state_dict(sd,strict=False)
    print(f"  loaded mlsd_tiny; missing {len(miss)} unexpected {len(unexp)}; params {sum(p.numel() for p in m.parameters())/1e6:.2f}M")
    return m
def opcheck(p,l):
    from ai_edge_litert.interpreter import Interpreter
    it=Interpreter(model_path=p)
    it.allocate_tensors()
    ops=collections.Counter(d.get("op_name","?") for d in it._get_ops_details())
    bad={k:v for k,v in ops.items() if k.upper() in BANNED}
    over=sum(1 for d in it.get_tensor_details() if len(d.get("shape",[]))>4)
    print(f"[{l}] ops:",dict(sorted(ops.items(),key=lambda kv:-kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB","GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it
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
    stage=sys.argv[1] if len(sys.argv)>1 else "all"
    m=build()
    x=torch.rand(1,4,SIZE,SIZE)
    with torch.no_grad(): ref=m(x)
    print(f"forward: out {tuple(ref.shape)} range [{ref.min():.2f},{ref.max():.2f}]")
    if stage=="forward": sys.exit()
    import litert_torch
    fp32=os.path.join(HERE,"mlsd.tflite")
    litert_torch.convert(m,(x,)).export(fp32)
    it=opcheck(fp32,"mlsd")
    d=it.get_input_details()[0]
    it.set_tensor(d["index"],x.numpy().astype(d["dtype"]))
    it.invoke()
    o=it.get_tensor(it.get_output_details()[0]["index"])
    print(f"tflite vs torch corr {np.corrcoef(o.ravel(),ref.numpy().ravel())[0,1]:.6f}")
    if stage=="all":
        to_fp16(fp32,os.path.join(HERE,"mlsd_fp16.tflite"))
        opcheck(os.path.join(HERE,"mlsd_fp16.tflite"),"mlsd_fp16")

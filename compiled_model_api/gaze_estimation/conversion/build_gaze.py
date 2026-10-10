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

"""L2CS-Net (ResNet-50, 90-bin yaw/pitch) -> LiteRT CompiledModel GPU.

Model-side re-authorings (numerically equivalent): stem MaxPool -> zero-pad +
valid max-pool, and the global average pool -> mean(3).mean(2).
"""
import sys
import os
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,"L2CS-Net"))
SIZE=int(os.environ.get("GZ_SIZE","448"))
BANNED={
    "GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT",
    "SELECT_V2","BROADCAST_TO","POW","TRANSPOSE_CONV","CAST",
    "EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT",
    "CUMSUM","MIRROR_PAD"}
class GAP(nn.Module):
    def forward(self,x):
        return x.mean(3,keepdim=True).mean(2,keepdim=True)
class ZeroPadMaxPool(nn.Module):
    def forward(self,x):
        return F.max_pool2d(F.pad(x,(1,1,1,1),value=0.0),
                            kernel_size=3,stride=2,padding=0)
class Wrap(nn.Module):
    def __init__(self,m):
        super().__init__()
        self.m=m
    def forward(self,x):
        y,p=self.m(x)
        # bake softmax; host does expectation
        return F.softmax(y,1), F.softmax(p,1)
def build():
    """Builds L2CS-Net (ResNet-50) and loads the gaze360 checkpoint.

    Returns:
        A Wrap module in eval mode that emits softmaxed 90-bin yaw and
        pitch distributions.
    """
    import torchvision
    import importlib.util as _u
    _sp=_u.spec_from_file_location("l2csm", os.path.join(HERE,"model.py"))
    _mod=_u.module_from_spec(_sp)
    _sp.loader.exec_module(_mod)
    L2CS=_mod.L2CS
    g=L2CS(torchvision.models.resnet.Bottleneck,[3,4,6,3],90)
    sd=torch.load(f"{HERE}/L2CSNet_gaze360.pkl",map_location="cpu",
                  weights_only=False)
    sd=(sd.get("model_state_dict",sd)
        if isinstance(sd,dict) and "model_state_dict" in sd else sd)
    miss,unexp=g.load_state_dict(sd,strict=False)
    g.maxpool=ZeroPadMaxPool()
    g.avgpool=GAP()
    g.eval()
    print(f"  loaded gaze360 resnet50; missing {len(miss)}"
          f" unexpected {len(unexp)};"
          f" params {sum(p.numel() for p in g.parameters())/1e6:.2f}M")
    return Wrap(g).eval()
def opcheck(p,l):
    """Static GPU-compat scan: read the op set straight from the
    .tflite flatbuffer.

    Args:
        p: Path to the .tflite model file to scan.
        l: Label used in the printed report line.
    """
    from ai_edge_litert import schema_py_generated as schema
    with open(p,"rb") as f:
        model=schema.ModelT.InitFromPackedBuf(f.read(),0)
    names={v:k for k,v in vars(schema.BuiltinOperator).items()
           if not k.startswith("_")}
    ops=collections.Counter()
    over=0
    for g in model.subgraphs:
        for op in g.operators:
            c=model.operatorCodes[op.opcodeIndex]
            code=max(c.builtinCode,c.deprecatedBuiltinCode)
            ops[c.customCode.decode() if c.customCode
                else names.get(code,str(code))]+=1
        over+=sum(1 for t in g.tensors
                  if t.shape is not None and len(t.shape)>4)
    bad={k:v for k,v in ops.items() if k.upper() in BANNED}
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over}"
          f" size {os.path.getsize(p)/1e6:.1f}MB",
          "GPU-CLEAN" if not bad and not over else "BLOCKERS")
def run_tflite(p,x):
    """Single inference through the LiteRT CompiledModel API.

    Args:
        p: Path to the .tflite model file.
        x: Input array written to the first input buffer as fp32.

    Returns:
        The first output tensor as a flat fp32 numpy array.
    """
    from ai_edge_litert.compiled_model import CompiledModel
    model=CompiledModel.from_file(p)
    ins=model.create_input_buffers(0)
    outs=model.create_output_buffers(0)
    ins[0].write(np.ascontiguousarray(x,dtype=np.float32))
    model.run_by_index(0,ins,outs)
    n=(model.get_output_buffer_requirements(0,0)["buffer_size"]
       //np.dtype(np.float32).itemsize)
    return outs[0].read(n,np.float32)
def to_fp16(fp32,fp16):
    """Quantizes an fp32 .tflite model to fp16 weights.

    Args:
        fp32: Path to the source fp32 .tflite model.
        fp16: Destination path for the fp16 .tflite model.

    Returns:
        The fp16 destination path.
    """
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm=recipe_manager.RecipeManager()
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(
                num_bits=16,dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT),
        algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16):
        os.remove(fp16)
    q=quantizer.Quantizer(float_model=fp32)
    q.load_quantization_recipe(rm.get_quantization_recipe())
    q.quantize().export_model(fp16)
    return fp16
if __name__=="__main__":
    m=build()
    x=torch.rand(1,3,SIZE,SIZE)
    with torch.no_grad():
        y,p=m(x)
    print(f"forward: yaw {tuple(y.shape)} pitch {tuple(p.shape)}")
    if (sys.argv[1] if len(sys.argv)>1 else "all")=="forward":
        sys.exit()
    import litert_torch
    fp32=f"{HERE}/gaze.tflite"
    litert_torch.convert(m,(x,)).export(fp32)
    opcheck(fp32,"gaze")
    o0=run_tflite(fp32,x.numpy())
    print(f"tflite vs torch yaw corr"
          f" {np.corrcoef(o0,y.numpy().ravel())[0,1]:.6f}")
    to_fp16(fp32,f"{HERE}/gaze_fp16.tflite")
    opcheck(f"{HERE}/gaze_fp16.tflite","gaze_fp16")

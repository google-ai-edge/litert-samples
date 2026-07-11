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

import sys
import os
import collections
import numpy as np
import torch
import torch.nn as nn
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(HERE,"MI-GAN"))
RES=int(os.environ.get("MG_RES","512"))
BANNED={"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE",
        "SELECT", "SELECT_V2", "BROADCAST_TO", "TRANSPOSE_CONV", "CAST",
        "EMBEDDING_LOOKUP", "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT",
        "IRFFT", "CUMSUM", "MIRROR_PAD"}
def build():
    """Builds the MI-GAN generator and loads its checkpoint.

    Returns:
        The MI-GAN generator in eval mode.
    """
    from lib.model_zoo.migan_inference import Generator as MIGANGenerator
    m=MIGANGenerator(resolution=RES)
    sd=torch.load(f"{HERE}/migan_models/migan_{RES}_places2.pt",
                  map_location="cpu", weights_only=False)
    sd=(sd.get("state_dict",sd)
        if isinstance(sd,dict) and "state_dict" in sd else sd)
    miss,unexp=m.load_state_dict(sd, strict=False)
    m.eval()
    n_params=sum(p.numel() for p in m.parameters())/1e6
    print(f"  loaded MIGAN res{RES}; missing {len(miss)} "
          f"unexpected {len(unexp)}; params {n_params:.2f}M")
    return m
def opcheck(p,l):
    """Static GPU-compat scan: reads the op set from the .tflite file.

    Args:
        p: Path to the .tflite model to scan.
        l: Label used to prefix the printed report.
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
            key=(c.customCode.decode() if c.customCode
                 else names.get(code,str(code)))
            ops[key]+=1
        over+=sum(1 for t in g.tensors
                  if t.shape is not None and len(t.shape)>4)
    bad={k:v for k,v in ops.items() if k.upper() in BANNED}
    ordered=dict(sorted(ops.items(),key=lambda kv:-kv[1]))
    print(f"[{l}] ops:",ordered)
    status="GPU-CLEAN" if not bad and not over else "BLOCKERS"
    size=os.path.getsize(p)/1e6
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {size:.1f}MB",
          status)
def run_tflite(p,x):
    """Single inference through the LiteRT CompiledModel API.

    Args:
        p: Path to the .tflite model.
        x: Input array to feed the model.

    Returns:
        The flat fp32 output array.
    """
    from ai_edge_litert.compiled_model import CompiledModel
    model=CompiledModel.from_file(p)
    ins=model.create_input_buffers(0)
    outs=model.create_output_buffers(0)
    ins[0].write(np.ascontiguousarray(x,dtype=np.float32))
    model.run_by_index(0,ins,outs)
    item=np.dtype(np.float32).itemsize
    n=model.get_output_buffer_requirements(0,0)["buffer_size"]//item
    return outs[0].read(n,np.float32)
def to_fp16(fp32,fp16):
    """Quantizes an fp32 .tflite to fp16 via float casting.

    Args:
        fp32: Path to the source fp32 .tflite model.
        fp16: Output path for the fp16 .tflite model.

    Returns:
        The fp16 output path.
    """
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm=recipe_manager.RecipeManager()
    weight_config=qtyping.TensorQuantizationConfig(
        num_bits=16, dtype=qtyping.TensorDataType.FLOAT)
    rm.add_quantization_config(
        regex=".*",
        operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=weight_config,
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
    x=torch.rand(1,4,RES,RES)*2-1
    with torch.no_grad():
        ref=m(x)
    print(f"forward: out {tuple(ref.shape)} "
          f"range [{ref.min():.2f},{ref.max():.2f}]")
    if (sys.argv[1] if len(sys.argv)>1 else "all")=="forward":
        sys.exit()
    import litert_torch
    fp32=f"{HERE}/migan.tflite"
    try:
        litert_torch.convert(m,(x,)).export(fp32)
        opcheck(fp32,"migan")
        o=run_tflite(fp32,x.numpy())
        corr=np.corrcoef(o,ref.numpy().ravel())[0,1]
        print(f"tflite vs torch corr {corr:.6f}")
        to_fp16(fp32,f"{HERE}/migan_fp16.tflite")
        opcheck(f"{HERE}/migan_fp16.tflite","migan_fp16")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("CONVERT FAIL:",repr(e)[:200])

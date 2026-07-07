"""Convert the 3DDFA_V2 MobileNetV1 regressor (120x120x3 -> 62 params) to fp16 tflite.
Run in ~/clipconv from tddfa-work/."""
import os, sys, types
import numpy as np
import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3DDFA_V2")
sys.path.insert(0, REPO); os.chdir(REPO)
_m = types.ModuleType("FaceBoxes.utils.nms.cpu_nms"); _m.cpu_nms = lambda *a, **k: []; _m.cpu_soft_nms = lambda *a, **k: []
sys.modules["FaceBoxes.utils.nms.cpu_nms"] = _m
import yaml
from TDDFA import TDDFA

WORK = os.path.dirname(REPO); OUT = os.path.join(WORK, "out"); os.makedirs(OUT, exist_ok=True)
REF = np.load(os.path.join(WORK, "ref", "ref_emma.npz"))

cfg = yaml.load(open("configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
tddfa = TDDFA(gpu_mode=False, **cfg)
model = tddfa.model.eval()

inp = torch.from_numpy(REF["inp"].astype(np.float32))          # [1,3,120,120]
with torch.no_grad():
    out = model(inp).numpy()[0]                                # 62 (normalized)
# de-normalize to compare to the reference param
param = out * tddfa.param_std + tddfa.param_mean
print("torch param corr vs ref:", float(np.corrcoef(param, REF["param"])[0, 1]),
      "maxdiff", float(np.abs(param - REF["param"]).max()))

import litert_torch, collections
BANNED = {"GATHER_ND","GATHER","SELECT","SELECT_V2","NOT_EQUAL","EQUAL","GREATER","LESS","TOPK_V2","CAST","PACK","SPLIT"}
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
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED or k.startswith("Flex")}
    ok = not bad and not over
    print(f"[{label}] {sum(ops.values())} ops {dict(ops)} banned:{bad or 'NONE'} >4D:{over} -> {'GPU-CLEAN' if ok else 'BLOCKERS'}")
    return ok
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
    qt = quantizer.Quantizer(float_model=fp32); qt.load_quantization_recipe(rm.get_quantization_recipe())
    qt.quantize().export_model(fp16); return fp16
fp32 = os.path.join(OUT, "tddfa_mb1.tflite")
litert_torch.convert(model, (inp,)).export(fp32)
clean = opcheck(fp32, "tddfa fp32")
if clean:
    fp16 = to_fp16(fp32, os.path.join(OUT, "tddfa_mb1_fp16.tflite"))
    opcheck(fp16, "tddfa fp16")
    # tflite parity
    o = run_tflite(fp16, REF["inp"].astype(np.float32))
    p16 = o * tddfa.param_std + tddfa.param_mean
    print("tflite fp16 param corr vs ref:", float(np.corrcoef(p16, REF["param"])[0, 1]),
          "maxdiff", float(np.abs(p16 - REF["param"]).max()))
    # what matters: reconstruct the 68 landmarks from the fp16 params, compare 2D pixel error
    ver16 = tddfa.recon_vers([p16.astype(np.float32)], [REF["roi"]], dense_flag=False)[0]  # [3,68]
    d2d = np.sqrt((ver16[0] - REF["ver"][0]) ** 2 + (ver16[1] - REF["ver"][1]) ** 2)
    print(f"landmark 2D pixel error (img coords, face ~{int(REF['roi'][2]-REF['roi'][0])}px): "
          f"mean {d2d.mean():.2f}px max {d2d.max():.2f}px")
    REF["inp"].astype("<f4").tofile(os.path.join(OUT, "tddfa_input.bin"))
    REF["param"].astype("<f4").tofile(os.path.join(OUT, "tddfa_param_ref.bin"))
    # host recon assets as raw f32 bins for Kotlin
    ra = np.load(os.path.join(WORK, "ref", "recon_assets.npz"))
    for k in ["u_base", "w_shp_base", "w_exp_base", "param_mean", "param_std"]:
        ra[k].astype("<f4").tofile(os.path.join(OUT, f"tddfa_{k}.bin"))
    print("wrote fp16 + fixtures + recon assets")

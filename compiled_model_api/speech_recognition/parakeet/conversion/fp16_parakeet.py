# fp16 quantize + op-check + desktop-fp16 parity (no NeMo: only ai_edge_litert/quantizer + numpy).
import os, collections, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FP32 = os.path.join(HERE, "parakeet_enc.tflite")
FP16 = os.path.join(HERE, "parakeet_enc_fp16.tflite")
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP",
          "RFFT2D", "FFT", "STFT", "COMPLEX", "RFFT", "IRFFT", "CUMSUM"}


def opcheck(path, label):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    print(f"[{label}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{label}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(path)/1e6:.1f}MB",
          "GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it


def to_fp16(fp32, fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm = recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
        op_config=qtyping.OpQuantizationConfig(
            weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
            compute_precision=qtyping.ComputePrecision.FLOAT), algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    q = quantizer.Quantizer(float_model=fp32); q.load_quantization_recipe(rm.get_quantization_recipe())
    q.quantize().export_model(fp16); return fp16


def run(path, mel):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors()
    di = it.get_input_details()[0]
    it.set_tensor(di["index"], mel.astype(di["dtype"])); it.invoke()
    return it.get_tensor(it.get_output_details()[0]["index"])


mel = np.load(os.path.join(HERE, "ref_mel.npy"))
ref_logp = np.load(os.path.join(HERE, "ref_logp.npy"))
ref_tok = ref_logp[0].argmax(-1)

opcheck(FP32, "fp32")
o32 = run(FP32, mel)
lp32 = o32 - o32.max(-1, keepdims=True)
lp32 = lp32 - np.log(np.exp(lp32).sum(-1, keepdims=True))
print("fp32 tflite: logp-corr %.5f tok %d/%d" % (
    np.corrcoef(lp32.ravel(), ref_logp.ravel())[0, 1],
    int((o32[0].argmax(-1) == ref_tok).sum()), len(ref_tok)))

to_fp16(FP32, FP16)
opcheck(FP16, "fp16")
o16 = run(FP16, mel)
lp16 = o16 - o16.max(-1, keepdims=True)
lp16 = lp16 - np.log(np.exp(lp16).sum(-1, keepdims=True))
print("fp16 tflite: logp-corr %.5f tok %d/%d  out absmax %.2f" % (
    np.corrcoef(lp16.ravel(), ref_logp.ravel())[0, 1],
    int((o16[0].argmax(-1) == ref_tok).sum()), len(ref_tok), float(np.abs(o16).max())))

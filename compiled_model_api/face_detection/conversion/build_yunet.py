import sys, os, collections
import numpy as np, torch, torch.nn as nn
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(HERE,"libfacedetection.train"))
SIZE=int(os.environ.get("YN_SIZE","640")); VARIANT=os.environ.get("YN_VAR","yunet_n")
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","RFFT","IRFFT","CUMSUM","MIRROR_PAD"}
class Wrap(nn.Module):
    def __init__(s,m): super().__init__(); s.m=m
    def forward(s,x):
        cls_scores,bbox_preds,objs,kps_preds=s.m(x); B=x.shape[0]
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
    miss,unexp=m.load_state_dict(sd,strict=False); m.eval()
    print(f"  loaded {VARIANT}; missing {len(miss)} unexpected {len(unexp)}; params {sum(p.numel() for p in m.parameters())/1e6:.3f}M")
    return Wrap(m).eval()
def opcheck(p,l):
    from ai_edge_litert.interpreter import Interpreter
    it=Interpreter(model_path=p); it.allocate_tensors()
    ops=collections.Counter(d.get("op_name","?") for d in it._get_ops_details())
    bad={k:v for k,v in ops.items() if k.upper() in BANNED}; over=sum(1 for d in it.get_tensor_details() if len(d.get("shape",[]))>4)
    print(f"[{l}] ops:",dict(sorted(ops.items(),key=lambda kv:-kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB outs:{len(it.get_output_details())}","GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it
def to_fp16(fp32,fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm=recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*",operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,op_config=qtyping.OpQuantizationConfig(weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16,dtype=qtyping.TensorDataType.FLOAT),compute_precision=qtyping.ComputePrecision.FLOAT),algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    q=quantizer.Quantizer(float_model=fp32); q.load_quantization_recipe(rm.get_quantization_recipe()); q.quantize().export_model(fp16); return fp16
if __name__=="__main__":
    m=build(); x=torch.rand(1,3,SIZE,SIZE)*255
    with torch.no_grad(): ys=m(x)
    print("forward outs:",len(ys),"shapes:",[tuple(y.shape) for y in ys[:4]],"...")
    if (sys.argv[1] if len(sys.argv)>1 else "all")=="forward": sys.exit()
    import litert_torch
    fp32=f"{HERE}/yunet.tflite"
    try:
        litert_torch.convert(m,(x,)).export(fp32); it=opcheck(fp32,"yunet")
        to_fp16(fp32,f"{HERE}/yunet_fp16.tflite"); opcheck(f"{HERE}/yunet_fp16.tflite","yunet_fp16")
    except Exception as e: import traceback; traceback.print_exc(); print("CONVERT FAIL:",repr(e)[:200])

import sys, os, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
R=int(os.environ.get("RF_RES","384")); HERE=os.path.dirname(os.path.abspath(__file__))
BANNED={"GATHER","GATHER_ND","TOPK_V2","GELU","ERF","WHERE","SELECT","SELECT_V2","BROADCAST_TO","POW","TRANSPOSE_CONV","CAST","EMBEDDING_LOOKUP","RFFT2D","FFT","STFT","COMPLEX","CUMSUM","MIRROR_PAD"}
from rfdetr.models.backbone import dinov2_with_windowed_attn as D
def sdpa_manual(self, hidden_states, output_attentions=False):
    q=self.transpose_for_scores(self.query(hidden_states))
    k=self.transpose_for_scores(self.key(hidden_states))
    v=self.transpose_for_scores(self.value(hidden_states))
    scale=1.0/(q.shape[-1]**0.5)
    s=torch.matmul(q,k.transpose(-1,-2))*scale
    a=torch.softmax(s,dim=-1)
    c=torch.matmul(a,v).permute(0,2,1,3).contiguous()
    c=c.view(c.size()[:-2]+(self.all_head_size,))
    return c,None
D.Dinov2WithRegistersSdpaSelfAttention.forward=sdpa_manual
def _win_part(t, B, nP, nWin):     # [B, nP*nP, C] -> [B*nWin^2, (nP/nWin)^2, C], all <=4D
    wp=nP//nWin; C=t.shape[-1]
    x=t.view(B,nP,nP,C)
    x=x.reshape(B*nWin,wp,nP,C)     # split h
    x=x.permute(0,2,1,3)            # [B*nWin,nP,wp,C]
    x=x.reshape(B*nWin*nWin,wp,wp,C)# split w
    x=x.permute(0,2,1,3)            # -> (h%wp, w%wp)
    return x.reshape(B*nWin*nWin,wp*wp,C)
def _emb_forward(self, pixel_values, bool_masked_pos=None):
    B=pixel_values.shape[0]; H=pixel_values.shape[2]; W=pixel_values.shape[3]
    emb=self.patch_embeddings(pixel_values)
    emb=torch.cat((self.cls_token.expand(B,-1,-1),emb),dim=1)
    emb=emb+self.interpolate_pos_encoding(emb,H,W)
    if self.config.num_windows>1:
        nP=H//self.config.patch_size; nWin=self.config.num_windows
        cls_t=emb[:,:1]; pix=emb[:,1:]
        pix=_win_part(pix,B,nP,nWin)
        emb=torch.cat((torch.cat([cls_t]*(nWin**2),dim=0),pix),dim=1)  # repeat->cat (no BROADCAST_TO)
    if self.config.num_register_tokens>0:
        emb=torch.cat((emb[:,:1],self.register_tokens.expand(emb.shape[0],-1,-1),emb[:,1:]),dim=1)
    return self.dropout(emb)
D.WindowedDinov2WithRegistersEmbeddings.forward=_emb_forward
def _win_reverse(t, B, nP, nWin):  # inverse of _win_part: [B*nWin^2, wp^2, C] -> [B, nP, nP, C], all <=4D
    wp=nP//nWin; C=t.shape[-1]
    x=t.reshape(B*nWin*nWin,wp,wp,C)
    x=x.permute(0,2,1,3)
    x=x.reshape(B*nWin,nP,wp,C)
    x=x.permute(0,2,1,3)
    return x.reshape(B,nP,nP,C)
from transformers.modeling_outputs import BackboneOutput as _BBOut
def _bb_forward(self, pixel_values, output_hidden_states=None, output_attentions=None, return_dict=None):
    emb=self.embeddings(pixel_values)
    outs=self.encoder(emb, output_hidden_states=True, output_attentions=False, return_dict=True)
    hidden_states=outs.hidden_states
    B=pixel_values.shape[0]; nP=pixel_values.shape[2]//self.config.patch_size; nWin=self.config.num_windows
    fmaps=()
    for stage, hs in zip(self.stage_names, hidden_states):
        if stage in self.out_features:
            if self.config.apply_layernorm: hs=self.layernorm(hs)
            if self.config.reshape_hidden_states:
                hs=hs[:, self.num_register_tokens+1:]
                hs=_win_reverse(hs,B,nP,nWin) if nWin>1 else hs.reshape(B,nP,nP,-1)
                hs=hs.permute(0,3,1,2).contiguous()
            fmaps+=(hs,)
    return _BBOut(feature_maps=fmaps)
D.WindowedDinov2WithRegistersBackbone.forward=_bb_forward


_TF=F.gelu
def _tg(x,*a,**k): return 0.5*x*(1.0+torch.tanh(0.7978845608*(x+0.044715*x*x*x)))
F.gelu=_tg
class TG(nn.Module):
    def forward(s,x): return _tg(x)
def build():
    from rfdetr import RFDETRNano
    m=RFDETRNano(); net=m.model.model.eval()
    bb=None
    for mod in net.modules():
        if hasattr(mod,"encoder") and hasattr(getattr(mod,"encoder"),"layer") and hasattr(mod,"embeddings"): bb=mod; break
    assert bb is not None, "backbone model not found"
    emb_mod=bb.embeddings; C=emb_mod.cls_token.shape[-1]; N=(R//16)**2+1
    _pos=emb_mod.interpolate_pos_encoding(torch.zeros(1,N,C),R,R).detach()
    emb_mod.interpolate_pos_encoding=lambda e,h,w,_p=_pos: _p     # baked (fixed res)
    for mod in bb.modules():
        for cn,ch in list(mod.named_children()):
            if isinstance(ch,nn.GELU) or type(ch).__name__ in ("GELUActivation","QuickGELUActivation"): setattr(mod,cn,TG())
    class BBWrap(nn.Module):
        def __init__(s): super().__init__(); s.bb=bb
        def forward(s,x):
            out=s.bb(pixel_values=x)
            fm=out.feature_maps if hasattr(out,"feature_maps") else (out[0] if isinstance(out,tuple) else out)
            return fm[-1] if isinstance(fm,(list,tuple)) else fm
    print(f"  backbone {type(bb).__name__}; layers {len(bb.encoder.layer)}; params {sum(p.numel() for p in bb.parameters())/1e6:.1f}M")
    return BBWrap().eval()
def opcheck(p,l):
    from ai_edge_litert.interpreter import Interpreter
    it=Interpreter(model_path=p); it.allocate_tensors()
    ops=collections.Counter(d.get("op_name","?") for d in it._get_ops_details())
    bad={k:v for k,v in ops.items() if k.upper() in BANNED}; over=sum(1 for d in it.get_tensor_details() if len(d.get("shape",[]))>4)
    print(f"[{l}] ops:",dict(sorted(ops.items(),key=lambda kv:-kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB","GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it
def to_fp16(fp32,fp16):
    from ai_edge_quantizer import quantizer, recipe_manager
    from ai_edge_quantizer.recipe import AlgorithmName, qtyping
    rm=recipe_manager.RecipeManager()
    rm.add_quantization_config(regex=".*",operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,op_config=qtyping.OpQuantizationConfig(weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16,dtype=qtyping.TensorDataType.FLOAT),compute_precision=qtyping.ComputePrecision.FLOAT),algorithm_key=AlgorithmName.FLOAT_CASTING)
    if os.path.exists(fp16): os.remove(fp16)
    q=quantizer.Quantizer(float_model=fp32); q.load_quantization_recipe(rm.get_quantization_recipe()); q.quantize().export_model(fp16); return fp16
if __name__=="__main__":
    g=build(); x=torch.randn(1,3,R,R)*0.5
    with torch.no_grad(): ref=g(x)
    print(f"backbone out {tuple(ref.shape)} range [{ref.min():.1f},{ref.max():.1f}] max|x| {ref.abs().max():.1f}")
    if (sys.argv[1] if len(sys.argv)>1 else "all")=="forward": sys.exit()
    import litert_torch
    fp32=f"{HERE}/rfbb.tflite"
    try:
        litert_torch.convert(g,(x,)).export(fp32); it=opcheck(fp32,"rfbb")
        d=it.get_input_details()[0]; it.set_tensor(d["index"],x.numpy().astype(d["dtype"])); it.invoke()
        o=it.get_tensor(it.get_output_details()[0]["index"])
        print(f"tflite vs torch corr {np.corrcoef(o.ravel(),ref.numpy().ravel())[0,1]:.6f}")
        to_fp16(fp32,f"{HERE}/rfbb_fp16.tflite"); opcheck(f"{HERE}/rfbb_fp16.tflite","rfbb_fp16")
        np.save(f"{HERE}/rfbb_ref.npy",ref.numpy()); x.numpy().astype(np.float32).tofile(f"{HERE}/rfbb_in.bin")
    except Exception as e: import traceback; traceback.print_exc(); print("CONVERT FAIL:",repr(e)[:300])

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
import torch.nn.functional as F
import build_rfdetr_bb as B            # applies backbone class patches (SDPA, windowing, win_reverse) on import
torch._shape_as_tensor=lambda t: torch.tensor(list(t.shape),dtype=torch.long)   # private ATen op untraceable -> constant (fixed res)
torch._assert=lambda *a,**k: None   # data-dependent sanity asserts (H*W==Len) -> no-op for export
R=int(os.environ.get("RF_RES","384"))
HERE=os.path.dirname(os.path.abspath(__file__))
BANNED=B.BANNED
# grid_sample -> rfdetr's manual bilinear (export-designed); check op-clean in the op-check
from rfdetr.utilities import tensors as _T
def _gs(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
    # GATHER/CAST-free bilinear (tent weights + BMM)
    N,C,H,W=input.shape
    Hg,Wg=grid.shape[1],grid.shape[2]
    ac=bool(align_corners)
    if ac:
        ix=(grid[...,0]+1)*(W-1)/2
        iy=(grid[...,1]+1)*(H-1)/2
    else:
        ix=(grid[...,0]+1)*W/2-0.5
        iy=(grid[...,1]+1)*H/2-0.5
    ix=ix.reshape(N,Hg*Wg,1)
    iy=iy.reshape(N,Hg*Wg,1)
    xs=torch.arange(W,dtype=input.dtype).reshape(1,1,W)
    ys=torch.arange(H,dtype=input.dtype).reshape(1,1,H)
    wx=torch.relu(1-(ix-xs).abs())
    wy=torch.relu(1-(iy-ys).abs())
    Wm=(wy.unsqueeze(-1)*wx.unsqueeze(-2)).reshape(N,Hg*Wg,H*W)
    return torch.matmul(input.reshape(N,C,H*W), Wm.transpose(1,2)).reshape(N,C,Hg,Wg)
F.grid_sample=_gs
_T._bilinear_grid_sample=lambda input,grid,padding_mode="zeros",align_corners=False: _gs(input,grid,padding_mode=padding_mode,align_corners=align_corners)

# bake the deformable level shapes (Python ints) so the value.split sizes are static (export-friendly)
_LEVEL_SHAPES=[]
def _core_patched(value, value_spatial_shapes, sampling_locations, attention_weights, value_spatial_shapes_hw=None):
    if not _LEVEL_SHAPES:
        try: _LEVEL_SHAPES.extend([(int(h),int(w)) for h,w in value_spatial_shapes.tolist()])
        except Exception: pass
    shapes=value_spatial_shapes_hw if value_spatial_shapes_hw is not None else _LEVEL_SHAPES
    bs,n_heads,head_dim,_=value.shape
    _,len_q,_,n_levels,n_points,_=sampling_locations.shape
    value_list=[value] if len(shapes)<=1 else list(value.split([h*w for h,w in shapes],dim=3))
    grids=2*sampling_locations-1
    svl=[]
    for li,(h,w) in enumerate(shapes):
        vl=value_list[li].reshape(bs*n_heads,head_dim,h,w)
        gl=grids[:,:,:,li].transpose(1,2).flatten(0,1)
        svl.append(_gs(vl,gl,padding_mode="zeros",align_corners=False))
    aw=attention_weights.transpose(1,2).reshape(bs*n_heads,1,len_q,n_levels*n_points)
    sv=torch.stack(svl,dim=-2).flatten(-2)
    return (sv*aw).sum(-1).view(bs,n_heads*head_dim,len_q).transpose(1,2).contiguous()
import rfdetr.models.ops.modules.ms_deform_attn as _MOD
_MOD.ms_deform_attn_core_pytorch=_core_patched
# re-author MSDeformAttn.forward to stay <=4D (n_levels=1): no 6D sampling tensors

# sine pos-embed: bake dim_t (no POW/FLOOR_DIV in graph) + reshape interleave (no strided-slice GATHER_ND)
import math as _math
import rfdetr.models.transformer as _TR
_DIMT={}
def _gen_sine(pos_tensor, dim=128):
    scale=2*_math.pi
    if dim not in _DIMT:
        dt=torch.arange(dim,dtype=torch.float32)
        _DIMT[dim]=(10000.0**(2*(dt//2)/dim)).detach()
    dim_t=_DIMT[dim]
    def il(emb):
        p=emb[:,:,None]*scale/dim_t
        pr=p.reshape(p.shape[0],p.shape[1],dim//2,2)
        return torch.stack((pr[...,0].sin(),pr[...,1].cos()),-1).flatten(2)
    pos_x=il(pos_tensor[:,:,0])
    pos_y=il(pos_tensor[:,:,1])
    if pos_tensor.size(-1)==2: return torch.cat((pos_y,pos_x),dim=2)
    return torch.cat((pos_y,pos_x,il(pos_tensor[:,:,2]),il(pos_tensor[:,:,3])),dim=2)
_TR.gen_sineembed_for_position=_gen_sine

import rfdetr.models.ops.modules.ms_deform_attn as _MSMOD
def _msda_forward(self, query, reference_points, input_flatten, input_spatial_shapes,
                  input_level_start_index, input_padding_mask=None, **kw):
    if not _LEVEL_SHAPES:
        try: _LEVEL_SHAPES.extend([(int(h),int(w)) for h,w in input_spatial_shapes.tolist()])
        except Exception: pass
    bs=query.shape[0]
    len_q=query.shape[1]
    nh=self.n_heads
    npnt=self.n_points
    dm=self.d_model
    hd=dm//nh
    value=self.value_proj(input_flatten)                         # [bs, Lin, dm]
    if input_padding_mask is not None:
        value=value.masked_fill(input_padding_mask[...,None], 0.0)
    so=self.sampling_offsets(query).view(bs,len_q,nh,npnt*2).permute(0,2,1,3).reshape(bs*nh,len_q,npnt,2)
    aw=self.attention_weights(query).view(bs,len_q,nh,npnt)
    aw=torch.softmax(aw,-1).permute(0,2,1,3).reshape(bs*nh,1,len_q,npnt)
    H,W=_LEVEL_SHAPES[0]
    ref=reference_points[:,:,0,:]                                # squeeze n_levels=1 -> [bs,len_q,2 or 4]
    rxy=ref[...,:2].unsqueeze(1).repeat(1,nh,1,1).reshape(bs*nh,len_q,1,2)
    if ref.shape[-1]==4:
        rwh=ref[...,2:].unsqueeze(1).repeat(1,nh,1,1).reshape(bs*nh,len_q,1,2)
        loc=rxy + so/npnt*rwh*0.5
    else:
        norm=torch.tensor([W,H],dtype=value.dtype).reshape(1,1,1,2)
        loc=rxy + so/norm
    val=value.transpose(1,2).reshape(bs*nh,hd,H,W)
    sampled=_gs(val, 2*loc-1, padding_mode="zeros", align_corners=False)   # [bs*nh,hd,len_q,npnt]
    out=(sampled*aw).sum(-1).reshape(bs,dm,len_q).transpose(1,2)            # [bs,len_q,dm]
    return self.output_proj(out)
_MSMOD.MSDeformAttn.forward=_msda_forward


class TG(nn.Module):
    def forward(self,x):
        return 0.5*x*(1.0+torch.tanh(0.7978845608*(x+0.044715*x*x*x)))
def build():
    from rfdetr import RFDETRNano
    m=RFDETRNano()
    net=m.model.model.eval()
    net.export()   # re-wire backbone for plain-tensor input + submodule export paths
    # backbone gelu + pos bake (instance-level, mirror the bb build)
    bb=None
    for mod in net.modules():
        if hasattr(mod,"encoder") and hasattr(getattr(mod,"encoder"),"layer") and hasattr(mod,"embeddings"):
            bb=mod
            break
    emb=bb.embeddings
    C=emb.cls_token.shape[-1]
    N=(R//16)**2+1
    _pos=emb.interpolate_pos_encoding(torch.zeros(1,N,C),R,R).detach()
    emb.interpolate_pos_encoding=lambda e,h,w,_p=_pos: _p
    for mod in net.modules():
        for cn,ch in list(mod.named_children()):
            if isinstance(ch,nn.GELU) or type(ch).__name__ in ("GELUActivation","QuickGELUActivation"): setattr(mod,cn,TG())
    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=net
        def forward(self,x):
            o=self.net.forward_export(x)
            if isinstance(o,dict): return o.get("pred_logits"), o.get("pred_boxes")
            if isinstance(o,(list,tuple)): return tuple(t for t in o if torch.is_tensor(t))[:2]
            return o
    print(f"  RFDETR full; net params {sum(p.numel() for p in net.parameters())/1e6:.1f}M")
    return Wrap().eval()
def opcheck(p,l):
    """Static GPU-compat scan: read the op set straight from the .tflite flatbuffer."""
    from ai_edge_litert import schema_py_generated as schema
    with open(p,"rb") as f:
        model=schema.ModelT.InitFromPackedBuf(f.read(),0)
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
def run_tflite(path,x):
    """Single-input inference through the LiteRT CompiledModel API; returns the flat fp32 outputs in signature order."""
    from ai_edge_litert.compiled_model import CompiledModel
    model=CompiledModel.from_file(path)
    inputs=model.create_input_buffers(0)
    outputs=model.create_output_buffers(0)
    inputs[0].write(np.ascontiguousarray(x,dtype=np.float32))
    model.run_by_index(0,inputs,outputs)
    outs=[]
    for j in range(len(outputs)):
        n=model.get_output_buffer_requirements(j,0)["buffer_size"]//np.dtype(np.float32).itemsize
        outs.append(outputs[j].read(n,np.float32))
    return outs
if __name__=="__main__":
    g=build()
    x=torch.randn(1,3,R,R)*0.5
    with torch.no_grad(): ref=g(x)
    print("forward out:", [tuple(t.shape) for t in (ref if isinstance(ref,(list,tuple)) else [ref]) if torch.is_tensor(t)])
    if (sys.argv[1] if len(sys.argv)>1 else "all")=="forward": sys.exit()
    import litert_torch
    fp32=f"{HERE}/rffull.tflite"
    try:
        litert_torch.convert(g,(x,)).export(fp32)
        opcheck(fp32,"rffull")
        for oi,o in enumerate(run_tflite(fp32,x.numpy())):
            r=ref[oi].numpy() if isinstance(ref,(list,tuple)) else ref.numpy()
            if o.size==r.size: print(f"  out{oi} corr {np.corrcoef(o,r.ravel())[0,1]:.5f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("CONVERT FAIL:",repr(e)[:400])

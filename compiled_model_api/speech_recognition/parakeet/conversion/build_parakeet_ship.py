# Shippable: FIXED mel window + padding, masking re-added GPU-clean (additive attn bias + conv frame-mask).
# Inputs: mel [1,80,MEL_LEN] (zero-padded), frame_mask [1,T'] (1.0 real / 0.0 pad at subsampled res).
# os._exit at end (NeMo<->litert teardown). MEL_LEN via env (default pad the real 744-sample to 1001).
import _stub, os, torch, numpy as np
import litert_torch
MEL_LEN = int(os.environ.get("MEL_LEN", "1001"))

d = torch.load("parakeet_modules.pt", map_location="cpu", weights_only=False)
enc, ctc = d["encoder"].eval(), d["ctc"].eval()
from nemo.collections.asr.parts.submodules import subsampling as SUB
from nemo.collections.asr.parts.submodules import multi_head_attention as MHA
from nemo.collections.asr.parts.submodules import conformer_modules as CM
from nemo.collections.asr.modules import conformer_encoder as CE

def _safe_ln(self, x):
    amax = x.abs().amax(-1, keepdim=True); S = (amax*(1.0/8.0)).clamp(min=1.0)
    xs = x/S; mu = xs.mean(-1, keepdim=True); dd = xs-mu
    var = (dd*dd).mean(-1, keepdim=True)
    y = dd*torch.rsqrt(var+self.eps)
    return y*self.weight+self.bias if self.elementwise_affine else y
torch.nn.LayerNorm.forward = _safe_ln

def _conv_sub(self, x, lengths):
    x = x.unsqueeze(1)
    for layer in self.conv: x = layer(x)
    b,c,t,f = x.size(); x = self.out(x.transpose(1,2).reshape(b,t,-1))
    return x, SUB.calc_length(lengths, all_paddings=self._left_padding+self._right_padding,
        kernel_size=self._kernel_size, stride=self._stride, ceil_mode=self._ceil_mode, repeat_num=self._sampling_num)
SUB.ConvSubsampling.forward = _conv_sub
CE.ConformerEncoder._create_masks = lambda self,*a,**k:(None,None)

M = {}   # holder; Wrap.forward fills attn_bias [1,1,1,T'] and frame_mask [1,1,T']
def _rel_mha(self, query, key, value, mask, pos_emb, cache=None):
    q,k,v = self.forward_qkv(query,key,value); q = q.transpose(1,2)
    p = self.linear_pos(pos_emb).view(pos_emb.size(0),-1,self.h,self.d_k).transpose(1,2)
    q_u = (q+self.pos_bias_u).transpose(1,2); q_v = (q+self.pos_bias_v).transpose(1,2)
    ac = torch.matmul(q_u,k.transpose(-2,-1)); bd = self.rel_shift(torch.matmul(q_v,p.transpose(-2,-1)))
    bd = bd[:,:,:,:ac.size(-1)]; scores = (ac+bd)/self.s_d_k + M['attn_bias']
    x = torch.matmul(torch.softmax(scores,dim=-1), v)
    return self.linear_out(x.transpose(1,2).reshape(value.size(0),-1,self.h*self.d_k))
MHA.RelPositionMultiHeadAttention.forward = _rel_mha

def _conv_mod(self, x, pad_mask=None, cache=None):
    x = x.transpose(1,2) * M['frame_mask']          # zero padded frames before convs
    x = self.pointwise_conv1(x)
    x = x[:,:self.d_model,:]*torch.sigmoid(x[:,self.d_model:,:]) if self.pointwise_activation=='glu_' else self.pointwise_activation(x)
    x = x * M['frame_mask']                          # re-zero before depthwise (kernel-9 leak guard)
    x = self.depthwise_conv(x)
    x = self.batch_norm(x) if self.norm_type!="layer_norm" else self.batch_norm(x.transpose(1,2)).transpose(1,2)
    return self.pointwise_conv2(self.activation(x)).transpose(1,2)
CM.ConformerConvolution.forward = _conv_mod

if hasattr(enc.pre_encode,"subsampling_conv_chunking_factor"): enc.pre_encode.subsampling_conv_chunking_factor=-1
for m in enc.modules():
    if hasattr(m,"use_pytorch_sdpa"): m.use_pytorch_sdpa=False

# subsampled length helper (T' for a given mel length)
def tprime(mel_len):
    pe = enc.pre_encode
    return int(SUB.calc_length(torch.tensor([mel_len]), all_paddings=pe._left_padding+pe._right_padding,
        kernel_size=pe._kernel_size, stride=pe._stride, ceil_mode=pe._ceil_mode, repeat_num=pe._sampling_num)[0])
TP = tprime(MEL_LEN)

class Wrap(torch.nn.Module):
    def __init__(s): super().__init__(); s.enc=enc; s.ctc=ctc
    def forward(s, mel, frame_mask):                 # mel [1,80,MEL_LEN], frame_mask [1,T']
        M['attn_bias'] = ((1.0-frame_mask).view(1,1,1,-1))*(-30000.0)
        M['frame_mask'] = frame_mask.view(1,1,-1)
        eo,_ = s.enc(audio_signal=mel, length=torch.full((1,), mel.shape[2], dtype=torch.long))
        return s.ctc.decoder_layers(eo).transpose(1,2)
w = Wrap().eval()

# build padded real-speech test input + mask
mel0 = np.load("ref_mel.npy"); RL = mel0.shape[2]                  # 744
mel = np.zeros((1,80,MEL_LEN), np.float32); mel[:,:,:RL] = mel0
TR = tprime(RL)                                                    # real subsampled len (93)
fmask = np.zeros((1,TP), np.float32); fmask[:,:TR] = 1.0
print(f"MEL_LEN={MEL_LEN} T'={TP} real T'={TR}", flush=True)
mt, ft = torch.from_numpy(mel), torch.from_numpy(fmask)
with torch.no_grad(): out = w(mt, ft).numpy()
np.save("ship_ref_logits.npy", out); np.save("ship_mel.npy", mel); np.save("ship_mask.npy", fmask)
np.save("ship_TR.npy", np.array([TR]))
print("CPU forward done; out", out.shape, "(decode/verify in ship_verify.py)", flush=True)

litert_torch.convert(w, (mt, ft)).export("parakeet_ship.tflite")
print("CONVERT OK", os.path.getsize("parakeet_ship.tflite"), flush=True)
os._exit(0)

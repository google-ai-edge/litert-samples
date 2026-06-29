# RT-DETRv2-S (r18vd) 2-graph split for CompiledModel GPU.
# ⚠ DEVICE-CORRECT ENTRY POINT = build_rtdetr_fix3.py (imports this as the patch/GraphB library).
#   The GraphA below is the ORIGINAL multi-output version; on device it hits the Mali 3D-sequence fanout
#   bug (memory_raw/output_memory/enc_coord corrupt -> cats lost). fix3 emits only enc_class+memory_raw*2
#   and moves enc_output+bbox_head to the host. See reference_mali_3d_sequence_fanout_bug in memory.
#   Graph A (GPU) = ResNet18-vd backbone + hybrid encoder + enc heads
#                   -> enc_class[1,N,80], enc_coord[1,N,4], output_memory[1,N,256], memory[1,N,256]
#   host = topk-300 by max class score -> gather coords(ref) + output_memory(target)
#   Graph B (GPU) = plain deformable DETR decoder (3 layers, direct box regression) -> logits, boxes
# RT-DETRv2 = D-FINE minus FDR/LQE (RF-DETR-like decoder -> fp16-friendlier). Re-authoring transfers from
# dfine: grid_sample->tent-matmul, multi-level deformable <=4D, SafeLayerNorm, tanh-GELU, inverse_sigmoid
# clamp(0,1) drop, anchors finfo.max->fp16 clamp, AIFI sine-embed bake. Two-stage topk/gather->host.
import sys, os, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import transformers.models.rt_detr_v2.modeling_rt_detr_v2 as M

torch._shape_as_tensor = lambda t: torch.tensor(list(t.shape), dtype=torch.long)
torch._assert = lambda *a, **k: None
R = int(os.environ.get("RT_RES", "640")); HERE = os.path.dirname(os.path.abspath(__file__))
MID = "PekingU/rtdetr_v2_r18vd"
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2", "BROADCAST_TO",
          "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP", "RFFT2D", "FFT", "STFT", "COMPLEX",
          "CUMSUM", "ARG_MAX", "ARG_MIN", "PACK", "UNPACK", "TILE", "RELU_0_TO_1", "PADV2"}
NQ = 300; NCLS = 80; HID = 256

def _gs(input, grid, mode="bilinear", padding_mode="zeros", align_corners=None):
    N, C, H, W = input.shape; Hg, Wg = grid.shape[1], grid.shape[2]; ac = bool(align_corners)
    if ac: ix = (grid[..., 0] + 1) * (W - 1) / 2; iy = (grid[..., 1] + 1) * (H - 1) / 2
    else:  ix = (grid[..., 0] + 1) * W / 2 - 0.5; iy = (grid[..., 1] + 1) * H / 2 - 0.5
    ix = ix.reshape(N, Hg * Wg, 1); iy = iy.reshape(N, Hg * Wg, 1)
    xs = torch.arange(W, dtype=input.dtype).reshape(1, 1, W); ys = torch.arange(H, dtype=input.dtype).reshape(1, 1, H)
    wx = torch.relu(1 - (ix - xs).abs()); wy = torch.relu(1 - (iy - ys).abs())
    Wm = (wy.unsqueeze(-1) * wx.unsqueeze(-2)).reshape(N, Hg * Wg, H * W)
    return torch.matmul(input.reshape(N, C, H * W), Wm.transpose(1, 2)).reshape(N, C, Hg, Wg)
F.grid_sample = _gs

def _tg(x, *a, **k): return 0.5 * x * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))
F.gelu = _tg
def _safe_ln(self, x):
    S = (x.abs().amax(-1, keepdim=True) * (1.0 / 8.0)).clamp(min=1.0)
    xs = x / S; mu = xs.mean(-1, keepdim=True); d = xs - mu
    var = (d * d).mean(-1, keepdim=True) * (S * S); d = d * S
    y = d * torch.rsqrt(var + self.eps)
    if self.elementwise_affine: y = y * self.weight + self.bias
    return y
nn.LayerNorm.forward = _safe_ln

def _msda_core(value, spatial_shapes_list, sampling_locations, attention_weights, num_points_list):
    bs, seq, nh, hd = value.shape
    nq = sampling_locations.shape[1]
    val_levels = value.split([h * w for h, w in spatial_shapes_list], dim=1)
    grids = (2 * sampling_locations - 1).split(num_points_list, dim=2)
    sampled = []
    for li, (h, w) in enumerate(spatial_shapes_list):
        vl = val_levels[li].permute(0, 2, 3, 1).reshape(bs * nh, hd, h, w)
        sampled.append(_gs(vl, grids[li], padding_mode="zeros", align_corners=False))
    sv = torch.cat(sampled, dim=-1)
    out = (sv * attention_weights).sum(-1)
    return out.reshape(bs, nh * hd, nq).transpose(1, 2).contiguous()

def _msda_forward(self, hidden_states, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                  position_embeddings=None, reference_points=None, spatial_shapes=None, spatial_shapes_list=None,
                  level_start_index=None, **kw):
    if position_embeddings is not None: hidden_states = hidden_states + position_embeddings
    bs, nq, _ = hidden_states.shape
    _, seq, _ = encoder_hidden_states.shape
    nh = self.n_heads; dm = self.d_model; hd = dm // nh; P = sum(self.n_points_list)
    value = self.value_proj(encoder_hidden_states).view(bs, seq, nh, hd)
    so = self.sampling_offsets(hidden_states).reshape(bs, nq, nh, P * 2).permute(0, 2, 1, 3).reshape(bs * nh, nq, P, 2)
    aw = self.attention_weights(hidden_states).reshape(bs, nq, nh, P)
    aw = F.softmax(aw, -1).permute(0, 2, 1, 3).reshape(bs * nh, nq, P).unsqueeze(1)
    ref = reference_points[:, :, 0, :]
    nps = self.n_points_scale.to(hidden_states.dtype).reshape(1, 1, P, 1)
    rxy = ref[..., :2].unsqueeze(1).repeat(1, nh, 1, 1).reshape(bs * nh, nq, 1, 2)
    rwh = ref[..., 2:].unsqueeze(1).repeat(1, nh, 1, 1).reshape(bs * nh, nq, 1, 2)
    loc = rxy + so * nps * rwh * self.offset_scale
    out = _msda_core(value, spatial_shapes_list, loc, aw, self.n_points_list)
    return self.output_proj(out), aw
M.RTDetrV2MultiscaleDeformableAttention.forward = _msda_forward

def _inverse_sigmoid(x, eps=1e-5):
    x1 = x.clamp(min=eps); x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
M.inverse_sigmoid = _inverse_sigmoid

_POS_CACHE = {}
def _sine_pos_forward(self, width, height, device, dtype):
    key = (int(width), int(height), self.embed_dim, self.temperature)
    if key not in _POS_CACHE:
        _POS_CACHE[key] = M.build_2d_sinusoidal_position_embedding(
            height=int(height), width=int(width), embed_dim=self.embed_dim,
            temperature=self.temperature, device="cpu", dtype=torch.float32).unsqueeze(0).detach()
    return _POS_CACHE[key]
if hasattr(M, "RTDetrV2SinePositionEmbedding"):
    M.RTDetrV2SinePositionEmbedding.forward = _sine_pos_forward


# ---- ResNet stem MaxPool -inf pad -> PADV2 (Mali-rejected). Zero-pad + valid pool (input>=0 post-ReLU, exact)
import transformers.models.rt_detr.modeling_rt_detr_resnet as _RN
def _emb_forward(self, pixel_values):
    embedding = self.embedder(pixel_values)
    embedding = F.pad(embedding, (1, 1, 1, 1), value=0.0)
    return F.max_pool2d(embedding, kernel_size=3, stride=2, padding=0)
_RN.RTDetrResNetEmbeddings.forward = _emb_forward


def build_net():
    from transformers import RTDetrV2ForObjectDetection
    net = RTDetrV2ForObjectDetection.from_pretrained(MID).eval()
    print(f"  RT-DETRv2-S params {sum(p.numel() for p in net.parameters())/1e6:.1f}M")
    return net


class GraphA(nn.Module):
    def __init__(s, net):
        super().__init__(); s.m = net.model
        x = torch.zeros(1, 3, R, R)
        with torch.no_grad(): ss_list = s._sources_shapes(x)
        anchors, valid = M.RTDetrV2Model._cached_generate_anchors(tuple(ss_list), 0.05, "cpu", torch.float32)
        anchors = anchors.clamp(min=-1e4, max=1e4)
        s.register_buffer("anchors", anchors, persistent=False)
        s.register_buffer("valid", valid.to(torch.float32), persistent=False)
        s.ss_list = ss_list

    def _sources(s, x):
        feats = s.m.backbone.model(x).feature_maps
        proj = [s.m.encoder_input_proj[i](f) for i, f in enumerate(feats)]
        enc = s.m.encoder(proj).last_hidden_state
        sources = [s.m.decoder_input_proj[i](f) for i, f in enumerate(enc)]
        if s.m.config.num_feature_levels > len(sources):
            sources.append(s.m.decoder_input_proj[len(sources)](enc[-1]))
            for i in range(len(sources), s.m.config.num_feature_levels):
                sources.append(s.m.decoder_input_proj[i](enc[-1]))
        return sources

    def _sources_shapes(s, x): return [(f.shape[-2], f.shape[-1]) for f in s._sources(x)]

    def forward(s, x):
        sources = s._sources(x)
        flat = [src.flatten(2).transpose(1, 2) for src in sources]
        memory_raw = torch.cat(flat, 1)
        memory = s.valid * memory_raw
        output_memory = s.m.enc_output(memory)
        enc_class = s.m.enc_score_head(output_memory)
        enc_coord = s.m.enc_bbox_head(output_memory) + s.anchors
        return enc_class, enc_coord, output_memory, memory_raw


def host_select(enc_class, enc_coord, output_memory):
    idx = enc_class.amax(-1).topk(NQ, dim=1).indices
    ref = torch.gather(enc_coord, 1, idx.unsqueeze(-1).expand(-1, -1, 4))
    tgt = torch.gather(output_memory, 1, idx.unsqueeze(-1).expand(-1, -1, HID))
    return tgt, ref, idx


class GraphB(nn.Module):
    def __init__(s, net):
        super().__init__(); s.dec = net.model.decoder; s.ss_list = None
    def forward(s, memory, target, ref_unact):
        ss = torch.tensor(s.ss_list, dtype=torch.long)
        lsi = torch.cat((ss.new_zeros((1,)), ss.prod(1).cumsum(0)[:-1]))
        out = s.dec(inputs_embeds=target, encoder_hidden_states=memory, reference_points=ref_unact,
                    spatial_shapes=ss, spatial_shapes_list=s.ss_list, level_start_index=lsi)
        return out.intermediate_reference_points[:, -1], out.intermediate_logits[:, -1]


def opcheck(p, l):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=p); it.allocate_tensors()
    ops = collections.Counter(d.get("op_name", "?") for d in it._get_ops_details())
    bad = {k: v for k, v in ops.items() if k.upper() in BANNED}
    over = sum(1 for d in it.get_tensor_details() if len(d.get("shape", [])) > 4)
    print(f"[{l}] ops:", dict(sorted(ops.items(), key=lambda kv: -kv[1])))
    print(f"[{l}] banned:{bad or 'NONE'} >4D:{over} size {os.path.getsize(p)/1e6:.1f}MB",
          "GPU-CLEAN" if not bad and not over else "BLOCKERS")
    return it, (not bad and not over)

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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "forward"
    net = build_net()
    x = torch.randn(1, 3, R, R) * 0.5
    ga = GraphA(net).eval(); gb = GraphB(net).eval(); gb.ss_list = ga.ss_list
    print("spatial shapes:", ga.ss_list, "N =", sum(h * w for h, w in ga.ss_list))
    with torch.no_grad():
        ref = net(pixel_values=x); ref_logits, ref_boxes = ref.logits, ref.pred_boxes
        ec, eco, om, mem = ga(x); tgt, rp, idx = host_select(ec, eco, om); sb, sl = gb(mem, tgt, rp)
    cl = np.corrcoef(sl.numpy().ravel(), ref_logits.numpy().ravel())[0, 1]
    cb = np.corrcoef(sb.numpy().ravel(), ref_boxes.numpy().ravel())[0, 1]
    print(f"split-vs-torch corr  logits {cl:.6f}  boxes {cb:.6f}")
    if cmd == "forward": sys.exit()
    import litert_torch
    pa = f"{HERE}/rtA.tflite"; litert_torch.convert(ga, (x,)).export(pa); opcheck(pa, "rtA")
    pb = f"{HERE}/rtB.tflite"; litert_torch.convert(gb, (mem, tgt, rp)).export(pb); opcheck(pb, "rtB")
    if cmd in ("all", "fp16"):
        to_fp16(pa, f"{HERE}/rtA_fp16.tflite"); opcheck(f"{HERE}/rtA_fp16.tflite", "rtA_fp16")
        to_fp16(pb, f"{HERE}/rtB_fp16.tflite"); opcheck(f"{HERE}/rtB_fp16.tflite", "rtB_fp16")
        x.numpy().astype(np.float32).tofile(f"{HERE}/rtA_in.bin")
        mem.numpy().astype(np.float32).tofile(f"{HERE}/rtB_in_memory.bin")
        tgt.numpy().astype(np.float32).tofile(f"{HERE}/rtB_in_target.bin")
        rp.numpy().astype(np.float32).tofile(f"{HERE}/rtB_in_ref.bin")
        for nm, arr in [("rtA_ec", ec), ("rtA_eco", eco), ("rtA_om", om), ("rtA_mem", mem), ("rtB_logits", sl), ("rtB_boxes", sb)]:
            np.save(f"{HERE}/{nm}.npy", arr.numpy())
        print("saved fp16 + device-probe artifacts")

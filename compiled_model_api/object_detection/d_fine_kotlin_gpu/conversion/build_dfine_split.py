# D-FINE-S 2-graph split for CompiledModel GPU.
#   Graph A (GPU) = backbone(HGNetV2) + hybrid encoder + enc heads
#                   -> enc_class[1,8400,80], enc_coord[1,8400,4], output_memory[1,8400,256], memory[1,8400,256]
#   host (Kotlin) = topk-300 by max class score -> gather coords(ref) + output_memory(target)
#   Graph B (GPU) = FDR decoder (3 layers, deformable 3-level, LQE) -> logits[1,300,80], boxes[1,300,4]
# Re-authoring (vs RF-DETR recipe): grid_sample->tent-matmul, DFineMultiscaleDeformableAttention <=4D
# (multi-level, per-level n_points), DFineLQE prob.topk->iterative-max, distance2bbox stack->cat,
# adaptive SafeLayerNorm, tanh-GELU. Two-stage topk/gather -> host. anchors/valid_mask baked.
import sys, os, collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import transformers.models.d_fine.modeling_d_fine as M

torch._shape_as_tensor = lambda t: torch.tensor(list(t.shape), dtype=torch.long)
torch._assert = lambda *a, **k: None
R = int(os.environ.get("DF_RES", "640")); HERE = os.path.dirname(os.path.abspath(__file__))
MID = "ustc-community/dfine-small-coco"
BANNED = {"GATHER", "GATHER_ND", "TOPK_V2", "GELU", "ERF", "WHERE", "SELECT", "SELECT_V2",
          "BROADCAST_TO", "POW", "TRANSPOSE_CONV", "CAST", "EMBEDDING_LOOKUP", "RFFT2D", "FFT",
          "STFT", "COMPLEX", "CUMSUM", "ARG_MAX", "ARG_MIN", "PACK", "UNPACK", "TILE"}

# ---- grid_sample -> GATHER/CAST-free tent-matmul ---------------------------------------------------
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

# ---- tanh-GELU (POW-free), adaptive SafeLayerNorm -------------------------------------------------
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

# ---- multi-level deformable attention re-authored <=4D --------------------------------------------
def _msda_core(value, spatial_shapes_list, sampling_locations, attention_weights, num_points_list):
    # value [bs, seq, nh, hd]; sampling_locations [bs*nh, nq, P, 2]; attention_weights [bs*nh, 1, nq, P]
    bs, seq, nh, hd = value.shape
    nq = sampling_locations.shape[1]
    val_levels = value.split([h * w for h, w in spatial_shapes_list], dim=1)   # per level [bs, h*w, nh, hd]
    grids = (2 * sampling_locations - 1).split(num_points_list, dim=2)         # per level [bs*nh, nq, np_l, 2]
    sampled = []
    for li, (h, w) in enumerate(spatial_shapes_list):
        vl = val_levels[li].permute(0, 2, 3, 1).reshape(bs * nh, hd, h, w)     # [bs*nh, hd, h, w]
        sampled.append(_gs(vl, grids[li], padding_mode="zeros", align_corners=False))  # [bs*nh, hd, nq, np_l]
    sv = torch.cat(sampled, dim=-1)                                            # [bs*nh, hd, nq, P]
    out = (sv * attention_weights).sum(-1)                                     # [bs*nh, hd, nq]
    return out.reshape(bs, nh * hd, nq).transpose(1, 2).contiguous()          # [bs, nq, dm]

def _msda_forward(self, hidden_states, attention_mask=None, reference_points=None,
                  encoder_hidden_states=None, spatial_shapes=None, spatial_shapes_list=None, **kw):
    bs, nq, _ = hidden_states.shape
    _, seq, _ = encoder_hidden_states.shape
    nh = self.n_heads; dm = self.d_model; hd = dm // nh; P = sum(self.num_points_list)
    value = encoder_hidden_states.reshape(bs, seq, nh, hd)
    so = self.sampling_offsets(hidden_states).reshape(bs, nq, nh, P * 2).permute(0, 2, 1, 3).reshape(bs * nh, nq, P, 2)
    aw = self.attention_weights(hidden_states).reshape(bs, nq, nh, P)
    aw = F.softmax(aw, -1).permute(0, 2, 1, 3).reshape(bs * nh, nq, P).unsqueeze(1)   # [bs*nh,1,nq,P]
    ref = reference_points[:, :, 0, :]                                          # [bs, nq, 4]  (n_levels-for-ref = 1)
    nps = self.num_points_scale.to(hidden_states.dtype).reshape(1, 1, P, 1)
    rxy = ref[..., :2].unsqueeze(1).repeat(1, nh, 1, 1).reshape(bs * nh, nq, 1, 2)   # repeat (expand->BROADCAST_TO)
    rwh = ref[..., 2:].unsqueeze(1).repeat(1, nh, 1, 1).reshape(bs * nh, nq, 1, 2)
    loc = rxy + so * nps * rwh * self.offset_scale
    out = _msda_core(value, spatial_shapes_list, loc, aw, self.num_points_list)
    return out, aw
M.DFineMultiscaleDeformableAttention.forward = _msda_forward

# ---- DFineLQE prob.topk -> iterative max-and-mask (GPU-clean, exact) ------------------------------
def _lqe_forward(self, scores, pred_corners):
    bs, length, _ = pred_corners.size()
    k = self.top_prob_values
    prob = F.softmax(pred_corners.reshape(bs, length * 4, self.max_num_bins + 1), dim=-1)  # 3D (4D amax breaks NHWC)
    cur = prob; vals = []
    for _ in range(k):
        m = cur.amax(-1, keepdim=True)                       # REDUCE_MAX over last dim of a 3D tensor
        vals.append(m)
        eq = torch.relu(1.0 - (m - cur) * 1e6)               # ~one-hot of the max (GPU-clean)
        cur = cur - eq * 1e4                                 # remove the max
    prob_topk = torch.cat(vals, -1)                          # [bs, length*4, k]
    stat = torch.cat([prob_topk, prob_topk.mean(dim=-1, keepdim=True)], dim=-1)  # [bs, length*4, k+1]
    quality_score = self.reg_conf(stat.reshape(bs, length, 4 * (k + 1)))
    return scores + quality_score
M.DFineLQE.forward = _lqe_forward

# ---- distance2bbox stack -> cat (avoid PACK) ------------------------------------------------------
_orig_d2b = M.distance2bbox
def _distance2bbox(points, distance, reg_scale):
    reg_scale = abs(reg_scale)
    tlx = points[..., 0] - (0.5 * reg_scale + distance[..., 0]) * (points[..., 2] / reg_scale)
    tly = points[..., 1] - (0.5 * reg_scale + distance[..., 1]) * (points[..., 3] / reg_scale)
    brx = points[..., 0] + (0.5 * reg_scale + distance[..., 2]) * (points[..., 2] / reg_scale)
    bry = points[..., 1] + (0.5 * reg_scale + distance[..., 3]) * (points[..., 3] / reg_scale)
    bboxes = torch.cat([tlx.unsqueeze(-1), tly.unsqueeze(-1), brx.unsqueeze(-1), bry.unsqueeze(-1)], -1)
    return M.corners_to_center_format(bboxes)
M.distance2bbox = _distance2bbox

# ---- inverse_sigmoid: drop the redundant clamp(0,1) (->RELU_0_TO_1, Mali-rejected) ----------------
# Its only caller passes sigmoid(...) which is already in (0,1), so the (0,1) clamp is a no-op.
def _inverse_sigmoid(x, eps=1e-5):
    x1 = x.clamp(min=eps); x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
M.inverse_sigmoid = _inverse_sigmoid

# ---- AIFI 2D sine pos-embed: bake (kills temperature**omega POW; image-independent fixed size) -----
_POS_CACHE = {}
def _sine_pos_forward(self, width, height, device, dtype):
    key = (int(width), int(height), self.embed_dim, self.temperature)
    if key not in _POS_CACHE:
        _POS_CACHE[key] = M.build_2d_sinusoidal_position_embedding(
            height=int(height), width=int(width), embed_dim=self.embed_dim,
            temperature=self.temperature, device="cpu", dtype=torch.float32).unsqueeze(0).detach()
    return _POS_CACHE[key]
M.DFineSinePositionEmbedding.forward = _sine_pos_forward


def build_net():
    from transformers import DFineForObjectDetection
    net = DFineForObjectDetection.from_pretrained(MID).eval()
    print(f"  D-FINE-S params {sum(p.numel() for p in net.parameters())/1e6:.1f}M")
    return net


NQ = 300; NCLS = 80; HID = 256


class GraphA(nn.Module):
    """image -> enc_class[1,N,80], enc_coord[1,N,4], output_memory[1,N,256], memory[1,N,256]."""
    def __init__(s, net):
        super().__init__(); s.m = net.model
        x = torch.zeros(1, 3, R, R)
        with torch.no_grad():
            ss_list = s._sources_shapes(x)
        # bake anchors + valid_mask (image-independent, from spatial shapes)
        anchors, valid = M.DFineModel._cached_generate_anchors(tuple(ss_list), 0.05, "cpu", torch.float32)
        anchors = anchors.clamp(min=-1e4, max=1e4)   # invalid anchors are finfo.max(3.4e38)->fp16 Inf; clamp finite
        s.register_buffer("anchors", anchors, persistent=False)               # [1, N, 4]
        s.register_buffer("valid", valid.to(torch.float32), persistent=False) # [1, N, 1]
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

    def _sources_shapes(s, x):
        return [(f.shape[-2], f.shape[-1]) for f in s._sources(x)]

    def forward(s, x):
        sources = s._sources(x)
        flat = [src.flatten(2).transpose(1, 2) for src in sources]
        memory_raw = torch.cat(flat, 1)                                       # [bs, N, 256]
        memory = s.valid * memory_raw
        output_memory = s.m.enc_output(memory)
        enc_class = s.m.enc_score_head(output_memory)                         # [bs, N, 80]
        enc_coord = s.m.enc_bbox_head(output_memory) + s.anchors             # [bs, N, 4]
        return enc_class, enc_coord, output_memory, memory_raw


def host_select(enc_class, enc_coord, output_memory):
    idx = enc_class.amax(-1).topk(NQ, dim=1).indices                         # [bs, 300]
    ref = torch.gather(enc_coord, 1, idx.unsqueeze(-1).expand(-1, -1, 4))    # [bs,300,4]
    tgt = torch.gather(output_memory, 1, idx.unsqueeze(-1).expand(-1, -1, HID))
    return tgt, ref, idx


class GraphB(nn.Module):
    """(memory[1,N,256], target[1,300,256], ref_unact[1,300,4]) -> logits[1,300,80], boxes[1,300,4]."""
    def __init__(s, net):
        super().__init__(); s.net = net; s.dec = net.model.decoder; s.ss_list = None

    def forward(s, memory, target, ref_unact):
        bs = memory.shape[0]
        ss = torch.tensor(s.ss_list, dtype=torch.long)
        lsi = torch.cat((ss.new_zeros((1,)), ss.prod(1).cumsum(0)[:-1]))
        out = s.dec(inputs_embeds=target, encoder_hidden_states=memory,
                    reference_points=ref_unact, spatial_shapes=ss,
                    spatial_shapes_list=s.ss_list, level_start_index=lsi)
        logits = out.intermediate_logits[:, -1]
        boxes = out.intermediate_reference_points[:, -1]
        return boxes, logits


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


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "forward"
    net = build_net()
    x = torch.randn(1, 3, R, R) * 0.5
    ga = GraphA(net).eval(); gb = GraphB(net).eval(); gb.ss_list = ga.ss_list
    print("spatial shapes:", ga.ss_list, "N =", sum(h * w for h, w in ga.ss_list))
    with torch.no_grad():
        ref = net(pixel_values=x)
        ref_logits, ref_boxes = ref.logits, ref.pred_boxes
        ec, eco, om, mem = ga(x)
        tgt, rp, idx = host_select(ec, eco, om)
        sb, sl = gb(mem, tgt, rp)
    print("ref :", tuple(ref_logits.shape), tuple(ref_boxes.shape))
    print("split:", tuple(sl.shape), tuple(sb.shape))
    cl = np.corrcoef(sl.numpy().ravel(), ref_logits.numpy().ravel())[0, 1]
    cb = np.corrcoef(sb.numpy().ravel(), ref_boxes.numpy().ravel())[0, 1]
    print(f"split-vs-torch corr  logits {cl:.6f}  boxes {cb:.6f}")
    if cmd == "forward": sys.exit()
    import litert_torch
    pa = f"{HERE}/dfA.tflite"; litert_torch.convert(ga, (x,)).export(pa); opcheck(pa, "dfA")
    pb = f"{HERE}/dfB.tflite"; litert_torch.convert(gb, (mem, tgt, rp)).export(pb); opcheck(pb, "dfB")
    if cmd in ("all", "fp16"):
        from ai_edge_quantizer import quantizer, recipe_manager
        from ai_edge_quantizer.recipe import AlgorithmName, qtyping
        def to_fp16(fp32, fp16):
            rm = recipe_manager.RecipeManager()
            rm.add_quantization_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                op_config=qtyping.OpQuantizationConfig(
                    weight_tensor_config=qtyping.TensorQuantizationConfig(num_bits=16, dtype=qtyping.TensorDataType.FLOAT),
                    compute_precision=qtyping.ComputePrecision.FLOAT), algorithm_key=AlgorithmName.FLOAT_CASTING)
            if os.path.exists(fp16): os.remove(fp16)
            q = quantizer.Quantizer(float_model=fp32); q.load_quantization_recipe(rm.get_quantization_recipe())
            q.quantize().export_model(fp16); return fp16
        to_fp16(pa, f"{HERE}/dfA_fp16.tflite"); opcheck(f"{HERE}/dfA_fp16.tflite", "dfA_fp16")
        to_fp16(pb, f"{HERE}/dfB_fp16.tflite"); opcheck(f"{HERE}/dfB_fp16.tflite", "dfB_fp16")
        x.numpy().astype(np.float32).tofile(f"{HERE}/dfA_in.bin")
        mem.numpy().astype(np.float32).tofile(f"{HERE}/dfB_in_memory.bin")
        tgt.numpy().astype(np.float32).tofile(f"{HERE}/dfB_in_target.bin")
        rp.numpy().astype(np.float32).tofile(f"{HERE}/dfB_in_ref.bin")
        np.save(f"{HERE}/dfA_ec.npy", ec.numpy()); np.save(f"{HERE}/dfA_eco.npy", eco.numpy())
        np.save(f"{HERE}/dfA_om.npy", om.numpy()); np.save(f"{HERE}/dfA_mem.npy", mem.numpy())
        np.save(f"{HERE}/dfB_logits.npy", sl.numpy()); np.save(f"{HERE}/dfB_boxes.npy", sb.numpy())
        print("saved fp16 + device-probe artifacts")

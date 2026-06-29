# D-FINE-S Graph A FIX v3 — apply the RT-DETRv2 3D-sequence-fanout fix to D-FINE's broken out3 (raw memory).
# D-FINE's GraphA is structurally identical to RT-DETRv2's (returns enc_class/enc_coord/output_memory/
# memory_raw, same flatten().transpose()); its device "out3 corr -0.016" was the same Mali 3D-token fan-out
# bug. Fix: Graph A emits only enc_class + memory_raw*2 (clean leaves); the per-token tail (enc_output +
# enc_bbox_head) runs on the host in fp32 over the 300 topk-selected tokens. GraphB (FDR decoder) unchanged.
# NOTE: D-FINE's GraphB FDR decoder ISOLATED corr was 0.752/0.778 with clean CPU inputs (a separate fp16
# issue from the memory bug) -> this fix gives GraphB CLEAN memory; whether the FDR decoder then yields real
# detections (RF-DETR shipped at 0.88) is what the E2E probe decides.
import sys, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import build_dfine_split as S
M = S.M
HERE = os.path.dirname(os.path.abspath(__file__)); R = 640
NQ, NCLS, HID = S.NQ, S.NCLS, S.HID
SCALE = 2.0


class GraphAFix3(S.GraphA):
    def forward(s, x):
        sources = s._sources(x)
        memory_raw = torch.cat([src.flatten(2).transpose(1, 2) for src in sources], 1)
        memory = s.valid * memory_raw
        output_memory = s.m.enc_output(memory)
        enc_class = s.m.enc_score_head(output_memory)   # clean 1-layer leaf out_0
        return enc_class, memory_raw * SCALE            # out_1 scaled leaf (clean)


def host_select(enc_class, memory_raw, ga, net):
    idx = enc_class.amax(-1).topk(NQ, dim=1).indices
    memory = ga.valid * memory_raw
    mem_sel = torch.gather(memory, 1, idx.unsqueeze(-1).expand(-1, -1, HID))
    with torch.no_grad():
        om_sel = net.model.enc_output(mem_sel)            # fp32 host: enc_output on 300 tokens
        coord_delta = net.model.enc_bbox_head(om_sel)     # fp32 host: bbox MLP on 300 tokens
    anc_sel = torch.gather(ga.anchors.expand(1, -1, -1), 1, idx.unsqueeze(-1).expand(-1, -1, 4))
    ref = coord_delta + anc_sel
    return om_sel, ref, idx


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fp16"
    net = S.build_net()
    _inp = f"{HERE}/dfA_real_in.bin"
    x = (torch.from_numpy(np.fromfile(_inp, dtype=np.float32).reshape(1, 3, R, R))
         if os.path.exists(_inp) else torch.randn(1, 3, R, R) * 0.5)
    ga = GraphAFix3(net).eval(); gb = S.GraphB(net).eval(); gb.ss_list = ga.ss_list
    with torch.no_grad():
        ec, mem2 = ga(x); mem = mem2 / SCALE
        tgt, rp, idx = host_select(ec, mem, ga, net)
        sb, sl = gb(mem, tgt, rp)
        ref = net(pixel_values=x)
    cl = np.corrcoef(sl.numpy().ravel(), ref.logits.numpy().ravel())[0, 1]
    cb = np.corrcoef(sb.numpy().ravel(), ref.pred_boxes.numpy().ravel())[0, 1]
    print(f"CPU split-vs-torch corr  logits {cl:.6f}  boxes {cb:.6f}")
    if cmd == "forward":
        sys.exit()
    import litert_torch
    pa = f"{HERE}/dfA_fix3.tflite"
    litert_torch.convert(ga, (x,)).export(pa); S.opcheck(pa, "dfA_fix3")
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
    to_fp16(pa, f"{HERE}/dfA_fix3_fp16.tflite"); S.opcheck(f"{HERE}/dfA_fix3_fp16.tflite", "dfA_fix3_fp16")
    eo = net.model.enc_output; bb = net.model.enc_bbox_head
    np.savez(f"{HERE}/dfine_host3_w.npz",
             eo_w=eo[0].weight.detach().numpy(), eo_b=eo[0].bias.detach().numpy(),
             eo_g=eo[1].weight.detach().numpy(), eo_beta=eo[1].bias.detach().numpy(), eo_eps=np.float32(eo[1].eps),
             bb0w=bb.layers[0].weight.detach().numpy(), bb0b=bb.layers[0].bias.detach().numpy(),
             bb1w=bb.layers[1].weight.detach().numpy(), bb1b=bb.layers[1].bias.detach().numpy(),
             bb2w=bb.layers[2].weight.detach().numpy(), bb2b=bb.layers[2].bias.detach().numpy(),
             valid=ga.valid.detach().numpy(), anchors=ga.anchors.detach().numpy())
    np.save(f"{HERE}/f3_ec.npy", ec.numpy()); np.save(f"{HERE}/f3_mem.npy", mem.numpy())
    print("saved dfA_fix3_fp16.tflite + dfine_host3_w.npz")

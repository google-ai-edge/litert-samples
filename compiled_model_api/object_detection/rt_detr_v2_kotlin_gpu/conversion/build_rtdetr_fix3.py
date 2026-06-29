# RT-DETRv2 Graph A FIX v3 — the clean, cheap split.
# The Mali bug: a 3D sequence tensor [1,N,C] (from flatten().transpose()) that FANS OUT to multiple
# consumers corrupts the longer branch. output_memory -> {score_head(ec), bbox_head(eco)} loses eco
# (3-layer MLP); ec (1 Linear) survives. (4D conv-map fanout is fine — build_loc's 7 outputs were clean.)
# Fix: Graph A emits only the two PROVEN-clean tensors: ec[1,8400,80] and memory_raw[1,8400,256]*2.
# eco (reference points) and target are needed ONLY at the 300 topk-selected tokens -> compute them on
# the HOST in fp32 from the clean memory_raw (enc_output + bbox_head on 300 tokens = negligible). This
# removes the corrupted eco tensor from the GPU graph entirely and makes reference boxes exact.
import sys, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import build_rtdetr_split as S
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
        enc_class = s.m.enc_score_head(output_memory)   # clean (1-layer) leaf out_0
        return enc_class, memory_raw * SCALE            # out_1 scaled leaf (clean)


def host_select(enc_class, memory_raw, ga, net):
    # enc_class[1,8400,80] (GPU), memory_raw[1,8400,256] (GPU). Compute eco+target for the 300 selected on host.
    idx = enc_class.amax(-1).topk(NQ, dim=1).indices               # [1,300]
    memory = ga.valid * memory_raw
    mem_sel = torch.gather(memory, 1, idx.unsqueeze(-1).expand(-1, -1, HID))   # [1,300,256] (pre-mask source)
    with torch.no_grad():
        om_sel = net.model.enc_output(mem_sel)                     # fp32 host: enc_output on 300 tokens
        coord_delta = net.model.enc_bbox_head(om_sel)              # fp32 host: bbox MLP on 300 tokens
    anc_sel = torch.gather(ga.anchors.expand(1, -1, -1), 1, idx.unsqueeze(-1).expand(-1, -1, 4))
    ref = coord_delta + anc_sel
    return om_sel, ref, idx


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fp16"
    net = S.build_net()
    _inp = f"{HERE}/rtA_real_in.bin"   # optional real-image fixture; random input is fine for conversion/parity
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
    pa = f"{HERE}/rtA_fix3.tflite"
    litert_torch.convert(ga, (x,)).export(pa); S.opcheck(pa, "rtA_fix3")
    S.to_fp16(pa, f"{HERE}/rtA_fix3_fp16.tflite"); S.opcheck(f"{HERE}/rtA_fix3_fp16.tflite", "rtA_fix3_fp16")
    eo = net.model.enc_output; bb = net.model.enc_bbox_head
    np.savez(f"{HERE}/host3_w.npz",
             eo_w=eo[0].weight.detach().numpy(), eo_b=eo[0].bias.detach().numpy(),
             eo_g=eo[1].weight.detach().numpy(), eo_beta=eo[1].bias.detach().numpy(), eo_eps=np.float32(eo[1].eps),
             bb0w=bb.layers[0].weight.detach().numpy(), bb0b=bb.layers[0].bias.detach().numpy(),
             bb1w=bb.layers[1].weight.detach().numpy(), bb1b=bb.layers[1].bias.detach().numpy(),
             bb2w=bb.layers[2].weight.detach().numpy(), bb2b=bb.layers[2].bias.detach().numpy(),
             valid=ga.valid.detach().numpy(), anchors=ga.anchors.detach().numpy())
    print("saved rtA_fix3_fp16.tflite + host3_w.npz")

"""
Build the GPU-compatible MoViNet-A0 streaming action-recognition model.

Re-authors Atze00/MoViNet-pytorch (causal streaming, Kinetics-600) into a
single-frame, 4D-only functional forward with explicit recurrent-state I/O,
verifies parity against the original model, and converts it to a LiteRT
CompiledModel-GPU .tflite via litert-torch.

Setup:
    pip install torch litert-torch fvcore einops
    git clone https://github.com/Atze00/MoViNet-pytorch.git   # or set MOVINET_PYTORCH

Run:
    MOVINET_PYTORCH=./MoViNet-pytorch python build_movinet.py

Output: movinet_a0_stream.tflite  (15 MB, 47 inputs / 28 outputs, all float32,
0 tensors rank>4, 0 banned ops — every op runs on the LiteRT GPU delegate). The
stream-buffer shift register and pool running-sum accumulation are done host-side
(see movinet/app .../ActionRecognizer.kt) to avoid Mali stateful-graph bugs.
"""
import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.environ.get("MOVINET_PYTORCH", "MoViNet-pytorch"))
from movinets import MoViNet
from movinets.config import _C
import stream_model as sm

m = MoViNet(_C.MODEL.MoViNetA0, causal=True, pretrained=True).eval()
net = sm.MoViNetA0Stream(m).eval()

# Reference: full-clip causal forward (== frame-by-frame streaming by design).
torch.manual_seed(0)
clip = torch.randn(1, 3, 8, 172, 172)
with torch.no_grad():
    m.clean_activation_buffers()
    out_full = m(clip)

# Parity: flat 4D single-frame module, streamed frame-by-frame with host-side
# stream-buffer shift register + pool-sum accumulation + frame counter (exactly
# what the Android app does).
with torch.no_grad():
    dummy = sm.make_dummy_states()          # 28 stream + 16 pool_sum + 1 count
    hist = {}                               # per-conv history of dim_pad frames
    off = 0
    for name, dp in sm.STREAM_SPEC:
        hist[name] = list(dummy[off:off + dp]); off += dp
    pool_sums = dummy[28:44]
    for t in range(clip.shape[2]):
        stream_inputs = []
        for name, dp in sm.STREAM_SPEC:
            stream_inputs += hist[name]
        inv_count = torch.full((1, 1, 1, 1), 1.0 / (t + 1))
        one = torch.ones(1, 1, 1, 1)
        out = net(clip[:, :, t], *(stream_inputs + pool_sums + [inv_count, one]))
        logits = out[0]
        cur = list(out[1:12])               # 11 current frames
        xspaces = list(out[12:28])          # 16 means
        for ci, (name, dp) in enumerate(sm.STREAM_SPEC):
            hist[name] = hist[name][1:] + [cur[ci]]       # push, drop oldest
        pool_sums = [pool_sums[i] + xspaces[i] for i in range(16)]
diff = (logits - out_full).abs().max().item()
print("flat-module top5:", logits[0].topk(5).indices.tolist())
print("reference  top5:", out_full[0].topk(5).indices.tolist())
print("max abs diff:", diff)
assert diff < 1e-4, "PARITY FAIL"
print("PARITY OK\n")

# Convert.
import litert_torch
dummy = (torch.randn(1, 3, 172, 172), *sm.make_dummy_states())
print("converting: %d inputs ..." % len(dummy))
res = litert_torch.convert(net, dummy)
out_path = "movinet_a0_stream.tflite"
res.export(out_path)
print("saved %s (%.2f MB)" % (out_path, os.path.getsize(out_path) / 1e6))

#!/usr/bin/env python3
# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""tapi_pack.py SRC_DIR OUT_DIR [--layers N] [--block 128] [--fc int4|int8|fp32] [--lm-head int8|int4|fp32]
               [--prefix model.] [--config CONFIG.json]

Weight pre-pass for the Tensor API Qwen2 decoder writer (graph/qwen2_writer.cc): reads a Qwen2-family HF
checkpoint (bf16 or fp32 safetensors, through model.safetensors.index.json when present), and writes
  OUT_DIR/weights.bin   every tensor the decoder needs, 64-byte aligned: fp32 (norms, biases), int4 blockwise
                        (packed two per byte) or int8 per channel, plus the fp32 scales of each quantized tensor
  OUT_DIR/index.tsv     name, kind (fp32 | bw4 | pc8), shape, data offset, data bytes, scales offset, scales count,
                        block size
  OUT_DIR/config.tsv    the decoder's dims (hidden, layers, heads, kv heads, head dim, MLP, vocab, eps, rope theta)
Quantization calls ai-edge-quantizer's own functions (tensor_zp_scale_from_min_max, uniform_quantize, pack_data)
with these choices: FC weights int4 blockwise symmetric min-max with the block scales floored at 2^-14
(a scale below 2^-14 is zero or subnormal in fp16: XNNPACK refuses such FC scales, and a zero-scale block holds w / 0), the
lm_head int8 per channel. So the int4 bytes and fp16 scales equal
what the converter path (litert-torch export_hf + ai-edge-quantizer, same recipe) produces for the same weights.
One tensor at a time: peak memory is one fp32 weight plus its quantized copy.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

ALIGN = 64
FLOOR = np.float32(2.0 ** -14)


def open_checkpoint(src):
    from safetensors import safe_open
    idx = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
    else:
        wm = None
    handles = {}

    def get(name):
        fname = wm[name] if wm else "model.safetensors"
        if fname not in handles:
            handles[fname] = safe_open(os.path.join(src, fname), framework="pt")
        return handles[fname].get_tensor(name).float().numpy()

    return get


def quant_bw4(w, block, stats):
    from ai_edge_quantizer import qtyping
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
    from ai_edge_quantizer.transformations import transformation_utils as tu
    out_f, in_f = w.shape
    assert in_f % block == 0, (w.shape, block)
    wb = w.reshape(out_f, in_f // block, block)
    gran = {32: qtyping.QuantGranularity.BLOCKWISE_32, 64: qtyping.QuantGranularity.BLOCKWISE_64,
            128: qtyping.QuantGranularity.BLOCKWISE_128, 256: qtyping.QuantGranularity.BLOCKWISE_256}[block]
    zp, scale = u.tensor_zp_scale_from_min_max(wb.min(axis=2), wb.max(axis=2), 4, True, gran)
    low = scale < FLOOR
    if low.any():
        stats["floored_tensors"] += 1
        stats["floored_scales"] += int(low.sum())
        scale = np.where(low, FLOOR, scale).astype(np.float32)
    params = qtyping.UniformQuantParams(scale=scale, zero_point=np.zeros_like(scale, dtype=np.int32), num_bits=4,
                                        symmetric=True, quantized_dimension=1, block_size=block)
    q = u.uniform_quantize(w, params, is_blockwise_quant=True)
    packed = tu.pack_data(4, np.ravel(np.asarray(q)).view(np.uint8))
    return np.asarray(packed, dtype=np.uint8), scale.astype(np.float32)


def quant_pc8(w):
    from ai_edge_quantizer import qtyping
    from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as u
    zp, scale = u.tensor_zp_scale_from_min_max(w.min(axis=1), w.max(axis=1), 8, True,
                                               qtyping.QuantGranularity.CHANNELWISE)
    params = qtyping.UniformQuantParams(scale=scale[:, None], zero_point=np.zeros((w.shape[0], 1), np.int32),
                                        num_bits=8, symmetric=True, quantized_dimension=0)
    q = u.uniform_quantize(w, params)
    return np.asarray(q, dtype=np.int8), scale.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--layers", type=int, help="keep the first N layers (a quick check)")
    ap.add_argument("--block", type=int, default=128)
    ap.add_argument("--fc", default="int4", choices=["int4", "int8", "fp32"])
    ap.add_argument("--lm-head", default="int8", choices=["int8", "int4", "fp32"])
    ap.add_argument("--prefix", default="model.", help="decoder prefix in the checkpoint (model. / model.language_model.)")
    ap.add_argument("--config", help="config.json with the text dims (default SRC/config.json; text_config if nested)")
    a = ap.parse_args()

    cfg = json.load(open(a.config or os.path.join(a.src, "config.json")))
    tc = cfg.get("text_config", cfg)
    dims = {
        "hidden": tc["hidden_size"], "layers": a.layers or tc["num_hidden_layers"],
        "heads": tc["num_attention_heads"], "kv_heads": tc["num_key_value_heads"],
        "head_dim": tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"],
        "mlp": tc["intermediate_size"], "vocab": tc["vocab_size"], "eps": tc["rms_norm_eps"],
        "rope_theta": tc.get("rope_theta") or (tc.get("rope_parameters") or {}).get("rope_theta"),
    }
    os.makedirs(a.out, exist_ok=True)
    get = open_checkpoint(a.src)
    p = a.prefix
    names = []
    for i in range(dims["layers"]):
        L = f"{p}layers.{i}."
        names += [(L + "input_layernorm.weight", "fp32"), (L + "post_attention_layernorm.weight", "fp32")]
        for n in ("q", "k", "v"):
            names += [(L + f"self_attn.{n}_proj.weight", a.fc), (L + f"self_attn.{n}_proj.bias", "fp32")]
        names += [(L + "self_attn.o_proj.weight", a.fc)]
        names += [(L + f"mlp.{n}_proj.weight", a.fc) for n in ("gate", "up", "down")]
    names += [(p + "norm.weight", "fp32"), ("lm_head.weight", a.lm_head)]

    stats = {"floored_tensors": 0, "floored_scales": 0}
    t0 = time.time()
    rows, off = [], 0
    with open(os.path.join(a.out, "weights.bin"), "wb") as f:
        def put(arr):
            nonlocal off
            pad = (-off) % ALIGN
            if pad:
                f.write(b"\0" * pad)
                off += pad
            start = off
            data = np.ascontiguousarray(arr).tobytes()
            f.write(data)
            off += len(data)
            return start, len(data)

        for name, kind in names:
            w = get(name)
            canon = ("model." + name[len(p):]) if name.startswith(p) else name
            if kind == "fp32" or w.ndim == 1:
                d0, dn = put(w.astype(np.float32))
                rows.append((canon, "fp32", w.shape, d0, dn, 0, 0, 0))
            elif kind == "int4":
                packed, scale = quant_bw4(w, a.block, stats)
                d0, dn = put(packed)
                s0, sn = put(scale)
                rows.append((canon, "bw4", w.shape, d0, dn, s0, scale.size, a.block))
            elif kind == "int8":
                q, scale = quant_pc8(w)
                d0, dn = put(q)
                s0, sn = put(scale)
                rows.append((canon, "pc8", w.shape, d0, dn, s0, scale.size, 0))
            print(f"{canon} {kind} {list(w.shape)}", flush=True)
    with open(os.path.join(a.out, "index.tsv"), "w") as f:
        for r in rows:
            f.write("\t".join([r[0], r[1], ",".join(str(x) for x in r[2])] + [str(x) for x in r[3:]]) + "\n")
    with open(os.path.join(a.out, "config.tsv"), "w") as f:
        for k, v in dims.items():
            f.write(f"{k}\t{v}\n")
    meta = {"src": os.path.abspath(a.src), "dims": dims, "fc": a.fc, "lm_head": a.lm_head, "block": a.block,
            "floor": stats, "bytes": off, "seconds": round(time.time() - t0, 1), "tensors": len(rows)}
    json.dump(meta, open(os.path.join(a.out, "pack.json"), "w"), indent=1)
    print(f"PACK_DONE {len(rows)} tensors {off} B in {meta['seconds']} s; floor {stats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

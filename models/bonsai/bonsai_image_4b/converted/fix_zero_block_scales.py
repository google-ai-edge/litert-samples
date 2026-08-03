#!/usr/bin/env python3
"""Replace zero blockwise-quantization scales in a .tflite with an epsilon.

Ternary models are sparse: a 32-weight block can be ALL zeros, so min-max
blockwise int4 emits scale = 0 for that block, and XNNPACK refuses to prepare
("unsupported scale value (0.000000) ... for INT4 tensor"). The quantized
values in such blocks are all 0, so dequantization is unchanged by ANY
positive scale — we substitute the tensor's smallest nonzero scale.

LiteRT stores blockwise scales NOT in QuantizationParameters.scale but in a
separate FLOAT16 tensor referenced by the BlockwiseQuantization details table,
so we patch those buffers in place via the raw (lazy) flatbuffers API — no
full model re-serialization needed.

Usage: fix_zero_block_scales.py <in.tflite> <out.tflite>
"""
import shutil
import sys

import numpy as np

from ai_edge_litert import schema_py_generated as schema


def patch_tflite(path):
    data = bytearray(open(path, "rb").read())
    model = schema.Model.GetRootAsModel(data, 0)
    n_fixed = n_tensors = 0
    seen_bufs = set()
    for s in range(model.SubgraphsLength()):
        sg = model.Subgraphs(s)
        for i in range(sg.TensorsLength()):
            q = sg.Tensors(i).Quantization()
            if q is None or q.DetailsType() != \
                    schema.QuantizationDetails.BlockwiseQuantization:
                continue
            bq = schema.BlockwiseQuantization()
            tab = q.Details()
            bq.Init(tab.Bytes, tab.Pos)
            st = sg.Tensors(bq.Scales())
            bidx = st.Buffer()
            if bidx in seen_bufs:
                continue
            seen_bufs.add(bidx)
            buf = model.Buffers(bidx)
            arr = buf.DataAsNumpy()
            if isinstance(arr, np.ndarray):
                f16 = arr.view(np.float16)  # view into `data` (inline buffer)
            else:
                # >2GB-style serialization: buffer data lives out-of-band at an
                # absolute file offset (Buffer.offset/size), not in the vector.
                off, size = buf.Offset(), buf.Size()
                if not size:
                    continue
                f16 = np.ndarray(size // 2, dtype=np.float16, buffer=data,
                                 offset=off)
            zeros = f16 == 0
            if zeros.any():
                nzmin = f16[~zeros].min() if (~zeros).any() else np.float16(1e-4)
                f16[zeros] = nzmin  # writes into `data`
                n_fixed += int(zeros.sum())
                n_tensors += 1
    print(f"patched {n_fixed} zero scales across {n_tensors} scale tensors")
    open(path, "wb").write(data)
    return n_fixed


def main():
    src, dst = sys.argv[1], sys.argv[2]
    if src != dst:
        shutil.copyfile(src, dst)
    patch_tflite(dst)


if __name__ == "__main__":
    main()

# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Replace zero blockwise-int4 scales INSIDE a .litertlm without repacking.

All-zero weight blocks (here: 13 dead rows in decoder layer 0's MLP; 32-weight
blocks) -> ai-edge-quantizer emits scale 0 -> XNNPACK refuses to prepare:
"unsupported scale value (0.000000) ... for INT4 tensor"; the quantized values in such
blocks are 0, so ANY positive scale dequantizes identically). A repack through
litert-lm-builder with one tflite section would drop the externalized embedder
section; this tool instead patches the fp16 scale bytes at their absolute offsets inside the
bundle, so every other byte of the file is untouched (size identical, sections
identical, only scale fp16 words change).

Usage: fix_zero_scales_inplace.py <in.litertlm> <out.litertlm>
Prints per-tensor counts and verifies afterwards that the only differing bytes are the
patched scale words.
"""
import os
import re
import sys
import tempfile

import numpy as np
from ai_edge_litert import schema_py_generated as schema
from litert_lm_builder import peek_litertlm_file


def tflite_sections(path):
  with tempfile.TemporaryDirectory(prefix="zsi_") as d:
    with open(os.path.join(d, "peek.txt"), "w") as out:
      peek_litertlm_file(path, d, out)
    with open(os.path.join(d, "peek.txt")) as f:
      peek = f.read()
  secs = []
  for m in re.finditer(r"Section (\d+):\n  Items:\n(.*?)\n  Begin Offset: (\d+)\n  End Offset:\s+(\d+)\n"
                       r"  Data Type:\s+(\S+)", peek, re.S):
    if m.group(5) == "TFLiteModel":
      secs.append((int(m.group(1)), int(m.group(3)), int(m.group(4)), m.group(2).strip()))
  return secs


def patch_bytes(data, base, end):
  """Patch zero blockwise scales of the tflite at data[base:end] in place. Returns (n_fixed, n_tensors, touched)."""
  view = memoryview(data)[base:end]
  model = schema.Model.GetRootAsModel(bytes(view), 0)  # parse a copy for offsets
  n_fixed = n_tensors = 0
  seen = set()
  touched = []
  for s in range(model.SubgraphsLength()):
    sg = model.Subgraphs(s)
    for i in range(sg.TensorsLength()):
      t = sg.Tensors(i)
      q = t.Quantization()
      if q is None or q.DetailsType() != schema.QuantizationDetails.BlockwiseQuantization:
        continue
      bq = schema.BlockwiseQuantization()
      tab = q.Details()
      bq.Init(tab.Bytes, tab.Pos)
      st = sg.Tensors(bq.Scales())
      bidx = st.Buffer()
      if bidx in seen:
        continue
      seen.add(bidx)
      buf = model.Buffers(bidx)
      if buf.DataLength():
        # inline vector: absolute position of element 0 inside the tflite bytes
        off = buf._tab.Vector(buf._tab.Offset(4))
        size = buf.DataLength()
      else:
        off, size = buf.Offset(), buf.Size()
      if not size:
        continue
      f16 = np.ndarray(size // 2, dtype=np.float16, buffer=data, offset=base + off)
      zeros = f16 == 0
      if zeros.any():
        nzmin = f16[~zeros].min() if (~zeros).any() else np.float16(1e-4)
        f16[zeros] = nzmin
        n_fixed += int(zeros.sum())
        n_tensors += 1
        touched.append((t.Name().decode(errors="replace"), int(zeros.sum()), int(f16.size), float(nzmin)))
  return n_fixed, n_tensors, touched


def main():
  src, dst = sys.argv[1], sys.argv[2]
  data = bytearray(open(src, "rb").read())
  orig = bytes(data)
  total = 0
  for sid, b, e, items in tflite_sections(src):
    n_fixed, n_tensors, touched = patch_bytes(data, b, e)
    print(f"section {sid} [{items}]: patched {n_fixed} zero scales across {n_tensors} scale tensors")
    for name, nz, n, eps in touched[:12]:
      print(f"    {name}: {nz}/{n} zero blocks -> eps {eps:.3g}")
    if len(touched) > 12:
      print(f"    ... {len(touched) - 12} more tensors")
    total += n_fixed
  diff = np.frombuffer(orig, dtype=np.uint8) != np.frombuffer(bytes(data), dtype=np.uint8)
  print(f"bytes changed: {int(diff.sum())} of {len(data)} (<= 2 per patched scale: {2 * total})")
  assert int(diff.sum()) <= 2 * total, "patched more bytes than scale words"
  open(dst, "wb").write(data)
  print("wrote", dst, os.path.getsize(dst), "B")


if __name__ == "__main__":
  main()

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
"""embedder_export.py SRC_DIR OUT.tflite [--name model.embed_tokens.weight] [--lightweight]

The EMBEDDER section of an externalized-embedder bundle for a plain token table (Qwen2 / Qwen2.5-VL: no scale,
no norm): signature "embedder", input token_ids [1,1] int32, output embeddings [1,1,D] float32, with the
`max(token_ids, 0)` guard export_hf puts in front of the lookup (core/external_emb/exportable_module.py). The table
is read from the checkpoint by name (bf16 -> fp32). Quantize afterwards to int8 per channel (EMBEDDING_LOOKUP only, ai-edge-quantizer).
"""
import argparse
import json
import os
import sys

import torch
from torch import nn


class Embedder(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.embed = nn.Embedding(table.shape[0], table.shape[1], _weight=table)

    def forward(self, token_ids):
        token_ids = torch.maximum(token_ids, torch.tensor(0, dtype=torch.int32))
        return {"embeddings": self.embed(token_ids).to(torch.float32)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--name", default="model.embed_tokens.weight")
    ap.add_argument("--lightweight", action="store_true")
    a = ap.parse_args()
    from safetensors import safe_open
    idx = os.path.join(a.src, "model.safetensors.index.json")
    fname = json.load(open(idx))["weight_map"][a.name] if os.path.exists(idx) else "model.safetensors"
    with safe_open(os.path.join(a.src, fname), framework="pt") as f:
        table = f.get_tensor(a.name).float()
    print(f"table {a.name} {tuple(table.shape)} from {fname}", flush=True)
    from litert_torch._convert import interface as converter_utils
    module = Embedder(table).eval()
    conv = converter_utils.Converter()
    conv.add_signature("embedder", module, sample_kwargs={"token_ids": torch.ones((1, 1), dtype=torch.int32)})
    model = conv.convert(strict_export=False, lightweight_conversion=a.lightweight)
    model.export(a.out)
    print(f"EMBEDDER_DONE {a.out} {os.path.getsize(a.out)} B", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

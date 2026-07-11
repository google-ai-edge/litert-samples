# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export the tied token-embedding table as fp16 for host-side lookup on device.

  embeddings_fp16.bin : [vocab, 1024] fp16, little-endian, row-major
    (row = token id)

Usage:
  python export_embeddings.py [MODEL_DIR] [OUT_DIR]
  # defaults: ./Qwen3-Reranker-0.6B next to this script
  #            ->  this script's directory
"""
import os
import sys

import torch
from transformers import AutoModel

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = (sys.argv[1] if len(sys.argv) > 1
             else os.path.join(HERE, "Qwen3-Reranker-0.6B"))
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else HERE


def main():
    """Writes the fp16 token-embedding table for host-side lookup."""
    out = os.path.join(OUT_DIR, "embeddings_fp16.bin")
    m = AutoModel.from_pretrained(MODEL_DIR, dtype=torch.float32).eval()
    # [vocab, 1024]
    w = m.embed_tokens.weight.detach().to(torch.float16).cpu().numpy()
    w.astype("<f2").tofile(out)
    print(f"wrote {out}  shape={w.shape}  {os.path.getsize(out)/1e6:.1f} MB")


if __name__ == "__main__":
    main()

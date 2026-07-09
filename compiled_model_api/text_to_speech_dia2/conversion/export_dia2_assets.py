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

"""Bake Dia2-1B host-side assets as fp16 for the Kotlin loop.

Embeddings plus depformer glue.

Temporal host embed = combined_main[main] + combined_second[second, if not pad]
                      + sum_i audio_embeds[i][audio_i]   (34 channels).
Depformer per stage = depformer_in[wi] @ hidden + audio_embeds[stage][prev];
logits[stage] @ hidden after the graph. (Dia2-1B sets depformer.text_embedding
= False, so no stage-0 text embedding is baked -- the guard below is for other
Dia2 variants.)
The temporal action/cb0 heads live inside the graph, so they are not baked here.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "dia2")
OUT = os.environ.get("DIA2_OUT", "out")
os.makedirs(OUT, exist_ok=True)


def save_f16(name, tensor):
    """Writes a tensor to OUT as a little-endian fp16 table.

    Args:
      name: Output file name.
      tensor: Tensor to bake.
    """
    arr = tensor.detach().float().numpy().astype("<f2")
    arr.tofile(f"{OUT}/{name}")
    print(f"  {name}: {arr.shape} {arr.nbytes / 1e6:.0f} MB")


def main():
    """Bakes the host-side embedding and projection tables."""
    from dia2.engine import Dia2
    d = Dia2.from_repo("nari-labs/Dia2-1B")
    d.set_device("cpu")
    rt = d._ensure_runtime()
    m = rt.model
    t = m.transformer.float().eval()
    dp = m.depformer.float().eval()
    cfg = rt.config

    # --- temporal text embeds (bake the two projections into lookup tables) ---
    te = t.text_embed
    emb_w = te.embedding.weight.data                       # [V, 1024]
    save_f16("dia2_combined_main.f16", emb_w @ te.main_proj.weight.data.t())
    save_f16("dia2_combined_second.f16", emb_w @ te.second_proj.weight.data.t())

    # --- temporal audio embeds (32 codebooks, stacked) ---
    # [32, 2050, 1024]
    aud = torch.stack([e.weight.data for e in t.audio_embeds], dim=0)
    save_f16("dia2_temporal_audio.f16", aud)

    # --- depformer glue ---
    # [31, 2050, 1024]
    dep_aud = torch.stack([e.weight.data for e in dp.audio_embeds], dim=0)
    save_f16("dia2_dep_audio.f16", dep_aud)
    dep_in = torch.stack([dp.depformer_in[str(w)].weight.data
                          for w in sorted(set(dp.weights_schedule))],
                         dim=0)   # [3, 1024, 1024]
    save_f16("dia2_dep_in.f16", dep_in)
    dep_logits = torch.stack([dp.logits[s].weight.data
                              for s in range(dp.num_depth)],
                             dim=0)   # [31, 2050, 1024]
    save_f16("dia2_dep_logits.f16", dep_logits)
    # depformer text_embed (stage 0)
    if dp.text_embed is not None:
        dte = dp.text_embed
        save_f16("dia2_dep_main.f16",
                 dte.embedding.weight.data @ dte.main_proj.weight.data.t())
        save_f16("dia2_dep_second.f16",
                 dte.embedding.weight.data @ dte.second_proj.weight.data.t())

    # --- constants for the host loop ---
    data = cfg.data
    consts = {
        "channels": data.channels,
        "audio_vocab_size": data.audio_vocab_size,
        "audio_pad": data.audio_pad_token_id,
        "audio_bos": data.audio_bos_token_id,
        "text_pad": data.text_pad_token_id,
        "text_bos": getattr(data, "text_bos_token_id", None),
        "text_zero": data.text_zero_token_id,
        "delay_pattern": list(data.delay_pattern),
        "audio_delays": [int(x) for x in rt.audio_delays],
        "weights_schedule": list(dp.weights_schedule),
        "num_depth": dp.num_depth,
        "audio_vocab_limit": dp.audio_vocab_limit,
        "sample_rate": int(rt.mimi.sample_rate),
        "frame_rate": float(rt.mimi.frame_rate),
        "hidden": 1024,
        "temporal_layers": 30,
        "temporal_q_heads": 16,
        "temporal_kv_heads": 8,
        "depformer_layers": 3,
        "depformer_heads": 8,
        "head_dim": 128,
        "rope_theta": 10000.0,
        "tokenizer_id": getattr(rt, "tokenizer_id", None),
    }
    with open(f"{OUT}/dia2_constants.json", "w") as f:
        json.dump(consts, f, indent=2)
    print(f"  dia2_constants.json ({len(consts)} keys)")


if __name__ == "__main__":
    main()

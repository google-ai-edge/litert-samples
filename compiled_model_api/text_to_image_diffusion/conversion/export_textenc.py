"""Export the Bonsai Image text encoder (Qwen3-4B) as a PROMPT EMBEDDER.

The pipeline does not generate text with it: it runs one forward pass with
output_hidden_states, takes layers (9, 18, 27), and interleaves them into a
(B, L, 3*2560=7680) tensor that feeds the DiT's encoder_hidden_states. So the
export target is that tensor, not logits -- and the lm_head is dead weight here.

Fixed shapes: batch 1, sequence 256 (matching the exported DiT's txt length).
"""
import os
import time

import torch
from transformers import Qwen3ForCausalLM

from huggingface_hub import snapshot_download

SNAP = os.environ.get("BONSAI_SNAPSHOT") or snapshot_download(
    "prism-ml/bonsai-image-ternary-4B-unpacked")
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "textenc_fp32.tflite")
SEQ = 256
LAYERS = (9, 18, 27)


class PromptEmbedder(torch.nn.Module):
    """input_ids + attention_mask -> (1, SEQ, 7680) prompt embeds."""

    def __init__(self, lm):
        super().__init__()
        self.model = lm.model          # drop lm_head: unused for embedding
        self.layers = LAYERS

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask,
                         output_hidden_states=True, use_cache=False)
        hs = torch.stack([out.hidden_states[k] for k in self.layers], dim=1)
        b, c, s, d = hs.shape
        return hs.permute(0, 2, 1, 3).reshape(b, s, c * d)


lm = Qwen3ForCausalLM.from_pretrained(f"{SNAP}/text_encoder", torch_dtype=torch.float32)
lm.eval()
print(f"text encoder {sum(p.numel() for p in lm.parameters())/1e9:.3f} B "
      f"hidden={lm.config.hidden_size} layers={lm.config.num_hidden_layers}", flush=True)

emb = PromptEmbedder(lm).eval()
ids = torch.ones(1, SEQ, dtype=torch.int32)
mask = torch.ones(1, SEQ, dtype=torch.int32)
with torch.no_grad():
    ref = emb(ids, mask)
print(f"embedder output {tuple(ref.shape)} (expect (1,{SEQ},7680)) "
      f"mean={ref.mean():.5f} std={ref.std():.5f}", flush=True)
torch.save(ref, os.path.join(HERE, "textenc_ref.pt"))

import litert_torch
t0 = time.time()
m = litert_torch.convert(emb, (ids, mask))
print(f"convert OK in {time.time()-t0:.0f}s", flush=True)
m.export(OUT)
print(f"export OK -> {os.path.getsize(OUT)/2**30:.2f} GiB", flush=True)

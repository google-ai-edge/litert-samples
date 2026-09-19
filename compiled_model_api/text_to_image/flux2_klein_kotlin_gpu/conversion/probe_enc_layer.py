"""Bisects one Qwen3 encoder layer on device, stage by stage.

Exports layer 0 as an fp32 graph with a tap at every stage, so a device run can
be diffed against the host one stage at a time. Using fp32 removes quantization
from the list of suspects.

Taps: q and k after their norm and RoPE, the repeated keys, the raw attention
logits, the softmax probabilities, the attention output before o_proj, and the
full layer output.

Reach for this when the device result is wrong but nothing errors. Note that
materializing an intermediate as a graph output can itself change the result --
that is a finding, not a nuisance: it means the delegate miscompiles the fused
form of whatever feeds it.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from build_klein_enc import REPO, SEQ, patch_repeat_kv

HEAD_DIM = 128
N_HEADS = 32
N_KV = 8
PROBE_HEADS = 2


def rotate_half(x):
    """Splits the last dimension in half and rotates it.

    Args:
      x: Input tensor.

    Returns:
      concat(-x2, x1), where x1 and x2 are the halves of the last dimension.
    """
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class LayerProbe(nn.Module):
    """Layer 0 of the encoder, re-authored so every stage can be tapped."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden, mask, cos, sin):
        attn = self.layer.self_attn
        residual = hidden
        normed = self.layer.input_layernorm(hidden)
        batch, seq, _ = normed.shape

        query = attn.q_norm(
            attn.q_proj(normed).view(batch, seq, N_HEADS, HEAD_DIM))
        key = attn.k_norm(
            attn.k_proj(normed).view(batch, seq, N_KV, HEAD_DIM))
        value = attn.v_proj(normed).view(
            batch, seq, N_KV, HEAD_DIM).transpose(1, 2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)

        cos4 = cos.unsqueeze(1)
        sin4 = sin.unsqueeze(1)
        query = query * cos4 + rotate_half(query) * sin4
        key = key * cos4 + rotate_half(key) * sin4

        rep = N_HEADS // N_KV
        key_rep = torch.cat([key[:, i:i + 1] for i in range(N_KV)
                             for _ in range(rep)], dim=1)
        value_rep = torch.cat([value[:, i:i + 1] for i in range(N_KV)
                               for _ in range(rep)], dim=1)

        raw = torch.matmul(query, key_rep.transpose(2, 3)) * attn.scaling
        logits = raw + mask
        probs = torch.softmax(logits, dim=-1)
        context = torch.matmul(probs, value_rep)
        context = context.transpose(1, 2).reshape(batch, seq, -1)
        hidden = residual + attn.o_proj(context)

        residual = hidden
        normed = self.layer.post_attention_layernorm(hidden)
        hidden = residual + self.layer.mlp(normed)
        return (query, key, key_rep[:, :PROBE_HEADS], raw[:, :PROBE_HEADS],
                probs[:, :PROBE_HEADS], context, hidden)


def main():
    """Exports the tapped layer and writes the host reference for each tap."""
    import litert_torch
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    torch.manual_seed(0)
    patch_repeat_kv()
    qwen = Qwen3Model.from_pretrained(REPO, subfolder="text_encoder",
                                      torch_dtype=torch.float32,
                                      attn_implementation="eager").eval()
    probe = LayerProbe(qwen.layers[0]).eval()

    bins = "klein_bins"

    def load(name, shape):
        import numpy as np
        array = np.fromfile(f"{bins}/{name}.bin", dtype="<f4")
        return torch.from_numpy(array.reshape(shape).copy())

    hidden = load("inputs_embeds", (1, SEQ, 2560))
    mask = load("enc_mask", (1, 1, SEQ, SEQ))
    cos = load("enc_cos", (1, SEQ, HEAD_DIM))
    sin = load("enc_sin", (1, SEQ, HEAD_DIM))

    with torch.no_grad():
        stock = qwen.layers[0](hidden, attention_mask=mask,
                               position_embeddings=(cos, sin))
        stock = stock[0] if isinstance(stock, tuple) else stock
        taps = probe(hidden, mask, cos, sin)

    diff = (stock - taps[-1]).abs().max()
    print(f"[probe] re-authored layer vs stock: max|diff| {diff:.2e}")
    names = ("q_rope", "k_rope", "key_rep", "raw", "probs", "context",
             "layer_out")
    for tap, name in zip(taps, names):
        tap.numpy().astype("<f4").tofile(f"probe_ref_{name}.bin")
        print(f"[ref] {name:10s} {tuple(tap.shape)}")

    inputs = (hidden, mask, cos, sin)
    litert_torch.convert(probe, inputs).export("ke_probe.tflite")
    print(f"[ke_probe] {os.path.getsize('ke_probe.tflite')/1e6:.0f} MB (fp32)")


if __name__ == "__main__":
    main()

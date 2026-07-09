"""FLUX.2-klein text encoder -> three int8 LiteRT chunks.

klein conditions the DiT on Qwen3-4B hidden states from layers 9 / 18 / 27
(stacked and interleaved to 7680 channels), so only 27 of the 36 layers are
needed — and the tap positions land exactly on 9-layer chunk boundaries:

  enc0 : embeddings -> layers 1..9   -> h9   (tap + input to enc1)
  enc1 : layers 10..18               -> h18
  enc2 : layers 19..27               -> h27
  [host] prompt_embeds = interleave(h9, h18, h27) -> [1, 512, 7680]

Each chunk is ~909 MB int8, under the 1 GB compile sweet spot and the >2 GB
flatbuffer load limit. The host does embed_tokens, the causal+padding mask and
the rotary tables, exactly as for Z-Image's encoder graph.

The mask is handed in already broadcast across heads, shaped [1, N_HEADS, S, S]
rather than the usual [1, 1, S, S]. ML Drift miscompiles an implicitly broadcast
ADD applied to a BATCH_MATMUL result, so `softmax(q @ k^T + mask[1,1,S,S])`
returns plausible-looking but wrong probabilities. This is device-only, and both
the CPU reference and the op checker are clean. Materializing the mask across
the head axis removes the broadcast and the attention becomes bit-exact.
"""
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "black-forest-labs/FLUX.2-klein-4B"
SEQ = 512
N_LAYERS = 27
CHUNK = 9
TAPS = (9, 18, 27)
N_HEADS = 32          # the mask is materialized across this axis


def repeat_kv_4d(hidden_states, n_rep):
    """GPU-clean `repeat_kv` for grouped-query attention.

    Two ML Drift constraints have to be dodged at once. Stock transformers does
    `x[:, :, None].expand(b, n_kv, n_rep, s, d).reshape(b, n_kv*n_rep, s, d)`,
    whose intermediate is rank 5 (>4D is rejected). Expanding a 4D slice instead
    stays 4D but lowers to `BROADCAST_TO`, which the delegate also rejects
    outright. Concatenating each KV head `n_rep` times is a plain
    `CONCATENATION` and reproduces the same head order.

    Args:
        hidden_states: Key or value states, shaped [batch, kv_heads, seq, dim].
        n_rep: Query heads per KV head.

    Returns:
        A [batch, kv_heads * n_rep, seq, dim] tensor, heads grouped per KV head.
    """
    if n_rep == 1:
        return hidden_states
    num_kv = hidden_states.shape[1]
    return torch.cat([hidden_states[:, i:i + 1] for i in range(num_kv)
                      for _ in range(n_rep)], dim=1)


def patch_repeat_kv():
    """Swaps transformers' `repeat_kv` for the GPU-clean concat equivalent."""
    from transformers.models.qwen3 import modeling_qwen3
    modeling_qwen3.repeat_kv = repeat_kv_4d


class EncChunk(nn.Module):
    """A run of Qwen3 decoder layers over a fixed-length sequence."""

    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, hidden, mask, cos, sin):
        for layer in self.layers:
            out = layer(hidden, attention_mask=mask,
                        position_embeddings=(cos, sin))
            hidden = out[0] if isinstance(out, tuple) else out
        return hidden


def q_export(model, inputs, name):
    """Quantize-exports a module to an INTEGER-int8 LiteRT graph.

    Args:
      model: The module to convert.
      inputs: Example inputs, in convert-argument order.
      name: Output basename, writes [name].tflite.
    """
    import litert_torch
    from litert_torch.generative.quantize import quant_recipes as qrs
    from litert_torch.generative.quantize.quant_attrs import Dtype, Granularity
    cfg = qrs.full_dynamic_recipe(weight_dtype=Dtype.INT8,
                                  granularity=Granularity.CHANNELWISE)
    out = f"{name}.tflite"
    litert_torch.convert(model, inputs, quant_config=cfg).export(out)
    print(f"[{name}] {os.path.getsize(out)/1e6:.0f} MB")


def main():
    """Splits the encoder into three chunks, checks parity, then exports."""
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

    torch.manual_seed(0)
    patch_repeat_kv()
    print("[patch] repeat_kv -> concat (>4D and BROADCAST_TO are rejected)")
    print("[load] Qwen3 text encoder (fp32) ...")
    qwen = Qwen3Model.from_pretrained(
        REPO, subfolder="text_encoder", torch_dtype=torch.float32,
        attn_implementation="eager").eval()
    # The reference must run the FULL stack: transformers applies the final
    # RMSNorm to hidden_states[num_layers], so truncating the model first would
    # normalize the h27 tap.
    print(f"[load] {len(qwen.layers)} layers, taps {TAPS} are raw")
    n = sum(p.numel() for p in qwen.layers[:N_LAYERS].parameters())
    print(f"[load] first {N_LAYERS} layers = {n/1e9:.2f} B params "
          f"-> int8 ~{n/1e9:.2f} GB")

    inputs_embeds = torch.randn(1, SEQ, 2560) * 0.02
    position_ids = torch.arange(SEQ)[None]
    valid = 37                                   # a 37-token example prompt
    causal = torch.full((SEQ, SEQ), -1e9).triu(1)
    pad = torch.zeros(SEQ)
    pad[valid:] = -1e9                           # padded keys are masked out
    mask = (causal + pad[None, :])[None, None]
    # [1,H,S,S]: materialized across heads, so the ADD does not broadcast.
    mask = mask.expand(1, N_HEADS, SEQ, SEQ).contiguous()

    with torch.no_grad():
        cos, sin = qwen.rotary_emb(inputs_embeds, position_ids)
        ref = qwen(inputs_embeds=inputs_embeds, attention_mask=mask,
                   position_ids=position_ids, use_cache=False,
                   output_hidden_states=True).hidden_states

        chunks = [EncChunk(list(qwen.layers[i:i + CHUNK])).eval()
                  for i in range(0, N_LAYERS, CHUNK)]
        hidden = inputs_embeds
        taps = []
        for chunk in chunks:
            hidden = chunk(hidden, mask, cos, sin)
            taps.append(hidden)

    for tap, layer_index in zip(taps, TAPS):
        want = ref[layer_index]
        pair = torch.stack([want.flatten(), tap.flatten()])
        corr = torch.corrcoef(pair)[0, 1]
        print(f"[parity] chunk tap h{layer_index}: corr {corr:.8f}  "
              f"max|diff| {(want - tap).abs().max():.2e}")

    # host-side interleave, exactly as the pipeline does
    stacked = torch.stack(taps, dim=1)                       # [1,3,S,2560]
    embeds = stacked.permute(0, 2, 1, 3).reshape(1, SEQ, 3 * 2560)
    print(f"[host] prompt_embeds {tuple(embeds.shape)}")

    if "--export" in sys.argv:
        hidden = inputs_embeds
        for i, chunk in enumerate(chunks):
            q_export(chunk, (hidden, mask, cos, sin), f"ke_enc{i}")
            with torch.no_grad():
                hidden = chunk(hidden, mask, cos, sin)


if __name__ == "__main__":
    main()

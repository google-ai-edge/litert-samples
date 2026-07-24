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
"""Authors and exports the MTP (code predictor) decode-step graph.

Qwen3-TTS predicts the 15 residual codebooks of each audio frame with a
5-layer transformer run as an inner autoregressive loop: prefill = [talker
hidden state, codebook-0 embedding], then 15 steps where step k uses
embedding table k-1 and lm_head k. This exports ONE fixed-shape decode-step
graph (17-slot KV cache, one-hot slot update in-graph, all 15 heads output
per step) that the host invokes 17 times per frame:

    inputs : embed [1,1,1024], pos [1] int32, mask [1,1,1,17] additive fp32,
             k_all/v_all [5,1,8,17,128]
    outputs: logits_all [15,2048], k_all, v_all

If ref/mtp_equiv_ref.npz exists (dump_mtp_ref.py), verifies the torch module
and the converted tflite reproduce the reference greedy inner loop 15/15.

Usage (convert env):
    python export_mtp.py
"""

import os

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors import safe_open
from torch import nn

OUT = 'out/mtp'
LAYERS, HEADS, KV_HEADS, HEAD_DIM = 5, 16, 8, 128
VOCAB, CACHE = 2048, 17
EPS, THETA = 1e-6, 1e6


def _rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Applies RMSNorm over the last dimension.

    Args:
        x: Input tensor normalized over its last dimension.
        weight: Per-channel scale of the same size as the last dimension.

    Returns:
        The normalized and scaled tensor with the shape of x.
    """
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x * torch.rsqrt(variance + EPS)) * weight


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the two halves of the last (head) dimension for RoPE.

    Args:
        x: Tensor whose last dimension has size HEAD_DIM.

    Returns:
        cat(-x2, x1) where x1/x2 are the two halves of the last dimension.
    """
    a, b = x[..., :HEAD_DIM // 2], x[..., HEAD_DIM // 2:]
    return torch.cat((-b, a), dim=-1)


class MtpStep(nn.Module):
    """One MTP decode step with an explicit 17-slot KV cache."""

    def __init__(self, weights: dict[str, torch.Tensor]):
        super().__init__()
        for key, tensor in weights.items():
            self.register_buffer(key.replace('.', '_'), tensor,
                                 persistent=False)
        inv_freq = 1.0 / (THETA ** (
            torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        self.register_buffer('slots',
                             torch.arange(CACHE, dtype=torch.float32),
                             persistent=False)

    def forward(self, embed, pos, mask, k_all, v_all):
        x = embed  # [1, 1, 1024]
        angles = pos.float().reshape(1, 1) * self.inv_freq.reshape(1, -1)
        angles = torch.cat((angles, angles), dim=-1)
        cos = angles.cos().reshape(1, 1, 1, HEAD_DIM)
        sin = angles.sin().reshape(1, 1, 1, HEAD_DIM)
        # One-hot cache-slot update keeps the graph free of dynamic scatter.
        one_hot = (self.slots == pos.float().reshape(1)).float().reshape(
            1, 1, CACHE, 1)

        k_new, v_new = [], []
        for i in range(LAYERS):
            def w(name: str, _i: int = i) -> torch.Tensor:
                return getattr(self, f'layers_{_i}_{name}'.replace('.', '_'))

            h = _rms(x, w('input_layernorm.weight'))
            q = F.linear(h, w('self_attn.q_proj.weight')).view(
                1, 1, HEADS, HEAD_DIM)
            k = F.linear(h, w('self_attn.k_proj.weight')).view(
                1, 1, KV_HEADS, HEAD_DIM)
            v = F.linear(h, w('self_attn.v_proj.weight')).view(
                1, 1, KV_HEADS, HEAD_DIM)
            q = _rms(q, w('self_attn.q_norm.weight')).transpose(1, 2)
            k = _rms(k, w('self_attn.k_norm.weight')).transpose(1, 2)
            v = v.transpose(1, 2)
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

            k_cache = k_all[i] * (1.0 - one_hot) + k * one_hot
            v_cache = v_all[i] * (1.0 - one_hot) + v * one_hot
            k_new.append(k_cache)
            v_new.append(v_cache)

            k_rep = k_cache.repeat_interleave(HEADS // KV_HEADS, dim=1)
            v_rep = v_cache.repeat_interleave(HEADS // KV_HEADS, dim=1)
            attn = torch.matmul(q, k_rep.transpose(2, 3))
            attn = attn * (HEAD_DIM ** -0.5) + mask
            attn = attn.softmax(dim=-1)
            out = torch.matmul(attn, v_rep).transpose(1, 2)
            out = out.reshape(1, 1, HEADS * HEAD_DIM)
            x = x + F.linear(out, w('self_attn.o_proj.weight'))

            h2 = _rms(x, w('post_attention_layernorm.weight'))
            gate = F.silu(F.linear(h2, w('mlp.gate_proj.weight')))
            ff = F.linear(gate * F.linear(h2, w('mlp.up_proj.weight')),
                          w('mlp.down_proj.weight'))
            x = x + ff

        x = _rms(x, self.norm_weight)
        logits_all = torch.matmul(self.heads, x.reshape(1024, 1))
        return (logits_all.reshape(15, VOCAB), torch.stack(k_new),
                torch.stack(v_new))


def _load_weights() -> dict[str, torch.Tensor]:
    """Loads code-predictor weights and saves the 15 embedding tables.

    Returns:
        Dict of fp32 layer/norm weights plus 'heads' [15, 2048, 1024];
        also writes out/mtp/mtp_embeddings.npy [15, 2048, 1024].
    """
    src = snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base')
    reader = safe_open(f'{src}/model.safetensors', framework='pt')
    prefix = 'talker.code_predictor.'
    weights = {}
    for key in reader.keys():
        if key.startswith(prefix + 'model.layers.') or (
                key == prefix + 'model.norm.weight'):
            weights[key[len(prefix + 'model.'):]] = reader.get_tensor(
                key).to(torch.float32)
    heads = [reader.get_tensor(f'{prefix}lm_head.{i}.weight').to(
        torch.float32) for i in range(15)]
    weights['heads'] = torch.stack(heads)
    embeddings = [reader.get_tensor(
        f'{prefix}model.codec_embedding.{i}.weight').to(
            torch.float32).numpy() for i in range(15)]
    os.makedirs(OUT, exist_ok=True)
    np.save(f'{OUT}/mtp_embeddings.npy', np.stack(embeddings))
    return weights


def _run_frame(step_fn, past_hidden, cb0_embed, mtp_embeddings):
    """Reference inner loop: 2 seed feeds, then 15 greedy steps.

    Args:
        step_fn: Callable (embed, pos, mask, k_all, v_all) -> (logits_all,
            k_all, v_all) running one decode step.
        past_hidden: Talker hidden state [1, 1, 1024] (seed feed 0).
        cb0_embed: Codebook-0 embedding [1, 1, 1024] (seed feed 1).
        mtp_embeddings: Embedding tables [15, 2048, 1024] for steps 2..16.

    Returns:
        The 15 greedy residual codebook ids as an int array.
    """
    k_all = np.zeros((LAYERS, 1, KV_HEADS, CACHE, HEAD_DIM), np.float32)
    v_all = np.zeros_like(k_all)
    feeds = [past_hidden, cb0_embed]
    codes = []
    for t in range(16):
        embed = (feeds[t] if t < 2 else
                 mtp_embeddings[t - 2][codes[-1]]).reshape(1, 1, 1024)
        mask = np.where(np.arange(CACHE) <= t, 0.0,
                        -1e9).astype(np.float32).reshape(1, 1, 1, -1)
        logits_all, k_all, v_all = step_fn(
            embed.astype(np.float32), np.array([t], np.int32), mask,
            k_all, v_all)
        if t >= 1:
            codes.append(int(np.asarray(logits_all)[t - 1].argmax()))
    return np.array(codes)


def main() -> None:
    """Exports the MTP decode-step graph and checks reference parity."""
    weights = _load_weights()
    module = MtpStep(weights).eval()
    mtp_embeddings = np.load(f'{OUT}/mtp_embeddings.npy')

    ref = None
    if os.path.exists('ref/mtp_equiv_ref.npz'):
        ref = np.load('ref/mtp_equiv_ref.npz')

        def torch_step(embed, pos, mask, k_all, v_all):
            with torch.no_grad():
                logits, k, v = module(torch.tensor(embed), torch.tensor(pos),
                                      torch.tensor(mask),
                                      torch.tensor(k_all),
                                      torch.tensor(v_all))
            return logits.numpy(), k.numpy(), v.numpy()

        codes = _run_frame(torch_step, ref['past_hidden'],
                           ref['last_id_hidden'], mtp_embeddings)
        matches = int((codes == ref['seq'][0]).sum())
        print(f'torch vs reference: {matches}/15 greedy tokens')
        assert matches == 15

    import litert_torch

    sample = (torch.zeros(1, 1, 1024), torch.zeros(1, dtype=torch.int32),
              torch.zeros(1, 1, 1, CACHE),
              torch.zeros(LAYERS, 1, KV_HEADS, CACHE, HEAD_DIM),
              torch.zeros(LAYERS, 1, KV_HEADS, CACHE, HEAD_DIM))
    litert_torch.convert(module, sample).export(f'{OUT}/mtp_step.tflite')
    print('converted ->', f'{OUT}/mtp_step.tflite')

    if ref is not None:
        from ai_edge_litert.compiled_model import CompiledModel

        tfl = CompiledModel.from_file(f'{OUT}/mtp_step.tflite')
        ins = tfl.create_input_buffers(0)
        outs = tfl.create_output_buffers(0)
        kv_shape = (LAYERS, 1, KV_HEADS, CACHE, HEAD_DIM)
        kv_size = LAYERS * KV_HEADS * CACHE * HEAD_DIM

        def tfl_step(embed, pos, mask, k_all, v_all):
            # Buffer order == signature order == forward() argument order.
            for buf, array in zip(ins, (embed, pos, mask, k_all, v_all)):
                buf.write(np.ascontiguousarray(array))
            tfl.run_by_index(0, ins, outs)
            return (outs[0].read(15 * VOCAB, np.float32).reshape(15, VOCAB),
                    outs[1].read(kv_size, np.float32).reshape(kv_shape),
                    outs[2].read(kv_size, np.float32).reshape(kv_shape))

        codes = _run_frame(tfl_step, ref['past_hidden'],
                           ref['last_id_hidden'], mtp_embeddings)
        print(f'tflite vs reference: {int((codes == ref["seq"][0]).sum())}'
              '/15 greedy tokens')


if __name__ == '__main__':
    main()

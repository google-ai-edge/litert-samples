#!/usr/bin/env python3
"""Validates the RWKV-7 fp16 step graph with the LiteRT CompiledModel API.

Two checks against the fp32 PyTorch reference (build_rwkv7_step.py):
  1. Prefill logits correlation on a prompt (expects > 0.999).
  2. 30-token greedy generation: every tflite-chosen token must be the fp32
     argmax for the same context, or a statistical near-tie (fp32 rank <= 2
     with a top1-top2 logit gap < 0.05). fp16 rounding legitimately flips
     near-tie argmaxes; a divergence with a LARGE fp32 gap is a real bug.

Requires rwkv7_0.1b.pth, rwkv_vocab_v20230424.txt and rwkv7_step_fp16.tflite
next to this script (see build_rwkv7_step.py).

Run:  python validate_rwkv7.py
"""

import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_rwkv7_step import (
    CKPT_PATH,
    FP16_PATH,
    HEAD_DIM,
    N_EMBD,
    N_HEAD,
    N_LAYER,
    PARITY_PROMPT,
    VOCAB_PATH,
    VOCAB_SIZE,
    Rwkv7Step,
    RwkvTokenizer,
)

GENERATION_STEPS = 30
NEAR_TIE_GAP = 0.05


class TfliteStepper:
    """Ping-pongs the recurrent state through the fp16 CompiledModel graph."""

    def __init__(self, path: str):
        from ai_edge_litert.compiled_model import CompiledModel

        self.model = CompiledModel.from_file(path)
        self.inputs = self.model.create_input_buffers(0)
        self.outputs = self.model.create_output_buffers(0)
        self.att = np.zeros((N_LAYER, N_EMBD), np.float32)
        self.ffn = np.zeros((N_LAYER, N_EMBD), np.float32)
        self.wkv = np.zeros(
            (N_LAYER * N_HEAD, HEAD_DIM, HEAD_DIM), np.float32
        )

    def step(self, emb_row: np.ndarray) -> np.ndarray:
        """Runs one token step; returns the logits."""
        self.inputs[0].write(np.ascontiguousarray(emb_row, np.float32))
        self.inputs[1].write(self.att.ravel())
        self.inputs[2].write(self.ffn.ravel())
        self.inputs[3].write(self.wkv.ravel())
        self.model.run_by_index(0, self.inputs, self.outputs)
        logits = self.outputs[0].read(VOCAB_SIZE, np.float32)
        self.att = self.outputs[1].read(N_LAYER * N_EMBD, np.float32).reshape(
            N_LAYER, N_EMBD
        )
        self.ffn = self.outputs[2].read(N_LAYER * N_EMBD, np.float32).reshape(
            N_LAYER, N_EMBD
        )
        self.wkv = (
            self.outputs[3]
            .read(N_LAYER * N_HEAD * HEAD_DIM * HEAD_DIM, np.float32)
            .reshape(N_LAYER * N_HEAD, HEAD_DIM, HEAD_DIM)
        )
        return logits


class TorchStepper:
    """fp32 PyTorch step-mode reference."""

    def __init__(self, sd):
        self.step_model = Rwkv7Step(sd).eval()
        self.att = torch.zeros(N_LAYER, N_EMBD)
        self.ffn = torch.zeros(N_LAYER, N_EMBD)
        self.wkv = torch.zeros(N_LAYER * N_HEAD, HEAD_DIM, HEAD_DIM)

    def step(self, emb_row: torch.Tensor) -> torch.Tensor:
        """Runs one token step; returns the logits."""
        with torch.no_grad():
            logits, self.att, self.ffn, self.wkv = self.step_model(
                emb_row, self.att, self.ffn, self.wkv
            )
        return logits[0]


def main() -> None:
    sd = torch.load(CKPT_PATH, map_location="cpu")
    sd = {k: v.float() for k, v in sd.items()}
    tok = RwkvTokenizer(VOCAB_PATH)
    emb = sd["emb.weight"].float()
    prompt_ids = tok.encode(PARITY_PROMPT)

    tfl = TfliteStepper(FP16_PATH)
    ref = TorchStepper(sd)
    tfl_logits = ref_logits = None
    for t in prompt_ids:
        tfl_logits = tfl.step(emb[t : t + 1].numpy())
        ref_logits = ref.step(emb[t : t + 1])
    corr = np.corrcoef(tfl_logits, ref_logits.numpy())[0, 1]
    print(f"prefill logits corr {corr:.7f}")
    assert corr > 0.999, "fp16 tflite diverges from fp32 torch on prefill"

    # Greedy generation: follow the TFLITE-chosen path; at each step check the
    # chosen token's rank in the fp32 logits computed on the SAME context.
    generated, hard_divergences = [], []
    for step_idx in range(GENERATION_STEPS):
        chosen = int(tfl_logits.argmax())
        order = torch.argsort(ref_logits, descending=True)
        rank = int((order == chosen).nonzero()[0])
        gap = float(ref_logits[order[0]] - ref_logits[order[1]])
        marker = "" if rank == 0 else f"  (fp32 rank {rank}, gap {gap:.4f})"
        print(f"step {step_idx:2d}: {tok.decode([chosen])!r}{marker}")
        if rank > 0 and (rank > 1 or gap > NEAR_TIE_GAP):
            hard_divergences.append(step_idx)
        generated.append(chosen)
        tfl_logits = tfl.step(emb[chosen : chosen + 1].numpy())
        ref_logits = ref.step(emb[chosen : chosen + 1])

    print("\ngeneration:", repr(tok.decode(generated)))
    print("hard divergences (rank>1 or gap>%.2f): %s"
          % (NEAR_TIE_GAP, hard_divergences or "none"))
    assert not hard_divergences, "generation left the fp32 near-tie envelope"
    print("PASS")


if __name__ == "__main__":
    main()

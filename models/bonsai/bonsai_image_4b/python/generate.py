"""Text-to-image with Bonsai Image 4B on LiteRT — no torch, no diffusers.

The whole diffusion pipeline runs in three LiteRT graphs (text encoder, DiT,
VAE decoder); this script is only the host loop: tokenize, FlowMatch-Euler
sampling, latent unpatchify, PNG save. Dependencies: numpy, Pillow,
ai-edge-litert, transformers (tokenizer only).

    python generate.py --model-dir <dir> --prompt "a bonsai tree" --out out.png

<dir> holds dit_int4b32.tflite / textenc_int4.tflite / vae_dec_fp32.tflite,
the tokenizer/ folder, and pipeline_meta.json (latent-normalization constants).
Graphs are fixed-shape: 512x512 output, 256 prompt tokens, 4 sampling steps by
default (the model is step-distilled; more steps also work).
"""
import argparse
import json
import math
import os
import time

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from PIL import Image
from transformers import AutoTokenizer

SEQ, TOKENS, LAT_GRID = 256, 1024, 32   # 512x512: 32x32 patch grid, 64x64 latent


class Graph:
    """Fixed-shape tflite runner, inputs mapped by ARGUMENT POSITION.

    Never map by shape: the text encoder's input_ids and attention_mask are both
    (1, 256), and a shape-keyed map silently drops one of them. litert-torch
    names inputs serving_default_args_<n> in the original forward() order.
    """

    def __init__(self, path, threads=os.cpu_count()):
        self.it = Interpreter(model_path=path, num_threads=threads)
        self.it.allocate_tensors()

        def argpos(d):
            parts = d["name"].rsplit("args_", 1)
            return int(parts[1].split(":")[0]) if len(parts) == 2 else 0

        self.inputs = sorted(self.it.get_input_details(), key=argpos)
        self.output = self.it.get_output_details()[0]

    def __call__(self, *tensors):
        for t, d in zip(tensors, self.inputs):
            self.it.set_tensor(d["index"], np.asarray(t, dtype=d["dtype"]))
        self.it.invoke()
        return self.it.get_tensor(self.output["index"]).copy()


def flowmatch_sigmas(steps):
    """FLUX.2-klein sigma schedule: linspace shifted by the empirical mu
    (diffusers compute_empirical_mu), exponential time-shift. timestep == sigma."""
    m200 = 0.00016927 * TOKENS + 0.45666666
    m10 = 8.73809524e-05 * TOKENS + 1.89833333
    a = (m200 - m10) / 190.0
    mu = a * steps + (m200 - 200.0 * a)
    lin = np.linspace(1.0, 1.0 / steps, steps)
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / lin - 1.0))
    return np.append(shifted, 0.0).astype(np.float32)


def unpatchify(lat, bn_scale, bn_shift):
    """(1024, 128) packed tokens -> (1, 32, 64, 64) VAE latent.

    Per-PACKED-channel affine (the VAE's BatchNorm running stats) comes first,
    then the 2x2 patch unfold: packed channel m = c*4 + i*2 + j lands at
    z[c, 2h+i, 2w+j] for token (h, w).
    """
    z = lat.astype(np.float32) * bn_scale + bn_shift          # (1024, 128)
    z = z.reshape(LAT_GRID, LAT_GRID, LAT_GRID, 2, 2)         # (h, w, c, i, j)
    return z.transpose(2, 0, 3, 1, 4).reshape(1, 32, 64, 64)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt", default="a small bonsai tree in a blue ceramic pot")
    p.add_argument("--out", default="out.png")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    args = p.parse_args()
    d = args.model_dir

    meta = json.load(open(os.path.join(d, "pipeline_meta.json")))
    bn_scale = np.asarray(meta["latent_bn_scale"], np.float32)   # 128
    bn_shift = np.asarray(meta["latent_bn_shift"], np.float32)   # 128

    tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
    text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(text, return_tensors="np", padding="max_length", truncation=True,
              max_length=SEQ)

    t0 = time.time()
    textenc = Graph(os.path.join(d, meta["files"]["textenc"]), args.threads)
    embeds = textenc(enc["input_ids"].astype(np.int32),
                     enc["attention_mask"].astype(np.int32))
    del textenc
    print(f"prompt encoded {time.time()-t0:.1f}s", flush=True)

    # position ids: image tokens on a (h, w) grid, text tokens along the sequence
    hh, ww = np.meshgrid(np.arange(LAT_GRID), np.arange(LAT_GRID), indexing="ij")
    img_ids = np.stack([np.zeros_like(hh), hh, ww, np.zeros_like(hh)],
                       -1).reshape(TOKENS, 4).astype(np.float32)
    txt_ids = np.stack([np.zeros(SEQ)] * 3 + [np.arange(SEQ)], -1).astype(np.float32)

    sigmas = flowmatch_sigmas(args.steps)
    lat = np.random.default_rng(args.seed).standard_normal(
        (1, TOKENS, 128)).astype(np.float32)
    if os.environ.get("BONSAI_INIT_LATENTS"):                 # numeric-parity testing
        lat = np.fromfile(os.environ["BONSAI_INIT_LATENTS"],
                          np.float32).reshape(1, TOKENS, 128)

    t0 = time.time()
    dit = Graph(os.path.join(d, meta["files"]["dit"]), args.threads)
    print(f"DiT loaded {time.time()-t0:.1f}s", flush=True)
    for k in range(args.steps):
        t0 = time.time()
        v = dit(lat, embeds, sigmas[k:k + 1], img_ids, txt_ids)
        lat = lat + (sigmas[k + 1] - sigmas[k]) * v
        print(f"step {k + 1}/{args.steps} sigma {sigmas[k]:.3f} "
              f"{time.time()-t0:.1f}s", flush=True)
    del dit

    t0 = time.time()
    vae = Graph(os.path.join(d, meta["files"]["vae"]), args.threads)
    y = vae(unpatchify(lat[0], bn_scale, bn_shift))           # (1, 3, 512, 512)
    print(f"VAE decoded {time.time()-t0:.1f}s", flush=True)

    rgb = (np.clip(y[0] / 2 + 0.5, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(rgb.transpose(1, 2, 0)).save(args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()

"""Text-to-image with Bonsai Image 4B on LiteRT — no torch, no diffusers.

The whole diffusion pipeline runs in three LiteRT graphs (text encoder, DiT,
VAE decoder); this script is only the host loop: tokenize, FlowMatch-Euler
sampling, latent unpatchify, PNG save. Dependencies: numpy, Pillow,
ai-edge-litert, transformers (tokenizer only) and jinja2 (its chat template).

    python generate.py --model-dir <dir> --prompt "a bonsai tree" --out out.png
    python generate.py --model-dir <dir> --size 256 --prompt "..." --out out256.png

<dir> holds the graphs listed in pipeline_meta.json (`files` for 512x512, and a
`variants` entry per other output size, e.g. "256": the 256x256 DiT and VAE
decoder), the tokenizer/ folder, and the latent-normalization constants.
Graphs are fixed-shape: one DiT + VAE pair per output size, 256 prompt tokens,
4 sampling steps by default (the model is step-distilled; more steps also work).
The text encoder is shared by every size.
"""
import argparse
import json
import math
import os
import time

import numpy as np
from ai_edge_litert.compiled_model import CompiledModel
from ai_edge_litert.options import CpuOptions, Options
from PIL import Image
from transformers import AutoTokenizer

SEQ = 256            # prompt tokens (fixed for every size)
LATENT_CH = 32       # VAE latent channels; 2x2 patchify packs them to 128 per token


class Graph:
    """Fixed-shape tflite runner on the CompiledModel API, inputs mapped by
    ARGUMENT POSITION.

    Never map by shape: the text encoder's input_ids and attention_mask are both
    (1, 256), and at 256x256 the DiT's img_ids and txt_ids are both (256, 4) —
    a shape-keyed map silently drops one of them. litert-torch names inputs
    args_<n> in the original forward() order. The tensor buffers are created
    once and rewritten on every call.
    """

    def __init__(self, path, threads=os.cpu_count()):
        self.model = CompiledModel.from_file(
            path, options=Options(cpu_options=CpuOptions(num_threads=threads)))
        self.sig = next(iter(self.model.get_signature_list()))

        def argpos(name):
            parts = name.rsplit("args_", 1)
            return int(parts[1]) if len(parts) == 2 else 0

        details = self.model.get_input_tensor_details(self.sig)
        self.inputs = sorted(details.items(), key=lambda kv: argpos(kv[0]))
        self.in_bufs = {n: self.model.create_input_buffer_by_name(self.sig, n)
                        for n, _ in self.inputs}
        self.out_name, self.output = next(
            iter(self.model.get_output_tensor_details(self.sig).items()))
        self.out_buf = self.model.create_output_buffer_by_name(self.sig,
                                                                 self.out_name)

    def __call__(self, *tensors):
        for t, (n, d) in zip(tensors, self.inputs):
            a = np.asarray(t, dtype=d["dtype"])
            if list(a.shape) != list(d["shape"]):   # write() does not check
                raise ValueError(f"{n}: expected {d['shape']}, got {list(a.shape)}")
            self.in_bufs[n].write(a)
        self.model.run_by_name(self.sig, self.in_bufs,
                               {self.out_name: self.out_buf})
        shape = self.output["shape"]
        return self.out_buf.read(int(np.prod(shape)),
                                 np.dtype(self.output["dtype"])).reshape(shape)


def flowmatch_sigmas(steps, tokens):
    """FLUX.2-klein sigma schedule: linspace shifted by the empirical mu
    (diffusers compute_empirical_mu, a function of the image-token count),
    exponential time-shift. timestep == sigma."""
    m200 = 0.00016927 * tokens + 0.45666666
    m10 = 8.73809524e-05 * tokens + 1.89833333
    a = (m200 - m10) / 190.0
    mu = a * steps + (m200 - 200.0 * a)
    lin = np.linspace(1.0, 1.0 / steps, steps)
    shifted = math.exp(mu) / (math.exp(mu) + (1.0 / lin - 1.0))
    return np.append(shifted, 0.0).astype(np.float32)


def unpatchify(lat, bn_scale, bn_shift, grid):
    """(grid*grid, 128) packed tokens -> (1, 32, 2*grid, 2*grid) VAE latent.

    Per-PACKED-channel affine (the VAE's BatchNorm running stats) comes first,
    then the 2x2 patch unfold: packed channel m = c*4 + i*2 + j lands at
    z[c, 2h+i, 2w+j] for token (h, w).
    """
    z = lat.astype(np.float32) * bn_scale + bn_shift          # (tokens, 128)
    z = z.reshape(grid, grid, LATENT_CH, 2, 2)                # (h, w, c, i, j)
    return z.transpose(2, 0, 3, 1, 4).reshape(1, LATENT_CH, 2 * grid, 2 * grid)


def graph_files(meta, size, overrides):
    """dit / textenc / vae file names for an output size (+ command-line overrides)."""
    files = dict(meta["files"])
    if size != 512:
        variants = meta.get("variants", {})
        if str(size) in variants:
            files.update(variants[str(size)])
        elif not (overrides.get("dit") and overrides.get("vae")):
            raise SystemExit(f"no {size}x{size} graphs in pipeline_meta.json "
                             f"(sizes: 512, {', '.join(variants)}); pass --dit and --vae")
    files.update({k: v for k, v in overrides.items() if v})
    return files


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--prompt", default="a small bonsai tree in a blue ceramic pot")
    p.add_argument("--out", default="out.png")
    p.add_argument("--size", type=int, default=512, help="output side in px (512 or 256)")
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    p.add_argument("--dit", help="override the DiT file for this run")
    p.add_argument("--vae", help="override the VAE decoder file for this run")
    p.add_argument("--textenc", help="override the text encoder file for this run")
    args = p.parse_args()
    d = args.model_dir
    if args.size % 16:
        raise SystemExit("--size must be a multiple of 16")
    grid = args.size // 16
    tokens = grid * grid

    meta = json.load(open(os.path.join(d, "pipeline_meta.json")))
    bn_scale = np.asarray(meta["latent_bn_scale"], np.float32)   # 128
    bn_shift = np.asarray(meta["latent_bn_shift"], np.float32)   # 128
    files = graph_files(meta, args.size,
                        {k: getattr(args, k) for k in ("dit", "vae", "textenc")})

    tok = AutoTokenizer.from_pretrained(os.path.join(d, "tokenizer"))
    text = tok.apply_chat_template([{"role": "user", "content": args.prompt}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    enc = tok(text, return_tensors="np", padding="max_length", truncation=True,
              max_length=SEQ)

    t0 = time.time()
    textenc = Graph(os.path.join(d, files["textenc"]), args.threads)
    embeds = textenc(enc["input_ids"].astype(np.int32),
                     enc["attention_mask"].astype(np.int32))
    del textenc
    print(f"prompt encoded {time.time()-t0:.1f}s", flush=True)

    # position ids: image tokens on a (h, w) grid, text tokens along the sequence
    hh, ww = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    img_ids = np.stack([np.zeros_like(hh), hh, ww, np.zeros_like(hh)],
                       -1).reshape(tokens, 4).astype(np.float32)
    txt_ids = np.stack([np.zeros(SEQ)] * 3 + [np.arange(SEQ)], -1).astype(np.float32)

    sigmas = flowmatch_sigmas(args.steps, tokens)
    lat = np.random.default_rng(args.seed).standard_normal(
        (1, tokens, 128)).astype(np.float32)
    if os.environ.get("BONSAI_INIT_LATENTS"):                 # numeric-parity testing
        lat = np.fromfile(os.environ["BONSAI_INIT_LATENTS"],
                          np.float32).reshape(1, tokens, 128)

    t0 = time.time()
    dit = Graph(os.path.join(d, files["dit"]), args.threads)
    print(f"DiT loaded {time.time()-t0:.1f}s ({files['dit']})", flush=True)
    for k in range(args.steps):
        t0 = time.time()
        v = dit(lat, embeds, sigmas[k:k + 1], img_ids, txt_ids)
        lat = lat + (sigmas[k + 1] - sigmas[k]) * v
        print(f"step {k + 1}/{args.steps} sigma {sigmas[k]:.3f} "
              f"{time.time()-t0:.1f}s", flush=True)
    del dit

    t0 = time.time()
    vae = Graph(os.path.join(d, files["vae"]), args.threads)
    y = vae(unpatchify(lat[0], bn_scale, bn_shift, grid))     # (1, 3, size, size)
    print(f"VAE decoded {time.time()-t0:.1f}s ({files['vae']})", flush=True)

    rgb = (np.clip(y[0] / 2 + 0.5, 0, 1) * 255).round().astype(np.uint8)
    Image.fromarray(rgb.transpose(1, 2, 0)).save(args.out)
    print(f"saved {args.out} ({args.size}x{args.size})")


if __name__ == "__main__":
    main()

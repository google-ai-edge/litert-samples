"""FLUX.2-klein int8 quality gate: real image from the actual LiteRT chunks.

Runs the stock Flux2KleinPipeline twice with the same seed: once with the fp32
transformer, once with the transformer swapped for the eight exported int8
chunks executed through the LiteRT `CompiledModel` API. Reports PSNR.

Random-input correlations are the wrong metric for a diffusion model. The
generated image is the metric, and this is the gate that chose int8 over int4.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

REPO = "black-forest-labs/FLUX.2-klein-4B"
PROMPT = "a red apple on a wooden table, studio lighting"
STEPS, SIZE, SEED = 4, 256, 1234
N_TXT = 512
CHUNK_DIR = os.path.dirname(os.path.abspath(__file__))

_CACHE = {}


def tfl_run(name, *inputs):
    """Runs one exported chunk through the LiteRT CompiledModel API.

    Args:
      name: Graph basename, loads [name].tflite.
      *inputs: Input arrays, in signature order.

    Returns:
      The graph's outputs, in signature order.
    """
    from ai_edge_litert.compiled_model import CompiledModel
    if name not in _CACHE:
        path = os.path.join(CHUNK_DIR, f"{name}.tflite")
        _CACHE[name] = CompiledModel.from_file(path)
    model = _CACHE[name]
    sigs = model.get_signature_list()
    key = list(sigs)[0]
    in_details = model.get_input_tensor_details(key)
    out_details = model.get_output_tensor_details(key)
    in_buffers = model.create_input_buffers(0)
    out_buffers = model.create_output_buffers(0)
    bindings = zip(sigs[key]["inputs"], in_buffers, inputs)
    for tensor_name, buffer, value in bindings:
        dtype = np.dtype(in_details[tensor_name]["dtype"])
        buffer.write(np.ascontiguousarray(value, dtype=dtype))
    model.run_by_index(0, in_buffers, out_buffers)
    outputs = []
    for tensor_name, buffer in zip(sigs[key]["outputs"], out_buffers):
        detail = out_details[tensor_name]
        count = int(np.prod(detail["shape"]))
        flat = buffer.read(count, np.dtype(detail["dtype"]))
        outputs.append(torch.from_numpy(flat.reshape(detail["shape"]).copy()))
    return outputs


def dit_chunks(rm, hidden_states, encoder_hidden_states, timestep, img_ids,
               txt_ids):
    """Runs the on-device DiT loop: host prep plus eight int8 graphs.

    Args:
      rm: The stock transformer, used only for its host-side helpers.
      hidden_states: Packed image tokens.
      encoder_hidden_states: Conditioning tensor.
      timestep: The scheduler timestep, already divided by 1000.
      img_ids: Image position ids.
      txt_ids: Text position ids.

    Returns:
      The predicted noise, [1, 256, 128].
    """
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]
    temb = rm.time_guidance_embed(timestep.to(hidden_states.dtype) * 1000, None)
    img_cos, img_sin = rm.pos_embed(img_ids)
    txt_cos, txt_sin = rm.pos_embed(txt_ids)
    cos = torch.cat([txt_cos, img_cos], dim=0)[:, 0::2][None, :, None, :]
    sin = torch.cat([txt_sin, img_sin], dim=0)[:, 0::2][None, :, None, :]

    hidden, encoder, mod_i, mod_t, mod_s = tfl_run(
        "kc_prep", hidden_states.numpy(), encoder_hidden_states.numpy(),
        temb.numpy())
    for i in range(2):
        hidden, encoder = tfl_run(
            f"kc_double{i}", hidden.numpy(), encoder.numpy(), cos.numpy(),
            sin.numpy(), mod_i.numpy(), mod_t.numpy())
    joint = torch.cat([encoder, hidden], dim=1)          # host concat
    for i in range(4):
        joint = tfl_run(f"kc_single{i}", joint.numpy(), cos.numpy(),
                        sin.numpy(), mod_s.numpy())[0]
    return tfl_run("kc_final", joint.numpy(), temb.numpy())[0]


def main():
    """Generates once with fp32 and once with the int8 chunks, reports PSNR."""
    from diffusers import Flux2KleinPipeline
    from diffusers.models.transformers.transformer_flux2 import (
        Flux2Transformer2DModelOutput)

    print("[load] Flux2KleinPipeline (fp32) ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        REPO, torch_dtype=torch.float32).to("cpu")
    rm = pipe.transformer
    stock_forward = rm.forward

    def run(tag):
        generator = torch.Generator("cpu").manual_seed(SEED)
        image = pipe(prompt=PROMPT, height=SIZE, width=SIZE,
                     num_inference_steps=STEPS, generator=generator).images[0]
        image.save(f"klein_{tag}.png")
        return np.asarray(image, dtype=np.float64)

    print(f"[ref] fp32 pipeline, {STEPS} steps ...")
    ref = run("fp32")

    def hooked(hidden_states=None, encoder_hidden_states=None, timestep=None,
               img_ids=None, txt_ids=None, return_dict=True, **kwargs):
        with torch.no_grad():
            out = dit_chunks(rm, hidden_states, encoder_hidden_states, timestep,
                             img_ids, txt_ids)
        if return_dict:
            return Flux2Transformer2DModelOutput(sample=out)
        return (out,)

    rm.forward = hooked
    print("[int8] pipeline driven by the eight LiteRT chunks ...")
    test = run("int8")
    rm.forward = stock_forward

    mse = ((ref - test) ** 2).mean()
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0
    corr = np.corrcoef(ref.flatten(), test.flatten())[0, 1]
    print(f"[gate] int8 chunks vs fp32: PSNR {psnr:.1f} dB  corr {corr:.5f}  "
          f"max px diff {np.abs(ref - test).max():.0f}/255")


if __name__ == "__main__":
    main()

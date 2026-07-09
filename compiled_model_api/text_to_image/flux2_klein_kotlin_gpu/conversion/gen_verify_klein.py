"""Standalone FLUX.2-klein loop driven only by the exported .tflite graphs.

No torch model is loaded: every learned weight comes from a graph, every other
tensor from the .bin files `gen_prep_klein.py` wrote. This is the exact loop the
Kotlin app will run, so if the PSNR against `ref_fp32.png` is good here, the
Kotlin port is a transcription rather than a redesign.

Graphs are loaded one at a time and dropped, mirroring the device's sequential
residency: never more than one ~912 MB graph is resident.
"""
import os

import numpy as np
import torch

STEPS = 4
SEQ_TXT, SEQ_IMG = 512, 256
DIM_ENC, TAPS, ENC_HEADS = 2560, 3, 32
LATENT_CH, LATENT_HW = 32, 32
PACKED_CH, PACKED_HW = 128, 16
DIR = os.path.dirname(os.path.abspath(__file__))
BINS = os.path.join(DIR, "klein_bins")


def load(name, shape):
    """Reads a staged float32 tensor.

    Args:
      name: Basename inside the bins directory.
      shape: Target shape.

    Returns:
      The tensor, reshaped.
    """
    array = np.fromfile(os.path.join(BINS, f"{name}.bin"), dtype="<f4")
    return torch.from_numpy(array.reshape(shape).copy())


def load_int(name):
    """Reads a staged int32 index map.

    Args:
      name: Basename inside the bins directory.

    Returns:
      A 1-D int64 index tensor.
    """
    array = np.fromfile(os.path.join(BINS, f"{name}.bin"), dtype="<i4")
    return torch.from_numpy(array.astype(np.int64))


def tfl_run(name, *inputs):
    """Runs one exported chunk, then releases it (sequential residency).

    Args:
      name: Graph basename, loads [name].tflite.
      *inputs: Input arrays, in signature order.

    Returns:
      The graph's outputs, in signature order.
    """
    from ai_edge_litert.compiled_model import CompiledModel
    model = CompiledModel.from_file(os.path.join(DIR, f"{name}.tflite"))
    signatures = model.get_signature_list()
    key = list(signatures)[0]
    in_details = model.get_input_tensor_details(key)
    out_details = model.get_output_tensor_details(key)
    in_buffers = model.create_input_buffers(0)
    out_buffers = model.create_output_buffers(0)
    bindings = zip(signatures[key]["inputs"], in_buffers, inputs)
    for tensor_name, buffer, value in bindings:
        dtype = np.dtype(in_details[tensor_name]["dtype"])
        buffer.write(np.ascontiguousarray(value, dtype=dtype))
    model.run_by_index(0, in_buffers, out_buffers)
    outputs = []
    for tensor_name, buffer in zip(signatures[key]["outputs"], out_buffers):
        detail = out_details[tensor_name]
        count = int(np.prod(detail["shape"]))
        flat = buffer.read(count, np.dtype(detail["dtype"]))
        outputs.append(torch.from_numpy(flat.reshape(detail["shape"]).copy()))
    del model
    return outputs


def encode_text(inputs_embeds, mask, cos, sin):
    """Runs the three encoder chunks and interleaves their taps.

    Args:
      inputs_embeds: Host token embeddings, [1, 512, 2560].
      mask: Head-expanded attention mask, [1, 32, 512, 512].
      cos: Encoder rotary cosine table.
      sin: Encoder rotary sine table.

    Returns:
      The [1, 512, 7680] conditioning tensor.
    """
    taps, hidden = [], inputs_embeds
    for i in range(TAPS):
        hidden = tfl_run(f"ke_enc{i}", hidden.numpy(), mask.numpy(),
                         cos.numpy(), sin.numpy())[0]
        taps.append(hidden)
    stacked = torch.stack(taps, dim=1)
    return stacked.permute(0, 2, 1, 3).reshape(1, SEQ_TXT, TAPS * DIM_ENC)


def denoise_step(latents, prompt_embeds, temb, cos, sin):
    """Runs one DiT step across the eight transformer chunks.

    Args:
      latents: Packed image tokens, [1, 256, 128].
      prompt_embeds: Conditioning tensor, [1, 512, 7680].
      temb: Timestep embedding for this step, [1, 3072].
      cos: Joint rotary cosine table.
      sin: Joint rotary sine table.

    Returns:
      The predicted noise, [1, 256, 128].
    """
    hidden, encoder, mod_img, mod_txt, mod_single = tfl_run(
        "kc_prep", latents.numpy(), prompt_embeds.numpy(), temb.numpy())
    for i in range(2):
        hidden, encoder = tfl_run(
            f"kc_double{i}", hidden.numpy(), encoder.numpy(), cos.numpy(),
            sin.numpy(), mod_img.numpy(), mod_txt.numpy())
    joint = torch.cat([encoder, hidden], dim=1)
    for i in range(4):
        joint = tfl_run(f"kc_single{i}", joint.numpy(), cos.numpy(),
                        sin.numpy(), mod_single.numpy())[0]
    return tfl_run("kc_final", joint.numpy(), temb.numpy())[0]


def decode(latents, unpack_perm, unpatch_perm, bn_mean, bn_std):
    """Turns packed latent tokens into an RGB tensor.

    Args:
      latents: Packed image tokens, [1, 256, 128].
      unpack_perm: Gather map for the position-id scatter.
      unpatch_perm: Gather map for the 2x2 unpatchify.
      bn_mean: Per-channel batch-norm mean.
      bn_std: Per-channel batch-norm standard deviation.

    Returns:
      The decoded image, [1, 3, 256, 256], in [-1, 1].
    """
    unpacked = latents.flatten()[unpack_perm].reshape(
        1, PACKED_CH, PACKED_HW, PACKED_HW)
    unpacked = unpacked * bn_std.view(1, -1, 1, 1) + bn_mean.view(1, -1, 1, 1)
    latent = unpacked.flatten()[unpatch_perm].reshape(
        1, LATENT_CH, LATENT_HW, LATENT_HW)
    return tfl_run("kv_vae", latent.numpy())[0]


def to_pixels(image):
    """Converts a VAE output to uint8 HWC, as VaeImageProcessor does.

    Args:
      image: Decoded image, [1, 3, H, W], in [-1, 1].

    Returns:
      A uint8 [H, W, 3] array.
    """
    array = (image[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255).round().astype(np.uint8)


def main():
    """Runs the device loop on the host and gates it against the reference."""
    from PIL import Image

    inputs_embeds = load("inputs_embeds", (1, SEQ_TXT, DIM_ENC))
    mask = load("enc_mask", (1, ENC_HEADS, SEQ_TXT, SEQ_TXT))
    enc_cos = load("enc_cos", (1, SEQ_TXT, -1))
    enc_sin = load("enc_sin", (1, SEQ_TXT, -1))
    cos = load("cos", (1, SEQ_TXT + SEQ_IMG, 1, 64))
    sin = load("sin", (1, SEQ_TXT + SEQ_IMG, 1, 64))
    temb = load("temb", (STEPS, -1))
    dsigma = load("dsigma", (STEPS,))
    latents = load("latents0", (1, SEQ_IMG, PACKED_CH))
    bn_mean, bn_std = load("bn_mean", (-1,)), load("bn_std", (-1,))
    want_embeds = load("prompt_embeds", (1, SEQ_TXT, TAPS * DIM_ENC))
    unpack_perm = load_int("unpack_perm")
    unpatch_perm = load_int("unpatch_perm")

    print("[enc] three int8 encoder chunks ...")
    prompt_embeds = encode_text(inputs_embeds, mask, enc_cos, enc_sin)
    corr = torch.corrcoef(torch.stack([want_embeds.flatten(),
                                       prompt_embeds.flatten()]))[0, 1]
    print(f"[enc] prompt_embeds vs fp32: corr {corr:.6f}  "
          f"max|diff| {(want_embeds - prompt_embeds).abs().max():.3f}  "
          f"norm ratio {prompt_embeds.norm() / want_embeds.norm():.4f}")

    for step in range(STEPS):
        noise_pred = denoise_step(
            latents, prompt_embeds, temb[step:step + 1], cos, sin)
        latents = latents + dsigma[step] * noise_pred
        print(f"[step {step}] |noise| {noise_pred.norm():.2f}  "
              f"|latents| {latents.norm():.2f}")

    print("[vae] decoding ...")
    decoded = decode(latents, unpack_perm, unpatch_perm, bn_mean, bn_std)
    pixels = to_pixels(decoded)
    Image.fromarray(pixels).save(os.path.join(BINS, "tflite_loop.png"))

    reference = Image.open(os.path.join(BINS, "ref_fp32.png"))
    ref = np.asarray(reference, dtype=np.float64)
    test = pixels.astype(np.float64)
    mse = ((ref - test) ** 2).mean()
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0
    corr = np.corrcoef(ref.flatten(), test.flatten())[0, 1]
    print(f"[gate] tflite-only loop vs fp32 pipeline: PSNR {psnr:.1f} dB  "
          f"corr {corr:.5f}  max px diff {np.abs(ref - test).max():.0f}/255")


if __name__ == "__main__":
    main()

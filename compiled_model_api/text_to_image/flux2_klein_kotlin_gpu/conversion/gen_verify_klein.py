"""Standalone FLUX.2-klein loop driven only by the exported .tflite graphs.

No torch model is loaded: every learned weight comes from a graph, every other
tensor from the .bin files `gen_prep_klein.py` wrote. This is the exact loop the
Kotlin app will run, so if the PSNR against `ref_fp32.png` is good here, the
Kotlin port is a transcription rather than a redesign.

Graphs are loaded one at a time and dropped, mirroring the device's sequential
residency: never more than one ~912 MB graph is resident.

With `--edit` the reference image's latent tokens (staged by
`gen_prep_klein.py --edit`) are concatenated onto the noise tokens before every
DiT step and the `kce_*` graphs are used. Only the first `SEQ_IMG` output tokens
are the noise prediction, exactly as `noise_pred[:, :latents.size(1)]` does.

`--device` swaps the executor for an adb-driven one: the identical loop runs on
the phone's GPU (`CompiledModel` + `Accelerator.GPU`, FP32), one graph resident
at a time, with the host doing only the arithmetic between graphs. Nothing else
changes, so a difference in the output is a difference in the delegate.

That mode needs a device-side runner: any binary that loads a `.tflite` with
`CompiledModel` + `Accelerator.GPU`, fills input *i* from `in.<i>` (float32,
little endian) and dumps output *i* to `out.bin.<i>`. Point `LITERT_GPU_RUNNER`
at it and `LITERT_GPU_RUNNER_LIBS` at the directory holding its `.so` files.

Usage:
    python gen_verify_klein.py                    # text-to-image, host CPU
    python gen_verify_klein.py --edit             # image editing, host CPU
    python gen_verify_klein.py --edit --device    # image editing, phone GPU
    python gen_verify_klein.py --edit --device --no-push  # graphs staged
"""
import os
import subprocess
import sys

import numpy as np
import torch

STEPS = 4
SEQ_TXT, SEQ_IMG = 512, 256
DIM_ENC, TAPS, ENC_HEADS = 2560, 3, 32
DIM_DIT = 3072
LATENT_CH, LATENT_HW = 32, 32
PACKED_CH, PACKED_HW = 128, 16
IMAGE_HW = 256
DIR = os.path.dirname(os.path.abspath(__file__))
BINS = os.path.join(DIR, "klein_bins")
if "--bins" in sys.argv:
    BINS = os.path.join(DIR, sys.argv[sys.argv.index("--bins") + 1])

DEVICE_DIR = "/data/local/tmp/klein_loop"
RUNNER = os.environ.get("LITERT_GPU_RUNNER", "")
RUNNER_LIBS = os.environ.get("LITERT_GPU_RUNNER_LIBS", "")


def load(name, shape):
    """Reads OUT/<name>.bin as float32 and reshapes it.

    Args:
        name: Tensor name, without the .bin suffix.
        shape: The shape to reshape the flat buffer into.

    Returns:
        A float32 tensor.
    """
    array = np.fromfile(os.path.join(BINS, f"{name}.bin"), dtype="<f4")
    return torch.from_numpy(array.reshape(shape).copy())


def load_int(name):
    """Reads OUT/<name>.bin as an int32 index array.

    Args:
        name: Index-map name, without the .bin suffix.

    Returns:
        An int64 tensor of indices.
    """
    return torch.from_numpy(
        np.fromfile(os.path.join(BINS, f"{name}.bin"),
                    dtype="<i4").astype(np.int64))


def tfl_run(name, *inputs):
    """Runs one exported chunk, then releases it (sequential residency).

    Args:
        name: Graph basename, without the .tflite suffix.
        *inputs: One array per graph input, in signature order.

    Returns:
        One tensor per graph output, in signature order.
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


def output_shapes(prefix, seq_img):
    """Every graph's output shapes, so the device dumps can be reshaped.

    Args:
        prefix: `kc` for text-to-image, `kce` for editing.
        seq_img: Image tokens the transformer sees.

    Returns:
        A map from graph name to its list of output shapes.
    """
    seq = SEQ_TXT + seq_img
    shapes = {f"ke_enc{i}": [(1, SEQ_TXT, DIM_ENC)] for i in range(TAPS)}
    shapes[f"{prefix}_prep"] = [(1, seq_img, DIM_DIT), (1, SEQ_TXT, DIM_DIT),
                                (1, 1, 6 * DIM_DIT), (1, 1, 6 * DIM_DIT),
                                (1, 1, 3 * DIM_DIT)]
    for i in range(2):
        shapes[f"{prefix}_double{i}"] = [(1, seq_img, DIM_DIT),
                                         (1, SEQ_TXT, DIM_DIT)]
    for i in range(4):
        shapes[f"{prefix}_single{i}"] = [(1, seq, DIM_DIT)]
    shapes[f"{prefix}_final"] = [(1, seq_img, PACKED_CH)]
    shapes["kv_vae"] = [(1, 3, IMAGE_HW, IMAGE_HW)]
    return shapes


def adb(command):
    """Runs one adb shell command and returns its stdout.

    Args:
        command: The shell command to run on the device.

    Returns:
        Whatever the command printed to stdout.
    """
    return subprocess.run(["adb", "shell", command], check=True,
                          capture_output=True, text=True).stdout


def stage_device(graph_names):
    """Pushes the runner, its libraries and every graph once.

    Args:
        graph_names: Every graph the loop will run.
    """
    if not RUNNER or not RUNNER_LIBS:
        raise SystemExit(
            "--device needs LITERT_GPU_RUNNER and LITERT_GPU_RUNNER_LIBS; "
            "see the module docstring for what the runner must do")
    adb(f"mkdir -p {DEVICE_DIR}")
    subprocess.run(["adb", "push", RUNNER, f"{DEVICE_DIR}/runner"], check=True,
                   capture_output=True)
    for lib in sorted(os.listdir(RUNNER_LIBS)):
        if lib.endswith(".so"):
            subprocess.run(["adb", "push", os.path.join(RUNNER_LIBS, lib),
                            f"{DEVICE_DIR}/"], check=True, capture_output=True)
    adb(f"chmod +x {DEVICE_DIR}/runner")
    for name in graph_names:
        print(f"[push] {name}.tflite")
        subprocess.run(["adb", "push", os.path.join(DIR, f"{name}.tflite"),
                        f"{DEVICE_DIR}/"], check=True, capture_output=True)


def device_run(name, *inputs, shapes=None):
    """Runs one graph on the phone GPU (FP32) and returns its outputs.

    Args:
        name: Graph basename, without the .tflite suffix.
        *inputs: One array per graph input, in signature order.
        shapes: The output shapes, to reshape the device dumps.

    Returns:
        One tensor per graph output.
    """
    adb(f"rm -f {DEVICE_DIR}/in.* {DEVICE_DIR}/out.bin.*")
    for index, value in enumerate(inputs):
        path = os.path.join(DIR, f"_dev_in{index}.bin")
        np.ascontiguousarray(value, dtype="<f4").tofile(path)
        subprocess.run(["adb", "push", path, f"{DEVICE_DIR}/in.{index}"],
                       check=True, capture_output=True)
        os.remove(path)

    log = adb(f"cd {DEVICE_DIR} && FP32=1 LD_LIBRARY_PATH=. "
              f"./runner {name}.tflite 1 in out.bin 2>&1")
    if "RUN OK" not in log:
        raise SystemExit(f"{name} failed on device:\n{log}")
    partial = [line for line in log.splitlines() if "Replacing" in line]
    if partial and " out of " in partial[0]:
        counts = partial[0].split("Replacing ")[1].split(" node")[0]
        taken, total = counts.split(" out of ")
        if taken != total:
            raise SystemExit(f"{name}: only {taken}/{total} nodes delegated")

    outputs = []
    for index, shape in enumerate(shapes):
        local = os.path.join(DIR, f"_dev_out{index}.bin")
        subprocess.run(["adb", "pull", f"{DEVICE_DIR}/out.bin.{index}", local],
                       check=True, capture_output=True)
        outputs.append(torch.from_numpy(
            np.fromfile(local, dtype=np.float32).reshape(shape).copy()))
        os.remove(local)
    return outputs


_USE_DEVICE = False
_SHAPES = {}


def run_graph(name, *inputs):
    """Runs one graph on whichever executor this invocation selected.

    Args:
        name: Graph basename, without the .tflite suffix.
        *inputs: One array per graph input, in signature order.

    Returns:
        One tensor per graph output.
    """
    if _USE_DEVICE:
        return device_run(name, *inputs, shapes=_SHAPES[name])
    return tfl_run(name, *inputs)


def encode_text(inputs_embeds, mask, cos, sin):
    """Three encoder chunks -> the interleaved 7680-channel prompt embedding.

    Args:
        inputs_embeds: Token embeddings staged by gen_prep.
        mask: The 4D causal-plus-padding attention mask.
        cos: Rotary cosine table.
        sin: Rotary sine table.

    Returns:
        The [1, 512, 7680] conditioning tensor.
    """
    taps, hidden = [], inputs_embeds
    for i in range(TAPS):
        hidden = run_graph(f"ke_enc{i}", hidden.numpy(), mask.numpy(),
                           cos.numpy(), sin.numpy())[0]
        taps.append(hidden)
    stacked = torch.stack(taps, dim=1)
    return stacked.permute(0, 2, 1, 3).reshape(1, SEQ_TXT, TAPS * DIM_ENC)


def denoise_step(tokens, prompt_embeds, temb, cos, sin, prefix):
    """One DiT step: prep -> doubles -> host concat -> singles -> final.

    Args:
        tokens: Noise tokens, plus reference tokens when editing.
        prompt_embeds: The conditioning tensor.
        temb: This step's timestep embedding.
        cos: Rotary cosine table.
        sin: Rotary sine table.
        prefix: `kc` for text-to-image, `kce` for editing.

    Returns:
        The predicted noise over every image token.
    """
    hidden, encoder, mod_img, mod_txt, mod_single = run_graph(
        f"{prefix}_prep", tokens.numpy(), prompt_embeds.numpy(), temb.numpy())
    for i in range(2):
        hidden, encoder = run_graph(f"{prefix}_double{i}", hidden.numpy(),
                                    encoder.numpy(), cos.numpy(), sin.numpy(),
                                    mod_img.numpy(), mod_txt.numpy())
    joint = torch.cat([encoder, hidden], dim=1)
    for i in range(4):
        joint = run_graph(f"{prefix}_single{i}", joint.numpy(), cos.numpy(),
                          sin.numpy(), mod_single.numpy())[0]
    return run_graph(f"{prefix}_final", joint.numpy(), temb.numpy())[0]


def decode(latents, unpack_perm, unpatch_perm, bn_mean, bn_std):
    """Token latents -> RGB: unpack, BN denorm, unpatchify, VAE decode.

    Args:
        latents: The final packed latent tokens.
        unpack_perm: Gather map from tokens to a packed latent.
        unpatch_perm: Gather map from packed latent to VAE latent.
        bn_mean: Per-channel batch-norm mean.
        bn_std: Per-channel batch-norm standard deviation.

    Returns:
        A [1, 3, 256, 256] image tensor in [-1, 1].
    """
    unpacked = latents.flatten()[unpack_perm].reshape(
        1, PACKED_CH, PACKED_HW, PACKED_HW)
    unpacked = unpacked * bn_std.view(1, -1, 1, 1) + bn_mean.view(1, -1, 1, 1)
    latent = unpacked.flatten()[unpatch_perm].reshape(
        1, LATENT_CH, LATENT_HW, LATENT_HW)
    return run_graph("kv_vae", latent.numpy())[0]


def to_pixels(image):
    """VAE output in [-1,1] -> uint8 HWC, like VaeImageProcessor.postprocess.

    Args:
        image: A [1, 3, 256, 256] tensor in [-1, 1].

    Returns:
        A uint8 HWC array.
    """
    array = (image[0] / 2 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy()
    return (array * 255).round().astype(np.uint8)


def main():
    """Runs the whole loop on the chosen executor and scores it."""
    from PIL import Image

    global _USE_DEVICE, _SHAPES
    editing = "--edit" in sys.argv
    _USE_DEVICE = "--device" in sys.argv
    prefix = "kce" if editing else "kc"
    seq_img = 2 * SEQ_IMG if editing else SEQ_IMG
    where = "phone GPU (FP32)" if _USE_DEVICE else "host CPU"
    print(f"[mode] {'image editing' if editing else 'text-to-image'}: "
          f"joint sequence {SEQ_TXT + seq_img}, graphs {prefix}_*, on {where}")

    if _USE_DEVICE:
        _SHAPES = output_shapes(prefix, seq_img)
        if "--no-push" not in sys.argv:
            stage_device(list(_SHAPES))

    inputs_embeds = load("inputs_embeds", (1, SEQ_TXT, DIM_ENC))
    mask = load("enc_mask", (1, ENC_HEADS, SEQ_TXT, SEQ_TXT))
    enc_cos = load("enc_cos", (1, SEQ_TXT, -1))
    enc_sin = load("enc_sin", (1, SEQ_TXT, -1))
    cos = load("cos", (1, SEQ_TXT + seq_img, 1, 64))
    sin = load("sin", (1, SEQ_TXT + seq_img, 1, 64))
    temb = load("temb", (STEPS, -1))
    dsigma = load("dsigma", (STEPS,))
    latents = load("latents0", (1, SEQ_IMG, PACKED_CH))
    image_latents = (load("image_latents", (1, SEQ_IMG, PACKED_CH))
                     if editing else None)
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
        tokens = (torch.cat([latents, image_latents], dim=1)
                  if editing else latents)
        noise_pred = denoise_step(tokens, prompt_embeds, temb[step:step + 1],
                                  cos, sin, prefix)
        noise_pred = noise_pred[:, :SEQ_IMG]  # drop the reference tokens
        latents = latents + dsigma[step] * noise_pred
        print(f"[step {step}] |noise| {noise_pred.norm():.2f}  "
              f"|latents| {latents.norm():.2f}")

    print("[vae] decoding ...")
    decoded = decode(latents, unpack_perm, unpatch_perm, bn_mean, bn_std)
    pixels = to_pixels(decoded)
    out_name = "device_loop.png" if _USE_DEVICE else "tflite_loop.png"
    Image.fromarray(pixels).save(os.path.join(BINS, out_name))

    reference_path = os.path.join(BINS, "ref_fp32.png")
    ref = np.asarray(Image.open(reference_path), dtype=np.float64)
    test = pixels.astype(np.float64)
    mse = ((ref - test) ** 2).mean()
    psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else 99.0
    corr = np.corrcoef(ref.flatten(), test.flatten())[0, 1]
    print(f"[gate] {where} loop vs fp32 pipeline: PSNR {psnr:.1f} dB  "
          f"corr {corr:.5f}  max px diff {np.abs(ref - test).max():.0f}/255")


if __name__ == "__main__":
    main()

"""Host prep for the on-device FLUX.2-klein loop: reference image + every .bin.

Runs the stock fp32 Flux2KleinPipeline once, recording the exact per-step inputs
the DiT sees, and writes the fixed tensors the device needs. The device then
runs only the twelve int8 graphs; everything else is cheap host arithmetic.

Device loop (no CFG, no sign flip -- klein is step-wise distilled):
    h9  = enc0(inputs_embeds, mask, enc_cos, enc_sin)
    h18 = enc1(h9, ...)   ;   h27 = enc2(h18, ...)
    prompt_embeds = interleave(h9, h18, h27)                    [1, 512, 7680]
    for s in steps:
        joint = double/single chunks over (latents, prompt_embeds, temb[s])
        noise_pred = kc_final(joint, temb[s])
        latents = latents + dsigma[s] * noise_pred
    x = unpack(latents) -> BN denorm -> unpatchify -> kv_vae -> image

With `--edit` the pipeline is given a reference image. Its VAE latent tokens are
appended to the noise tokens every step, `cat([latents, image_latents], dim=1)`,
which doubles the image sequence and needs the `kce_*` graphs. The reference is
constant across steps, so the host stages it once as `image_latents.bin`. Only
the first `SEQ_IMG` output tokens are the noise prediction.

The reference must be square and `SIZE` px: `Flux2KleinPipeline` derives the
reference token count from the image's OWN size, so a 1546x1213 photo would ask
the DiT for 7200 tokens instead of 256. The host center-crops and resizes, which
is what the app does too.

Usage:
    python gen_prep_klein.py           # text-to-image
    python gen_prep_klein.py --edit    # image editing
    python gen_prep_klein.py --edit --image photo.jpg --prompt "..." --bins out/
"""
import argparse
import os
import sys

import numpy as np
import torch

REPO = "black-forest-labs/FLUX.2-klein-4B"
PROMPT = "a red apple on a wooden table, studio lighting"
EDIT_PROMPT = "turn the apple into a green apple"
EDIT_IMAGE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "docs", "pixel8a_generated.png")
STEPS, SIZE, SEED = 4, 256, 1234
SEQ_TXT, SEQ_IMG = 512, 256
LATENT_CH, LATENT_HW = 32, 32
TAPS = (9, 18, 27)
ENC_HEADS = 32
NEG_INF = -1e9
OUT = "klein_bins"


def parse_args():
    """Parses the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit", action="store_true")
    parser.add_argument("--image", default=EDIT_IMAGE)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--bins", default=OUT)
    return parser.parse_args()


def square_reference(path):
    """Center-crops and resizes to SIZE, so the DiT sees SEQ_IMG tokens.

    Args:
        path: Any image file, squared about its center.

    Returns:
        A SIZE x SIZE PIL image.
    """
    from PIL import Image

    image = Image.open(path).convert("RGB")
    side = min(image.size)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    return image.crop((left, top, left + side, top + side)).resize(
        (SIZE, SIZE), Image.BICUBIC)


def save(tensor, name, out_dir=OUT):
    """Writes a tensor as little-endian float32 to <out_dir>/<name>.bin.

    Args:
        tensor: A torch tensor or numpy array.
        name: File name, without the .bin suffix.
        out_dir: Directory to write into.
    """
    array = (tensor.detach().cpu().numpy() if torch.is_tensor(tensor)
             else np.asarray(tensor))
    array.astype("<f4").tofile(f"{out_dir}/{name}.bin")


def save_int(array, name, out_dir=OUT):
    """Writes an index array as little-endian int32 to <out_dir>/<name>.bin.

    Args:
        array: An index array.
        name: File name, without the .bin suffix.
        out_dir: Directory to write into.
    """
    np.asarray(array, dtype="<i4").tofile(f"{out_dir}/{name}.bin")


def build_text_inputs(pipe, prompt):
    """Tokenizes the prompt and builds the encoder's host-side inputs.

    Returns (inputs_embeds, mask4d, cos, sin, valid_token_count). The 4D mask is
    causal plus a padded-key block: padded query rows still attend to real keys,
    and their outputs do reach the DiT (klein feeds all 512 text tokens), so the
    mask has to match what transformers builds from the 2D attention_mask.

    It is materialized across the head axis ([1, H, S, S], not [1, 1, S, S]):
    ML Drift miscompiles an implicitly broadcast ADD on a BATCH_MATMUL result.

    Args:
        pipe: The stock `Flux2KleinPipeline`.
        prompt: The prompt to encode.

    Returns:
        (inputs_embeds, mask4d, cos, sin, valid_token_count).
    """
    qwen = pipe.text_encoder.model
    tokenizer = pipe.tokenizer
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    encoded = tokenizer(text, return_tensors="pt", padding="max_length",
                        truncation=True, max_length=SEQ_TXT)
    with torch.no_grad():
        inputs_embeds = qwen.embed_tokens(encoded["input_ids"])
        position_ids = torch.arange(SEQ_TXT)[None]
        cos, sin = qwen.rotary_emb(inputs_embeds, position_ids)
    keep = encoded["attention_mask"][0].float()
    causal = torch.full((SEQ_TXT, SEQ_TXT), NEG_INF).triu(1)
    pad = torch.where(keep > 0, 0.0, NEG_INF)
    mask = (causal + pad[None, :])[None, None]
    mask = mask.expand(1, ENC_HEADS, SEQ_TXT, SEQ_TXT).contiguous()
    return inputs_embeds, mask, cos, sin, int(keep.sum())


def check_encoder_mask(pipe, inputs_embeds, mask, want):
    """Confirms the host 4D mask reproduces the pipeline's prompt_embeds.

    Args:
        pipe: The stock `Flux2KleinPipeline`.
        inputs_embeds: Token embeddings fed to the encoder.
        mask: The host-built 4D mask.
        want: The pipeline's own prompt_embeds.
    """
    qwen = pipe.text_encoder.model
    with torch.no_grad():
        states = qwen(inputs_embeds=inputs_embeds, attention_mask=mask,
                      position_ids=torch.arange(SEQ_TXT)[None], use_cache=False,
                      output_hidden_states=True).hidden_states
    stacked = torch.stack([states[k] for k in TAPS], dim=1)
    got = stacked.permute(0, 2, 1, 3).reshape(1, SEQ_TXT, -1)
    corr = torch.corrcoef(torch.stack([want.flatten(), got.flatten()]))[0, 1]
    print(f"[enc-mask] prompt_embeds corr {corr:.8f}  "
          f"max|diff| {(want - got).abs().max():.2e}")


def patchify_permutation(pipe):
    """The 2x2 latent patchify, recovered as a flat gather index map.

    Editing needs the reverse of the decode tail: the app VAE-encodes the picked
    image to [32, 32, 32] and must reach the DiT's [128, 16, 16] packing. That
    reshape/permute is a pure permutation, so an arange probe recovers it and
    the app applies it as a gather -- no reshape logic duplicated in Kotlin.

    Args:
        pipe: The stock `Flux2KleinPipeline`.

    Returns:
        A flat gather index map.
    """
    probe = torch.arange(LATENT_CH * LATENT_HW * LATENT_HW).float()
    probe = probe.reshape(1, LATENT_CH, LATENT_HW, LATENT_HW)
    patched = pipe._patchify_latents(probe)
    perm = patched.flatten().round().long().numpy()
    pure = np.array_equal(np.sort(perm), np.arange(perm.size))
    print(f"[perm] patchify {tuple(patched.shape)} pure={pure}")
    return perm


def tail_permutations(pipe, img_ids):
    """Recovers the two tail reorderings as flat gather indices.

    Both `_unpack_latents_with_ids` (scatter by position id) and
    `_unpatchify_latents` (2x2 reshape/permute) are pure permutations, so an
    arange probe recovers them exactly and the device applies them as a gather.

    `img_ids` must be the NOISE ids only: editing appends reference ids, but the
    pipeline unpacks with `latent_ids`, never with the concatenated table.

    Args:
        pipe: The stock `Flux2KleinPipeline`.
        img_ids: The NOISE position ids only.

    Returns:
        (unpack_perm, unpatch_perm) as flat gather index maps.
    """
    probe = torch.arange(SEQ_IMG * 128).float().reshape(1, SEQ_IMG, 128)
    unpacked = pipe._unpack_latents_with_ids(probe, img_ids[None], 16, 16)
    unpack_perm = unpacked.flatten().round().long().numpy()

    probe = torch.arange(128 * 16 * 16).float().reshape(1, 128, 16, 16)
    unpatched = pipe._unpatchify_latents(probe)
    unpatch_perm = unpatched.flatten().round().long().numpy()

    pure_unpack = np.array_equal(np.sort(unpack_perm), np.arange(SEQ_IMG * 128))
    unpatch_size = 128 * 16 * 16
    pure_unpatch = np.array_equal(np.sort(unpatch_perm),
                                  np.arange(unpatch_size))
    print(f"[perm] unpack {tuple(unpacked.shape)} pure={pure_unpack}   "
          f"unpatchify {tuple(unpatched.shape)} pure={pure_unpatch}")
    return unpack_perm, unpatch_perm


def main():
    """Runs the fp32 pipeline once and stages everything the app needs."""
    from diffusers import Flux2KleinPipeline

    args = parse_args()
    out_dir = args.bins
    os.makedirs(out_dir, exist_ok=True)

    editing = args.edit
    prompt = args.prompt or (EDIT_PROMPT if editing else PROMPT)
    reference = square_reference(args.image) if editing else None
    mode = "image editing" if editing else "text-to-image"
    print(f"[mode] {mode}: {prompt!r}")
    if editing:
        print(f"[ref ] {args.image} -> {SIZE}x{SIZE}")

    print("[load] Flux2KleinPipeline (fp32) ...")
    pipe = Flux2KleinPipeline.from_pretrained(
        REPO, torch_dtype=torch.float32).to("cpu")
    transformer = pipe.transformer
    stock_forward = transformer.forward
    rec = {"temb": []}

    def hooked(hidden_states=None, encoder_hidden_states=None, timestep=None,
               guidance=None, img_ids=None, txt_ids=None, **kwargs):
        """Records the DiT's inputs, then defers to the stock fp32 forward."""
        with torch.no_grad():
            scaled = timestep.to(hidden_states.dtype) * 1000
            embedded = transformer.time_guidance_embed(scaled, guidance)
            rec["temb"].append(embedded.clone())
            if "cos" not in rec:
                image_ids = img_ids[0] if img_ids.ndim == 3 else img_ids
                text_ids = txt_ids[0] if txt_ids.ndim == 3 else txt_ids
                img_cos, img_sin = transformer.pos_embed(image_ids)
                txt_cos, txt_sin = transformer.pos_embed(text_ids)
                full_cos = torch.cat([txt_cos, img_cos], 0)
                full_sin = torch.cat([txt_sin, img_sin], 0)
                rec["cos"] = full_cos[:, 0::2][None, :, None, :]
                rec["sin"] = full_sin[:, 0::2][None, :, None, :]
                rec["img_ids"] = image_ids.clone()
                rec["prompt_embeds"] = encoder_hidden_states.clone()
                rec["latents0"] = hidden_states.clone()
        return stock_forward(hidden_states=hidden_states, timestep=timestep,
                             guidance=guidance,
                             encoder_hidden_states=encoder_hidden_states,
                             img_ids=img_ids, txt_ids=txt_ids, **kwargs)

    transformer.forward = hooked
    generator = torch.Generator("cpu").manual_seed(SEED)
    image = pipe(image=reference, prompt=prompt, height=SIZE, width=SIZE,
                 guidance_scale=1.0, num_inference_steps=STEPS,
                 generator=generator).images[0]
    transformer.forward = stock_forward
    image.save(f"{out_dir}/ref_fp32.png")
    if editing:
        reference.save(f"{out_dir}/edit_source.png")

    sigmas = pipe.scheduler.sigmas.detach().cpu().numpy()
    dsigma = sigmas[1:STEPS + 1] - sigmas[:STEPS]
    print(f"[sched] sigmas {[round(float(s), 4) for s in sigmas]}")
    print(f"[sched] dsigma {[round(float(s), 4) for s in dsigma]}")

    inputs_embeds, mask, enc_cos, enc_sin, valid = build_text_inputs(
        pipe, prompt)
    print(f"[enc] {valid}/{SEQ_TXT} real tokens, rest padded")
    check_encoder_mask(pipe, inputs_embeds, mask, rec["prompt_embeds"])

    # Editing hands the DiT `cat([latents, image_latents])`. The reference half
    # is constant across steps; the tail permutations index the noise half only.
    image_latents = None
    noise_latents = rec["latents0"]
    noise_ids = rec["img_ids"]
    if editing:
        noise_latents = rec["latents0"][:, :SEQ_IMG]
        image_latents = rec["latents0"][:, SEQ_IMG:]
        noise_ids = rec["img_ids"][:SEQ_IMG]
        joint = rec["latents0"].shape[1]
        print(f"[edit] joint image tokens {joint} = "
              f"{noise_latents.shape[1]} noise + "
              f"{image_latents.shape[1]} reference")
        print(f"[edit] noise T={noise_ids[0, 0]:.0f}  "
              f"reference T={rec['img_ids'][SEQ_IMG, 0]:.0f}")

    unpack_perm, unpatch_perm = tail_permutations(pipe, noise_ids)
    patch_perm = patchify_permutation(pipe) if editing else None

    bn_mean = pipe.vae.bn.running_mean.detach()
    bn_std = torch.sqrt(pipe.vae.bn.running_var.detach()
                        + pipe.vae.config.batch_norm_eps)

    save(inputs_embeds, "inputs_embeds", out_dir)
    save(mask, "enc_mask", out_dir)
    save(enc_cos, "enc_cos", out_dir)
    save(enc_sin, "enc_sin", out_dir)
    save(rec["prompt_embeds"], "prompt_embeds", out_dir)
    save(rec["cos"], "cos", out_dir)
    save(rec["sin"], "sin", out_dir)
    save(torch.cat(rec["temb"], 0), "temb", out_dir)
    save(dsigma, "dsigma", out_dir)
    save(noise_latents, "latents0", out_dir)
    save(bn_mean, "bn_mean", out_dir)
    save(bn_std, "bn_std", out_dir)
    save_int(unpack_perm, "unpack_perm", out_dir)
    save_int(unpatch_perm, "unpatch_perm", out_dir)
    if editing:
        save(image_latents, "image_latents", out_dir)
        save_int(patch_perm, "patch_perm", out_dir)
    print(f"[gen_prep] wrote {out_dir}/ ({STEPS} steps, {SIZE} px)")


if __name__ == "__main__":
    main()

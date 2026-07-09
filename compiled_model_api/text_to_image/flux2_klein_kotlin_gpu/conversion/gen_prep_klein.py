"""Host prep for the on-device FLUX.2-klein loop: reference image + every .bin.

Runs the stock fp32 Flux2KleinPipeline once, recording the exact per-step inputs
the DiT sees, and writes the fixed tensors the device needs. The device runs
only the twelve int8 graphs. Everything else is cheap host arithmetic.

Device loop (no CFG, no sign flip -- klein is step-wise distilled):
    h9  = enc0(inputs_embeds, mask, enc_cos, enc_sin)
    h18 = enc1(h9, ...)
    h27 = enc2(h18, ...)
    prompt_embeds = interleave(h9, h18, h27)                    [1, 512, 7680]
    for s in steps:
        joint = double/single chunks over (latents, prompt_embeds, temb[s])
        noise_pred = kc_final(joint, temb[s])
        latents = latents + dsigma[s] * noise_pred
    x = unpack(latents) -> BN denorm -> unpatchify -> kv_vae -> image
"""
import os

import numpy as np
import torch

REPO = "black-forest-labs/FLUX.2-klein-4B"
PROMPT = "a red apple on a wooden table, studio lighting"
STEPS, SIZE, SEED = 4, 256, 1234
SEQ_TXT, SEQ_IMG = 512, 256
TAPS = (9, 18, 27)
ENC_HEADS = 32
NEG_INF = -1e9
OUT = "klein_bins"


def save(tensor, name):
    """Writes a tensor as little-endian float32.

    Args:
      tensor: Tensor or array to write.
      name: Output basename, writes {OUT}/[name].bin.
    """
    if torch.is_tensor(tensor):
        array = tensor.detach().cpu().numpy()
    else:
        array = np.asarray(tensor)
    array.astype("<f4").tofile(f"{OUT}/{name}.bin")


def save_int(array, name):
    """Writes an index array as little-endian int32.

    Args:
      array: Index array to write.
      name: Output basename, writes {OUT}/[name].bin.
    """
    np.asarray(array, dtype="<i4").tofile(f"{OUT}/{name}.bin")


def build_text_inputs(pipe):
    """Tokenizes the prompt and builds the encoder's host-side inputs.

    The 4D mask is causal plus a padded-key block. Padded query rows still
    attend to real keys, and their outputs do reach the DiT, since klein feeds
    all 512 text tokens, so the mask has to match what transformers builds from
    the 2D attention_mask. It is materialized across the head axis
    ([1, H, S, S], not [1, 1, S, S]) because ML Drift miscompiles an implicitly
    broadcast ADD on a BATCH_MATMUL result.

    Args:
      pipe: The loaded Flux2KleinPipeline.

    Returns:
      A tuple (inputs_embeds, mask, cos, sin, valid_token_count).
    """
    qwen = pipe.text_encoder.model
    tokenizer = pipe.tokenizer
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}], tokenize=False,
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
      pipe: The loaded Flux2KleinPipeline.
      inputs_embeds: Host-built token embeddings.
      mask: Host-built 4D attention mask.
      want: The prompt_embeds the pipeline itself produced.
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


def tail_permutations(pipe, img_ids):
    """Recovers the two tail reorderings as flat gather indices.

    Both `_unpack_latents_with_ids` (scatter by position id) and
    `_unpatchify_latents` (2x2 reshape/permute) are pure permutations, so an
    arange probe recovers them exactly and the device applies them as a gather.

    Args:
      pipe: The loaded Flux2KleinPipeline.
      img_ids: The image position ids the pipeline gave to the DiT.

    Returns:
      A tuple (unpack_perm, unpatch_perm) of flat int index maps.
    """
    probe = torch.arange(SEQ_IMG * 128).float().reshape(1, SEQ_IMG, 128)
    unpacked = pipe._unpack_latents_with_ids(probe, img_ids[None], 16, 16)
    unpack_perm = unpacked.flatten().round().long().numpy()

    probe = torch.arange(128 * 16 * 16).float().reshape(1, 128, 16, 16)
    unpatched = pipe._unpatchify_latents(probe)
    unpatch_perm = unpatched.flatten().round().long().numpy()

    pure_unpack = np.array_equal(
        np.sort(unpack_perm), np.arange(SEQ_IMG * 128))
    pure_unpatch = np.array_equal(
        np.sort(unpatch_perm), np.arange(128 * 16 * 16))
    print(f"[perm] unpack {tuple(unpacked.shape)} pure={pure_unpack}   "
          f"unpatchify {tuple(unpatched.shape)} pure={pure_unpatch}")
    return unpack_perm, unpatch_perm


def main():
    """Renders the fp32 reference image and stages every device input."""
    from diffusers import Flux2KleinPipeline
    os.makedirs(OUT, exist_ok=True)

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
                cos = torch.cat([txt_cos, img_cos], 0)[:, 0::2]
                sin = torch.cat([txt_sin, img_sin], 0)[:, 0::2]
                rec["cos"] = cos[None, :, None, :]
                rec["sin"] = sin[None, :, None, :]
                rec["img_ids"] = image_ids.clone()
                rec["prompt_embeds"] = encoder_hidden_states.clone()
                rec["latents0"] = hidden_states.clone()
        return stock_forward(hidden_states=hidden_states, timestep=timestep,
                             guidance=guidance,
                             encoder_hidden_states=encoder_hidden_states,
                             img_ids=img_ids, txt_ids=txt_ids, **kwargs)

    transformer.forward = hooked
    generator = torch.Generator("cpu").manual_seed(SEED)
    image = pipe(prompt=PROMPT, height=SIZE, width=SIZE, guidance_scale=1.0,
                 num_inference_steps=STEPS, generator=generator).images[0]
    transformer.forward = stock_forward
    image.save(f"{OUT}/ref_fp32.png")

    sigmas = pipe.scheduler.sigmas.detach().cpu().numpy()
    dsigma = sigmas[1:STEPS + 1] - sigmas[:STEPS]
    print(f"[sched] sigmas {[round(float(s), 4) for s in sigmas]}")
    print(f"[sched] dsigma {[round(float(s), 4) for s in dsigma]}")

    inputs_embeds, mask, enc_cos, enc_sin, valid = build_text_inputs(pipe)
    print(f"[enc] {valid}/{SEQ_TXT} real tokens, rest padded")
    check_encoder_mask(pipe, inputs_embeds, mask, rec["prompt_embeds"])
    unpack_perm, unpatch_perm = tail_permutations(pipe, rec["img_ids"])

    bn_mean = pipe.vae.bn.running_mean.detach()
    bn_std = torch.sqrt(pipe.vae.bn.running_var.detach()
                        + pipe.vae.config.batch_norm_eps)

    save(inputs_embeds, "inputs_embeds")
    save(mask, "enc_mask")
    save(enc_cos, "enc_cos")
    save(enc_sin, "enc_sin")
    save(rec["prompt_embeds"], "prompt_embeds")
    save(rec["cos"], "cos")
    save(rec["sin"], "sin")
    save(torch.cat(rec["temb"], 0), "temb")
    save(dsigma, "dsigma")
    save(rec["latents0"], "latents0")
    save(bn_mean, "bn_mean")
    save(bn_std, "bn_std")
    save_int(unpack_perm, "unpack_perm")
    save_int(unpatch_perm, "unpatch_perm")
    print(f"[gen_prep] wrote {OUT}/ ({STEPS} steps, {SIZE} px) + ref_fp32.png")


if __name__ == "__main__":
    main()

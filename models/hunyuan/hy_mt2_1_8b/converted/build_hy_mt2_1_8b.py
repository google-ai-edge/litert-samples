# Copyright 2026 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""tencent/Hy-MT2-1.8B (dense HunYuanDenseV1ForCausalLM) -> .litertlm.

Converts the Hugging Face checkpoint into a LiteRT-LM bundle through
litert-torch's plain HF export path with the stock int8 recipe
(`dynamic_wi8_afp32`: dynamic int8 weights on linears + embedding): 1.82 GB for
1.79 B unique parameters. The architecture itself needs nothing — GQA with
QK-norm, RMSNorm, SiLU MLP, a tied embedding — and the stock exporter handles
it. What this script adds is two things the checkpoint's *config* and
*template* make necessary, both measured:

  * A bitwise-equal rope bake. The config declares
    `rope_scaling: {type: dynamic, alpha: 1000}`. transformers 5.14 resolves
    that STATICALLY at init (`base = rope_theta * alpha**(dim/(dim-2))`; the
    code comment calls it "DynamicNTKAlphaRotary"), but the forward still
    carries the generic
    dynamic-rope growth branch, and its data-dependent
    `if seq_len > max_seq_len_cached` guard kills torch.export
    (GuardOnDataDependentSymNode: Could not guard on Eq(u0, 1)). Baking the
    resolved base into `rope_theta` (11,158,839.925…) and dropping
    `rope_scaling` removes the dead branch and nothing else: `inv_freq` and
    teacher-forced logits are bitwise-equal to the HF reference, valid for
    every position up to `max_position_embeddings` (262,144) — far past this
    bundle's 4,096 context. `verify_hy_mt2_1_8b.py --rope-ab` re-measures it.

  * No metadata `start_token`. The chat template renders
    `<｜hy_begin▁of▁sentence｜>` itself as a LITERAL, the tokenizer adds no BOS
    at encode time, and the LiteRT-LM engine prepends the metadata
    `start_token` to every rendered prompt unconditionally — so the default
    export fed the model two BOS tokens. The proof runs inside the runtime:
    `[start_token] + rest` (default bundle, `--no-template`) generates
    byte-identical greedy output to `[template BOS] + rest` (this bundle,
    normal templating) on 3 of 3 probes (`--bos-discriminator`). Dropping the
    field makes the on-device stream equal the training stream.

Usage:
  python build_hy_mt2_1_8b.py --out out_int8
  python build_hy_mt2_1_8b.py --out out_int8_bos --keep-start-token   # study
"""

import argparse
import json
import os
import re
import shutil
import tempfile
from pathlib import Path

MODEL_ID = "tencent/Hy-MT2-1.8B"
FINAL_NAME = "Hy-MT2-1.8B_int8.litertlm"
EOS_ID = 120020  # <｜hy_place▁holder▁no▁2｜>, the tokenizer's eos


def normalize_rope(snapshot, out_dir):
  """Builds a hub-format dir whose config carries the statically resolved rope.

  Everything but config.json is a symlink into the snapshot. The bake is the
  exact expression transformers' hunyuan rotary embedding evaluates at init.
  """
  cfg = json.loads((snapshot / "config.json").read_text())
  rope = cfg.get("rope_scaling") or {}
  if rope.get("type") != "dynamic" or not rope.get("alpha"):
    raise SystemExit(
        f"unexpected rope_scaling {rope!r} — this recipe bakes the "
        "dynamic-alpha form only; re-check the checkpoint's config")
  dim = cfg["head_dim"]
  baked = cfg["rope_theta"] * rope["alpha"] ** (dim / (dim - 2))

  norm = out_dir / "rope_normalized_src"
  norm.mkdir(parents=True, exist_ok=True)
  for f in snapshot.iterdir():
    if f.name == "config.json" or f.is_dir():
      continue
    dst = norm / f.name
    if not dst.exists():
      dst.symlink_to(f.resolve())
  norm_cfg = {k: v for k, v in cfg.items() if k != "rope_scaling"}
  norm_cfg["rope_theta"] = baked
  (norm / "config.json").write_text(json.dumps(norm_cfg, indent=1))
  print(f"rope: type=dynamic alpha={rope['alpha']} resolves statically -> "
        f"rope_theta={baked!r} ({baked.hex()}), rope_scaling dropped; "
        f"bitwise-equal below {cfg.get('max_position_embeddings')} positions")
  return norm


def strip_block(pbtext, field="start_token"):
  """Drops a top-level `field { ... }` block from a text-format proto.

  Brace-counting rather than a regex: the block nests braces, and the
  metadata's stop_tokens use the same shape.
  """
  m = re.search(rf"^{field}\s*\{{", pbtext, re.M)
  if not m:
    return pbtext, False
  i = pbtext.index("{", m.start())
  depth = 0
  for j in range(i, len(pbtext)):
    if pbtext[j] == "{":
      depth += 1
    elif pbtext[j] == "}":
      depth -= 1
      if depth == 0:
        end = j + 1
        while end < len(pbtext) and pbtext[end] == "\n":
          end += 1
        return pbtext[:m.start()] + pbtext[end:], True
  raise SystemExit(f"unbalanced braces in {field} block")


def bos_evidence(tokenizer):
  """Counts the leading BOS the model was trained to see vs what the bundle
  would feed: (reference, from_template). The engine renders the template
  with `bos_token` unbound and prepends the metadata start_token, so the
  bundle count is from_template + 1."""
  import jinja2

  def render(bos):
    env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])
    return env.from_string(tokenizer.chat_template).render(
        messages=[{"role": "user", "content": "x"}], add_generation_prompt=True,
        bos_token=bos, eos_token="")

  def leading(text, tok):
    n = 0
    while tok and text.startswith(tok):
      text, n = text[len(tok):], n + 1
    return n

  bos = tokenizer.bos_token
  return leading(render(bos), bos), leading(render(""), bos)


def finalize_metadata(bundle, keep_start_token):
  """Post-export pass over LlmMetadata: assert the eos stop, drop start_token."""
  from litert_lm_builder import pack_litertlm_file, unpack_litertlm_file

  with tempfile.TemporaryDirectory(dir=bundle.parent) as td:
    toml = unpack_litertlm_file(str(bundle), td)
    pb = Path(td) / "LlmMetadataProto.pbtext"
    text = pb.read_text()
    stop_ids = {int(x) for x in re.findall(r"ids:\s*(\d+)", text)}
    if EOS_ID not in stop_ids:
      raise SystemExit(f"eos {EOS_ID} missing from stop_tokens {sorted(stop_ids)} "
                       "— the exporter's stop derivation changed; re-check")
    text, had = strip_block(text, "start_token")
    if not had:
      raise SystemExit("no start_token in the export — the bundler's behaviour "
                       "changed; re-check the BOS story before shipping")
    if keep_start_token:
      print("start_token: KEPT on request (double-BOS bundle, for study only)")
      return
    pb.write_text(text)
    fixed = bundle.parent / (bundle.name + ".nobos")
    pack_litertlm_file(str(toml), str(fixed))
    fixed.replace(bundle)
  print("start_token: dropped (template renders the BOS literal; the engine "
        "would have prepended a second one)")


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default=MODEL_ID,
                  help="HF model id or a local checkout of it")
  ap.add_argument("--out", default="out_int8", help="output dir")
  ap.add_argument("--cache", type=int, default=4096, help="KV cache length")
  ap.add_argument("--prefill", default="128",
                  help="comma-separated prefill signature lengths (the "
                       "published bundle carries the exporter's stock single "
                       "128-token signature)")
  ap.add_argument("--keep-start-token", action="store_true",
                  help="do not drop the metadata start_token: reproduces the "
                       "default double-BOS export for --bos-discriminator")
  ap.add_argument("--keep-src", action="store_true",
                  help="keep the rope-normalized source dir under --out")
  args = ap.parse_args()

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)

  from huggingface_hub import snapshot_download
  from transformers import AutoTokenizer

  src = Path(args.model)
  if not src.exists():
    src = Path(snapshot_download(args.model,
                                 ignore_patterns=["train/*", "imgs/*", "*.png"]))
  tok = AutoTokenizer.from_pretrained(str(src))
  want, from_template = bos_evidence(tok)
  print(f"BOS: reference stream starts with {want} x {tok.bos_token!r}; the "
        f"template renders {from_template} itself and the engine prepends the "
        f"metadata start_token -> a default export feeds {from_template + 1}")
  assert (want, from_template) == (1, 1), "BOS shape changed — re-derive the story"

  norm = normalize_rope(src, out)

  from litert_torch.generative.export_hf.export import export

  export(
      model=str(norm),
      output_dir=str(out),
      prefill_lengths=[int(x) for x in args.prefill.split(",") if x.strip()],
      cache_length=args.cache,
      # The exporter's stock int8: dynamic int8 weights on linears + embedding,
      # fp32 activations. Same string the 0.9.3 defaults resolve to.
      quantization_recipe="dynamic_wi8_afp32",
      # The vendor Jinja is embedded verbatim and rendered on device.
      use_jinja_template=True,
      externalize_embedder=False,
  )
  bundle = out / "model.litertlm"
  if not bundle.exists():
    raise SystemExit("export finished without producing model.litertlm")

  finalize_metadata(bundle, args.keep_start_token)
  final = out / FINAL_NAME
  bundle.replace(final)
  if not args.keep_src:
    shutil.rmtree(norm, ignore_errors=True)
  print(f"EXPORT_DONE {final} ({os.path.getsize(final):,} B)")


if __name__ == "__main__":
  main()

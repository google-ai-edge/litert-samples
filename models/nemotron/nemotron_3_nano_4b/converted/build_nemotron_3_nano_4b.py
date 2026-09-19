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

"""nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 (NemotronHForCausalLM hybrid) -> .litertlm.

A three-kind hybrid: 21 Mamba2 selective-scan layers + 17 plain MLP layers +
4 grouped-query attention layers (42 in all, pattern
`M-M-M-MM-M-M*-M-M*-M-M-M*-M-M-MM*-MMM-M-M-`). No released litert-torch
carries a Mamba2 cache layer, so this recipe pins a litert-torch checkout
(115a1360, 2026-06-19) and applies `nemotron_h_litert_torch.patch` — the
folded rank<=4 SSD scan port, a cache-less layer type for the MLP blocks, and
the generic hybrid cache — then runs the family recipe:

  1. float export through the patched exporter (bundle, embedded Jinja,
     4096-token KV budget for the 4 attention layers, reduced 7-signature
     prefill ladder 1024/256/64/16/4/1 + decode);
  2. post-hoc dynamic int8 on the linears and the embedding ONLY — the
     causal convolutions and the selective scan stay float — the rule every
     published bundle of this and the sibling hybrid families ships with
     (an export-time-int8 comparison on another family's hybrid scored
     lower; cause not isolated);
  3. an ExecutorMetadata section: litert-lm >= 0.15 binds per-layer state
     buffers through it, and the pinned exporter does not write one. The
     50 buffers (42 mamba conv+SSM, 8 attention K/V) are named from the
     decode signature;
  4. `<|im_end|>` (id 11) added to the stop set — generation_config declares
     only id 2, and the vendor template's ChatML turns end with <|im_end|>;
  5. the metadata `start_token` (`<s>`) dropped: the tokenizer says
     `add_bos_token: False` and the template opens with `<|im_start|>`, so a
     start token would be a stream the model never trained on. At 4B the
     model is robust either way (gate 7/8 with and without; HF greedy
     byte-identical on 2 of 3 probes) — this aligns the runtime stream with
     the training stream, it does not rescue anything;
  6. `prefer_activation_type = fp32` declared on the model section: the GPU
     executor runs activations in fp16 by default and the SSM scan's
     intermediates do not fit that range.

Every step edits the bundle through the litert-lm-builder pack/unpack API;
the weights are quantized once and never touched again.

Usage:
  python build_nemotron_3_nano_4b.py --setup             # one-time: clone + pin + patch
  python build_nemotron_3_nano_4b.py --out out_int8      # ~9 min on an M4 Max; 16 GB float intermediate
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MODEL_ID = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
FINAL_NAME = "Nemotron-3-Nano-4B_int8.litertlm"
HERE = Path(__file__).resolve().parent
PATCH = HERE / "nemotron_h_litert_torch.patch"
PIN = "115a13607c730c81018bb9789138a3e5e5119e3d"
LITERT_TORCH_GIT = "https://github.com/google-ai-edge/litert-torch"
IM_END_ID = 11  # <|im_end|>, the tokenizer's eos


def setup_checkout(checkout):
  """Clones litert-torch at the pinned commit and applies the hybrid patch."""
  if (checkout / "litert_torch/generative/export_hf/model_ext/nemotron_h/patch.py").exists():
    print(f"checkout already set up: {checkout}")
    return
  if not checkout.exists():
    subprocess.run(["git", "clone", LITERT_TORCH_GIT, str(checkout)], check=True)
  subprocess.run(["git", "-C", str(checkout), "fetch", "origin", PIN], check=True)
  subprocess.run(["git", "-C", str(checkout), "checkout", PIN], check=True)
  subprocess.run(["git", "-C", str(checkout), "apply", str(PATCH)], check=True)
  print(f"checkout ready: {checkout} @ {PIN[:8]} + {PATCH.name}")


def use_checkout(checkout):
  """Puts the patched tree first on sys.path and proves it is what imports."""
  ext = checkout / "litert_torch/generative/export_hf/model_ext/nemotron_h/patch.py"
  if not ext.exists():
    raise SystemExit(f"{checkout} is not the patched checkout — run --setup first")
  sys.path.insert(0, str(checkout))
  import litert_torch  # noqa: E402
  got = Path(litert_torch.__file__).resolve()
  if checkout.resolve() not in got.parents:
    raise SystemExit(f"litert_torch imported from {got}, not the pinned checkout")
  print(f"litert_torch: {got.parent} (pinned {PIN[:8]} + patch)")


def expected_state_buffers(model_dir):
  cfg = json.loads((Path(model_dir) / "config.json").read_text())
  pat = cfg["hybrid_override_pattern"]
  return 2 * pat.count("M"), 2 * pat.count("*"), pat


def float_export(model, out, prefill, cache):
  from litert_torch.generative.export_hf.export import export

  export(
      model=model,
      output_dir=str(out),
      prefill_lengths=prefill,
      cache_length=cache,
      # "" (not None) — None would fall through to the exporter's default
      # dynamic int8, which quantizes the convolutions too. The int8 recipe
      # this family ships is applied post hoc in quantize_int8().
      quantization_recipe="",
      use_jinja_template=True,
      bundle_litert_lm=True,
  )
  bundle = out / "model.litertlm"
  if not bundle.exists():
    raise SystemExit("export finished without producing model.litertlm")
  print(f"float export: {bundle} ({os.path.getsize(bundle):,} B)")
  return bundle


def quantize_int8(tflite):
  """Dynamic int8 on FULLY_CONNECTED + EMBEDDING_LOOKUP (channelwise); convs
  and the scan stay float. Rewrites the tflite in place."""
  from ai_edge_quantizer import qtyping, quantizer, recipe_manager

  rm = recipe_manager.RecipeManager()
  rm.add_dynamic_config(regex=".*", operation_name=qtyping.TFLOperationName.FULLY_CONNECTED,
                        num_bits=8)
  rm.add_dynamic_config(regex=".*", operation_name=qtyping.TFLOperationName.EMBEDDING_LOOKUP,
                        num_bits=8, granularity=qtyping.QuantGranularity.CHANNELWISE)
  qt = quantizer.Quantizer(str(tflite), rm.get_quantization_recipe())
  if qt.need_calibration:
    raise SystemExit("recipe unexpectedly needs calibration")
  before = os.path.getsize(tflite)
  qpath = tflite.with_suffix(".int8.tflite")
  qt.quantize().export_model(str(qpath))
  qpath.replace(tflite)
  print(f"int8 (linears + embedding): {before / 1e9:.2f} GB -> "
        f"{os.path.getsize(tflite) / 1e9:.2f} GB")


def read_state_buffers(tflite):
  """[(name, shape)] for every kv_cache_* input of the decode signature."""
  from ai_edge_litert.interpreter import Interpreter

  it = Interpreter(model_path=str(tflite))
  sigs = it.get_signature_list()
  key = "decode" if "decode" in sigs else sorted(sigs)[0]
  details = it.get_signature_runner(key).get_input_details()
  names = sorted(n for n in sigs[key]["inputs"] if n.startswith("kv_cache_"))
  return [(n, [int(x) for x in details[n]["shape"]]) for n in names]


def executor_pbtext(buffers):
  """The ExecutorMetadata the runtime binds hybrid state through.

  Exporter naming: kv_cache_mc_N the mamba conv state, kv_cache_mr_N the
  mamba recurrent (SSM) state — both TYPE_LINEAR_ATTENTION, opaque to the
  executor (c/s and lc/lr are the ShortConv and linear-attention spellings of
  other hybrids); kv_cache_k_N / _v_N the attention caches with their
  sequence axis.
  """
  blocks = []
  for name, shape in buffers:
    kind = name.split("_")[2]
    if kind in ("mc", "mr", "c", "s", "lc", "lr"):
      typ, extra = "TYPE_LINEAR_ATTENTION", ""
    elif kind in ("k", "v"):
      typ = "TYPE_GLOBAL_KEY_CACHE" if kind == "k" else "TYPE_GLOBAL_VALUE_CACHE"
      axis = shape.index(max(shape))
      extra = f"\n    sequence_axis: {axis}\n    maximum_sequence_length: {max(shape)}"
    else:
      raise SystemExit(f"unknown state tensor kind for {name}")
    blocks.append(
        "  state_buffers {\n"
        f'    prefill_input_name: "{name}"\n'
        f'    prefill_output_name: "{name}"\n'
        f'    decode_input_name: "{name}"\n'
        f'    decode_output_name: "{name}"\n'
        f"    type: {typ}{extra}\n"
        "  }")
  return "llm_executor_metadata {\n  max_history_size: 0\n" + "\n".join(blocks) + "\n}\n"


def strip_block(pbtext, field="start_token"):
  """Drops a top-level `field { ... }` block from a text-format proto
  (brace-counting: the block nests, and stop_tokens share the shape)."""
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


def finalize(float_bundle, out, model_dir, keep_start_token, fp16_activations):
  """One unpack -> quantize + executor metadata + metadata edits -> one pack."""
  from litert_lm_builder import pack_litertlm_file, unpack_litertlm_file

  work = out / "work_unpack"
  shutil.rmtree(work, ignore_errors=True)
  toml_path = Path(unpack_litertlm_file(str(float_bundle), str(work)))
  toml = toml_path.read_text()
  m = re.search(r'section_type = "TFLiteModel"\n(?:.*\n)*?data_path = "([^"]+)"', toml)
  if not m:
    raise SystemExit("model.toml has no TFLiteModel section")
  tflite = work / m.group(1)

  quantize_int8(tflite)

  buffers = read_state_buffers(tflite)
  n_la = sum(1 for n, _ in buffers if n.split("_")[2] in ("mc", "mr", "c", "s", "lc", "lr"))
  n_kv = sum(1 for n, _ in buffers if n.split("_")[2] in ("k", "v"))
  want_la, want_kv, pat = expected_state_buffers(model_dir)
  print(f"state buffers: {len(buffers)} = {n_la} mamba conv+SSM + {n_kv} attention K/V "
        f"(layer pattern {pat})")
  if (n_la, n_kv) != (want_la, want_kv):
    raise SystemExit(f"expected {want_la} + {want_kv} state buffers from the config "
                     "— the patched cache layer did not export the layout it should")
  (work / "ExecutorMetadataProto.pbtext").write_text(executor_pbtext(buffers))
  marker = 'data_path = "LlmMetadataProto.pbtext"\n'
  if marker not in toml or "ExecutorMetadata" in toml:
    raise SystemExit("unexpected model.toml layout")
  toml = toml.replace(marker, marker + '\n[[section]]\nsection_type = "ExecutorMetadata"\n'
                      'data_path = "ExecutorMetadataProto.pbtext"\n')
  if not fp16_activations:
    toml = toml.replace('section_type = "TFLiteModel"',
                        'section_type = "TFLiteModel"\nprefer_activation_type = "fp32"')
  toml_path.write_text(toml)

  pb = work / "LlmMetadataProto.pbtext"
  text = pb.read_text()
  stop_ids = {int(x) for x in re.findall(r"ids:\s*(\d+)", text)}
  if IM_END_ID not in stop_ids:
    block = re.search(r"stop_tokens \{.*?\n\}", text, re.S)
    if not block:
      raise SystemExit("no stop_tokens block to extend")
    text = text.replace(block.group(0), block.group(0) +
                        "\nstop_tokens {\n  token_ids {\n    ids: %d\n  }\n}" % IM_END_ID, 1)
    print(f"stop tokens: added <|im_end|> ({IM_END_ID}) to {sorted(stop_ids)}")
  if not re.search(r"^start_token\s*\{", text, re.M):
    raise SystemExit("no start_token in the export — the bundler's behaviour changed; "
                     "re-check the BOS story before shipping")
  if keep_start_token:
    print("start_token: KEPT on request (<s>, a stream the model never trained on; "
          "for study only)")
  else:
    text, _ = strip_block(text, "start_token")
    print("start_token: dropped (add_bos_token False, template opens with <|im_start|>)")
  pb.write_text(text)

  final = out / FINAL_NAME
  if final.exists():
    final.unlink()
  pack_litertlm_file(str(toml_path), str(final))
  shutil.rmtree(work, ignore_errors=True)
  return final


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("--model", default=MODEL_ID,
                  help="HF model id or a local checkout of it")
  ap.add_argument("--out", default="out_int8", help="output dir")
  ap.add_argument("--checkout", default=str(HERE / "litert-torch-nemotron"),
                  help="the pinned, patched litert-torch tree (see --setup)")
  ap.add_argument("--setup", action="store_true",
                  help="clone litert-torch at the pinned commit into --checkout "
                       "and apply the patch, then exit")
  ap.add_argument("--cache", type=int, default=4096,
                  help="KV budget for the 4 attention layers (mamba state is "
                       "constant-size, MLP layers hold none)")
  ap.add_argument("--prefill", default="1024,256,64,16,4,1",
                  help="comma-separated prefill signature ladder (the published "
                       "bundle carries these six: every exported signature costs "
                       "engine RAM whether or not it is called)")
  ap.add_argument("--keep-float", action="store_true",
                  help="keep the float model.litertlm next to the int8 bundle")
  ap.add_argument("--keep-start-token", action="store_true",
                  help="do not drop the metadata start_token (study only)")
  ap.add_argument("--fp16-activations", action="store_true",
                  help="do not declare fp32 activations (study only: the GPU "
                       "executor then runs the scan in fp16)")
  args = ap.parse_args()

  checkout = Path(args.checkout)
  if args.setup:
    setup_checkout(checkout)
    return
  use_checkout(checkout)

  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  model_dir = Path(args.model)
  if not model_dir.exists():
    from huggingface_hub import snapshot_download
    model_dir = Path(snapshot_download(args.model))

  prefill = [int(x) for x in args.prefill.split(",") if x.strip()]
  float_bundle = float_export(args.model, out, prefill, args.cache)
  final = finalize(float_bundle, out, model_dir, args.keep_start_token,
                   args.fp16_activations)
  if not args.keep_float:
    float_bundle.unlink()
  print(f"EXPORT_DONE {final} ({os.path.getsize(final):,} B)")


if __name__ == "__main__":
  main()

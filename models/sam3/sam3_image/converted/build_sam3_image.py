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

"""Converts the SAM 3 image side to three LiteRT CompiledModel GPU graphs.

Order of operations, so that a failure is attributed to the right stage:

  1. Build the detector from the checkpoint and run the STOCK modules on a
     real image and prompt -- these outputs are the reference.
  2. Apply the re-authoring recipe (sam3_recipe.py) in place.
  3. Assert the re-authored torch modules still match the reference. Every
     rewrite is exact, so this gate is corr 1.0, not "close enough".
  4. Convert each graph with litert-torch, static-op-check the flatbuffer,
     fp16-quantize the matmul weights, and re-check parity of the fp16 graph
     through the CompiledModel CPU path.
  5. Emit the host-side artifacts: the fp16 token-embedding table and the CLIP
     BPE vocabulary.

Run:
  python build_sam3_image.py --ckpt sam3.1_multiplex.pt
  python build_sam3_image.py --ckpt sam3.1_multiplex.pt --only vision
"""

import argparse
import collections
import os
import shutil
import time

import numpy as np
import torch

import sam3_recipe as recipe


def op_histogram(path):
  """Reads a .tflite flatbuffer and counts operators and >4-D tensors.

  Uses the generated schema rather than an interpreter so the check works
  without a runtime that can allocate the graph.

  Args:
    path: Path to the .tflite file.

  Returns:
    A (Counter of op name -> count, number of >4-D tensors) tuple.
  """
  from ai_edge_litert import schema_py_generated as schema

  names = {
      value: key
      for key, value in vars(schema.BuiltinOperator).items()
      if isinstance(value, int)
  }
  with open(path, "rb") as f:
    buf = bytearray(f.read())
  model = schema.Model.GetRootAsModel(buf, 0)
  codes = []
  for i in range(model.OperatorCodesLength()):
    opcode = model.OperatorCodes(i)
    builtin = max(opcode.DeprecatedBuiltinCode(), opcode.BuiltinCode())
    custom = opcode.CustomCode()
    if custom:
      codes.append(custom.decode())
    else:
      codes.append(names.get(builtin, f"OP_{builtin}"))
  histogram = collections.Counter()
  over_rank4 = 0
  for s in range(model.SubgraphsLength()):
    subgraph = model.Subgraphs(s)
    for o in range(subgraph.OperatorsLength()):
      histogram[codes[subgraph.Operators(o).OpcodeIndex()]] += 1
    for t in range(subgraph.TensorsLength()):
      if subgraph.Tensors(t).ShapeLength() > 4:
        over_rank4 += 1
  return histogram, over_rank4


def check_ops(path, tag):
  """Prints the op histogram and fails if the graph cannot run on the GPU.

  Args:
    path: Path to the .tflite file.
    tag: Short graph name used in the log line.

  Raises:
    AssertionError: if a GPU-banned op or a >4-D tensor survived.
  """
  histogram, over_rank4 = op_histogram(path)
  banned = {
      k: v
      for k, v in histogram.items()
      if k in recipe.GPU_BANNED_OPS or k.startswith("Flex")
  }
  total = sum(histogram.values())
  top = dict(sorted(histogram.items(), key=lambda kv: -kv[1])[:8])
  print(f"[ops {tag}] total={total} kinds={len(histogram)} top={top}")
  print(f"[ops {tag}] banned={banned or 'NONE'} rank>4={over_rank4}")
  assert not banned, f"{tag}: GPU-banned ops {banned}"
  assert not over_rank4, f"{tag}: {over_rank4} tensors above rank 4"


def parity(tag, produced, reference):
  """Prints correlation and max absolute difference against the reference.

  Args:
    tag: Short name used in the log line.
    produced: Tensor or array under test.
    reference: Tensor or array taken as truth.

  Returns:
    The Pearson correlation as a float.
  """
  a = np.asarray(produced, dtype=np.float64).reshape(-1)
  b = np.asarray(reference, dtype=np.float64).reshape(-1)
  corr = float(np.corrcoef(a, b)[0, 1])
  max_diff = float(np.abs(a - b).max())
  print(f"[parity {tag}] corr={corr:.7f} max|diff|={max_diff:.3g}")
  return corr


def run_cpu(path, flat_input, n_out):
  """Runs one inference through the CompiledModel CPU path.

  Args:
    path: Path to the .tflite file.
    flat_input: Flat float32 input array.
    n_out: Number of output floats to read back.

  Returns:
    The flat float32 output array.
  """
  from ai_edge_litert.compiled_model import CompiledModel
  from ai_edge_litert.hardware_accelerator import HardwareAccelerator

  model = CompiledModel.from_file(path, HardwareAccelerator.CPU)
  inputs = model.create_input_buffers(0)
  outputs = model.create_output_buffers(0)
  inputs[0].write(np.ascontiguousarray(flat_input, np.float32).ravel())
  started = time.time()
  model.run_by_index(0, inputs, outputs)
  result = np.array(outputs[0].read(n_out, np.float32))
  elapsed_ms = (time.time() - started) * 1e3
  print(f"[cpu {os.path.basename(path)}] {elapsed_ms:.0f} ms")
  return result


def quantize_fp16(src, dst):
  """Casts the matmul-class weights of a graph to fp16.

  Args:
    src: Path to the fp32 .tflite file.
    dst: Path to write the fp16 .tflite file to.

  Returns:
    The size of the written file in megabytes.
  """
  from ai_edge_quantizer import quantizer

  if os.path.exists(dst):
    os.remove(dst)
  quant = quantizer.Quantizer(src)
  quant.load_quantization_recipe(recipe.FP16_RECIPE)
  quant.quantize().export_model(dst)
  return os.path.getsize(dst) / 1e6


def convert_graph(module, example_input, name, out_dir):
  """Converts one module with litert-torch and fp16-quantizes it.

  Args:
    module: The torch module to convert.
    example_input: Example input tensor defining the fixed shape.
    name: Output file stem.
    out_dir: Directory to write into.

  Returns:
    A (fp32 path, fp16 path) tuple.
  """
  import litert_torch

  fp32_path = os.path.join(out_dir, f"{name}_fp32.tflite")
  started = time.time()
  exported = litert_torch.convert(module.eval(), (example_input,))
  exported.export(fp32_path)
  size = os.path.getsize(fp32_path) / 1e6
  print(f"[convert {name}] {time.time() - started:.0f}s fp32={size:.1f} MB")
  check_ops(fp32_path, name)
  fp16_path = os.path.join(out_dir, f"{name}.tflite")
  print(f"[fp16 {name}] {quantize_fp16(fp32_path, fp16_path):.1f} MB")
  return fp32_path, fp16_path


def export_host_assets(det, out_dir):
  """Writes the fp16 token-embedding table and the CLIP BPE vocabulary.

  EMBEDDING_LOOKUP has no GPU lowering, so the token embedding is a host table
  and the text graph takes embeddings instead of ids.

  Args:
    det: The detector.
    out_dir: Directory to write into.
  """
  import pkg_resources

  encoder = det.backbone.language_backbone.encoder
  table = encoder.token_embedding.weight.detach()
  path = os.path.join(out_dir, "sam3_token_embed.bin")
  table.to(torch.float16).contiguous().numpy().tofile(path)
  size = os.path.getsize(path) / 1e6
  print(f"[table] {tuple(table.shape)} fp16 {size:.1f} MB -> {path}")
  bpe = pkg_resources.resource_filename(
      "sam3", "assets/bpe_simple_vocab_16e6.txt.gz"
  )
  tokenizer_dir = os.path.join(out_dir, "sam3_tokenizer")
  os.makedirs(tokenizer_dir, exist_ok=True)
  shutil.copy(bpe, os.path.join(tokenizer_dir, os.path.basename(bpe)))
  print(f"[tokenizer] {os.path.basename(bpe)} -> {tokenizer_dir}")


def main():
  """Builds, verifies and writes the three graphs plus the host assets."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--ckpt", required=True, help="sam3.1_multiplex.pt")
  parser.add_argument("--out", default=".", help="output directory")
  parser.add_argument("--image", required=True, help="parity image")
  parser.add_argument("--prompt", default="wheel", help="parity prompt")
  parser.add_argument(
      "--chunks",
      type=int,
      default=9,
      help="global-attention query chunks (exact; any divisor of 5184)",
  )
  parser.add_argument(
      "--only",
      choices=["vision", "text", "head"],
      default=None,
      help="convert a single graph",
  )
  parser.add_argument(
      "--no-convert",
      action="store_true",
      help="run the torch parity gate only",
  )
  args = parser.parse_args()
  os.makedirs(args.out, exist_ok=True)
  wanted = {args.only} if args.only else {"vision", "text", "head"}
  torch.manual_seed(0)

  det = recipe.build_detector(args.ckpt)
  image, _ = recipe.preprocess_image(args.image)
  tokenizer = det.backbone.language_backbone.tokenizer
  tokens = tokenizer([args.prompt], context_length=recipe.CONTEXT)
  print(f"[input] '{args.prompt}' -> {tokens[0].tolist()[:8]}...")

  # Stage 1: stock references, before any patch touches the model.
  with torch.inference_mode():
    vision_ref = recipe.VisionFlat(det)(image)
    text_full = recipe.TextFlatStock(det)(tokens)
    head_ref = recipe.HeadFlatStock(det)(torch.cat([vision_ref, text_full], 1))
  text_ref = text_full[:, :recipe.CONTEXT * 256]
  text_pad = text_full[:, recipe.CONTEXT * 256:]

  # Stage 2: re-author in place.
  recipe.patch_vit_4d(
      det.backbone.vision_backbone.trunk, global_chunks=args.chunks
  )
  recipe.patch_neck(det.backbone.vision_backbone)
  n_text = recipe.apply_text_patches(det)
  n_head = recipe.apply_head_patches(det)
  print(f"[patch] text blocks={n_text} attention modules={n_head}")

  vision_module = recipe.VisionFlat(det)
  text_module = recipe.TextFlat4d(det)
  head_module = recipe.HeadFlat4d(det)
  embeddings = det.backbone.language_backbone.encoder.token_embedding(
      tokens
  ).detach()
  head_input = torch.cat([vision_ref, text_ref, text_pad], 1)

  # Stage 3: the re-authored torch modules must reproduce the reference.
  with torch.inference_mode():
    if "vision" in wanted:
      parity("vision torch", vision_module(image), vision_ref)
    if "text" in wanted:
      parity("text torch", text_module(embeddings), text_ref)
    if "head" in wanted:
      parity("head torch", head_module(head_input), head_ref)
  if args.no_convert:
    return

  # Stages 4 and 5.
  export_host_assets(det, args.out)
  jobs = []
  if "text" in wanted:
    jobs.append(("sam3_text", text_module, embeddings, text_ref))
  if "head" in wanted:
    jobs.append(("sam3_head", head_module, head_input, head_ref))
  if "vision" in wanted:
    jobs.append(("sam3_vision", vision_module, image, vision_ref))
  for name, module, example, reference in jobs:
    _, fp16_path = convert_graph(module, example, name, args.out)
    produced = run_cpu(fp16_path, example.numpy(), reference.numel())
    parity(f"{name} fp16 tflite", produced, reference.numpy())


if __name__ == "__main__":
  main()

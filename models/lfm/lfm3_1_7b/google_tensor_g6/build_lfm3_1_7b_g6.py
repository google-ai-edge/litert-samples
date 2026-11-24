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

"""LiquidAI/LFM3-1.7B -> one .litertlm bundle for the Google Tensor G6 NPU.

Five stages, each re-runnable on its own. Each prints the size and sha256 of
the files it writes under --work:

  download  the checkpoint at REVISION into WORK/LFM3-1.7B, and the option file
            Google published with its Gemma 3 1B bundle for Tensor G6, at
            OPTIONS_REVISION, into WORK/ (a gated repository: accept the Gemma
            terms on Hugging Face and `hf auth login` first)
  export    the stock HF exporter with lfm3_g6_extension.py: int8 weights,
            split cache, prefill 128, cache 2,048, ring caches of 256 in the
            sliding layers -> WORK/export/
  embedder  the embedder's EMBEDDING_LOOKUP on the int8 table itself, without
            the DEQUANTIZE of the whole table in front of it
            -> WORK/export/embedder_direct.tflite
  compile   ai_edge_litert's aot_compile for Tensor G6 -> WORK/compiled/; run
            this stage with the Python of the environment where the Tensor SDK
            is installed (it imports nothing else)
  bundle    the six sections with litert-lm-builder -> WORK/BUNDLE, read back
            and compared with their sources

Usage:
  python build_lfm3_1_7b_g6.py download export embedder --work out_g6
  <SDK environment>/bin/python build_lfm3_1_7b_g6.py compile --work out_g6
  python build_lfm3_1_7b_g6.py bundle --work out_g6
"""

import argparse
import hashlib
import io
import os
import subprocess
import sys
import zlib
from pathlib import Path

MODEL_ID = "LiquidAI/LFM3-1.7B"
REVISION = "1da06bec6a819da10b54c67e5c90972b46d81871"
SOURCE_FILES = [
    "config.json", "generation_config.json", "configuration_lfm3.py",
    "modeling_lfm3.py", "lfm3_conv.py", "model.safetensors", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
]
OPTIONS_REPO = "litert-community/Gemma3-1B-IT"
OPTIONS_REVISION = "a9bd85643f05cdb1e5cd5af7ba4e74b0e8210ca0"
OPTIONS_FILE = "Gemma3-1B-IT_Google_Tensor_G6.pb"
OPTIONS_SHA256 = "f5c3266111fd93e9c369cf150dc5422885ebb4c57a5a407295d4b39c44525335"
HERE = Path(__file__).resolve().parent
EXTENSION = HERE / "lfm3_g6_extension.py"
BUNDLE = "LFM3-1.7B_Google_Tensor_G6.litertlm"
EXPORT_FLAGS = [
    "--trust_remote_code=True", "--split_cache=True", "--prefill_lengths=128",
    "--cache_length=2048", "--sliding_window_ring_buffer_size=256",
    "--quantization_recipe=weight_only_wi8_afp32",
    "--keep_temporary_files=True", "--bundle_litert_lm=True",
]
EXPORT_FILES = [
    "model_quantized.tflite", "embedder_quantized.tflite", "auxiliary.tflite",
    "executor_metadata.pb", "llm_metadata.pb", "tokenizer.json",
]


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for chunk in iter(lambda: f.read(1 << 24), b""):
      h.update(chunk)
  return h.hexdigest()


def show(*paths):
  for p in paths:
    print(f"{os.path.getsize(p):>15,} B  {sha256(p)}  {p}", flush=True)


def check_toolchain():
  """Stops early on a litert-torch that cannot run this export."""
  import torch  # pylint: disable=g-import-not-at-top
  from litert_torch.backend.torch_tfl import _ops  # pylint: disable=g-import-not-at-top,unused-import
  from litert_torch.generative.export_hf.core.split_cache import exportable_module  # pylint: disable=g-import-not-at-top

  missing = []
  if not hasattr(torch.ops.tfl, "sum"):
    missing.append("torch.ops.tfl.sum")
  # shortcut: keyed on a helper litert-torch #1505 adds; drop once a release has it.
  if not hasattr(exportable_module, "_has_conv_layer"):
    missing.append("the split-cache export of conv layers (#1505)")
  if missing:
    raise SystemExit(f"this litert-torch lacks {' and '.join(missing)};"
                     " install the litert-torch of the README")


def download(args):
  from huggingface_hub import hf_hub_download, snapshot_download  # pylint: disable=g-import-not-at-top

  if not args.model:
    snapshot_download(MODEL_ID, revision=REVISION, local_dir=args.model_dir,
                      allow_patterns=SOURCE_FILES)
  if not args.options:
    hf_hub_download(OPTIONS_REPO, OPTIONS_FILE, revision=OPTIONS_REVISION,
                    local_dir=args.work)
  show(*(args.model_dir / f for f in SOURCE_FILES), args.options_file)


def export(args):
  check_toolchain()
  if args.export_dir.exists():
    raise SystemExit(f"{args.export_dir} exists; remove it to export again")
  # The extension goes in by module name, as tensor names carry it (a file
  # path would put a hash of the path in them). PYTHONHASHSEED=0 as for the
  # export checked on a Pixel 11; offline, as the checkpoint is local.
  env = dict(os.environ, PYTHONHASHSEED="0", HF_HUB_OFFLINE="1",
             PYTHONPATH=os.pathsep.join(
                 filter(None, [str(HERE), os.environ.get("PYTHONPATH")])))
  subprocess.run(
      [sys.executable, "-m", "litert_torch.generative.export_hf",
       f"--model={args.model_dir}", f"--output_dir={args.export_dir}",
       f"--extension_modules={EXTENSION.stem}", *EXPORT_FLAGS],
      env=env, check=True)
  show(*(args.export_dir / f for f in EXPORT_FILES))


def embedder(args):
  """Points EMBEDDING_LOOKUP at the int8 table, which it dequantizes per row."""
  from ai_edge_litert import schema_py_generated as schema  # pylint: disable=g-import-not-at-top
  from ai_edge_litert.tools import flatbuffer_utils  # pylint: disable=g-import-not-at-top

  ops = schema.BuiltinOperator
  model = flatbuffer_utils.read_model(
      str(args.export_dir / "embedder_quantized.tflite"))
  code = lambda op: model.operatorCodes[op.opcodeIndex].builtinCode
  dropped = 0
  for sg in model.subgraphs:
    table = {op.outputs[0]: op.inputs[0] for op in sg.operators
             if code(op) == ops.DEQUANTIZE}
    for op in sg.operators:
      if code(op) == ops.EMBEDDING_LOOKUP and op.inputs[1] in table:
        op.inputs = [op.inputs[0], table[op.inputs[1]]]
    read = {t for op in sg.operators if code(op) != ops.DEQUANTIZE
            for t in op.inputs}
    kept = [op for op in sg.operators
            if code(op) != ops.DEQUANTIZE or op.outputs[0] in read]
    dropped += len(sg.operators) - len(kept)
    sg.operators = kept
  if not dropped:
    raise SystemExit("no DEQUANTIZE in front of EMBEDDING_LOOKUP; the"
                     " exporter's embedder changed")
  out = args.export_dir / "embedder_direct.tflite"
  flatbuffer_utils.write_model(model, str(out))
  print(f"embedder: {dropped} DEQUANTIZE dropped")
  show(out)


def compile_model(args):
  from ai_edge_litert.aot import aot_compile as aot_lib  # pylint: disable=g-import-not-at-top
  from ai_edge_litert.aot.vendors.google_tensor import target as gt_target  # pylint: disable=g-import-not-at-top

  if sha256(args.options_file) != OPTIONS_SHA256:
    raise SystemExit(f"{args.options_file} is not {OPTIONS_FILE} of"
                     f" {OPTIONS_REPO} at {OPTIONS_REVISION}")
  out = args.work / "compiled"
  if out.exists():
    raise SystemExit(f"{out} exists; remove it to compile again")
  result = aot_lib.aot_compile(
      str(args.export_dir / "model_quantized.tflite"),
      output_dir=out,
      target=[gt_target.Target(gt_target.SocModel.TENSOR_G6)],
      keep_going=False,
      # As for the bundle checked on a Pixel 11: 16-bit floats, which the
      # model changes of lfm3_g6_extension.py keep in range, and the compiler
      # options of Google's Gemma 3 1B G6 bundle.
      google_tensor_truncation_type="half",
      google_tensor_extra_options_path=str(args.options_file),
  )
  print(result.compilation_report(), flush=True)
  (compiled,) = result.models
  partial = [s for s in compiled.partition_stats.subgraph_stats
             if s.num_ops_offloaded != s.num_total_ops]
  if partial:
    raise SystemExit(f"not every op runs on the TPU: {partial}")
  show(compiled.path)


def bundle(args):
  from litert_lm_builder import litertlm_builder  # pylint: disable=g-import-not-at-top
  from litert_lm_builder import litertlm_peek  # pylint: disable=g-import-not-at-top
  from litert_lm_builder.runtime.proto import llm_metadata_pb2  # pylint: disable=g-import-not-at-top

  ex = args.export_dir
  (compiled,) = sorted((args.work / "compiled").glob("*.tflite"))
  kind = llm_metadata_pb2.LlmMetadata
  sources = [ex / "llm_metadata.pb", ex / "executor_metadata.pb",
             ex / "tokenizer.json", compiled, ex / "embedder_direct.tflite",
             ex / "auxiliary.tflite"]
  # The sections and system metadata of the exporter's own bundle, with the
  # compiled model and the direct embedder in place of the uncompiled ones.
  builder = litertlm_builder.LitertLmFileBuilder()
  builder.add_system_metadata(litertlm_builder.Metadata(
      key="Authors", value="ODML", dtype=litertlm_builder.DType.STRING))
  builder.add_llm_metadata(str(sources[0]))
  builder.add_executor_metadata(str(sources[1]))
  builder.add_hf_tokenizer(str(sources[2]))
  builder.add_tflite_model(str(sources[3]), kind.TF_LITE_PREFILL_DECODE)
  builder.add_tflite_model(str(sources[4]), kind.TF_LITE_EMBEDDER)
  builder.add_tflite_model(str(sources[5]), kind.TF_LITE_AUX)
  out = args.work / BUNDLE
  with open(out, "wb") as f:
    builder.build(f, weights_in_model_graph=True)

  header = litertlm_peek.read_litertlm_header(str(out), io.StringIO())
  sections = header.SectionMetadata()
  if sections.ObjectsLength() != len(sources):
    raise SystemExit(f"{out}: {sections.ObjectsLength()} sections, expected"
                     f" {len(sources)}")
  with open(out, "rb") as f:
    for i, source in enumerate(sources):
      section = sections.Objects(i)
      f.seek(section.BeginOffset())
      data = f.read(section.EndOffset() - section.BeginOffset())
      want = source.read_bytes()
      # The tokenizer section is its size (8 bytes, little endian) and zlib.
      got = zlib.decompress(data[8:]) if source.suffix == ".json" else data
      if got != want:
        raise SystemExit(f"{out}: section {i} differs from {source}")
      print(f"[{i}] {len(data):>15,} B  {hashlib.sha256(data).hexdigest()}"
            f"  = {source.name}")
  show(out)


STAGES = {"download": download, "export": export, "embedder": embedder,
          "compile": compile_model, "bundle": bundle}


def main():
  ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
  ap.add_argument("stages", nargs="+", choices=list(STAGES),
                  help="stages to run, in the order given")
  ap.add_argument("--work", type=Path, default=Path("out_g6"),
                  help="working directory")
  ap.add_argument("--model", type=Path,
                  help="a local copy of the checkpoint at REVISION in a folder"
                       " named LFM3-1.7B, which every tensor name carries"
                       " (default: WORK/LFM3-1.7B, downloaded)")
  ap.add_argument("--options", type=Path,
                  help=f"a local copy of {OPTIONS_FILE} (default: WORK/,"
                       " downloaded)")
  args = ap.parse_args()
  args.work = args.work.resolve()
  args.model_dir = (args.model or args.work / "LFM3-1.7B").resolve()
  args.options_file = (args.options or args.work / OPTIONS_FILE).resolve()
  args.export_dir = args.work / "export"
  args.work.mkdir(parents=True, exist_ok=True)
  for stage in args.stages:
    print(f"== {stage}", flush=True)
    STAGES[stage](args)


if __name__ == "__main__":
  main()

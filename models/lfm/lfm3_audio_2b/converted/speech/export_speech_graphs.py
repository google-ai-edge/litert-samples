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

"""Exports the speech-output graphs of LiquidAI/LFM3-Audio-2B (the part that runs outside LiteRT-LM).

  python export_speech_graphs.py --hf SRC --out-dir OUT [--part P ...] [--ref-dir DIR]

SRC is a local copy of LiquidAI/LFM3-Audio-2B (the files read: model.safetensors and
audio_tokenizer/model.safetensors). Parts, all by default, each in its own process:

  frame       lfm3_audio_frame_embed_f32.tflite         frame_embed: codes [1,12] int32 -> embeds [1,1,2048], the
              sum of the 12 codebook embeddings that is the decoder's next input during speech
  depth       lfm3_audio_depth_proj_r4b.tflite          depth_proj: hidden [1,2048] -> depth_in [1,12,1024], once per
              80 ms speech frame
              lfm3_audio_depth_pass_r4b.tflite          depth_pass: depth_in, codes [1,12] int32, masked [1,12]
              -> logits [1,12,2017], three passes per frame reveal 4 + 4 + 4 codebooks
  depth_int8  lfm3_audio_depth_pass_r4b_int8.tflite     the float32 pass through ai-edge-quantizer's dynamic_wi8_afp32
              recipe: int8 per-channel weights, float32 activations
  codec12     lfm3_audio_codec_decoder_f12_r4b2.tflite  decode: codes [1,12,12] + streaming state -> 72 windowed
              frames of 1,280 samples + the next state (12 codec frames = 0.96 s of 24 kHz audio per call)
  window      codec_window_f32.npy                      the codec's synthesis window (decoder.istft.window), which
              the host overlap-add divides by

With litert-torch 0.10.0, litert-converter 0.5.0, ai-edge-quantizer 0.10.0 and torch 2.14.0 (the toolchain of the
published files) the five .tflite files come out byte for byte as published. --ref-dir compares each file's sha256
with the file of the same name there. A record of the run goes to OUT/speech_graphs.json.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
PARTS = ("frame", "depth", "depth_int8", "codec12", "window")
FILES = {
    "frame": ("lfm3_audio_frame_embed_f32.tflite",),
    "depth": ("lfm3_audio_depth_proj_r4b.tflite", "lfm3_audio_depth_pass_r4b.tflite"),
    "depth_int8": ("lfm3_audio_depth_pass_r4b_int8.tflite",),
    "codec12": ("lfm3_audio_codec_decoder_f12_r4b2.tflite",),
    "window": ("codec_window_f32.npy",),
}
# The toolchain the published files were exported with (their bytes depend on it).
PUBLISHED_STACK = {"litert-torch": "0.10.0", "litert-converter": "0.5.0", "ai-edge-quantizer": "0.10.0",
                   "torch": "2.14.0"}


def load_by_path(name, path, register=True):
  """Loads a module file under `name`: the module name reaches the graphs' tensor names."""
  if register and name in sys.modules:
    return sys.modules[name]
  spec = importlib.util.spec_from_file_location(name, path)
  mod = importlib.util.module_from_spec(spec)
  if register:
    sys.modules[name] = mod
  spec.loader.exec_module(mod)
  return mod


def versions():
  out = {}
  for p in PUBLISHED_STACK:
    try:
      out[p] = importlib.metadata.version(p)
    except importlib.metadata.PackageNotFoundError:
      out[p] = None
  return out


def sha256(path):
  h = hashlib.sha256()
  with open(path, "rb") as f:
    for c in iter(lambda: f.read(1 << 24), b""):
      h.update(c)
  return h.hexdigest()


def export_part(part, hf, out_dir):
  main_st, codec_st = hf / "model.safetensors", hf / "audio_tokenizer" / "model.safetensors"
  if part == "frame":
    g = load_by_path("r1b_graphs", HERE / "r1b_graphs.py", register=False)
    g.export_frame_embed(g.load_frame_state_dict(main_st, "audio2b"), out_dir / FILES["frame"][0], "audio2b")
  elif part == "depth":
    p = load_by_path("r4b_parts", HERE / "r4b_parts.py")
    p.export_depth_split(p.G.load_depth_state_dict(main_st), out_dir / FILES["depth"][0], out_dir / FILES["depth"][1])
  elif part == "depth_int8":
    try:
      from ai_edge_quantizer import quantizer, recipe  # pylint: disable=g-import-not-at-top
    except ImportError:  # the package's newer name (litert-quantizer)
      from litert_quantizer import quantizer, recipe  # pylint: disable=g-import-not-at-top
    src = out_dir / FILES["depth"][1]
    if not src.exists():
      raise SystemExit(f"depth_int8 reads {src}: run the depth part first")
    res = quantizer.Quantizer(str(src), recipe.dynamic_wi8_afp32()).quantize()
    res.export_model(str(out_dir / FILES["depth_int8"][0]), overwrite=True)
  elif part == "codec12":
    p = load_by_path("r4b_parts", HERE / "r4b_parts.py")
    p.export_codec(p.G.load_codec_state_dict(codec_st), 12, out_dir / FILES["codec12"][0])
  elif part == "window":
    import numpy as np  # pylint: disable=g-import-not-at-top
    g = load_by_path("r1b_graphs", HERE / "r1b_graphs.py", register=False)
    win = g.read_tensors(codec_st, names={"decoder.istft.window"})["decoder.istft.window"]
    np.save(out_dir / FILES["window"][0], win.numpy().astype(np.float32))
  else:
    raise ValueError(part)


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  ap.add_argument("--hf", required=True, help="local copy of LiquidAI/LFM3-Audio-2B")
  ap.add_argument("--out-dir", required=True)
  ap.add_argument("--part", action="append", choices=PARTS, help="default: all, in the order listed")
  ap.add_argument("--ref-dir", help="compare each file's sha256 with the same name in this folder")
  a = ap.parse_args()
  hf = pathlib.Path(a.hf).resolve()
  out_dir = pathlib.Path(a.out_dir).resolve()
  out_dir.mkdir(parents=True, exist_ok=True)
  parts = a.part or list(PARTS)
  if len(parts) == 1 and not a.ref_dir:
    t0 = time.time()
    export_part(parts[0], hf, out_dir)
    for f in FILES[parts[0]]:
      p = out_dir / f
      print(f"{f}: {p.stat().st_size} B sha256 {sha256(p)} ({time.time() - t0:.1f} s)", flush=True)
    return 0
  # Several parts: one process per part (as the published files were made), then the record.
  stack = versions()
  if stack != PUBLISHED_STACK:
    print(f"note: toolchain {stack} differs from the published files' {PUBLISHED_STACK}; the graphs may come out "
          "with other bytes", flush=True)
  rows, rc = [], 0
  for part in parts:
    t0 = time.time()
    r = subprocess.run([sys.executable, str(pathlib.Path(__file__).resolve()), "--hf", str(hf), "--out-dir",
                        str(out_dir), "--part", part], check=False)
    if r.returncode:
      print(f"part {part}: exit {r.returncode}", flush=True)
      rc = r.returncode
      break
    for f in FILES[part]:
      p = out_dir / f
      row = {"part": part, "file": f, "bytes": p.stat().st_size, "sha256": sha256(p),
             "seconds": round(time.time() - t0, 1)}
      if a.ref_dir:
        ref = pathlib.Path(a.ref_dir) / f
        row["ref_sha256"] = sha256(ref) if ref.exists() else None
        row["verdict"] = ("IDENTICAL" if row["sha256"] == row["ref_sha256"] else "DIFFERENT") if ref.exists() \
            else "NO_REF"
      rows.append(row)
      print(f"{f}: {row['bytes']} B sha256 {row['sha256']} {row.get('verdict', '')}", flush=True)
  rec = {"source": str(hf), "toolchain": stack, "published_toolchain": PUBLISHED_STACK, "python": sys.version.split()[0],
         "files": rows, "ok": rc == 0}
  (out_dir / "speech_graphs.json").write_text(json.dumps(rec, indent=1) + "\n")
  return rc


if __name__ == "__main__":
  sys.exit(main())

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

"""Real-image end-to-end check of the SAM 3 image graphs.

Runs a photo and a text prompt through the full converted chain via the LiteRT
CompiledModel API -- host BPE, host embedding lookup, vision graph, text graph,
head graph, host thresholding -- and prints the detections. With `--ckpt` it
also runs the official PyTorch model on the same input and reports which
queries survive the threshold in both, plus box IoU, mask IoU and score deltas
for the kept set. Correlation alone does not prove the decoded output is right,
so the kept set is the ship criterion.

The prompt matters: run several phrases. The CLIP text tower is the part that
breaks first in fp16 (see the README), and a wrong prompt embedding produces a
confidently empty result rather than an error.

Run (after build_sam3_image.py has emitted the artifacts):
  python verify_sam3_image.py --image photo.jpg --prompt wheel
  python verify_sam3_image.py --image photo.jpg --prompt wheel \
      --accelerator gpu --ckpt sam3.1_multiplex.pt
"""

import argparse
import os
import time

import numpy as np

import sam3_recipe as recipe

N_VISION = 256 * sum(h * w for h, w in recipe.FPN_SIZES)
N_TEXT = recipe.CONTEXT * 256
N_HEAD = (
    recipe.NUM_QUERIES * 5
    + 1
    + recipe.NUM_QUERIES * recipe.MASK_SIZE * recipe.MASK_SIZE
)


class Graph:
  """One compiled .tflite with a flat float input and output.

  Args:
    path: Path to the .tflite file.
    accelerator: "cpu", "gpu", or "gpu_f32" for GPU with enforce_f32.
    n_out: Number of output floats to read back.
  """

  def __init__(self, path, accelerator, n_out):
    from ai_edge_litert.compiled_model import CompiledModel
    from ai_edge_litert.hardware_accelerator import HardwareAccelerator

    self.name = os.path.basename(path)
    self.n_out = n_out
    self.last_ms = 0.0
    started = time.time()
    if accelerator == "cpu":
      self.model = CompiledModel.from_file(path, HardwareAccelerator.CPU)
    elif accelerator == "gpu":
      self.model = CompiledModel.from_file(path, HardwareAccelerator.GPU)
    elif accelerator == "gpu_f32":
      from ai_edge_litert.options import Options

      options = Options.create()
      options.hardware_accelerators = HardwareAccelerator.GPU
      options.gpu_options.enforce_f32 = True
      self.model = CompiledModel.from_file(path, options=options)
    else:
      raise ValueError(f"unknown accelerator {accelerator}")
    self.compile_s = time.time() - started
    self.inputs = self.model.create_input_buffers(0)
    self.outputs = self.model.create_output_buffers(0)
    try:
      self.fully_accelerated = self.model.is_fully_accelerated()
    except (AttributeError, RuntimeError):
      self.fully_accelerated = None

  def __call__(self, flat_input):
    """Runs one inference and returns the flat float32 output."""
    self.inputs[0].write(np.ascontiguousarray(flat_input, np.float32).ravel())
    started = time.time()
    self.model.run_by_index(0, self.inputs, self.outputs)
    result = np.array(self.outputs[0].read(self.n_out, np.float32))
    self.last_ms = (time.time() - started) * 1e3
    return result


def load_tokenizer(artifacts):
  """Builds the CLIP BPE tokenizer from the exported vocabulary.

  Args:
    artifacts: Directory holding sam3_tokenizer/.

  Returns:
    A callable taking a list of strings and a context length.
  """
  from sam3.model.tokenizer_ve import SimpleTokenizer

  vocab = os.path.join(
      artifacts, "sam3_tokenizer", "bpe_simple_vocab_16e6.txt.gz"
  )
  return SimpleTokenizer(bpe_path=vocab)


def box_iou_cxcywh(a, b):
  """Intersection over union of two cxcywh boxes normalized to [0, 1].

  Args:
    a: Array of four values, cx, cy, w, h.
    b: Array of four values, cx, cy, w, h.

  Returns:
    The IoU as a float.
  """

  def corners(box):
    cx, cy, w, h = box
    return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

  ax0, ay0, ax1, ay1 = corners(a)
  bx0, by0, bx1, by1 = corners(b)
  ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
  iy = max(0.0, min(ay1, by1) - max(ay0, by0))
  inter = ix * iy
  union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
  return float(inter / union) if union > 0 else 0.0


def mask_iou(a, b):
  """Intersection over union of two boolean masks.

  Args:
    a: Boolean array.
    b: Boolean array of the same shape.

  Returns:
    The IoU as a float.
  """
  union = np.logical_or(a, b).sum()
  if union == 0:
    return 1.0
  return float(np.logical_and(a, b).sum() / union)


def reference_detections(ckpt, image, tokens):
  """Runs the unmodified PyTorch model on the same input.

  Args:
    ckpt: Path to the checkpoint.
    image: (1, 3, 1008, 1008) input tensor.
    tokens: (1, 32) token id tensor.

  Returns:
    The flat head output as a numpy array.
  """
  import torch

  det = recipe.build_detector(ckpt, verbose=False)
  with torch.inference_mode():
    vision = recipe.VisionFlat(det)(image)
    text = recipe.TextFlatStock(det)(tokens)
    head = recipe.HeadFlatStock(det)(torch.cat([vision, text], 1))
  return head.numpy()


def main():
  """Runs the converted chain and, optionally, compares against PyTorch."""
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--artifacts", default=".", help="directory of .tflite")
  parser.add_argument("--image", required=True, help="input photo")
  parser.add_argument("--prompt", default="wheel", help="text prompt")
  parser.add_argument("--threshold", type=float, default=0.5)
  parser.add_argument(
      "--accelerator",
      default="cpu",
      choices=["cpu", "gpu", "gpu_f32"],
      help="how to run every graph; the text graph is forced to CPU on gpu",
  )
  parser.add_argument(
      "--ckpt",
      default=None,
      help="checkpoint for the PyTorch comparison (optional)",
  )
  args = parser.parse_args()

  tokenizer = load_tokenizer(args.artifacts)
  table = np.fromfile(
      os.path.join(args.artifacts, "sam3_token_embed.bin"), dtype=np.float16
  ).reshape(-1, 1024)
  image, original_size = recipe.preprocess_image(args.image)
  tokens = tokenizer([args.prompt], context_length=recipe.CONTEXT)
  ids = tokens[0].numpy()
  embeddings = table[ids].astype(np.float32)[None]
  pad = (ids == 0).astype(np.float32)[None]

  # The text tower runs on the CPU unless the GPU is asked for f32: its
  # residual stream overflows fp16 and silently corrupts some prompts.
  text_accelerator = "cpu" if args.accelerator == "gpu" else args.accelerator
  vision = Graph(
      os.path.join(args.artifacts, "sam3_vision.tflite"),
      args.accelerator,
      N_VISION,
  )
  text = Graph(
      os.path.join(args.artifacts, "sam3_text.tflite"),
      text_accelerator,
      N_TEXT,
  )
  head = Graph(
      os.path.join(args.artifacts, "sam3_head.tflite"),
      args.accelerator,
      N_HEAD,
  )
  for graph in (vision, text, head):
    print(
        f"[compile] {graph.name} {graph.compile_s:.1f}s "
        f"fully_accelerated={graph.fully_accelerated}"
    )

  started = time.time()
  vision_out = vision(image.numpy())
  text_out = text(embeddings)
  head_out = head(np.concatenate([vision_out, text_out, pad.ravel()]))
  total_ms = (time.time() - started) * 1e3
  scores, boxes, masks, kept = recipe.decode_detections(
      head_out, args.threshold
  )
  print(
      f"[run] '{args.prompt}' {total_ms:.0f} ms "
      f"(vision {vision.last_ms:.0f} + text {text.last_ms:.0f} + "
      f"head {head.last_ms:.0f})"
  )
  print(f"[detections] image {original_size[0]}x{original_size[1]}")
  for query in kept:
    cx, cy, w, h = boxes[query]
    area = int((masks[query] > 0).sum())
    print(
        f"  query {query:3d} score={scores[query]:.3f} "
        f"box=({cx:.3f},{cy:.3f},{w:.3f},{h:.3f}) mask_px={area}"
    )
  if not len(kept):  # pylint: disable=g-explicit-length-test
    print("  none above threshold")

  if not args.ckpt:
    return
  ref_out = reference_detections(args.ckpt, image, tokens)
  ref_scores, ref_boxes, ref_masks, ref_kept = recipe.decode_detections(
      ref_out, args.threshold
  )
  same = set(kept.tolist()) == set(ref_kept.tolist())
  print(
      f"[reference] kept={ref_kept.tolist()} "
      f"converted={kept.tolist()} same_set={same}"
  )
  for query in ref_kept:
    b_iou = box_iou_cxcywh(boxes[query], ref_boxes[query])
    m_iou = mask_iou(masks[query] > 0, ref_masks[query] > 0)
    delta = abs(float(scores[query] - ref_scores[query]))
    print(
        f"  query {query:3d} box IoU={b_iou:.3f} mask IoU={m_iou:.3f} "
        f"|dscore|={delta:.4f}"
    )


if __name__ == "__main__":
  main()

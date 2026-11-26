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

"""Converts fastino/gliner2-privacy-filter-PII-multi to LiteRT (.tflite).

Writes into --out-dir:
  gliner2_pii.tflite       float32 LiteRT model
  gliner2_pii_int8.tflite  dynamic-range int8 quantized LiteRT model (unless
                           --no-quantize)
  tokenizer.json           HuggingFace tokenizer including GLiNER2's special
                           tokens
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import pathlib
import time

from ai_edge_quantizer import quantizer
from ai_edge_quantizer import recipe
import gliner2
import litert_torch
import tokenizers
import torch
from torch import nn

MODEL_ID = "fastino/gliner2-privacy-filter-PII-multi"
DEFAULT_SEQ_LEN = 512
DEFAULT_MAX_WORDS = 256
DEFAULT_MAX_LABELS = 42


def gather(seq: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
  """Gathers embeddings at `positions` from `seq` along the sequence dimension.

  Args:
    seq: Encoded hidden states of shape `[1, S, D]`.
    positions: Token index positions of shape `[1, N]`.

  Returns:
    Gathered embeddings of shape `[1, N, D]`.
  """
  index = positions.long().unsqueeze(-1).expand(-1, -1, seq.shape[-1])
  return torch.gather(seq, 1, index)


class EntityExtractionGraph(nn.Module):
  """Static-shape entity-extraction forward pass of a GLiNER2 span model.

  Attributes:
    encoder: Pretrained transformer backbone mapping token IDs and attention
      masks to contextual hidden states.
    span_marker: Span representation layer computing width-bounded span features
      from word embeddings.
    count_pred: Linear projection head predicting entity count logits from the
      prompt embedding.
    count_embed: Count-aware label projection module transforming label
      embeddings into structural query vectors.
    max_width: Maximum entity span width in words (`end_word - start_word + 1`).
  """

  encoder: nn.Module
  span_marker: nn.Module
  count_pred: nn.Module
  count_embed: nn.Module
  max_width: int

  def __init__(self, model: nn.Module):
    """Initializes the instance.

    Args:
      model: Loaded GLiNER2 model instance containing the encoder, span
        representation layer, count prediction head, and count embedding module.
    """
    super().__init__()
    self.encoder = model.encoder
    self.span_marker = model.span_rep.span_rep_layer
    self.count_pred = model.count_pred
    self.count_embed = model.count_embed
    self.max_width = model.max_width

  def forward(
      self,
      input_ids: torch.Tensor,
      attention_mask: torch.Tensor,
      word_positions: torch.Tensor,
      label_positions: torch.Tensor,
      prompt_position: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Computes entity count logits and per-label span probabilities.

    Args:
      input_ids: Subword token IDs of shape `[1, seq_len]`.
      attention_mask: Subword attention mask of shape `[1, seq_len]`.
      word_positions: First-subword indices per text word of shape `[1,
        max_words]`.
      label_positions: Subword indices of `[E]` label markers of shape `[1,
        max_labels]`.
      prompt_position: Subword index of the `[P]` prompt marker of shape `[1]`.

    Returns:
      A dictionary mapping `"count_logits"` (`[1, 20]`) and `"span_scores"`
      (`[1, max_labels, max_words, max_width]`) to their output tensors.
    """
    hidden = self.encoder(
        input_ids=input_ids.long(), attention_mask=attention_mask.long()
    ).last_hidden_state  # [1, S, D]

    words = gather(hidden, word_positions)  # [1, W, D]
    num_words = words.shape[1]
    starts = (
        torch.arange(num_words, device=words.device)
        .unsqueeze(1)
        .expand(-1, self.max_width)
    )
    ends = (
        starts + torch.arange(self.max_width, device=words.device).unsqueeze(0)
    ).clamp(max=num_words - 1)
    span_idx = torch.stack(
        [starts.reshape(-1), ends.reshape(-1)], dim=-1
    ).unsqueeze(0)
    span_rep = self.span_marker(words, span_idx)  # [1, W, max_width, D]

    prompt = gather(hidden, prompt_position.unsqueeze(0))[:, 0]  # [1, D]
    count_logits = self.count_pred(prompt)  # [1, 20]

    labels = gather(hidden, label_positions)[0]  # [K, D]
    struct = self.count_embed(labels, 1)[0]  # [K, D]

    logits = torch.einsum("bwmd,kd->bkwm", span_rep, struct)
    return {
        "count_logits": count_logits,
        "span_scores": torch.sigmoid(logits),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
  """Parses command-line flags for the GLiNER2 TFLite conversion script.

  Args:
    argv: Optional sequence of command-line argument strings. If `None`, reads
      arguments from `sys.argv[1:]`.

  Returns:
    Parsed command-line arguments namespace.
  """
  parser = argparse.ArgumentParser(
      description=__doc__,
      formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
      "--model-id",
      default=MODEL_ID,
      help="HuggingFace repo ID or local checkpoint directory.",
  )
  parser.add_argument(
      "--out-dir",
      type=pathlib.Path,
      default=pathlib.Path("out"),
      help="Output directory for the converted .tflite models and tokenizer.",
  )
  parser.add_argument(
      "--seq-len",
      type=int,
      default=DEFAULT_SEQ_LEN,
      help="Total subwords (label prompt + text); mDeBERTa supports up to 512.",
  )
  parser.add_argument(
      "--max-words",
      type=int,
      default=DEFAULT_MAX_WORDS,
      help="Maximum number of text words per inference call.",
  )
  parser.add_argument(
      "--max-labels",
      type=int,
      default=DEFAULT_MAX_LABELS,
      help="Maximum number of entity labels per inference call.",
  )
  parser.add_argument(
      "--no-quantize",
      action="store_true",
      help="Skip generating the dynamic-range int8 quantized model.",
  )
  return parser.parse_args(argv)


def export_tokenizer(model: nn.Module, out_dir: pathlib.Path) -> None:
  """Saves `tokenizer.json` (including GLiNER2's special tokens) into `out_dir`.

  Args:
    model: Loaded GLiNER2 model containing `processor.tokenizer` and `encoder`.
    out_dir: Destination directory where `tokenizer.json` will be written.

  Raises:
    RuntimeError: If the exported tokenizer vocabulary exceeds the encoder's
      embedding table size.
  """
  model.processor.tokenizer.save_pretrained(out_dir)
  for extra in (
      "tokenizer_config.json",
      "special_tokens_map.json",
      "added_tokens.json",
      "spm.model",
      "sentencepiece.bpe.model",
  ):
    (out_dir / extra).unlink(missing_ok=True)
  tokenizer_path = out_dir / "tokenizer.json"
  tokenizer = tokenizers.Tokenizer.from_file(str(tokenizer_path))
  vocab = tokenizer.get_vocab_size(with_added_tokens=True)
  embeddings = model.encoder.get_input_embeddings().num_embeddings
  if vocab > embeddings:
    raise RuntimeError(
        f"Tokenizer has {vocab} IDs, but encoder only has {embeddings}"
        " embeddings."
    )
  print(f"Wrote {tokenizer_path}")


def convert_to_tflite(
    graph: nn.Module,
    sample_kwargs: Mapping[str, torch.Tensor],
    float_path: pathlib.Path,
) -> None:
  """Traces and converts `graph` to a float32 `.tflite` model at `float_path`.

  Args:
    graph: Static-shape `EntityExtractionGraph` module to trace and export.
    sample_kwargs: Example input tensors keyed by `forward` parameter name.
    float_path: Output file path for the float32 `.tflite` model.
  """
  started = time.time()
  with torch.no_grad():
    edge_model = litert_torch.convert(graph, sample_kwargs=dict(sample_kwargs))
  edge_model.export(str(float_path))
  size_mib = float_path.stat().st_size / (2**20)
  elapsed = time.time() - started
  print(f"Wrote {float_path} ({size_mib:.0f} MiB) in {elapsed:.0f}s")


def quantize_int8(float_path: pathlib.Path, int8_path: pathlib.Path) -> None:
  """Applies dynamic-range int8 weight quantization to `float_path`.

  Args:
    float_path: Input path to the unquantized float32 `.tflite` model.
    int8_path: Output path for the dynamic-range int8 quantized `.tflite` model.
  """
  qt = quantizer.Quantizer(str(float_path), recipe.dynamic_wi8_afp32())
  qt.quantize().export_model(str(int8_path), overwrite=True)
  size_mib = int8_path.stat().st_size / (2**20)
  print(f"Wrote {int8_path} ({size_mib:.0f} MiB)")


def main() -> None:
  """Downloads the GLiNER2 checkpoint and exports LiteRT models."""
  args = parse_args()
  args.out_dir.mkdir(parents=True, exist_ok=True)

  model = gliner2.GLiNER2.from_pretrained(args.model_id)
  model.eval()
  model.to("cpu")

  export_tokenizer(model, args.out_dir)

  sample_kwargs = {
      "input_ids": torch.zeros((1, args.seq_len), dtype=torch.int32),
      "attention_mask": torch.ones((1, args.seq_len), dtype=torch.int32),
      "word_positions": torch.zeros((1, args.max_words), dtype=torch.int32),
      "label_positions": torch.zeros((1, args.max_labels), dtype=torch.int32),
      "prompt_position": torch.zeros((1,), dtype=torch.int32),
  }

  graph = EntityExtractionGraph(model).eval()
  float_path = args.out_dir / "gliner2_pii.tflite"
  convert_to_tflite(graph, sample_kwargs, float_path)

  if not args.no_quantize:
    int8_path = args.out_dir / "gliner2_pii_int8.tflite"
    quantize_int8(float_path, int8_path)


if __name__ == "__main__":
  main()

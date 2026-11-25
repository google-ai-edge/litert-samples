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

"""Unit tests for EntityExtractionGraph and helpers in convert.py."""

from __future__ import annotations

import pathlib
import sys
import types
from unittest import mock

import torch
from torch import nn

from google3.testing.pybase import googletest

# Provide lightweight module stubs for standalone OSS CLI packages that are not
# invoked when testing the PyTorch graph definition.
for _mod_name in (
    "ai_edge_quantizer",
    "ai_edge_quantizer.quantizer",
    "ai_edge_quantizer.recipe",
    "gliner2",
    "litert_torch",
    "tokenizers",
):
  sys.modules.setdefault(_mod_name, mock.MagicMock())

# pylint: disable=g-import-not-at-top
from google3.third_party.odml.litert.litert.samples.text_classifier.tflite_conversion import convert
# pylint: enable=g-import-not-at-top

_EXTRA_TOKENIZER_FILES = (
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "spm.model",
    "sentencepiece.bpe.model",
)


class _FakeEncoder(nn.Module):
  """Synthetic encoder returning deterministic hidden states."""

  def __init__(self, hidden_dim: int):
    super().__init__()
    self.hidden_dim = hidden_dim

  def forward(
      self, input_ids: torch.Tensor, attention_mask: torch.Tensor
  ) -> types.SimpleNamespace:
    batch_size, seq_len = input_ids.shape
    base = (input_ids * attention_mask).unsqueeze(-1).float()
    offsets = torch.arange(self.hidden_dim, dtype=torch.float32)
    return types.SimpleNamespace(
        last_hidden_state=(base + offsets).reshape(
            batch_size, seq_len, self.hidden_dim
        )
    )


class _FakeSpanRepLayer(nn.Module):
  """Synthetic span representation layer summing start and end word vectors."""

  def __init__(self, max_width: int):
    super().__init__()
    self.max_width = max_width

  def forward(
      self, words: torch.Tensor, span_idx: torch.Tensor
  ) -> torch.Tensor:
    batch_size, num_words, hidden_dim = words.shape
    starts = convert.gather(words, span_idx[:, :, 0])
    ends = convert.gather(words, span_idx[:, :, 1])
    return (starts + ends).reshape(
        batch_size, num_words, self.max_width, hidden_dim
    )


class _FakeCountEmbed(nn.Module):
  """Synthetic count-aware label embedding projection."""

  def forward(self, labels: torch.Tensor, count: int) -> torch.Tensor:
    del count
    return labels.unsqueeze(0)


class _FakeGliner2Model(nn.Module):
  """Synthetic GLiNER2 model container for testing EntityExtractionGraph."""

  def __init__(self, hidden_dim: int = 4, max_width: int = 3):
    super().__init__()
    self.encoder = _FakeEncoder(hidden_dim)
    self.span_rep = types.SimpleNamespace(
        span_rep_layer=_FakeSpanRepLayer(max_width)
    )
    self.count_pred = nn.Linear(hidden_dim, 20, bias=False)
    nn.init.constant_(self.count_pred.weight, 0.1)
    self.count_embed = _FakeCountEmbed()
    self.max_width = max_width


def _save_fake_pretrained_tokenizer(target_dir: pathlib.Path) -> None:
  """Writes a fake `tokenizer.json` and extra tokenizer artifacts."""
  (target_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
  for extra in _EXTRA_TOKENIZER_FILES:
    (target_dir / extra).write_text("extra", encoding="utf-8")


class ConvertTest(googletest.TestCase):

  def test_gather_selects_expected_sequence_positions(self):
    seq = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
    positions = torch.tensor([[4, 1, 0]], dtype=torch.int32)

    gathered = convert.gather(seq, positions)

    self.assertEqual(tuple(gathered.shape), (1, 3, 4))
    torch.testing.assert_close(gathered[0, 0], seq[0, 4])
    torch.testing.assert_close(gathered[0, 1], seq[0, 1])
    torch.testing.assert_close(gathered[0, 2], seq[0, 0])

  def test_entity_extraction_graph_forward_shapes_and_bounds(self):
    fake_model = _FakeGliner2Model(hidden_dim=4, max_width=3)
    graph = convert.EntityExtractionGraph(fake_model).eval()

    seq_len = 10
    max_words = 5
    max_labels = 2
    outputs = graph(
        input_ids=torch.arange(seq_len, dtype=torch.int32).unsqueeze(0),
        attention_mask=torch.ones((1, seq_len), dtype=torch.int32),
        word_positions=torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.int32),
        label_positions=torch.tensor([[6, 7]], dtype=torch.int32),
        prompt_position=torch.tensor([0], dtype=torch.int32),
    )

    self.assertCountEqual(outputs.keys(), ["count_logits", "span_scores"])
    self.assertEqual(tuple(outputs["count_logits"].shape), (1, 20))
    self.assertEqual(
        tuple(outputs["span_scores"].shape), (1, max_labels, max_words, 3)
    )
    self.assertTrue(torch.all(outputs["span_scores"] >= 0.0))
    self.assertTrue(torch.all(outputs["span_scores"] <= 1.0))

  def test_entity_extraction_graph_forward_values_and_clamping(self):
    fake_model = _FakeGliner2Model(hidden_dim=2, max_width=3)
    graph = convert.EntityExtractionGraph(fake_model).eval()

    outputs = graph(
        input_ids=torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        attention_mask=torch.ones((1, 4), dtype=torch.int32),
        word_positions=torch.tensor([[1, 2]], dtype=torch.int32),
        label_positions=torch.tensor([[3]], dtype=torch.int32),
        prompt_position=torch.tensor([0], dtype=torch.int32),
    )

    # Prompt at pos 0 has input_id=1 -> hidden=[1.0, 2.0]. Linear weight=0.1
    # gives 0.1 * (1.0 + 2.0) = 0.3 across all 20 count logits.
    torch.testing.assert_close(
        outputs["count_logits"], torch.full((1, 20), 0.3)
    )
    # For the last word (index 1, num_words=2), width=0, 1, 2 all clamp end to
    # index 1, so their span representations and scores are identical.
    last_word_scores = outputs["span_scores"][0, 0, 1]
    torch.testing.assert_close(last_word_scores[0], last_word_scores[1])
    torch.testing.assert_close(last_word_scores[1], last_word_scores[2])

  def test_export_tokenizer_writes_tokenizer_and_removes_extras(self):
    out_dir = pathlib.Path(self.create_tempdir().full_path)

    fake_model = mock.MagicMock()
    fake_model.processor.tokenizer.save_pretrained.side_effect = (
        _save_fake_pretrained_tokenizer
    )
    fake_model.encoder.get_input_embeddings.return_value.num_embeddings = 100

    with mock.patch.object(
        convert.tokenizers.Tokenizer, "from_file"
    ) as mock_from_file:
      mock_from_file.return_value.get_vocab_size.return_value = 100
      convert.export_tokenizer(fake_model, out_dir)

    self.assertTrue((out_dir / "tokenizer.json").exists())
    for extra in _EXTRA_TOKENIZER_FILES:
      self.assertFalse((out_dir / extra).exists())

  def test_export_tokenizer_raises_when_vocab_exceeds_embeddings(self):
    out_dir = pathlib.Path(self.create_tempdir().full_path)
    fake_model = mock.MagicMock()
    fake_model.encoder.get_input_embeddings.return_value.num_embeddings = 50

    with mock.patch.object(
        convert.tokenizers.Tokenizer, "from_file"
    ) as mock_from_file:
      mock_from_file.return_value.get_vocab_size.return_value = 55
      with self.assertRaisesRegex(
          RuntimeError, "Tokenizer has 55 IDs, but encoder only has 50"
      ):
        convert.export_tokenizer(fake_model, out_dir)

  def test_convert_to_tflite_and_quantize_int8(self):
    out_dir = pathlib.Path(self.create_tempdir().full_path)
    float_path = out_dir / "model.tflite"
    int8_path = out_dir / "model_int8.tflite"
    graph = nn.Identity()
    sample_kwargs = {"x": torch.zeros((1, 4), dtype=torch.int32)}

    with mock.patch.object(convert.litert_torch, "convert") as mock_convert:
      mock_convert.return_value.export.side_effect = (
          lambda path: pathlib.Path(path).write_bytes(b"float32")
      )
      convert.convert_to_tflite(graph, sample_kwargs, float_path)
      mock_convert.assert_called_once_with(graph, sample_kwargs=sample_kwargs)
      self.assertTrue(float_path.exists())

    with (
        mock.patch.object(
            convert.quantizer, "Quantizer"
        ) as mock_quantizer_cls,
        mock.patch.object(
            convert.recipe, "dynamic_wi8_afp32"
        ) as mock_recipe_fn,
    ):
      mock_export = (
          mock_quantizer_cls.return_value.quantize.return_value.export_model
      )
      mock_export.side_effect = (
          lambda path, overwrite: pathlib.Path(path).write_bytes(b"int8")
      )
      convert.quantize_int8(float_path, int8_path)
      mock_quantizer_cls.assert_called_once_with(
          str(float_path), mock_recipe_fn.return_value
      )
      mock_export.assert_called_once_with(str(int8_path), overwrite=True)
      self.assertTrue(int8_path.exists())

  def test_parse_args_defaults_and_overrides(self):
    defaults = convert.parse_args([])
    self.assertEqual(defaults.model_id, convert.MODEL_ID)
    self.assertEqual(defaults.out_dir, pathlib.Path("out"))
    self.assertEqual(defaults.seq_len, convert.DEFAULT_SEQ_LEN)
    self.assertEqual(defaults.max_words, convert.DEFAULT_MAX_WORDS)
    self.assertEqual(defaults.max_labels, convert.DEFAULT_MAX_LABELS)
    self.assertFalse(defaults.no_quantize)

    custom = convert.parse_args([
        "--model-id=custom/model",
        "--out-dir=/tmp/out",
        "--seq-len=256",
        "--max-words=128",
        "--max-labels=10",
        "--no-quantize",
    ])
    self.assertEqual(custom.model_id, "custom/model")
    self.assertEqual(custom.out_dir, pathlib.Path("/tmp/out"))
    self.assertEqual(custom.seq_len, 256)
    self.assertEqual(custom.max_words, 128)
    self.assertEqual(custom.max_labels, 10)
    self.assertTrue(custom.no_quantize)


if __name__ == "__main__":
  googletest.main()

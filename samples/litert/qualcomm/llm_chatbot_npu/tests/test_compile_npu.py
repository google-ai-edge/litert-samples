# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for compile_npu.py."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compile_npu  # pylint: disable=g-import-not-at-top


class CompileNpuTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._temp_dir = tempfile.TemporaryDirectory()
    self.temp_path = Path(self._temp_dir.name)
    self._saved_env = os.environ.copy()

  def tearDown(self):
    os.environ.clear()
    os.environ.update(self._saved_env)
    self._temp_dir.cleanup()
    super().tearDown()

  def test_rejects_litertlm_magic_header_with_actionable_message(self):
    litertlm_file = self.temp_path / "gemma-4-E2B-it.litertlm"
    litertlm_file.write_bytes(b"LITERTLM\x01\x00\x00\x00payload")

    with self.assertRaises(compile_npu.CompilationError) as ctx:
      compile_npu.validate_tflite_model(litertlm_file)

    self.assertIn("RTLM", str(ctx.exception))
    self.assertIn("huggingface.co/litert-community", str(ctx.exception))

  def test_rejects_corrupt_header_identifier(self):
    corrupt_file = self.temp_path / "bad.tflite"
    corrupt_file.write_bytes(b"\x1c\x00\x00\x00XXXX\x00\x00\x00\x00")

    with self.assertRaises(compile_npu.CompilationError) as ctx:
      compile_npu.validate_tflite_model(corrupt_file)

    self.assertIn("XXXX", str(ctx.exception))

  def test_rejects_missing_file_with_nonzero_exit(self):
    rc = compile_npu.main(
        ["--model_path", str(self.temp_path / "does_not_exist.tflite")]
    )
    self.assertEqual(rc, 1)

  def test_qairt_env_auto_detection_leaves_unset_when_none(self):
    os.environ.pop("QAIRT_ROOT", None)
    self.assertIsNone(compile_npu.configure_qairt_environment(None))
    self.assertNotIn("QAIRT_ROOT", os.environ)

  def test_qairt_env_sets_bin_and_lib_paths_when_valid(self):
    qairt_dir = self.temp_path / "qairt" / "2.44.0"
    (qairt_dir / "bin" / "x86_64-linux-clang").mkdir(parents=True)
    (qairt_dir / "lib" / "x86_64-linux-clang").mkdir(parents=True)

    resolved = compile_npu.configure_qairt_environment(str(qairt_dir))
    self.assertEqual(resolved, qairt_dir.resolve())
    self.assertEqual(os.environ["QAIRT_ROOT"], str(qairt_dir.resolve()))
    self.assertIn("x86_64-linux-clang", os.environ["PATH"])
    self.assertIn("x86_64-linux-clang", os.environ["LD_LIBRARY_PATH"])

  def test_compiles_valid_tfl3_model_with_mocked_aot_backend(self):
    tflite_file = self.temp_path / "model.tflite"
    tflite_file.write_bytes(b"\x1c\x00\x00\x00TFL3\x00\x00\x00\x00")
    out_dir = self.temp_path / "compiled_out"

    fake_aot = types.SimpleNamespace(aot_compile=mock.MagicMock())
    fake_soc = types.SimpleNamespace(SM8750="SM8750_ENUM")
    fake_target = types.SimpleNamespace(
        SocModel=fake_soc,
        Target=lambda soc: f"Target({soc})",
    )

    os.environ.pop("QAIRT_ROOT", None)
    with mock.patch.object(
        compile_npu, "load_aot_modules", return_value=(fake_aot, fake_target)
    ):
      rc = compile_npu.main([
          "--model-path",
          str(tflite_file),
          "--output-dir",
          str(out_dir),
          "--soc-model",
          "sm8750",
      ])

    self.assertEqual(rc, 0)
    self.assertTrue(out_dir.is_dir())
    fake_aot.aot_compile.assert_called_once_with(
        str(tflite_file.resolve()),
        output_dir=str(out_dir.resolve()),
        target=["Target(SM8750_ENUM)"],
    )


if __name__ == "__main__":
  unittest.main()

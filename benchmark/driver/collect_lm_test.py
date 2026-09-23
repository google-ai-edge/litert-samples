#!/usr/bin/env python3
"""Tests for collect_lm.py: a row from a `litert benchmark --ddp` session, and the exit code of each failure path.

No protoc is needed: the decoded proto text is stubbed. Run from this directory:
  python3 -m unittest collect_lm_test
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import collect_lm

SHA = "adac974bea147273b5bc64232d808905667eee69587161368146680df92e2d06"
MATRIX = f"""
platforms:
  android:
    devices:
      measured:
        - {{id: caiman-35, name: Pixel 9 Pro, os: Android 15 (API 35)}}
runtime_lm:
  version: "latest@2026-09-18"
  binary: latest/android_arm64/litert_lm/litert_lm_advanced_main
  sha256: {SHA}
  proto_ref: v0.17.0
  warmup_iterations: 1
lm_models:
  - {{repo: litert-community/Qwen3-0.6B, file: qwen3_0_6b_mixed_int4.litertlm, task: text-generation, backends: [cpu, gpu]}}
"""
PROVENANCE = f"""date: Fri Sep 18 20:10:57 PDT 2026
product: Pixel 9 Pro (caiman)
args: --backend=gpu --model_path=/data/local/tmp/litert-cli/qwen3_0_6b_mixed_int4.litertlm --benchmark=true --benchmark_prefill_tokens=1024 --benchmark_decode_tokens=256 --max_num_tokens=1280 --num_iterations=5 --metric_proto_file_path=/data/local/tmp/litert-cli/metrics.pb
sha256:
{SHA}  litert_lm_advanced_main
b1baab462f6be49d70eada79d715c2c52cd9ece0cad00bddf6a2c097d23498e9  /data/local/tmp/litert-cli/qwen3_0_6b_mixed_int4.litertlm
d9dd64c36f6e18661588e71cf6bad5860598e6ccd3a56dfb304c517f89ac609e  ./libLiteRtOpenClAccelerator.so
8e69e7d010e6787e34369b124b1eade0aa25d60b9d4bbed9dbb1a710901ff582  ./libLiteRtTopKOpenClSampler.so
"""
LOGCAT_GPU = "09-18 20:10:58.395 21780 21780 I litert  : [gpu_environment.cc:234] Created OpenCL device from provided device id and platform id.\n"


def decoded(iterations: list[tuple[float, float, float]], params: bool = True) -> str:
    """Decoded LitertLmMetricsList text with one block per (prefill tok/s, decode tok/s, ttft s)."""
    blocks = []
    for prefill, decode, ttft in iterations:
        blocks.append(
            "metrics {\n"
            + ("  benchmark_params {\n    num_prefill_tokens: 1024\n    num_decode_tokens: 256\n  }\n" if params else "")
            + '  init_phase_durations_us {\n    key: "Init Total"\n    value: 5684800\n  }\n'
            + f"  prefill_turns {{\n    tokens_per_second: {prefill}\n  }}\n"
            + f"  decode_turns {{\n    tokens_per_second: {decode}\n  }}\n"
            + f"  time_to_first_token_seconds: {ttft}\n}}\n"
        )
    return "".join(blocks)


class CollectLmTest(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.matrix = self.tmp / "matrix.yaml"
        self.matrix.write_text(MATRIX)
        self.data = self.tmp / "data"

    def job(self, session: str, name: str, provenance: str = PROVENANCE, logcat: str = LOGCAT_GPU, metrics: bool = True) -> pathlib.Path:
        d = self.tmp / session / name
        d.mkdir(parents=True)
        (d / "provenance.txt").write_text(provenance)
        (d / "logcat.txt").write_text(logcat)
        if metrics:
            (d / "metrics.pb").write_bytes(b"\x0a\x00")
        return d

    def run_collect(self, *session_dirs: pathlib.Path, text: str | None = decoded([(583.4, 19.9, 1.81), (585.3, 31.2, 1.78), (597.5, 24.0, 1.76)]),
                    extra=(), model: str = "litert-community/Qwen3-0.6B") -> tuple[int, str]:
        argv = ["collect_lm.py", *map(str, session_dirs), "--model", model,
                "--matrix", str(self.matrix), "--data-dir", str(self.data), *extra]
        err = io.StringIO()
        with mock.patch.object(collect_lm, "decode", return_value=text), mock.patch.object(sys, "argv", argv), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            try:
                code = collect_lm.main()
            except SystemExit as e:  # sys.exit(message): the message is the exit status the interpreter would print
                code = e.code if isinstance(e.code, int) else 1
                err.write(str(e.code))
        return code, err.getvalue()

    def rows(self) -> list[dict]:
        path = self.data / "measurements-lm.jsonl"
        return [json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []

    def test_row_from_a_gpu_job(self):
        code, _ = self.run_collect(self.job("session-1", "gpu-caiman-35"), extra=["--model-size-mb", "497.66"])
        self.assertEqual(code, 0)
        [row] = self.rows()
        self.assertEqual(row["row_id"], "litert-community/Qwen3-0.6B:qwen3_0_6b_mixed_int4.litertlm@latest@2026-09-18/android/caiman-35/gpu/p1024-d256-n1280")
        self.assertEqual((row["device"], row["accelerator"], row["delegate"], row["session"], row["job"]), ("Pixel 9 Pro", "gpu", "OpenCL", "session-1", "gpu-caiman-35"))
        self.assertEqual(row["metrics"]["prefill_tok_s"], 591.4)  # median of the iterations after the warm-up
        self.assertEqual(row["metrics"]["decode_tok_s"], 27.6)
        self.assertEqual(row["metrics"]["init_total_ms"], 5684.8)
        self.assertEqual(row["conditions"], {"prefill_tokens": 1024, "decode_tokens": 256, "max_num_tokens": 1280, "iterations": 3, "warmup_iterations": 1})
        self.assertEqual([l["name"] for l in row["libs"]], ["libLiteRtOpenClAccelerator.so", "libLiteRtTopKOpenClSampler.so"])
        self.assertEqual(row["libs"][0]["source"], "latest/android_arm64/litert_lm")
        self.assertEqual(row["binary"], f"latest/android_arm64/litert_lm/litert_lm_advanced_main, sha256 {SHA}")

    def test_same_row_id_replaces_the_earlier_row(self):
        job = self.job("session-1", "gpu-caiman-35")
        self.run_collect(job)
        code, _ = self.run_collect(job, text=decoded([(1, 1, 1), (600, 30, 1.7)]))
        self.assertEqual(code, 0)
        [row] = self.rows()
        self.assertEqual(row["metrics"]["prefill_tok_s"], 600)

    def test_binary_sha_mismatch_exits_1(self):
        job = self.job("session-1", "gpu-caiman-35", provenance=PROVENANCE.replace(SHA, "f" * 64))
        code, err = self.run_collect(job)
        self.assertEqual(code, 1)
        self.assertIn("is not the one matrix.yaml runtime_lm names", err)
        self.assertEqual(self.rows(), [])

    def test_provenance_without_the_binary_sha_exits_1(self):
        job = self.job("session-1", "gpu-caiman-35", provenance=PROVENANCE.replace(f"{SHA}  litert_lm_advanced_main\n", ""))
        code, err = self.run_collect(job)
        self.assertEqual((code, self.rows()), (1, []))
        self.assertIn("names no sha256 for the binary", err)

    def test_explicit_binary_turns_the_sha_check_off(self):
        job = self.job("session-1", "gpu-caiman-35", provenance=PROVENANCE.replace(SHA, "f" * 64))
        code, _ = self.run_collect(job, extra=["--binary", "some/dir/litert_lm_advanced_main", "--runtime-version", "v0.18.0"])
        self.assertEqual(code, 0)
        self.assertTrue(self.rows()[0]["row_id"].endswith("@v0.18.0/android/caiman-35/gpu/p1024-d256-n1280"))

    def test_gpu_job_without_a_gpu_api_exits_1(self):
        code, err = self.run_collect(self.job("session-1", "gpu-caiman-35", logcat="no accelerator lines\n"))
        self.assertEqual((code, self.rows()), (1, []))
        self.assertIn("names no GPU API", err)

    def test_missing_token_counts_exit_1(self):
        code, err = self.run_collect(self.job("session-1", "gpu-caiman-35"), text=decoded([(1, 1, 1), (2, 2, 2)], params=False))
        self.assertEqual((code, self.rows()), (1, []))
        self.assertIn("token counts unknown", err)
        code, err = self.run_collect(self.job("session-2", "gpu-caiman-35", provenance=PROVENANCE.replace(" --max_num_tokens=1280", "")))
        self.assertEqual((code, self.rows()), (1, []))

    def test_undecodable_metrics_exits_1(self):
        code, err = self.run_collect(self.job("session-1", "gpu-caiman-35"), text=None)
        self.assertEqual((code, self.rows()), (1, []))
        self.assertIn("metrics.pb not decoded", err)

    def test_no_iteration_beyond_the_warmup_exits_1(self):
        code, err = self.run_collect(self.job("session-1", "gpu-caiman-35"), text=decoded([(1, 1, 1)]))
        self.assertEqual((code, self.rows()), (1, []))
        self.assertIn("none beyond the 1 warm-up", err)

    def test_file_mismatch_exits_1(self):
        code, err = self.run_collect(self.job("session-1", "gpu-caiman-35"), extra=["--file", "other.litertlm"])
        self.assertEqual((code, self.rows()), (1, []))
        self.assertIn("--file says other.litertlm", err)

    def test_unknown_task_exits_1(self):
        code, err = self.run_collect(self.job("session-1", "gpu-caiman-35"), model="litert-community/NotListed")
        self.assertEqual((code, self.rows()), (1, []))
        self.assertIn("task unknown", err)

    def test_session_without_job_dirs_exits_1(self):
        (self.tmp / "session-empty").mkdir()
        code, err = self.run_collect(self.tmp / "session-empty")
        self.assertEqual((code, self.rows()), (1, []))
        self.assertIn("no job directories", err)

    def test_a_bad_path_stops_before_any_row(self):
        code, err = self.run_collect(self.job("session-1", "gpu-caiman-35"), self.tmp / "missing")
        self.assertEqual((code, self.rows()), (1, []))
        self.assertIn("not a directory", err)

    def test_one_failed_job_still_writes_the_others_and_exits_1(self):
        good = self.job("session-1", "gpu-caiman-35")
        self.job("session-1", "cpu-caiman-35", provenance=PROVENANCE.replace(" --max_num_tokens=1280", ""), logcat="")
        code, _ = self.run_collect(good.parent)
        self.assertEqual(code, 1)
        self.assertEqual([r["accelerator"] for r in self.rows()], ["gpu"])


if __name__ == "__main__":
    unittest.main()

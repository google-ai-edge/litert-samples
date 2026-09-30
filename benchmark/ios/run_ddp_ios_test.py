#!/usr/bin/env python3
# Copyright 2026 The AI Edge LiteRT Authors.
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
"""Tests for run_ddp_ios.py: one exit status per failure path, against a stand-in gcloud.

The stand-in is a shell script put first on PATH. It answers `devices list`, `sessions submit`,
`sessions list`, `sessions describe` and `storage cp` from files the test writes, and what it
does is switched by FAKE_GCLOUD_MODE. Its answers have the shapes seen on 2026-09-30 with gcloud
587: `devices list` and `sessions list` print lists, `sessions submit --async` printed `[]`, and
`sessions describe --full` a session whose pulled files sit flat under artifacts/.
Run: python3 run_ddp_ios_test.py
"""

import json
import os
import pathlib
import plistlib
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

HERE = pathlib.Path(__file__).resolve().parent
SCRIPT = HERE / "run_ddp_ios.py"
SID = "session-0123abcd-0000-4000-8000-000000000001"
PREFIX = f"gs://my-project-devicerun/automation/sessions/{SID}/job-000/execution-1111-2222/"
DEVICE = "iphone16pro-18-3"

FAKE_GCLOUD = r"""#!/bin/bash
# Stand-in gcloud for run_ddp_ios_test.py: one answer per command, switched by FAKE_GCLOUD_MODE.
echo "$*" >> "$FAKE_GCLOUD_LOG"
if [[ "$*" == *"devices list"* ]]; then
  if [ "$FAKE_GCLOUD_MODE" = auth_fails ]; then
    echo "ERROR: (gcloud.beta.device-run.devices.list) You do not currently have an active account selected." >&2
    exit 1
  fi
  cat "$FAKE_GCLOUD_DIR/devices.json"
elif [[ "$*" == *"sessions submit xctest"* ]]; then
  if [ "$FAKE_GCLOUD_MODE" = submit_fails ]; then
    echo "ERROR: (gcloud.beta.device-run.sessions.submit.xctest) PERMISSION_DENIED: Device Run API has not been used" >&2
    exit 1
  fi
  cat "$FAKE_GCLOUD_DIR/submit.json"
elif [[ "$*" == *"sessions list"* ]]; then
  # The list carries the submitted session with the run label the script passed, and a decoy.
  run="$(grep -o 'run=[0-9a-f]*' "$FAKE_GCLOUD_LOG" | tail -1)"
  sed "s/RUN_LABEL/${run#run=}/" "$FAKE_GCLOUD_DIR/list.json"
elif [[ "$*" == *"sessions describe"* ]]; then
  if [ "$FAKE_GCLOUD_MODE" = describe_not_json ]; then
    echo "ERROR: something else"
    exit 0
  fi
  cat "$FAKE_GCLOUD_DIR/describe.json"
elif [[ "$*" == *"storage cp"* ]]; then
  if [ "$FAKE_GCLOUD_MODE" = download_fails ]; then
    echo "ERROR: (gcloud.storage.cp) The following URLs matched no objects" >&2
    exit 1
  fi
  # Copies only what the session report lists, the way the real bucket would.
  dest="${@: -1}"
  mkdir -p "$dest"
  if grep -q '/artifacts/' "$FAKE_GCLOUD_DIR/describe.json"; then
    cp -R "$FAKE_GCLOUD_DIR/artifacts" "$dest/"
  fi
  cp "$FAKE_GCLOUD_DIR/system.log" "$dest/"
else
  echo "unexpected gcloud call: $*" >&2
  exit 1
fi
"""


def describe_json(result: str = "PASSED", pulled: bool = True, device: str = DEVICE,
                  accelerator: str = "cpu", model: str = "mobilenet_v2.tflite") -> dict:
    """A session in the shape of a real `sessions describe --full` (session-f25a0b01, 2026-09-30)."""
    files = [{"gcsOutputFile": {"path": f"{PREFIX}system.log"}}, {"gcsOutputFile": {"path": f"{PREFIX}junit.xml"}}]
    if pulled:
        files += [{"gcsOutputFile": {"path": f"{PREFIX}artifacts/{n}"}}
                  for n in ("benchmark.done", "results.pb", "stdout.txt", "runtime_info.pb")]
    return {
        "name": f"projects/my-project/locations/global/sessions/{SID}",
        "sessionConfig": {
            "displayName": "xctest",
            "jobConfigs": [{
                "action": {"iosXcTest": {"xcTestTimeout": "900s", "xcodeVersion": "xcode-26-2"}},
                "allocationConfig": {"deviceConfigs": [{"requirement": {"deviceId": device}}]},
                "labels": {"tool": "litert-samples-ios", "model": model, "accelerator": accelerator, "run": "abc"},
            }],
        },
        "sessionReport": {
            "id": "a0a7b280-not-the-session-id", "startTime": "2026-09-30T00:00:00Z", "endTime": "2026-09-30T00:01:05Z",
            "status": {"statusType": "DONE"}, "result": {"resultType": result},
            "jobReports": [{"displayName": "job-000", "id": "job-000",
                            "status": {"statusType": "DONE"}, "result": {"resultType": result},
                            "executionReports": [{"id": "execution-1111-2222", "outputFiles": files,
                                                  "result": {"resultType": result},
                                                  "status": {"statusType": "DONE",
                                                             "progressMessages": ["1 test cases passed"]}}]}],
        },
    }


class RunDdpIosTest(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp(prefix="run_ddp_ios_test."))
        self.fake_dir = self.tmp / "fake"
        self.fake_dir.mkdir()
        (self.fake_dir / "gcloud").write_text(FAKE_GCLOUD)
        (self.fake_dir / "gcloud").chmod(0o755)
        # Two catalog entries in the shape `devices list` prints (2026-09-30): the iPhone and a Pixel.
        (self.fake_dir / "devices.json").write_text(json.dumps([
            {"displayName": "Pixel 9 Pro", "osVersion": "35", "platform": "ANDROID", "formFactor": "PHONE",
             "name": "projects/my-project/locations/global/devices/caiman-35"},
            {"displayName": "iPhone 16 Pro", "osVersion": "18.3", "platform": "IOS", "formFactor": "PHONE",
             "hardwareType": "PHYSICAL", "iosDetails": {},
             "name": f"projects/my-project/locations/global/devices/{DEVICE}"}]))
        # `sessions submit xctest --async --format=json` printed an empty list; the script then finds the
        # session by its run label in `sessions list`, next to a decoy with another label.
        (self.fake_dir / "submit.json").write_text("[]\n")
        (self.fake_dir / "list.json").write_text(json.dumps(
            [{"name": "projects/my-project/locations/global/sessions/session-other",
              "sessionConfig": {"jobConfigs": [{"labels": {"tool": "litert-samples-ios", "run": "000000000000"}}]}},
             {"name": f"projects/my-project/locations/global/sessions/{SID}",
              "sessionConfig": {"jobConfigs": [{"labels": {"tool": "litert-samples-ios", "run": "RUN_LABEL"}}]},
              "sessionReport": {"status": {"statusType": "PENDING"}}}]))
        (self.fake_dir / "describe.json").write_text(json.dumps(describe_json()))
        out = self.fake_dir / "artifacts"  # flat, as the lab returns a pulled directory
        out.mkdir(parents=True)
        (out / "results.pb").write_bytes(b"\x08\x01")
        (out / "runtime_info.pb").write_bytes(b"\x08\x01")
        (out / "stdout.txt").write_text("INFO: Inference (avg): 2300 us\nINFO: Model initialization: 15 ms\n"
                                        "INFO: Overall footprint: 25 MB\n")
        (out / "benchmark.done").write_text("0\n")
        (self.fake_dir / "system.log").write_text("device log\n")
        self.model = self.tmp / "mobilenet_v2.tflite"
        self.model.write_bytes(b"TFL3")
        self.archive = self.tmp / "LiteRTBenchmark-xctest.zip"
        self.write_archive()
        self.script = self.tmp / "run_ddp_ios.py"
        # The copy under test retries `describe` once and finds the run label in one try, so a
        # failure path costs no 30 s waits.
        self.script.write_text(SCRIPT.read_text().replace("DESCRIBE_TRIES = 3", "DESCRIBE_TRIES = 1")
                               .replace("tries: int = 6", "tries: int = 1"))
        self.sessions = self.tmp / "sessions"
        self.log = self.tmp / "gcloud.log"

    def write_archive(self, manifest: dict | None = None, build_info: str | None = "source v2.2.0 (145c7523f)\n"):
        if manifest is None:  # FormatVersion 1 is what Xcode 26.1 wrote on 2026-09-30
            manifest = {"__xctestrun_metadata__": {"FormatVersion": 1},
                        "LiteRTBenchmarkTests": {"TestHostBundleIdentifier": "com.example.LiteRTBenchmark"}}
        with zipfile.ZipFile(self.archive, "w") as zf:
            zf.writestr("LiteRTBenchmark_iphoneos26.1-arm64.xctestrun", plistlib.dumps(manifest))
            zf.writestr("Release-iphoneos/LiteRTBenchmark.app/Info.plist", b"")
            if build_info is not None:
                zf.writestr("BUILD_INFO", build_info)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_script(self, *extra: str, mode: str = "ok", gcloud: bool = True, accelerator: str = "cpu",
                   device: str = DEVICE):
        env = {**os.environ, "FAKE_GCLOUD_MODE": mode, "FAKE_GCLOUD_DIR": str(self.fake_dir),
               "FAKE_GCLOUD_LOG": str(self.log), "LITERT_GCP_PROJECT": "my-project"}
        env["PATH"] = (f"{self.fake_dir}:" if gcloud else "") + "/usr/bin:/bin"
        cmd = [sys.executable, str(self.script), str(self.model), "--device", device,
               "--accelerator", accelerator, "--test-zip", str(self.archive), "--session-dir", str(self.sessions), *extra]
        return subprocess.run(cmd, capture_output=True, text=True, env=env)

    def gcloud_calls(self) -> list[str]:
        return self.log.read_text().splitlines() if self.log.exists() else []

    def test_ok(self):
        (self.fake_dir / "describe.json").write_text(json.dumps(describe_json(accelerator="gpu")))
        res = self.run_script(accelerator="gpu")
        self.assertEqual(res.returncode, 0, res.stderr)
        job_dir = self.sessions / SID / f"gpu-{DEVICE}"
        for name in ("results.pb", "runtime_info.pb", "stdout.txt", "benchmark.done"):
            self.assertTrue((job_dir / name).exists(), name)
        self.assertTrue((self.sessions / SID / "system.log").exists())
        meta = json.loads((self.sessions / SID / "session.json").read_text())
        self.assertEqual((meta["platform"], meta["runner"], meta["device"], meta["os"], meta["runtime_version"]),
                         ("ios", "ddp", "iPhone 16 Pro", "iOS 18.3", "2.2.0"))
        self.assertIn("source v2.2.0 (145c7523f)", meta["binary"])
        self.assertIn("Inference (avg)", res.stdout)
        self.assertIn('"--use_gpu=true"', res.stdout)
        submit = next(c for c in self.gcloud_calls() if "sessions submit xctest" in c)
        self.assertIn("com.example.LiteRTBenchmark:/Documents/mobilenet_v2.tflite", submit)
        self.assertIn("--paths-to-pull=com.example.LiteRTBenchmark:/Documents/out", submit)
        self.assertIn("--xcode-version=xcode-26-2", submit)
        self.assertIn("--async", submit)
        self.assertTrue(any("sessions list" in c for c in self.gcloud_calls()))  # found by the run label

    def test_submit_prints_session(self):
        (self.fake_dir / "submit.json").write_text(json.dumps(
            [{"name": f"projects/my-project/locations/global/sessions/{SID}", "sessionConfig": {}}]))
        res = self.run_script()
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertFalse(any("sessions list" in c for c in self.gcloud_calls()))

    def test_dry_run(self):
        res = self.run_script("--dry-run", "--", "--num_threads=4")
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertEqual(self.gcloud_calls(), [])
        self.assertIn("sessions submit xctest", res.stdout)
        self.assertIn('"--num_threads=4"', res.stdout)
        self.assertFalse((self.sessions / SID).exists())

    def test_missing_inputs(self):
        self.model.unlink()
        self.assertEqual(self.run_script().returncode, 2)
        self.model.write_bytes(b"TFL3")
        self.assertEqual(self.run_script("--timeout", "900").returncode, 2)
        self.assertEqual(self.run_script("--timeout", "2h").returncode, 2)
        self.assertEqual(self.run_script("--", "--use_gpu=true").returncode, 2)
        self.archive.unlink()
        self.assertEqual(self.run_script().returncode, 2)
        self.write_archive(build_info=None)
        self.assertIn("BUILD_INFO", self.run_script().stderr)
        self.write_archive(manifest={"__xctestrun_metadata__": {"FormatVersion": 1}})
        self.assertIn("TestHostBundleIdentifier", self.run_script().stderr)
        self.archive.write_bytes(b"not a zip")
        self.assertEqual(self.run_script().returncode, 2)
        with zipfile.ZipFile(self.archive, "w") as zf:
            zf.writestr("x.xctestrun", b"not a plist")
            zf.writestr("BUILD_INFO", "x")
        self.assertEqual(self.run_script().returncode, 2)
        self.assertEqual(self.gcloud_calls(), [])

    def test_session_dir_is_a_file(self):
        self.sessions.write_text("")
        res = self.run_script()
        self.assertEqual(res.returncode, 2)
        self.assertIn("--session-dir", res.stderr)
        self.assertEqual(self.gcloud_calls(), [])

    def test_device_not_ios(self):
        for device, word in (("caiman-35", "ANDROID"), ("iphone99-1-0", "not in the catalog")):
            res = self.run_script(device=device)
            self.assertEqual(res.returncode, 2, res.stderr)
            self.assertIn(word, res.stderr)
            self.assertIn("not an iPhone", res.stderr)
        self.assertFalse(any("sessions submit" in c for c in self.gcloud_calls()))

    def test_gcloud_missing(self):
        res = self.run_script(gcloud=False)
        self.assertEqual(res.returncode, 3)
        self.assertIn("gcloud is not on PATH", res.stderr)

    def test_auth_fails(self):
        res = self.run_script(mode="auth_fails")
        self.assertEqual(res.returncode, 3)
        self.assertIn("active account", res.stderr)

    def test_submit_fails(self):
        res = self.run_script(mode="submit_fails")
        self.assertEqual(res.returncode, 3)
        self.assertIn("PERMISSION_DENIED", res.stderr)
        self.assertFalse((self.sessions / SID).exists())

    def test_run_label_not_found(self):
        (self.fake_dir / "list.json").write_text(json.dumps(
            [{"name": "projects/my-project/locations/global/sessions/session-other",
              "sessionConfig": {"jobConfigs": [{"labels": {"tool": "litert-samples-ios", "run": "000000000000"}}]}}]))
        res = self.run_script()
        self.assertEqual(res.returncode, 3)
        self.assertIn("no session carries the label", res.stderr)

    def test_describe_shape(self):
        res = self.run_script(mode="describe_not_json")
        self.assertEqual(res.returncode, 3)
        self.assertIn("no JSON", res.stderr)
        (self.fake_dir / "describe.json").write_text(json.dumps({"name": "x", "sessionReport": {"endTime": "t"}}))
        res = self.run_script()
        self.assertEqual(res.returncode, 3)
        self.assertIn("0 jobs", res.stderr)

    def test_job_failed(self):
        (self.fake_dir / "describe.json").write_text(json.dumps(describe_json(result="FAILED", pulled=False)))
        res = self.run_script()
        self.assertEqual(res.returncode, 4)
        self.assertIn("FAILED", res.stderr)
        self.assertIn("system.log ends", res.stderr)
        self.assertTrue((self.sessions / SID / "ddp_session.json").exists())

    def test_download_fails(self):
        res = self.run_script(mode="download_fails")
        self.assertEqual(res.returncode, 5)
        self.assertIn("no results.pb came back", res.stderr)

    def test_missing_results(self):
        (self.fake_dir / "artifacts" / "results.pb").unlink()
        (self.fake_dir / "artifacts" / "benchmark.done").write_text("1\n")
        res = self.run_script()
        self.assertEqual(res.returncode, 5)
        self.assertIn("benchmark.done says 1", res.stderr)
        self.assertIn("Inference (avg)", res.stderr)

    def test_collect_existing_session(self):
        res = self.run_script("--collect", SID)
        self.assertEqual(res.returncode, 0, res.stderr)
        self.assertFalse(any("sessions submit" in c for c in self.gcloud_calls()))
        self.assertTrue((self.sessions / SID / f"cpu-{DEVICE}" / "results.pb").exists())
        res = self.run_script("--collect", SID)
        self.assertEqual(res.returncode, 2)
        self.assertIn("--force", res.stderr)
        self.assertEqual(self.run_script("--collect", SID, "--force").returncode, 0)

    def test_collect_mismatch(self):
        (self.fake_dir / "devices.json").write_text(json.dumps([
            {"displayName": "iPhone SE 3", "osVersion": "26.3", "platform": "IOS",
             "name": "projects/my-project/locations/global/devices/iphonese3-26-3"},
            {"displayName": "iPhone 16 Pro", "osVersion": "18.3", "platform": "IOS",
             "name": f"projects/my-project/locations/global/devices/{DEVICE}"}]))
        for kwargs, word in (({"accelerator": "gpu"}, "accelerator 'cpu'"),
                             ({"device": "iphonese3-26-3"}, "device 'iphone16pro-18-3'")):
            res = self.run_script("--collect", SID, **kwargs)
            self.assertEqual(res.returncode, 2, res.stderr)
            self.assertIn(word, res.stderr)
            self.assertFalse((self.sessions / SID).exists())


if __name__ == "__main__":
    unittest.main()

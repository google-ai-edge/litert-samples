# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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
# ==============================================================================

import sys

from emulator.env import environment_snapshot


def test_snapshot_has_required_keys():
    snap = environment_snapshot()
    for key in ("host", "platform", "machine", "chip", "os", "python",
                "packages", "cpu_count"):
        assert key in snap, f"missing key {key}"


def test_snapshot_reports_running_python():
    snap = environment_snapshot()
    assert snap["python"].startswith(
        f"{sys.version_info.major}.{sys.version_info.minor}"
    )


def test_snapshot_reports_package_versions():
    # Shape, not exact versions: the snapshot records whatever is installed (or
    # "not-installed"), so asserting a pinned version would make this test fail
    # on any checkout with a slightly different environment (e.g. no litert-lm
    # wheel on macOS). We only check the tracked packages appear as strings.
    snap = environment_snapshot()
    for name in ("ai-edge-litert", "litert-lm", "litert-lm-api"):
        assert name in snap["packages"]
        assert isinstance(snap["packages"][name], str) and snap["packages"][name]


def test_snapshot_records_background_load():
    # The machine's load at measurement time is what distinguishes a clean
    # run from a run on a machine with someone else's services running.
    # Without it, contamination is invisible.
    snap = environment_snapshot()
    assert "loadavg" in snap
    assert len(snap["loadavg"]) == 3
    assert all(isinstance(x, float) and x >= 0 for x in snap["loadavg"])


def test_snapshot_records_kernel_cmdline():
    snap = environment_snapshot()
    assert "kernel_cmdline" in snap
    assert isinstance(snap["kernel_cmdline"], str)

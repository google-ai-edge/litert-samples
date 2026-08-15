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

"""Environment snapshot embedded in every result row.

This is the only module that distinguishes Mac from Raspberry Pi. Everything
else in the rig is platform-neutral.
"""

from __future__ import annotations

import os
import platform
import shutil
import socket
import subprocess
import sys
from importlib import metadata

_PACKAGES = ("ai-edge-litert", "litert-lm", "litert-lm-api")


def _package_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in _PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out


def parse_cpuinfo_model(text: str) -> str | None:
    """Pull the Model line out of /proc/cpuinfo. A pure function for testability."""
    for line in text.splitlines():
        if line.startswith("Model"):
            return line.split(":", 1)[1].strip()
    return None


def parse_cpuinfo_features(text: str) -> list[str]:
    """Pull the Features list out of /proc/cpuinfo.

    This isn't just for reporting. The `asimddp` flag marks the Armv8.2
    dot-product instructions that int8 acceleration in XNNPACK and KleidiAI
    relies on. The A76 in Pi 5 has it; the A72 in Pi 4 and Compute Module 4
    doesn't. Recording it in every result row turns the claim "CM4 has no
    SDOT" from an assumption into a fixed fact.
    """
    for line in text.splitlines():
        if line.startswith("Features"):
            return line.split(":", 1)[1].split()
    return []


def _read_cpuinfo() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return ""


def _sysctl(name: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", name],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _chip() -> str:
    if sys.platform == "darwin":
        return _sysctl("machdep.cpu.brand_string") or "unknown"
    return parse_cpuinfo_model(_read_cpuinfo()) or platform.processor() or "unknown"


def _cpu_features() -> list[str]:
    if sys.platform == "darwin":
        return []
    return parse_cpuinfo_features(_read_cpuinfo())


def _has_dotprod() -> bool | None:
    """Whether dot-product instructions are available. None if it couldn't be determined."""
    if sys.platform == "darwin":
        raw = _sysctl("hw.optional.arm.FEAT_DotProd")
        return None if raw is None else raw == "1"
    features = _cpu_features()
    return "asimddp" in features if features else None


# The low bits of vcgencmd get_throttled are the state right now; the high
# bits (0x10000 and above) mean "has happened since boot" and stay set
# until reboot.
_THROTTLE_NOW_MASK = 0xF


def parse_throttle_outputs(temp_out: str, throttled_out: str) -> dict[str, object]:
    """Parse vcgencmd output: "temp=52.0'C" and "throttled=0x0".

    A separate pure function because vcgencmd doesn't exist at all on Mac,
    and without this the parsing logic would first run for real on a
    Raspberry Pi.
    """
    raw = throttled_out.strip().split("=", 1)[-1]
    try:
        value = int(raw, 16)
    except ValueError:
        value = -1  # vcgencmd failed — treat the run as suspect
    return {
        "temp": temp_out.strip().split("=", 1)[-1],
        "throttled": raw,
        "value": value,
        "now": value & _THROTTLE_NOW_MASK if value >= 0 else -1,
    }


def throttled_during_run(
    before: dict[str, object] | None, after: dict[str, object] | None
) -> bool | None:
    """Whether throttling happened specifically during this run.

    Checking `throttled == 0x0` won't work: the high bits are sticky and
    persist until reboot. Once the board overheats, it would mark every
    subsequent run as invalid, even completely cold ones. So we compare the
    value before and after: if it changed, throttling happened right here.

    Returns None if there's no data (e.g. on Mac).
    """
    if before is None or after is None:
        return None
    b, a = before.get("value", -1), after.get("value", -1)
    if b < 0 or a < 0:
        return True  # vcgencmd is broken — safer to treat the run as spoiled
    return a != b or bool(a & _THROTTLE_NOW_MASK)


def throttle_state() -> dict[str, object] | None:
    """Raspberry Pi temperature and throttle flags.

    Returns None on anything without vcgencmd — i.e. on Mac.
    """
    if shutil.which("vcgencmd") is None:
        return None

    outputs = []
    for args in (["measure_temp"], ["get_throttled"]):
        try:
            outputs.append(
                subprocess.run(
                    ["vcgencmd", *args], capture_output=True, text=True, check=True,
                ).stdout
            )
        except (OSError, subprocess.CalledProcessError):
            outputs.append("=error")
    return parse_throttle_outputs(outputs[0], outputs[1])


def load_average() -> list[float]:
    """Average load over 1, 5, and 15 minutes.

    The rig can flag runs spoiled by throttling, but it knew nothing about
    CPU contention. A board bought secondhand arrived with someone else's
    services running, and one of them was eating two-thirds of a core —
    such a run would look valid while being skewed low. Recording the load
    alongside the result makes the contamination visible in the data.
    """
    try:
        with open("/proc/loadavg", encoding="utf-8") as fh:
            return [float(x) for x in fh.read().split()[:3]]
    except (OSError, ValueError):
        pass
    try:
        return list(os.getloadavg())
    except (OSError, AttributeError):
        return []


def kernel_cmdline() -> str:
    """Kernel boot parameters.

    On Raspberry Pi, this shows traces of the machine's past life: e.g.
    cgroup_enable=memory left over from Docker, which a clean system
    wouldn't have. It affects memory and overhead, so it goes into the
    environment snapshot.
    """
    try:
        with open("/proc/cmdline", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def environment_snapshot() -> dict[str, object]:
    return {
        "host": socket.gethostname(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "chip": _chip(),
        "os": platform.platform(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "cpu_count": os.cpu_count(),
        "cpu_features": _cpu_features(),
        "has_dotprod": _has_dotprod(),
        "loadavg": load_average(),
        "kernel_cmdline": kernel_cmdline(),
    }

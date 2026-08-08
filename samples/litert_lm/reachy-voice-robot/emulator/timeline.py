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

"""Pipeline event timeline and its reconciliation against a measured budget.

The only way to tell whether the pipeline works as a whole is to compare
actual timestamps against numbers gathered on the bench. That's why the
timeline comes before all the stages: without it, every subsequent task
would be unverifiable.

Total time is measured by the wall clock, not by summing durations: stages
run in threads, and the sum would exceed real elapsed time.
"""

from __future__ import annotations

import contextlib
import dataclasses
import time

from emulator.env import environment_snapshot, throttle_state

# Stage costs measured on a Raspberry Pi 5.
#
# The keys MUST match the span names the pipeline actually emits (see
# emulator/pipeline.py): detector / asr_full / llm_reply / tts_call. An earlier
# version keyed this on finer names (asr_encode / asr_decode_token / llm_ttft)
# that no stage ever emits, so check_budget silently skipped the two heaviest
# stages (ASR and the LLM) and always reported "over budget: none".
BUDGET_MS: dict[str, float] = {
    "detector": 220.0,   # yolox-tiny, 416x416 frame (the 's' variant caught small objects, but called a cup a dog at 0.51 confidence and cost +0.5s)
    "asr_full": 390.0,   # moonshine, 5s window: ~51ms encode + ~9 decode tokens at ~37ms each ≈ 384ms (matches the end-to-end STT budget)
    "llm_reply": 6100.0, # Gemma 4 E2B, full offline reply (streaming hides this; the offline pipeline path pays it in full — TTFT alone is ~1.2s)
    "tts_call": 225.0,   # inflect (default --tts), ~210ms per phrase measured (225 budget/ceiling)
}


@dataclasses.dataclass(frozen=True)
class Span:
    name: str
    start_ms: float
    end_ms: float
    failed: bool = False

    @property
    def duration_ms(self) -> float:
        return self.end_ms - self.start_ms


class Timeline:
    """Collects timestamps for the stages of a single pipeline pass."""

    def __init__(self) -> None:
        self._origin = time.perf_counter()
        self._spans: list[Span] = []

    def _now_ms(self) -> float:
        return (time.perf_counter() - self._origin) * 1000.0

    @contextlib.contextmanager
    def stage(self, name: str):
        start = self._now_ms()
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            # The stage is recorded even on exception: otherwise it's unclear
            # exactly where the pipeline broke and how long it took.
            self._spans.append(
                Span(name=name, start_ms=start, end_ms=self._now_ms(),
                     failed=failed)
            )

    def mark(self, name: str, duration_ms: float) -> None:
        """Record a stage that was measured externally."""
        start = self._now_ms()
        self._spans.append(
            Span(name=name, start_ms=start, end_ms=start + duration_ms)
        )

    def spans(self) -> list[Span]:
        return sorted(self._spans, key=lambda s: s.start_ms)

    def total_ms(self) -> float:
        if not self._spans:
            return 0.0
        return max(s.end_ms for s in self._spans)

    def check_budget(self, tolerance: float = 1.5) -> list[str]:
        """Stages that exceeded budget by more than `tolerance`x."""
        problems = []
        for span in self.spans():
            budget = BUDGET_MS.get(span.name)
            if budget is None:
                continue
            if span.duration_ms > budget * tolerance:
                problems.append(
                    f"{span.name}: {span.duration_ms:.0f}ms vs "
                    f"budget {budget:.0f}ms "
                    f"({span.duration_ms / budget:.1f}x)"
                )
        return problems

    def to_record(self) -> dict:
        return {
            "total_ms": self.total_ms(),
            "spans": [dataclasses.asdict(s) | {"duration_ms": s.duration_ms}
                      for s in self.spans()],
            "budget_problems": self.check_budget(),
            "env": environment_snapshot(),
            "throttle": throttle_state(),
        }

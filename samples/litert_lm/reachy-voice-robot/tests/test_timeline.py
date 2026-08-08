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

"""The timeline is the only way to know whether the pipeline works.

Separately verifies that a stage is recorded even when it raises: without
this, a failure leaves no way to tell exactly where the pipeline stopped and
how long it took — and a failure in the middle of a four-stage pipeline has
nothing else to debug it with.
"""

import time

import pytest

from emulator.timeline import BUDGET_MS, Timeline


def test_stage_records_duration():
    tl = Timeline()
    with tl.stage("detector"):
        time.sleep(0.02)
    spans = tl.spans()
    assert len(spans) == 1
    assert spans[0].name == "detector"
    assert 15 <= spans[0].duration_ms <= 200


def test_stages_are_ordered_by_start():
    tl = Timeline()
    with tl.stage("asr"):
        pass
    with tl.stage("llm"):
        pass
    assert [s.name for s in tl.spans()] == ["asr", "llm"]


def test_total_is_wall_clock_not_sum():
    # Stages can run in parallel across threads, so the sum of durations
    # exceeds wall-clock time. The budget is checked against the wall clock.
    tl = Timeline()
    with tl.stage("a"):
        time.sleep(0.01)
    with tl.stage("b"):
        time.sleep(0.01)
    assert tl.total_ms() >= 15


def test_budget_keys_match_the_stages_the_pipeline_emits():
    # BUDGET_MS must be keyed on the span names Pipeline.step actually emits
    # (emulator/pipeline.py: detector / asr_full / llm_reply / tts_call) — NOT
    # finer names like asr_encode/llm_ttft that no stage ever emits, which made
    # check_budget silently skip the two heaviest stages (ASR and the LLM) and
    # always report "over budget: none".
    assert set(BUDGET_MS) == {"detector", "asr_full", "llm_reply", "tts_call"}


def test_check_budget_covers_asr_and_llm_the_two_heaviest_stages():
    # Regression guard for that blind spot: a slow ASR or LLM must be flagged,
    # not silently skipped because its span name has no budget entry.
    for stage in ("asr_full", "llm_reply"):
        tl = Timeline()
        tl.mark(stage, BUDGET_MS[stage] * 2.0)
        assert any(stage in p for p in tl.check_budget(tolerance=1.5)), \
            f"{stage} over budget was not flagged"


def test_tts_call_budget_matches_inflect():
    # The synthesizer (Inflect-Nano-v2) runs ~225 ms/phrase. The budget must
    # reflect that, or check_budget wouldn't catch even a threefold regression.
    assert BUDGET_MS["tts_call"] == pytest.approx(225.0)


def test_check_budget_flags_slow_stage():
    tl = Timeline()
    # Twice the budget — clearly outside tolerance, we expect a complaint.
    # Relative to BUDGET_MS so the test survives a change of detector.
    tl.mark("detector", BUDGET_MS["detector"] * 2.0)
    problems = tl.check_budget(tolerance=1.5)
    assert any("detector" in p for p in problems)


def test_check_budget_silent_within_tolerance():
    tl = Timeline()
    tl.mark("detector", BUDGET_MS["detector"] * 1.2)
    assert tl.check_budget(tolerance=1.5) == []


def test_record_carries_environment():
    tl = Timeline()
    with tl.stage("detector"):
        pass
    record = tl.to_record()
    assert record["env"]["machine"] in {"aarch64", "arm64", "x86_64"}
    assert "throttle" in record
    assert record["spans"][0]["name"] == "detector"


def test_exception_inside_stage_still_records_span():
    tl = Timeline()
    with pytest.raises(ValueError):
        with tl.stage("tts"):
            raise ValueError("synthesis failure")
    # The stage must be recorded with a failed flag, otherwise debugging is impossible.
    assert tl.spans()[0].name == "tts"
    assert tl.spans()[0].failed is True

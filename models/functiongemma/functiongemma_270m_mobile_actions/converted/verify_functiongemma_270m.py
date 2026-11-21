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

"""Tool-routing gate for the published FunctionGemma 270M mobile-actions bundle.

Four modes. The first is the gate; the rest each isolate one variable that
changes the verdict, so a deployment can tell which knob is responsible.

  gate (default) — scores tool routing against a fixed prompt set, one fresh
    conversation per prompt, greedy, with `automatic_tool_calling=False` so the
    model emits exactly one call and the reply is compared to the expectation.
    Each prompt has a paraphrased twin, so a model that matches keywords rather
    than meaning cannot pass by accident.

      python verify_functiongemma_270m.py model.litertlm --mode gate

  --mode scaling — re-runs the same prompts against 1-, 2-, 4- and 8-tool
    sets. Accuracy is scored only over prompts whose expected tool is in the
    offered set, so a 1-tool cell is not credited for the 16 prompts it could
    not answer. The published bundle peaks at 4.

  --mode sensitivity — same prompts, same tools, tool descriptions reworded.
    Reports how many verdicts flip. This is the reason a tool set cannot be
    chosen by intuition.

  --mode multi-turn — three turns through one conversation with automatic tool
    calling, which exercises the bundle's `<start_function_response>` round trip
    and its template's prefix contract.

  --mode floor — the cookbook's eight-question floor gate, printed as a
    non-pass. This bundle is a function-calling finetune: it answers every one
    of the eight with its tool-refusal line, so the gate scores 0/8 by design
    rather than by defect. Kept here so nobody re-derives that.

Needs `pip install litert-lm`. Every number in the recipe README comes from one
of these modes; `--out` writes the JSON the README's tables are transcribed from.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Any

import litert_lm
from litert_lm import interfaces

# The eight actions of the mobile-actions finetune. Names, descriptions and
# argument names are all variables under test: the bundle is a keyword matcher,
# so a schema it has not seen routes by whatever tokens happen to overlap.
TOOL_SPECS: dict[str, tuple[str, dict[str, str]]] = {
    "open_flashlight": ("Turns the phone's flashlight on.", {}),
    "close_flashlight": ("Turns the phone's flashlight off.", {}),
    "query_calendar": ("Lists the events on the user's calendar for today.", {}),
    "take_photo": ("Takes a photo with the phone's camera.", {}),
    "set_alarm": (
        "Sets an alarm on the phone.",
        {"time": "The 24-hour alarm time as HH:MM."},
    ),
    "send_message": (
        "Sends a text message to a contact.",
        {
            "contact_name": "Name of the contact to message.",
            "message": "The text of the message to send.",
        },
    ),
    "append_note": (
        "Appends a note with a title to the notes file.",
        {"title": "The title of the note."},
    ),
    "noop": ("Does nothing.", {}),
}

# Tool sets to compare. The prefixes keep the flashlight pair first because
# those are the routes that resolve; that is the finding, not the hypothesis.
TOOL_SETS: dict[str, list[str]] = {
    "1_tool": ["query_calendar"],
    "2_tools": ["open_flashlight", "query_calendar"],
    "4_tools": [
        "open_flashlight",
        "close_flashlight",
        "query_calendar",
        "take_photo",
    ],
    "8_tools": list(TOOL_SPECS),
}

# prompt -> (expected tool, expected argument subset, prompt kind)
# `kind` is literal when the prompt carries the tool's own vocabulary and
# paraphrase when it does not. A semantic router scores both alike.
PROBES: list[tuple[str, str, dict[str, Any], str]] = [
    ("what is on my calendar today", "query_calendar", {}, "literal"),
    ("whats on my calender today", "query_calendar", {}, "paraphrase"),
    ("calendar today", "query_calendar", {}, "terse"),
    ("what's my schedule", "query_calendar", {}, "paraphrase"),
    ("list my calendar events", "query_calendar", {}, "paraphrase"),
    ("do i have meetings today", "query_calendar", {}, "paraphrase"),
    ("am i busy today", "query_calendar", {}, "paraphrase"),
    ("show my events", "query_calendar", {}, "paraphrase"),
    ("turn on the flashlight", "open_flashlight", {}, "literal"),
    ("i need some light in here", "open_flashlight", {}, "paraphrase"),
    ("it is dark in here", "open_flashlight", {}, "paraphrase"),
    ("lights on please", "open_flashlight", {}, "paraphrase"),
    ("turn off the flashlight", "close_flashlight", {}, "literal"),
    ("kill the torch now", "close_flashlight", {}, "paraphrase"),
    ("shut the light", "close_flashlight", {}, "paraphrase"),
    ("no more light", "close_flashlight", {}, "paraphrase"),
]

# Additional tools the 4-tool cell cannot score; they carry the only
# argument-passing cases in the set.
ARGUMENT_PROBES: list[tuple[str, str, dict[str, Any], str]] = [
    ("take a photo", "take_photo", {}, "literal"),
    ("snap a picture of the room", "take_photo", {}, "paraphrase"),
    ("set an alarm for 07:30", "set_alarm", {"time": "07:30"}, "literal"),
    ("wake me up at 06:00", "set_alarm", {"time": "06:00"}, "paraphrase"),
    (
        "send a message to mom saying hi",
        "send_message",
        {"contact_name": "mom"},
        "paraphrase",
    ),
    (
        "text alex that i am running late",
        "send_message",
        {"contact_name": "alex"},
        "paraphrase",
    ),
    ("write a note titled groceries", "append_note", {"title": "groceries"}, "literal"),
    ("remind me to buy milk", "append_note", {"title": "milk"}, "paraphrase"),
    ("what is the weather in nairobi", "noop", {}, "paraphrase"),
    ("thanks, that was helpful", "noop", {}, "literal"),
]

FLOOR_QUESTIONS = [
    ("What is 17 + 25?", "42"),
    ("What is the capital of Japan?", "Tokyo"),
    ('What is the opposite of "hot"?', "cold"),
    ("How many days are in a week?", "seven"),
    ('How do you say "thank you" in French?', "merci"),
    ("What is 8 times 7?", "56"),
    ("Which is larger: 0.9 or 0.11?", "0.9"),
    ('Complete the rhyme: "Roses are red, violets are ___"', "blue"),
]


class SchemaTool(interfaces.Tool):
  """A tool with a hand-written OpenAPI schema.

  `litert_lm.tools` derives schemas from Python signatures via `inspect`, which
  cannot express a description-only tool, and cannot put a description on an
  argument without a Google-style docstring. Both are needed here because the
  description text is one of the variables under test.
  """

  def __init__(self, spec: dict[str, Any]) -> None:
    self._spec = spec

  def get_tool_description(self) -> dict[str, Any]:
    return self._spec

  def execute(self, param: Any) -> str:
    del param
    return "ok"


def make_tool(name: str) -> SchemaTool:
  """Builds a SchemaTool from TOOL_SPECS."""
  description, props = TOOL_SPECS[name]
  return SchemaTool({
      "type": "function",
      "function": {
          "name": name,
          "description": description,
          "parameters": {
              "type": "object",
              "properties": {
                  key: {"type": "string", "description": text}
                  for key, text in props.items()
              },
              "required": list(props),
          },
      },
  })


def make_engine(model_path: str, backend: str) -> litert_lm.Engine:
  """Creates an engine with a cache dir that is not beside the bundle.

  The default disk cache writes delegate caches of up to twice the model size
  next to the .litertlm, which for a 289 MB bundle means ~600 MB of unrequested
  writes in the caller's model directory.
  """
  return litert_lm.Engine(
      model_path=model_path,
      backend=(
          interfaces.Backend.GPU()
          if backend == "gpu"
          else interfaces.Backend.CPU()
      ),
      max_num_tokens=2048,
      cache_dir=os.path.abspath(".litertlm-cache"),
  )


def read_reply(response: Any) -> tuple[str | None, dict[str, Any], str]:
  """Returns (tool name, arguments, prose) from a non-automatic tool reply.

  With `automatic_tool_calling=False` the model emits one call and the
  conversation returns it to the caller in `tool_calls`; the handler never
  fires. The model's refusal to call is prose in `content`, not an empty
  `tool_calls` list, so both are read here.
  """
  calls = response.get("tool_calls") or []
  if calls:
    function = calls[0].get("function", {})
    args = function.get("arguments") or {}
    if isinstance(args, str):
      try:
        args = json.loads(args)
      except json.JSONDecodeError:
        args = {"_unparsed": args}
    if not isinstance(args, dict):
      args = {"_unexpected": args}
    return function.get("name"), args, ""

  prose = "".join(
      part.get("text", "") for part in response.get("content") or []
  )
  return None, {}, prose.strip()


def args_match(expected: dict[str, Any], actual: dict[str, Any]) -> bool:
  """True when every expected argument is present and equal, case-insensitively."""
  for key, value in expected.items():
    got = actual.get(key)
    if got is None:
      return False
    if str(got).strip().lower() != str(value).strip().lower():
      return False
  return True


def ask(
    engine: Any, tools: list[SchemaTool], prompt: str
) -> dict[str, Any]:
  """One prompt, one fresh conversation, greedy. No KV cache is shared."""
  with engine.create_conversation(
      tools=tools, automatic_tool_calling=False
  ) as convo:
    start = time.monotonic()
    response = convo.send_message(prompt)
    elapsed = time.monotonic() - start
  got_tool, got_args, prose = read_reply(response)
  return {
      "got_tool": got_tool,
      "got_args": got_args,
      "refusal": got_tool is None,
      "prose": prose,
      "seconds": round(elapsed, 3),
  }


def score(rows: list[dict[str, Any]], offered: set[str]) -> dict[str, Any]:
  """Accuracy over the rows whose expected tool was actually on offer."""
  scored = [r for r in rows if r["expected_tool"] in offered]
  tool_hits = sum(r["tool_match"] for r in scored)
  full_hits = sum(r["tool_match"] and r["args_match"] for r in scored)
  refusals = sum(1 for r in rows if r["refusal"])
  return {
      "prompts_scored": len(scored),
      "prompts_total": len(rows),
      "tool_name_accuracy": round(tool_hits / len(scored), 4) if scored else 0.0,
      "exact_accuracy": round(full_hits / len(scored), 4) if scored else 0.0,
      "refusal_rate": round(refusals / len(rows), 4) if rows else 0.0,
      "median_seconds": round(
          statistics.median([r["seconds"] for r in rows]), 3
      )
      if rows
      else 0.0,
  }


def mode_gate(engine: Any, tool_names: list[str]) -> dict[str, Any]:
  """Routing accuracy over the full prompt set at one tool-set size."""
  tools = [make_tool(name) for name in tool_names]
  rows = []
  for prompt, expected, expected_args, kind in PROBES + ARGUMENT_PROBES:
    result = ask(engine, tools, prompt)
    tool_ok = result["got_tool"] == expected
    rows.append({
        "prompt": prompt,
        "expected_tool": expected,
        "expected_args": expected_args,
        "kind": kind,
        "got_tool": result["got_tool"],
        "got_args": result["got_args"],
        "tool_match": tool_ok,
        "args_match": tool_ok and args_match(expected_args, result["got_args"]),
        "refusal": result["refusal"],
        "prose": result["prose"],
        "seconds": result["seconds"],
    })
    print(
        f"  {'OK ' if tool_ok else 'BAD'} {kind:11} {prompt!r:44} "
        f"want={expected:16} got={result['got_tool'] or 'refusal'}"
    )
  return {"tools": tool_names, "rows": rows, **score(rows, set(tool_names))}


def mode_scaling(engine: Any) -> dict[str, Any]:
  """The same prompts at four tool-set sizes."""
  results = []
  for label, names in TOOL_SETS.items():
    print(f"\n== {label}: {', '.join(names)}")
    cell = mode_gate(engine, names)
    cell["tool_set"] = label
    cell["n_tools"] = len(names)
    results.append(cell)
  return {"mode": "scaling", "cells": results}


def mode_sensitivity(engine: Any) -> dict[str, Any]:
  """Literal vs paraphrase routing, then the same prompts with reworded tools."""
  worded = [make_tool(name) for name in TOOL_SETS["4_tools"]]

  print("== literal vs paraphrase (4 tools, finetune wording)")
  base_rows = []
  for prompt, expected, expected_args, kind in PROBES:
    result = ask(engine, worded, prompt)
    base_rows.append({
        "prompt": prompt,
        "expected_tool": expected,
        "expected_args": expected_args,
        "kind": kind,
        "tool_match": result["got_tool"] == expected,
        "refusal": result["refusal"],
        "prose": result["prose"],
        "seconds": result["seconds"],
    })
    print(
        f"  {'OK ' if result['got_tool'] == expected else 'BAD'} "
        f"{kind:11} {prompt!r:38} -> {result['got_tool'] or 'refusal'}"
    )

  by_kind: dict[str, list[bool]] = {}
  for row in base_rows:
    by_kind.setdefault(row["kind"], []).append(row["tool_match"])
  kind_summary = {
      kind: {"n": len(hits), "accuracy": round(sum(hits) / len(hits), 4)}
      for kind, hits in sorted(by_kind.items())
  }

  # Reword every description, keep the names, and ask the same questions again.
  # A router that reads the description moves; one that matches prompt tokens
  # against tool names does not.
  reworded = [
      SchemaTool({
          "type": "function",
          "function": {
              "name": name,
              "description": description,
              "parameters": {
                  "type": "object",
                  "properties": {},
                  "required": [],
              },
          },
      })
      for name, description in [
          ("open_flashlight", "Enables the LED torch."),
          ("close_flashlight", "Disables the LED torch."),
          ("query_calendar", "Shows the user's schedule."),
          ("take_photo", "Captures a still image."),
      ]
  ]

  print("\n== same prompts, tool descriptions reworded")
  wording_rows = []
  for row in base_rows:
    before = ask(engine, worded, row["prompt"])
    after = ask(engine, reworded, row["prompt"])
    wording_rows.append({
        "prompt": row["prompt"],
        "expected_tool": row["expected_tool"],
        "kind": row["kind"],
        "match_worded": before["got_tool"] == row["expected_tool"],
        "match_reworded": after["got_tool"] == row["expected_tool"],
    })
    print(
        f"  {row['prompt']!r:38} worded={before['got_tool'] or 'refusal':16} "
        f"reworded={after['got_tool'] or 'refusal'}"
    )

  flips = [
      r for r in wording_rows if r["match_worded"] != r["match_reworded"]
  ]
  return {
      "mode": "sensitivity",
      "literal_vs_paraphrase": {"by_kind": kind_summary, "rows": base_rows},
      "description_wording": {
          "n_prompts": len(wording_rows),
          "n_flipped": len(flips),
          "flipped": flips,
          "rows": wording_rows,
      },
  }


def mode_multi_turn(engine: Any) -> dict[str, Any]:
  """Three turns, automatic tool calling, one conversation."""
  tools = [make_tool(name) for name in TOOL_SETS["4_tools"]]
  turns = [
      "turn on the flashlight",
      "what is on my calendar today",
      "now turn it off",
  ]
  rows = []
  with engine.create_conversation(tools=tools, automatic_tool_calling=True) as convo:
    for prompt in turns:
      start = time.monotonic()
      response = convo.send_message(prompt)
      elapsed = time.monotonic() - start
      text = "".join(
          part.get("text", "") for part in response.get("content") or []
      )
      rows.append({"prompt": prompt, "reply": text, "seconds": round(elapsed, 3)})
      print(f"  {prompt!r:34} -> {text!r}")
  return {"mode": "multi_turn", "turns": rows}


def mode_floor(engine: Any) -> dict[str, Any]:
  """The cookbook's eight-question floor gate, reported as expected to fail."""
  rows = []
  for question, expected in FLOOR_QUESTIONS:
    with engine.create_conversation() as convo:
      response = convo.send_message(f"{question} Answer briefly.")
    text = "".join(
        part.get("text", "") for part in response.get("content") or []
    )
    correct = expected.lower() in text.lower()
    rows.append({
        "question": question,
        "expected": expected,
        "reply": text.strip(),
        "correct": correct,
    })
    print(f"  {'OK ' if correct else 'BAD'} {question!r:58} -> {text.strip()!r}")
  return {
      "mode": "floor",
      "note": (
          "A function-calling finetune answers general questions with its "
          "tool-refusal line, so this gate is 0/8 by design. It is not a "
          "quality verdict on the bundle."
      ),
      "correct": sum(r["correct"] for r in rows),
      "total": len(rows),
      "rows": rows,
  }


def main() -> int:
  parser = argparse.ArgumentParser(
      description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
  )
  parser.add_argument("model", help="path to the .litertlm bundle")
  parser.add_argument(
      "--mode",
      default="gate",
      choices=["gate", "scaling", "sensitivity", "multi-turn", "floor"],
  )
  parser.add_argument("--backend", default="cpu", choices=["cpu", "gpu"])
  parser.add_argument("--out", default="", help="write the JSON result here")
  args = parser.parse_args()

  if not os.path.exists(args.model):
    parser.error(f"model not found: {args.model}")

  engine = make_engine(args.model, args.backend)

  if args.mode == "scaling":
    report = mode_scaling(engine)
  elif args.mode == "sensitivity":
    report = mode_sensitivity(engine)
  elif args.mode == "multi-turn":
    report = mode_multi_turn(engine)
  elif args.mode == "floor":
    report = mode_floor(engine)
  else:
    print(f"== gate, {len(TOOL_SETS['8_tools'])} tools")
    report = {"mode": "gate", **mode_gate(engine, TOOL_SETS["8_tools"])}

  report["model"] = os.path.basename(args.model)
  report["backend"] = args.backend

  if args.out:
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as handle:
      json.dump(report, handle, indent=2)
    print(f"\nwrote {args.out}")

  if args.mode == "scaling":
    print("\n| tool set | n | scored | tool acc | exact | refusal | median s |")
    print("|---|---|---|---|---|---|---|")
    for cell in report["cells"]:
      print(
          f"| {cell['tool_set']} | {cell['n_tools']} | "
          f"{cell['prompts_scored']} | {cell['tool_name_accuracy']:.0%} | "
          f"{cell['exact_accuracy']:.0%} | {cell['refusal_rate']:.0%} | "
          f"{cell['median_seconds']:.2f} |"
      )
  elif args.mode == "gate":
    print(
        f"\ntool name accuracy {report['tool_name_accuracy']:.0%} "
        f"over {report['prompts_scored']} prompts, "
        f"exact {report['exact_accuracy']:.0%}, "
        f"refusals {report['refusal_rate']:.0%}"
    )
  elif args.mode == "sensitivity":
    print("\n| prompt kind | n | routing accuracy |")
    print("|---|---|---|")
    for kind, stats in report["literal_vs_paraphrase"]["by_kind"].items():
      print(f"| {kind} | {stats['n']} | {stats['accuracy']:.0%} |")
    wording = report["description_wording"]
    print(
        f"\nreworded every tool description: "
        f"{wording['n_flipped']}/{wording['n_prompts']} verdicts flipped"
    )
  elif args.mode == "floor":
    print(f"\n{report['correct']}/{report['total']} — {report['note']}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())

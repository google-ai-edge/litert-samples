# FunctionGemma 270M mobile-actions

[litert-community/functiongemma-270m-ft-mobile-actions](https://huggingface.co/litert-community/functiongemma-270m-ft-mobile-actions)'s `mobile_actions_q8_ekv1024.litertlm` (289 MB), a `.litertlm` bundle for the [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) engine: Gemma 3 270M finetuned to route natural-language requests to phone actions. The bundle is published by the LiteRT team, not built by this recipe, so there is no `build_*.py` here — [`converted/`](converted/) holds the measurement harness that establishes what the bundle can and cannot route, which is the part that was not written down anywhere.

**Read the four-tool result before shipping this model.** The bundle is a keyword matcher, not a semantic router. On a fixed prompt set it routes **100% of prompts that carry a tool's own vocabulary and 17% of paraphrases that do not**, and its accuracy is highest at four tools and *falls* at eight. Both numbers were measured; neither is in the model's card.

| Prompt phrasing | n | Routing accuracy |
|---|---|---|
| Literal — uses the tool's own words ("what is on my calendar today") | 3 | 100% |
| Paraphrase — same intent, different words ("do i have meetings today") | 12 | 17% |
| Terse — keywords only ("calendar today") | 1 | 0% |

## Run

The bundle is a function-calling model: with no tools offered it declines every request with a fixed line ("I am FunctionGemma, a model optimized for function calls…"). It is useless without a tool set, and its tool set is the whole design problem.

```bash
pip install litert-lm
python - <<'EOF'
import litert_lm

def open_flashlight() -> str:
  """Turns the phone's flashlight on."""
  return "on"

def close_flashlight() -> str:
  """Turns the phone's flashlight off."""
  return "off"

def query_calendar() -> str:
  """Lists the events on the user's calendar for today."""
  return "standup at 10:00"

def take_photo() -> str:
  """Takes a photo with the phone's camera."""
  return "saved"

with litert_lm.Engine("mobile_actions_q8_ekv1024.litertlm",
                      backend=litert_lm.Backend.CPU()) as engine:
  with engine.create_conversation(tools=[open_flashlight, close_flashlight,
                                         query_calendar, take_photo],
                                  automatic_tool_calling=True) as convo:
    print(convo.send_message("turn on the flashlight")["content"][0]["text"])
EOF
```

```bash
hf download litert-community/functiongemma-270m-ft-mobile-actions \
    mobile_actions_q8_ekv1024.litertlm --local-dir .   # access-gated; needs a token
litert-lm run mobile_actions_q8_ekv1024.litertlm --prompt "turn on the flashlight"
```

The bare `litert-lm run` above answers with the refusal line, because a run with no tool set gives the model nothing to route to. That is expected.

## Which file

| File | Size | Use it for |
|---|---|---|
| `mobile_actions_q8_ekv1024.litertlm` | 289 MB | CPU (measured here). int8, 1024-token KV cache, a `.litertlm` from LiteRT-LM builder 1.5.0 |
| `functiongemma-270m-ft-mobile-actions_Google_Tensor_G5.litertlm` | 574 MB | Not measured. Named for a Tensor G5 phone; the same actions, AOT-compiled for that NPU |
| `functiongemma-270m-ft-mobile-actions_Google_Tensor_G6.litertlm` | 570 MB | Not measured. As above, for Tensor G6 |

All three are 270M int8; the two larger files are NPU-targeted builds of the same checkpoint and were not run. The 289 MB file is the only one measured in this recipe. On the access-gated repository, `hf download` needs an accepted license and a token.

## The tool set is the design

Same bundle, same prompts, greedy, one fresh conversation per prompt. Accuracy is scored only over prompts whose expected tool was actually offered, so a one-tool cell is not credited for the 18 prompts it had no way to answer.

```bash
python converted/verify_functiongemma_270m.py mobile_actions_q8_ekv1024.litertlm --mode scaling
```

| Tool set offered | n tools | Prompts scored | Tool-name accuracy | Refusal rate | Median |
|---|---|---|---|---|---|
| 1 tool | 1 | 8 | 12% | 96% | 0.56 s |
| 2 tools | 2 | 12 | 8% | 88% | 0.57 s |
| **4 tools** | 4 | 18 | **28%** | 81% | 0.66 s |
| 8 tools | 8 | 26 | 19% | 81% | 0.88 s |

**More tools make it worse.** Accuracy peaks at four and drops at eight while median latency rises 33%. The 8-tool set is the obvious design for a phone agent — flashlight, calendar, photo, alarm, message, note, no-op — and it is the worst-scoring configuration measured here. Two findings behind the drop:

- **Literal prompts are the only ones that route.** Every one of the three literal-phrasing probes resolves; the paraphrases almost never do. With eight tools the schema block is long enough that a paraphrase's tokens no longer overlap the tool it means.
- **Rewording the tool descriptions moves the answers.** Keeping the four names and replacing every description ("Turns the phone's flashlight on." → "Enables the LED torch.") flipped **3 of 16** verdicts, both directions: `"what's my schedule"` started resolving to `query_calendar`, and `"shut the light"` stopped resolving to `close_flashlight`. The model is sensitive to prompt wording the developer chose casually.

Practical consequence: match prompts to the tool's vocabulary rather than to natural user phrasing, and treat the tool descriptions as part of the prompt contract that has to be gated, not as documentation.

```bash
python converted/verify_functiongemma_270m.py mobile_actions_q8_ekv1024.litertlm --mode sensitivity
```

## What it routes

The four tools that resolve, and the argument cases. Note that none of the six argument-taking tools passed an argument at all — every one either refused or emitted an empty argument object.

| Prompt | Routs to | Arguments |
|---|---|---|
| "turn on the flashlight" | `open_flashlight` | — |
| "switch the flashlight on please" | `open_flashlight` | — |
| "turn off the flashlight" | `close_flashlight` | — |
| "shut the light" | `close_flashlight` | — |
| "what is on my calendar today" | `query_calendar` | — |
| "set an alarm for 07:30" | refusal (asks for the time) | — |
| "send a message to mom saying hi" | refusal (asks for the recipient) | — |
| "write a note titled groceries" | refusal (asks for the title) | — |
| "what is the weather in nairobi" | refusal (no such tool) | — |

The refusals are the model's own line ("I need the time for the alarm to set…"), not a runtime error. The bundle routes **intent to a zero-argument tool** and does not populate arguments; for an agent that must pass a time, a recipient or a title, this is a hard ceiling at 270M with this finetune.

## Multi-turn

Three turns, one conversation, automatic tool calling, four tools. The tool round trip works — the bundle's `<start_function_response>` prefix and its template's prefix contract hold across turns, and no state corruption appeared. Routing degrades on the middle turn: "what is on my calendar today" is answered *"The flashlight is now on."* — a fluent, confident wrong answer rather than a refusal, which is worse to ship than a refusal.

```bash
python converted/verify_functiongemma_270m.py mobile_actions_q8_ekv1024.litertlm --mode multi-turn
```

```
'turn on the flashlight'       -> 'The flashlight has been turned on.'
'what is on my calendar today' -> 'The flashlight is now on.'      # wrong, fluent
'now turn it off'              -> 'The flashlight has been turned off.'
```

**Gate multi-turn, not single-turn.** A single-turn evaluation of this bundle looks far better than it behaves: the second prompt above is answered confidently and incorrectly, and single-turn scoring cannot see that.

## The floor gate does not apply

The cookbook's eight-question floor gate scores **0/8** on this bundle: every question is answered with the tool-refusal line. That is the finetune working as intended, not a defect — a function-calling model has no reason to answer arithmetic. `--mode floor` runs the gate anyway so the number is on record and nobody re-derives it.

```bash
python converted/verify_functiongemma_270m.py mobile_actions_q8_ekv1024.litertlm --mode floor
```

Do not read 0/8 as a quality verdict, and do not calibrate an unquantized baseline against it either: the refusal line contains none of the expected answers, so every comparison against it is uninformative.

## Tested on

```bash
litert-lm benchmark mobile_actions_q8_ekv1024.litertlm --backend cpu -p 256 -d 256 --runs 3 --cache no
litert-lm benchmark mobile_actions_q8_ekv1024.litertlm --backend gpu -p 256 -d 256 --runs 3 --cache no
```

Decode and prefill in tokens per second, 256-token prompt and reply, three runs, no compile cache. Host: Intel Core i7-1185G7 (4 cores / 8 threads, 3.0 GHz), 15 GB RAM, Linux 6.18, litert-lm 0.16.1.

| Backend | Prefill tok/s | Decode tok/s | Init | TTFT |
|---|---|---|---|---|
| CPU (XNNPACK) | 941 | 47.2 | 2.02 s | 0.293 s |
| GPU (WebGPU/Vulkan) | 1529 | 28.4 | 6.14 s | 0.203 s |

**The GPU row does not produce a usable tool call.** With `Backend.GPU()` the same prompt that returns `open_flashlight` on the CPU returns a refusal, and a no-tools prompt returns 2,032 repetitions of the literal string `<pad>`. The bundle's own metadata declares a `function_gemma` model type whose GPU path is not exercised here; the speed numbers above are real, and the routing is not. Gate a backend on generated text, not on benchmark output — this backend prints healthy prefill and decode numbers for a model it cannot actually run.

Not measured on a phone. The two Tensor-targeted files in the same repository were not run, and the NPU dispatch library is absent from this host, so no NPU number is claimed.

## Conversion

Not in this recipe: the bundle is published by the LiteRT team, and the source checkpoint ([google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it), `gemma` license, manually gated) is not something this repository rebuilds. The harness in [`converted/`](converted/) and the bundle's own layout are documented here so the published file can be evaluated rather than assumed.

Measured on the published file: LiteRT-LM container version 1.5.0, authors "Google AI Edge", three sections — `LlmMetadata` (15 KB), `TF_LITE_PREFILL_DECODE` model (284,216,288 B), SentencePiece tokenizer (4,689,144 B), 1.05 bytes per parameter at 270M. Metadata declares `llm_model_type: function_gemma`, `start_token` id 2, and stop tokens `<end_of_turn>` and `<start_function_response>`. The embedded Jinja template renders tool declarations as `<start_function_declaration>` blocks and tool calls as `<start_function_call>call:{name}{k:v}` — the format the harness in `converted/` emits through `SchemaTool`, which is why the tool descriptions and argument names are part of the prompt contract.

## References

- [litert-community/functiongemma-270m-ft-mobile-actions](https://huggingface.co/litert-community/functiongemma-270m-ft-mobile-actions), the bundles (access-gated).
- [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it), the source checkpoint.
- [FunctionGemma](https://ai.google.dev/edge/functiongemma), the model family.
- LiteRT-LM guides: [CLI](https://ai.google.dev/edge/litert-lm/cli), [Python](https://ai.google.dev/edge/litert-lm/python), [Kotlin](https://ai.google.dev/edge/litert-lm/android), [Swift](https://ai.google.dev/edge/litert-lm/swift).

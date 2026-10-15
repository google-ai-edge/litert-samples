# FunctionGemma 270M mobile-actions: measurement harness

Measurement scripts for
[`mobile_actions_q8_ekv1024.litertlm`](https://huggingface.co/litert-community/functiongemma-270m-ft-mobile-actions),
the 289 MB int8 `.litertlm` bundle published by the LiteRT team. There is no
build script: the bundle is published, and the source checkpoint
([google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it))
is manually gated. What this directory adds is the measurement the model card
does not carry — what this bundle actually routes, and what happens to its
accuracy as the tool set grows.

Every number on the [recipe page](../README.md) came from one of the modes
below, run on the published file with `litert-lm` 0.16.1 on Linux x86-64.

## Environment

```bash
pip install litert-lm            # 0.16.1 used here
hf download litert-community/functiongemma-270m-ft-mobile-actions \
    mobile_actions_q8_ekv1024.litertlm --local-dir .    # access-gated
```

## Run

```bash
V=verify_functiongemma_270m.py
M=mobile_actions_q8_ekv1024.litertlm

python $V $M --mode gate                      # routing accuracy, 8 tools
python $V $M --mode scaling                   # the same at 1, 2, 4 and 8 tools
python $V $M --mode sensitivity               # literal vs paraphrase, and description rewording
python $V $M --mode multi-turn                # three turns, automatic tool calling
python $V $M --mode floor                     # the cookbook's 8 questions, expected 0/8
python $V $M --mode scaling --out scaling.json
```

`--backend gpu` runs the same modes on the GPU backend. On this host that
backend returns refusals and `<pad>` repetition for tool prompts, so the
numbers are a backend check, not a routing measurement — see the recipe page.

## Modes

| Mode | What it isolates |
|---|---|
| `gate` | Tool-name and argument accuracy over 26 prompts, one fresh conversation each, greedy. The publish guardrail for an agent. |
| `scaling` | The same prompts at four tool-set sizes. Accuracy is scored only over prompts whose expected tool was offered, so a narrow cell is not credited for prompts it could not answer. |
| `sensitivity` | Two variables: whether the model routes on meaning or on keywords (literal vs paraphrase prompts), and whether it reads the tool descriptions at all (same names, reworded descriptions). |
| `multi-turn` | Three turns through one conversation with automatic tool calling — the template prefix contract and the tool-response round trip, which single-turn scoring cannot reach. |
| `floor` | The cookbook's eight-question floor gate, run so its 0/8 is on record. A function-calling finetune declines general questions; this is not a quality verdict. |

## Why `SchemaTool` and not a Python function

`litert_lm.tools.tool_from_function` builds a tool schema by calling
`inspect.signature` on a Python function and parsing its docstring for
per-argument descriptions. That is fine for a real integration, and wrong for
this measurement, which needs to vary the description text and the argument
names independently of any Python signature. `SchemaTool` implements
`litert_lm.interfaces.Tool` directly and takes a hand-written OpenAPI schema, so
the description is a variable under test.

## Measurement rules

- **One fresh conversation per prompt.** The bundle's Jinja template enforces a
  prefix contract (`new rendered template string does not start with the
  previous`), so a shared conversation fails on history, not on the prompt.
- **Greedy, `max_output_tokens=128`.** These are routing probes. A longer budget
  lets a refusal finish its sentence, which does not change the verdict.
- **`automatic_tool_calling=False`** for every single-turn mode. With automatic
  calling the model emits a call, the tool runs, and the reply is prose about
  the result — which throws away the thing being measured. The call comes back
  in `tool_calls`; a refusal comes back as prose in `content`, and both are read.
- **`cache_dir` is set explicitly.** The default disk cache writes delegate
  caches of up to twice the model size next to the `.litertlm` — about 600 MB
  into the caller's model directory for this bundle.
- **Prompts are paired.** Each intent appears once with the tool's own
  vocabulary and once without it, so a keyword matcher cannot score well by
  accident. This is the whole reason the headline number is low.

## Files

| File | What |
|---|---|
| `verify_functiongemma_270m.py` | The five modes above, plus the prompt set, the eight tool schemas, and the literal/paraphrase pairing. |

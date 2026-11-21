---
name: add-litert-omni-model
description: >-
  Guides end-to-end bring-up of a new LiteRT Omni multimodal model (Any-to-Any,
  ASR, TTS, or Text-to-Image) under models/ in google-ai-edge/litert-samples
  without modifying upstream LiteRT-LM or litert-torch packages. Use when adding
  a new Omni C++ runtime pipeline, custom protobuf metadata schema, standalone
  runner binary, out-of-tree Hugging Face PyTorch exporter extension, or
  hermetic cross-language tests under models/. Don't use for LiteRT Tensor API
  LLM bring-up (see google-ai-edge/LiteRT tensor/README.md), upstreaming
  stabilized models into LiteRT-LM, or general model inference debugging.
---

# Add a New LiteRT Omni Model

Bring up a new multimodal model under `//models/{model_family}/{model_name}/`
(for example, `lfm/lfm2_5_audio/`, `deepseek/janus_pro/`, `omnigen/omnigen2/`,
or `tongyi/z_image_turbo/`) without modifying upstream
`google-ai-edge/LiteRT-LM` (`@litert_lm`) or `google-ai-edge/litert-torch`
(`@litert_torch`). Use the
[LFM2.5-Audio implementation](https://github.com/google-ai-edge/litert-samples/tree/main/models/lfm/lfm2_5_audio)
as the canonical end-to-end template.

## 1. Scope & Zero-Touch Mainline Invariants

-   **Keep all new files under `//models/{model_family}/{model_name}/`.**
    Register C++ runtimes, custom protos, and Python exporters out-of-tree
    rather than adding per-model code, enum values, or visibility exceptions to
    `google-ai-edge/LiteRT-LM` (`@litert_lm`) or `google-ai-edge/litert-torch`
    (`@litert_torch`).
-   **Keep package visibility scoped cleanly.** Set
    `package(default_visibility = ["//models:__subpackages__"])` in every
    `BUILD` file so cross-model targets stay encapsulated while tests can depend
    on their sibling package targets.
-   **Split delivery into two stacked pull requests** and keep edits strictly
    inside `models/{model_family}/{model_name}/` (no upstream `LiteRT-LM` or
    `litert-torch` changes in the same PR):
    1.  **PR 1 (C++ runtime + proto + runner)**:
        `//models/{model_family}/{model_name}/` containing `README.md`, the
        `.proto` (with C++ and Python proto rules), C++ pipeline stages,
        `OmniRegistry` registration, `omni_runner_binary`, per-stage unit tests,
        and integration + runner tests. Include runnable `bazel run` command(s)
        in the PR description to invoke
        `//models/{model_family}/{model_name}:{model_name}_runner` with
        representative flags and modality inputs.
    2.  **PR 2 (Python exporter + E2E)**:
        `//models/{model_family}/{model_name}/export/` containing the
        `ModelExtension`, `export_main` CLI, exporter unit tests, and
        cross-language `export_and_run_test` (plus export instructions in
        `{model_family}/{model_name}/README.md`). Include runnable `bazel run`
        command(s) in the PR description for both
        `//models/{model_family}/{model_name}/export:export_main` and
        `:{model_name}_runner` on the exported `.litertlm` artifact.

## 2. Directory Layout & Modality Mapping

Mirror this layout under `models/{model_family}/{model_name}/`:

```text
{model_family}/{model_name}/
├── BUILD                       # proto, per-stage cc_library, alwayslink reg, runner, tests
├── README.md                   # overview, bundle sections, export & run commands
├── {model_name}.proto          # model-intrinsic config packed into custom_model = 100
├── {model_name}_model_config.h # inline constexpr constants and C++ config struct
├── io_types.h                  # inter-stage I/O structs (:io_types)
├── stage_utils.{h,cc}          # optional shared tensor/validation helpers
├── {stage_name}_stage.{h,cc}   # one header/source/cc_library per pipeline stage
├── {stage_name}_stage_test.cc  # dedicated cc_test unit tests per pipeline stage
├── {model_name}_factory.{h,cc} # ModelResources init + stage assembly
├── {model_name}.cc             # metadata unpack, session lease, OmniRegistry spec
├── {model_name}_test_utils.{h,cc} # testonly synthetic .litertlm / WAV / image builders
├── {model_name}_test.cc        # factory + OmniEngine/OmniRegistry integration tests
├── {model_name}_runner_test.cc # RunOmniRunner + portable posix_spawn binary tests
└── export/
    ├── BUILD                   # all targets tagged ["nomacos"]
    ├── modules.py              # PyTorch exportable submodules
    ├── {model_name}.py         # submodel export, verify_checkpoint_keys, metadata packing
    ├── extension.py            # @register_model_extension() ModelExtension subclass
    ├── export_main.py          # thin CLI importing extension as _ and invoking fire.Fire
    ├── test_{model_name}.py    # config, keys, submodel export parity, litertlm_peek tests
    └── export_and_run_test.py  # subprocess :export_main -> :{model_name}_runner E2E test
```

Apply these structural conventions to avoid review rework:

-   **Include a self-contained `README.md` inside
    `{model_family}/{model_name}/`.** Document supported modalities,
    architecture and `.litertlm` bundle sections, directory layout, verified
    `export:export_main` and `:{model_name}_runner` commands using `bazel run`
    with `//models/{model_family}/{model_name}` targets (including explicit
    flags matching the actual source code and filename auto-detection aliases),
    and hermetic `bazel test` invocations. Use only public URLs and relative
    links so the README renders cleanly on GitHub.
-   **Isolate inter-stage structs in `io_types.h` (`:io_types`)** so stage
    headers do not cross-include each other and downstream changes minimize
    rebuilds.
-   **Give every pipeline stage its own `{stage_name}_stage.{h,cc}`,
    `cc_library`, and `{stage_name}_stage_test.cc` `cc_test` target.** Bundling
    multiple stages or their tests into one file entangles dependencies and
    weakens per-stage test isolation.
-   **Use `inline constexpr` for namespace-scope constants in headers** (such as
    `{model_name}_model_config.h`) so C++17+ gives them external linkage instead
    of per-translation-unit internal duplicates, avoiding ODR address
    mismatches. Keep plain `static constexpr` for class/struct members and plain
    `constexpr` for functions (both already implicitly `inline`) and inside
    `.cc` files.
-   **Include standard Apache 2.0 license headers** on every source, proto, and
    `BUILD` file:
    -   Runtime files (`*.cc`, `*.h`, `*.proto`, and
        `{model_family}/{model_name}/BUILD`): `//` (or `#` for `BUILD`) Apache
        2.0 header carrying `Copyright {year} The ODML Authors.` followed by one
        blank line.
    -   Exporter files (`export/*.py` and `export/BUILD`): `#` Apache 2.0 header
        carrying `Copyright {year} The LiteRT Torch Authors.` ending with the
        80-column `# ===...` divider (no blank line before the docstring in
        `*.py`; one blank line before `load(...)` in `BUILD`).

Match your modality across C++ and Python contracts:

<!-- mdformat off(prevent table wrapping) -->

| Modality | `session_config_type_id` | Metadata Proto & Field | ModelType Proto (`custom_model = 100`) | `ExportTask` | `ModelExtension` Hook |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Any-to-Any | `"Any2Any"` | `Any2AnyMetadata.any2any_model_type` | `Any2AnyModelType` | `ExportTask.ANY_TO_ANY` | `get_any2any_model_cls` |
| ASR | `"Speech2Text"` | `Speech2TextMetadata.asr_model_type` | `Speech2TextModelType` | `ExportTask.AUTOMATIC_SPEECH_RECOGNITION` | `get_speech_model_cls` |
| TTS | `"Text2Speech"` | `Text2SpeechMetadata.tts_model_type` | `Text2SpeechModelType` | `ExportTask.TEXT_TO_SPEECH` | `get_tts_model_cls` |
| Image Gen | `"Text2Image"` | `Text2ImageMetadata.image_gen_model_type` | `ImageGenModelType` | `ExportTask.TEXT_TO_IMAGE` | `get_image_gen_model_cls` |

<!-- mdformat on -->

## 3. C++ Runtime Integration

### Custom Proto & `google.protobuf.Any` Packing

Define model-intrinsic hyperparameters in `{model_name}.proto` under
`package prototypes.litert_models.{model_name};` (matching the
`google.protobuf.Any` `type_url` packed in existing `.litertlm` bundles) and
declare `proto_library`, `cc_proto_library`, and `py_proto_library` targets in
`BUILD`:

```protobuf
syntax = "proto3";

package prototypes.litert_models.{model_name};

option java_multiple_files = true;

message {proto_message} {
  optional int32 hidden_size = 1;
  optional int32 max_new_tokens = 2;
}
```

Pack and unpack this message via field `100`
(`google.protobuf.Any custom_model`) on the modality's `*ModelType` proto.
`OmniRegistry::CreateSessionFactory` already merges `.litertlm`-embedded
metadata with caller overrides via `MergeMetadata` before invoking
`get_session_factory`. Copy `metadata` into a fresh modality proto via
`CheckTypeAndMergeFrom` and do not re-read embedded metadata from
`lm_resources`—re-merging duplicates repeated fields such as
`supported_languages`:

```cpp
absl::StatusOr<lm::proto::Any2AnyMetadata> ResolveMetadata(
    const MessageLite* absl_nullable metadata) {
  lm::proto::Any2AnyMetadata resolved;
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  if (resolved.any2any_model_type().model_type_case() ==
      lm::proto::Any2AnyModelType::MODEL_TYPE_NOT_SET) {
    prototypes::litert_models::{model_name}::{proto_message} default_proto;
    if (!resolved.mutable_any2any_model_type()
             ->mutable_custom_model()
             ->PackFrom(default_proto)) {
      return absl::InternalError("Failed to pack default custom_model proto.");
    }
  }
  return resolved;
}
```

Note that `lfm2_audio.cc`'s `ResolveMetadata` still carries the legacy
`lm_resources` re-read with defensive clears; follow the single-merge pattern
above instead (matching upstream `LiteRT-LM` factories such as `kokoro.cc` and
`flux2_klein.cc`). Unpack with `UnpackTo(&proto)` and return
`absl::InvalidArgumentError` if `custom_model` is missing (when set to another
`oneof` arm) or holds a different type URL.

### `OmniRegistry` Registration & `omni_runner_binary`

Register the model factory in `{model_name}.cc` via the
[OmniRegistry interface](https://github.com/google-ai-edge/LiteRT-LM/blob/main/omni/omni_registry.h):

```cpp
constexpr absl::string_view kModelNames[] = {
    "{canonical_name}",
    "{model_name}",
};

[[maybe_unused]] const bool kModelRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetModelSessionFactory,
      .session_config_type_id = "Any2Any",  // Match modality table above.
  }));
  return true;
}();
```

-   List the canonical name first in `model_names`, followed by aliases.
    `OmniRegistry::FindModelNameForPath` matches the longest normalized alias
    (lowercase, ignoring `-` and `_`) inside the `.litertlm` filename to infer
    `--model_name` automatically.
-   Set `alwayslink = 1` on the registration `cc_library` in `BUILD` so the
    linker keeps the static registration translation unit. Include
    `":{model_name}"` in `deps` of `cc_test` targets that rely on link-time
    registration without including a header directly.
-   Instantiate a standalone CLI binary using `omni_runner_binary` from
    [the Omni runner Bazel macro](https://github.com/google-ai-edge/LiteRT-LM/blob/main/omni/omni_runner.bzl):

```starlark
load("@litert_lm//omni:omni_runner.bzl", "omni_runner_binary")

omni_runner_binary(
    name = "{model_name}_runner",
    models = [":{model_name}"],
)
```

### Session Leasing & `ModelResources` Lifetime

-   **Guard shared stateful runners with a single-active-session atomic lease.**
    `ModelResources` shares compiled runners (such as a stateful
    `LiteRtLmRunner` KV cache) across sessions from the same
    `OmniSessionFactory`. Acquire a `std::shared_ptr<std::atomic<bool>>` via
    `compare_exchange_strong` in `CreateWithInputSource` (returning
    `absl::FailedPreconditionError` if already active), protect setup error
    paths with `absl::Cleanup`, transfer ownership into the `MultiStagedSession`
    subclass, and call `Reset()` in the session destructor *before* releasing
    the lease so async worker threads stop first.
-   **Pass `std::shared_ptr<ModelResources>` directly without copying by
    value.** In the
    [ModelResources container](https://github.com/google-ai-edge/LiteRT-LM/blob/main/omni/base/model_resources.h),
    member declaration order guarantees `lm_runners_` and `models_`
    (`CompiledModel`) are destroyed before `lm_resources_` (mmap'd `.litertlm`
    buffers) and `env_` (`litert::Environment`). Shallow copies or detached
    `CompiledModel` handles break that teardown order and trigger use-after-free
    crashes under macOS `opt` builds.

## 4. Python Exporter Extension (`export_hf`)

### Subclassing `ModelExtension`

In `export/extension.py`, subclass `extension_lib.ModelExtension`
([export_hf extension contract](https://github.com/google-ai-edge/litert-torch/blob/main/litert_torch/generative/export_hf/model_ext/extension.py))
and decorate with `@extension_lib.register_model_extension()`:

```python
@extension_lib.register_model_extension()
class {extension_cls}(extension_lib.ModelExtension):
  model_types = ('{model_name}',)
  extra_kwargs_keys = {model_name}_lib.EXTRA_KWARGS

  @classmethod
  def match_checkpoint(
      cls,
      model_path: str,
      task: Any = None,
      config_dict: Mapping[str, Any] | None = None,
  ) -> bool:
    ...

  def get_export_defaults(
      self,
      config_dict: Mapping[str, Any],
      task: str | None = None,
      target: str = 'cpu_gpu',
  ) -> extension_lib.ExportDefaults:
    del config_dict, task, target
    return extension_lib.ExportDefaults(task=extension_lib.ExportTask.ANY_TO_ANY)

  def get_any2any_model_cls(self, model_type: str) -> type[{model_cls}]:
    del model_type
    return {model_cls}
```

-   Also register `extra_kwargs_keys` via
    `exportable_module_config.register_extra_kwargs_keys(...)` in
    `export/{model_name}.py` if callers construct `ExportableModuleConfig`
    directly in unit tests before importing `extension.py`.
-   Implement `match_checkpoint` as a `@classmethod` whenever upstream
    checkpoints omit a top-level Hugging Face `model_type` (checking
    `architectures`, config keys, and `model_path` fallback).

### Thin CLI Entrypoint, Proto Packing & Checkpoint Verification

-   **`export/export_main.py`**: Import `extension as _` (with
    `# pylint: disable=unused-import`) to trigger registration on import and
    call `fire.Fire(lib.export)` from `main()` (with
    `if __name__ == '__main__': main()`). Keep `":extension"` in
    `export_main_lib`'s `deps` so the side-effect import remains linked.
-   **Pack the custom proto**: Import `{model_name}_pb2` from
    `//models/{model_family}/{model_name}:{model_name}_py_proto` and pack it
    into the modality metadata message:

```python
metadata = any2any_metadata_pb2.Any2AnyMetadata(...)
metadata.any2any_model_type.custom_model.Pack(model_proto)
```

-   **Verify checkpoint keys across layouts**: Add `verify_checkpoint_keys()` to
    reject unconsumed safetensors tensors when `use_random_weights=False`.
    Support both single-file checkpoints (submodel tensors prefixed in
    `model_dir`) and split-directory checkpoints (submodel weights in a
    subdirectory), and test both layouts.
-   **Tag every `export/BUILD` target with `tags = ["nomacos"]`.** PyTorch and
    `litert_torch` pull Linux-only CUDA dependencies when built under Bazel, so
    `nomacos` keeps macOS C++ runtime builds clean.

## 5. Required Hermetic Test Suite

Because multi-GB Hugging Face checkpoints cannot be checked into Git, build all
five hermetic, non-mocked test layers using tiny deterministic synthetic models
so real LiteRT compilation, `.litertlm` serialization, and CLI execution run in
CI on every change:

1.  **`{stage_name}_stage_test.cc` (per-stage C++ unit tests)**: Add a dedicated
    unit test file and `cc_test` target for every pipeline stage covering shape
    and tensor contracts, preprocessing math, control token escaping,
    scheduling, error paths, and round-trip inference on real LiteRT compiled
    models. (Note: `lfm/lfm2_5_audio` currently groups stage tests inside
    `lfm2_audio_test.cc`; new models should give each stage its own
    `{stage_name}_stage_test.cc`.)
2.  **`{model_name}_test.cc` (factory & registry integration tests)**:
    `OmniRegistry` and `OmniEngine` integration tests covering `custom_model`
    pack/unpack, metadata override handling (including no repeated-field
    duplication), wrong-proto rejection, and concurrent-session lease errors.
3.  **`{model_name}_runner_test.cc` (runner library & binary tests)**:
    In-process `RunOmniRunner()` tests
    ([the Omni runner library API](https://github.com/google-ai-edge/LiteRT-LM/blob/main/omni/omni_runner_lib.h))
    for each modality path plus portable `posix_spawn` execution of the compiled
    `:{model_name}_runner` binary (explicit `--model_name` and filename-inferred
    auto-detection). Guard `environ` with `#ifdef __APPLE__`
    (`*_NSGetEnviron()`) so the test passes on both Linux and macOS.
4.  **`export/test_{model_name}.py` (exporter unit & parity tests)**: Extension
    auto-discovery, checkpoint key verification (single-file and
    split-directory), `--dry_run`, per-submodel LiteRT `CompiledModel` numerical
    parity against PyTorch, and `.litertlm` metadata inspection via
    `litertlm_peek`.
5.  **`export/export_and_run_test.py` (cross-language subprocess E2E test)**:
    Hermetic E2E test (`size = "large"`, `data = [":export_main",
    "//models/{model_family}/{model_name}:{model_name}_runner"]`) that generates
    a tiny deterministic safetensors checkpoint, runs `:export_main` to emit
    `model.litertlm`, executes C++ `:{model_name}_runner` on that file, and
    asserts valid non-empty outputs (text transcript, valid WAV with non-zero
    PCM, or valid PNG).

## 6. Static Analysis, Formatting & Verification

### Code Hygiene Guidelines

-   **C++**:
    -   Prefix Abseil macros with `ABSL_` (`ABSL_CHECK_OK`,
        `ABSL_ASSIGN_OR_RETURN`, `ABSL_RETURN_IF_ERROR`).
    -   Use `absl::c_copy` / `absl::c_fill` instead of `std::copy` /
        `std::fill`.
    -   Use `absl::string_view::substr` instead of pointer-and-length slicing.
    -   Maintain direct include-what-you-use (`IWYU`) headers (such as
        `<cstring>`, `<iterator>`, `<optional>`, `<utility>`).
-   **Python**:
    -   Prefer `@classmethod` or module-level functions over `@staticmethod`.
    -   Annotate all parameters and return types on public interfaces.

### Local Verification Loop

Run from the repository root before opening a pull request:

```bash
# 1. BUILD formatting
buildifier -mode=check models/{model_family}/{model_name}/BUILD models/{model_family}/{model_name}/export/BUILD

# 2. Linux C++ runtime tests (C++20 required for @litert_lm)
bazel test //models/{model_family}/{model_name}:all

# 3. Python exporter scripts via pip (`litert-samples`'s default `@pypi` lockfile
#    only includes `numpy`, so install `litert-torch`, `litert-lm-builder`, and
#    `ai-edge-litert` into a Python venv unless a separate pip_parse repo is wired)
#    pip install litert-torch litert-lm-builder ai-edge-litert

# 4. macOS C++ runtime build and tests (excluding nomacos exporter targets)
bazel test --build_tag_filters=-nomacos --test_tag_filters=-nomacos //models/{model_family}/{model_name}:all

# 5. ASan check on C++ stage, integration, and runner tests
#    (litert-samples/.bazelrc does not define a `build:asan` config)
bazel test \
  --copt=-fsanitize=address --copt=-fno-omit-frame-pointer \
  --linkopt=-fsanitize=address \
  //models/{model_family}/{model_name}:all
```

### Graduation to Upstream Repositories

Once validated on real Hugging Face checkpoints and reviewed, stabilized C++
model runtimes can graduate to
[google-ai-edge/LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM) and
exporter extensions to
[google-ai-edge/litert-torch](https://github.com/google-ai-edge/litert-torch).

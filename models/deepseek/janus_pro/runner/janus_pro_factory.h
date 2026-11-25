// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MODELS_DEEPSEEK_JANUS_PRO_RUNNER_JANUS_PRO_FACTORY_H_
#define MODELS_DEEPSEEK_JANUS_PRO_RUNNER_JANUS_PRO_FACTORY_H_

#include <memory>
#include <vector>

#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_environment.h"
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::any2any {

// Creates shared `ModelResources` for DeepSeek Janus-Pro, compiling
// `text_embedder`, optional `vision_encoder`, `llm_backbone`, and the visual
// generation sections (`vq_image_embedder`, `gen_head`, `vq_image_decoder`, and
// `uncond_llm_backbone`, required when `config.generate_image` is true or
// compiled whenever present in the `.litertlm` bundle) in a new
// `ModelResources` retaining `env`. If `lm_resources` is non-null, it is reused
// directly without reopening `model_path`.
absl::StatusOr<ModelResources> CreateJanusProResources(
    const JanusProModelConfig& config, absl::string_view model_path,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources = nullptr);

// Creates the Janus-Pro stage pipeline (`JanusProInputFanOutStage`,
// `JanusProTextStage`, optional `JanusProImageStage`, `JanusProFrontendStage`,
// `JanusProLlmStage`, and `JanusProImageGenStage`) and appends all stages in
// upstream-to-downstream order to `stages`, returning the terminal output
// stage.
absl::StatusOr<Stage<Output>* absl_nonnull> CreateJanusProComponents(
    const JanusProModelConfig& config,
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages);

}  // namespace litert::omni::any2any

#endif  // MODELS_DEEPSEEK_JANUS_PRO_RUNNER_JANUS_PRO_FACTORY_H_

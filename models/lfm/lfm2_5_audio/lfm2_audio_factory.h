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

#ifndef MODELS_LFM_LFM2_5_AUDIO_LFM2_AUDIO_FACTORY_H_
#define MODELS_LFM_LFM2_5_AUDIO_LFM2_AUDIO_FACTORY_H_

#include <memory>
#include <vector>

#include "models/lfm/lfm2_5_audio/lfm2_audio_model_config.h"
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

// Creates shared `ModelResources` for LFM2.5-Audio-1.5B, compiling the
// sub-models (`text_embedder`, optional `audio_encoder`, `llm_backbone`, and,
// when `config.generate_audio` is true, required `tts_projector`,
// `tts_decoder`, and `audio_vocoder`) from a `.litertlm` bundle in a new
// `ModelResources` owning `env`. If `lm_resources` is non-null, it is reused
// directly and `model_path` is not reopened.
absl::StatusOr<ModelResources> CreateLfm2AudioResources(
    const Lfm2AudioModelConfig& config, absl::string_view model_path,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources = nullptr);

// Instantiates all stage components for an LFM2.5-Audio-1.5B Any2Any inference
// session and appends them in upstream-to-downstream order to `stages`.
//
// Args:
//   config: Session-level model configuration for LFM2.5-Audio-1.5B.
//   input_source: Non-null source stage providing `OmniSession::Input` items;
//     ownership is transferred into `stages[0]`.
//   resources: Non-null initialized `ModelResources` containing the compiled
//     LFM2.5-Audio sub-models, tokenizer, and LLM backbone runner.
//   stages: Output vector that takes ownership of all created pipeline stages.
//
// Returns:
//   A non-owning pointer to the terminal `Stage<Output>` owned by `stages`
//   (`stages.back().get()`), or an error status if stage creation fails.
absl::StatusOr<Stage<Output>* absl_nonnull> CreateLfm2AudioComponents(
    const Lfm2AudioModelConfig& config,
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages);

}  // namespace litert::omni::any2any

#endif  // MODELS_LFM_LFM2_5_AUDIO_LFM2_AUDIO_FACTORY_H_

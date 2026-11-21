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

#ifndef MODELS_OMNIGEN_OMNIGEN2_OMNIGEN2_FACTORY_H_
#define MODELS_OMNIGEN_OMNIGEN2_OMNIGEN2_FACTORY_H_

#include <memory>
#include <vector>

#include "models/omnigen/omnigen2/omnigen2_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_environment.h"
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/text2image/prompt_source.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::text2image::omnigen2 {

// Creates shared `ModelResources` for `OmniGen2/OmniGen2`, compiling all 13
// LiteRT sub-models (`text_encoder`, `cap_embedder`, `context_refiner`,
// `x_embedder`, `noise_refiner`, `main_blocks[0..5]`, `final`, and
// `vae_decoder`) from a `.litertlm` container into a new `ModelResources`
// retaining `env`.
//
// If `lm_resources` is non-null, it is reused directly and `model_path` is not
// reopened. Standard capability sections (`text_encoder`, `main_blocks[0..3]`,
// `final`, `vae_decoder`) are loaded via `GetTFLiteModelBuffer`, and
// OmniGen2-specific extra shards (`cap_embedder`, `context_refiner`,
// `x_embedder`, `noise_refiner`, `main_blocks[4..5]`) are loaded zero-copy via
// `GetGenericBinaryDataBuffer`.
absl::StatusOr<ModelResources> CreateOmniGen2Resources(
    const OmniGen2ModelConfig& config, absl::string_view model_path,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources = nullptr);

// Instantiates all stage components for an `OmniGen2` image generation
// inference session and appends them in upstream-to-downstream order to
// `stages`.
//
// Returns a non-owning pointer to the terminal `Stage<Output>` owned by
// `stages` (`stages.back().get()`), or an error status if stage creation fails.
absl::StatusOr<Stage<Output>* absl_nonnull> CreateOmniGen2Components(
    const OmniGen2ModelConfig& config,
    std::unique_ptr<PromptSource> absl_nonnull prompt_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages);

}  // namespace litert::omni::text2image::omnigen2

#endif  // MODELS_OMNIGEN_OMNIGEN2_OMNIGEN2_FACTORY_H_

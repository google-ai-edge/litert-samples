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

#ifndef MODELS_TONGYI_Z_IMAGE_TURBO_RUNNER_Z_IMAGE_TURBO_FACTORY_H_
#define MODELS_TONGYI_Z_IMAGE_TURBO_RUNNER_Z_IMAGE_TURBO_FACTORY_H_

#include <memory>
#include <vector>

#include "models/tongyi/z_image_turbo/runner/z_image_turbo_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_environment.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/text2image/io_types.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/vae_decoder_stage.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"

namespace litert::omni::text2image {

// Creates shared `ModelResources` for Z-Image-Turbo, compiling all submodels
// (`z_text_encoder`, `z_embc`, `z_refc`, `z_embx`, `z_refx`, `zc_main0`..`5`,
// `zc_final`, `zvae`) from a `.litertlm` container or loose directory into a
// new `ModelResources` owning `env` and `lm_resources`.
//
// Standard submodels (`text_encoder`, `dit_single_block_0..3`, `dit_final`,
// `vae_decoder`) are loaded from `TFLiteModel` sections; model-specific shards
// (`dit_cap_embedder`, `dit_context_refiner`, `dit_x_embedder`,
// `dit_noise_refiner`, `dit_single_block_4`, `dit_single_block_5`) are loaded
// zero-copy from named `GenericBinaryData` sections.
absl::StatusOr<ModelResources> CreateZImageTurboResources(
    const ZImageTurboModelConfig& config, absl::string_view model_folder,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources = nullptr);

// Creates the shared `VaeDecoderStage` for Z-Image-Turbo (`AutoencoderKL`),
// wired with a callback that unpatchifies the denoised S3-DiT tokens
// `[1, (H/2)*(W/2), packed_ch]` into NCHW `[1, in_channels, H/8, W/8]` and
// applies inverse latent scaling and shift
// (`z = z * (1 / vae_scaling_factor) + vae_shift_factor`). Returns
// `InvalidArgumentError` if `denoiser`, `runner`, `config`, or the runner's
// tensor buffers are invalid.
absl::StatusOr<std::unique_ptr<VaeDecoderStage>> CreateZImageVaeDecoderStage(
    Stage<DenoisedLatents>* absl_nonnull denoiser,
    const ZImageTurboModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner);

// Instantiates all stage components for a Z-Image-Turbo image generation
// session (`PromptSource` -> `TextEncoderStage` -> `ZImageDenoiserStage` ->
// `VaeDecoderStage`) and appends them in upstream-to-downstream order to
// `stages`.
//
// Returns a non-owning pointer to the terminal `Stage<Output>` owned by
// `stages` (`stages.back().get()`), or an error status on failure.
absl::StatusOr<Stage<Output>* absl_nonnull> CreateZImageTurboComponents(
    const ZImageTurboModelConfig& config, absl::string_view model_folder,
    std::unique_ptr<PromptSource> absl_nonnull prompt_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages);

}  // namespace litert::omni::text2image

#endif  // MODELS_TONGYI_Z_IMAGE_TURBO_RUNNER_Z_IMAGE_TURBO_FACTORY_H_

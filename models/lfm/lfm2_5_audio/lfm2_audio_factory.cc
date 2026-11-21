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

#include "models/lfm/lfm2_5_audio/lfm2_audio_factory.h"

#include <memory>
#include <utility>
#include <vector>

#include "models/lfm/lfm2_5_audio/audio_stage.h"
#include "models/lfm/lfm2_5_audio/frontend_stage.h"
#include "models/lfm/lfm2_5_audio/generation_stage.h"
#include "models/lfm/lfm2_5_audio/input_fan_out_stage.h"
#include "models/lfm/lfm2_5_audio/lfm2_audio_model_config.h"
#include "models/lfm/lfm2_5_audio/text_stage.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "omni/any2any/any2any_resources.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {

absl::StatusOr<ModelResources> CreateLfm2AudioResources(
    const Lfm2AudioModelConfig& config, absl::string_view model_path,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  const Any2AnyResourceOptions resource_options{
      .model_display_name = "LFM2.5-Audio",
      .generate_audio = config.generate_audio,
      .audio_encoder_backend = config.audio_encoder_backend,
      .llm_backend = config.llm_backend,
      .tts_backend = config.tts_backend,
      .vocoder_backend = config.vocoder_backend,
      .force_fp32_on_gpu = config.force_fp32_on_gpu,
  };
  return CreateAny2AnyResources(resource_options, model_path, cache_dir,
                                backend, num_threads, std::move(env),
                                std::move(lm_resources));
}

absl::StatusOr<Stage<Output>* absl_nonnull> CreateLfm2AudioComponents(
    const Lfm2AudioModelConfig& config,
    std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages) {
  if (resources == nullptr) {
    return absl::InvalidArgumentError("ModelResources must not be null.");
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> text_embedder_model,
                        resources->GetCompiledModel("text_embedder"));

  if (!resources->HasLmModelResources()) {
    return absl::FailedPreconditionError(
        "LmModelResources must be initialized for LFM2.5-Audio.");
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<lm::Tokenizer> lm_tokenizer,
                        resources->GetLmModelResources()->GetTokenizer());
  std::shared_ptr<support::Tokenizer> tokenizer = std::move(lm_tokenizer);

  const bool has_audio_encoder = resources->HasCompiledModel("audio_encoder");
  ABSL_ASSIGN_OR_RETURN(
      auto fan_out,
      Lfm2AudioInputFanOutStage::Create(input_source.get(), has_audio_encoder));
  auto text_embedder_runner =
      std::make_shared<LiteRtRunnerImpl>(text_embedder_model.get());
  ABSL_ASSIGN_OR_RETURN(auto text_stage,
                        Lfm2AudioTextStage::Create(
                            fan_out->text_queue(), fan_out->end_of_turn_queue(),
                            tokenizer, text_embedder_runner, config));

  std::unique_ptr<Lfm2AudioStage> audio_stage;
  if (resources->HasCompiledModel("audio_encoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> audio_encoder_model,
                          resources->GetCompiledModel("audio_encoder"));
    ABSL_ASSIGN_OR_RETURN(
        audio_stage,
        Lfm2AudioStage::Create(
            fan_out->audio_queue(), fan_out->audio_input_metadata_queue(),
            std::make_unique<LiteRtRunnerImpl>(audio_encoder_model.get()),
            config));
  }

  // `Lfm2AudioFrontendStage::Create` uses `text_embedder_runner` synchronously
  // to precompute static control-token embeddings and does not retain it, so
  // `Lfm2AudioTextStage` is the sole runtime caller of `text_embedder_model`.
  ABSL_ASSIGN_OR_RETURN(
      auto frontend,
      Lfm2AudioFrontendStage::Create(text_stage.get(), audio_stage.get(),
                                     tokenizer, text_embedder_runner, config));

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<LiteRtLmRunner> llm_runner,
                        resources->GetLmRunner("llm_backbone"));
  ABSL_RETURN_IF_ERROR(llm_runner->Reset());

  std::unique_ptr<LiteRtRunner> tts_projector_runner;
  if (resources->HasCompiledModel("tts_projector")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> tts_projector_model,
                          resources->GetCompiledModel("tts_projector"));
    tts_projector_runner =
        std::make_unique<LiteRtRunnerImpl>(tts_projector_model.get());
  }
  std::unique_ptr<LiteRtRunner> tts_decoder_runner;
  if (resources->HasCompiledModel("tts_decoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> tts_decoder_model,
                          resources->GetCompiledModel("tts_decoder"));
    tts_decoder_runner =
        std::make_unique<LiteRtRunnerImpl>(tts_decoder_model.get());
  }
  std::unique_ptr<LiteRtRunner> vocoder_runner;
  if (resources->HasCompiledModel("audio_vocoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vocoder_model,
                          resources->GetCompiledModel("audio_vocoder"));
    vocoder_runner = std::make_unique<LiteRtRunnerImpl>(vocoder_model.get());
  }

  ABSL_ASSIGN_OR_RETURN(
      auto generation,
      Lfm2AudioGenerationStage::Create(
          frontend.get(), tokenizer, std::move(llm_runner),
          std::move(tts_projector_runner), std::move(tts_decoder_runner),
          std::move(vocoder_runner), config));

  Stage<Output>* output_stage = generation.get();
  // The first stage must be `InputSource`.
  stages.push_back(std::move(input_source));
  stages.push_back(std::move(fan_out));
  stages.push_back(std::move(text_stage));
  if (audio_stage != nullptr) {
    stages.push_back(std::move(audio_stage));
  }
  stages.push_back(std::move(frontend));
  stages.push_back(std::move(generation));
  return output_stage;
}

}  // namespace litert::omni::any2any

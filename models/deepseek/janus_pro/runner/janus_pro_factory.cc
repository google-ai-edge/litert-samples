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

#include "models/deepseek/janus_pro/runner/janus_pro_factory.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/deepseek/janus_pro/runner/fan_out_stage.h"
#include "models/deepseek/janus_pro/runner/frontend_stage.h"
#include "models/deepseek/janus_pro/runner/image_gen_stage.h"
#include "models/deepseek/janus_pro/runner/image_stage.h"
#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
#include "models/deepseek/janus_pro/runner/llm_stage.h"
#include "models/deepseek/janus_pro/runner/text_stage.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "omni/any2any/any2any_resources.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::any2any {
namespace {

ModelOptions MakeModelOptions(absl::string_view model_path,
                              absl::string_view cache_dir, lm::Backend backend,
                              int num_threads, bool force_fp32_on_gpu) {
  return ModelOptions{
      .model_dir = std::string(model_path),
      .cache_dir = std::string(cache_dir),
      .backend = backend,
      .num_threads = num_threads,
      .force_fp32_on_gpu = force_fp32_on_gpu,
  };
}

}  // namespace

absl::StatusOr<ModelResources> CreateJanusProResources(
    const JanusProModelConfig& config, absl::string_view model_path,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (env == nullptr) {
    return absl::InvalidArgumentError("Environment must not be null.");
  }
  if (lm_resources == nullptr) {
    if (!absl::EndsWithIgnoreCase(model_path, ".litertlm")) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Janus-Pro requires a .litertlm model file, got: ", model_path));
    }
    ABSL_ASSIGN_OR_RETURN(lm_resources, CreateLmModelResources(model_path));
  }
  if (lm_resources == nullptr) {
    return absl::InternalError("CreateLmModelResources returned null.");
  }
  std::shared_ptr<lm::ModelResources> absl_nonnull resolved_lm_resources =
      std::move(lm_resources);

  ModelResources resources(env);
  resources.SetLmModelResources(resolved_lm_resources);

  const ModelOptions cpu_options =
      MakeModelOptions(model_path, cache_dir, lm::Backend::CPU, num_threads,
                       config.force_fp32_on_gpu);
  const ModelOptions vision_options = MakeModelOptions(
      model_path, cache_dir, config.vision_backend.value_or(backend),
      num_threads, config.force_fp32_on_gpu);
  const ModelOptions llm_options = MakeModelOptions(
      model_path, cache_dir, config.llm_backend.value_or(backend), num_threads,
      config.force_fp32_on_gpu);
  const ModelOptions image_gen_options = MakeModelOptions(
      model_path, cache_dir, config.image_gen_backend.value_or(backend),
      num_threads, config.force_fp32_on_gpu);

  auto compile_and_add =
      [&](lm::proto::Any2AnyMetadata::TfLiteModelType section_type,
          absl::string_view key, const ModelOptions& section_options,
          bool required) -> absl::Status {
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<CompiledModel> model,
        TryCompileSection(*env, *resolved_lm_resources, section_type, key,
                          section_options, required));
    if (model != nullptr) {
      return resources.AddCompiledModel(key, std::move(model));
    }
    return absl::OkStatus();
  };

  ABSL_RETURN_IF_ERROR(
      compile_and_add(lm::proto::Any2AnyMetadata::TF_LITE_TEXT_EMBEDDER,
                      "text_embedder", cpu_options, /*required=*/true));
  ABSL_RETURN_IF_ERROR(
      compile_and_add(lm::proto::Any2AnyMetadata::TF_LITE_VISION_ENCODER,
                      "vision_encoder", vision_options, /*required=*/false));

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<LiteRtLmRunner> runner,
      CreateBackboneLmRunner(*env, llm_options, resolved_lm_resources));
  ABSL_RETURN_IF_ERROR(
      resources.AddLmRunner("llm_backbone", std::move(runner)));

  ABSL_RETURN_IF_ERROR(compile_and_add(
      lm::proto::Any2AnyMetadata::TF_LITE_VQ_IMAGE_EMBEDDER,
      "vq_image_embedder", cpu_options, /*required=*/config.generate_image));
  ABSL_RETURN_IF_ERROR(
      compile_and_add(lm::proto::Any2AnyMetadata::TF_LITE_GEN_HEAD, "gen_head",
                      cpu_options, /*required=*/config.generate_image));
  ABSL_RETURN_IF_ERROR(compile_and_add(
      lm::proto::Any2AnyMetadata::TF_LITE_VQ_IMAGE_DECODER, "vq_image_decoder",
      image_gen_options, /*required=*/config.generate_image));

  if (config.generate_image ||
      (resources.HasCompiledModel("vq_image_embedder") &&
       resources.HasCompiledModel("gen_head") &&
       resources.HasCompiledModel("vq_image_decoder"))) {
    ABSL_ASSIGN_OR_RETURN(
        std::unique_ptr<LiteRtLmRunner> uncond_runner,
        CreateBackboneLmRunner(*env, llm_options, resolved_lm_resources));
    ABSL_RETURN_IF_ERROR(
        resources.AddLmRunner("uncond_llm_backbone", std::move(uncond_runner)));
  }

  return resources;
}

absl::StatusOr<Stage<Output>* absl_nonnull> CreateJanusProComponents(
    const JanusProModelConfig& config,
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
        "LmModelResources must be initialized for Janus-Pro.");
  }
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<lm::Tokenizer> lm_tokenizer,
                        resources->GetLmModelResources()->GetTokenizer());
  std::shared_ptr<support::Tokenizer> tokenizer = std::move(lm_tokenizer);

  const bool has_vision_encoder = resources->HasCompiledModel("vision_encoder");
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<JanusProInputFanOutStage> fan_out,
      JanusProInputFanOutStage::Create(input_source.get(), has_vision_encoder));
  auto text_embedder_runner =
      std::make_shared<LiteRtRunnerImpl>(text_embedder_model.get());
  ABSL_ASSIGN_OR_RETURN(std::unique_ptr<JanusProTextStage> text_stage,
                        JanusProTextStage::Create(
                            fan_out->text_queue(), fan_out->end_of_turn_queue(),
                            fan_out->end_of_input_queue(), tokenizer,
                            text_embedder_runner, config));

  std::unique_ptr<JanusProImageStage> image_stage;
  if (has_vision_encoder) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vision_encoder_model,
                          resources->GetCompiledModel("vision_encoder"));
    ABSL_ASSIGN_OR_RETURN(
        image_stage,
        JanusProImageStage::Create(
            fan_out->image_queue(),
            std::make_unique<LiteRtRunnerImpl>(vision_encoder_model.get()),
            config));
  }

  // `JanusProFrontendStage::Create` uses `text_embedder_runner` synchronously
  // to precompute static control-token and CFG embeddings and does not retain
  // it, so `JanusProTextStage` is the sole runtime caller of
  // `text_embedder_model`.
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<JanusProFrontendStage> frontend,
      JanusProFrontendStage::Create(text_stage.get(), image_stage.get(),
                                    tokenizer, text_embedder_runner, config));

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<LiteRtLmRunner> llm_runner,
                        resources->GetLmRunner("llm_backbone"));
  ABSL_RETURN_IF_ERROR(llm_runner->Reset());

  std::shared_ptr<LiteRtLmRunner> uncond_llm_runner;
  if (resources->HasLmRunner("uncond_llm_backbone")) {
    ABSL_ASSIGN_OR_RETURN(uncond_llm_runner,
                          resources->GetLmRunner("uncond_llm_backbone"));
    ABSL_RETURN_IF_ERROR(uncond_llm_runner->Reset());
  } else if (config.generate_image) {
    return absl::NotFoundError(
        "uncond_llm_backbone is required when generate_image is enabled.");
  }

  std::unique_ptr<LiteRtRunner> vq_image_embedder_runner;
  if (resources->HasCompiledModel("vq_image_embedder")) {
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<CompiledModel> vq_image_embedder_model,
        resources->GetCompiledModel("vq_image_embedder"));
    vq_image_embedder_runner =
        std::make_unique<LiteRtRunnerImpl>(vq_image_embedder_model.get());
  } else if (config.generate_image) {
    return absl::NotFoundError(
        "vq_image_embedder is required when generate_image is enabled.");
  }

  std::unique_ptr<LiteRtRunner> gen_head_runner;
  if (resources->HasCompiledModel("gen_head")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> gen_head_model,
                          resources->GetCompiledModel("gen_head"));
    gen_head_runner = std::make_unique<LiteRtRunnerImpl>(gen_head_model.get());
  } else if (config.generate_image) {
    return absl::NotFoundError(
        "gen_head is required when generate_image is enabled.");
  }

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<JanusProLlmStage> llm,
      JanusProLlmStage::Create(frontend.get(), tokenizer, std::move(llm_runner),
                               std::move(uncond_llm_runner),
                               std::move(vq_image_embedder_runner),
                               std::move(gen_head_runner), config));

  std::unique_ptr<LiteRtRunner> vq_image_decoder_runner;
  if (resources->HasCompiledModel("vq_image_decoder")) {
    ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vq_image_decoder_model,
                          resources->GetCompiledModel("vq_image_decoder"));
    vq_image_decoder_runner =
        std::make_unique<LiteRtRunnerImpl>(vq_image_decoder_model.get());
  } else if (config.generate_image) {
    return absl::NotFoundError(
        "vq_image_decoder is required when generate_image is enabled.");
  }

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<JanusProImageGenStage> image_gen,
      JanusProImageGenStage::Create(llm.get(), config,
                                    std::move(vq_image_decoder_runner)));

  Stage<Output>* output_stage = image_gen.get();
  // The first stage must be `InputSource`.
  stages.push_back(std::move(input_source));
  stages.push_back(std::move(fan_out));
  stages.push_back(std::move(text_stage));
  if (image_stage != nullptr) {
    stages.push_back(std::move(image_stage));
  }
  stages.push_back(std::move(frontend));
  stages.push_back(std::move(llm));
  stages.push_back(std::move(image_gen));
  return output_stage;
}

}  // namespace litert::omni::any2any

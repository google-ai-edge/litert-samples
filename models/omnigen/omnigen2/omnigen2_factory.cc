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

#include "models/omnigen/omnigen2/omnigen2_factory.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/omnigen/omnigen2/denoiser_stage.h"
#include "models/omnigen/omnigen2/omnigen2_model_config.h"
#include "models/omnigen/omnigen2/text_encoder_stage.h"
#include "models/omnigen/omnigen2/vae_decoder_stage.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/text2image/prompt_source.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::text2image::omnigen2 {
namespace {

constexpr std::array<lm::proto::Text2ImageMetadata::TfLiteModelType, 4>
    kStandardMainBlockTypes = {
        lm::proto::Text2ImageMetadata::
            TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_0,
        lm::proto::Text2ImageMetadata::
            TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_1,
        lm::proto::Text2ImageMetadata::
            TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_2,
        lm::proto::Text2ImageMetadata::
            TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_3,
    };

constexpr std::array<absl::string_view, 2> kExtraMainBlockSectionNames = {
    kSectionDitSingleBlock4,
    kSectionDitSingleBlock5,
};

ModelOptions MakeModelOptions(absl::string_view model_dir,
                              absl::string_view cache_dir, lm::Backend backend,
                              int num_threads) {
  ModelOptions options;
  options.model_dir = model_dir;
  options.cache_dir = cache_dir;
  options.backend = backend;
  options.num_threads = num_threads;
  options.force_fp32_on_gpu = true;
  return options;
}

absl::StatusOr<size_t> ResolveSignatureArgInputIndex(
    const CompiledModel& model, size_t arg_index,
    absl::string_view fallback_name = "") {
  const std::string exact_arg = absl::StrCat("args_", arg_index);
  const std::string suffixed_arg = absl::StrCat("_", exact_arg);
  const std::string prefixed_arg = absl::StrCat("args_", arg_index, ":");
  auto names_res = model.GetSignatureInputNames();
  if (names_res.HasValue()) {
    const auto& names = *names_res;
    for (size_t i = 0; i < names.size(); ++i) {
      if (names[i] == exact_arg || absl::EndsWith(names[i], suffixed_arg) ||
          absl::StrContains(names[i], prefixed_arg)) {
        return i;
      }
    }
    if (!fallback_name.empty()) {
      for (size_t i = 0; i < names.size(); ++i) {
        if (names[i] == fallback_name) {
          return i;
        }
      }
    }
    if (arg_index < names.size()) {
      return arg_index;
    }
    return absl::NotFoundError(absl::StrCat(
        "Input argument ", exact_arg, " exceeds model signature input count (",
        names.size(), ")."));
  }
  return arg_index;
}

absl::Status RegisterCompiledModelFromTfLiteSection(
    ::litert::Environment& env, lm::ModelResources& lm_resources,
    const ModelOptions& options,
    lm::proto::Text2ImageMetadata::TfLiteModelType model_type,
    absl::string_view resource_name, ModelResources& resources) {
  ABSL_ASSIGN_OR_RETURN(absl::string_view buffer,
                        lm_resources.GetTFLiteModelBuffer(model_type));
  ABSL_ASSIGN_OR_RETURN(
      CompiledModel compiled,
      CreateCompiledModelFromBuffer(env, options, buffer, resource_name));
  return resources.AddCompiledModel(
      resource_name, std::make_shared<CompiledModel>(std::move(compiled)));
}

absl::Status RegisterCompiledModelFromGenericBinarySection(
    ::litert::Environment& env, lm::ModelResources& lm_resources,
    const ModelOptions& options, absl::string_view section_name,
    absl::string_view resource_name, ModelResources& resources) {
  ABSL_ASSIGN_OR_RETURN(absl::string_view buffer,
                        lm_resources.GetGenericBinaryDataBuffer(section_name));
  ABSL_ASSIGN_OR_RETURN(
      CompiledModel compiled,
      CreateCompiledModelFromBuffer(env, options, buffer, resource_name));
  return resources.AddCompiledModel(
      resource_name, std::make_shared<CompiledModel>(std::move(compiled)));
}

}  // namespace

absl::StatusOr<ModelResources> CreateOmniGen2Resources(
    const OmniGen2ModelConfig& config, absl::string_view model_path,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  ABSL_RETURN_IF_ERROR(ValidateOmniGen2ModelConfig(config));
  if (env == nullptr) {
    return absl::InvalidArgumentError("OmniGen2 Environment must not be null.");
  }

  if (lm_resources == nullptr) {
    if (model_path.empty()) {
      return absl::NotFoundError(
          "No .litertlm model container provided for OmniGen2.");
    }
    const std::string litertlm_path = ResolveLitertLmPath(model_path);
    if (litertlm_path.empty()) {
      return absl::NotFoundError(
          absl::StrCat("No .litertlm model container found in: ", model_path));
    }
    ABSL_ASSIGN_OR_RETURN(lm_resources, CreateLmModelResources(litertlm_path));
  }

  ModelResources resources(env);
  resources.SetLmModelResources(lm_resources);

  const lm::Backend textenc_backend = config.textenc_backend.value_or(backend);
  const lm::Backend dit_backend = config.dit_backend.value_or(backend);
  const lm::Backend vae_backend = config.vae_backend.value_or(backend);

  ModelOptions textenc_options =
      MakeModelOptions(model_path, cache_dir, textenc_backend, num_threads);
  ModelOptions dit_options =
      MakeModelOptions(model_path, cache_dir, dit_backend, num_threads);
  ModelOptions vae_options =
      MakeModelOptions(model_path, cache_dir, vae_backend, num_threads);

  ABSL_RETURN_IF_ERROR(RegisterCompiledModelFromTfLiteSection(
      *env, *lm_resources, textenc_options,
      lm::proto::Text2ImageMetadata::TF_LITE_TEXT_ENCODER, "omnigen2_textenc",
      resources));

  ABSL_RETURN_IF_ERROR(RegisterCompiledModelFromGenericBinarySection(
      *env, *lm_resources, dit_options, kSectionDitCapEmbedder,
      "omnigen2_dit_cap_embedder", resources));

  ABSL_RETURN_IF_ERROR(RegisterCompiledModelFromGenericBinarySection(
      *env, *lm_resources, dit_options, kSectionDitContextRefiner,
      "omnigen2_dit_context_refiner", resources));

  ABSL_RETURN_IF_ERROR(RegisterCompiledModelFromGenericBinarySection(
      *env, *lm_resources, dit_options, kSectionDitXEmbedder,
      "omnigen2_dit_x_embedder", resources));

  ABSL_RETURN_IF_ERROR(RegisterCompiledModelFromGenericBinarySection(
      *env, *lm_resources, dit_options, kSectionDitNoiseRefiner,
      "omnigen2_dit_noise_refiner", resources));

  for (size_t i = 0; i < kStandardMainBlockTypes.size(); ++i) {
    ABSL_RETURN_IF_ERROR(RegisterCompiledModelFromTfLiteSection(
        *env, *lm_resources, dit_options, kStandardMainBlockTypes[i],
        absl::StrCat("omnigen2_dit_main_", i), resources));
  }
  for (size_t i = 0; i < kExtraMainBlockSectionNames.size(); ++i) {
    const size_t shard_idx = kStandardMainBlockTypes.size() + i;
    ABSL_RETURN_IF_ERROR(RegisterCompiledModelFromGenericBinarySection(
        *env, *lm_resources, dit_options, kExtraMainBlockSectionNames[i],
        absl::StrCat("omnigen2_dit_main_", shard_idx), resources));
  }

  ABSL_RETURN_IF_ERROR(RegisterCompiledModelFromTfLiteSection(
      *env, *lm_resources, dit_options,
      lm::proto::Text2ImageMetadata::TF_LITE_DIFFUSION_TRANSFORMER_FINAL,
      "omnigen2_dit_final", resources));

  ABSL_RETURN_IF_ERROR(RegisterCompiledModelFromTfLiteSection(
      *env, *lm_resources, vae_options,
      lm::proto::Text2ImageMetadata::TF_LITE_IMAGE_DECODER, "omnigen2_vae",
      resources));

  return resources;
}

absl::StatusOr<Stage<Output>* absl_nonnull> CreateOmniGen2Components(
    const OmniGen2ModelConfig& config,
    std::unique_ptr<PromptSource> absl_nonnull prompt_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages) {
  ABSL_RETURN_IF_ERROR(ValidateOmniGen2ModelConfig(config));
  if (prompt_source == nullptr) {
    return absl::InvalidArgumentError(
        "OmniGen2 prompt_source must not be null.");
  }
  if (resources == nullptr || !resources->HasLmModelResources()) {
    return absl::InvalidArgumentError(
        "OmniGen2 requires a .litertlm model container in ModelResources.");
  }

  auto lm_resources = resources->GetLmModelResources();
  auto tok = lm_resources->GetTokenizer();
  if (!tok.ok()) {
    tok = lm_resources->GetTokenizer(
        lm::proto::Text2ImageMetadata::TF_LITE_TEXT_ENCODER);
  }
  if (!tok.ok()) {
    return tok.status();
  }
  std::unique_ptr<support::Tokenizer> tokenizer = *std::move(tok);

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> textenc_model,
                        resources->GetCompiledModel("omnigen2_textenc"));
  OmniGen2TextEncoderStage::Config textenc_config;
  textenc_config.seq_len = config.seq_len;
  textenc_config.pad_token_id = ResolveOmniGen2PadTokenId(tokenizer.get());
  textenc_config.prompt_prefix = "user\n";
  textenc_config.prefix_token_ids =
      ResolveOmniGen2PrefixTokenIds(tokenizer.get());
  textenc_config.suffix_token_ids =
      ResolveOmniGen2SuffixTokenIds(tokenizer.get());
  textenc_config.encode_negative_prompt = config.text_guidance_scale > 1.0f;
  ABSL_ASSIGN_OR_RETURN(
      textenc_config.input_indices.input_ids,
      ResolveSignatureArgInputIndex(*textenc_model, 0, "input_ids"));
  ABSL_ASSIGN_OR_RETURN(
      textenc_config.input_indices.attention_mask,
      ResolveSignatureArgInputIndex(*textenc_model, 1, "attention_mask"));

  ABSL_ASSIGN_OR_RETURN(
      auto text_encoder,
      OmniGen2TextEncoderStage::Create(
          prompt_source.get(), std::move(tokenizer),
          std::make_unique<LiteRtRunnerImpl>(textenc_model.get()),
          textenc_config));

  OmniGen2DenoiserStage::Runners dit_runners;
  OmniGen2DenoiserStage::InputIndices dit_indices;

  ABSL_ASSIGN_OR_RETURN(
      std::shared_ptr<CompiledModel> cap_emb_model,
      resources->GetCompiledModel("omnigen2_dit_cap_embedder"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.cap_embedder.cap,
      ResolveSignatureArgInputIndex(*cap_emb_model, 0, "cap_feats"));
  dit_runners.cap_embedder =
      std::make_unique<LiteRtRunnerImpl>(cap_emb_model.get());

  ABSL_ASSIGN_OR_RETURN(
      std::shared_ptr<CompiledModel> ctx_ref_model,
      resources->GetCompiledModel("omnigen2_dit_context_refiner"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.context_refiner.cap,
      ResolveSignatureArgInputIndex(*ctx_ref_model, 0, "cap"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.context_refiner.cos,
      ResolveSignatureArgInputIndex(*ctx_ref_model, 1, "cos"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.context_refiner.sin,
      ResolveSignatureArgInputIndex(*ctx_ref_model, 2, "sin"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.context_refiner.mask,
      ResolveSignatureArgInputIndex(*ctx_ref_model, 3, "mask"));
  dit_runners.context_refiner =
      std::make_unique<LiteRtRunnerImpl>(ctx_ref_model.get());

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> x_emb_model,
                        resources->GetCompiledModel("omnigen2_dit_x_embedder"));
  ABSL_ASSIGN_OR_RETURN(dit_indices.x_embedder.x,
                        ResolveSignatureArgInputIndex(*x_emb_model, 0, "x"));
  ABSL_ASSIGN_OR_RETURN(dit_indices.x_embedder.t,
                        ResolveSignatureArgInputIndex(*x_emb_model, 1, "t"));
  dit_runners.x_embedder =
      std::make_unique<LiteRtRunnerImpl>(x_emb_model.get());

  ABSL_ASSIGN_OR_RETURN(
      std::shared_ptr<CompiledModel> noise_ref_model,
      resources->GetCompiledModel("omnigen2_dit_noise_refiner"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.noise_refiner.x,
      ResolveSignatureArgInputIndex(*noise_ref_model, 0, "x"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.noise_refiner.cos,
      ResolveSignatureArgInputIndex(*noise_ref_model, 1, "cos"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.noise_refiner.sin,
      ResolveSignatureArgInputIndex(*noise_ref_model, 2, "sin"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.noise_refiner.temb,
      ResolveSignatureArgInputIndex(*noise_ref_model, 3, "temb"));
  dit_runners.noise_refiner =
      std::make_unique<LiteRtRunnerImpl>(noise_ref_model.get());

  for (size_t i = 0; i < kNumMainBlockShards; ++i) {
    ABSL_ASSIGN_OR_RETURN(
        std::shared_ptr<CompiledModel> main_model,
        resources->GetCompiledModel(absl::StrCat("omnigen2_dit_main_", i)));
    ABSL_ASSIGN_OR_RETURN(
        dit_indices.main_blocks[i].joint,
        ResolveSignatureArgInputIndex(*main_model, 0, "joint"));
    ABSL_ASSIGN_OR_RETURN(dit_indices.main_blocks[i].cos,
                          ResolveSignatureArgInputIndex(*main_model, 1, "cos"));
    ABSL_ASSIGN_OR_RETURN(dit_indices.main_blocks[i].sin,
                          ResolveSignatureArgInputIndex(*main_model, 2, "sin"));
    ABSL_ASSIGN_OR_RETURN(
        dit_indices.main_blocks[i].mask,
        ResolveSignatureArgInputIndex(*main_model, 3, "mask"));
    ABSL_ASSIGN_OR_RETURN(
        dit_indices.main_blocks[i].temb,
        ResolveSignatureArgInputIndex(*main_model, 4, "temb"));
    dit_runners.main_blocks[i] =
        std::make_unique<LiteRtRunnerImpl>(main_model.get());
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> final_model,
                        resources->GetCompiledModel("omnigen2_dit_final"));
  ABSL_ASSIGN_OR_RETURN(
      dit_indices.final_stage.joint,
      ResolveSignatureArgInputIndex(*final_model, 0, "joint"));
  ABSL_ASSIGN_OR_RETURN(dit_indices.final_stage.temb,
                        ResolveSignatureArgInputIndex(*final_model, 1, "temb"));
  dit_runners.final_stage =
      std::make_unique<LiteRtRunnerImpl>(final_model.get());

  ABSL_ASSIGN_OR_RETURN(
      auto denoiser,
      OmniGen2DenoiserStage::Create(text_encoder.get(), config,
                                    std::move(dit_runners), dit_indices));

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vae_model,
                        resources->GetCompiledModel("omnigen2_vae"));
  ABSL_ASSIGN_OR_RETURN(
      auto vae_decoder,
      OmniGen2VaeDecoderStage::Create(
          denoiser.get(), config,
          std::make_unique<LiteRtRunnerImpl>(vae_model.get())));

  Stage<Output>* output_stage = vae_decoder.get();
  stages.push_back(std::move(prompt_source));
  stages.push_back(std::move(text_encoder));
  stages.push_back(std::move(denoiser));
  stages.push_back(std::move(vae_decoder));
  return output_stage;
}

}  // namespace litert::omni::text2image::omnigen2

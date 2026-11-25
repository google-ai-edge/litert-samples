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

#include "models/tongyi/z_image_turbo/runner/z_image_turbo_factory.h"

#include <array>
#include <cstddef>
#include <filesystem>  // NOLINT: Required for path manipulation.
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "models/tongyi/z_image_turbo/runner/denoiser_stage.h"
#include "models/tongyi/z_image_turbo/runner/z_image_turbo_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/model_cache.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/text2image/io_types.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/text_encoder_stage.h"
#include "omni/text2image/vae_decoder_stage.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/embedding_metadata.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::text2image {
namespace {

using ::litert::lm::proto::Text2ImageMetadata;

constexpr absl::string_view kTextEncModel = "z_text_encoder";
constexpr absl::string_view kCapEmbModel = "z_embc";
constexpr absl::string_view kContextRefModel = "z_refc";
constexpr absl::string_view kXEmbModel = "z_embx";
constexpr absl::string_view kNoiseRefModel = "z_refx";
constexpr std::array<absl::string_view, 6> kMainBlockModels = {
    "zc_main0", "zc_main1", "zc_main2", "zc_main3", "zc_main4", "zc_main5",
};
constexpr absl::string_view kFinalModel = "zc_final";
constexpr absl::string_view kVaeDecModel = "zvae";

constexpr std::array<Text2ImageMetadata::TfLiteModelType, 4>
    kStandardMainBlockTypes = {
        Text2ImageMetadata::TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_0,
        Text2ImageMetadata::TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_1,
        Text2ImageMetadata::TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_2,
        Text2ImageMetadata::TF_LITE_DIFFUSION_TRANSFORMER_SINGLE_BLOCK_3,
    };

constexpr std::array<absl::string_view, 2> kExtraMainBlockSections = {
    z_image_turbo::kSingleBlock4SectionName,
    z_image_turbo::kSingleBlock5SectionName,
};

std::string JoinPath(absl::string_view dir, absl::string_view file) {
  return (std::filesystem::path(std::string_view(dir)) / std::string_view(file))
      .string();
}

absl::StatusOr<CompiledModel> CompileTfLiteSectionOrFile(
    Environment& env, const ModelOptions& options,
    lm::ModelResources* absl_nullable lm_resources,
    Text2ImageMetadata::TfLiteModelType model_type,
    absl::string_view model_name, absl::string_view fallback_file) {
  if (lm_resources != nullptr) {
    ABSL_ASSIGN_OR_RETURN(absl::string_view buf,
                          lm_resources->GetTFLiteModelBuffer(model_type));
    return CreateCompiledModelFromBuffer(env, options, buf, model_name);
  }
  return CreateCompiledModel(env, options, fallback_file);
}

absl::StatusOr<CompiledModel> CompileGenericBinarySectionOrFile(
    Environment& env, const ModelOptions& options,
    lm::ModelResources* absl_nullable lm_resources,
    absl::string_view section_name, absl::string_view model_name,
    absl::string_view fallback_file) {
  if (lm_resources != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        absl::string_view buf,
        lm_resources->GetGenericBinaryDataBuffer(section_name));
    return CreateCompiledModelFromBuffer(env, options, buf, model_name);
  }
  return CreateCompiledModel(env, options, fallback_file);
}

absl::StatusOr<std::optional<CompiledModel>> TryCompileMainBlockShard(
    Environment& env, const ModelOptions& options,
    lm::ModelResources* absl_nullable lm_resources, const size_t shard_idx,
    absl::string_view model_name, absl::string_view fallback_file) {
  if (lm_resources != nullptr) {
    if (shard_idx < kStandardMainBlockTypes.size()) {
      const Text2ImageMetadata::TfLiteModelType model_type =
          kStandardMainBlockTypes[shard_idx];
      if (!lm_resources->HasTFLiteModel(model_type)) {
        return std::nullopt;
      }
      ABSL_ASSIGN_OR_RETURN(absl::string_view buf,
                            lm_resources->GetTFLiteModelBuffer(model_type));
      ABSL_ASSIGN_OR_RETURN(
          CompiledModel compiled,
          CreateCompiledModelFromBuffer(env, options, buf, model_name));
      return std::move(compiled);
    }
    const size_t extra_idx = shard_idx - kStandardMainBlockTypes.size();
    const absl::string_view section_name = kExtraMainBlockSections[extra_idx];
    absl::StatusOr<absl::string_view> buf =
        lm_resources->GetGenericBinaryDataBuffer(section_name);
    if (absl::IsNotFound(buf.status())) {
      return std::nullopt;
    }
    if (!buf.ok()) {
      return buf.status();
    }
    ABSL_ASSIGN_OR_RETURN(
        CompiledModel compiled,
        CreateCompiledModelFromBuffer(env, options, *buf, model_name));
    return std::move(compiled);
  }

  if (shard_idx > 0 &&
      !CheckFileReadable(JoinPath(options.model_dir, fallback_file)).ok()) {
    return std::nullopt;
  }
  ABSL_ASSIGN_OR_RETURN(CompiledModel compiled,
                        CreateCompiledModel(env, options, fallback_file));
  return std::move(compiled);
}

}  // namespace

absl::StatusOr<ModelResources> CreateZImageTurboResources(
    const ZImageTurboModelConfig& config, absl::string_view model_folder,
    absl::string_view cache_dir, lm::Backend backend, int num_threads,
    std::shared_ptr<Environment> absl_nonnull env,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (env == nullptr) {
    return absl::InvalidArgumentError(
        "CreateZImageTurboResources requires a non-null Environment.");
  }
  if (model_folder.empty() && lm_resources == nullptr) {
    return absl::InvalidArgumentError(
        "CreateZImageTurboResources requires a non-empty model_folder or "
        "lm_resources.");
  }

  if (lm_resources == nullptr && !model_folder.empty()) {
    const std::string litertlm_path = ResolveLitertLmPath(model_folder);
    if (!litertlm_path.empty()) {
      ABSL_ASSIGN_OR_RETURN(lm_resources,
                            CreateLmModelResources(litertlm_path));
    }
  }

  // Own `env` and `lm_resources` inside `resources` before compiling any models
  // so member destruction order tears down `CompiledModel`s first.
  ModelResources resources(std::move(env));
  if (lm_resources != nullptr) {
    resources.SetLmModelResources(std::move(lm_resources));
  }
  std::shared_ptr<Environment> owned_env = resources.environment();
  Environment& env_ref = *owned_env;
  std::shared_ptr<lm::ModelResources> container =
      resources.GetLmModelResources();

  ModelOptions options;
  options.model_dir = std::string(model_folder);
  options.cache_dir =
      cache_dir.empty() ? std::string(kNoCacheDir) : std::string(cache_dir);
  options.backend = backend;
  options.num_threads = num_threads > 0 ? num_threads : 4;
  options.force_fp32_on_gpu = true;

  ABSL_ASSIGN_OR_RETURN(
      CompiledModel text_enc,
      CompileTfLiteSectionOrFile(env_ref, options, container.get(),
                                 Text2ImageMetadata::TF_LITE_TEXT_ENCODER,
                                 kTextEncModel, config.text_encoder_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      kTextEncModel, std::make_shared<CompiledModel>(std::move(text_enc))));

  ABSL_ASSIGN_OR_RETURN(CompiledModel embc,
                        CompileGenericBinarySectionOrFile(
                            env_ref, options, container.get(),
                            z_image_turbo::kCapEmbedderSectionName,
                            kCapEmbModel, config.cap_embedder_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      kCapEmbModel, std::make_shared<CompiledModel>(std::move(embc))));

  ABSL_ASSIGN_OR_RETURN(CompiledModel refc,
                        CompileGenericBinarySectionOrFile(
                            env_ref, options, container.get(),
                            z_image_turbo::kContextRefinerSectionName,
                            kContextRefModel, config.context_refiner_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      kContextRefModel, std::make_shared<CompiledModel>(std::move(refc))));

  ABSL_ASSIGN_OR_RETURN(
      CompiledModel embx,
      CompileGenericBinarySectionOrFile(env_ref, options, container.get(),
                                        z_image_turbo::kXEmbedderSectionName,
                                        kXEmbModel, config.x_embedder_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      kXEmbModel, std::make_shared<CompiledModel>(std::move(embx))));

  ABSL_ASSIGN_OR_RETURN(CompiledModel refx,
                        CompileGenericBinarySectionOrFile(
                            env_ref, options, container.get(),
                            z_image_turbo::kNoiseRefinerSectionName,
                            kNoiseRefModel, config.noise_refiner_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      kNoiseRefModel, std::make_shared<CompiledModel>(std::move(refx))));

  size_t loaded_main_blocks = 0;
  for (size_t i = 0;
       i < kMainBlockModels.size() && i < config.main_block_files.size(); ++i) {
    ABSL_ASSIGN_OR_RETURN(std::optional<CompiledModel> main_block,
                          TryCompileMainBlockShard(
                              env_ref, options, container.get(), i,
                              kMainBlockModels[i], config.main_block_files[i]));
    if (!main_block.has_value()) {
      break;
    }
    ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
        kMainBlockModels[i],
        std::make_shared<CompiledModel>(std::move(*main_block))));
    ++loaded_main_blocks;
  }
  if (loaded_main_blocks == 0) {
    return absl::NotFoundError(
        "Z-Image-Turbo requires at least one main_blocks shard "
        "(tf_lite_diffusion_transformer_single_block_0).");
  }

  ABSL_ASSIGN_OR_RETURN(
      CompiledModel final_layer,
      CompileTfLiteSectionOrFile(
          env_ref, options, container.get(),
          Text2ImageMetadata::TF_LITE_DIFFUSION_TRANSFORMER_FINAL, kFinalModel,
          config.final_layer_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      kFinalModel, std::make_shared<CompiledModel>(std::move(final_layer))));

  ABSL_ASSIGN_OR_RETURN(
      CompiledModel vae_dec,
      CompileTfLiteSectionOrFile(env_ref, options, container.get(),
                                 Text2ImageMetadata::TF_LITE_IMAGE_DECODER,
                                 kVaeDecModel, config.vae_decoder_file));
  ABSL_RETURN_IF_ERROR(resources.AddCompiledModel(
      kVaeDecModel, std::make_shared<CompiledModel>(std::move(vae_dec))));

  return resources;
}

absl::StatusOr<std::unique_ptr<VaeDecoderStage>> CreateZImageVaeDecoderStage(
    Stage<DenoisedLatents>* absl_nonnull denoiser,
    const ZImageTurboModelConfig& config,
    std::unique_ptr<LiteRtRunner> absl_nonnull runner) {
  if (config.vae_scaling_factor == 0.0f || config.img_size <= 0 ||
      config.in_channels <= 0 || config.patch_size <= 0 ||
      config.GridSize() <= 0 ||
      config.img_size % z_image_turbo::kVaeSpatialDownsampleFactor != 0 ||
      config.LatentSize() % config.patch_size != 0) {
    return absl::InvalidArgumentError(
        "ZImageVaeDecoderStage received invalid ZImageTurboModelConfig.");
  }

  const int latent_size = config.LatentSize();
  const size_t expected_latent_elements =
      static_cast<size_t>(config.in_channels) *
      static_cast<size_t>(latent_size) * static_cast<size_t>(latent_size);
  VaeDecoderStage::Config vae_config{
      .image_width = config.img_size,
      .image_height = config.img_size,
      .expected_latent_elements = expected_latent_elements,
      .stage_name = "ZImageVaeDecoderStage",
  };
  return VaeDecoderStage::Create(
      denoiser, std::move(vae_config), std::move(runner),
      [latent_size, expected_latent_elements, patch_size = config.patch_size,
       in_channels = config.in_channels,
       inv_scale = 1.0f / config.vae_scaling_factor,
       shift = config.vae_shift_factor](const DenoisedLatents& denoised)
          -> absl::StatusOr<std::vector<float>> {
        std::vector<float> nchw_latents(expected_latent_elements);
        ABSL_RETURN_IF_ERROR(
            UnpackZImageLatents(absl::MakeConstSpan(denoised.packed_latents),
                                absl::MakeSpan(nchw_latents), latent_size,
                                latent_size, patch_size, in_channels));
        for (float& value : nchw_latents) {
          value = value * inv_scale + shift;
        }
        return nchw_latents;
      });
}

absl::StatusOr<Stage<Output>* absl_nonnull> CreateZImageTurboComponents(
    const ZImageTurboModelConfig& config, absl::string_view model_folder,
    std::unique_ptr<PromptSource> absl_nonnull prompt_source,
    std::shared_ptr<ModelResources> absl_nonnull resources,
    std::vector<std::unique_ptr<internal::StageBase>>& stages) {
  stages.clear();

  if (prompt_source == nullptr) {
    return absl::InvalidArgumentError("prompt_source must not be null.");
  }
  if (resources == nullptr) {
    return absl::InvalidArgumentError("resources must not be null.");
  }

  std::unique_ptr<support::Tokenizer> tokenizer;
  std::shared_ptr<lm::ModelResources> lm_resources =
      resources->GetLmModelResources();
  if (lm_resources != nullptr) {
    absl::StatusOr<std::unique_ptr<lm::Tokenizer>> tok =
        lm_resources->GetTokenizer();
    if (!tok.ok()) {
      tok = lm_resources->GetTokenizer(
          lm::proto::EmbeddingMetadata::TF_LITE_TEXT_ENCODER);
    }
    if (!tok.ok()) {
      return tok.status();
    }
    tokenizer = *std::move(tok);
  } else if (!model_folder.empty()) {
    const std::string tokenizer_path =
        JoinPath(model_folder, config.tokenizer_file);
    ABSL_ASSIGN_OR_RETURN(
        tokenizer, CreateTokenizer(/*lm_resources=*/nullptr, tokenizer_path));
  } else {
    return absl::InvalidArgumentError(
        "Z-Image-Turbo requires a tokenizer from .litertlm or model_folder.");
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> text_enc_model,
                        resources->GetCompiledModel(kTextEncModel));
  TextEncoderStage::Config textenc_config;
  textenc_config.seq_len = config.seq_len;
  textenc_config.pad_token_id = config.pad_token_id;
  textenc_config.prefix_token_ids = config.prefix_token_ids;
  textenc_config.suffix_token_ids = config.suffix_token_ids;
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<TextEncoderStage> text_encoder,
      TextEncoderStage::Create(
          prompt_source.get(), std::move(tokenizer),
          std::make_unique<LiteRtRunnerImpl>(text_enc_model.get()),
          textenc_config));

  ZImageDenoiserStage::Runners denoiser_runners;
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> embc_model,
                        resources->GetCompiledModel(kCapEmbModel));
  denoiser_runners.cap_embedder =
      std::make_unique<LiteRtRunnerImpl>(embc_model.get());
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> refc_model,
                        resources->GetCompiledModel(kContextRefModel));
  denoiser_runners.context_refiner =
      std::make_unique<LiteRtRunnerImpl>(refc_model.get());
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> embx_model,
                        resources->GetCompiledModel(kXEmbModel));
  denoiser_runners.x_embedder =
      std::make_unique<LiteRtRunnerImpl>(embx_model.get());
  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> refx_model,
                        resources->GetCompiledModel(kNoiseRefModel));
  denoiser_runners.noise_refiner =
      std::make_unique<LiteRtRunnerImpl>(refx_model.get());

  for (absl::string_view block_name : kMainBlockModels) {
    absl::StatusOr<std::shared_ptr<CompiledModel>> block_model =
        resources->GetCompiledModel(block_name);
    if (!block_model.ok()) {
      break;
    }
    denoiser_runners.main_blocks.push_back(
        std::make_unique<LiteRtRunnerImpl>(block_model->get()));
  }
  if (denoiser_runners.main_blocks.empty()) {
    return absl::FailedPreconditionError(
        "Z-Image-Turbo ModelResources contains no compiled main_blocks "
        "shards.");
  }

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> final_model,
                        resources->GetCompiledModel(kFinalModel));
  denoiser_runners.final_layer =
      std::make_unique<LiteRtRunnerImpl>(final_model.get());

  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<ZImageDenoiserStage> denoiser_stage,
      ZImageDenoiserStage::Create(text_encoder.get(), config,
                                  std::move(denoiser_runners)));

  ABSL_ASSIGN_OR_RETURN(std::shared_ptr<CompiledModel> vae_model,
                        resources->GetCompiledModel(kVaeDecModel));
  ABSL_ASSIGN_OR_RETURN(
      std::unique_ptr<VaeDecoderStage> vae_stage,
      CreateZImageVaeDecoderStage(
          denoiser_stage.get(), config,
          std::make_unique<LiteRtRunnerImpl>(vae_model.get())));

  Stage<Output>* output_stage = vae_stage.get();
  stages.push_back(std::move(prompt_source));
  stages.push_back(std::move(text_encoder));
  stages.push_back(std::move(denoiser_stage));
  stages.push_back(std::move(vae_stage));
  return output_stage;
}

}  // namespace litert::omni::text2image

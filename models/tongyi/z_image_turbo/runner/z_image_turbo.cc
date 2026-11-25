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

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "models/tongyi/z_image_turbo/runner/z_image_turbo.pb.h"
#include "models/tongyi/z_image_turbo/runner/z_image_turbo_factory.h"
#include "models/tongyi/z_image_turbo/runner/z_image_turbo_model_config.h"
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "omni/base/custom_model_utils.h"
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/text2image_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"

namespace litert::omni::text2image {
namespace {

constexpr absl::string_view kModelNames[] = {
    "z-image-turbo", "z_image_turbo", "Tongyi-MAI/Z-Image-Turbo",
    "z-image",       "z_image",
};

absl::StatusOr<lm::proto::Text2ImageMetadata> ResolveMetadata(
    const MessageLite* absl_nullable metadata) {
  lm::proto::Text2ImageMetadata resolved;
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  ABSL_RETURN_IF_ERROR(EnsureDefaultCustomModel<lm::proto::ZImageTurbo>(
      *resolved.mutable_image_gen_model_type()));
  return resolved;
}

absl::Status PopulateConfigFromMetadata(
    const lm::proto::Text2ImageMetadata& metadata,
    ZImageTurboModelConfig& config) {
  ABSL_ASSIGN_OR_RETURN(std::optional<lm::proto::ZImageTurbo> proto,
                        UnpackCustomModel<lm::proto::ZImageTurbo>(
                            metadata.image_gen_model_type()));
  if (proto.has_value()) {
    PopulateZImageTurboConfigFromProto(*proto, config);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<OmniSessionFactory>>
GetZImageTurboSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata, Environment& env,
    Options& /*options*/,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }

  ABSL_ASSIGN_OR_RETURN(lm::proto::Text2ImageMetadata resolved_metadata,
                        ResolveMetadata(metadata));
  ZImageTurboModelConfig config;
  ABSL_RETURN_IF_ERROR(PopulateConfigFromMetadata(resolved_metadata, config));

  auto shared_env = std::make_shared<Environment>(std::move(env));
  const int num_threads = settings.num_threads > 0 ? settings.num_threads : 4;
  ABSL_ASSIGN_OR_RETURN(ModelResources model_resources,
                        CreateZImageTurboResources(
                            config, settings.model_path, settings.cache_dir,
                            ToLmBackend(settings.backend), num_threads,
                            std::move(shared_env), std::move(lm_resources)));
  auto resources = std::make_shared<ModelResources>(std::move(model_resources));

  LeasedText2ImageSessionOptions lease_options{
      .model_display_name = "Z-Image-Turbo",
      .default_prompt_params =
          ImageGenInputMetadata{
              .width = config.img_size,
              .height = config.img_size,
              .num_inference_steps = config.steps,
              .seed = config.default_seed,
          },
  };
  LeasedText2ImageSessionFactory::ComponentBuilder component_builder =
      [base_config = std::move(config), model_path = settings.model_path](
          const lm::proto::Text2ImageSessionConfig& effective_config,
          std::unique_ptr<PromptSource> absl_nonnull prompt_source,
          std::shared_ptr<ModelResources> absl_nonnull model_resources,
          std::vector<std::unique_ptr<internal::StageBase>>& stages)
      -> absl::StatusOr<Stage<Output>* absl_nonnull> {
    ZImageTurboModelConfig session_model_config = base_config;
    if (effective_config.width() > 0 || effective_config.height() > 0) {
      if (effective_config.width() != effective_config.height()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Z-Image-Turbo requires square image dimensions (width == height), "
            "got ",
            effective_config.width(), "x", effective_config.height()));
      }
      if (effective_config.width() != base_config.img_size) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Z-Image-Turbo session image size (", effective_config.width(),
            ") must match exported model img_size (", base_config.img_size,
            ")."));
      }
    }
    if (effective_config.num_inference_steps() > 0) {
      session_model_config.steps = effective_config.num_inference_steps();
    }
    if (effective_config.has_seed()) {
      session_model_config.default_seed =
          static_cast<uint64_t>(effective_config.seed());
    }
    return CreateZImageTurboComponents(session_model_config, model_path,
                                       std::move(prompt_source),
                                       std::move(model_resources), stages);
  };
  return std::make_unique<LeasedText2ImageSessionFactory>(
      settings, session_config, resolved_metadata, std::move(resources),
      std::move(component_builder), std::move(lease_options));
}

[[maybe_unused]] const bool kZImageTurboRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetZImageTurboSessionFactory,
      .session_config_type_id = "Text2Image",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::text2image

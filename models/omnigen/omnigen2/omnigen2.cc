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

#include <atomic>
#include <cstdint>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/omnigen/omnigen2/omnigen2.pb.h"
#include "models/omnigen/omnigen2/omnigen2_factory.h"
#include "models/omnigen/omnigen2/omnigen2_model_config.h"
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/text2image_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"

namespace litert::omni::text2image::omnigen2 {
namespace {

constexpr absl::string_view kModelNames[] = {
    "omnigen2",
    "omnigen-2",
    "omnigen_2",
    "OmniGen2/OmniGen2",
};

absl::StatusOr<lm::proto::Text2ImageMetadata> ResolveMetadata(
    const MessageLite* absl_nullable metadata) {
  lm::proto::Text2ImageMetadata resolved;
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  if (resolved.image_gen_model_type().model_type_case() ==
      lm::proto::ImageGenModelType::MODEL_TYPE_NOT_SET) {
    prototypes::litert_models::omnigen2::OmniGen2 default_omnigen2;
    if (!resolved.mutable_image_gen_model_type()
             ->mutable_custom_model()
             ->PackFrom(default_omnigen2)) {
      return absl::InternalError(
          "Failed to pack default OmniGen2 proto into Text2ImageMetadata.");
    }
  }
  return resolved;
}

absl::Status PopulateOmniGen2ConfigFromMetadata(
    const lm::proto::Text2ImageMetadata& metadata,
    OmniGen2ModelConfig& config) {
  if (metadata.image_gen_model_type().model_type_case() ==
      lm::proto::ImageGenModelType::MODEL_TYPE_NOT_SET) {
    return absl::OkStatus();
  }
  if (!metadata.image_gen_model_type().has_custom_model()) {
    return absl::InvalidArgumentError(
        "Expected ImageGenModelType.custom_model packing "
        "prototypes.litert_models.omnigen2.OmniGen2.");
  }
  prototypes::litert_models::omnigen2::OmniGen2 proto;
  if (!metadata.image_gen_model_type().custom_model().UnpackTo(&proto)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to unpack ImageGenModelType.custom_model as "
        "prototypes.litert_models.omnigen2.OmniGen2 (got type_url='",
        metadata.image_gen_model_type().custom_model().type_url(), "')."));
  }
  PopulateOmniGen2ConfigFromProto(proto, config);
  return absl::OkStatus();
}

class OmniGen2Session : public MultiStagedSession {
 public:
  OmniGen2Session(std::vector<std::unique_ptr<internal::StageBase>> stages,
                  Stage<Output>* absl_nonnull output_stage,
                  ::litert::lm::ThreadPool* absl_nullable thread_pool,
                  std::shared_ptr<std::atomic<bool>> active_session)
      : MultiStagedSession(std::move(stages), output_stage, thread_pool),
        active_session_(std::move(active_session)) {}

  ~OmniGen2Session() override {
    Reset();
    if (active_session_ != nullptr) {
      active_session_->store(false, std::memory_order_release);
    }
  }

 private:
  std::shared_ptr<std::atomic<bool>> active_session_;
};

class OmniGen2SessionFactory : public Text2ImageSessionFactory {
 public:
  OmniGen2SessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Text2ImageMetadata& metadata,
      std::shared_ptr<ModelResources> absl_nullable model_resources,
      OmniGen2ModelConfig config)
      : Text2ImageSessionFactory(settings, session_config, metadata,
                                 std::move(model_resources)),
        config_(std::move(config)) {}

  ImageGenInputMetadata ResolveDefaultPromptParams(
      const lm::proto::Text2ImageSessionConfig& session_config) const override {
    ImageGenInputMetadata params{
        .width = session_config.has_width() ? session_config.width()
                                            : this->session_config().width(),
        .height = session_config.has_height() ? session_config.height()
                                              : this->session_config().height(),
        .num_inference_steps =
            session_config.has_num_inference_steps()
                ? session_config.num_inference_steps()
                : this->session_config().num_inference_steps(),
        .seed =
            session_config.has_seed()
                ? static_cast<uint64_t>(session_config.seed())
                : (this->session_config().has_seed()
                       ? static_cast<uint64_t>(this->session_config().seed())
                       : config_.default_seed),
    };
    if (params.width <= 0) {
      params.width = config_.img_size;
    }
    if (params.height <= 0) {
      params.height = config_.img_size;
    }
    if (params.num_inference_steps <= 0) {
      params.num_inference_steps = config_.steps;
    }
    return params;
  }

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithPromptSource(
      const lm::proto::Text2ImageSessionConfig& session_config,
      std::unique_ptr<PromptSource> absl_nullable prompt_source) override {
    if (model_resources() == nullptr) {
      return absl::FailedPreconditionError("ModelResources must not be null.");
    }

    bool expected = false;
    if (!active_session_->compare_exchange_strong(expected, true,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_relaxed)) {
      return absl::FailedPreconditionError(
          "Another OmniGen2 session is already active on this engine.");
    }
    bool lease_transferred = false;
    absl::Cleanup release_lease = [&] {
      if (!lease_transferred) {
        active_session_->store(false, std::memory_order_release);
      }
    };

    lm::proto::Text2ImageSessionConfig effective_config =
        this->session_config();
    effective_config.MergeFrom(session_config);
    if (prompt_source == nullptr) {
      prompt_source = std::make_unique<PushPromptSource>(
          ResolveDefaultPromptParams(effective_config));
    }

    OmniGen2ModelConfig session_model_config = config_;
    if (effective_config.width() > 0 || effective_config.height() > 0) {
      if (effective_config.width() != effective_config.height()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "OmniGen2 requires square image dimensions (width == height), got ",
            effective_config.width(), "x", effective_config.height()));
      }
      session_model_config.img_size = effective_config.width();
    }
    if (effective_config.num_inference_steps() > 0) {
      session_model_config.steps = effective_config.num_inference_steps();
    }
    if (effective_config.has_seed()) {
      session_model_config.default_seed = effective_config.seed();
    }

    std::vector<std::unique_ptr<internal::StageBase>> stages;
    ABSL_ASSIGN_OR_RETURN(
        Stage<Output> * output_stage,
        CreateOmniGen2Components(session_model_config, std::move(prompt_source),
                                 model_resources(), stages));
    if (output_stage == nullptr) {
      return absl::InternalError("Output stage must not be null.");
    }
    lease_transferred = true;
    return std::make_unique<OmniGen2Session>(std::move(stages), output_stage,
                                             thread_pool(), active_session_);
  }

 private:
  OmniGen2ModelConfig config_;
  std::shared_ptr<std::atomic<bool>> active_session_ =
      std::make_shared<std::atomic<bool>>(false);
};

absl::StatusOr<std::unique_ptr<OmniSessionFactory>> GetOmniGen2SessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata, ::litert::Environment& env,
    ::litert::Options& /*options*/,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }
  ABSL_ASSIGN_OR_RETURN(lm::proto::Text2ImageMetadata resolved_metadata,
                        ResolveMetadata(metadata));
  OmniGen2ModelConfig config;
  ABSL_RETURN_IF_ERROR(
      PopulateOmniGen2ConfigFromMetadata(resolved_metadata, config));
  ABSL_RETURN_IF_ERROR(ValidateOmniGen2ModelConfig(config));

  auto shared_env = std::make_shared<Environment>(std::move(env));
  const int num_threads = settings.num_threads > 0 ? settings.num_threads : 4;
  ABSL_ASSIGN_OR_RETURN(
      ModelResources model_resources,
      CreateOmniGen2Resources(config, settings.model_path, settings.cache_dir,
                              ToLmBackend(settings.backend), num_threads,
                              std::move(shared_env), std::move(lm_resources)));
  auto resources = std::make_shared<ModelResources>(std::move(model_resources));

  return std::make_unique<OmniGen2SessionFactory>(
      settings, session_config, resolved_metadata, std::move(resources),
      std::move(config));
}

[[maybe_unused]] const bool kOmniGen2Registered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetOmniGen2SessionFactory,
      .session_config_type_id = "Text2Image",
      .model_remote_url =
          "https://huggingface.co/litert-community/OmniGen2-LiteRT/"
          "resolve/main/omnigen2.litertlm",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::text2image::omnigen2

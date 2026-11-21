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

#include "models/deepseek/janus_pro/janus_pro.pb.h"
#include "models/deepseek/janus_pro/janus_pro_factory.h"
#include "models/deepseek/janus_pro/janus_pro_model_config.h"
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
#include "omni/any2any/any2any_session.h"
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/multi_staged_session.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/framework/threadpool.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"

namespace litert::omni::any2any {
namespace {

constexpr absl::string_view kModelNames[] = {
    "janus-pro-1b",
    "janus_pro_1b",
    "janus-pro",
    "janus_pro",
    "deepseek-ai/Janus-Pro-1B",
    "deepseek-ai/janus-pro-1b",
    "janus-pro-7b",
    "janus_pro_7b",
    "deepseek-ai/Janus-Pro-7B",
    "deepseek-ai/janus-pro-7b",
};

absl::StatusOr<lm::proto::Any2AnyMetadata> ResolveMetadata(
    const MessageLite* absl_nullable metadata) {
  lm::proto::Any2AnyMetadata resolved;
  if (metadata != nullptr) {
    resolved.CheckTypeAndMergeFrom(*metadata);
  }
  if (resolved.any2any_model_type().model_type_case() ==
      lm::proto::Any2AnyModelType::MODEL_TYPE_NOT_SET) {
    prototypes::litert_models::janus_pro::JanusPro default_janus_pro;
    if (!resolved.mutable_any2any_model_type()
             ->mutable_custom_model()
             ->PackFrom(default_janus_pro)) {
      return absl::InternalError(
          "Failed to pack default JanusPro proto into Any2AnyMetadata.");
    }
  }
  if (resolved.supported_languages().empty()) {
    resolved.add_supported_languages("en-US");
    resolved.add_supported_languages("zh-CN");
  }
  return resolved;
}

absl::Status PopulateJanusProConfigFromProto(
    const lm::proto::Any2AnyMetadata& metadata, JanusProModelConfig& config) {
  if (metadata.has_system_prompt() && !metadata.system_prompt().empty()) {
    config.system_prompt = metadata.system_prompt();
  }
  if (metadata.any2any_model_type().model_type_case() ==
      lm::proto::Any2AnyModelType::MODEL_TYPE_NOT_SET) {
    return absl::OkStatus();
  }
  if (!metadata.any2any_model_type().has_custom_model()) {
    return absl::InvalidArgumentError(
        "Expected Any2AnyModelType.custom_model packing "
        "prototypes.litert_models.janus_pro.JanusPro.");
  }
  prototypes::litert_models::janus_pro::JanusPro proto;
  if (!metadata.any2any_model_type().custom_model().UnpackTo(&proto)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to unpack Any2AnyModelType.custom_model as "
        "prototypes.litert_models.janus_pro.JanusPro (got type_url='",
        metadata.any2any_model_type().custom_model().type_url(), "')."));
  }
  if (proto.has_llm_hidden_size() && proto.llm_hidden_size() > 0) {
    config.llm_hidden_size = proto.llm_hidden_size();
  }
  if (proto.has_llm_vocab_size() && proto.llm_vocab_size() > 0) {
    config.llm_vocab_size = proto.llm_vocab_size();
  }
  if (proto.has_vq_codebook_size() && proto.vq_codebook_size() > 0) {
    config.vq_codebook_size = proto.vq_codebook_size();
  }
  if (proto.has_vq_codebook_embed_dim() && proto.vq_codebook_embed_dim() > 0) {
    config.vq_embed_dim = proto.vq_codebook_embed_dim();
  }
  if (proto.has_image_size() && proto.image_size() > 0) {
    config.image_size = proto.image_size();
  }
  if (proto.has_patch_size() && proto.patch_size() > 0) {
    config.vision_patch_size = proto.patch_size();
  }
  if (proto.has_num_image_tokens() && proto.num_image_tokens() > 0) {
    config.num_image_tokens = proto.num_image_tokens();
  }
  if (proto.has_cfg_weight() && proto.cfg_weight() > 0.0f) {
    config.cfg_weight = proto.cfg_weight();
  }
  if (proto.has_image_temperature()) {
    config.image_temperature = proto.image_temperature();
  }
  if (proto.has_generate_image()) {
    config.generate_image = proto.generate_image();
  }
  if (proto.has_max_new_tokens() && proto.max_new_tokens() > 0) {
    config.max_new_tokens = proto.max_new_tokens();
  }
  return absl::OkStatus();
}

void ApplySessionConfigToModelConfig(
    const lm::proto::Any2AnySessionConfig& session_config,
    JanusProModelConfig& config) {
  if (session_config.has_generate_image()) {
    config.generate_image = session_config.generate_image();
  }
  if (session_config.has_max_new_tokens() &&
      session_config.max_new_tokens() > 0) {
    config.max_new_tokens = session_config.max_new_tokens();
  }
  if (session_config.has_temperature()) {
    config.temperature = session_config.temperature();
    config.do_sample = session_config.temperature() > 0.0f;
  }
  if (session_config.has_top_p() && session_config.top_p() > 0.0f) {
    config.top_p = session_config.top_p();
  }
  if (session_config.has_top_k() && session_config.top_k() > 0) {
    config.top_k = session_config.top_k();
  }
  if (session_config.has_cfg_weight() && session_config.cfg_weight() > 0.0f) {
    config.cfg_weight = session_config.cfg_weight();
  }
  if (session_config.has_image_temperature()) {
    config.image_temperature = session_config.image_temperature();
  }
  if (session_config.has_seed()) {
    config.seed = static_cast<uint64_t>(session_config.seed());
  }
  if (session_config.has_system_prompt()) {
    config.system_prompt = session_config.system_prompt();
  }
}

class JanusProSession : public MultiStagedSession {
 public:
  JanusProSession(std::vector<std::unique_ptr<internal::StageBase>> stages,
                  Stage<Output>* absl_nonnull output_stage,
                  ::litert::lm::ThreadPool* absl_nullable thread_pool,
                  std::shared_ptr<std::atomic<bool>> active_session)
      : MultiStagedSession(std::move(stages), output_stage, thread_pool),
        active_session_(std::move(active_session)) {}

  ~JanusProSession() override {
    Reset();
    if (active_session_ != nullptr) {
      active_session_->store(false, std::memory_order_release);
    }
  }

 private:
  std::shared_ptr<std::atomic<bool>> active_session_;
};

class JanusProSessionFactory : public Any2AnySessionFactory {
 public:
  JanusProSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Any2AnyMetadata& metadata,
      std::vector<std::string> available_voices,
      std::shared_ptr<ModelResources> absl_nullable model_resources,
      JanusProModelConfig config = {})
      : Any2AnySessionFactory(settings, session_config, metadata,
                              std::move(available_voices),
                              std::move(model_resources)),
        config_(std::move(config)) {}

  std::string GetDefaultVoice(absl::string_view /*language*/) const override {
    return "";
  }

 protected:
  absl::StatusOr<std::unique_ptr<MultiStagedSession>> CreateWithInputSource(
      const lm::proto::Any2AnySessionConfig& session_config,
      std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source)
      override {
    if (model_resources() == nullptr) {
      return absl::FailedPreconditionError("ModelResources must not be null.");
    }

    bool expected = false;
    if (!active_session_->compare_exchange_strong(expected, true,
                                                  std::memory_order_acq_rel,
                                                  std::memory_order_relaxed)) {
      return absl::FailedPreconditionError(
          "Another Janus-Pro session is already active on this engine.");
    }
    bool lease_transferred = false;
    absl::Cleanup release_lease = [&] {
      if (!lease_transferred) {
        active_session_->store(false, std::memory_order_release);
      }
    };

    lm::proto::Any2AnySessionConfig effective =
        MergeSessionConfig(session_config);
    JanusProModelConfig session_model_config = config_;
    ApplySessionConfigToModelConfig(effective, session_model_config);

    if (input_source == nullptr) {
      input_source = std::make_unique<PushInputSource>();
    }

    std::vector<std::unique_ptr<internal::StageBase>> stages;
    ABSL_ASSIGN_OR_RETURN(
        Stage<Output> * output_stage,
        CreateJanusProComponents(session_model_config, std::move(input_source),
                                 model_resources(), stages));
    if (output_stage == nullptr) {
      return absl::InternalError("Output stage must not be null.");
    }
    lease_transferred = true;
    return std::make_unique<JanusProSession>(std::move(stages), output_stage,
                                             thread_pool(), active_session_);
  }

 private:
  JanusProModelConfig config_;
  std::shared_ptr<std::atomic<bool>> active_session_ =
      std::make_shared<std::atomic<bool>>(false);
};

absl::StatusOr<std::unique_ptr<OmniSessionFactory>> GetJanusProSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata, ::litert::Environment& env,
    ::litert::Options& /*options*/,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }

  ABSL_ASSIGN_OR_RETURN(lm::proto::Any2AnyMetadata resolved_metadata,
                        ResolveMetadata(metadata));
  JanusProModelConfig config;
  config.force_fp32_on_gpu = settings.force_fp32_on_gpu;
  ABSL_RETURN_IF_ERROR(
      PopulateJanusProConfigFromProto(resolved_metadata, config));
  if (session_config != nullptr) {
    lm::proto::Any2AnySessionConfig init_session_cfg;
    init_session_cfg.CheckTypeAndMergeFrom(*session_config);
    ApplySessionConfigToModelConfig(init_session_cfg, config);
  }

  auto shared_env = std::make_shared<Environment>(std::move(env));
  const int num_threads = settings.num_threads > 0 ? settings.num_threads : 4;
  ABSL_ASSIGN_OR_RETURN(
      ModelResources model_resources,
      CreateJanusProResources(config, settings.model_path, settings.cache_dir,
                              ToLmBackend(settings.backend), num_threads,
                              std::move(shared_env), std::move(lm_resources)));
  auto resources = std::make_shared<ModelResources>(std::move(model_resources));

  return std::make_unique<JanusProSessionFactory>(
      settings, session_config, resolved_metadata,
      /*available_voices=*/std::vector<std::string>{}, std::move(resources),
      std::move(config));
}

[[maybe_unused]] const bool kJanusProRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetJanusProSessionFactory,
      .session_config_type_id = "Any2Any",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::any2any

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
#include <string>
#include <utility>
#include <vector>

#include "models/deepseek/janus_pro/runner/janus_pro.pb.h"
#include "models/deepseek/janus_pro/runner/janus_pro_factory.h"
#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
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
#include "omni/any2any/any2any_session.h"
#include "omni/base/custom_model_utils.h"
#include "omni/base/io_types.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
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
  lm::proto::Any2AnyMetadata resolved = ResolveAny2AnyMetadata(metadata);
  ABSL_RETURN_IF_ERROR(EnsureDefaultCustomModel<lm::proto::JanusPro>(
      *resolved.mutable_any2any_model_type()));
  return resolved;
}

absl::Status PopulateJanusProConfigFromProto(
    const lm::proto::Any2AnyMetadata& metadata, JanusProModelConfig& config) {
  if (metadata.has_system_prompt() && !metadata.system_prompt().empty()) {
    config.system_prompt = metadata.system_prompt();
  }
  ABSL_ASSIGN_OR_RETURN(
      std::optional<lm::proto::JanusPro> maybe_proto,
      UnpackCustomModel<lm::proto::JanusPro>(metadata.any2any_model_type()));
  if (!maybe_proto.has_value()) {
    return absl::OkStatus();
  }
  const lm::proto::JanusPro& proto = *maybe_proto;
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

  LeasedAny2AnySessionOptions lease_options{
      .model_display_name = "Janus-Pro",
  };
  LeasedAny2AnySessionFactory::ComponentBuilder component_builder =
      [base_config = std::move(config)](
          const lm::proto::Any2AnySessionConfig& effective_config,
          std::unique_ptr<OmniSession::InputSource> absl_nonnull input_source,
          std::shared_ptr<ModelResources> absl_nonnull model_resources,
          std::vector<std::unique_ptr<internal::StageBase>>& stages)
      -> absl::StatusOr<Stage<Output>* absl_nonnull> {
    JanusProModelConfig session_model_config = base_config;
    ApplySessionConfigToModelConfig(effective_config, session_model_config);
    return CreateJanusProComponents(session_model_config,
                                    std::move(input_source),
                                    std::move(model_resources), stages);
  };

  return std::make_unique<LeasedAny2AnySessionFactory>(
      settings, session_config, resolved_metadata,
      /*available_voices=*/std::vector<std::string>{}, std::move(resources),
      std::move(component_builder), std::move(lease_options));
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

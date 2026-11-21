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
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "models/lfm/lfm2_5_audio/lfm2_audio.pb.h"
#include "models/lfm/lfm2_5_audio/lfm2_audio_factory.h"
#include "models/lfm/lfm2_5_audio/lfm2_audio_model_config.h"
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
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
    "lfm2.5-audio-1.5b",
    "LiquidAI/LFM2.5-Audio-1.5B",
    "lfm2_audio",
    "lfm2-audio",
};

absl::StatusOr<lm::proto::Any2AnyMetadata> ResolveMetadata(
    lm::ModelResources* absl_nullable lm_resources,
    const MessageLite* absl_nullable metadata) {
  lm::proto::Any2AnyMetadata resolved;
  if (lm_resources != nullptr) {
    absl::StatusOr<const lm::proto::Any2AnyMetadata*> embedded =
        lm_resources->GetAny2AnyMetadata();
    if (embedded.ok() && *embedded != nullptr) {
      resolved = **embedded;
    }
  }
  if (metadata != nullptr) {
    lm::proto::Any2AnyMetadata override_meta;
    override_meta.CheckTypeAndMergeFrom(*metadata);
    if (override_meta.any2any_model_type().has_custom_model()) {
      resolved.mutable_any2any_model_type()->clear_custom_model();
    }
    if (!override_meta.supported_languages().empty()) {
      resolved.clear_supported_languages();
    }
    resolved.MergeFrom(override_meta);
  }
  if (resolved.any2any_model_type().model_type_case() ==
      lm::proto::Any2AnyModelType::MODEL_TYPE_NOT_SET) {
    prototypes::litert_models::lfm2_audio::Lfm2Audio default_lfm2_audio;
    if (!resolved.mutable_any2any_model_type()
             ->mutable_custom_model()
             ->PackFrom(default_lfm2_audio)) {
      return absl::InternalError(
          "Failed to pack default Lfm2Audio proto into Any2AnyMetadata.");
    }
  }
  if (resolved.supported_languages().empty()) {
    resolved.add_supported_languages("en-US");
  }
  return resolved;
}

absl::Status PopulateLfm2AudioConfigFromProto(
    const lm::proto::Any2AnyMetadata& metadata, Lfm2AudioModelConfig& config) {
  if (metadata.has_input_sample_rate_hz() &&
      metadata.input_sample_rate_hz() > 0) {
    config.audio_input_sample_rate_hz = metadata.input_sample_rate_hz();
  }
  if (metadata.has_output_sample_rate_hz() &&
      metadata.output_sample_rate_hz() > 0) {
    config.audio_output_sample_rate_hz = metadata.output_sample_rate_hz();
  }
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
        "prototypes.litert_models.lfm2_audio.Lfm2Audio.");
  }
  prototypes::litert_models::lfm2_audio::Lfm2Audio proto;
  if (!metadata.any2any_model_type().custom_model().UnpackTo(&proto)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to unpack Any2AnyModelType.custom_model as "
        "prototypes.litert_models.lfm2_audio.Lfm2Audio (got type_url='",
        metadata.any2any_model_type().custom_model().type_url(), "')."));
  }
  if (proto.has_llm_hidden_size() && proto.llm_hidden_size() > 0) {
    config.llm_hidden_size = proto.llm_hidden_size();
  }
  if (proto.has_llm_vocab_size() && proto.llm_vocab_size() > 0) {
    config.llm_vocab_size = proto.llm_vocab_size();
  }
  if (proto.has_tts_hidden_size() && proto.tts_hidden_size() > 0) {
    config.tts_hidden_size = proto.tts_hidden_size();
  }
  if (proto.has_tts_vocab_size() && proto.tts_vocab_size() > 0) {
    config.tts_vocab_size = proto.tts_vocab_size();
  }
  if (proto.has_tts_num_codebooks() && proto.tts_num_codebooks() > 0) {
    config.tts_num_codebooks = proto.tts_num_codebooks();
  }
  if (proto.has_audio_mel_bins() && proto.audio_mel_bins() > 0) {
    config.audio_mel_bins = proto.audio_mel_bins();
  }
  if (proto.has_audio_subsampling_factor() &&
      proto.audio_subsampling_factor() > 0) {
    config.audio_subsampling_factor = proto.audio_subsampling_factor();
  }
  if (proto.has_generate_audio()) {
    config.generate_audio = proto.generate_audio();
  }
  if (proto.has_max_new_tokens() && proto.max_new_tokens() > 0) {
    config.max_new_tokens = proto.max_new_tokens();
  }
  if (proto.has_max_tts_tokens() && proto.max_tts_tokens() > 0) {
    config.max_tts_tokens = proto.max_tts_tokens();
  }
  if (proto.has_voice() && !proto.voice().empty()) {
    config.voice_name = proto.voice();
  }
  if (proto.has_language() && !proto.language().empty()) {
    config.language = proto.language();
  }
  return absl::OkStatus();
}

void ApplySessionConfigToModelConfig(
    const lm::proto::Any2AnySessionConfig& session_config,
    Lfm2AudioModelConfig& config) {
  if (session_config.has_generate_audio()) {
    config.generate_audio = session_config.generate_audio();
  }
  if (session_config.has_max_new_tokens() &&
      session_config.max_new_tokens() > 0) {
    config.max_new_tokens = session_config.max_new_tokens();
  }
  if (session_config.has_max_tts_tokens() &&
      session_config.max_tts_tokens() > 0) {
    config.max_tts_tokens = session_config.max_tts_tokens();
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
  if (session_config.has_tts_temperature()) {
    config.tts_temperature = session_config.tts_temperature();
  }
  if (session_config.has_tts_top_p() && session_config.tts_top_p() > 0.0f) {
    config.tts_top_p = session_config.tts_top_p();
  }
  if (session_config.has_tts_top_k() && session_config.tts_top_k() > 0) {
    config.tts_top_k = session_config.tts_top_k();
  }
  if (session_config.has_system_prompt()) {
    config.system_prompt = session_config.system_prompt();
  }
  if (session_config.has_voice() && !session_config.voice().empty()) {
    config.voice_name = session_config.voice();
  }
  if (session_config.has_language() && !session_config.language().empty()) {
    config.language = session_config.language();
  }
}

class Lfm2AudioSession : public MultiStagedSession {
 public:
  Lfm2AudioSession(std::vector<std::unique_ptr<internal::StageBase>> stages,
                   Stage<Output>* absl_nonnull output_stage,
                   ::litert::lm::ThreadPool* absl_nullable thread_pool,
                   std::shared_ptr<std::atomic<bool>> active_session)
      : MultiStagedSession(std::move(stages), output_stage, thread_pool),
        active_session_(std::move(active_session)) {}

  ~Lfm2AudioSession() override {
    Reset();
    if (active_session_ != nullptr) {
      active_session_->store(false, std::memory_order_release);
    }
  }

 private:
  std::shared_ptr<std::atomic<bool>> active_session_;
};

class Lfm2AudioSessionFactory : public Any2AnySessionFactory {
 public:
  Lfm2AudioSessionFactory(
      const OmniEngine::Settings& settings,
      const MessageLite* absl_nullable session_config,
      const lm::proto::Any2AnyMetadata& metadata,
      std::vector<std::string> available_voices,
      std::shared_ptr<ModelResources> absl_nullable model_resources,
      Lfm2AudioModelConfig config = {})
      : Any2AnySessionFactory(settings, session_config, metadata,
                              std::move(available_voices),
                              std::move(model_resources)),
        config_(std::move(config)) {}

  std::string GetDefaultVoice(absl::string_view /*language*/) const override {
    if (!available_voices().empty()) {
      return available_voices().front();
    }
    return config_.voice_name;
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
          "Another LFM2.5-Audio session is already active on this engine.");
    }
    bool lease_transferred = false;
    absl::Cleanup release_lease = [&] {
      if (!lease_transferred) {
        active_session_->store(false, std::memory_order_release);
      }
    };

    lm::proto::Any2AnySessionConfig effective =
        MergeSessionConfig(session_config);
    Lfm2AudioModelConfig session_model_config = config_;
    ApplySessionConfigToModelConfig(effective, session_model_config);

    if (input_source == nullptr) {
      input_source = std::make_unique<PushInputSource>();
    }

    std::vector<std::unique_ptr<internal::StageBase>> stages;
    ABSL_ASSIGN_OR_RETURN(
        Stage<Output> * output_stage,
        CreateLfm2AudioComponents(session_model_config, std::move(input_source),
                                  model_resources(), stages));
    if (output_stage == nullptr) {
      return absl::InternalError("Output stage must not be null.");
    }
    lease_transferred = true;
    return std::make_unique<Lfm2AudioSession>(std::move(stages), output_stage,
                                              thread_pool(), active_session_);
  }

 private:
  Lfm2AudioModelConfig config_;
  std::shared_ptr<std::atomic<bool>> active_session_ =
      std::make_shared<std::atomic<bool>>(false);
};

absl::StatusOr<std::unique_ptr<OmniSessionFactory>> GetLfm2AudioSessionFactory(
    absl::string_view model_name, const OmniEngine::Settings& settings,
    const MessageLite* absl_nullable session_config,
    const MessageLite* absl_nullable metadata, ::litert::Environment& env,
    ::litert::Options& /*options*/,
    std::shared_ptr<lm::ModelResources> absl_nullable lm_resources) {
  if (!absl::c_linear_search(kModelNames, model_name)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected model_name: ", model_name));
  }
  if (lm_resources == nullptr &&
      absl::EndsWithIgnoreCase(settings.model_path, ".litertlm")) {
    ABSL_ASSIGN_OR_RETURN(lm_resources,
                          CreateLmModelResources(settings.model_path));
  }

  ABSL_ASSIGN_OR_RETURN(lm::proto::Any2AnyMetadata resolved_metadata,
                        ResolveMetadata(lm_resources.get(), metadata));
  Lfm2AudioModelConfig config;
  config.force_fp32_on_gpu = settings.force_fp32_on_gpu;
  ABSL_RETURN_IF_ERROR(
      PopulateLfm2AudioConfigFromProto(resolved_metadata, config));
  if (session_config != nullptr) {
    lm::proto::Any2AnySessionConfig init_session_cfg;
    init_session_cfg.CheckTypeAndMergeFrom(*session_config);
    ApplySessionConfigToModelConfig(init_session_cfg, config);
  }

  auto shared_env = std::make_shared<Environment>(std::move(env));
  const int num_threads = settings.num_threads > 0 ? settings.num_threads : 4;
  ABSL_ASSIGN_OR_RETURN(
      ModelResources model_resources,
      CreateLfm2AudioResources(config, settings.model_path, settings.cache_dir,
                               ToLmBackend(settings.backend), num_threads,
                               std::move(shared_env), std::move(lm_resources)));
  auto resources = std::make_shared<ModelResources>(std::move(model_resources));

  std::vector<std::string> available_voices;
  if (!config.voice_name.empty()) {
    available_voices.push_back(config.voice_name);
  }
  for (absl::string_view preset_voice :
       {"default", "us_male", "us_female", "uk_male", "uk_female"}) {
    if (!absl::c_linear_search(available_voices, preset_voice)) {
      available_voices.emplace_back(preset_voice);
    }
  }

  return std::make_unique<Lfm2AudioSessionFactory>(
      settings, session_config, resolved_metadata, std::move(available_voices),
      std::move(resources), std::move(config));
}

[[maybe_unused]] const bool kLfm2AudioRegistered = [] {
  ABSL_CHECK_OK(OmniRegistry::Get().RegisterSessionFactory(RegistrySpec{
      .model_names = {std::begin(kModelNames), std::end(kModelNames)},
      .get_session_factory = &GetLfm2AudioSessionFactory,
      .session_config_type_id = "Any2Any",
  }));
  return true;
}();

}  // namespace
}  // namespace litert::omni::any2any

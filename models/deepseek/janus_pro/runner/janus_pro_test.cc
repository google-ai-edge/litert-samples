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
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "models/deepseek/janus_pro/runner/janus_pro.pb.h"
#include "models/deepseek/janus_pro/runner/janus_pro_factory.h"
#include "models/deepseek/janus_pro/runner/janus_pro_model_config.h"
#include "models/deepseek/janus_pro/runner/janus_pro_test_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_expected.h"
#include "omni/any2any/any2any_session.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_lm_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "runtime/components/model_resources.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/any2any_metadata.pb.h"
#include "runtime/proto/any2any_model_type.pb.h"
#include "runtime/proto/any2any_session_config.pb.h"
#include "runtime/proto/llm_metadata.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep

namespace litert::omni::any2any {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::IsSupersetOf;
using ::testing::Not;

TEST(JanusProFactoryTest, CreateJanusProResourcesRejectsInvalidInputs) {
  JanusProModelConfig config;
  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<Environment>(std::move(*env));

  std::shared_ptr<Environment> null_env;
  EXPECT_THAT(
      CreateJanusProResources(config, "/tmp/nonexistent_janus_pro.litertlm", "",
                              lm::Backend::CPU, /*num_threads=*/1, null_env),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(
      CreateJanusProResources(config, "/tmp/not_a_litertlm_bundle_dir", "",
                              lm::Backend::CPU, /*num_threads=*/1, shared_env),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Janus-Pro requires a .litertlm model file")));
  EXPECT_THAT(
      CreateJanusProResources(config, "/tmp/nonexistent_janus_pro.litertlm", "",
                              lm::Backend::CPU, /*num_threads=*/1, shared_env),
      Not(IsOk()));
}

TEST(JanusProFactoryTest,
     CreateJanusProResourcesRetainsEnvDelegatesBackboneAndCompilesSections) {
  JanusProTestBundleOptions full_opts;
  full_opts.include_vision_encoder = true;
  full_opts.include_image_gen_models = true;
  ASSERT_OK_AND_ASSIGN(
      std::string full_bundle_path,
      WriteJanusProBundle("janus_pro_factory_full.litertlm", full_opts));

  JanusProTestBundleOptions und_only_opts;
  und_only_opts.include_vision_encoder = true;
  und_only_opts.include_image_gen_models = false;
  ASSERT_OK_AND_ASSIGN(
      std::string und_only_bundle_path,
      WriteJanusProBundle("janus_pro_factory_und_only.litertlm",
                          und_only_opts));

  Expected<Environment> env = Environment::Create({});
  ASSERT_TRUE(env.HasValue());
  auto shared_env = std::make_shared<Environment>(std::move(*env));

  // 1. Full bundle compiles all sections, retains `env`, and delegates
  //    `BackboneLmModelResources`.
  {
    JanusProModelConfig config;
    config.generate_image = false;
    ASSERT_OK_AND_ASSIGN(ModelResources resources,
                         CreateJanusProResources(
                             config, full_bundle_path, ":nocache",
                             lm::Backend::CPU, /*num_threads=*/1, shared_env));
    EXPECT_EQ(resources.environment(), shared_env);
    EXPECT_TRUE(resources.HasLmModelResources());
    EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
    EXPECT_TRUE(resources.HasCompiledModel("vision_encoder"));
    EXPECT_TRUE(resources.HasCompiledModel("vq_image_embedder"));
    EXPECT_TRUE(resources.HasCompiledModel("gen_head"));
    EXPECT_TRUE(resources.HasCompiledModel("vq_image_decoder"));
    EXPECT_TRUE(resources.HasLmRunner("llm_backbone"));
    EXPECT_TRUE(resources.HasLmRunner("uncond_llm_backbone"));

    ASSERT_OK_AND_ASSIGN(std::shared_ptr<LiteRtLmRunner> llm_runner,
                         resources.GetLmRunner("llm_backbone"));
    lm::ModelResources* backbone_res = llm_runner->mutable_model_resources();
    ASSERT_NE(backbone_res, nullptr);
    using lm::proto::LlmMetadata;
    EXPECT_TRUE(
        backbone_res->HasTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE));
    EXPECT_TRUE(backbone_res->HasTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER));
    EXPECT_THAT(
        backbone_res->GetTFLiteModel(LlmMetadata::TF_LITE_PREFILL_DECODE),
        IsOk());
    EXPECT_THAT(backbone_res->GetTFLiteModel(LlmMetadata::TF_LITE_EMBEDDER),
                IsOk());
  }

  // 2. Understanding-only bundle omits image-gen sections when
  //    generate_image=false, and returns kNotFound when generate_image=true.
  {
    JanusProModelConfig und_cfg;
    und_cfg.generate_image = false;
    ASSERT_OK_AND_ASSIGN(ModelResources resources,
                         CreateJanusProResources(
                             und_cfg, und_only_bundle_path, ":nocache",
                             lm::Backend::CPU, /*num_threads=*/1, shared_env));
    EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
    EXPECT_TRUE(resources.HasCompiledModel("vision_encoder"));
    EXPECT_FALSE(resources.HasCompiledModel("vq_image_embedder"));
    EXPECT_FALSE(resources.HasCompiledModel("gen_head"));
    EXPECT_FALSE(resources.HasCompiledModel("vq_image_decoder"));
    EXPECT_FALSE(resources.HasLmRunner("uncond_llm_backbone"));

    JanusProModelConfig gen_cfg;
    gen_cfg.generate_image = true;
    EXPECT_THAT(CreateJanusProResources(gen_cfg, und_only_bundle_path,
                                        ":nocache", lm::Backend::CPU,
                                        /*num_threads=*/1, shared_env),
                StatusIs(absl::StatusCode::kNotFound));
  }

  // 3. Pre-populated `lm::ModelResources` is reused directly without reopening
  //    `model_path`.
  {
    ASSERT_OK_AND_ASSIGN(std::shared_ptr<lm::ModelResources> lm_resources,
                         CreateLmModelResources(und_only_bundle_path));
    JanusProModelConfig config;
    config.generate_image = false;
    ASSERT_OK_AND_ASSIGN(
        ModelResources resources,
        CreateJanusProResources(config, /*model_path=*/"", ":nocache",
                                lm::Backend::CPU, /*num_threads=*/1, shared_env,
                                lm_resources));
    EXPECT_EQ(resources.environment(), shared_env);
    EXPECT_EQ(resources.GetLmModelResources(), lm_resources);
    EXPECT_TRUE(resources.HasCompiledModel("text_embedder"));
    EXPECT_THAT(resources.GetLmRunner("llm_backbone"), IsOk());
  }
}

TEST(JanusProSessionTest, RegistersAllModelNamesInOmniRegistry) {
  constexpr absl::string_view kExpectedNames[] = {
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
  EXPECT_THAT(OmniRegistry::Get().GetRegisteredModels(),
              IsSupersetOf(kExpectedNames));
  for (absl::string_view name : kExpectedNames) {
    EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered(name)) << name;
    ASSERT_OK_AND_ASSIGN(OmniRegistry::RegisteredModel reg,
                         OmniRegistry::Get().GetRegisteredModel(name));
    EXPECT_EQ(reg.canonical_model_name, "janus-pro-1b");
    EXPECT_EQ(reg.session_config_proto_type_id,
              "litert.lm.proto.Any2AnySessionConfig");
    EXPECT_EQ(reg.metadata_proto_type_id, "litert.lm.proto.Any2AnyMetadata");
  }
}

TEST(JanusProSessionTest,
     RunsUnderstandingEnforcesSingleActiveSessionLeaseAndReleasesOnDestroy) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteJanusProBundle("janus_pro_understanding.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  lm::proto::Any2AnySessionConfig init_cfg;
  init_cfg.set_generate_image(false);
  init_cfg.set_max_new_tokens(2);
  init_cfg.set_temperature(0.0f);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniEngine> engine,
                       OmniEngine::Create("janus-pro-1b", settings, &init_cfg));

  {
    auto input_source = std::make_unique<PushInputSource>();
    PushInputSource* raw_input = input_source.get();
    ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                         engine->CreateSession(std::move(input_source)));

    // Second concurrent session on the same engine fails with
    // kFailedPrecondition.
    EXPECT_THAT(
        engine->CreateSession(std::make_unique<PushInputSource>()),
        StatusIs(absl::StatusCode::kFailedPrecondition,
                 HasSubstr("Another Janus-Pro session is already active")));

    ASSERT_OK_AND_ASSIGN(
        std::vector<uint8_t> png_bytes,
        EncodePngBytes(std::vector<uint8_t>(4 * 4 * 3, 180), /*width=*/4,
                       /*height=*/4, /*channels=*/3));
    ASSERT_OK(raw_input->PushInput(
        OmniSession::ImageInput{.image_bytes = std::move(png_bytes)}));
    ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "Describe"}));
    ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

    ASSERT_OK_AND_ASSIGN(OmniSession::Output text_out, session->ProcessNext());
    EXPECT_TRUE(std::holds_alternative<TextOutput>(text_out));
    ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session->ProcessNext());
    EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));
  }

  // Once the first session is destroyed, its lease is released and a new
  // session succeeds.
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSession> second_session,
      engine->CreateSession(std::make_unique<PushInputSource>()));
  EXPECT_NE(second_session, nullptr);
}

TEST(JanusProSessionTest, RunsNonMockedVisualGenerationEndToEndViaOmniEngine) {
  // Split the 32-element decode output of `checksum_fake_lm_embedding_input`
  // into 28 vocab logits + 4 hidden-state dimensions so visual generation
  // extracts real hidden states without mocks.
  JanusProTestBundleOptions gen_opts;
  gen_opts.llm_vocab_size = 28;
  gen_opts.llm_hidden_size = 4;
  gen_opts.vq_codebook_size = 8;
  gen_opts.image_size = 16;
  gen_opts.patch_size = 8;
  gen_opts.num_image_tokens = 4;
  gen_opts.generate_image = true;
  gen_opts.cfg_weight = 5.0f;
  gen_opts.image_temperature = 0.0f;
  gen_opts.vq_decoder_r = 1.0f;
  gen_opts.vq_decoder_g = 0.0f;
  gen_opts.vq_decoder_b = -1.0f;
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteJanusProBundle("janus_pro_visual_gen.litertlm", gen_opts));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  lm::proto::Any2AnySessionConfig init_cfg;
  init_cfg.set_generate_image(true);
  init_cfg.set_cfg_weight(5.0f);
  init_cfg.set_image_temperature(0.0f);
  init_cfg.set_seed(42);

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniEngine> engine,
      OmniEngine::Create("deepseek-ai/Janus-Pro-1B", settings, &init_cfg));

  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                       engine->CreateSession(std::move(input_source)));

  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "fox"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));

  ASSERT_OK_AND_ASSIGN(OmniSession::Output img_out, session->ProcessNext());
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(img_out));
  const auto& img = std::get<ImageOutput>(img_out);
  EXPECT_EQ(img.width, 16);
  EXPECT_EQ(img.height, 16);
  EXPECT_EQ(img.channels, 3);
  ASSERT_EQ(img.rgb_data.size(), 16u * 16u * 3u);
  EXPECT_EQ(img.rgb_data[0], 255);
  EXPECT_EQ(img.rgb_data[1], 127);
  EXPECT_EQ(img.rgb_data[2], 0);

  ASSERT_OK_AND_ASSIGN(OmniSession::Output end_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<OmniSession::EndOfOutputTurn>(end_out));
}

TEST(JanusProSessionTest, DefaultsLanguageAndCustomModelWhenOmitted) {
  lm::proto::Any2AnyMetadata bare_meta;
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteJanusProBundle("janus_pro_defaults_test.litertlm", &bare_meta));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  lm::proto::Any2AnySessionConfig init_cfg;
  init_cfg.set_generate_image(false);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSessionFactory> base_factory,
                       OmniRegistry::Get().CreateSessionFactory(
                           "janus-pro-1b", settings, &init_cfg));
  auto* factory = dynamic_cast<Any2AnySessionFactory*>(base_factory.get());
  ASSERT_NE(factory, nullptr);
  EXPECT_THAT(factory->GetAvailableLanguages(), ElementsAre("en-US", "zh-CN"));
  EXPECT_EQ(factory->GetDefaultVoice("en-US"), "");
  ASSERT_TRUE(factory->metadata().any2any_model_type().has_custom_model());
  lm::proto::JanusPro unpacked;
  EXPECT_TRUE(factory->metadata().any2any_model_type().custom_model().UnpackTo(
      &unpacked));
}

TEST(JanusProSessionTest,
     MetadataOverridesPreserveCustomModelAndDoNotDuplicateLanguages) {
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteJanusProBundle("janus_pro_metadata_overrides.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  // 1. Scalar-only metadata override preserves embedded `custom_model` AND does
  //    not duplicate repeated `supported_languages` from single-merge
  //    `ResolveMetadata`.
  lm::proto::Any2AnyMetadata scalar_override;
  scalar_override.set_system_prompt("Custom system prompt.");
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSessionFactory> scalar_base_factory,
      OmniRegistry::Get().CreateSessionFactory(
          "janus_pro", settings, /*session_config=*/nullptr, &scalar_override));
  auto* scalar_factory =
      dynamic_cast<Any2AnySessionFactory*>(scalar_base_factory.get());
  ASSERT_NE(scalar_factory, nullptr);
  EXPECT_THAT(scalar_factory->GetAvailableLanguages(),
              ElementsAre("en-US", "zh-CN"));

  // 2. `custom_model` metadata override replaces embedded `custom_model`
  //    (enabling visual generation with llm_vocab_size=28 on a bundle whose
  //    embedded metadata had generate_image=false and llm_vocab_size=32).
  lm::proto::Any2AnyMetadata custom_override;
  lm::proto::JanusPro override_proto;
  override_proto.set_llm_hidden_size(4);
  override_proto.set_llm_vocab_size(28);
  override_proto.set_vq_codebook_size(8);
  override_proto.set_vq_codebook_embed_dim(4);
  override_proto.set_image_size(16);
  override_proto.set_patch_size(8);
  override_proto.set_num_image_tokens(4);
  override_proto.set_generate_image(true);
  override_proto.set_cfg_weight(5.0f);
  override_proto.set_image_temperature(0.0f);
  ASSERT_TRUE(custom_override.mutable_any2any_model_type()
                  ->mutable_custom_model()
                  ->PackFrom(override_proto));

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniSessionFactory> custom_base_factory,
      OmniRegistry::Get().CreateSessionFactory(
          "janus-pro", settings, /*session_config=*/nullptr, &custom_override));
  auto input_source = std::make_unique<PushInputSource>();
  PushInputSource* raw_input = input_source.get();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                       custom_base_factory->Create(std::move(input_source)));
  ASSERT_OK(raw_input->PushInput(OmniSession::TextInput{.text = "fox"}));
  ASSERT_OK(raw_input->PushInput(OmniSession::EndOfInputTurn{}));
  ASSERT_OK_AND_ASSIGN(OmniSession::Output img_out, session->ProcessNext());
  EXPECT_TRUE(std::holds_alternative<ImageOutput>(img_out));
}

TEST(JanusProSessionTest,
     RejectsWrongCustomModelProtoTypeAndNonCustomModelType) {
  // 1. Embedded `custom_model` packing an unexpected proto type (`MiniCpmO`).
  lm::proto::Any2AnyMetadata wrong_any_meta;
  lm::proto::MiniCpmO wrong_payload;
  wrong_payload.set_llm_hidden_size(4);
  ASSERT_TRUE(wrong_any_meta.mutable_any2any_model_type()
                  ->mutable_custom_model()
                  ->PackFrom(wrong_payload));
  ASSERT_OK_AND_ASSIGN(
      std::string wrong_any_bundle,
      WriteJanusProBundle("janus_pro_wrong_any.litertlm", &wrong_any_meta));

  OmniEngine::Settings settings;
  settings.model_path = wrong_any_bundle;
  settings.cache_dir = ":nocache";
  settings.backend = OmniEngine::Settings::Backend::kCpu;
  settings.num_threads = 1;

  EXPECT_THAT(
      OmniRegistry::Get().CreateSessionFactory("janus-pro-1b", settings),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(OmniEngine::Create("janus-pro-1b", settings),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // 2. Metadata override packing an unexpected proto type into `custom_model`.
  ASSERT_OK_AND_ASSIGN(
      std::string valid_bundle,
      WriteJanusProBundle("janus_pro_valid_for_override.litertlm"));
  settings.model_path = valid_bundle;
  EXPECT_THAT(OmniRegistry::Get().CreateSessionFactory(
                  "janus-pro-1b", settings, /*session_config=*/nullptr,
                  &wrong_any_meta),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // 3. Embedded non-custom `any2any_model_type` oneof branch (`minicpm_o`).
  lm::proto::Any2AnyMetadata non_custom_meta;
  non_custom_meta.mutable_any2any_model_type()->mutable_minicpm_o();
  ASSERT_OK_AND_ASSIGN(
      std::string non_custom_bundle,
      WriteJanusProBundle("janus_pro_non_custom.litertlm", &non_custom_meta));
  settings.model_path = non_custom_bundle;
  EXPECT_THAT(
      OmniRegistry::Get().CreateSessionFactory("janus-pro-1b", settings),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::any2any

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

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "models/omnigen/omnigen2/runner/omnigen2.pb.h"
#include "models/omnigen/omnigen2/runner/omnigen2_factory.h"
#include "models/omnigen/omnigen2/runner/omnigen2_model_config.h"
#include "models/omnigen/omnigen2/runner/omnigen2_test_utils.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_layout.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"
#include "omni/any2any/any2any_test_utils.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/mock_litert_runner.h"
#include "omni/base/model_resources.h"
#include "omni/base/model_utils.h"
#include "omni/base/stage.h"
#include "omni/omni_engine.h"
#include "omni/omni_registry.h"
#include "omni/omni_session.h"
#include "omni/text2image/io_types.h"
#include "omni/text2image/prompt_source.h"
#include "omni/text2image/vae_decoder_stage.h"
#include "runtime/executor/executor_settings_base.h"
#include "runtime/proto/image_gen_model_type.pb.h"
#include "runtime/proto/text2image_metadata.pb.h"
#include "runtime/proto/text2image_session_config.pb.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK
#include "tflite/schema/schema_generated.h"

namespace litert::omni::text2image::omnigen2 {
namespace {

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::FloatNear;
using ::testing::HasSubstr;
using ::testing::IsEmpty;

class StubDenoiserStage : public SingleThreadedStageWithDeque<DenoisedLatents> {
 public:
  void Push(DenoisedLatents output) { PushOutput(std::move(output)); }

 protected:
  bool NeedScheduleInternal() const override { return false; }
  absl::Status ScheduleInternal() override { return absl::OkStatus(); }
};

absl::StatusOr<TensorBuffer> CreateHostFloatBuffer(
    Environment& env, const std::vector<int>& dims) {
  RankedTensorType type(ElementType::Float32,
                        Layout(Dimensions(dims.begin(), dims.end())));
  size_t num_elements = 1;
  for (int d : dims) num_elements *= static_cast<size_t>(d);
  LITERT_ASSIGN_OR_RETURN(
      TensorBuffer buf, TensorBuffer::CreateManaged(
                            env, TensorBufferType::kHostMemory, std::move(type),
                            num_elements * sizeof(float)));
  return buf;
}

OmniGen2ModelConfig MakeSmallModelConfig() {
  OmniGen2ModelConfig config;
  config.seq_len = 32;
  config.img_size = 32;
  config.in_channels = 16;
  config.patch_size = 2;
  config.packed_ch = 64;
  config.text_hidden_size = 8;
  config.dit_hidden_size = 48;
  config.dit_num_heads = 3;
  config.dit_num_kv_heads = 1;
  config.dit_head_dim = 16;
  config.axes_dims_rope = {4, 6, 6};
  config.steps = 2;
  config.text_guidance_scale = 2.0f;
  return config;
}

TEST(OmniGen2RegistryTest, RegistersAllAliasesAndMatchesFilename) {
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("omnigen2"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("omnigen-2"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("omnigen_2"));
  EXPECT_TRUE(OmniRegistry::Get().IsModelRegistered("OmniGen2/OmniGen2"));

  ASSERT_OK_AND_ASSIGN(OmniRegistry::RegisteredModel reg,
                       OmniRegistry::Get().GetRegisteredModel("omnigen2"));
  EXPECT_EQ(reg.canonical_model_name, "omnigen2");
  EXPECT_EQ(reg.session_config_proto_type_id,
            "litert.lm.proto.Text2ImageSessionConfig");
  EXPECT_EQ(reg.metadata_proto_type_id, "litert.lm.proto.Text2ImageMetadata");
  EXPECT_THAT(reg.model_remote_url, IsEmpty());

  ASSERT_OK_AND_ASSIGN(
      std::string matched,
      OmniRegistry::Get().FindModelNameForPath("/tmp/omnigen2_fp32.litertlm"));
  EXPECT_THAT(matched, HasSubstr("omnigen"));
}

TEST(OmniGen2FactoryTest,
     CreateOmniGen2ResourcesLoadsBothTfLiteAndGenericBinaryShards) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteOmniGen2Bundle("omnigen2_resources_ok.litertlm"));
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::make_shared<Environment>(std::move(*env_or));

  ASSERT_OK_AND_ASSIGN(
      ModelResources resources,
      CreateOmniGen2Resources(MakeSmallModelConfig(), bundle_path, ":nocache",
                              lm::Backend::CPU, /*num_threads=*/1, env));

  EXPECT_EQ(resources.environment(), env);
  EXPECT_TRUE(resources.HasLmModelResources());
  EXPECT_TRUE(resources.HasCompiledModel("omnigen2_textenc"));
  EXPECT_TRUE(resources.HasCompiledModel("omnigen2_dit_cap_embedder"));
  EXPECT_TRUE(resources.HasCompiledModel("omnigen2_dit_context_refiner"));
  EXPECT_TRUE(resources.HasCompiledModel("omnigen2_dit_x_embedder"));
  EXPECT_TRUE(resources.HasCompiledModel("omnigen2_dit_noise_refiner"));
  for (size_t i = 0; i < kNumMainBlockShards; ++i) {
    EXPECT_TRUE(
        resources.HasCompiledModel(absl::StrCat("omnigen2_dit_main_", i)))
        << "Missing main shard " << i;
  }
  EXPECT_TRUE(resources.HasCompiledModel("omnigen2_dit_final"));
  EXPECT_TRUE(resources.HasCompiledModel("omnigen2_vae"));

  // Confirm all 6 GenericBinaryData shards are accessible via
  // `GetGenericBinaryDataBuffer`.
  auto lm_res = resources.GetLmModelResources();
  ASSERT_NE(lm_res, nullptr);
  for (absl::string_view key :
       {kSectionDitCapEmbedder, kSectionDitContextRefiner, kSectionDitXEmbedder,
        kSectionDitNoiseRefiner, kSectionDitSingleBlock4,
        kSectionDitSingleBlock5}) {
    ASSERT_OK_AND_ASSIGN(absl::string_view buf,
                         lm_res->GetGenericBinaryDataBuffer(key));
    EXPECT_FALSE(buf.empty()) << "Empty GenericBinaryData section: " << key;
  }
}

TEST(OmniGen2FactoryTest, RejectsMissingTfLiteOrGenericBinarySection) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  auto env = std::make_shared<Environment>(std::move(*env_or));
  const OmniGen2ModelConfig config = MakeSmallModelConfig();

  // Missing standard TFLiteModel section.
  OmniGen2TestBundleOptions missing_final;
  missing_final.omit_section = "tf_lite_diffusion_transformer_final";
  ASSERT_OK_AND_ASSIGN(
      std::string no_final_path,
      WriteOmniGen2Bundle("omnigen2_missing_final.litertlm", missing_final));
  EXPECT_THAT(CreateOmniGen2Resources(config, no_final_path, ":nocache",
                                      lm::Backend::CPU, 1, env),
              StatusIs(absl::StatusCode::kNotFound));

  // Missing GenericBinaryData section (`dit_cap_embedder`).
  OmniGen2TestBundleOptions missing_cap;
  missing_cap.omit_section = std::string(kSectionDitCapEmbedder);
  ASSERT_OK_AND_ASSIGN(
      std::string no_cap_path,
      WriteOmniGen2Bundle("omnigen2_missing_cap.litertlm", missing_cap));
  EXPECT_THAT(CreateOmniGen2Resources(config, no_cap_path, ":nocache",
                                      lm::Backend::CPU, 1, env),
              StatusIs(absl::StatusCode::kNotFound));

  // Missing GenericBinaryData main shard (`dit_single_block_4`).
  OmniGen2TestBundleOptions missing_block4;
  missing_block4.omit_section = std::string(kSectionDitSingleBlock4);
  ASSERT_OK_AND_ASSIGN(
      std::string no_b4_path,
      WriteOmniGen2Bundle("omnigen2_missing_b4.litertlm", missing_block4));
  EXPECT_THAT(CreateOmniGen2Resources(config, no_b4_path, ":nocache",
                                      lm::Backend::CPU, 1, env),
              StatusIs(absl::StatusCode::kNotFound));

  // Missing container file and empty ModelResources in
  // CreateOmniGen2Components.
  EXPECT_THAT(CreateOmniGen2Resources(config, "/nonexistent/omnigen2.litertlm",
                                      ":nocache", lm::Backend::CPU, 1, env),
              StatusIs(absl::StatusCode::kNotFound));

  auto empty_resources = std::make_shared<ModelResources>(env);
  std::vector<std::unique_ptr<internal::StageBase>> stages;
  EXPECT_THAT(
      CreateOmniGen2Components(
          config, std::make_unique<PushPromptSource>(ImageGenInputMetadata{}),
          empty_resources, stages),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OmniGen2EngineTest, EndToEndGenerationWithCustomModelMetadataAndInt64Te) {
  OmniGen2TestBundleOptions opts;
  opts.vae_output_value = 0.5f;  // (0.5 + 1.0) / 2.0 * 255 = 191.25 -> 191
  opts.text_input_type = tflite::TensorType_INT64;
  ASSERT_OK_AND_ASSIGN(
      std::string bundle_path,
      WriteOmniGen2Bundle("omnigen2_engine_e2e.litertlm", opts));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.num_threads = 1;

  lm::proto::Text2ImageSessionConfig session_cfg;
  session_cfg.set_num_inference_steps(2);
  session_cfg.set_seed(123);

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniEngine> engine,
                       OmniEngine::Create("omnigen2", settings, &session_cfg,
                                          /*metadata=*/nullptr));

  auto input_source = std::make_unique<PushInputSource>();
  ASSERT_OK(input_source->PushInput(
      OmniSession::TextInput{.text = "a robot painting a canvas"}));
  input_source->Finish();

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> session,
                       engine->CreateSession(std::move(input_source)));

  // Concurrent second session must fail while first session holds the lease.
  auto second_input = std::make_unique<PushInputSource>();
  second_input->Finish();
  EXPECT_THAT(
      engine->CreateSession(std::move(second_input)),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Another OmniGen2 session is already active")));

  ImageOutput image;
  bool got_image = false;
  while (true) {
    auto out = session->ProcessNext();
    if (absl::IsOutOfRange(out.status())) {
      break;
    }
    if (absl::IsNotFound(out.status())) {
      auto flushed = session->Flush();
      if (flushed.ok()) {
        if (const auto* img = std::get_if<ImageOutput>(&*flushed)) {
          image = *img;
          got_image = true;
        }
      }
      break;
    }
    ASSERT_OK(out.status());
    if (const auto* img = std::get_if<ImageOutput>(&*out)) {
      image = *img;
      got_image = true;
    }
  }
  ASSERT_TRUE(got_image);
  EXPECT_EQ(image.width, 32);
  EXPECT_EQ(image.height, 32);
  EXPECT_EQ(image.channels, 3);
  ASSERT_EQ(image.rgb_data.size(), 32u * 32u * 3u);
  EXPECT_NEAR(image.rgb_data[0], 191, 1);

  // Destroying the first session releases the active-session lease so a new
  // session can be created.
  session.reset();
  auto third_input = std::make_unique<PushInputSource>();
  ASSERT_OK(third_input->PushInput(OmniSession::TextInput{.text = "a cat"}));
  third_input->Finish();
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniSession> next_session,
                       engine->CreateSession(std::move(third_input)));
  EXPECT_NE(next_session, nullptr);
}

TEST(OmniGen2EngineTest, AppliesCallerCustomModelOverrideAndRejectsWrongProto) {
  ASSERT_OK_AND_ASSIGN(std::string bundle_path,
                       WriteOmniGen2Bundle("omnigen2_custom_meta.litertlm"));

  OmniEngine::Settings settings;
  settings.model_path = bundle_path;
  settings.cache_dir = ":nocache";
  settings.num_threads = 1;

  // 1. Caller override packing OmniGen2 with `text_guidance_scale = 1.0`
  // (disables CFG) succeeds and round-trips through `custom_model`.
  OmniGen2TestBundleOptions override_opts;
  override_opts.text_guidance_scale = 1.0f;
  ASSERT_OK_AND_ASSIGN(lm::proto::Text2ImageMetadata override_meta,
                       BuildDefaultTestMetadata(override_opts));
  lm::proto::OmniGen2 unpacked;
  ASSERT_TRUE(
      override_meta.image_gen_model_type().custom_model().UnpackTo(&unpacked));
  EXPECT_FLOAT_EQ(unpacked.text_guidance_scale(), 1.0f);
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<OmniEngine> ok_engine,
      OmniEngine::Create("omnigen2", settings, /*session_config=*/nullptr,
                         &override_meta));
  EXPECT_NE(ok_engine, nullptr);

  // 1b. Caller override with invalid OmniGen2 `img_size` must take effect (not
  // be overwritten by re-reading embedded bundle metadata) and be rejected.
  lm::proto::Text2ImageMetadata invalid_override_meta;
  lm::proto::OmniGen2 bad_omnigen2 = unpacked;
  bad_omnigen2.set_img_size(500);
  ASSERT_TRUE(invalid_override_meta.mutable_image_gen_model_type()
                  ->mutable_custom_model()
                  ->PackFrom(bad_omnigen2));
  EXPECT_THAT(
      OmniEngine::Create("omnigen2", settings, /*session_config=*/nullptr,
                         &invalid_override_meta),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // 2. Wrong proto packed into `custom_model` -> InvalidArgumentError.
  lm::proto::Text2ImageMetadata wrong_custom_meta;
  lm::proto::Text2ImageSessionConfig wrong_payload;
  wrong_payload.set_width(32);
  ASSERT_TRUE(wrong_custom_meta.mutable_image_gen_model_type()
                  ->mutable_custom_model()
                  ->PackFrom(wrong_payload));
  EXPECT_THAT(
      OmniEngine::Create("omnigen2", settings, /*session_config=*/nullptr,
                         &wrong_custom_meta),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // 3. Non-custom oneof arm (`flux2_klein`) -> InvalidArgumentError.
  lm::proto::Text2ImageMetadata non_custom_meta;
  non_custom_meta.mutable_image_gen_model_type()->mutable_flux2_klein();
  EXPECT_THAT(OmniEngine::Create("omnigen2", settings,
                                 /*session_config=*/nullptr, &non_custom_meta),
              StatusIs(absl::StatusCode::kInvalidArgument));

  // 4. Non-square session config (width != height) -> InvalidArgumentError,
  // and must release the active-session lease so a retry sees InvalidArgument
  // rather than FailedPrecondition.
  lm::proto::Text2ImageSessionConfig nonsquare_cfg;
  nonsquare_cfg.set_width(32);
  nonsquare_cfg.set_height(64);
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<OmniEngine> sq_engine,
                       OmniEngine::Create("omnigen2", settings, &nonsquare_cfg,
                                          /*metadata=*/nullptr));
  auto in = std::make_unique<PushInputSource>();
  in->Finish();
  EXPECT_THAT(sq_engine->CreateSession(std::move(in)),
              StatusIs(absl::StatusCode::kInvalidArgument));
  auto retry_in = std::make_unique<PushInputSource>();
  retry_in->Finish();
  EXPECT_THAT(sq_engine->CreateSession(std::move(retry_in)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OmniGen2FactoryTest,
     CreateOmniGen2VaeDecoderStageUnpacksScalesAndConvertsToClampedRgb888) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  Environment env = std::move(*env_or);

  OmniGen2ModelConfig config = MakeSmallModelConfig();
  config.vae_scaling_factor = 0.5f;
  config.vae_shift_factor = 0.25f;
  constexpr int kLatentDim = 4;
  constexpr int kImgPixels = 32 * 32;

  StubDenoiserStage denoiser;
  auto runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*runner, CreateInputBuffers(absl::string_view("")))
      .WillOnce(
          [&](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            std::vector<TensorBuffer> bufs;
            ABSL_ASSIGN_OR_RETURN(
                TensorBuffer buf,
                CreateHostFloatBuffer(
                    env, {1, config.in_channels, kLatentDim, kLatentDim}));
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_CALL(*runner, CreateOutputBuffers(absl::string_view("")))
      .WillOnce(
          [&](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            std::vector<TensorBuffer> bufs;
            ABSL_ASSIGN_OR_RETURN(
                TensorBuffer buf,
                CreateHostFloatBuffer(
                    env, {1, 3, config.img_size, config.img_size}));
            bufs.push_back(std::move(buf));
            return bufs;
          });

  std::vector<float> nchw_expected(config.in_channels * kLatentDim *
                                   kLatentDim);
  for (size_t i = 0; i < nchw_expected.size(); ++i) {
    nchw_expected[i] = static_cast<float>(i) * 0.05f;
  }
  ASSERT_OK_AND_ASSIGN(
      std::vector<float> packed,
      PackOmniGen2Latents(nchw_expected, config.in_channels, kLatentDim,
                          kLatentDim, config.patch_size));

  EXPECT_CALL(*runner, Run(absl::string_view(""), _, _))
      .WillOnce([&](absl::string_view, absl::Span<const TensorBuffer> inputs,
                    absl::Span<const TensorBuffer> outputs) {
        std::vector<float> z_in(nchw_expected.size(), 0.0f);
        EXPECT_TRUE(const_cast<TensorBuffer&>(inputs[0])
                        .Read<float>(absl::MakeSpan(z_in))
                        .HasValue());
        for (size_t i = 0; i < nchw_expected.size(); ++i) {
          const float expected = nchw_expected[i] / config.vae_scaling_factor +
                                 config.vae_shift_factor;
          EXPECT_THAT(z_in[i], FloatNear(expected, 1e-5f));
        }

        // Write planar NCHW with R=-1.5 (clamps to 0), G=0.0 (128), B=1.5
        // (clamps to 255).
        std::vector<float> img_nchw(3 * kImgPixels, 0.0f);
        for (int p = 0; p < kImgPixels; ++p) {
          img_nchw[p] = -1.5f;
          img_nchw[kImgPixels + p] = 0.0f;
          img_nchw[2 * kImgPixels + p] = 1.5f;
        }
        EXPECT_TRUE(const_cast<TensorBuffer&>(outputs[0])
                        .Write<float>(img_nchw)
                        .HasValue());
        return absl::OkStatus();
      });

  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VaeDecoderStage> stage,
      CreateOmniGen2VaeDecoderStage(&denoiser, config, std::move(runner)));

  EXPECT_FALSE(stage->NeedSchedule());
  denoiser.Push(DenoisedLatents{.packed_latents = std::move(packed)});
  EXPECT_TRUE(stage->NeedSchedule());
  ASSERT_OK(stage->Schedule());
  ASSERT_OK(stage->Flush());
  ASSERT_TRUE(stage->HasOutput());

  ASSERT_OK_AND_ASSIGN(Output out, stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(out));
  const auto& img = std::get<ImageOutput>(out);
  EXPECT_EQ(img.width, 32);
  EXPECT_EQ(img.height, 32);
  EXPECT_EQ(img.channels, 3);
  ASSERT_EQ(img.rgb_data.size(), static_cast<size_t>(kImgPixels * 3));
  EXPECT_EQ(img.rgb_data[0], 0);
  EXPECT_EQ(img.rgb_data[1], 128);
  EXPECT_EQ(img.rgb_data[2], 255);
}

TEST(OmniGen2FactoryTest,
     CreateOmniGen2VaeDecoderStageRunsWithRealCompiledModel) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  Environment env = std::move(*env_or);

  OmniGen2ModelConfig config = MakeSmallModelConfig();
  config.vae_scaling_factor = 0.5f;
  config.vae_shift_factor = 0.25f;
  constexpr int kLatentDim = 4;
  constexpr int kImgPixels = 32 * 32;

  std::vector<float> decoded_vals(3 * kImgPixels, 1.0f);
  const std::string tflite_buf = any2any::BuildMinimalTfliteModelBuffer(
      {{"args_0",
        {1, config.in_channels, kLatentDim, kLatentDim},
        tflite::TensorType_FLOAT32}},
      {{"image",
        {1, 3, config.img_size, config.img_size},
        tflite::TensorType_FLOAT32,
        decoded_vals}});
  ModelOptions options;
  options.backend = lm::Backend::CPU;
  options.num_threads = 1;
  options.cache_dir = ":nocache";
  ASSERT_OK_AND_ASSIGN(
      CompiledModel compiled,
      CreateCompiledModelFromBuffer(env, options, tflite_buf, "test_vae"));

  StubDenoiserStage denoiser;
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VaeDecoderStage> real_stage,
      CreateOmniGen2VaeDecoderStage(
          &denoiser, config, std::make_unique<LiteRtRunnerImpl>(&compiled)));
  denoiser.Push(DenoisedLatents{
      .packed_latents = std::vector<float>(4 * config.packed_ch, 0.0f)});
  ASSERT_OK(real_stage->Schedule());
  ASSERT_OK_AND_ASSIGN(Output real_out, real_stage->GetOutput());
  ASSERT_TRUE(std::holds_alternative<ImageOutput>(real_out));
  const auto& real_img = std::get<ImageOutput>(real_out);
  EXPECT_EQ(real_img.width, 32);
  EXPECT_EQ(real_img.height, 32);
  EXPECT_EQ(real_img.rgb_data[0], 255);
}

TEST(OmniGen2FactoryTest,
     CreateOmniGen2VaeDecoderStageRejectsInvalidConfigAndBuffers) {
  auto env_or = Environment::Create({});
  ASSERT_TRUE(env_or.HasValue());
  Environment env = std::move(*env_or);

  OmniGen2ModelConfig config = MakeSmallModelConfig();
  constexpr int kLatentDim = 4;
  StubDenoiserStage denoiser;

  // Wrong output buffer shape rejected at Create() time.
  auto bad_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*bad_runner, CreateInputBuffers(absl::string_view("")))
      .WillOnce(
          [&](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            std::vector<TensorBuffer> bufs;
            ABSL_ASSIGN_OR_RETURN(
                TensorBuffer buf,
                CreateHostFloatBuffer(
                    env, {1, config.in_channels, kLatentDim, kLatentDim}));
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_CALL(*bad_runner, CreateOutputBuffers(absl::string_view("")))
      .WillOnce(
          [&](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            std::vector<TensorBuffer> bufs;
            ABSL_ASSIGN_OR_RETURN(TensorBuffer buf,
                                  CreateHostFloatBuffer(env, {1, 3, 16, 16}));
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_THAT(
      CreateOmniGen2VaeDecoderStage(&denoiser, config, std::move(bad_runner)),
      StatusIs(absl::StatusCode::kInvalidArgument));

  // Wrong packed_latents size rejected at Schedule() time.
  auto ok_runner = std::make_unique<MockLiteRtRunner>();
  EXPECT_CALL(*ok_runner, CreateInputBuffers(absl::string_view("")))
      .WillOnce(
          [&](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            std::vector<TensorBuffer> bufs;
            ABSL_ASSIGN_OR_RETURN(
                TensorBuffer buf,
                CreateHostFloatBuffer(
                    env, {1, config.in_channels, kLatentDim, kLatentDim}));
            bufs.push_back(std::move(buf));
            return bufs;
          });
  EXPECT_CALL(*ok_runner, CreateOutputBuffers(absl::string_view("")))
      .WillOnce(
          [&](absl::string_view) -> absl::StatusOr<std::vector<TensorBuffer>> {
            std::vector<TensorBuffer> bufs;
            ABSL_ASSIGN_OR_RETURN(
                TensorBuffer buf,
                CreateHostFloatBuffer(
                    env, {1, 3, config.img_size, config.img_size}));
            bufs.push_back(std::move(buf));
            return bufs;
          });
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VaeDecoderStage> stage,
      CreateOmniGen2VaeDecoderStage(&denoiser, config, std::move(ok_runner)));
  denoiser.Push(DenoisedLatents{.packed_latents = {1.0f, 2.0f}});
  EXPECT_THAT(stage->Schedule(), StatusIs(absl::StatusCode::kInvalidArgument));

  // Invalid config is rejected at Create() time.
  OmniGen2ModelConfig bad_config = config;
  bad_config.img_size = 30;
  EXPECT_THAT(CreateOmniGen2VaeDecoderStage(
                  &denoiser, bad_config, std::make_unique<MockLiteRtRunner>()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::text2image::omnigen2

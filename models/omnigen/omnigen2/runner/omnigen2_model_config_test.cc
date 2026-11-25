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

#include "models/omnigen/omnigen2/runner/omnigen2_model_config.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "models/omnigen/omnigen2/runner/omnigen2.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "support/tokenizer/tokenizer.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::text2image::omnigen2 {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FloatNear;

class FakeTokenizer : public support::Tokenizer {
 public:
  FakeTokenizer(int im_start_id, int im_end_id, int pad_id,
                support::TokenIds sys_tokens, support::TokenIds nl_tokens)
      : im_start_id_(im_start_id),
        im_end_id_(im_end_id),
        pad_id_(pad_id),
        sys_tokens_(std::move(sys_tokens)),
        nl_tokens_(std::move(nl_tokens)) {}

  support::TokenizerType GetTokenizerType() const override {
    return support::TokenizerType::kHuggingFace;
  }

  absl::StatusOr<support::TokenIds> TextToTokenIds(
      absl::string_view text) override {
    if (text == "\n") {
      return nl_tokens_;
    }
    if (text ==
        "system\nYou are a helpful assistant that generates high-quality "
        "images based on user instructions.") {
      return sys_tokens_;
    }
    return support::TokenIds{10, 20};
  }

  absl::StatusOr<int> TokenToId(absl::string_view token) override {
    if (token == "<|im_start|>" && im_start_id_ >= 0) return im_start_id_;
    if (token == "<|im_end|>" && im_end_id_ >= 0) return im_end_id_;
    if (token == "<|endoftext|>" && pad_id_ >= 0) return pad_id_;
    return absl::NotFoundError("not implemented");
  }

  absl::StatusOr<std::string> TokenIdsToText(
      absl::Span<const int> /*token_ids*/,
      bool /*skip_special_tokens*/) override {
    return "";
  }

  std::vector<std::string> GetTokens() const override { return {}; }

  int GetVocabSize() const override { return 151936; }

 private:
  int im_start_id_;
  int im_end_id_;
  int pad_id_;
  support::TokenIds sys_tokens_;
  support::TokenIds nl_tokens_;
};

TEST(OmniGen2ModelConfigTest, DefaultAndDynamicFramingAndPadResolution) {
  const auto default_prefix = DefaultOmniGen2PrefixTokenIds();
  const auto default_suffix = DefaultOmniGen2SuffixTokenIds();

  FakeTokenizer qwen_tok(
      /*im_start_id=*/151644, /*im_end_id=*/151645, /*pad_id=*/151643,
      /*sys_tokens=*/
      {8948, 198, 2610, 525, 264, 10950, 17847, 429, 26885, 1550, 22092, 5335,
       3118, 389, 1196, 11221, 13},
      /*nl_tokens=*/{198});

  EXPECT_THAT(default_prefix,
              ElementsAre(151644, 8948, 198, 2610, 525, 264, 10950, 17847, 429,
                          26885, 1550, 22092, 5335, 3118, 389, 1196, 11221, 13,
                          151645, 198, 151644));
  EXPECT_EQ(ResolveOmniGen2PrefixTokenIds(&qwen_tok), default_prefix);
  EXPECT_EQ(ResolveOmniGen2PrefixTokenIds(nullptr), default_prefix);
  EXPECT_THAT(default_suffix, ElementsAre(151645, 198));
  EXPECT_EQ(ResolveOmniGen2SuffixTokenIds(&qwen_tok), default_suffix);
  EXPECT_EQ(ResolveOmniGen2SuffixTokenIds(nullptr), default_suffix);
  EXPECT_EQ(ResolveOmniGen2PadTokenId(&qwen_tok), kOmniGen2PadTokenId);
  EXPECT_EQ(ResolveOmniGen2PadTokenId(nullptr), kOmniGen2PadTokenId);

  // Tiny synthetic tokenizer with custom IDs resolves dynamically.
  FakeTokenizer tiny_tok(/*im_start_id=*/1, /*im_end_id=*/2, /*pad_id=*/0,
                         /*sys_tokens=*/{4, 5, 6, 7}, /*nl_tokens=*/{3});
  EXPECT_THAT(ResolveOmniGen2PrefixTokenIds(&tiny_tok),
              ElementsAre(1, 4, 5, 6, 7, 2, 3, 1));
  EXPECT_THAT(ResolveOmniGen2SuffixTokenIds(&tiny_tok), ElementsAre(2, 3));
  EXPECT_EQ(ResolveOmniGen2PadTokenId(&tiny_tok), 0);

  // Tokenizer without ChatML special tokens falls back to defaults.
  FakeTokenizer fallback_tok(/*im_start_id=*/-1, /*im_end_id=*/-1,
                             /*pad_id=*/-1, /*sys_tokens=*/{1},
                             /*nl_tokens=*/{});
  EXPECT_EQ(ResolveOmniGen2PrefixTokenIds(&fallback_tok), default_prefix);
  EXPECT_EQ(ResolveOmniGen2SuffixTokenIds(&fallback_tok), default_suffix);
  EXPECT_EQ(ResolveOmniGen2PadTokenId(&fallback_tok), kOmniGen2PadTokenId);
}

TEST(OmniGen2ModelConfigTest, ValidateOmniGen2ModelConfigChecksAllBounds) {
  OmniGen2ModelConfig config;
  EXPECT_OK(ValidateOmniGen2ModelConfig(config));

  OmniGen2ModelConfig bad_seq = config;
  bad_seq.seq_len = 0;
  EXPECT_THAT(ValidateOmniGen2ModelConfig(bad_seq),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig bad_img = config;
  bad_img.img_size = 500;
  EXPECT_THAT(ValidateOmniGen2ModelConfig(bad_img),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig bad_packed = config;
  bad_packed.packed_ch = 32;
  EXPECT_THAT(ValidateOmniGen2ModelConfig(bad_packed),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig bad_heads = config;
  bad_heads.dit_num_heads = 20;
  EXPECT_THAT(ValidateOmniGen2ModelConfig(bad_heads),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig bad_kv = config;
  bad_kv.dit_num_kv_heads = 4;
  EXPECT_THAT(ValidateOmniGen2ModelConfig(bad_kv),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig bad_rope = config;
  bad_rope.axes_dims_rope = {40, 40, 20};
  EXPECT_THAT(ValidateOmniGen2ModelConfig(bad_rope),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig odd_rope = config;
  odd_rope.axes_dims_rope = {39, 41, 40};
  EXPECT_THAT(ValidateOmniGen2ModelConfig(odd_rope),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig bad_theta = config;
  bad_theta.rope_theta = 0.0f;
  EXPECT_THAT(ValidateOmniGen2ModelConfig(bad_theta),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig bad_scale = config;
  bad_scale.vae_scaling_factor = 0.0f;
  EXPECT_THAT(ValidateOmniGen2ModelConfig(bad_scale),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig bad_shift = config;
  bad_shift.vae_shift_factor = std::numeric_limits<float>::quiet_NaN();
  EXPECT_THAT(ValidateOmniGen2ModelConfig(bad_shift),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig bad_steps = config;
  bad_steps.steps = 0;
  EXPECT_THAT(ValidateOmniGen2ModelConfig(bad_steps),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig bad_cfg = config;
  bad_cfg.text_guidance_scale = -1.0f;
  EXPECT_THAT(ValidateOmniGen2ModelConfig(bad_cfg),
              StatusIs(absl::StatusCode::kInvalidArgument));

  OmniGen2ModelConfig nan_cfg = config;
  nan_cfg.text_guidance_scale = std::numeric_limits<float>::infinity();
  EXPECT_THAT(ValidateOmniGen2ModelConfig(nan_cfg),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OmniGen2ModelConfigTest, PopulateOmniGen2ConfigFromProtoPopulatesFields) {
  lm::proto::OmniGen2 proto;
  proto.set_seq_len(64);
  proto.set_img_size(256);
  proto.set_in_channels(16);
  proto.set_patch_size(2);
  proto.set_hidden_size(48);
  proto.set_prompt_dim(32);
  proto.set_default_steps(12);
  proto.set_vae_scaling_factor(0.3611f);
  proto.set_vae_shift_factor(0.1159f);
  proto.set_timestep_scale(500.0f);
  proto.add_axes_dim_rope(4);
  proto.add_axes_dim_rope(6);
  proto.add_axes_dim_rope(6);
  proto.set_dynamic_time_shift(false);
  proto.set_text_guidance_scale(3.5f);

  OmniGen2ModelConfig config;
  PopulateOmniGen2ConfigFromProto(proto, config);

  EXPECT_OK(ValidateOmniGen2ModelConfig(config));
  EXPECT_EQ(config.seq_len, 64);
  EXPECT_EQ(config.img_size, 256);
  EXPECT_EQ(config.in_channels, 16);
  EXPECT_EQ(config.patch_size, 2);
  EXPECT_EQ(config.packed_ch, 64);
  EXPECT_EQ(config.dit_hidden_size, 48);
  EXPECT_EQ(config.text_hidden_size, 32);
  EXPECT_EQ(config.dit_head_dim, 16);
  EXPECT_EQ(config.dit_num_heads, 3);
  EXPECT_EQ(config.dit_num_kv_heads, 1);
  EXPECT_EQ(config.steps, 12);
  EXPECT_FLOAT_EQ(config.vae_scaling_factor, 0.3611f);
  EXPECT_FLOAT_EQ(config.vae_shift_factor, 0.1159f);
  EXPECT_FLOAT_EQ(config.timestep_scale, 500.0f);
  EXPECT_THAT(config.axes_dims_rope, ElementsAre(4, 6, 6));
  EXPECT_FALSE(config.dynamic_time_shift);
  EXPECT_FLOAT_EQ(config.text_guidance_scale, 3.5f);
}

TEST(OmniGen2MathTest, ComputeOmniGen2TimestepsMatchesOfficialFormula) {
  // At 512x512, latent_h = 64, latent_w = 64, m = sqrt(64*64) / 40 = 1.6.
  ASSERT_OK_AND_ASSIGN(
      std::vector<float> shifted,
      ComputeOmniGen2Timesteps(/*steps=*/4, /*height=*/512, /*width=*/512,
                               /*dynamic_time_shift=*/true));
  ASSERT_OK_AND_ASSIGN(
      std::vector<float> linear,
      ComputeOmniGen2Timesteps(/*steps=*/4, /*height=*/512, /*width=*/512,
                               /*dynamic_time_shift=*/false));

  ASSERT_EQ(shifted.size(), 5);
  EXPECT_FLOAT_EQ(shifted[0], 0.0f);
  EXPECT_FLOAT_EQ(shifted[4], 1.0f);
  const float m = 64.0f / 40.0f;
  for (int i = 1; i < 4; ++i) {
    const float t = static_cast<float>(i) / 4.0f;
    const float expected = t / (m - m * t + t);
    EXPECT_THAT(shifted[i], FloatNear(expected, 1e-6f));
    EXPECT_THAT(linear[i], FloatNear(t, 1e-6f));
  }

  EXPECT_THAT(ComputeOmniGen2Timesteps(0, 512, 512),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(ComputeOmniGen2Timesteps(4, 500, 512),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OmniGen2MathTest, BuildOmniGen2PositionIdsRotaryEmbedAndMasks) {
  constexpr int kSeqLen = 6;
  constexpr int kActiveCapLen = 4;
  constexpr int kGridH = 2;
  constexpr int kGridW = 2;
  constexpr int kImgTokens = kGridH * kGridW;
  const std::array<int, 3> kAxesDims = {4, 6, 6};  // head_dim = 16

  ASSERT_OK_AND_ASSIGN(std::vector<int32_t> cap_ids,
                       BuildOmniGen2CaptionPositionIds(kSeqLen, kActiveCapLen));
  ASSERT_OK_AND_ASSIGN(
      std::vector<int32_t> img_ids,
      BuildOmniGen2ImagePositionIds(kGridH, kGridW, kActiveCapLen));
  ASSERT_OK_AND_ASSIGN(auto cap_rope, BuildOmniGen2RotaryPosEmbed(
                                          cap_ids, kSeqLen, kAxesDims,
                                          /*theta=*/10000.0f, kActiveCapLen));
  ASSERT_OK_AND_ASSIGN(
      auto img_rope, BuildOmniGen2RotaryPosEmbed(img_ids, kImgTokens, kAxesDims,
                                                 /*theta=*/10000.0f));
  ASSERT_OK_AND_ASSIGN(auto masks, BuildOmniGen2AttentionMasks(
                                       kSeqLen, kActiveCapLen, kImgTokens));

  for (int i = 0; i < kActiveCapLen; ++i) {
    EXPECT_EQ(cap_ids[3 * i + 0], i);
    EXPECT_EQ(cap_ids[3 * i + 1], i);
    EXPECT_EQ(cap_ids[3 * i + 2], i);
  }
  for (int i = kActiveCapLen; i < kSeqLen; ++i) {
    EXPECT_EQ(cap_ids[3 * i + 0], 0);
    EXPECT_EQ(cap_ids[3 * i + 1], 0);
    EXPECT_EQ(cap_ids[3 * i + 2], 0);
  }
  for (int t = kActiveCapLen; t < kSeqLen; ++t) {
    for (int d = 0; d < 16; ++d) {
      EXPECT_FLOAT_EQ(cap_rope.first[t * 16 + d], 0.0f);
      EXPECT_FLOAT_EQ(cap_rope.second[t * 16 + d], 0.0f);
    }
  }
  EXPECT_EQ(img_ids[3 * 3 + 0], kActiveCapLen);
  EXPECT_EQ(img_ids[3 * 3 + 1], 1);
  EXPECT_EQ(img_ids[3 * 3 + 2], 1);
  EXPECT_THAT(img_rope.first[0], FloatNear(std::cos(4.0f), 1e-6f));
  EXPECT_THAT(img_rope.first[1], FloatNear(std::cos(4.0f), 1e-6f));
  EXPECT_THAT(img_rope.second[0], FloatNear(std::sin(4.0f), 1e-6f));
  EXPECT_THAT(img_rope.second[1], FloatNear(std::sin(4.0f), 1e-6f));

  // Verify exact bit-for-bit double-precision rotary formula across all 3 axes
  // for image token (h=1, w=1, id=4) at offset t=3.
  const double theta = 10000.0;
  const int token_base = 3 * 16;
  const std::array<int, 3> token_pos = {kActiveCapLen, 1, 1};
  int axis_offset = 0;
  for (int axis = 0; axis < 3; ++axis) {
    const int axis_dim = kAxesDims[axis];
    for (int k = 0; k < axis_dim / 2; ++k) {
      const double inv_freq =
          1.0 / std::pow(theta, static_cast<double>(2 * k) / axis_dim);
      const double angle = static_cast<double>(token_pos[axis]) * inv_freq;
      const float expected_cos = static_cast<float>(std::cos(angle));
      const float expected_sin = static_cast<float>(std::sin(angle));
      EXPECT_EQ(img_rope.first[token_base + axis_offset + 2 * k], expected_cos);
      EXPECT_EQ(img_rope.first[token_base + axis_offset + 2 * k + 1],
                expected_cos);
      EXPECT_EQ(img_rope.second[token_base + axis_offset + 2 * k],
                expected_sin);
      EXPECT_EQ(img_rope.second[token_base + axis_offset + 2 * k + 1],
                expected_sin);
    }
    axis_offset += axis_dim;
  }

  // active_tokens = 0 returns all-zero cos and sin tensors.
  ASSERT_OK_AND_ASSIGN(
      auto zero_rope,
      BuildOmniGen2RotaryPosEmbed(cap_ids, kSeqLen, kAxesDims,
                                  /*theta=*/10000.0f, /*active_tokens=*/0));
  ASSERT_EQ(zero_rope.first.size(), static_cast<size_t>(kSeqLen * 16));
  ASSERT_EQ(zero_rope.second.size(), static_cast<size_t>(kSeqLen * 16));
  for (size_t i = 0; i < zero_rope.first.size(); ++i) {
    EXPECT_FLOAT_EQ(zero_rope.first[i], 0.0f);
    EXPECT_FLOAT_EQ(zero_rope.second[i], 0.0f);
  }

  // Invalid theta, active_tokens, and out-of-exact-float32 position ID.
  EXPECT_THAT(
      BuildOmniGen2RotaryPosEmbed(cap_ids, kSeqLen, kAxesDims,
                                  std::numeric_limits<float>::quiet_NaN()),
      StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildOmniGen2RotaryPosEmbed(cap_ids, kSeqLen, kAxesDims,
                                          /*theta=*/10000.0f,
                                          /*active_tokens=*/kSeqLen + 1),
              StatusIs(absl::StatusCode::kInvalidArgument));
  const std::vector<int32_t> large_pos_ids = {(1 << 24) + 1, 0, 0};
  EXPECT_THAT(
      BuildOmniGen2RotaryPosEmbed(large_pos_ids, /*num_tokens=*/1, kAxesDims),
      StatusIs(absl::StatusCode::kInvalidArgument));

  EXPECT_THAT(masks.first, ElementsAre(0.0f, 0.0f, 0.0f, 0.0f, -1e9f, -1e9f));
  EXPECT_THAT(masks.second, ElementsAre(0.0f, 0.0f, 0.0f, 0.0f, -1e9f, -1e9f,
                                        0.0f, 0.0f, 0.0f, 0.0f));

  EXPECT_THAT(BuildOmniGen2CaptionPositionIds(4, 5),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildOmniGen2ImagePositionIds(0, 2, 0),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(BuildOmniGen2AttentionMasks(4, 5, 4),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(OmniGen2MathTest, PackAndUnpackAndScaleLatentsRoundTrip) {
  constexpr int kChannels = 16;
  constexpr int kLatentH = 4;
  constexpr int kLatentW = 4;
  constexpr int kPatchSize = 2;
  std::vector<float> nchw(kChannels * kLatentH * kLatentW);
  for (size_t i = 0; i < nchw.size(); ++i) {
    nchw[i] = static_cast<float>(i + 1) * 0.1f;
  }

  ASSERT_OK_AND_ASSIGN(
      std::vector<float> packed,
      PackOmniGen2Latents(nchw, kChannels, kLatentH, kLatentW, kPatchSize));
  ASSERT_OK_AND_ASSIGN(
      std::vector<float> unpacked_identity,
      UnpackAndScaleOmniGen2Latents(packed, kChannels, kLatentH, kLatentW,
                                    kPatchSize, /*vae_scaling_factor=*/1.0f,
                                    /*vae_shift_factor=*/0.0f));
  ASSERT_OK_AND_ASSIGN(
      std::vector<float> unpacked_scaled,
      UnpackAndScaleOmniGen2Latents(packed, kChannels, kLatentH, kLatentW,
                                    kPatchSize, /*vae_scaling_factor=*/0.3611f,
                                    /*vae_shift_factor=*/0.1159f));

  ASSERT_EQ(unpacked_identity.size(), nchw.size());
  for (size_t i = 0; i < nchw.size(); ++i) {
    EXPECT_FLOAT_EQ(unpacked_identity[i], nchw[i]);
    EXPECT_THAT(unpacked_scaled[i],
                FloatNear(nchw[i] / 0.3611f + 0.1159f, 1e-5f));
  }

  EXPECT_THAT(PackOmniGen2Latents({}, kChannels, kLatentH, kLatentW, 2),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(UnpackAndScaleOmniGen2Latents(packed, kChannels, kLatentH,
                                            kLatentW, 2, 0.0f, 0.0f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(UnpackAndScaleOmniGen2Latents(
                  packed, kChannels, kLatentH, kLatentW, 2,
                  std::numeric_limits<float>::quiet_NaN(), 0.0f),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(UnpackAndScaleOmniGen2Latents(
                  packed, kChannels, kLatentH, kLatentW, 2, 0.3611f,
                  std::numeric_limits<float>::infinity()),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(UnpackAndScaleOmniGen2Latents({1.0f, 2.0f}, kChannels, kLatentH,
                                            kLatentW, 2, 0.3611f, 0.1159f),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::text2image::omnigen2

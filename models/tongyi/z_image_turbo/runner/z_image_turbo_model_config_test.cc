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

#include "models/tongyi/z_image_turbo/runner/z_image_turbo_model_config.h"

#include <cstddef>
#include <vector>

#include "models/tongyi/z_image_turbo/runner/z_image_turbo.pb.h"
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/types/span.h"
#include "support/util/test_utils.h"  // IWYU pragma: keep for ASSERT_OK

namespace litert::omni::text2image {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::FloatNear;
using ::testing::IsEmpty;
using ::testing::Pointwise;

TEST(ZImageTurboModelConfigTest, DefaultDerivedDimensions) {
  ZImageTurboModelConfig config;
  EXPECT_EQ(config.seq_len, 64);
  EXPECT_EQ(config.img_size, 256);
  EXPECT_EQ(config.in_channels, 16);
  EXPECT_EQ(config.patch_size, 2);
  EXPECT_EQ(config.dim, 3840);
  EXPECT_EQ(config.prompt_dim, 2560);
  EXPECT_EQ(config.steps, 8);
  EXPECT_FLOAT_EQ(config.vae_scaling_factor, 0.3611f);
  EXPECT_FLOAT_EQ(config.vae_shift_factor, 0.1159f);
  EXPECT_FLOAT_EQ(config.rope_theta, 256.0f);
  EXPECT_FLOAT_EQ(config.shift, 3.0f);
  EXPECT_FALSE(config.use_dynamic_shifting);

  EXPECT_EQ(config.LatentSize(), 32);
  EXPECT_EQ(config.GridSize(), 16);
  EXPECT_EQ(config.ImageTokens(), 256);
  EXPECT_EQ(config.TotalTokens(), 320);
  EXPECT_EQ(config.PackedChannels(), 64);
  EXPECT_EQ(config.HeadDim(), 128);
  EXPECT_EQ(config.AdalnDim(), 256);

  config.dim = 64;
  EXPECT_EQ(config.AdalnDim(), 64);
}

TEST(ZImageTurboModelConfigTest, PopulateFromProtoOverridesAllFields) {
  lm::proto::ZImageTurbo proto;
  proto.set_seq_len(128);
  proto.set_img_size(512);
  proto.set_in_channels(8);
  proto.set_patch_size(4);
  proto.set_dim(1024);
  proto.set_prompt_dim(512);
  proto.set_default_steps(4);
  proto.set_vae_scaling_factor(0.5f);
  proto.set_vae_shift_factor(-0.25f);
  proto.add_axes_dims(16);
  proto.add_axes_dims(24);
  proto.add_axes_dims(24);
  proto.set_rope_theta(1000.0f);
  proto.set_shift(2.5f);
  proto.set_use_dynamic_shifting(true);
  proto.set_pad_token_id(0);
  proto.add_prefix_token_ids(1);
  proto.add_suffix_token_ids(2);
  proto.add_suffix_token_ids(3);

  ZImageTurboModelConfig config;
  PopulateZImageTurboConfigFromProto(proto, config);

  EXPECT_EQ(config.seq_len, 128);
  EXPECT_EQ(config.img_size, 512);
  EXPECT_EQ(config.in_channels, 8);
  EXPECT_EQ(config.patch_size, 4);
  EXPECT_EQ(config.dim, 1024);
  EXPECT_EQ(config.prompt_dim, 512);
  EXPECT_EQ(config.steps, 4);
  EXPECT_FLOAT_EQ(config.vae_scaling_factor, 0.5f);
  EXPECT_FLOAT_EQ(config.vae_shift_factor, -0.25f);
  EXPECT_THAT(config.axes_dims, ElementsAre(16, 24, 24));
  EXPECT_FLOAT_EQ(config.rope_theta, 1000.0f);
  EXPECT_FLOAT_EQ(config.shift, 2.5f);
  EXPECT_TRUE(config.use_dynamic_shifting);
  EXPECT_EQ(config.pad_token_id, 0);
  EXPECT_THAT(config.prefix_token_ids, ElementsAre(1));
  EXPECT_THAT(config.suffix_token_ids, ElementsAre(2, 3));
}

TEST(ZImageTurboModelConfigTest, PopulateFromProtoIgnoresInvalidValues) {
  lm::proto::ZImageTurbo proto;
  proto.set_seq_len(-1);
  proto.set_img_size(0);
  proto.add_axes_dims(32);
  proto.add_axes_dims(48);  // Only 2 elements instead of 3.
  proto.set_pad_token_id(-5);

  ZImageTurboModelConfig config;
  PopulateZImageTurboConfigFromProto(proto, config);

  EXPECT_EQ(config.seq_len, 64);
  EXPECT_EQ(config.img_size, 256);
  EXPECT_THAT(config.axes_dims, ElementsAre(32, 48, 48));
  EXPECT_EQ(config.pad_token_id, z_image_turbo::kDefaultPadTokenId);
}

TEST(ZImageTurboModelConfigTest, BuildCaptionAndPatchIds) {
  std::vector<float> cap_ids;
  BuildZImageCaptionIds(3, cap_ids);
  EXPECT_THAT(cap_ids, ElementsAre(1.0f, 0.0f, 0.0f, 2.0f, 0.0f, 0.0f, 3.0f,
                                   0.0f, 0.0f));

  std::vector<float> patch_ids;
  BuildZImagePatchIds(/*grid_h=*/2, /*grid_w=*/2, /*seq_len=*/32, patch_ids);
  EXPECT_THAT(patch_ids, ElementsAre(33.0f, 0.0f, 0.0f, 33.0f, 0.0f, 1.0f,
                                     33.0f, 1.0f, 0.0f, 33.0f, 1.0f, 1.0f));
}

TEST(ZImageTurboModelConfigTest, ComputePaddedCapLenRoundsToMultipleOf32) {
  EXPECT_EQ(ComputeZImagePaddedCapLen(0, 64), 64);
  EXPECT_EQ(ComputeZImagePaddedCapLen(-4, 64), 64);
  EXPECT_EQ(ComputeZImagePaddedCapLen(1, 64), 32);
  EXPECT_EQ(ComputeZImagePaddedCapLen(9, 64), 32);
  EXPECT_EQ(ComputeZImagePaddedCapLen(32, 64), 32);
  EXPECT_EQ(ComputeZImagePaddedCapLen(33, 64), 64);
  EXPECT_EQ(ComputeZImagePaddedCapLen(96, 64), 64);
  EXPECT_EQ(ComputeZImagePaddedCapLen(5, 8), 8);
}

TEST(ZImageTurboModelConfigTest, ComputeSigmasStaticMatchesDiffusers) {
  EXPECT_THAT(ComputeZImageSigmas(0, 256), IsEmpty());
  EXPECT_THAT(ComputeZImageSigmas(-2, 256), IsEmpty());

  EXPECT_THAT(ComputeZImageSigmas(1, 256), ElementsAre(1.0f, 0.0f));

  const std::vector<float> sigmas = ComputeZImageSigmas(8, 256, 3.0f, false);
  const std::vector<float> expected = {
      1.0f, 0.95454544f, 0.9f, 0.8333333f, 0.75f, 0.64285713f, 0.5f, 0.3f, 0.0f,
  };
  EXPECT_THAT(sigmas, Pointwise(FloatNear(1e-5f), expected));
}

TEST(ZImageTurboModelConfigTest, ComputeSigmasDynamicShiftingMonotonic) {
  const std::vector<float> sigmas_small =
      ComputeZImageSigmas(8, /*image_tokens=*/256, 3.0f,
                          /*use_dynamic_shifting=*/true);
  const std::vector<float> sigmas_large =
      ComputeZImageSigmas(8, /*image_tokens=*/4096, 3.0f,
                          /*use_dynamic_shifting=*/true);

  ASSERT_EQ(sigmas_small.size(), 9u);
  ASSERT_EQ(sigmas_large.size(), 9u);
  EXPECT_FLOAT_EQ(sigmas_small.front(), 1.0f);
  EXPECT_FLOAT_EQ(sigmas_small.back(), 0.0f);
  EXPECT_FLOAT_EQ(sigmas_large.front(), 1.0f);
  EXPECT_FLOAT_EQ(sigmas_large.back(), 0.0f);
  for (size_t i = 0; i + 1 < sigmas_small.size(); ++i) {
    EXPECT_GT(sigmas_small[i], sigmas_small[i + 1]);
    EXPECT_GT(sigmas_large[i], sigmas_large[i + 1]);
  }
  EXPECT_GT(sigmas_large[4], sigmas_small[4]);
}

TEST(ZImageTurboModelConfigTest, PackAndUnpackLatentsRoundTripAndValidation) {
  constexpr int kLatentH = 4;
  constexpr int kLatentW = 4;
  constexpr int kPatch = 2;
  constexpr int kChannels = 3;
  const size_t total = static_cast<size_t>(kChannels) * kLatentH * kLatentW;

  std::vector<float> nchw(total);
  for (size_t i = 0; i < total; ++i) {
    nchw[i] = static_cast<float>(i + 1);
  }
  std::vector<float> packed(total, 0.0f);
  std::vector<float> unpacked(total, 0.0f);

  ASSERT_OK(PackZImageLatents(nchw, absl::MakeSpan(packed), kLatentH, kLatentW,
                              kPatch, kChannels));
  // First token `(gh=0, gw=0)` packs `(ph, pw)` outer, `c` inner:
  // ph=0,pw=0 -> (0,0) across c=0,1,2 -> values 1, 17, 33.
  EXPECT_FLOAT_EQ(packed[0], 1.0f);
  EXPECT_FLOAT_EQ(packed[1], 17.0f);
  EXPECT_FLOAT_EQ(packed[2], 33.0f);
  // ph=0,pw=1 -> (0,1) across c=0,1,2 -> values 2, 18, 34.
  EXPECT_FLOAT_EQ(packed[3], 2.0f);
  EXPECT_FLOAT_EQ(packed[4], 18.0f);
  EXPECT_FLOAT_EQ(packed[5], 34.0f);

  ASSERT_OK(UnpackZImageLatents(packed, absl::MakeSpan(unpacked), kLatentH,
                                kLatentW, kPatch, kChannels));
  EXPECT_EQ(unpacked, nchw);

  EXPECT_THAT(PackZImageLatents(absl::MakeConstSpan(nchw).subspan(0, 4),
                                absl::MakeSpan(packed), kLatentH, kLatentW,
                                kPatch, kChannels),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(UnpackZImageLatents(packed, absl::MakeSpan(unpacked),
                                  /*latent_h=*/3, kLatentW, kPatch, kChannels),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace litert::omni::text2image

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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include "models/tongyi/z_image_turbo/runner/z_image_turbo.pb.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"

namespace litert::omni::text2image {
namespace {

constexpr int kAdalnEmbedDim = 256;

absl::Status ValidateLatentLayout(absl::Span<const float> src,
                                  absl::Span<float> dst, int latent_h,
                                  int latent_w, int patch_size,
                                  int in_channels) {
  if (latent_h <= 0 || latent_w <= 0 || patch_size <= 0 || in_channels <= 0) {
    return absl::InvalidArgumentError(
        "Latent dimensions, patch_size, and in_channels must be positive.");
  }
  if (latent_h % patch_size != 0 || latent_w % patch_size != 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("Latent spatial size (", latent_h, "x", latent_w,
                     ") must be divisible by patch_size (", patch_size, ")."));
  }
  const size_t expected_elems = static_cast<size_t>(in_channels) *
                                static_cast<size_t>(latent_h) *
                                static_cast<size_t>(latent_w);
  if (src.size() != expected_elems || dst.size() != expected_elems) {
    return absl::InvalidArgumentError(
        absl::StrCat("Latent buffer size mismatch: expected ", expected_elems,
                     ", got src=", src.size(), ", dst=", dst.size(), "."));
  }
  return absl::OkStatus();
}

}  // namespace

int ZImageTurboModelConfig::LatentSize() const {
  return img_size / z_image_turbo::kVaeSpatialDownsampleFactor;
}

int ZImageTurboModelConfig::GridSize() const {
  if (patch_size <= 0) {
    return 0;
  }
  return LatentSize() / patch_size;
}

int ZImageTurboModelConfig::ImageTokens() const {
  const int grid = GridSize();
  return grid * grid;
}

int ZImageTurboModelConfig::TotalTokens() const {
  return ImageTokens() + seq_len;
}

int ZImageTurboModelConfig::PackedChannels() const {
  return patch_size * patch_size * in_channels;
}

int ZImageTurboModelConfig::HeadDim() const {
  return axes_dims[0] + axes_dims[1] + axes_dims[2];
}

int ZImageTurboModelConfig::AdalnDim() const {
  return std::min(dim, kAdalnEmbedDim);
}

void PopulateZImageTurboConfigFromProto(const lm::proto::ZImageTurbo& proto,
                                        ZImageTurboModelConfig& config) {
  if (proto.has_seq_len() && proto.seq_len() > 0) {
    config.seq_len = proto.seq_len();
  }
  if (proto.has_img_size() && proto.img_size() > 0) {
    config.img_size = proto.img_size();
  }
  if (proto.has_in_channels() && proto.in_channels() > 0) {
    config.in_channels = proto.in_channels();
  }
  if (proto.has_patch_size() && proto.patch_size() > 0) {
    config.patch_size = proto.patch_size();
  }
  if (proto.has_dim() && proto.dim() > 0) {
    config.dim = proto.dim();
  }
  if (proto.has_prompt_dim() && proto.prompt_dim() > 0) {
    config.prompt_dim = proto.prompt_dim();
  }
  if (proto.has_default_steps() && proto.default_steps() > 0) {
    config.steps = proto.default_steps();
  }
  if (proto.has_vae_scaling_factor() && proto.vae_scaling_factor() > 0.0f) {
    config.vae_scaling_factor = proto.vae_scaling_factor();
  }
  if (proto.has_vae_shift_factor()) {
    config.vae_shift_factor = proto.vae_shift_factor();
  }
  if (proto.axes_dims_size() == 3 && proto.axes_dims(0) > 0 &&
      proto.axes_dims(1) > 0 && proto.axes_dims(2) > 0) {
    config.axes_dims = {proto.axes_dims(0), proto.axes_dims(1),
                        proto.axes_dims(2)};
  }
  if (proto.has_rope_theta() && proto.rope_theta() > 0.0f) {
    config.rope_theta = proto.rope_theta();
  }
  if (proto.has_shift() && proto.shift() > 0.0f) {
    config.shift = proto.shift();
  }
  if (proto.has_use_dynamic_shifting()) {
    config.use_dynamic_shifting = proto.use_dynamic_shifting();
  }
  if (proto.has_pad_token_id() && proto.pad_token_id() >= 0) {
    config.pad_token_id = proto.pad_token_id();
  }
  if (!proto.prefix_token_ids().empty()) {
    config.prefix_token_ids.assign(proto.prefix_token_ids().begin(),
                                   proto.prefix_token_ids().end());
  }
  if (!proto.suffix_token_ids().empty()) {
    config.suffix_token_ids.assign(proto.suffix_token_ids().begin(),
                                   proto.suffix_token_ids().end());
  }
}

void BuildZImageCaptionIds(const int seq_len, std::vector<float>& ids) {
  if (seq_len <= 0) {
    ids.clear();
    return;
  }
  ids.assign(static_cast<size_t>(seq_len) * 3, 0.0f);
  for (int i = 0; i < seq_len; ++i) {
    ids[static_cast<size_t>(i) * 3 + 0] = static_cast<float>(i + 1);
  }
}

void BuildZImagePatchIds(const int grid_h, const int grid_w, const int seq_len,
                         std::vector<float>& ids) {
  if (grid_h <= 0 || grid_w <= 0) {
    ids.clear();
    return;
  }
  ids.resize(static_cast<size_t>(grid_h) * static_cast<size_t>(grid_w) * 3);
  size_t idx = 0;
  const float t_coord = static_cast<float>(seq_len + 1);
  for (int h = 0; h < grid_h; ++h) {
    for (int w = 0; w < grid_w; ++w) {
      ids[idx++] = t_coord;
      ids[idx++] = static_cast<float>(h);
      ids[idx++] = static_cast<float>(w);
    }
  }
}

int ComputeZImagePaddedCapLen(const int valid_tokens, const int seq_len) {
  if (seq_len <= 0) {
    return 0;
  }
  if (valid_tokens <= 0) {
    return seq_len;
  }
  const int rounded = ((valid_tokens + z_image_turbo::kSeqMultiOf - 1) /
                       z_image_turbo::kSeqMultiOf) *
                      z_image_turbo::kSeqMultiOf;
  return std::min(seq_len, std::max(1, rounded));
}

std::vector<float> ComputeZImageSigmas(const int steps, const int image_tokens,
                                       const float shift,
                                       const bool use_dynamic_shifting) {
  if (steps <= 0) {
    return {};
  }
  constexpr double kBaseSeqLen = 256.0;
  constexpr double kMaxSeqLen = 4096.0;
  constexpr double kBaseShift = 0.5;
  constexpr double kMaxShift = 1.15;
  const double slope = (kMaxShift - kBaseShift) / (kMaxSeqLen - kBaseSeqLen);
  const double intercept = kBaseShift - slope * kBaseSeqLen;
  const double mu = static_cast<double>(image_tokens) * slope + intercept;
  const double exp_mu = std::exp(mu);
  const double shift_d = static_cast<double>(shift);

  std::vector<float> sigmas;
  sigmas.reserve(static_cast<size_t>(steps) + 1);
  for (int i = 0; i < steps; ++i) {
    const double lin = steps == 1 ? 1.0
                                  : 1.0 - static_cast<double>(i) *
                                              (1.0 - 1.0 / steps) / (steps - 1);
    if (lin >= 1.0) {
      sigmas.push_back(1.0f);
    } else if (use_dynamic_shifting) {
      const double shifted = exp_mu / (exp_mu + (1.0 / lin - 1.0));
      sigmas.push_back(static_cast<float>(shifted));
    } else {
      const double shifted = (shift_d * lin) / (1.0 + (shift_d - 1.0) * lin);
      sigmas.push_back(static_cast<float>(shifted));
    }
  }
  sigmas.push_back(0.0f);
  return sigmas;
}

absl::Status PackZImageLatents(absl::Span<const float> src,
                               absl::Span<float> dst, const int latent_h,
                               const int latent_w, const int patch_size,
                               const int in_channels) {
  absl::Status status = ValidateLatentLayout(src, dst, latent_h, latent_w,
                                             patch_size, in_channels);
  if (!status.ok()) {
    return status;
  }
  const int grid_h = latent_h / patch_size;
  const int grid_w = latent_w / patch_size;
  const int packed_ch = patch_size * patch_size * in_channels;
  for (int gh = 0; gh < grid_h; ++gh) {
    for (int gw = 0; gw < grid_w; ++gw) {
      const size_t token_offset =
          static_cast<size_t>(gh * grid_w + gw) * packed_ch;
      for (int ph = 0; ph < patch_size; ++ph) {
        for (int pw = 0; pw < patch_size; ++pw) {
          const int h = gh * patch_size + ph;
          const int w = gw * patch_size + pw;
          const size_t sub_offset =
              static_cast<size_t>(ph * patch_size + pw) * in_channels;
          for (int c = 0; c < in_channels; ++c) {
            const size_t src_idx =
                static_cast<size_t>((c * latent_h + h) * latent_w + w);
            dst[token_offset + sub_offset + c] = src[src_idx];
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status UnpackZImageLatents(absl::Span<const float> src,
                                 absl::Span<float> dst, const int latent_h,
                                 const int latent_w, const int patch_size,
                                 const int in_channels) {
  absl::Status status = ValidateLatentLayout(src, dst, latent_h, latent_w,
                                             patch_size, in_channels);
  if (!status.ok()) {
    return status;
  }
  const int grid_h = latent_h / patch_size;
  const int grid_w = latent_w / patch_size;
  const int packed_ch = patch_size * patch_size * in_channels;
  for (int gh = 0; gh < grid_h; ++gh) {
    for (int gw = 0; gw < grid_w; ++gw) {
      const size_t token_offset =
          static_cast<size_t>(gh * grid_w + gw) * packed_ch;
      for (int ph = 0; ph < patch_size; ++ph) {
        for (int pw = 0; pw < patch_size; ++pw) {
          const int h = gh * patch_size + ph;
          const int w = gw * patch_size + pw;
          const size_t sub_offset =
              static_cast<size_t>(ph * patch_size + pw) * in_channels;
          for (int c = 0; c < in_channels; ++c) {
            const size_t dst_idx =
                static_cast<size_t>((c * latent_h + h) * latent_w + w);
            dst[dst_idx] = src[token_offset + sub_offset + c];
          }
        }
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::text2image

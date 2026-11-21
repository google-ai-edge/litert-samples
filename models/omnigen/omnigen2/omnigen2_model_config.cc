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

#include "models/omnigen/omnigen2/omnigen2_model_config.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "models/omnigen/omnigen2/omnigen2.pb.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::text2image::omnigen2 {
namespace {

constexpr int kMaxOmniGen2SeqLen = 4096;
constexpr int kMaxOmniGen2ImgSize = 4096;
constexpr int kMaxOmniGen2InChannels = 256;
constexpr int kMaxOmniGen2HiddenSize = 65536;
constexpr int kMaxOmniGen2Steps = 1000;

}  // namespace

std::vector<int32_t> DefaultOmniGen2PrefixTokenIds() {
  // `<|im_start|>system\nYou are a helpful assistant that generates
  // high-quality images based on user instructions.<|im_end|>\n<|im_start|>`
  // in `Qwen2.5-VL-3B-Instruct` (`OmniGen2/OmniGen2/processor/tokenizer.json`):
  // 151644=`<|im_start|>`, 8948=`system`, 198=`\n`, 2610=`You`, 525=` are`,
  // 264=` a`, 10950=` helpful`, 17847=` assistant`, 429=` that`,
  // 26885=` generates`, 1550=` high`, 22092=`-quality`, 5335=` images`,
  // 3118=` based`, 389=` on`, 1196=` user`, 11221=` instructions`, 13=`.`,
  // 151645=`<|im_end|>`, 198=`\n`, 151644=`<|im_start|>`.
  return {
      151644, 8948, 198,  2610, 525,  264,   10950, 17847,  429, 26885,  1550,
      22092,  5335, 3118, 389,  1196, 11221, 13,    151645, 198, 151644,
  };
}

std::vector<int32_t> DefaultOmniGen2SuffixTokenIds() {
  // `<|im_end|>\n` in `Qwen2.5-VL-3B-Instruct`.
  return {151645, 198};
}

std::vector<int32_t> ResolveOmniGen2PrefixTokenIds(
    support::Tokenizer* absl_nullable tokenizer) {
  if (tokenizer == nullptr) {
    return DefaultOmniGen2PrefixTokenIds();
  }
  const absl::StatusOr<int> im_start = tokenizer->TokenToId("<|im_start|>");
  const absl::StatusOr<int> im_end = tokenizer->TokenToId("<|im_end|>");
  const absl::StatusOr<support::TokenIds> sys_body = tokenizer->TextToTokenIds(
      "system\nYou are a helpful assistant that generates high-quality images "
      "based on user instructions.");
  const absl::StatusOr<support::TokenIds> nl = tokenizer->TextToTokenIds("\n");
  if (!im_start.ok() || !im_end.ok() || !sys_body.ok() || !nl.ok() ||
      sys_body->size() < 4 || nl->empty()) {
    return DefaultOmniGen2PrefixTokenIds();
  }
  std::vector<int32_t> prefix;
  prefix.reserve(3 + sys_body->size() + nl->size());
  prefix.push_back(static_cast<int32_t>(*im_start));
  for (int id : *sys_body) {
    prefix.push_back(static_cast<int32_t>(id));
  }
  prefix.push_back(static_cast<int32_t>(*im_end));
  for (int id : *nl) {
    prefix.push_back(static_cast<int32_t>(id));
  }
  prefix.push_back(static_cast<int32_t>(*im_start));
  return prefix;
}

std::vector<int32_t> ResolveOmniGen2SuffixTokenIds(
    support::Tokenizer* absl_nullable tokenizer) {
  if (tokenizer == nullptr) {
    return DefaultOmniGen2SuffixTokenIds();
  }
  const absl::StatusOr<int> im_end = tokenizer->TokenToId("<|im_end|>");
  const absl::StatusOr<support::TokenIds> nl = tokenizer->TextToTokenIds("\n");
  if (!im_end.ok() || !nl.ok() || nl->empty()) {
    return DefaultOmniGen2SuffixTokenIds();
  }
  std::vector<int32_t> suffix;
  suffix.reserve(1 + nl->size());
  suffix.push_back(static_cast<int32_t>(*im_end));
  for (int id : *nl) {
    suffix.push_back(static_cast<int32_t>(id));
  }
  return suffix;
}

int32_t ResolveOmniGen2PadTokenId(support::Tokenizer* absl_nullable tokenizer) {
  if (tokenizer == nullptr) {
    return kOmniGen2PadTokenId;
  }
  if (const absl::StatusOr<int> id = tokenizer->TokenToId("<|endoftext|>");
      id.ok()) {
    return static_cast<int32_t>(*id);
  }
  if (const absl::StatusOr<int> id = tokenizer->TokenToId("<|pad|>"); id.ok()) {
    return static_cast<int32_t>(*id);
  }
  return kOmniGen2PadTokenId;
}

absl::Status ValidateOmniGen2ModelConfig(const OmniGen2ModelConfig& config) {
  if (config.img_size <= 0 || config.img_size > kMaxOmniGen2ImgSize ||
      config.img_size % 16 != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2 img_size (%d) must be a positive multiple of 16 and at most "
        "%d.",
        config.img_size, kMaxOmniGen2ImgSize));
  }
  if (config.seq_len <= 0 || config.seq_len > kMaxOmniGen2SeqLen) {
    return absl::InvalidArgumentError(
        absl::StrFormat("OmniGen2 seq_len (%d) must be in [1, %d].",
                        config.seq_len, kMaxOmniGen2SeqLen));
  }
  if (config.in_channels <= 0 || config.in_channels > kMaxOmniGen2InChannels ||
      config.patch_size <= 0 || config.patch_size > 16 ||
      (config.img_size / 8) % config.patch_size != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2 in_channels (%d) and patch_size (%d) must be valid positive "
        "values dividing latent size (%d).",
        config.in_channels, config.patch_size, config.img_size / 8));
  }
  const int expected_packed_ch =
      config.in_channels * config.patch_size * config.patch_size;
  if (config.packed_ch != expected_packed_ch) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2 packed_ch (%d) must equal in_channels * patch_size^2 (%d).",
        config.packed_ch, expected_packed_ch));
  }
  if (config.text_hidden_size <= 0 ||
      config.text_hidden_size > kMaxOmniGen2HiddenSize ||
      config.dit_hidden_size <= 0 ||
      config.dit_hidden_size > kMaxOmniGen2HiddenSize) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2 text_hidden_size (%d) and dit_hidden_size (%d) must be "
        "within [1, %d].",
        config.text_hidden_size, config.dit_hidden_size,
        kMaxOmniGen2HiddenSize));
  }
  if (config.dit_num_heads <= 0 || config.dit_num_kv_heads <= 0 ||
      config.dit_num_heads % config.dit_num_kv_heads != 0 ||
      config.dit_head_dim <= 0 ||
      config.dit_num_heads * config.dit_head_dim != config.dit_hidden_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2 attention dimensions invalid: dit_hidden_size=%d, "
        "dit_num_heads=%d, dit_num_kv_heads=%d, dit_head_dim=%d.",
        config.dit_hidden_size, config.dit_num_heads, config.dit_num_kv_heads,
        config.dit_head_dim));
  }
  int rope_sum = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (config.axes_dims_rope[axis] <= 0 ||
        config.axes_dims_rope[axis] % 2 != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "OmniGen2 axes_dims_rope[%d] (%d) must be a positive even integer.",
          axis, config.axes_dims_rope[axis]));
    }
    rope_sum += config.axes_dims_rope[axis];
  }
  if (rope_sum != config.dit_head_dim) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2 sum(axes_dims_rope) (%d) must equal dit_head_dim (%d).",
        rope_sum, config.dit_head_dim));
  }
  if (!std::isfinite(config.rope_theta) || config.rope_theta <= 0.0f ||
      !std::isfinite(config.text_rope_theta) ||
      config.text_rope_theta <= 0.0f || !std::isfinite(config.timestep_scale) ||
      config.timestep_scale <= 0.0f ||
      !std::isfinite(config.vae_scaling_factor) ||
      config.vae_scaling_factor <= 0.0f ||
      !std::isfinite(config.vae_shift_factor)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2 rope_theta (%f), text_rope_theta (%f), timestep_scale (%f), "
        "vae_scaling_factor (%f) must be positive finite values, and "
        "vae_shift_factor (%f) must be finite.",
        config.rope_theta, config.text_rope_theta, config.timestep_scale,
        config.vae_scaling_factor, config.vae_shift_factor));
  }
  if (config.steps <= 0 || config.steps > kMaxOmniGen2Steps) {
    return absl::InvalidArgumentError(
        absl::StrFormat("OmniGen2 steps (%d) must be in [1, %d].", config.steps,
                        kMaxOmniGen2Steps));
  }
  if (config.text_guidance_scale < 0.0f ||
      !std::isfinite(config.text_guidance_scale)) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2 text_guidance_scale (%f) must be finite and non-negative.",
        config.text_guidance_scale));
  }
  return absl::OkStatus();
}

void PopulateOmniGen2ConfigFromProto(
    const prototypes::litert_models::omnigen2::OmniGen2& proto,
    OmniGen2ModelConfig& config) {
  if (proto.has_seq_len() && proto.seq_len() > 0 &&
      proto.seq_len() <= kMaxOmniGen2SeqLen) {
    config.seq_len = proto.seq_len();
  }
  if (proto.has_img_size() && proto.img_size() > 0 &&
      proto.img_size() <= kMaxOmniGen2ImgSize) {
    config.img_size = proto.img_size();
  }
  if (proto.has_in_channels() && proto.in_channels() > 0 &&
      proto.in_channels() <= kMaxOmniGen2InChannels) {
    config.in_channels = proto.in_channels();
  }
  if (proto.has_patch_size() && proto.patch_size() > 0 &&
      proto.patch_size() <= 16) {
    config.patch_size = proto.patch_size();
  }
  config.packed_ch = config.in_channels * config.patch_size * config.patch_size;

  if (proto.has_hidden_size() && proto.hidden_size() > 0 &&
      proto.hidden_size() <= kMaxOmniGen2HiddenSize) {
    config.dit_hidden_size = proto.hidden_size();
  }
  if (proto.has_prompt_dim() && proto.prompt_dim() > 0 &&
      proto.prompt_dim() <= kMaxOmniGen2HiddenSize) {
    config.text_hidden_size = proto.prompt_dim();
  }
  if (proto.has_default_steps() && proto.default_steps() > 0 &&
      proto.default_steps() <= kMaxOmniGen2Steps) {
    config.steps = proto.default_steps();
  }
  if (proto.has_vae_scaling_factor() && proto.vae_scaling_factor() > 0.0f) {
    config.vae_scaling_factor = proto.vae_scaling_factor();
  }
  if (proto.has_vae_shift_factor()) {
    config.vae_shift_factor = proto.vae_shift_factor();
  }
  if (proto.has_timestep_scale() && proto.timestep_scale() > 0.0f) {
    config.timestep_scale = proto.timestep_scale();
  }
  if (proto.axes_dim_rope_size() == 3 && proto.axes_dim_rope(0) > 0 &&
      proto.axes_dim_rope(1) > 0 && proto.axes_dim_rope(2) > 0 &&
      proto.axes_dim_rope(0) % 2 == 0 && proto.axes_dim_rope(1) % 2 == 0 &&
      proto.axes_dim_rope(2) % 2 == 0) {
    config.axes_dims_rope = {
        proto.axes_dim_rope(0),
        proto.axes_dim_rope(1),
        proto.axes_dim_rope(2),
    };
    config.dit_head_dim = config.axes_dims_rope[0] + config.axes_dims_rope[1] +
                          config.axes_dims_rope[2];
  }
  if (config.dit_head_dim > 0 &&
      config.dit_hidden_size % config.dit_head_dim == 0) {
    config.dit_num_heads = config.dit_hidden_size / config.dit_head_dim;
    if (config.dit_num_heads % config.dit_num_kv_heads != 0) {
      config.dit_num_kv_heads = 1;
    }
  }
  if (proto.has_dynamic_time_shift()) {
    config.dynamic_time_shift = proto.dynamic_time_shift();
  }
  if (proto.has_text_guidance_scale() && proto.text_guidance_scale() >= 0.0f &&
      std::isfinite(proto.text_guidance_scale())) {
    config.text_guidance_scale = proto.text_guidance_scale();
  }
}

absl::StatusOr<std::vector<float>> ComputeOmniGen2Timesteps(
    int steps, int height, int width, bool dynamic_time_shift) {
  if (steps <= 0 || steps > kMaxOmniGen2Steps) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ComputeOmniGen2Timesteps steps (%d) must be in [1, %d].", steps,
        kMaxOmniGen2Steps));
  }
  if (height <= 0 || height % 16 != 0 || width <= 0 || width % 16 != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "ComputeOmniGen2Timesteps height (%d) and width (%d) must be positive "
        "multiples of 16.",
        height, width));
  }
  const double latent_h = static_cast<double>(height / 8);
  const double latent_w = static_cast<double>(width / 8);
  const double m = std::sqrt(latent_h * latent_w) / 40.0;

  std::vector<float> timesteps(static_cast<size_t>(steps) + 1, 0.0f);
  for (int i = 0; i <= steps; ++i) {
    double t = static_cast<double>(i) / static_cast<double>(steps);
    if (dynamic_time_shift && i > 0 && i < steps) {
      t = t / (m - m * t + t);
    }
    timesteps[i] = static_cast<float>(t);
  }
  return timesteps;
}

absl::StatusOr<std::vector<int32_t>> BuildOmniGen2CaptionPositionIds(
    int seq_len, int active_cap_len) {
  if (seq_len <= 0 || seq_len > kMaxOmniGen2SeqLen) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "BuildOmniGen2CaptionPositionIds seq_len (%d) must be in [1, %d].",
        seq_len, kMaxOmniGen2SeqLen));
  }
  if (active_cap_len < 0 || active_cap_len > seq_len) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "BuildOmniGen2CaptionPositionIds active_cap_len (%d) must be in "
        "[0, seq_len (%d)].",
        active_cap_len, seq_len));
  }
  std::vector<int32_t> pos_ids(static_cast<size_t>(seq_len) * 3, 0);
  for (int i = 0; i < active_cap_len; ++i) {
    pos_ids[3 * i + 0] = i;
    pos_ids[3 * i + 1] = i;
    pos_ids[3 * i + 2] = i;
  }
  return pos_ids;
}

absl::StatusOr<std::vector<int32_t>> BuildOmniGen2ImagePositionIds(
    int grid_h, int grid_w, int id_offset) {
  if (grid_h <= 0 || grid_w <= 0 || id_offset < 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "BuildOmniGen2ImagePositionIds grid_h (%d) and grid_w (%d) must be "
        "positive and id_offset (%d) must be non-negative.",
        grid_h, grid_w, id_offset));
  }
  const size_t num_tokens =
      static_cast<size_t>(grid_h) * static_cast<size_t>(grid_w);
  std::vector<int32_t> pos_ids(num_tokens * 3, 0);
  size_t idx = 0;
  for (int h = 0; h < grid_h; ++h) {
    for (int w = 0; w < grid_w; ++w) {
      pos_ids[3 * idx + 0] = id_offset;
      pos_ids[3 * idx + 1] = h;
      pos_ids[3 * idx + 2] = w;
      ++idx;
    }
  }
  return pos_ids;
}

absl::StatusOr<std::pair<std::vector<float>, std::vector<float>>>
BuildOmniGen2RotaryPosEmbed(absl::Span<const int32_t> pos_ids, int num_tokens,
                            const std::array<int, 3>& axes_dims_rope,
                            float theta, int active_tokens) {
  if (num_tokens <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "BuildOmniGen2RotaryPosEmbed num_tokens (%d) must be positive.",
        num_tokens));
  }
  if (active_tokens > num_tokens) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "BuildOmniGen2RotaryPosEmbed active_tokens (%d) must not exceed "
        "num_tokens (%d).",
        active_tokens, num_tokens));
  }
  if (pos_ids.size() != static_cast<size_t>(num_tokens) * 3) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "BuildOmniGen2RotaryPosEmbed pos_ids size (%d) must equal 3 * "
        "num_tokens (%d).",
        pos_ids.size(), 3 * num_tokens));
  }
  if (theta <= 0.0f) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "BuildOmniGen2RotaryPosEmbed theta (%f) must be positive.", theta));
  }
  int head_dim = 0;
  for (int axis = 0; axis < 3; ++axis) {
    if (axes_dims_rope[axis] <= 0 || axes_dims_rope[axis] % 2 != 0) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "BuildOmniGen2RotaryPosEmbed axes_dims_rope[%d] (%d) must be a "
          "positive even integer.",
          axis, axes_dims_rope[axis]));
    }
    head_dim += axes_dims_rope[axis];
  }

  std::array<std::vector<double>, 3> inv_freqs;
  for (int axis = 0; axis < 3; ++axis) {
    const int axis_dim = axes_dims_rope[axis];
    const int half_dim = axis_dim / 2;
    inv_freqs[axis].resize(half_dim);
    for (int k = 0; k < half_dim; ++k) {
      inv_freqs[axis][k] = 1.0 / std::pow(static_cast<double>(theta),
                                          static_cast<double>(2 * k) /
                                              static_cast<double>(axis_dim));
    }
  }

  const size_t total_elements =
      static_cast<size_t>(num_tokens) * static_cast<size_t>(head_dim);
  std::vector<float> cos_out(total_elements, 0.0f);
  std::vector<float> sin_out(total_elements, 0.0f);

  const int eff_tokens = (active_tokens >= 0 && active_tokens < num_tokens)
                             ? active_tokens
                             : num_tokens;
  for (int t = 0; t < eff_tokens; ++t) {
    const size_t out_base =
        static_cast<size_t>(t) * static_cast<size_t>(head_dim);
    int axis_offset = 0;
    for (int axis = 0; axis < 3; ++axis) {
      const double pos = static_cast<double>(pos_ids[3 * t + axis]);
      const int half_dim = axes_dims_rope[axis] / 2;
      for (int k = 0; k < half_dim; ++k) {
        const double angle = pos * inv_freqs[axis][k];
        const float c = static_cast<float>(std::cos(angle));
        const float s = static_cast<float>(std::sin(angle));
        const size_t idx = out_base + axis_offset + 2 * k;
        cos_out[idx] = c;
        cos_out[idx + 1] = c;
        sin_out[idx] = s;
        sin_out[idx + 1] = s;
      }
      axis_offset += axes_dims_rope[axis];
    }
  }
  return std::make_pair(std::move(cos_out), std::move(sin_out));
}

absl::StatusOr<std::pair<std::vector<float>, std::vector<float>>>
BuildOmniGen2AttentionMasks(int seq_len, int active_cap_len, int img_tokens,
                            float neg_inf) {
  if (seq_len <= 0 || seq_len > kMaxOmniGen2SeqLen || active_cap_len < 0 ||
      active_cap_len > seq_len || img_tokens <= 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "BuildOmniGen2AttentionMasks invalid dimensions: seq_len=%d, "
        "active_cap_len=%d, img_tokens=%d.",
        seq_len, active_cap_len, img_tokens));
  }
  std::vector<float> cap_mask(static_cast<size_t>(seq_len), 0.0f);
  for (int k = active_cap_len; k < seq_len; ++k) {
    cap_mask[k] = neg_inf;
  }

  const int joint_tokens = seq_len + img_tokens;
  std::vector<float> joint_mask(static_cast<size_t>(joint_tokens), 0.0f);
  for (int k = active_cap_len; k < seq_len; ++k) {
    joint_mask[k] = neg_inf;
  }
  return std::make_pair(std::move(cap_mask), std::move(joint_mask));
}

absl::StatusOr<std::vector<float>> PackOmniGen2Latents(
    absl::Span<const float> latents_nchw, int channels, int latent_h,
    int latent_w, int patch_size) {
  if (channels <= 0 || latent_h <= 0 || latent_w <= 0 || patch_size <= 0 ||
      latent_h % patch_size != 0 || latent_w % patch_size != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "PackOmniGen2Latents invalid dimensions: channels=%d, latent_h=%d, "
        "latent_w=%d, patch_size=%d.",
        channels, latent_h, latent_w, patch_size));
  }
  const size_t expected_size = static_cast<size_t>(channels) *
                               static_cast<size_t>(latent_h) *
                               static_cast<size_t>(latent_w);
  if (latents_nchw.size() != expected_size) {
    return absl::InvalidArgumentError(
        absl::StrFormat("PackOmniGen2Latents expected %d elements, got %d.",
                        expected_size, latents_nchw.size()));
  }

  const int grid_h = latent_h / patch_size;
  const int grid_w = latent_w / patch_size;
  const int packed_ch = patch_size * patch_size * channels;
  std::vector<float> packed(expected_size, 0.0f);

  // Matches `einops.rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)')`.
  for (int gh = 0; gh < grid_h; ++gh) {
    for (int gw = 0; gw < grid_w; ++gw) {
      const size_t token_idx =
          static_cast<size_t>(gh * grid_w + gw) * packed_ch;
      for (int p1 = 0; p1 < patch_size; ++p1) {
        const int y = gh * patch_size + p1;
        for (int p2 = 0; p2 < patch_size; ++p2) {
          const int x = gw * patch_size + p2;
          const size_t patch_offset =
              static_cast<size_t>(p1 * patch_size + p2) * channels;
          for (int c = 0; c < channels; ++c) {
            const size_t nchw_idx =
                (static_cast<size_t>(c) * latent_h + y) * latent_w + x;
            packed[token_idx + patch_offset + c] = latents_nchw[nchw_idx];
          }
        }
      }
    }
  }
  return packed;
}

absl::StatusOr<std::vector<float>> UnpackAndScaleOmniGen2Latents(
    absl::Span<const float> packed_latents, int channels, int latent_h,
    int latent_w, int patch_size, float vae_scaling_factor,
    float vae_shift_factor) {
  if (channels <= 0 || latent_h <= 0 || latent_w <= 0 || patch_size <= 0 ||
      latent_h % patch_size != 0 || latent_w % patch_size != 0) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "UnpackAndScaleOmniGen2Latents invalid dimensions: channels=%d, "
        "latent_h=%d, latent_w=%d, patch_size=%d.",
        channels, latent_h, latent_w, patch_size));
  }
  if (vae_scaling_factor <= 0.0f) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "UnpackAndScaleOmniGen2Latents vae_scaling_factor (%f) must be "
        "positive.",
        vae_scaling_factor));
  }
  const size_t expected_size = static_cast<size_t>(channels) *
                               static_cast<size_t>(latent_h) *
                               static_cast<size_t>(latent_w);
  if (packed_latents.size() != expected_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "UnpackAndScaleOmniGen2Latents expected %d elements, got %d.",
        expected_size, packed_latents.size()));
  }

  const int grid_h = latent_h / patch_size;
  const int grid_w = latent_w / patch_size;
  const int packed_ch = patch_size * patch_size * channels;
  std::vector<float> nchw(expected_size, 0.0f);

  for (int gh = 0; gh < grid_h; ++gh) {
    for (int gw = 0; gw < grid_w; ++gw) {
      const size_t token_idx =
          static_cast<size_t>(gh * grid_w + gw) * packed_ch;
      for (int p1 = 0; p1 < patch_size; ++p1) {
        const int y = gh * patch_size + p1;
        for (int p2 = 0; p2 < patch_size; ++p2) {
          const int x = gw * patch_size + p2;
          const size_t patch_offset =
              static_cast<size_t>(p1 * patch_size + p2) * channels;
          for (int c = 0; c < channels; ++c) {
            const size_t nchw_idx =
                (static_cast<size_t>(c) * latent_h + y) * latent_w + x;
            const float z_packed = packed_latents[token_idx + patch_offset + c];
            nchw[nchw_idx] = z_packed / vae_scaling_factor + vae_shift_factor;
          }
        }
      }
    }
  }
  return nchw;
}

}  // namespace litert::omni::text2image::omnigen2

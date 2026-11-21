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

#ifndef MODELS_OMNIGEN_OMNIGEN2_OMNIGEN2_MODEL_CONFIG_H_
#define MODELS_OMNIGEN_OMNIGEN2_OMNIGEN2_MODEL_CONFIG_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "models/omnigen/omnigen2/omnigen2.pb.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "runtime/executor/executor_settings_base.h"
#include "support/tokenizer/tokenizer.h"

namespace litert::omni::text2image::omnigen2 {

// Official `OmniGen2Pipeline._apply_chat_template` system prompt.
inline constexpr absl::string_view kOmniGen2SystemPrompt =
    "You are a helpful assistant that generates high-quality images based on "
    "user instructions.";

// Default `<|endoftext|>` padding token ID for `Qwen2.5-VL-3B-Instruct`.
inline constexpr int32_t kOmniGen2PadTokenId = 151643;

// Number of sharded main transformer block graphs (`dit_single_block_0..5`).
inline constexpr size_t kNumMainBlockShards = 6;

// Named `GenericBinaryData` section keys for OmniGen2-specific sharded TFLite
// subgraphs that are not part of the closed
// `Text2ImageMetadata::TfLiteModelType` enum.
inline constexpr absl::string_view kSectionDitCapEmbedder = "dit_cap_embedder";
inline constexpr absl::string_view kSectionDitContextRefiner =
    "dit_context_refiner";
inline constexpr absl::string_view kSectionDitXEmbedder = "dit_x_embedder";
inline constexpr absl::string_view kSectionDitNoiseRefiner =
    "dit_noise_refiner";
inline constexpr absl::string_view kSectionDitSingleBlock4 =
    "dit_single_block_4";
inline constexpr absl::string_view kSectionDitSingleBlock5 =
    "dit_single_block_5";

// Returns the default 21-token ChatML system + user prefix token IDs for
// `OmniGen2/OmniGen2` (`Qwen2.5-VL-3B-Instruct` tokenizer):
// `<|im_start|>system\nYou are a helpful assistant that generates high-quality
// images based on user instructions.<|im_end|>\n<|im_start|>`
// (followed by `prompt_prefix = "user\n"` + user prompt).
std::vector<int32_t> DefaultOmniGen2PrefixTokenIds();

// Returns the default 2-token ChatML suffix token IDs for `OmniGen2/OmniGen2`:
// `<|im_end|>\n` (`{151645, 198}`).
std::vector<int32_t> DefaultOmniGen2SuffixTokenIds();

// Resolves the ChatML system + user prefix token IDs dynamically using
// `tokenizer` when `tokenizer` supports ChatML control tokens (`<|im_start|>`,
// `<|im_end|>`), falling back to `DefaultOmniGen2PrefixTokenIds()` otherwise.
std::vector<int32_t> ResolveOmniGen2PrefixTokenIds(
    support::Tokenizer* absl_nullable tokenizer);

// Resolves the ChatML suffix token IDs (`<|im_end|>\n`) dynamically using
// `tokenizer`, falling back to `DefaultOmniGen2SuffixTokenIds()` otherwise.
std::vector<int32_t> ResolveOmniGen2SuffixTokenIds(
    support::Tokenizer* absl_nullable tokenizer);

// Resolves the `<|endoftext|>` (or `<|pad|>`) padding token ID dynamically
// using `tokenizer`, falling back to `kOmniGen2PadTokenId` otherwise.
int32_t ResolveOmniGen2PadTokenId(support::Tokenizer* absl_nullable tokenizer);

// Model-specific configuration settings for the `OmniGen2/OmniGen2` unified
// image generation pipeline (`Qwen2.5-VL-3B-Instruct` text backbone +
// `OmniGen2Transformer2DModel` 32-layer DiT with 2-layer context/noise
// refiners and 3D `(id, h, w)` Omni-RoPE + 16-channel `AutoencoderKL` VAE).
struct OmniGen2ModelConfig {
  // Padded text sequence length expected by text encoder and DiT shards.
  int seq_len = 256;
  // Target output image width and height in pixels (multiple of 16).
  int img_size = 512;
  // Number of VAE latent channels (`16`).
  int in_channels = 16;
  // Spatial patchify factor (`2`).
  int patch_size = 2;
  // Packed latent channel count (`in_channels * patch_size * patch_size = 64`).
  int packed_ch = 64;
  // `Qwen2.5-VL-3B-Instruct` text hidden dimension (`2048`).
  int text_hidden_size = 2048;
  // `OmniGen2Transformer2DModel` hidden dimension (`2520`).
  int dit_hidden_size = 2520;
  // Number of query attention heads (`21`).
  int dit_num_heads = 21;
  // Number of key/value attention heads (`7`).
  int dit_num_kv_heads = 7;
  // Attention head dimension (`120`).
  int dit_head_dim = 120;
  // 3D `(id, h, w)` Omni-RoPE axis dimensions (`{40, 40, 40}`).
  std::array<int, 3> axes_dims_rope = {40, 40, 40};
  // 3D Omni-RoPE base frequency theta (`10000.0f`).
  float rope_theta = 10000.0f;
  // 1D Qwen2.5-VL text encoder RoPE base frequency theta (`1000000.0f`).
  float text_rope_theta = 1000000.0f;
  // Sinusoidal timestep projection scale factor (`1000.0f`).
  float timestep_scale = 1000.0f;
  // 16-channel `AutoencoderKL` latent scaling factor (`0.3611f`).
  float vae_scaling_factor = 0.3611f;
  // 16-channel `AutoencoderKL` latent shift factor (`0.1159f`).
  float vae_shift_factor = 0.1159f;
  // Whether resolution-dependent dynamic time shifting is enabled in
  // `FlowMatchEulerDiscreteScheduler` (`m = sqrt(latent_h * latent_w) / 40`).
  bool dynamic_time_shift = true;
  // Default number of FlowMatch Euler denoising steps (`28` in
  // `OmniGen2Pipeline`).
  int steps = 28;
  // Classifier-Free Guidance scale for text conditioning (`4.0f` in
  // `OmniGen2Pipeline`; `<= 1.0f` disables CFG).
  float text_guidance_scale = 4.0f;
  // Default random seed for initial Gaussian latent noise.
  uint64_t default_seed = 42;

  // Optional per-stage backend overrides. If unset, defaults to the engine's
  // backend.
  std::optional<lm::Backend> textenc_backend;
  std::optional<lm::Backend> dit_backend;
  std::optional<lm::Backend> vae_backend;
};

// Validates that all dimensions and step counts in `config` are within
// supported bounds.
absl::Status ValidateOmniGen2ModelConfig(const OmniGen2ModelConfig& config);

// Populates `config` from an unpacked `OmniGen2` custom metadata protobuf.
void PopulateOmniGen2ConfigFromProto(
    const prototypes::litert_models::omnigen2::OmniGen2& proto,
    OmniGen2ModelConfig& config);

// Computes the `OmniGen2` `FlowMatchEulerDiscreteScheduler` ascending timestep
// schedule of length `steps + 1` in `[0.0f, 1.0f]` (`t = 0.0 -> 1.0`).
// When `dynamic_time_shift == true`, applies
// `t_shifted = t / (m - m * t + t)` where
// `m = sqrt((height / 8) * (width / 8)) / 40.0f`.
absl::StatusOr<std::vector<float>> ComputeOmniGen2Timesteps(
    int steps, int height, int width, bool dynamic_time_shift = true);

// Builds 3D `(id, h, w)` position IDs of shape `[seq_len, 3]` (flattened to
// `3 * seq_len` elements) for caption tokens, where active tokens
// `0 <= i < active_cap_len` have `(i, i, i)` and padded tokens
// `i >= active_cap_len` have `(0, 0, 0)`.
absl::StatusOr<std::vector<int32_t>> BuildOmniGen2CaptionPositionIds(
    int seq_len, int active_cap_len);

// Builds 3D `(id, h, w)` position IDs of shape `[grid_h * grid_w, 3]`
// (flattened to `3 * grid_h * grid_w` elements) for an image patch grid of
// size `grid_h x grid_w` with constant `id = id_offset`, `h in [0, grid_h)`,
// and `w in [0, grid_w)`.
absl::StatusOr<std::vector<int32_t>> BuildOmniGen2ImagePositionIds(
    int grid_h, int grid_w, int id_offset);

// Computes 3D `(id, h, w)` `OmniGen2RotaryPosEmbed` `(cos, sin)` tensors of
// shape `[1, num_tokens, 1, sum(axes_dims_rope)]` from `pos_ids` of shape
// `[num_tokens, 3]`, repeating each frequency pair twice (`repeat_interleave`
// by 2) along each axis. If `active_tokens >= 0` and
// `active_tokens < num_tokens`, trailing padded token rows
// `t >= active_tokens` are zeroed out (`cos = 0, sin = 0`) to match
// `OmniGen2RotaryPosEmbed`.
absl::StatusOr<std::pair<std::vector<float>, std::vector<float>>>
BuildOmniGen2RotaryPosEmbed(absl::Span<const int32_t> pos_ids, int num_tokens,
                            const std::array<int, 3>& axes_dims_rope,
                            float theta = 10000.0f, int active_tokens = -1);

// Builds the 4D additive attention masks `(cap_mask, joint_mask)` for
// `OmniGen2ContextRefinerShard` (`[1, 1, 1, seq_len]`) and
// `OmniGen2MainBlocksShard` (`[1, 1, 1, seq_len + img_tokens]`),
// masking padded caption key columns `active_cap_len <= k < seq_len` with
// `neg_inf`.
absl::StatusOr<std::pair<std::vector<float>, std::vector<float>>>
BuildOmniGen2AttentionMasks(int seq_len, int active_cap_len, int img_tokens,
                            float neg_inf = -1e9f);

// Packs NCHW latents `[1, channels, latent_h, latent_w]` into `OmniGen2`
// 2x2-patchified token sequence
// `[1, (latent_h / p) * (latent_w / p), p * p * channels]` matching
// `einops.rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)')`.
absl::StatusOr<std::vector<float>> PackOmniGen2Latents(
    absl::Span<const float> latents_nchw, int channels, int latent_h,
    int latent_w, int patch_size = 2);

// Unpacks `OmniGen2` 2x2-patchified latents
// `[1, (latent_h / p) * (latent_w / p), p * p * channels]` into NCHW
// `[1, channels, latent_h, latent_w]` and applies the `OmniGen2Pipeline`
// inverse VAE scaling and shift:
// `z = z_unpacked / vae_scaling_factor + vae_shift_factor`.
absl::StatusOr<std::vector<float>> UnpackAndScaleOmniGen2Latents(
    absl::Span<const float> packed_latents, int channels, int latent_h,
    int latent_w, int patch_size, float vae_scaling_factor,
    float vae_shift_factor);

}  // namespace litert::omni::text2image::omnigen2

#endif  // MODELS_OMNIGEN_OMNIGEN2_OMNIGEN2_MODEL_CONFIG_H_

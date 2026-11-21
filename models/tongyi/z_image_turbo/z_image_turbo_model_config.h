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

#ifndef MODELS_TONGYI_Z_IMAGE_TURBO_Z_IMAGE_TURBO_MODEL_CONFIG_H_
#define MODELS_TONGYI_Z_IMAGE_TURBO_Z_IMAGE_TURBO_MODEL_CONFIG_H_

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "models/tongyi/z_image_turbo/z_image_turbo.pb.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"

namespace litert::omni::text2image {
namespace z_image_turbo {

// Caption sequence padding multiple used by `ZImageTransformer2DModel`
// (`SEQ_MULTI_OF = 32` in Diffusers `transformer_z_image.py`).
inline constexpr int kSeqMultiOf = 32;

// Qwen3 pad token ID (`<|endoftext|>`).
inline constexpr int32_t kDefaultPadTokenId = 151643;

// Qwen3 chat template prefix token IDs (`<|im_start|>`).
inline constexpr std::array<int32_t, 1> kDefaultPrefixTokenIds = {151644};

// Qwen3 chat template suffix token IDs for Z-Image-Turbo with
// `add_generation_prompt=True, enable_thinking=True`
// (`<|im_end|>\n<|im_start|>assistant\n`):
// 151645 = `<|im_end|>`, 198 = `\n`, 151644 = `<|im_start|>`,
// 77091 = `assistant`, 198 = `\n`.
inline constexpr std::array<int32_t, 5> kDefaultSuffixTokenIds = {
    151645, 198, 151644, 77091, 198};

// Default architectural hyperparameters for Tongyi-MAI/Z-Image-Turbo.
inline constexpr int kDefaultSeqLen = 64;
inline constexpr int kDefaultImgSize = 256;
inline constexpr int kDefaultInChannels = 16;
inline constexpr int kDefaultPatchSize = 2;
inline constexpr int kDefaultDim = 3840;
inline constexpr int kDefaultPromptDim = 2560;
inline constexpr int kDefaultSteps = 8;
inline constexpr uint64_t kDefaultSeed = 42;
inline constexpr float kDefaultVaeScalingFactor = 0.3611f;
inline constexpr float kDefaultVaeShiftFactor = 0.1159f;
inline constexpr float kDefaultRopeTheta = 256.0f;
inline constexpr float kDefaultShift = 3.0f;
inline constexpr int kVaeSpatialDownsampleFactor = 8;

// GenericBinaryData section names for Z-Image-Turbo shards not covered by
// mainline `Text2ImageMetadata::TfLiteModelType`.
inline constexpr absl::string_view kCapEmbedderSectionName = "dit_cap_embedder";
inline constexpr absl::string_view kContextRefinerSectionName =
    "dit_context_refiner";
inline constexpr absl::string_view kXEmbedderSectionName = "dit_x_embedder";
inline constexpr absl::string_view kNoiseRefinerSectionName =
    "dit_noise_refiner";
inline constexpr absl::string_view kSingleBlock4SectionName =
    "dit_single_block_4";
inline constexpr absl::string_view kSingleBlock5SectionName =
    "dit_single_block_5";

}  // namespace z_image_turbo

// Configuration parameters and asset filenames for the Z-Image-Turbo (S3-DiT)
// text-to-image pipeline.
struct ZImageTurboModelConfig {
  // Padded token sequence length for the Qwen3 text encoder.
  int seq_len = z_image_turbo::kDefaultSeqLen;
  // Output image width and height in pixels (square).
  int img_size = z_image_turbo::kDefaultImgSize;
  // Number of latent channels before patchification (`AutoencoderKL`).
  int in_channels = z_image_turbo::kDefaultInChannels;
  // Spatial patch size used by the S3-DiT patch embedder (`2x2`).
  int patch_size = z_image_turbo::kDefaultPatchSize;
  // Hidden dimension of the S3-DiT transformer blocks.
  int dim = z_image_turbo::kDefaultDim;
  // Hidden dimension of the Qwen3 text encoder penultimate layer.
  int prompt_dim = z_image_turbo::kDefaultPromptDim;
  // Default number of FlowMatch Euler denoising steps.
  int steps = z_image_turbo::kDefaultSteps;
  // Default PRNG seed for initial latent Gaussian noise.
  uint64_t default_seed = z_image_turbo::kDefaultSeed;
  // VAE latent scaling factor (`z = z / vae_scaling_factor +
  // vae_shift_factor`).
  float vae_scaling_factor = z_image_turbo::kDefaultVaeScalingFactor;
  // VAE latent shift factor (`z = z / vae_scaling_factor + vae_shift_factor`).
  float vae_shift_factor = z_image_turbo::kDefaultVaeShiftFactor;
  // Per-axis 3D rotary position embedding dimensions `(d_t, d_h, d_w)`.
  std::array<int, 3> axes_dims = {32, 48, 48};
  // Base frequency theta for 3D axial RoPE.
  float rope_theta = z_image_turbo::kDefaultRopeTheta;
  // FlowMatch Euler static timestep shift (`scheduler_config.json`: 3.0).
  float shift = z_image_turbo::kDefaultShift;
  // Whether to use resolution-dependent dynamic shifting (`false` for
  // Z-Image-Turbo).
  bool use_dynamic_shifting = false;

  // Token IDs used by `TextEncoderStage` for Qwen3 chat-template framing.
  int32_t pad_token_id = z_image_turbo::kDefaultPadTokenId;
  std::vector<int32_t> prefix_token_ids = {
      z_image_turbo::kDefaultPrefixTokenIds.begin(),
      z_image_turbo::kDefaultPrefixTokenIds.end()};
  std::vector<int32_t> suffix_token_ids = {
      z_image_turbo::kDefaultSuffixTokenIds.begin(),
      z_image_turbo::kDefaultSuffixTokenIds.end()};

  // Fallback filenames when loading from a directory of loose files.
  std::string tokenizer_file = "tokenizer.json";
  std::string text_encoder_file = "z_text_encoder.tflite";
  std::string cap_embedder_file = "z_embc.tflite";
  std::string context_refiner_file = "z_refc.tflite";
  std::string x_embedder_file = "z_embx.tflite";
  std::string noise_refiner_file = "z_refx.tflite";
  std::vector<std::string> main_block_files = {
      "zc_main0.tflite", "zc_main1.tflite", "zc_main2.tflite",
      "zc_main3.tflite", "zc_main4.tflite", "zc_main5.tflite",
  };
  std::string final_layer_file = "zc_final.tflite";
  std::string vae_decoder_file = "zvae.tflite";

  // Returns spatial latent size (`img_size / 8`).
  int LatentSize() const;

  // Returns patch grid spatial size (`LatentSize() / patch_size`).
  int GridSize() const;

  // Returns number of patchified image tokens (`GridSize() * GridSize()`).
  int ImageTokens() const;

  // Returns total unified sequence length (`ImageTokens() + seq_len`).
  int TotalTokens() const;

  // Returns channels per patchified image token (`patch_size^2 * in_channels`).
  int PackedChannels() const;

  // Returns attention head dimension (`axes_dims[0] + axes_dims[1] +
  // axes_dims[2]`).
  int HeadDim() const;

  // Returns AdaLN conditioning embedding dimension (`min(dim, 256)`).
  int AdalnDim() const;
};

// Populates `config` with any fields present in `proto`.
void PopulateZImageTurboConfigFromProto(
    const prototypes::litert_models::z_image_turbo::ZImageTurbo& proto,
    ZImageTurboModelConfig& config);

// Builds 3D axial position IDs `(t, h, w)` for `seq_len` caption tokens.
// Each token `i` in `[0, seq_len)` gets `(1.0f + i, 0.0f, 0.0f)`.
// Resizes `ids` to `seq_len * 3`.
void BuildZImageCaptionIds(int seq_len, std::vector<float>& ids);

// Builds 3D axial position IDs `(t, h, w)` for `grid_h * grid_w` image tokens.
// Each patch `(h, w)` gets `(seq_len + 1.0f, h, w)`.
// Resizes `ids` to `grid_h * grid_w * 3`.
void BuildZImagePatchIds(int grid_h, int grid_w, int seq_len,
                         std::vector<float>& ids);

// Computes the active padded caption length by rounding `valid_tokens` up to a
// multiple of `z_image_turbo::kSeqMultiOf` (32), clamped to `[1, seq_len]`. If
// `valid_tokens <= 0`, returns `seq_len`.
int ComputeZImagePaddedCapLen(int valid_tokens, int seq_len);

// Computes the shifted FlowMatch Euler sigma schedule for Z-Image-Turbo.
// Uses static shifting (`shift * s / (1 + (shift - 1) * s)` with `shift = 3.0`)
// when `use_dynamic_shifting` is false (the Z-Image-Turbo default), or
// resolution-dependent exponential shifting when `use_dynamic_shifting` is
// true. Returns `steps + 1` sigma values starting at `1.0f` and ending at
// `0.0f` (or an empty vector if `steps <= 0`).
std::vector<float> ComputeZImageSigmas(
    int steps, int image_tokens, float shift = z_image_turbo::kDefaultShift,
    bool use_dynamic_shifting = false);

// Packs NCHW latents `[1, in_channels, latent_h, latent_w]` into patchified
// tokens `[1, (latent_h / patch_size) * (latent_w / patch_size),
// patch_size * patch_size * in_channels]`.
absl::Status PackZImageLatents(absl::Span<const float> src,
                               absl::Span<float> dst, int latent_h,
                               int latent_w, int patch_size, int in_channels);

// Unpacks patchified tokens `[1, (latent_h / patch_size) * (latent_w /
// patch_size), patch_size * patch_size * in_channels]` into NCHW latents
// `[1, in_channels, latent_h, latent_w]`.
absl::Status UnpackZImageLatents(absl::Span<const float> src,
                                 absl::Span<float> dst, int latent_h,
                                 int latent_w, int patch_size, int in_channels);

}  // namespace litert::omni::text2image

#endif  // MODELS_TONGYI_Z_IMAGE_TURBO_Z_IMAGE_TURBO_MODEL_CONFIG_H_

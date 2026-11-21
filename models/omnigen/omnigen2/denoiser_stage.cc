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

#include "models/omnigen/omnigen2/denoiser_stage.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "models/omnigen/omnigen2/io_types.h"
#include "models/omnigen/omnigen2/omnigen2_model_config.h"
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/flux2/flux2_math.h"
#include "omni/utils/tensor_utils.h"

namespace litert::omni::text2image::omnigen2 {
namespace {

struct BranchRopeAndMasks {
  std::vector<float> cap_cos;
  std::vector<float> cap_sin;
  std::vector<float> img_cos;
  std::vector<float> img_sin;
  std::vector<float> joint_cos;
  std::vector<float> joint_sin;
  std::vector<float> cap_mask;
  std::vector<float> joint_mask;
};

absl::StatusOr<size_t> GetBufferNumElements(const TensorBuffer& buffer) {
  LITERT_ASSIGN_OR_RETURN(const RankedTensorType tensor_type,
                          buffer.TensorType());
  LITERT_ASSIGN_OR_RETURN(const size_t num_elements,
                          tensor_type.Layout().NumElements());
  return num_elements;
}

absl::Status ValidateDistinctIndices(absl::Span<const size_t> indices,
                                     size_t num_buffers,
                                     absl::string_view stage_name) {
  if (num_buffers < indices.size()) {
    return absl::InvalidArgumentError(
        absl::StrFormat("%s expected at least %d input buffers, got %d.",
                        stage_name, indices.size(), num_buffers));
  }
  for (size_t i = 0; i < indices.size(); ++i) {
    if (indices[i] >= num_buffers) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "%s input index %d out of bounds for %d input buffers.", stage_name,
          indices[i], num_buffers));
    }
    for (size_t j = i + 1; j < indices.size(); ++j) {
      if (indices[i] == indices[j]) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "%s input indices must be distinct, got duplicate %d.", stage_name,
            indices[i]));
      }
    }
  }
  return absl::OkStatus();
}

int InferActiveCaptionLength(absl::Span<const float> prompt_embeds, int seq_len,
                             int text_hidden_size) {
  for (int s = seq_len - 1; s >= 0; --s) {
    const size_t row_offset =
        static_cast<size_t>(s) * static_cast<size_t>(text_hidden_size);
    for (int d = 0; d < text_hidden_size; ++d) {
      if (std::abs(prompt_embeds[row_offset + d]) > 0.0f) {
        return s + 1;
      }
    }
  }
  return seq_len;
}

int ResolveActiveCaptionLength(absl::Span<const float> prompt_embeds,
                               int requested_active_len, int seq_len,
                               int text_hidden_size) {
  if (requested_active_len > 0 && requested_active_len <= seq_len) {
    return requested_active_len;
  }
  return InferActiveCaptionLength(prompt_embeds, seq_len, text_hidden_size);
}

std::vector<float> ConcatFloatVectors(absl::Span<const float> first,
                                      absl::Span<const float> second) {
  std::vector<float> combined;
  combined.reserve(first.size() + second.size());
  combined.insert(combined.end(), first.begin(), first.end());
  combined.insert(combined.end(), second.begin(), second.end());
  return combined;
}

absl::StatusOr<BranchRopeAndMasks> BuildBranchRopeAndMasks(
    const OmniGen2ModelConfig& config, int active_cap_len, int grid_h,
    int grid_w) {
  const int seq_len = config.seq_len;
  const int img_tokens = grid_h * grid_w;
  ABSL_ASSIGN_OR_RETURN(
      std::vector<int32_t> cap_ids,
      BuildOmniGen2CaptionPositionIds(seq_len, active_cap_len));
  ABSL_ASSIGN_OR_RETURN(
      auto cap_rope,
      BuildOmniGen2RotaryPosEmbed(cap_ids, seq_len, config.axes_dims_rope,
                                  config.rope_theta, active_cap_len));
  ABSL_ASSIGN_OR_RETURN(
      std::vector<int32_t> img_ids,
      BuildOmniGen2ImagePositionIds(grid_h, grid_w, active_cap_len));
  ABSL_ASSIGN_OR_RETURN(
      auto img_rope,
      BuildOmniGen2RotaryPosEmbed(img_ids, img_tokens, config.axes_dims_rope,
                                  config.rope_theta));
  ABSL_ASSIGN_OR_RETURN(auto masks, BuildOmniGen2AttentionMasks(
                                        seq_len, active_cap_len, img_tokens));

  BranchRopeAndMasks result;
  result.joint_cos = ConcatFloatVectors(cap_rope.first, img_rope.first);
  result.joint_sin = ConcatFloatVectors(cap_rope.second, img_rope.second);
  result.cap_cos = std::move(cap_rope.first);
  result.cap_sin = std::move(cap_rope.second);
  result.img_cos = std::move(img_rope.first);
  result.img_sin = std::move(img_rope.second);
  result.cap_mask = std::move(masks.first);
  result.joint_mask = std::move(masks.second);
  return result;
}

void ApplyClassifierFreeGuidance(absl::Span<float> velocity,
                                 absl::Span<const float> uncond_velocity,
                                 float guidance_scale) {
  for (size_t i = 0; i < velocity.size(); ++i) {
    velocity[i] = uncond_velocity[i] +
                  guidance_scale * (velocity[i] - uncond_velocity[i]);
  }
}

void ApplyEulerStep(absl::Span<float> latents, absl::Span<const float> velocity,
                    float dt) {
  for (size_t i = 0; i < latents.size(); ++i) {
    latents[i] += dt * velocity[i];
  }
}

OmniGen2DenoiserOutput BuildDenoiserOutput(
    const ImageGenInputMetadata& input_metadata, int width, int height,
    int steps, uint64_t seed, std::vector<float> packed_latents) {
  OmniGen2DenoiserOutput output;
  output.metadata = input_metadata;
  output.metadata.width = width;
  output.metadata.height = height;
  output.metadata.num_inference_steps = steps;
  output.metadata.seed = seed;
  output.packed_latents = std::move(packed_latents);
  return output;
}

}  // namespace

absl::StatusOr<std::unique_ptr<OmniGen2DenoiserStage>>
OmniGen2DenoiserStage::Create(
    Stage<OmniGen2TextEncoderOutput>* absl_nonnull text_encoder,
    const OmniGen2ModelConfig& config, Runners runners,
    InputIndices input_indices) {
  ABSL_RETURN_IF_ERROR(ValidateOmniGen2ModelConfig(config));
  if (runners.cap_embedder == nullptr || runners.context_refiner == nullptr ||
      runners.x_embedder == nullptr || runners.noise_refiner == nullptr ||
      runners.final_stage == nullptr) {
    return absl::InvalidArgumentError(
        "OmniGen2DenoiserStage cap_embedder, context_refiner, x_embedder, "
        "noise_refiner, and final_stage runners must not be null.");
  }
  for (size_t i = 0; i < runners.main_blocks.size(); ++i) {
    if (runners.main_blocks[i] == nullptr) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "OmniGen2DenoiserStage main_blocks[%d] runner must not be null.", i));
    }
  }

  const size_t latent_dim = static_cast<size_t>(config.img_size / 8);
  const size_t grid_dim = latent_dim / static_cast<size_t>(config.patch_size);
  const size_t img_tokens = grid_dim * grid_dim;
  const size_t seq_len = static_cast<size_t>(config.seq_len);
  const size_t joint_tokens = seq_len + img_tokens;
  const size_t text_hidden_size = static_cast<size_t>(config.text_hidden_size);
  const size_t dit_hidden_size = static_cast<size_t>(config.dit_hidden_size);
  const size_t head_dim = static_cast<size_t>(config.dit_head_dim);
  const size_t packed_ch = static_cast<size_t>(config.packed_ch);

  const size_t prompt_elements = seq_len * text_hidden_size;
  const size_t cap_hidden_elements = seq_len * dit_hidden_size;
  const size_t img_hidden_elements = img_tokens * dit_hidden_size;
  const size_t joint_hidden_elements = joint_tokens * dit_hidden_size;
  const size_t latent_elements = img_tokens * packed_ch;
  const size_t cap_rope_elements = seq_len * head_dim;
  const size_t img_rope_elements = img_tokens * head_dim;
  const size_t joint_rope_elements = joint_tokens * head_dim;

  Buffers buffers;

  // 1. Validate `cap_embedder`.
  ABSL_ASSIGN_OR_RETURN(buffers.cap_emb_in,
                        runners.cap_embedder->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(buffers.cap_emb_out,
                        runners.cap_embedder->CreateOutputBuffers(""));
  const std::array<size_t, 1> cap_emb_indices = {
      input_indices.cap_embedder.cap};
  ABSL_RETURN_IF_ERROR(
      ValidateDistinctIndices(cap_emb_indices, buffers.cap_emb_in.size(),
                              "OmniGen2DenoiserStage cap_embedder"));
  if (buffers.cap_emb_out.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2DenoiserStage cap_embedder expected 1 output buffer, got %d.",
        buffers.cap_emb_out.size()));
  }
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.cap_emb_in[input_indices.cap_embedder.cap], prompt_elements,
      "OmniGen2DenoiserStage cap_embedder cap"));
  ABSL_RETURN_IF_ERROR(
      ValidateFloatBuffer(buffers.cap_emb_out[0], cap_hidden_elements,
                          "OmniGen2DenoiserStage cap_embedder output"));

  // 2. Validate `context_refiner`.
  ABSL_ASSIGN_OR_RETURN(buffers.ctx_ref_in,
                        runners.context_refiner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(buffers.ctx_ref_out,
                        runners.context_refiner->CreateOutputBuffers(""));
  const std::array<size_t, 4> ctx_ref_indices = {
      input_indices.context_refiner.cap,
      input_indices.context_refiner.cos,
      input_indices.context_refiner.sin,
      input_indices.context_refiner.mask,
  };
  ABSL_RETURN_IF_ERROR(
      ValidateDistinctIndices(ctx_ref_indices, buffers.ctx_ref_in.size(),
                              "OmniGen2DenoiserStage context_refiner"));
  if (buffers.ctx_ref_out.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2DenoiserStage context_refiner expected 1 output buffer, got "
        "%d.",
        buffers.ctx_ref_out.size()));
  }
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.ctx_ref_in[input_indices.context_refiner.cap],
      cap_hidden_elements, "OmniGen2DenoiserStage context_refiner cap"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.ctx_ref_in[input_indices.context_refiner.cos], cap_rope_elements,
      "OmniGen2DenoiserStage context_refiner cos"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.ctx_ref_in[input_indices.context_refiner.sin], cap_rope_elements,
      "OmniGen2DenoiserStage context_refiner sin"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.ctx_ref_in[input_indices.context_refiner.mask], seq_len,
      "OmniGen2DenoiserStage context_refiner mask"));
  ABSL_RETURN_IF_ERROR(
      ValidateFloatBuffer(buffers.ctx_ref_out[0], cap_hidden_elements,
                          "OmniGen2DenoiserStage context_refiner output"));

  // 3. Validate `x_embedder`.
  ABSL_ASSIGN_OR_RETURN(buffers.x_emb_in,
                        runners.x_embedder->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(buffers.x_emb_out,
                        runners.x_embedder->CreateOutputBuffers(""));
  const std::array<size_t, 2> x_emb_indices = {
      input_indices.x_embedder.x,
      input_indices.x_embedder.t,
  };
  ABSL_RETURN_IF_ERROR(
      ValidateDistinctIndices(x_emb_indices, buffers.x_emb_in.size(),
                              "OmniGen2DenoiserStage x_embedder"));
  if (buffers.x_emb_out.size() != 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2DenoiserStage x_embedder expected 2 output buffers "
        "(x_emb, temb), got %d.",
        buffers.x_emb_out.size()));
  }
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.x_emb_in[input_indices.x_embedder.x], latent_elements,
      "OmniGen2DenoiserStage x_embedder x"));
  ABSL_RETURN_IF_ERROR(
      ValidateFloatBuffer(buffers.x_emb_in[input_indices.x_embedder.t], 1,
                          "OmniGen2DenoiserStage x_embedder t"));
  ABSL_RETURN_IF_ERROR(
      ValidateFloatBuffer(buffers.x_emb_out[0], img_hidden_elements,
                          "OmniGen2DenoiserStage x_embedder x_emb"));
  ABSL_ASSIGN_OR_RETURN(const size_t temb_elements,
                        GetBufferNumElements(buffers.x_emb_out[1]));
  if (temb_elements == 0) {
    return absl::InvalidArgumentError(
        "OmniGen2DenoiserStage x_embedder temb output must not be empty.");
  }
  ABSL_RETURN_IF_ERROR(
      ValidateFloatBuffer(buffers.x_emb_out[1], temb_elements,
                          "OmniGen2DenoiserStage x_embedder temb"));

  // 4. Validate `noise_refiner`.
  ABSL_ASSIGN_OR_RETURN(buffers.noise_ref_in,
                        runners.noise_refiner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(buffers.noise_ref_out,
                        runners.noise_refiner->CreateOutputBuffers(""));
  const std::array<size_t, 4> noise_ref_indices = {
      input_indices.noise_refiner.x,
      input_indices.noise_refiner.cos,
      input_indices.noise_refiner.sin,
      input_indices.noise_refiner.temb,
  };
  ABSL_RETURN_IF_ERROR(
      ValidateDistinctIndices(noise_ref_indices, buffers.noise_ref_in.size(),
                              "OmniGen2DenoiserStage noise_refiner"));
  if (buffers.noise_ref_out.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2DenoiserStage noise_refiner expected 1 output buffer, got "
        "%d.",
        buffers.noise_ref_out.size()));
  }
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.noise_ref_in[input_indices.noise_refiner.x], img_hidden_elements,
      "OmniGen2DenoiserStage noise_refiner x"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.noise_ref_in[input_indices.noise_refiner.cos], img_rope_elements,
      "OmniGen2DenoiserStage noise_refiner cos"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.noise_ref_in[input_indices.noise_refiner.sin], img_rope_elements,
      "OmniGen2DenoiserStage noise_refiner sin"));
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.noise_ref_in[input_indices.noise_refiner.temb], temb_elements,
      "OmniGen2DenoiserStage noise_refiner temb"));
  ABSL_RETURN_IF_ERROR(
      ValidateFloatBuffer(buffers.noise_ref_out[0], img_hidden_elements,
                          "OmniGen2DenoiserStage noise_refiner output"));

  // 5. Validate `main_blocks[0..kNumMainBlockShards-1]`.
  for (size_t i = 0; i < kNumMainBlockShards; ++i) {
    const std::string label =
        absl::StrCat("OmniGen2DenoiserStage main_block_", i);
    ABSL_ASSIGN_OR_RETURN(buffers.main_in[i],
                          runners.main_blocks[i]->CreateInputBuffers(""));
    ABSL_ASSIGN_OR_RETURN(buffers.main_out[i],
                          runners.main_blocks[i]->CreateOutputBuffers(""));
    const auto& idx = input_indices.main_blocks[i];
    const std::array<size_t, 5> main_indices = {idx.joint, idx.cos, idx.sin,
                                                idx.mask, idx.temb};
    ABSL_RETURN_IF_ERROR(ValidateDistinctIndices(
        main_indices, buffers.main_in[i].size(), label));
    if (buffers.main_out[i].size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrFormat("%s expected 1 output buffer, got %d.", label,
                          buffers.main_out[i].size()));
    }
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.main_in[i][idx.joint],
                                             joint_hidden_elements,
                                             absl::StrCat(label, " joint")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.main_in[i][idx.cos],
                                             joint_rope_elements,
                                             absl::StrCat(label, " cos")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.main_in[i][idx.sin],
                                             joint_rope_elements,
                                             absl::StrCat(label, " sin")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.main_in[i][idx.mask],
                                             joint_tokens,
                                             absl::StrCat(label, " mask")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.main_in[i][idx.temb],
                                             temb_elements,
                                             absl::StrCat(label, " temb")));
    ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(buffers.main_out[i][0],
                                             joint_hidden_elements,
                                             absl::StrCat(label, " output")));
  }

  // 6. Validate `final_stage`.
  ABSL_ASSIGN_OR_RETURN(buffers.final_in,
                        runners.final_stage->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(buffers.final_out,
                        runners.final_stage->CreateOutputBuffers(""));
  const std::array<size_t, 2> final_indices = {
      input_indices.final_stage.joint,
      input_indices.final_stage.temb,
  };
  ABSL_RETURN_IF_ERROR(ValidateDistinctIndices(
      final_indices, buffers.final_in.size(), "OmniGen2DenoiserStage final"));
  if (buffers.final_out.size() != 1) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "OmniGen2DenoiserStage final expected 1 output buffer, got %d.",
        buffers.final_out.size()));
  }
  ABSL_RETURN_IF_ERROR(ValidateFloatBuffer(
      buffers.final_in[input_indices.final_stage.joint], joint_hidden_elements,
      "OmniGen2DenoiserStage final joint"));
  ABSL_RETURN_IF_ERROR(
      ValidateFloatBuffer(buffers.final_in[input_indices.final_stage.temb],
                          temb_elements, "OmniGen2DenoiserStage final temb"));
  ABSL_RETURN_IF_ERROR(
      ValidateFloatBuffer(buffers.final_out[0], latent_elements,
                          "OmniGen2DenoiserStage final output"));

  return absl::WrapUnique(new OmniGen2DenoiserStage(
      text_encoder, config, std::move(runners), input_indices,
      std::move(buffers), static_cast<int>(temb_elements)));
}

absl::StatusOr<std::unique_ptr<OmniGen2DenoiserStage>>
OmniGen2DenoiserStage::Create(
    Stage<OmniGen2TextEncoderOutput>* absl_nonnull text_encoder,
    const OmniGen2ModelConfig& config, Runners runners) {
  return Create(text_encoder, config, std::move(runners), InputIndices{});
}

OmniGen2DenoiserStage::OmniGen2DenoiserStage(
    Stage<OmniGen2TextEncoderOutput>* absl_nonnull text_encoder,
    OmniGen2ModelConfig config, Runners runners, InputIndices input_indices,
    Buffers buffers, int temb_dim)
    : text_encoder_(*text_encoder),
      config_(std::move(config)),
      runners_(std::move(runners)),
      input_indices_(input_indices),
      buffers_(std::move(buffers)),
      temb_dim_(temb_dim) {}

bool OmniGen2DenoiserStage::NeedScheduleInternal() const {
  return text_encoder_.HasOutput();
}

absl::Status OmniGen2DenoiserStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  absl::StatusOr<OmniGen2TextEncoderOutput> cond = text_encoder_.GetOutput();
  if (absl::IsNotFound(cond.status())) {
    return absl::OkStatus();
  }
  if (!cond.ok()) {
    return cond.status();
  }

  const int width = config_.img_size;
  const int height = config_.img_size;
  const int latent_h = height / 8;
  const int latent_w = width / 8;
  const int grid_h = latent_h / config_.patch_size;
  const int grid_w = latent_w / config_.patch_size;
  const int steps = cond->metadata.num_inference_steps > 0
                        ? cond->metadata.num_inference_steps
                        : config_.steps;
  if (steps != config_.steps) {
    OmniGen2ModelConfig step_config = config_;
    step_config.steps = steps;
    ABSL_RETURN_IF_ERROR(ValidateOmniGen2ModelConfig(step_config));
  }

  ABSL_ASSIGN_OR_RETURN(
      PreparedBranch cond_branch,
      PrepareBranchConditioning(cond->prompt_embeds, cond->active_seq_len,
                                grid_h, grid_w, "prompt_embeds"));

  const bool use_cfg = config_.text_guidance_scale > 1.0f &&
                       !cond->negative_prompt_embeds.empty();
  std::optional<PreparedBranch> uncond_branch;
  if (use_cfg) {
    ABSL_ASSIGN_OR_RETURN(
        uncond_branch,
        PrepareBranchConditioning(cond->negative_prompt_embeds,
                                  cond->negative_active_seq_len, grid_h, grid_w,
                                  "negative_prompt_embeds"));
  }

  ABSL_RETURN_IF_ERROR(WriteBranchRopeAndMask(cond_branch));

  const size_t latent_elements =
      static_cast<size_t>(grid_h * grid_w) * config_.packed_ch;
  const uint64_t seed =
      cond->metadata.seed > 0 ? cond->metadata.seed : config_.default_seed;
  std::vector<float> nchw_noise = SampleGaussianLatents(latent_elements, seed);
  ABSL_ASSIGN_OR_RETURN(
      std::vector<float> latents,
      PackOmniGen2Latents(nchw_noise, config_.in_channels, latent_h, latent_w,
                          config_.patch_size));
  ABSL_ASSIGN_OR_RETURN(std::vector<float> timesteps,
                        ComputeOmniGen2Timesteps(steps, height, width,
                                                 config_.dynamic_time_shift));

  ABSL_RETURN_IF_ERROR(RunFlowMatchEulerLoop(
      steps, timesteps, cond_branch,
      uncond_branch.has_value() ? &*uncond_branch : nullptr, latents));

  PushOutput(BuildDenoiserOutput(cond->metadata, width, height, steps, seed,
                                 std::move(latents)));
  return absl::OkStatus();
}

absl::StatusOr<OmniGen2DenoiserStage::PreparedBranch>
OmniGen2DenoiserStage::PrepareBranchConditioning(
    absl::Span<const float> prompt_embeds, int requested_active_len, int grid_h,
    int grid_w, absl::string_view field_name) {
  const size_t prompt_elements =
      static_cast<size_t>(config_.seq_len) * config_.text_hidden_size;
  if (prompt_embeds.size() != prompt_elements) {
    return absl::InvalidArgumentError(
        absl::StrFormat("OmniGen2DenoiserStage expected %s size %d, got %d.",
                        field_name, prompt_elements, prompt_embeds.size()));
  }

  const int active_cap_len =
      ResolveActiveCaptionLength(prompt_embeds, requested_active_len,
                                 config_.seq_len, config_.text_hidden_size);
  ABSL_ASSIGN_OR_RETURN(
      BranchRopeAndMasks rope_and_masks,
      BuildBranchRopeAndMasks(config_, active_cap_len, grid_h, grid_w));

  const size_t cap_hidden_elements =
      static_cast<size_t>(config_.seq_len) * config_.dit_hidden_size;
  std::vector<float> cap_emb(cap_hidden_elements, 0.0f);
  std::vector<float> cap_refined(cap_hidden_elements, 0.0f);

  LITERT_RETURN_IF_ERROR(
      buffers_.cap_emb_in[input_indices_.cap_embedder.cap].Write<float>(
          prompt_embeds));
  ABSL_RETURN_IF_ERROR(runners_.cap_embedder->Run("", buffers_.cap_emb_in,
                                                  buffers_.cap_emb_out));
  LITERT_RETURN_IF_ERROR(
      buffers_.cap_emb_out[0].Read<float>(absl::MakeSpan(cap_emb)));

  LITERT_RETURN_IF_ERROR(
      buffers_.ctx_ref_in[input_indices_.context_refiner.cap].Write<float>(
          absl::MakeConstSpan(cap_emb)));
  LITERT_RETURN_IF_ERROR(
      buffers_.ctx_ref_in[input_indices_.context_refiner.cos].Write<float>(
          absl::MakeConstSpan(rope_and_masks.cap_cos)));
  LITERT_RETURN_IF_ERROR(
      buffers_.ctx_ref_in[input_indices_.context_refiner.sin].Write<float>(
          absl::MakeConstSpan(rope_and_masks.cap_sin)));
  LITERT_RETURN_IF_ERROR(
      buffers_.ctx_ref_in[input_indices_.context_refiner.mask].Write<float>(
          absl::MakeConstSpan(rope_and_masks.cap_mask)));
  ABSL_RETURN_IF_ERROR(runners_.context_refiner->Run("", buffers_.ctx_ref_in,
                                                     buffers_.ctx_ref_out));
  LITERT_RETURN_IF_ERROR(
      buffers_.ctx_ref_out[0].Read<float>(absl::MakeSpan(cap_refined)));

  PreparedBranch branch;
  branch.cap_refined = std::move(cap_refined);
  branch.img_cos = std::move(rope_and_masks.img_cos);
  branch.img_sin = std::move(rope_and_masks.img_sin);
  branch.joint_cos = std::move(rope_and_masks.joint_cos);
  branch.joint_sin = std::move(rope_and_masks.joint_sin);
  branch.joint_mask = std::move(rope_and_masks.joint_mask);
  return branch;
}

absl::Status OmniGen2DenoiserStage::WriteBranchRopeAndMask(
    const PreparedBranch& branch) {
  LITERT_RETURN_IF_ERROR(
      buffers_.noise_ref_in[input_indices_.noise_refiner.cos].Write<float>(
          absl::MakeConstSpan(branch.img_cos)));
  LITERT_RETURN_IF_ERROR(
      buffers_.noise_ref_in[input_indices_.noise_refiner.sin].Write<float>(
          absl::MakeConstSpan(branch.img_sin)));
  for (size_t i = 0; i < runners_.main_blocks.size(); ++i) {
    const auto& idx = input_indices_.main_blocks[i];
    LITERT_RETURN_IF_ERROR(buffers_.main_in[i][idx.cos].Write<float>(
        absl::MakeConstSpan(branch.joint_cos)));
    LITERT_RETURN_IF_ERROR(buffers_.main_in[i][idx.sin].Write<float>(
        absl::MakeConstSpan(branch.joint_sin)));
    LITERT_RETURN_IF_ERROR(buffers_.main_in[i][idx.mask].Write<float>(
        absl::MakeConstSpan(branch.joint_mask)));
  }
  return absl::OkStatus();
}

absl::Status OmniGen2DenoiserStage::RunXEmbedder(
    absl::Span<const float> latents, float timestep, absl::Span<float> x_emb,
    absl::Span<float> temb) {
  LITERT_RETURN_IF_ERROR(
      buffers_.x_emb_in[input_indices_.x_embedder.x].Write<float>(latents));
  LITERT_RETURN_IF_ERROR(
      buffers_.x_emb_in[input_indices_.x_embedder.t].Write<float>(
          absl::MakeConstSpan(&timestep, 1)));
  ABSL_RETURN_IF_ERROR(
      runners_.x_embedder->Run("", buffers_.x_emb_in, buffers_.x_emb_out));
  LITERT_RETURN_IF_ERROR(buffers_.x_emb_out[0].Read<float>(x_emb));
  LITERT_RETURN_IF_ERROR(buffers_.x_emb_out[1].Read<float>(temb));
  return absl::OkStatus();
}

absl::Status OmniGen2DenoiserStage::RunDenoiserBranch(
    const PreparedBranch& branch, bool write_branch_rope,
    DenoisingWorkspace& workspace, absl::Span<float> out_velocity) {
  if (write_branch_rope) {
    ABSL_RETURN_IF_ERROR(WriteBranchRopeAndMask(branch));
  }
  LITERT_RETURN_IF_ERROR(
      buffers_.noise_ref_in[input_indices_.noise_refiner.x].Write<float>(
          absl::MakeConstSpan(workspace.x_emb)));
  LITERT_RETURN_IF_ERROR(
      buffers_.noise_ref_in[input_indices_.noise_refiner.temb].Write<float>(
          absl::MakeConstSpan(workspace.temb)));
  ABSL_RETURN_IF_ERROR(runners_.noise_refiner->Run("", buffers_.noise_ref_in,
                                                   buffers_.noise_ref_out));
  LITERT_RETURN_IF_ERROR(buffers_.noise_ref_out[0].Read<float>(
      absl::MakeSpan(workspace.x_refined)));

  absl::c_copy(branch.cap_refined, workspace.joint.begin());
  absl::c_copy(workspace.x_refined,
               workspace.joint.begin() + branch.cap_refined.size());

  for (size_t i = 0; i < runners_.main_blocks.size(); ++i) {
    const auto& idx = input_indices_.main_blocks[i];
    LITERT_RETURN_IF_ERROR(buffers_.main_in[i][idx.joint].Write<float>(
        absl::MakeConstSpan(workspace.joint)));
    LITERT_RETURN_IF_ERROR(buffers_.main_in[i][idx.temb].Write<float>(
        absl::MakeConstSpan(workspace.temb)));
    ABSL_RETURN_IF_ERROR(runners_.main_blocks[i]->Run("", buffers_.main_in[i],
                                                      buffers_.main_out[i]));
    LITERT_RETURN_IF_ERROR(
        buffers_.main_out[i][0].Read<float>(absl::MakeSpan(workspace.joint)));
  }

  LITERT_RETURN_IF_ERROR(
      buffers_.final_in[input_indices_.final_stage.joint].Write<float>(
          absl::MakeConstSpan(workspace.joint)));
  LITERT_RETURN_IF_ERROR(
      buffers_.final_in[input_indices_.final_stage.temb].Write<float>(
          absl::MakeConstSpan(workspace.temb)));
  ABSL_RETURN_IF_ERROR(
      runners_.final_stage->Run("", buffers_.final_in, buffers_.final_out));
  LITERT_RETURN_IF_ERROR(buffers_.final_out[0].Read<float>(out_velocity));
  return absl::OkStatus();
}

absl::Status OmniGen2DenoiserStage::RunFlowMatchEulerLoop(
    int steps, absl::Span<const float> timesteps,
    const PreparedBranch& cond_branch,
    const PreparedBranch* absl_nullable uncond_branch,
    std::vector<float>& latents) {
  const size_t img_tokens =
      latents.size() / static_cast<size_t>(config_.packed_ch);
  const size_t img_hidden_elements =
      img_tokens * static_cast<size_t>(config_.dit_hidden_size);
  const size_t cap_hidden_elements = cond_branch.cap_refined.size();
  const bool use_cfg = uncond_branch != nullptr;

  DenoisingWorkspace workspace;
  workspace.x_emb.assign(img_hidden_elements, 0.0f);
  workspace.x_refined.assign(img_hidden_elements, 0.0f);
  workspace.temb.assign(static_cast<size_t>(temb_dim_), 0.0f);
  workspace.joint.assign(cap_hidden_elements + img_hidden_elements, 0.0f);
  workspace.velocity.assign(latents.size(), 0.0f);
  if (use_cfg) {
    workspace.uncond_velocity.assign(latents.size(), 0.0f);
  }

  for (int step = 0; step < steps; ++step) {
    const float t_cur = timesteps[step];
    const float dt = timesteps[step + 1] - t_cur;

    ABSL_RETURN_IF_ERROR(RunXEmbedder(latents, t_cur,
                                      absl::MakeSpan(workspace.x_emb),
                                      absl::MakeSpan(workspace.temb)));
    ABSL_RETURN_IF_ERROR(RunDenoiserBranch(
        cond_branch, /*write_branch_rope=*/use_cfg && step > 0, workspace,
        absl::MakeSpan(workspace.velocity)));
    if (use_cfg) {
      ABSL_RETURN_IF_ERROR(RunDenoiserBranch(
          *uncond_branch, /*write_branch_rope=*/true, workspace,
          absl::MakeSpan(workspace.uncond_velocity)));
      ApplyClassifierFreeGuidance(absl::MakeSpan(workspace.velocity),
                                  workspace.uncond_velocity,
                                  config_.text_guidance_scale);
    }
    ApplyEulerStep(absl::MakeSpan(latents), workspace.velocity, dt);
  }
  return absl::OkStatus();
}

}  // namespace litert::omni::text2image::omnigen2

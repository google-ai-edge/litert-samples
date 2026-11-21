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

#include "models/tongyi/z_image_turbo/denoiser_stage.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "models/tongyi/z_image_turbo/io_types.h"
#include "models/tongyi/z_image_turbo/z_image_turbo_model_config.h"
#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"
#include "absl/cleanup/cleanup.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/io_types.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"
#include "omni/text2image/flux2/flux2_math.h"
#include "omni/text2image/text_encoder_stage.h"

namespace litert::omni::text2image {
namespace {

absl::Status CreateAndValidateBuffers(LiteRtRunner* absl_nullable runner,
                                      absl::string_view name,
                                      size_t expected_inputs,
                                      size_t expected_outputs,
                                      std::vector<TensorBuffer>& in_buffers,
                                      std::vector<TensorBuffer>& out_buffers) {
  if (runner == nullptr) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ZImageDenoiserStage runner '", name, "' must not be null."));
  }
  ABSL_ASSIGN_OR_RETURN(in_buffers, runner->CreateInputBuffers(""));
  ABSL_ASSIGN_OR_RETURN(out_buffers, runner->CreateOutputBuffers(""));
  if (in_buffers.size() != expected_inputs) {
    return absl::InvalidArgumentError(
        absl::StrCat("ZImageDenoiserStage runner '", name, "' expected ",
                     expected_inputs, " inputs, got ", in_buffers.size()));
  }
  if (out_buffers.size() != expected_outputs) {
    return absl::InvalidArgumentError(
        absl::StrCat("ZImageDenoiserStage runner '", name, "' expected ",
                     expected_outputs, " outputs, got ", out_buffers.size()));
  }
  return absl::OkStatus();
}

absl::Status WriteRotaryBuffers(
    const ZImageTurboModelConfig& config, const int active_cap_len,
    std::vector<TensorBuffer>& refc_in, std::vector<TensorBuffer>& refx_in,
    std::vector<std::vector<TensorBuffer>>& main_in) {
  const size_t head_dim = static_cast<size_t>(config.HeadDim());
  std::vector<float> cap_ids;
  BuildZImageCaptionIds(config.seq_len, cap_ids);
  ABSL_ASSIGN_OR_RETURN(
      auto cap_rope,
      BuildFlux2RotaryPosEmbed(cap_ids, config.seq_len, config.axes_dims,
                               config.rope_theta));
  if (active_cap_len < config.seq_len) {
    const size_t start_elem = static_cast<size_t>(active_cap_len) * head_dim;
    auto cap_cos_tail = absl::MakeSpan(cap_rope.first).subspan(start_elem);
    auto cap_sin_tail = absl::MakeSpan(cap_rope.second).subspan(start_elem);
    absl::c_fill(cap_cos_tail, 0.0f);
    absl::c_fill(cap_sin_tail, 0.0f);
  }

  const int grid = config.GridSize();
  std::vector<float> img_ids;
  BuildZImagePatchIds(grid, grid, active_cap_len, img_ids);
  ABSL_ASSIGN_OR_RETURN(
      auto img_rope,
      BuildFlux2RotaryPosEmbed(img_ids, config.ImageTokens(), config.axes_dims,
                               config.rope_theta));

  std::vector<float> unified_cos;
  unified_cos.reserve(img_rope.first.size() + cap_rope.first.size());
  unified_cos.insert(unified_cos.end(), img_rope.first.begin(),
                     img_rope.first.end());
  unified_cos.insert(unified_cos.end(), cap_rope.first.begin(),
                     cap_rope.first.end());

  std::vector<float> unified_sin;
  unified_sin.reserve(img_rope.second.size() + cap_rope.second.size());
  unified_sin.insert(unified_sin.end(), img_rope.second.begin(),
                     img_rope.second.end());
  unified_sin.insert(unified_sin.end(), cap_rope.second.begin(),
                     cap_rope.second.end());

  LITERT_RETURN_IF_ERROR(
      refc_in[1].Write<float>(absl::MakeConstSpan(cap_rope.first)));
  LITERT_RETURN_IF_ERROR(
      refc_in[2].Write<float>(absl::MakeConstSpan(cap_rope.second)));
  LITERT_RETURN_IF_ERROR(
      refx_in[1].Write<float>(absl::MakeConstSpan(img_rope.first)));
  LITERT_RETURN_IF_ERROR(
      refx_in[2].Write<float>(absl::MakeConstSpan(img_rope.second)));
  for (std::vector<TensorBuffer>& block_in : main_in) {
    LITERT_RETURN_IF_ERROR(
        block_in[1].Write<float>(absl::MakeConstSpan(unified_cos)));
    LITERT_RETURN_IF_ERROR(
        block_in[2].Write<float>(absl::MakeConstSpan(unified_sin)));
  }
  return absl::OkStatus();
}

// Counts active caption tokens by locating the last non-zero row in the
// `[seq_len, prompt_dim]` text encoder output.
int CountActiveCaptionTokens(absl::Span<const float> prompt_embeds,
                             const size_t seq_len, const size_t prompt_dim) {
  int valid_cap_tokens = 0;
  for (size_t i = 0; i < seq_len; ++i) {
    const absl::Span<const float> row =
        prompt_embeds.subspan(i * prompt_dim, prompt_dim);
    if (absl::c_any_of(row, [](const float v) { return v != 0.0f; })) {
      valid_cap_tokens = static_cast<int>(i + 1);
    }
  }
  return valid_cap_tokens;
}

void ApplyEulerUpdate(absl::Span<float> latents,
                      absl::Span<const float> velocity, const float dt) {
  for (size_t j = 0; j < latents.size(); ++j) {
    latents[j] += dt * velocity[j];
  }
}

ZImageDenoiserOutput BuildDenoiserOutput(
    const ImageGenInputMetadata& input_metadata,
    const ZImageTurboModelConfig& config, const int steps, const uint64_t seed,
    std::vector<float> latents) {
  ZImageDenoiserOutput output;
  output.metadata = input_metadata;
  output.metadata.width = config.img_size;
  output.metadata.height = config.img_size;
  output.metadata.num_inference_steps = steps;
  output.metadata.seed = seed;
  output.packed_latents = std::move(latents);
  return output;
}

}  // namespace

absl::StatusOr<std::unique_ptr<ZImageDenoiserStage>>
ZImageDenoiserStage::Create(Stage<TextEncoderOutput>* absl_nonnull text_encoder,
                            const ZImageTurboModelConfig& config,
                            Runners runners) {
  if (text_encoder == nullptr) {
    return absl::InvalidArgumentError(
        "ZImageDenoiserStage text_encoder must not be null.");
  }
  if (config.seq_len <= 0 || config.img_size <= 0 || config.patch_size <= 0 ||
      config.in_channels <= 0 || config.dim <= 0 || config.prompt_dim <= 0 ||
      config.GridSize() <= 0 || config.HeadDim() <= 0) {
    return absl::InvalidArgumentError(
        "ZImageDenoiserStage requires positive model dimensions.");
  }
  if (config.img_size % z_image_turbo::kVaeSpatialDownsampleFactor != 0 ||
      config.LatentSize() % config.patch_size != 0) {
    return absl::InvalidArgumentError(
        "ZImageDenoiserStage requires img_size divisible by 8 * patch_size.");
  }
  if (runners.main_blocks.empty()) {
    return absl::InvalidArgumentError(
        "ZImageDenoiserStage requires at least one main_blocks runner.");
  }

  Buffers buffers;
  ABSL_RETURN_IF_ERROR(
      CreateAndValidateBuffers(runners.cap_embedder.get(), "cap_embedder", 1, 1,
                               buffers.embc_in, buffers.embc_out));
  ABSL_RETURN_IF_ERROR(
      CreateAndValidateBuffers(runners.context_refiner.get(), "context_refiner",
                               3, 1, buffers.refc_in, buffers.refc_out));
  ABSL_RETURN_IF_ERROR(
      CreateAndValidateBuffers(runners.x_embedder.get(), "x_embedder", 2, 2,
                               buffers.embx_in, buffers.embx_out));
  ABSL_RETURN_IF_ERROR(
      CreateAndValidateBuffers(runners.noise_refiner.get(), "noise_refiner", 4,
                               1, buffers.refx_in, buffers.refx_out));

  buffers.main_in.resize(runners.main_blocks.size());
  buffers.main_out.resize(runners.main_blocks.size());
  for (size_t i = 0; i < runners.main_blocks.size(); ++i) {
    ABSL_RETURN_IF_ERROR(
        CreateAndValidateBuffers(runners.main_blocks[i].get(), "main_block", 4,
                                 1, buffers.main_in[i], buffers.main_out[i]));
  }
  ABSL_RETURN_IF_ERROR(
      CreateAndValidateBuffers(runners.final_layer.get(), "final_layer", 2, 1,
                               buffers.final_in, buffers.final_out));

  ABSL_RETURN_IF_ERROR(WriteRotaryBuffers(config, config.seq_len,
                                          buffers.refc_in, buffers.refx_in,
                                          buffers.main_in));

  return absl::WrapUnique(
      new ZImageDenoiserStage(text_encoder, config, std::move(runners),
                              std::move(buffers), config.seq_len));
}

ZImageDenoiserStage::~ZImageDenoiserStage() = default;

bool ZImageDenoiserStage::NeedScheduleInternal() const {
  return text_encoder_.HasOutput();
}

absl::Status ZImageDenoiserStage::ScheduleInternal() {
  absl::Cleanup cleanup = [this] { SetState(State::kIdle); };

  absl::StatusOr<TextEncoderOutput> cond = text_encoder_.GetOutput();
  if (absl::IsNotFound(cond.status())) {
    return absl::OkStatus();
  }
  if (!cond.ok()) {
    return cond.status();
  }

  const size_t seq_len = static_cast<size_t>(config_.seq_len);
  const size_t prompt_dim = static_cast<size_t>(config_.prompt_dim);
  if (cond->prompt_embeds.size() != seq_len * prompt_dim) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ZImageDenoiserStage expected ", seq_len * prompt_dim,
        " prompt embedding elements, got ", cond->prompt_embeds.size(), "."));
  }

  // Round active caption tokens up to `z_image_turbo::kSeqMultiOf` (32) and
  // refresh 3D RoPE tables so padded blocks are masked (`0.0f`) and image
  // temporal coordinates start at `active_cap_len + 1`.
  const int valid_cap_tokens =
      CountActiveCaptionTokens(cond->prompt_embeds, seq_len, prompt_dim);
  ABSL_RETURN_IF_ERROR(UpdateRotaryBuffers(
      ComputeZImagePaddedCapLen(valid_cap_tokens, config_.seq_len)));

  ABSL_ASSIGN_OR_RETURN(std::vector<float> cap_refined,
                        EncodeAndRefineCaption(cond->prompt_embeds));

  const uint64_t seed =
      cond->metadata.seed > 0 ? cond->metadata.seed : config_.default_seed;
  const int steps = cond->metadata.num_inference_steps > 0
                        ? cond->metadata.num_inference_steps
                        : config_.steps;
  if (steps <= 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ZImageDenoiserStage requires positive steps, got ", steps, "."));
  }

  ABSL_ASSIGN_OR_RETURN(std::vector<float> latents,
                        RunEulerDenoisingLoop(cap_refined, steps, seed));

  PushOutput(BuildDenoiserOutput(cond->metadata, config_, steps, seed,
                                 std::move(latents)));
  return absl::OkStatus();
}

ZImageDenoiserStage::ZImageDenoiserStage(
    Stage<TextEncoderOutput>* absl_nonnull text_encoder,
    ZImageTurboModelConfig config, Runners runners, Buffers buffers,
    const int last_active_cap_len)
    : text_encoder_(*text_encoder),
      config_(std::move(config)),
      runners_(std::move(runners)),
      buffers_(std::move(buffers)),
      last_active_cap_len_(last_active_cap_len) {}

absl::Status ZImageDenoiserStage::UpdateRotaryBuffers(
    const int active_cap_len) {
  if (active_cap_len == last_active_cap_len_) {
    return absl::OkStatus();
  }
  ABSL_RETURN_IF_ERROR(WriteRotaryBuffers(config_, active_cap_len,
                                          buffers_.refc_in, buffers_.refx_in,
                                          buffers_.main_in));
  last_active_cap_len_ = active_cap_len;
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> ZImageDenoiserStage::EncodeAndRefineCaption(
    absl::Span<const float> prompt_embeds) {
  const size_t cap_elems =
      static_cast<size_t>(config_.seq_len) * static_cast<size_t>(config_.dim);

  // 1. Run `z_embc` (caption embedder) once per prompt.
  LITERT_RETURN_IF_ERROR(buffers_.embc_in[0].Write<float>(prompt_embeds));
  ABSL_RETURN_IF_ERROR(
      runners_.cap_embedder->Run("", buffers_.embc_in, buffers_.embc_out));
  std::vector<float> cap(cap_elems);
  LITERT_RETURN_IF_ERROR(buffers_.embc_out[0].Read<float>(absl::MakeSpan(cap)));

  // 2. Run `z_refc` (context refiner) once per prompt.
  LITERT_RETURN_IF_ERROR(
      buffers_.refc_in[0].Write<float>(absl::MakeConstSpan(cap)));
  ABSL_RETURN_IF_ERROR(
      runners_.context_refiner->Run("", buffers_.refc_in, buffers_.refc_out));
  std::vector<float> cap_refined(cap_elems);
  LITERT_RETURN_IF_ERROR(
      buffers_.refc_out[0].Read<float>(absl::MakeSpan(cap_refined)));
  return cap_refined;
}

absl::StatusOr<std::vector<float>> ZImageDenoiserStage::RunEulerDenoisingLoop(
    absl::Span<const float> cap_refined, const int steps, const uint64_t seed) {
  const size_t img_tokens = static_cast<size_t>(config_.ImageTokens());
  const size_t total_tokens = static_cast<size_t>(config_.TotalTokens());
  const size_t dim = static_cast<size_t>(config_.dim);
  const size_t packed_ch = static_cast<size_t>(config_.PackedChannels());
  const size_t adaln_dim = static_cast<size_t>(config_.AdalnDim());

  std::vector<float> latents =
      SampleGaussianLatents(img_tokens * packed_ch, seed);
  const std::vector<float> sigmas =
      ComputeZImageSigmas(steps, config_.ImageTokens(), config_.shift,
                          config_.use_dynamic_shifting);

  StepScratch scratch{
      .x_emb = std::vector<float>(img_tokens * dim),
      .adaln = std::vector<float>(adaln_dim),
      .x_refined = std::vector<float>(img_tokens * dim),
      .unified = std::vector<float>(total_tokens * dim),
      .model_out = std::vector<float>(img_tokens * packed_ch),
  };
  for (int step_idx = 0; step_idx < steps; ++step_idx) {
    ABSL_RETURN_IF_ERROR(RunDenoisingStep(sigmas[step_idx],
                                          sigmas[step_idx + 1], cap_refined,
                                          absl::MakeSpan(latents), scratch));
  }
  return latents;
}

absl::Status ZImageDenoiserStage::RunDenoisingStep(
    const float sigma, const float next_sigma,
    absl::Span<const float> cap_refined, absl::Span<float> latents,
    StepScratch& scratch) {
  // The exported `x_embedder` graph bakes `t_scale` (1000.0) internally and
  // expects unscaled continuous timesteps `t = 1 - sigma` in `[0, 1]`.
  const float t_val = 1.0f - sigma;

  LITERT_RETURN_IF_ERROR(buffers_.embx_in[0].Write<float>(latents));
  LITERT_RETURN_IF_ERROR(
      buffers_.embx_in[1].Write<float>(absl::MakeConstSpan(&t_val, 1)));
  ABSL_RETURN_IF_ERROR(
      runners_.x_embedder->Run("", buffers_.embx_in, buffers_.embx_out));
  LITERT_RETURN_IF_ERROR(
      buffers_.embx_out[0].Read<float>(absl::MakeSpan(scratch.x_emb)));
  LITERT_RETURN_IF_ERROR(
      buffers_.embx_out[1].Read<float>(absl::MakeSpan(scratch.adaln)));

  LITERT_RETURN_IF_ERROR(
      buffers_.refx_in[0].Write<float>(absl::MakeConstSpan(scratch.x_emb)));
  LITERT_RETURN_IF_ERROR(
      buffers_.refx_in[3].Write<float>(absl::MakeConstSpan(scratch.adaln)));
  ABSL_RETURN_IF_ERROR(
      runners_.noise_refiner->Run("", buffers_.refx_in, buffers_.refx_out));
  LITERT_RETURN_IF_ERROR(
      buffers_.refx_out[0].Read<float>(absl::MakeSpan(scratch.x_refined)));

  absl::c_copy(scratch.x_refined, scratch.unified.begin());
  absl::c_copy(cap_refined, scratch.unified.begin() + scratch.x_refined.size());

  for (size_t k = 0; k < runners_.main_blocks.size(); ++k) {
    LITERT_RETURN_IF_ERROR(buffers_.main_in[k][0].Write<float>(
        absl::MakeConstSpan(scratch.unified)));
    LITERT_RETURN_IF_ERROR(buffers_.main_in[k][3].Write<float>(
        absl::MakeConstSpan(scratch.adaln)));
    ABSL_RETURN_IF_ERROR(runners_.main_blocks[k]->Run("", buffers_.main_in[k],
                                                      buffers_.main_out[k]));
    LITERT_RETURN_IF_ERROR(
        buffers_.main_out[k][0].Read<float>(absl::MakeSpan(scratch.unified)));
  }

  LITERT_RETURN_IF_ERROR(
      buffers_.final_in[0].Write<float>(absl::MakeConstSpan(scratch.unified)));
  LITERT_RETURN_IF_ERROR(
      buffers_.final_in[1].Write<float>(absl::MakeConstSpan(scratch.adaln)));
  ABSL_RETURN_IF_ERROR(
      runners_.final_layer->Run("", buffers_.final_in, buffers_.final_out));
  LITERT_RETURN_IF_ERROR(
      buffers_.final_out[0].Read<float>(absl::MakeSpan(scratch.model_out)));

  ApplyEulerUpdate(latents, scratch.model_out, sigma - next_sigma);
  return absl::OkStatus();
}

}  // namespace litert::omni::text2image

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

#ifndef MODELS_OMNIGEN_OMNIGEN2_DENOISER_STAGE_H_
#define MODELS_OMNIGEN_OMNIGEN2_DENOISER_STAGE_H_

#include <array>
#include <cstddef>
#include <memory>
#include <vector>

#include "models/omnigen/omnigen2/io_types.h"
#include "models/omnigen/omnigen2/omnigen2_model_config.h"
#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/cc/litert_tensor_buffer.h"
#include "omni/base/litert_runner.h"
#include "omni/base/stage.h"

namespace litert::omni::text2image::omnigen2 {

// Stage 2 for `OmniGen2/OmniGen2`:
// Orchestrates the 11 DiT subgraphs (`cap_embedder`, `context_refiner`,
// `x_embedder`, `noise_refiner`, `main_blocks[0..5]`, `final_stage`) across the
// FlowMatch Euler integration loop (`t = 0.0 -> 1.0`,
// `lat += (t[k+1] - t[k]) * v_t`) to produce denoised packed latents
// `[1, grid_h * grid_w, packed_ch]`.
class OmniGen2DenoiserStage
    : public SingleThreadedStageWithDeque<OmniGen2DenoiserOutput> {
 public:
  // Input buffer indices for the 11 sharded `OmniGen2Transformer2DModel`
  // subgraphs (`cap_embedder`, `context_refiner`, `x_embedder`,
  // `noise_refiner`, `main_blocks[0..5]`, and `final_stage`).
  struct InputIndices {
    struct CapEmbedder {
      size_t cap = 0;  // args_0: [1, seq_len, text_hidden_size]
    };
    struct ContextRefiner {
      size_t cap = 0;   // args_0: [1, seq_len, dit_hidden_size]
      size_t cos = 1;   // args_1: [1, seq_len, 1, dit_head_dim]
      size_t sin = 2;   // args_2: [1, seq_len, 1, dit_head_dim]
      size_t mask = 3;  // args_3: [1, 1, 1, seq_len]
    };
    struct XEmbedder {
      size_t x = 0;  // args_0: [1, img_tokens, packed_ch]
      size_t t = 1;  // args_1: [1]
    };
    struct NoiseRefiner {
      size_t x = 0;     // args_0: [1, img_tokens, dit_hidden_size]
      size_t cos = 1;   // args_1: [1, img_tokens, 1, dit_head_dim]
      size_t sin = 2;   // args_2: [1, img_tokens, 1, dit_head_dim]
      size_t temb = 3;  // args_3: [1, temb_dim]
    };
    struct MainBlock {
      size_t joint = 0;  // args_0: [1, seq_len + img_tokens, dit_hidden_size]
      size_t cos = 1;    // args_1: [1, seq_len + img_tokens, 1, dit_head_dim]
      size_t sin = 2;    // args_2: [1, seq_len + img_tokens, 1, dit_head_dim]
      size_t mask = 3;   // args_3: [1, 1, 1, seq_len + img_tokens]
      size_t temb = 4;   // args_4: [1, temb_dim]
    };
    struct Final {
      size_t joint = 0;  // args_0: [1, seq_len + img_tokens, dit_hidden_size]
      size_t temb = 1;   // args_1: [1, temb_dim]
    };

    CapEmbedder cap_embedder;
    ContextRefiner context_refiner;
    XEmbedder x_embedder;
    NoiseRefiner noise_refiner;
    std::array<MainBlock, kNumMainBlockShards> main_blocks = {};
    Final final_stage;
  };

  struct Runners {
    std::unique_ptr<LiteRtRunner> cap_embedder;
    std::unique_ptr<LiteRtRunner> context_refiner;
    std::unique_ptr<LiteRtRunner> x_embedder;
    std::unique_ptr<LiteRtRunner> noise_refiner;
    std::array<std::unique_ptr<LiteRtRunner>, kNumMainBlockShards> main_blocks;
    std::unique_ptr<LiteRtRunner> final_stage;
  };

  static absl::StatusOr<std::unique_ptr<OmniGen2DenoiserStage>> Create(
      Stage<OmniGen2TextEncoderOutput>* absl_nonnull text_encoder,
      const OmniGen2ModelConfig& config, Runners runners,
      InputIndices input_indices);

  static absl::StatusOr<std::unique_ptr<OmniGen2DenoiserStage>> Create(
      Stage<OmniGen2TextEncoderOutput>* absl_nonnull text_encoder,
      const OmniGen2ModelConfig& config, Runners runners);

  ~OmniGen2DenoiserStage() override = default;

 protected:
  bool NeedScheduleInternal() const override;

  absl::Status ScheduleInternal() override;

 private:
  struct Buffers {
    std::vector<TensorBuffer> cap_emb_in;
    std::vector<TensorBuffer> cap_emb_out;
    std::vector<TensorBuffer> ctx_ref_in;
    std::vector<TensorBuffer> ctx_ref_out;
    std::vector<TensorBuffer> x_emb_in;
    std::vector<TensorBuffer> x_emb_out;
    std::vector<TensorBuffer> noise_ref_in;
    std::vector<TensorBuffer> noise_ref_out;
    std::array<std::vector<TensorBuffer>, kNumMainBlockShards> main_in;
    std::array<std::vector<TensorBuffer>, kNumMainBlockShards> main_out;
    std::vector<TensorBuffer> final_in;
    std::vector<TensorBuffer> final_out;
  };

  // Precomputed caption embeddings, 3D Omni-RoPE tables, and joint attention
  // mask for one CFG branch (conditional or unconditional).
  struct PreparedBranch {
    std::vector<float> cap_refined;
    std::vector<float> img_cos;
    std::vector<float> img_sin;
    std::vector<float> joint_cos;
    std::vector<float> joint_sin;
    std::vector<float> joint_mask;
  };

  // Reusable per-step host buffers shared across Euler iterations.
  struct DenoisingWorkspace {
    std::vector<float> x_emb;
    std::vector<float> x_refined;
    std::vector<float> temb;
    std::vector<float> joint;
    std::vector<float> velocity;
    std::vector<float> uncond_velocity;
  };

  OmniGen2DenoiserStage(
      Stage<OmniGen2TextEncoderOutput>* absl_nonnull text_encoder,
      OmniGen2ModelConfig config, Runners runners, InputIndices input_indices,
      Buffers buffers, int temb_dim);

  // Validates `prompt_embeds`, builds per-branch 3D Omni-RoPE and masks, and
  // runs `cap_embedder` + `context_refiner`.
  absl::StatusOr<PreparedBranch> PrepareBranchConditioning(
      absl::Span<const float> prompt_embeds, int requested_active_len,
      int grid_h, int grid_w, absl::string_view field_name);

  // Writes image/joint RoPE tables and joint attention mask into the
  // `noise_refiner` and `main_blocks` input buffers.
  absl::Status WriteBranchRopeAndMask(const PreparedBranch& branch);

  // Runs `x_embedder(latents, timestep)` and populates `x_emb` and `temb`.
  absl::Status RunXEmbedder(absl::Span<const float> latents, float timestep,
                            absl::Span<float> x_emb, absl::Span<float> temb);

  // Runs `noise_refiner` -> `main_blocks` -> `final_stage` for one CFG branch
  // and writes predicted velocity into `out_velocity`.
  absl::Status RunDenoiserBranch(const PreparedBranch& branch,
                                 bool write_branch_rope,
                                 DenoisingWorkspace& workspace,
                                 absl::Span<float> out_velocity);

  // Executes the FlowMatch Euler integration loop (`t = 0.0 -> 1.0`) in place
  // on `latents`, applying CFG when `uncond_branch` is non-null.
  absl::Status RunFlowMatchEulerLoop(
      int steps, absl::Span<const float> timesteps,
      const PreparedBranch& cond_branch,
      const PreparedBranch* absl_nullable uncond_branch,
      std::vector<float>& latents);

  Stage<OmniGen2TextEncoderOutput>& text_encoder_;
  const OmniGen2ModelConfig config_;
  Runners runners_;
  const InputIndices input_indices_;
  Buffers buffers_;
  const int temb_dim_;
};

}  // namespace litert::omni::text2image::omnigen2

#endif  // MODELS_OMNIGEN_OMNIGEN2_DENOISER_STAGE_H_

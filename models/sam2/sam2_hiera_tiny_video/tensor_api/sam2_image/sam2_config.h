// SAM2.1 hiera-tiny image path on the C++ Tensor API — configuration.
//
// Matches the reference pipelines this port is verified against:
// the mlx-swift app (LiteRT-Models/sam2-mlx-ios, corr 0.9999 vs PyTorch at
// 512) and transformers' facebook/sam2.1-hiera-tiny. Sizes are for the
// 512x512 input the apps run (token grid 128, image embeddings 32x32).

#ifndef MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_IMAGE_SAM2_CONFIG_H_
#define MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_IMAGE_SAM2_CONFIG_H_

#include <set>
#include <vector>

namespace litert::tensor::examples::sam2 {

struct Sam2Config {
  int image_size = 512;

  // Hiera-tiny trunk.
  int embed_dim = 96;
  int num_heads = 1;
  std::vector<int> stages{1, 2, 7, 2};
  std::vector<int> window_spec{8, 4, 14, 7};
  std::set<int> global_blocks{5, 7, 9};
  int q_pool = 3;
  int q_stride = 2;
  float dim_mul = 2.0f;
  float head_mul = 2.0f;
  int mlp_ratio = 4;

  // Neck / SAM.
  int d_model = 256;

  // Eps values from the reference implementations (Hiera LN explicit 1e-6;
  // decoder LayerNorm = torch/mlx default 1e-5; LayerNorm2d = 1e-6).
  float ln_eps_hiera = 1e-6f;
  float ln_eps_decoder = 1e-5f;
  float ln_eps_2d = 1e-6f;

  // Emission toggles (raw decompositions and composites are numerically
  // identical on CPU; composites unlock fused delegate kernels).
  bool use_sdpa_composite = false;
  // odml.runtime_bmm pair per attention (QK + AV composites with an int32
  // [1,1,1,7] control input, element 2 = full length — SAM2 shapes are
  // static, so both sides are bounded at full fill, which is complete
  // computation; the audio-side stale-tail hazard needs active < S).
  bool use_rbmm_attention = false;
  // The hypernetwork mask projection (activation x activation) as a single
  // dst-bounded odml.runtime_bmm at rank 4.
  bool use_rbmm_hypernet = false;
  bool use_layer_norm_composite = false;
  // Mask upsampler form: TRANSPOSE_CONV directly, or the exact
  // 4x(1x1)+concat+DepthToSpace expansion (k2/s2 transposed conv has zero
  // tap overlap, so the two are the same sums) for delegates that reject
  // TRANSPOSE_CONV (Mali ML Drift).
  bool use_d2s_upsampler = false;

  int token_grid() const { return image_size / 4; }
  int embed_grid() const { return image_size / 16; }  // 32 at 512
  int mask_grid() const { return image_size / 4; }    // 128 at 512

  struct BlockSpec {
    int dim;
    int dim_out;
    int heads;
    int window;    // 0 = global attention
    int q_stride;  // 0 = no query pooling
    int grid_in;
    int grid_out;
  };

  // Per-block specs following the reference construction exactly: the
  // window size uses the CURRENT stage spec fixed before the stage
  // increment, dim/heads scale at stage boundaries, q-pool hits the first
  // block of stages 2..4.
  std::vector<BlockSpec> Blocks() const {
    int depth = 0;
    for (int s : stages) depth += s;
    std::vector<int> ends;  // last block index of each stage
    int acc = 0;
    for (int s : stages) {
      acc += s;
      ends.push_back(acc - 1);
    }
    std::set<int> end_set(ends.begin(), ends.end());
    std::set<int> pool_blocks;
    for (int i = 0; i + 1 < static_cast<int>(ends.size()) && i < q_pool; ++i) {
      pool_blocks.insert(ends[i] + 1);
    }

    std::vector<BlockSpec> specs;
    int ed = embed_dim;
    int nh = num_heads;
    int cur_stage = 1;
    int grid = token_grid();
    for (int i = 0; i < depth; ++i) {
      int dim_out = ed;
      int ws = window_spec[cur_stage - 1];
      if (global_blocks.count(i)) ws = 0;
      if (i > 0 && end_set.count(i - 1)) {
        dim_out = static_cast<int>(ed * dim_mul);
        nh = static_cast<int>(nh * head_mul);
        ++cur_stage;
      }
      int qs = pool_blocks.count(i) ? q_stride : 0;
      int grid_out = qs ? grid / qs : grid;
      specs.push_back({ed, dim_out, nh, ws, qs, grid, grid_out});
      ed = dim_out;
      grid = grid_out;
    }
    return specs;
  }

  // Stage-end block indices (whose outputs feed the FPN neck).
  std::vector<int> StageEnds() const {
    std::vector<int> ends;
    int acc = 0;
    for (int s : stages) {
      acc += s;
      ends.push_back(acc - 1);
    }
    return ends;
  }
};

}  // namespace litert::tensor::examples::sam2

#endif  // MODELS_SAM2_SAM2_HIERA_TINY_VIDEO_TENSOR_API_SAM2_IMAGE_SAM2_CONFIG_H_

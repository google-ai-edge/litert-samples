// Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================

// Tensor API graphs for the SAM 2 ModelChain pipeline.
//
// Two models, both authored here with the LiteRT Tensor API and serialized by
// ModelFactory (no converter):
//
//   SAM 2 model (per input size, shared weights across signatures):
//     encode          pixels -> pix_raw, feat_s1, feat_s0 (Hiera + FPN)
//     prompt{1..8}    decode k clicks + in-graph post + memory encoder
//     track{2,7}      memory attention over N memory slots + 16 object
//                     pointers, each bound as its OWN input tensor (the bank
//                     is concatenated in-graph), then decode + post + memorize
//
//   Frame model (per video geometry, built at run time):
//     preprocess      RGBA frame [1,H,stride,4] in [0,1] -> pixels [1,S,S,3]
//                     (crop, anti-alias average pool, bilinear resize,
//                     ImageNet normalization)
//     composite       frame + up to kMaxObjects low-res mask logits + effect
//                     parameters -> display RGB [1,H,W,3] (bilinear upsample,
//                     signed distance to the boundary, fill / outline over the
//                     frame, or a union matte for spotlight / green cutout)
//
// Taking every memory slot as a separate input makes the memory bank pure
// buffer binding: the host points mem_k / ptr_k at buffers earlier steps
// wrote, with no copies.

#ifndef SAM2_WEB_TENSORAPI_CC_CHAIN_GRAPHS_H_
#define SAM2_WEB_TENSORAPI_CC_CHAIN_GRAPHS_H_

#include <array>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video/sam2v_graph.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"

namespace sam2_chain {

namespace s2v = ::litert::tensor::examples::sam2_video;
using TfTensor = ::litert::tensor::examples::sam2::TfTensor;
using WeightMap = ::litert::tensor::examples::sam2::WeightMap;
using ::litert::tensor::ModelFactory;

inline constexpr int kMaxObjects = 5;
inline constexpr int kMaxClicks = 8;
inline constexpr int kNumPtrFrames = 16;
inline constexpr int kPtrSplit = 4;
inline constexpr int kMemCh = 64;
inline constexpr int kHidden = 256;

using Rgb = std::array<float, 3>;  // 0..1

// Object colours (the web demo's palette) and the cutout background.
inline constexpr std::array<Rgb, kMaxObjects> kPalette = {{
    {76 / 255.f, 141 / 255.f, 255 / 255.f},
    {255 / 255.f, 158 / 255.f, 44 / 255.f},
    {62 / 255.f, 207 / 255.f, 142 / 255.f},
    {240 / 255.f, 82 / 255.f, 156 / 255.f},
    {170 / 255.f, 110 / 255.f, 255 / 255.f},
}};
inline constexpr Rgb kCutoutGreen = {0.f, 177 / 255.f, 64 / 255.f};

// Row layout of the RGBA frame tensor. stride >= width: WebGPU texture ->
// buffer copies need 256-byte rows (16 rgba32float texels), so the browser
// passes padded rows and the graph crops them.
struct FrameGeometry {
  int width = 0;
  int height = 0;
  int stride = 0;
  static FrameGeometry For(int width, int height) {
    return {width, height, (width + 15) / 16 * 16};
  }
};

// Anti-alias average-pool factors applied before the bilinear resize to S.
struct PoolFactors {
  int ky = 1, kx = 1;
};
PoolFactors PreprocessPool(const FrameGeometry& geo, int image_size);

// composite's `fx` input [1,1,1,kFxCount], one parameter per entry.
enum FxIndex {
  kFxFill = 0,    // overlay fill opacity (0.5), 0 in matte modes
  kFxRing = 1,    // outline opacity (0.95), 0 = no outline
  kFxStroke = 2,  // outline width, display px
  kFxMatte = 3,   // 1 = spotlight / cutout (union matte), 0 = overlay
  kFxSpot = 4,    // matte background: 1 = dimmed grey frame, 0 = green
  kFxCount = 8,
};
enum class Effect { kOverlay, kSpotlight, kCutout };
std::array<float, kFxCount> EffectParams(Effect effect, float stroke_px);

// Encode, prompt{1..kMaxClicks} and track{2,7}. `track_sparse` is the baked
// sparse prompt row pair for tracked frames (host consts "track_sparse").
absl::Status AddSam2Signatures(ModelFactory& factory,
                               const s2v::Sam2VideoConfig& config,
                               const WeightMap& weights,
                               const std::vector<float>& track_sparse);

// Preprocess + composite for one frame geometry.
absl::Status AddFrameSignatures(ModelFactory& factory,
                                const FrameGeometry& geo, int image_size);

// Signature names.
std::string PromptSignature(int clicks);   // "prompt{k}"
std::string TrackSignature(int nmm);       // "track{n}"
std::string MemInput(int slot);            // "mem_{k}"
std::string PtrInput(int slot);            // "ptr_{k}"
std::string MaskInput(int object);         // "mask_{k}"

}  // namespace sam2_chain

#endif  // SAM2_WEB_TENSORAPI_CC_CHAIN_GRAPHS_H_

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

// Host-side SAM 2 bookkeeping for the ModelChain pipeline, in plain C++:
// click encoding, object-pointer temporal encodings, and the memory plan —
// which stored frames fill the track{n} memory slots / pointer inputs this
// frame, plus the small per-slot tables. Port of sam2v_main.cc's host loop
// (and the web demo's host.ts planMemCond); no LiteRT dependency, so it is
// unit-tested on its own.

#ifndef SAM2_WEB_TENSORAPI_CC_HOST_PLAN_H_
#define SAM2_WEB_TENSORAPI_CC_HOST_PLAN_H_

#include <functional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl

namespace sam2_chain {

// Host tables exported from the checkpoint (web/tools/export_host_assets.py).
struct HostConsts {
  std::vector<float> gaussian;      // (2,128) prompt-encoder Fourier matrix
  std::vector<float> point_embed0;  // (256) negative-click embedding
  std::vector<float> point_embed1;  // (256) positive-click embedding
  std::vector<float> point_embed2;  // (256) box top-left corner (empty: no box support)
  std::vector<float> point_embed3;  // (256) box bottom-right corner
  std::vector<float> not_a_point;   // (256) pad token
  std::vector<float> track_sparse;  // (2,256) baked sparse rows, tracked frames
  std::vector<float> mtpe;          // (7,64) memory temporal position rows
  std::vector<float> tpos_w;        // (64,256) pointer temporal projection
  std::vector<float> tpos_b;        // (64)

  static absl::StatusOr<HostConsts> Load(const std::string& safetensors_path);
};

// A prompt point in model space (the S x S squashed frame). label: 1 =
// positive click, 0 = negative click, 2 / 3 = a box's top-left /
// bottom-right corner. As in SAM 2 video, a box is two corner points placed
// before any clicks, so a box prompt runs through the same prompt{k} graphs.
struct Click {
  float x = 0, y = 0;
  int label = 1;
};

// Sparse prompt rows for k points + the not-a-point pad: [(k+1) * 256].
// Point i: Fourier position encoding + the embedding of its label.
std::vector<float> PointsSparse(const HostConsts& c,
                                const std::vector<Click>& clicks,
                                int image_size);

// tpos_proj(get_1d_sine_pe(t_diff / 15, 256)) -> (64).
std::vector<float> PtrPos(const HostConsts& c, int t_diff);

// Which stored frames feed track{nmm} at frame t, and the per-slot tables.
// Slots are compact (used slots first); unused slot / pointer inputs are
// masked out by key_mask and may be bound to any finite buffer.
struct MemPlan {
  std::vector<int> slot_frames;  // frame per used memory slot (<= nmm)
  std::vector<int> ptr_frames;   // frame per used pointer input (<= 16)
  std::vector<float> slot_tpe;   // [nmm, 64]
  std::vector<float> ptr_pos;    // [64, 64] (16 pointers x 4 rows)
  std::vector<float> key_mask;   // [nmm * hw + 64], 0 = attend, kMaskNeg = off
};

// -30000: fp16-safe additive mask (exactly zero softmax weight in fp32).
inline constexpr float kMaskNeg = -30000.0f;

// cond_frame: the object's prompt frame (slot 0, temporal row 6).
// has_mem(f) / has_ptr(f): whether frame f's memory / pointer is stored.
MemPlan PlanMemory(const HostConsts& c, int cond_frame, int t, int nmm, int hw,
                   const std::function<bool(int)>& has_mem,
                   const std::function<bool(int)>& has_ptr);

}  // namespace sam2_chain

#endif  // SAM2_WEB_TENSORAPI_CC_HOST_PLAN_H_

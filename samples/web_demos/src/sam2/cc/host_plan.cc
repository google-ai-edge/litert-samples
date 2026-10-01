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

#include "host_plan.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/examples/gemma3/safetensors.h"

namespace sam2_chain {
namespace {

constexpr int kHidden = 256;
constexpr int kMemCh = 64;
constexpr int kNumPtrFrames = 16;
constexpr int kPtrSplit = 4;
constexpr float kTwoPi = 6.283185307179586f;

}  // namespace

absl::StatusOr<HostConsts> HostConsts::Load(const std::string& path) {
  safetensors::safetensors_t st;
  std::string warn, err;
  if (!safetensors::load_from_file(path, &st, &warn, &err)) {
    return absl::NotFoundError(absl::StrCat("host consts ", path, ": ", err));
  }
  const uint8_t* base =
      st.mmaped ? st.databuffer_addr : st.storage.data();
  auto get = [&](const std::string& key, size_t n,
                 std::vector<float>& out) -> absl::Status {
    safetensors::tensor_t t;
    if (!st.tensors.at(key, &t)) {
      return absl::NotFoundError(absl::StrCat("host consts: missing ", key));
    }
    if (t.dtype != safetensors::dtype::kFLOAT32 ||
        safetensors::get_shape_size(t) != n) {
      return absl::InvalidArgumentError(
          absl::StrCat("host consts: ", key, " must be fp32 with ", n,
                       " elements"));
    }
    out.resize(n);
    std::memcpy(out.data(), base + t.data_offsets[0], n * sizeof(float));
    return absl::OkStatus();
  };
  HostConsts c;
  // Box corners (labels 2 / 3): optional, older host-table files lack them.
  safetensors::tensor_t probe;
  if (st.tensors.at("point_embed2", &probe)) {
    for (auto st2 : {get("point_embed2", kHidden, c.point_embed2),
                     get("point_embed3", kHidden, c.point_embed3)}) {
      if (!st2.ok()) return st2;
    }
  }
  for (auto st2 : {get("gaussian", 2 * 128, c.gaussian),
                   get("point_embed0", kHidden, c.point_embed0),
                   get("point_embed1", kHidden, c.point_embed1),
                   get("not_a_point", kHidden, c.not_a_point),
                   get("track_sparse", 2 * kHidden, c.track_sparse),
                   get("mtpe", 7 * kMemCh, c.mtpe),
                   get("tpos_w", kMemCh * kHidden, c.tpos_w),
                   get("tpos_b", kMemCh, c.tpos_b)}) {
    if (!st2.ok()) return st2;
  }
  return c;
}

std::vector<float> PointsSparse(const HostConsts& c,
                                const std::vector<Click>& clicks,
                                int image_size) {
  const size_t k = clicks.size();
  std::vector<float> out((k + 1) * kHidden);
  for (size_t p = 0; p < k; ++p) {
    const float xn = 2.0f * ((clicks[p].x + 0.5f) / image_size) - 1.0f;
    const float yn = 2.0f * ((clicks[p].y + 0.5f) / image_size) - 1.0f;
    const int label = clicks[p].label;
    const std::vector<float>& emb = label == 1   ? c.point_embed1
                                    : label == 2 ? c.point_embed2
                                    : label == 3 ? c.point_embed3
                                                 : c.point_embed0;
    float* o = out.data() + p * kHidden;
    for (int i = 0; i < 128; ++i) {
      const float proj = kTwoPi * (xn * c.gaussian[i] + yn * c.gaussian[128 + i]);
      o[i] = std::sin(proj) + emb[i];
      o[128 + i] = std::cos(proj) + emb[128 + i];
    }
  }
  std::copy(c.not_a_point.begin(), c.not_a_point.end(),
            out.begin() + k * kHidden);
  return out;
}

std::vector<float> PtrPos(const HostConsts& c, int t_diff) {
  // Double accumulation like the reference (fp32 rounding only at the end).
  double pe[kHidden];
  const double pos = static_cast<double>(t_diff) / (kNumPtrFrames - 1.0);
  for (int i = 0; i < 128; ++i) {
    const double dim_t = std::pow(10000.0, 2.0 * (i / 2) / 128.0);
    pe[i] = std::sin(pos / dim_t);
    pe[128 + i] = std::cos(pos / dim_t);
  }
  std::vector<float> out(kMemCh);
  for (int r = 0; r < kMemCh; ++r) {
    double acc = c.tpos_b[r];
    for (int k = 0; k < kHidden; ++k) {
      acc += static_cast<double>(c.tpos_w[r * kHidden + k]) * pe[k];
    }
    out[r] = static_cast<float>(acc);
  }
  return out;
}

MemPlan PlanMemory(const HostConsts& c, int cond_frame, int t, int nmm, int hw,
                   const std::function<bool(int)>& has_mem,
                   const std::function<bool(int)>& has_ptr) {
  MemPlan plan;
  plan.slot_tpe.assign(nmm * kMemCh, 0.f);
  plan.ptr_pos.assign(kNumPtrFrames * kPtrSplit * kMemCh, 0.f);
  plan.key_mask.assign(static_cast<size_t>(nmm) * hw +
                           kNumPtrFrames * kPtrSplit,
                       kMaskNeg);
  auto put_slot = [&](int frame, int row) {
    const int k = static_cast<int>(plan.slot_frames.size());
    std::copy(c.mtpe.begin() + row * kMemCh, c.mtpe.begin() + (row + 1) * kMemCh,
              plan.slot_tpe.begin() + k * kMemCh);
    std::fill(plan.key_mask.begin() + static_cast<size_t>(k) * hw,
              plan.key_mask.begin() + static_cast<size_t>(k + 1) * hw, 0.f);
    plan.slot_frames.push_back(frame);
  };
  // The conditioning frame always takes slot 0 with the "conditioning"
  // temporal row; then the most recent nmm-1 non-conditioning frames.
  put_slot(cond_frame, 6);
  for (int off = nmm - 1; off >= 1; --off) {
    const int pf = t - off;
    if (pf != cond_frame && has_mem(pf)) put_slot(pf, off - 1);
  }

  auto add_ptr = [&](int frame, int t_diff) {
    const std::vector<float> pos = PtrPos(c, t_diff);
    for (int j = 0; j < kPtrSplit; ++j) {
      const int row = static_cast<int>(plan.ptr_frames.size()) * kPtrSplit + j;
      std::copy(pos.begin(), pos.end(), plan.ptr_pos.begin() + row * kMemCh);
      plan.key_mask[static_cast<size_t>(nmm) * hw + row] = 0.f;
    }
    plan.ptr_frames.push_back(frame);
  };
  add_ptr(cond_frame, t - cond_frame);
  for (int td = 1; td < kNumPtrFrames; ++td) {
    const int pf = t - td;
    if (pf < 0) break;
    if (pf != cond_frame && has_ptr(pf)) add_ptr(pf, td);
  }
  return plan;
}

}  // namespace sam2_chain

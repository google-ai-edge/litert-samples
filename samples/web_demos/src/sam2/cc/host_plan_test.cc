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

// Host plan unit tests: click / pointer encodings against Hugging Face
// goldens (web/tools/export_host_assets.py), and the memory plan's slot /
// pointer selection and tables.

#include "host_plan.h"

#include <cmath>
#include <cstring>
#include <set>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "tensor/examples/gemma3/safetensors.h"

namespace sam2_chain {
namespace {

std::string g_goldens;  // --goldens=<path>
std::string ConstsPath() {
  return g_goldens.substr(0, g_goldens.rfind('/')) +
         "/sam2_host_consts.safetensors";
}

class Goldens {
 public:
  explicit Goldens(const std::string& path) {
    std::string warn, err;
    ok_ = safetensors::load_from_file(path, &st_, &warn, &err);
  }
  bool ok() const { return ok_; }
  std::vector<float> Get(const std::string& key) const {
    safetensors::tensor_t t;
    if (!st_.tensors.at(key, &t)) return {};
    std::vector<float> out(safetensors::get_shape_size(t));
    const uint8_t* base = st_.mmaped ? st_.databuffer_addr : st_.storage.data();
    std::memcpy(out.data(), base + t.data_offsets[0], out.size() * 4);
    return out;
  }

 private:
  safetensors::safetensors_t st_;
  bool ok_ = false;
};

float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b) {
  EXPECT_EQ(a.size(), b.size());
  float m = 0;
  for (size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    m = std::max(m, std::abs(a[i] - b[i]));
  }
  return m;
}

class HostPlanTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto c = HostConsts::Load(ConstsPath());
    ASSERT_TRUE(c.ok()) << c.status();
    consts_ = *c;
    goldens_ = std::make_unique<Goldens>(g_goldens);
    ASSERT_TRUE(goldens_->ok());
  }
  HostConsts consts_;
  std::unique_ptr<Goldens> goldens_;
};

TEST_F(HostPlanTest, SingleClickMatchesHf) {
  for (int i = 0; i < 3; ++i) {
    const auto xy = goldens_->Get("click_" + std::to_string(i) + "_xy");
    const auto got = PointsSparse(consts_, {{xy[0], xy[1], 1}}, 1024);
    EXPECT_LT(MaxAbsDiff(got, goldens_->Get("click_" + std::to_string(i))), 1e-4)
        << "click " << i;
  }
}

TEST_F(HostPlanTest, MultiClickPositiveNegativeMatchesHf) {
  for (const std::string name : {"mp_a", "mp_b"}) {
    const auto pts = goldens_->Get(name + "_pts");  // (k, 3): x, y, label
    std::vector<Click> clicks;
    for (size_t i = 0; i + 2 < pts.size(); i += 3) {
      clicks.push_back({pts[i], pts[i + 1], static_cast<int>(pts[i + 2])});
    }
    const auto got = PointsSparse(consts_, clicks, 1024);
    EXPECT_EQ(got.size(), (clicks.size() + 1) * 256u);
    EXPECT_LT(MaxAbsDiff(got, goldens_->Get(name)), 1e-4) << name;
  }
}

TEST_F(HostPlanTest, BoxPromptMatchesHf) {
  // A box is its two corners (labels 2, 3) before any clicks, as HF's video
  // processor builds it; box_b adds a positive and a negative click.
  ASSERT_FALSE(consts_.point_embed2.empty());
  for (const std::string name : {"box_a", "box_b"}) {
    const auto pts = goldens_->Get(name + "_pts");
    std::vector<Click> clicks;
    for (size_t i = 0; i + 2 < pts.size(); i += 3) {
      clicks.push_back({pts[i], pts[i + 1], static_cast<int>(pts[i + 2])});
    }
    const auto got = PointsSparse(consts_, clicks, 1024);
    EXPECT_LT(MaxAbsDiff(got, goldens_->Get(name)), 1e-4) << name;
  }
}

TEST_F(HostPlanTest, PointerTemporalEncodingMatchesHf) {
  const auto ref = goldens_->Get("ptr_pos");  // (16, 64), t_diff 0..15
  for (int td = 0; td < 16; ++td) {
    const auto got = PtrPos(consts_, td);
    std::vector<float> row(ref.begin() + td * 64, ref.begin() + (td + 1) * 64);
    EXPECT_LT(MaxAbsDiff(got, row), 1e-5) << "t_diff " << td;
  }
}

TEST_F(HostPlanTest, FirstTrackedFrameUsesOnlyTheConditioningFrame) {
  const int hw = 4;
  const MemPlan p = PlanMemory(consts_, /*cond=*/0, /*t=*/1, /*nmm=*/7, hw,
                               [](int f) { return f == 0; },
                               [](int f) { return f == 0; });
  EXPECT_EQ(p.slot_frames, std::vector<int>({0}));
  EXPECT_EQ(p.ptr_frames, std::vector<int>({0}));
  // Slot 0 attends, slots 1..6 masked; pointer rows 0..3 attend.
  for (int i = 0; i < 7 * hw; ++i) {
    EXPECT_EQ(p.key_mask[i], i < hw ? 0.f : kMaskNeg) << i;
  }
  for (int r = 0; r < 64; ++r) {
    EXPECT_EQ(p.key_mask[7 * hw + r], r < 4 ? 0.f : kMaskNeg) << r;
  }
  // Conditioning slot takes temporal row 6.
  for (int c = 0; c < 64; ++c) EXPECT_EQ(p.slot_tpe[c], consts_.mtpe[6 * 64 + c]);
}

TEST_F(HostPlanTest, SteadyStateSlotsAreCondPlusRecentOldestFirst) {
  const int hw = 4, t = 40;
  auto all = [](int f) { return f >= 0; };
  const MemPlan p = PlanMemory(consts_, /*cond=*/3, t, /*nmm=*/7, hw, all, all);
  EXPECT_EQ(p.slot_frames, std::vector<int>({3, 34, 35, 36, 37, 38, 39}));
  // Recent frame t-off takes temporal row off-1.
  for (int k = 1; k < 7; ++k) {
    const int off = t - p.slot_frames[k];
    for (int c = 0; c < 64; ++c) {
      ASSERT_EQ(p.slot_tpe[k * 64 + c], consts_.mtpe[(off - 1) * 64 + c]);
    }
  }
  // Pointers: the conditioning frame first, then t-1 .. t-15.
  ASSERT_EQ(p.ptr_frames.size(), 16u);
  EXPECT_EQ(p.ptr_frames[0], 3);
  for (int i = 1; i < 16; ++i) EXPECT_EQ(p.ptr_frames[i], t - i);
  // Conditioning pointer's position encodes its true distance.
  const auto pos = PtrPos(consts_, t - 3);
  for (int c = 0; c < 64; ++c) EXPECT_EQ(p.ptr_pos[c], pos[c]);
  for (float v : p.key_mask) EXPECT_EQ(v, 0.f);
}

TEST_F(HostPlanTest, TwoSlotBankAndConditioningFrameNotDuplicated) {
  auto all = [](int f) { return f >= 0; };
  // t-1 is the conditioning frame: it must not appear twice.
  const MemPlan p = PlanMemory(consts_, /*cond=*/9, /*t=*/10, /*nmm=*/2, 4, all, all);
  EXPECT_EQ(p.slot_frames, std::vector<int>({9}));
  const MemPlan q = PlanMemory(consts_, /*cond=*/0, /*t=*/10, /*nmm=*/2, 4, all, all);
  EXPECT_EQ(q.slot_frames, std::vector<int>({0, 9}));
  std::set<int> uniq(q.ptr_frames.begin(), q.ptr_frames.end());
  EXPECT_EQ(uniq.size(), q.ptr_frames.size());
  EXPECT_EQ(q.ptr_frames.size(), 10u);  // cond 0 + frames 9..1
}

TEST_F(HostPlanTest, MissingFramesLeaveGapsCompacted) {
  // Frames 36 and 38 were never stored (e.g. skipped): slots stay compact.
  auto has = [](int f) { return f != 36 && f != 38; };
  const MemPlan p = PlanMemory(consts_, 0, 40, 7, 4, has, has);
  EXPECT_EQ(p.slot_frames, std::vector<int>({0, 34, 35, 37, 39}));
  EXPECT_EQ(p.key_mask[5 * 4], kMaskNeg);  // slot 5 unused
}

}  // namespace
}  // namespace sam2_chain

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a.rfind("--goldens=", 0) == 0) sam2_chain::g_goldens = a.substr(10);
  }
  return RUN_ALL_TESTS();
}

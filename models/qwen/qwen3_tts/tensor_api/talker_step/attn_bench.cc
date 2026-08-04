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

// Attention micro-benchmark: raw ops vs odml.scaled_dot_product_attention vs
// odml.runtime_bmm, at the talker decode shapes (heads 16 / kv 8 / head_dim
// 128 / 1024-slot static cache, q_seq = 1).
//
//   bazel build -c opt //models/qwen/qwen3_tts/tensor_api/talker_step:attn_bench
//   attn_bench --variant=raw|sdpa|rbmm --accelerator=cpu|gpu \
//              --fills=64,512,1024 [--mha] [--blocks=28]
//
// One graph = `--blocks` chained attention blocks sharing the same cache
// inputs (q_{i+1} = out_i), so per-run time ~= the attention load of one
// talker decode frame. All variants consume identical logical inputs and
// produce outputs in the same linear element order, so results are directly
// comparable across variants and backends.
//
//   raw : the talker's GqaAttention fold (BatchMatMul + mask add + Softmax).
//   sdpa: odml.scaled_dot_product_attention composite, BSND inputs
//         (q [1,1,H,D], k/v [1,S,G,D], additive mask 4th input, scale attr);
//         decomposition = the raw fold, so CPU runs are the reference.
//   rbmm: odml.runtime_bmm composite pair (QK^T with is_src=false, then
//         scores x V with is_src=true), fp32 BMM path (transpose_right),
//         runtime control tensor int32 [1,1,1,7] with element 2 = active
//         token count; V is provided transposed [1,G,D,S] so valid positions
//         sit on channels for the src-side runtime bound. The additive mask
//         stays in-graph between the two composites, so the CPU
//         decomposition (plain BatchMatMul, ignores the control tensor) is
//         exact and the GPU kernel's runtime bound is observable as a pure
//         perf effect (and any tail-garbage leak shows up as a numeric
//         mismatch vs the CPU reference).
//
// The active length is runtime input only (mask contents / control tensor),
// so one flatbuffer serves every fill level. K/V rows at or beyond the
// active length are filled with `--canary` to expose reads past the bound.
// A CPU runner on the same flatbuffer is always created as the numeric
// reference; GPU runs report max |rel| vs that reference per fill.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/runners/litert/feedback_loop_config.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, variant, "raw", "raw|sdpa|rbmm|cache_rbmm");
ABSL_FLAG(std::string, multi_host_cache, "zeros",
          "cache_rbmm multi_steps: what the measured GPU runner's host cache "
          "inputs carry per step. zeros = all-zero every step; real = true "
          "prefix rows every step; once = bind zeros at pos=0 then never "
          "re-set (probes whether re-binding resets delegate cache state)");
ABSL_FLAG(bool, feedback_loops, false,
          "cache_rbmm multi_steps: register cache_k/v as FeedbackLoopConfig "
          "pairs (PR #8796 runner API): caches become signature outputs, the "
          "runner swaps input<->output buffers between Run() calls, and the "
          "GPU runs in external-tensors mode. Host cache inputs are never "
          "bound (the swap chain owns them; unwritten rows are never read).");
ABSL_FLAG(int, multi_steps, 0,
          "cache_rbmm only: run N chained decode steps (pos = 0..N-1) on one "
          "runner instance and verify each step against the CPU reference. "
          "The GPU runner gets ZEROED host cache inputs, so agreement proves "
          "the delegate-managed cache accumulates across Run() calls.");
ABSL_FLAG(bool, cache_qk_bound, false,
          "cache_rbmm only: bound the QK side with the active length instead "
          "of running it unbounded (probes the intended composition for "
          "delegate-managed caches, where the host cannot pre-zero the "
          "packed cache storage)");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu|gpu");
ABSL_FLAG(int, blocks, 28, "chained attention blocks (one talker frame = 28)");
ABSL_FLAG(int, heads, 16, "query heads");
ABSL_FLAG(int, kv_groups, 8, "kv groups (GQA); ignored with --mha");
ABSL_FLAG(int, head_dim, 128, "head dim");
ABSL_FLAG(int, max_seq, 1024, "static cache slots");
ABSL_FLAG(bool, mha, false,
          "repeat K/V host-side to one group per head (MHA-shaped control; "
          "isolates GQA handling in the composite kernels)");
ABSL_FLAG(std::string, fills, "64,512,1024",
          "comma list of active lengths to measure");
ABSL_FLAG(bool, param_align4, false,
          "round the control tensor's active count up to a multiple of 4 "
          "(probes the 'aligned' semantics of element 2; mask stays exact)");
ABSL_FLAG(int, runs, 50, "timed runs per fill");
ABSL_FLAG(int, warmup, 5, "untimed warmup runs per fill");
ABSL_FLAG(float, canary, 1000.0f,
          "K/V value at rows >= active (leak detector; 0 to disable)");
ABSL_FLAG(int, seed, 123, "input RNG seed");
ABSL_FLAG(std::string, gpu_precision, "fp16", "fp16|fp32");
ABSL_FLAG(std::string, gpu_buffer_storage, "default",
          "default|buffer|texture2d (Metal needs buffer at max_seq >= 1024)");
ABSL_FLAG(std::string, tflite_path, "/tmp/attn_bench.tflite",
          "where to write the flatbuffer");
ABSL_FLAG(std::string, dump_out, "",
          "optional path to write the final output floats (cross-variant "
          "exact diffing)");
ABSL_FLAG(bool, debug_softmax, false,
          "rbmm only: also output block-0 softmax [1,G,gsz,S] and print its "
          "tail statistics on the measured runner (mask-vs-bound probe)");
ABSL_FLAG(std::string, rbmm_mode, "corrected",
          "rbmm only. 'masked': scale+mask add between the composites (the "
          "additive mask gets fused into the bounded QK kernel's epilogue on "
          "ML Drift, so the zero tail reaches the softmax unmasked - kept as "
          "the repro of that boundary). 'nomask': rbmm_qk -> softmax -> "
          "rbmm_av with pre-scaled Q, no correction (shows the raw boundary "
          "semantics). 'corrected': nomask + exact renormalization - the "
          "tail is known to be uniform e^{-m}/Z, so out * 1/(1 - "
          "tail_count*p0) with p0 = attn[..., S-1] recovers masked attention "
          "exactly, PROVIDED cache slots at/after the active length are "
          "zero (the natural DUS cache state; canary is forced to 0)");

namespace {

using ::litert::tensor::Add;
using ::litert::tensor::BatchMatMul;
using ::litert::tensor::Create;
using ::litert::tensor::LitertDynamicRunner;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::Mul;
using ::litert::tensor::OwningCpuBuffer;
using ::litert::tensor::Div;
using ::litert::tensor::DynamicUpdateSlice;
using ::litert::tensor::Reshape;
using ::litert::tensor::Slice;
using ::litert::tensor::Softmax;
using ::litert::tensor::Sub;
using ::litert::tensor::StableHLOComposite;
using ::litert::tensor::StableHLOCompositeOptions;
using ::litert::tensor::TensorHandle;
using ::litert::tensor::Transpose;
using ::litert::tensor::Type;
using TfTensor = ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>;

// fp16-safe additive mask floor (talker convention).
constexpr float kMaskFloor = -30000.0f;

struct Geometry {
  int heads;      // query heads H
  int groups;     // kv groups G (== H with --mha)
  int group_sz;   // H / G
  int head_dim;   // D
  int max_seq;    // S
};

TfTensor Const1(float value) {
  return TfTensor({.type = Type::kFP32,
                   .shape = {1},
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>({value})});
}

std::vector<uint8_t> RuntimeBmmAttributes(bool is_src) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Bool("is_global", true);
    fbb.Bool("is_src", is_src);
    fbb.Bool("rhs_cache_update", false);
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

std::vector<uint8_t> SdpaAttributes(float scale) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Float("scale", scale); });
  fbb.Finish();
  return fbb.GetBuffer();
}

// The talker's GqaAttention fold on pre-shaped tensors. q [1,H,seq,D],
// k/v [1,G,S,D] -> [1,H,seq,D]. Used directly by the raw variant and as the
// sdpa decomposition body (after BSND->BNSD transposes).
TfTensor FoldAttention(const Geometry& g, const TfTensor& q, const TfTensor& k,
                       const TfTensor& v, const TfTensor& mask) {
  const int seq = q.GetShape()[2];
  const float scale = 1.0f / std::sqrt(static_cast<float>(g.head_dim));
  TfTensor qg = Reshape(q, {g.groups, 1, g.group_sz * seq, g.head_dim});
  TfTensor kg = Reshape(k, {g.groups, 1, g.max_seq, g.head_dim});
  TfTensor vg = Reshape(v, {g.groups, 1, g.max_seq, g.head_dim});
  TfTensor scores = BatchMatMul(qg, kg, /*adj_x=*/false, /*adj_y=*/true);
  scores = Reshape(scores, {1, g.heads, seq, g.max_seq});
  scores = Mul(scores, Const1(scale));
  scores = Add(scores, mask);
  TfTensor attn = Softmax(scores);
  attn = Reshape(attn, {g.groups, 1, g.group_sz * seq, g.max_seq});
  TfTensor out = BatchMatMul(attn, vg);
  return Reshape(out, {1, g.heads, seq, g.head_dim});
}

// One sdpa composite block. q [1,1,H,D] BSND, k/v [1,S,G,D], mask
// [1,1,1,S] additive -> [1,1,H,D].
TfTensor SdpaBlock(const Geometry& g, const TfTensor& q, const TfTensor& k,
                   const TfTensor& v, const TfTensor& mask) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(g.head_dim));
  StableHLOCompositeOptions opts{
      .name = "odml.scaled_dot_product_attention",
      .composite_attributes = SdpaAttributes(scale)};
  auto out = StableHLOComposite(
      opts,
      [&g](TfTensor dq, TfTensor dk, TfTensor dv, TfTensor dm) {
        TfTensor qt = Transpose(dq, {0, 2, 1, 3});   // [1,H,1,D]
        TfTensor kt = Transpose(dk, {0, 2, 1, 3});   // [1,G,S,D]
        TfTensor vt = Transpose(dv, {0, 2, 1, 3});
        TfTensor o = FoldAttention(g, qt, kt, vt, dm);
        return Transpose(o, {0, 2, 1, 3});           // back to [1,1,H,D]
      },
      q, k, v, mask);
  return out;
}

// One runtime_bmm block on pre-folded tensors. qf [1,G,gsz,D],
// k [1,G,S,D], v_t [1,G,D,S], mask [1,1,1,S], param int32 [1,1,1,7]
// -> [1,G,gsz,D]. The decompositions are plain BatchMatMuls that ignore the
// control tensor; correctness on CPU comes from the in-graph mask.
enum class RbmmMode { kMasked, kNoMask, kCorrected, kAvBound };

TfTensor RuntimeBmmBlock(const Geometry& g, const TfTensor& qf,
                         const TfTensor& k, const TfTensor& v_t,
                         const TfTensor& mask, const TfTensor& param,
                         const TfTensor& param_full,
                         const TfTensor& tail_count, RbmmMode mode,
                         TfTensor* attn_out = nullptr) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(g.head_dim));
  const bool pre_scale =
      mode == RbmmMode::kNoMask || mode == RbmmMode::kCorrected;
  TfTensor lhs = pre_scale ? Mul(qf, Const1(scale)) : qf;
  StableHLOCompositeOptions qk_opts{
      .name = "odml.runtime_bmm",
      .composite_attributes = RuntimeBmmAttributes(/*is_src=*/false)};
  // avbound: the QK side runs UNBOUNDED (param_full has element 2 == S) so
  // every tail score is deterministically written (zero K rows give zero
  // scores) and the fused scale+mask epilogue covers the full width; only
  // the AV side is runtime-bounded. This is the sound composition when a
  // full-width softmax sits between the two composites - a dst-bounded QK
  // leaves its tail UNWRITTEN (stale memory), which a full-width softmax
  // then reads.
  TfTensor scores = StableHLOComposite(
      qk_opts,
      [](TfTensor l, TfTensor r, TfTensor /*p*/) {
        return BatchMatMul(l, r, /*adj_x=*/false, /*adj_y=*/true);
      },
      lhs, k, mode == RbmmMode::kAvBound ? param_full : param);  // [1,G,gsz,S]
  if (mode == RbmmMode::kMasked || mode == RbmmMode::kAvBound) {
    scores = Mul(scores, Const1(scale));
    scores = Add(scores, mask);
  }
  TfTensor attn = Softmax(scores);
  if (attn_out != nullptr) *attn_out = attn;
  StableHLOCompositeOptions av_opts{
      .name = "odml.runtime_bmm",
      .composite_attributes = RuntimeBmmAttributes(/*is_src=*/true)};
  TfTensor out = StableHLOComposite(
      av_opts,
      [](TfTensor l, TfTensor r, TfTensor /*p*/) {
        return BatchMatMul(l, r, /*adj_x=*/false, /*adj_y=*/true);
      },
      attn, v_t, param);  // [1,G,gsz,D]
  if (mode == RbmmMode::kCorrected) {
    // Tail slots enter the softmax as exp(0 - rowmax) each (bounded kernels
    // zero-fill; zero cache rows give zero scores on the decomposition), so
    // the denominator is inflated by tail_count * p0 where p0 is any tail
    // probability - the last slot is one whenever active < S, and the factor
    // degenerates to 1 at full fill. Rescaling the src-bounded AV output by
    // 1 / (1 - tail_count * p0) is exact renormalization, not an
    // approximation.
    TfTensor p0 = Slice(attn, {0, 0, 0, g.max_seq - 1},
                        {1, g.groups, g.group_sz, 1});
    TfTensor corr =
        Div(Const1(1.0f), Sub(Const1(1.0f), Mul(p0, tail_count)));
    out = Mul(out, corr);
  }
  return out;
}

std::vector<uint8_t> CacheUpdateAttributes(int kv_cache_batch_size,
                                           int cache_size, int head_size) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Int("kv_cache_batch_size", kv_cache_batch_size);
    fbb.Int("cache_size", cache_size);
    fbb.Int("head_size", head_size);
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

// One decode step through the odml.cache_update + odml.runtime_bmm pair -
// the fused write/read pattern the delegate implements (cache_update stores
// K in the QK FC-weights layout and V in the AV layout; runtime_bmm's
// FC-external-weights path engages because the RHS producer is
// add_values_to_cache). The composite carries 7 inputs so the CPU
// decomposition can express the same update as DYNAMIC_UPDATE_SLICE
// (src_k, src_v, control tensor, k cache, v cache, and the two write-index
// vectors); the GPU parser reads only the first three. Control tensor
// element 0 = write offset, element 1 = active tokens (cache_update),
// element 2 = active tokens (runtime_bmm bound).
TfTensor CacheRbmmStep(const Geometry& g, const TfTensor& q,
                       const TfTensor& src_k, const TfTensor& src_v,
                       const TfTensor& mask, const TfTensor& param,
                       const TfTensor& param_full, const TfTensor& cache_k_in,
                       const TfTensor& cache_v_in, const TfTensor& pos_k,
                       const TfTensor& pos_v, bool qk_bound,
                       TfTensor* cache_k_out = nullptr,
                       TfTensor* cache_v_out = nullptr) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(g.head_dim));
  StableHLOCompositeOptions cu_opts{
      .name = "odml.cache_update",
      .composite_attributes = CacheUpdateAttributes(
          g.groups, g.max_seq, g.head_dim)};
  auto caches = StableHLOComposite(
      cu_opts,
      [](TfTensor sk, TfTensor sv, TfTensor /*p*/, TfTensor ck, TfTensor cv,
         TfTensor pk, TfTensor pv) {
        TfTensor k_new = DynamicUpdateSlice(ck, sk, pk);
        // V cache is stored transposed ([1, G, D, S]); write this step's row
        // as a column.
        TfTensor sv_t = Transpose(sv, {0, 1, 3, 2});
        TfTensor v_new = DynamicUpdateSlice(cv, sv_t, pv);
        return std::make_tuple(k_new, v_new);
      },
      src_k, src_v, param, cache_k_in, cache_v_in, pos_k, pos_v);
  TfTensor cache_k_new = std::get<0>(caches);
  TfTensor cache_v_new = std::get<1>(caches);
  cache_k_new.SetName("cache_k_new");
  cache_v_new.SetName("cache_v_new");
  if (cache_k_out != nullptr) *cache_k_out = cache_k_new;
  if (cache_v_out != nullptr) *cache_v_out = cache_v_new;

  TfTensor lhs = Mul(q, Const1(scale));
  StableHLOCompositeOptions qk_opts{
      .name = "odml.runtime_bmm",
      .composite_attributes = RuntimeBmmAttributes(/*is_src=*/false)};
  TfTensor scores = StableHLOComposite(
      qk_opts,
      [](TfTensor l, TfTensor r, TfTensor /*p*/) {
        return BatchMatMul(l, r, /*adj_x=*/false, /*adj_y=*/true);
      },
      lhs, cache_k_new, qk_bound ? param : param_full);
  scores = Add(scores, mask);
  TfTensor attn = Softmax(scores);
  StableHLOCompositeOptions av_opts{
      .name = "odml.runtime_bmm",
      .composite_attributes = RuntimeBmmAttributes(/*is_src=*/true)};
  TfTensor out = StableHLOComposite(
      av_opts,
      [](TfTensor l, TfTensor r, TfTensor /*p*/) {
        return BatchMatMul(l, r, /*adj_x=*/false, /*adj_y=*/true);
      },
      attn, cache_v_new, param);
  return out;  // [1, G, gsz, D]
}

struct HostData {
  std::vector<float> q;         // H*D, block-0 query
  std::vector<float> k;         // G*S*D logical [g][s][d]
  std::vector<float> v;         // G*S*D logical [g][s][d]
};

HostData MakeHostData(const Geometry& g, int seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  HostData d;
  d.q.resize(static_cast<size_t>(g.heads) * g.head_dim);
  for (auto& x : d.q) x = dist(rng);
  const size_t kv = static_cast<size_t>(g.groups) * g.max_seq * g.head_dim;
  d.k.resize(kv);
  d.v.resize(kv);
  for (auto& x : d.k) x = dist(rng);
  for (auto& x : d.v) x = dist(rng);
  return d;
}

// Layout packers. Rows >= active are overwritten with the canary value.
std::vector<float> PackBnsd(const Geometry& g, const std::vector<float>& src,
                            int active, float canary) {
  std::vector<float> out(src);
  for (int gi = 0; gi < g.groups; ++gi)
    for (int s = active; s < g.max_seq; ++s)
      for (int di = 0; di < g.head_dim; ++di)
        out[(static_cast<size_t>(gi) * g.max_seq + s) * g.head_dim + di] =
            canary;
  return out;
}

std::vector<float> PackBsnd(const Geometry& g, const std::vector<float>& src,
                            int active, float canary) {
  std::vector<float> out(src.size());
  for (int s = 0; s < g.max_seq; ++s)
    for (int gi = 0; gi < g.groups; ++gi)
      for (int di = 0; di < g.head_dim; ++di)
        out[(static_cast<size_t>(s) * g.groups + gi) * g.head_dim + di] =
            s < active
                ? src[(static_cast<size_t>(gi) * g.max_seq + s) * g.head_dim +
                      di]
                : canary;
  return out;
}

std::vector<float> PackTransposed(const Geometry& g,
                                  const std::vector<float>& src, int active,
                                  float canary) {
  std::vector<float> out(src.size());
  for (int gi = 0; gi < g.groups; ++gi)
    for (int di = 0; di < g.head_dim; ++di)
      for (int s = 0; s < g.max_seq; ++s)
        out[(static_cast<size_t>(gi) * g.head_dim + di) * g.max_seq + s] =
            s < active
                ? src[(static_cast<size_t>(gi) * g.max_seq + s) * g.head_dim +
                      di]
                : canary;
  return out;
}

std::vector<float> MakeMask(int max_seq, int active) {
  std::vector<float> m(max_seq, kMaskFloor);
  std::fill(m.begin(), m.begin() + active, 0.0f);
  return m;
}

// One cache_rbmm decode step's inputs: the step writes position `pos` and
// attends over [0, active) (active = pos + 1 in a plain decode loop). The
// host cache inputs carry the prefix rows [0, pos) — they feed only the CPU
// decomposition. zero_host_caches hands the measured GPU runner all-zero
// host caches instead: the delegate path never reads them, so agreement
// with the CPU reference then PROVES the delegate-managed cache persisted
// across Run() calls.
enum class HostCacheMode { kReal, kZeros, kSkip };

absl::Status SetCacheStepInputs(LitertDynamicRunner& runner,
                                const Geometry& g, const HostData& data,
                                int pos, int active,
                                HostCacheMode cache_mode) {
  auto st = runner.SetInput("main", "q",
                            Create("q", Type::kFP32,
                                   {1, g.groups, g.group_sz, g.head_dim},
                                   std::vector<float>(data.q)));
  if (!st.ok())
    return absl::InternalError(absl::StrCat("q: ", st.message()));
  std::vector<float> row_k(static_cast<size_t>(g.groups) * g.head_dim);
  std::vector<float> row_v(row_k.size());
  for (int gi = 0; gi < g.groups; ++gi)
    for (int di = 0; di < g.head_dim; ++di) {
      size_t src = (static_cast<size_t>(gi) * g.max_seq + pos) * g.head_dim +
                   di;
      row_k[static_cast<size_t>(gi) * g.head_dim + di] = data.k[src];
      row_v[static_cast<size_t>(gi) * g.head_dim + di] = data.v[src];
    }
  st = runner.SetInput("main", "src_k",
                       Create("src_k", Type::kFP32,
                              {1, g.groups, 1, g.head_dim},
                              std::move(row_k)));
  if (!st.ok()) return absl::InternalError(absl::StrCat("src_k: ", st.message()));
  st = runner.SetInput("main", "src_v",
                       Create("src_v", Type::kFP32,
                              {1, g.groups, 1, g.head_dim},
                              std::move(row_v)));
  if (!st.ok()) return absl::InternalError(absl::StrCat("src_v: ", st.message()));
  const size_t cache_elems =
      static_cast<size_t>(g.groups) * g.max_seq * g.head_dim;
  if (cache_mode != HostCacheMode::kSkip) {
    st = runner.SetInput("main", "cache_k_in",
                         Create("cache_k_in", Type::kFP32,
                                {1, g.groups, g.max_seq, g.head_dim},
                                cache_mode == HostCacheMode::kZeros
                                    ? std::vector<float>(cache_elems, 0.0f)
                                    : PackBnsd(g, data.k, pos, 0.0f)));
    if (!st.ok()) return absl::InternalError(absl::StrCat("cache_k_in: ", st.message()));
    st = runner.SetInput("main", "cache_v_in",
                         Create("cache_v_in", Type::kFP32,
                                {1, g.groups, g.head_dim, g.max_seq},
                                cache_mode == HostCacheMode::kZeros
                                    ? std::vector<float>(cache_elems, 0.0f)
                                    : PackTransposed(g, data.v, pos, 0.0f)));
    if (!st.ok()) return absl::InternalError(absl::StrCat("cache_v_in: ", st.message()));
  }
  std::vector<int32_t> p(7, active);
  p[0] = pos;
  st = runner.SetInput("main", "param",
                       Create("param", Type::kI32, {1, 1, 1, 7},
                              std::move(p)));
  if (!st.ok()) return absl::InternalError(absl::StrCat("param: ", st.message()));
  std::vector<int32_t> pf(7, g.max_seq);
  pf[0] = pos;
  pf[1] = active;
  st = runner.SetInput("main", "param_full",
                       Create("param_full", Type::kI32, {1, 1, 1, 7},
                              std::move(pf)));
  if (!st.ok()) return absl::InternalError(absl::StrCat("param_full: ", st.message()));
  st = runner.SetInput("main", "pos_k",
                       Create("pos_k", Type::kI32, {4},
                              std::vector<int32_t>{0, 0, pos, 0}));
  if (!st.ok()) return absl::InternalError(absl::StrCat("pos_k: ", st.message()));
  st = runner.SetInput("main", "pos_v",
                       Create("pos_v", Type::kI32, {4},
                              std::vector<int32_t>{0, 0, 0, pos}));
  if (!st.ok()) return absl::InternalError(absl::StrCat("pos_v: ", st.message()));
  st = runner.SetInput("main", "mask",
                       Create("mask", Type::kFP32, {1, 1, 1, g.max_seq},
                              MakeMask(g.max_seq, active)));
  if (!st.ok()) return absl::InternalError(absl::StrCat("mask: ", st.message()));
  return absl::OkStatus();
}

absl::Status SetCommonInputs(LitertDynamicRunner& runner,
                             const Geometry& g, const std::string& variant,
                             const HostData& data, int active, float canary,
                             bool param_align4) {
  auto st = absl::OkStatus();
  if (variant == "cache_rbmm") {
    // The step writes position active-1; the cache inputs carry positions
    // [0, active-1) and the composite adds the last row.
    return SetCacheStepInputs(runner, g, data, /*pos=*/active - 1, active,
                              HostCacheMode::kReal);
  }
  if (variant == "raw" || variant == "rbmm") {
    std::vector<int> q_shape = variant == "raw"
                                   ? std::vector<int>{1, g.heads, 1, g.head_dim}
                                   : std::vector<int>{1, g.groups, g.group_sz,
                                                      g.head_dim};
    st = runner.SetInput("main", "q",
                         Create("q", Type::kFP32, q_shape,
                                std::vector<float>(data.q)));
    if (!st.ok()) return st;
    st = runner.SetInput(
        "main", "k_cache",
        Create("k_cache", Type::kFP32, {1, g.groups, g.max_seq, g.head_dim},
               PackBnsd(g, data.k, active, canary)));
    if (!st.ok()) return st;
    if (variant == "raw") {
      st = runner.SetInput(
          "main", "v_cache",
          Create("v_cache", Type::kFP32, {1, g.groups, g.max_seq, g.head_dim},
                 PackBnsd(g, data.v, active, canary)));
    } else {
      st = runner.SetInput(
          "main", "v_cache_t",
          Create("v_cache_t", Type::kFP32,
                 {1, g.groups, g.head_dim, g.max_seq},
                 PackTransposed(g, data.v, active, canary)));
    }
    if (!st.ok()) return st;
  } else {  // sdpa
    st = runner.SetInput("main", "q",
                         Create("q", Type::kFP32, {1, 1, g.heads, g.head_dim},
                                std::vector<float>(data.q)));
    if (!st.ok()) return st;
    st = runner.SetInput(
        "main", "k_cache",
        Create("k_cache", Type::kFP32, {1, g.max_seq, g.groups, g.head_dim},
               PackBsnd(g, data.k, active, canary)));
    if (!st.ok()) return st;
    st = runner.SetInput(
        "main", "v_cache",
        Create("v_cache", Type::kFP32, {1, g.max_seq, g.groups, g.head_dim},
               PackBsnd(g, data.v, active, canary)));
    if (!st.ok()) return st;
  }
  const std::string rbmm_mode = absl::GetFlag(FLAGS_rbmm_mode);
  const bool rbmm_has_mask =
      rbmm_mode == "masked" || rbmm_mode == "avbound";
  if (variant != "rbmm" || rbmm_has_mask) {
    st = runner.SetInput("main", "mask",
                         Create("mask", Type::kFP32, {1, 1, 1, g.max_seq},
                                MakeMask(g.max_seq, active)));
    if (!st.ok()) return st;
  }
  if (variant == "rbmm") {
    int p2 = param_align4 ? (active + 3) / 4 * 4 : active;
    std::vector<int32_t> param(7, p2);
    st = runner.SetInput("main", "param",
                         Create("param", Type::kI32, {1, 1, 1, 7},
                                std::move(param)));
    if (!st.ok()) return st;
    if (rbmm_mode == "corrected") {
      st = runner.SetInput(
          "main", "tail_count",
          Create("tail_count", Type::kFP32, {1},
                 std::vector<float>{static_cast<float>(g.max_seq - active)}));
      if (!st.ok()) return st;
    }
    if (rbmm_mode == "avbound") {
      st = runner.SetInput("main", "param_full",
                           Create("param_full", Type::kI32, {1, 1, 1, 7},
                                  std::vector<int32_t>(7, g.max_seq)));
      if (!st.ok()) return st;
    }
  }
  return st;
}

absl::Status RunBench() {
  const std::string variant = absl::GetFlag(FLAGS_variant);
  if (variant != "raw" && variant != "sdpa" && variant != "rbmm" &&
      variant != "cache_rbmm") {
    return absl::InvalidArgumentError(
        "--variant must be raw|sdpa|rbmm|cache_rbmm");
  }
  const bool mha = absl::GetFlag(FLAGS_mha);
  Geometry g;
  g.heads = absl::GetFlag(FLAGS_heads);
  g.groups = mha ? g.heads : absl::GetFlag(FLAGS_kv_groups);
  g.group_sz = g.heads / g.groups;
  g.head_dim = absl::GetFlag(FLAGS_head_dim);
  g.max_seq = absl::GetFlag(FLAGS_max_seq);
  const int blocks = absl::GetFlag(FLAGS_blocks);

  // ---- Build the graph.
  std::vector<TensorHandle> inputs;
  TfTensor out;
  TfTensor attn0;
  TfTensor cache_k_new_out;
  TfTensor cache_v_new_out;
  TfTensor mask({.name = "mask",
                 .type = Type::kFP32,
                 .shape = {1, 1, 1, g.max_seq}});
  if (variant == "raw") {
    TfTensor q({.name = "q",
                .type = Type::kFP32,
                .shape = {1, g.heads, 1, g.head_dim}});
    TfTensor k({.name = "k_cache",
                .type = Type::kFP32,
                .shape = {1, g.groups, g.max_seq, g.head_dim}});
    TfTensor v({.name = "v_cache",
                .type = Type::kFP32,
                .shape = {1, g.groups, g.max_seq, g.head_dim}});
    TfTensor x = q;
    for (int b = 0; b < blocks; ++b) x = FoldAttention(g, x, k, v, mask);
    out = x;
    inputs = {q, k, v, mask};
  } else if (variant == "sdpa") {
    TfTensor q({.name = "q",
                .type = Type::kFP32,
                .shape = {1, 1, g.heads, g.head_dim}});
    TfTensor k({.name = "k_cache",
                .type = Type::kFP32,
                .shape = {1, g.max_seq, g.groups, g.head_dim}});
    TfTensor v({.name = "v_cache",
                .type = Type::kFP32,
                .shape = {1, g.max_seq, g.groups, g.head_dim}});
    TfTensor x = q;
    for (int b = 0; b < blocks; ++b) x = SdpaBlock(g, x, k, v, mask);
    out = x;
    inputs = {q, k, v, mask};
  } else if (variant == "rbmm") {
    TfTensor q({.name = "q",
                .type = Type::kFP32,
                .shape = {1, g.groups, g.group_sz, g.head_dim}});
    TfTensor k({.name = "k_cache",
                .type = Type::kFP32,
                .shape = {1, g.groups, g.max_seq, g.head_dim}});
    TfTensor v_t({.name = "v_cache_t",
                  .type = Type::kFP32,
                  .shape = {1, g.groups, g.head_dim, g.max_seq}});
    TfTensor param({.name = "param",
                    .type = Type::kI32,
                    .shape = {1, 1, 1, 7}});
    TfTensor tail_count({.name = "tail_count",
                         .type = Type::kFP32,
                         .shape = {1}});
    TfTensor param_full({.name = "param_full",
                         .type = Type::kI32,
                         .shape = {1, 1, 1, 7}});
    const std::string mode_str = absl::GetFlag(FLAGS_rbmm_mode);
    if (mode_str != "masked" && mode_str != "nomask" &&
        mode_str != "corrected" && mode_str != "avbound") {
      return absl::InvalidArgumentError(
          "--rbmm_mode must be masked|nomask|corrected|avbound");
    }
    const RbmmMode mode = mode_str == "masked"      ? RbmmMode::kMasked
                          : mode_str == "nomask"    ? RbmmMode::kNoMask
                          : mode_str == "corrected" ? RbmmMode::kCorrected
                                                    : RbmmMode::kAvBound;
    TfTensor x = q;
    for (int b = 0; b < blocks; ++b)
      x = RuntimeBmmBlock(g, x, k, v_t, mask, param, param_full, tail_count,
                          mode,
                          (b == 0 && absl::GetFlag(FLAGS_debug_softmax))
                              ? &attn0
                              : nullptr);
    out = Reshape(x, {1, g.heads, 1, g.head_dim});
    if (mode == RbmmMode::kMasked) {
      inputs = {q, k, v_t, mask, param};
    } else if (mode == RbmmMode::kNoMask) {
      inputs = {q, k, v_t, param};
    } else if (mode == RbmmMode::kCorrected) {
      inputs = {q, k, v_t, param, tail_count};
    } else {
      inputs = {q, k, v_t, mask, param, param_full};
    }
  }
  if (variant == "cache_rbmm") {
    // Single decode step through cache_update + runtime_bmm; the cache is
    // threaded through the composite, so block-chaining does not apply.
    TfTensor q({.name = "q",
                .type = Type::kFP32,
                .shape = {1, g.groups, g.group_sz, g.head_dim}});
    TfTensor src_k({.name = "src_k",
                    .type = Type::kFP32,
                    .shape = {1, g.groups, 1, g.head_dim}});
    TfTensor src_v({.name = "src_v",
                    .type = Type::kFP32,
                    .shape = {1, g.groups, 1, g.head_dim}});
    TfTensor param({.name = "param",
                    .type = Type::kI32,
                    .shape = {1, 1, 1, 7}});
    TfTensor param_full({.name = "param_full",
                         .type = Type::kI32,
                         .shape = {1, 1, 1, 7}});
    TfTensor cache_k_in({.name = "cache_k_in",
                         .type = Type::kFP32,
                         .shape = {1, g.groups, g.max_seq, g.head_dim}});
    TfTensor cache_v_in({.name = "cache_v_in",
                         .type = Type::kFP32,
                         .shape = {1, g.groups, g.head_dim, g.max_seq}});
    TfTensor pos_k({.name = "pos_k", .type = Type::kI32, .shape = {4}});
    TfTensor pos_v({.name = "pos_v", .type = Type::kI32, .shape = {4}});
    TfTensor step =
        CacheRbmmStep(g, q, src_k, src_v, mask, param, param_full, cache_k_in,
                      cache_v_in, pos_k, pos_v,
                      absl::GetFlag(FLAGS_cache_qk_bound), &cache_k_new_out,
                      &cache_v_new_out);
    out = Reshape(step, {1, g.heads, 1, g.head_dim});
    inputs = {q,          src_k, src_v, mask,  param,
              param_full, cache_k_in,   cache_v_in, pos_k, pos_v};
  }
  out.SetName("out");

  ModelFactory factory;
  std::vector<TensorHandle> outputs = {out};
  if (variant == "cache_rbmm" && absl::GetFlag(FLAGS_feedback_loops)) {
    // Feedback-loop mode: the caches are signature outputs the runner swaps
    // back into the inputs between Run() calls (PR #8796 contract).
    outputs.push_back(cache_k_new_out);
    outputs.push_back(cache_v_new_out);
  }
  const bool debug_softmax =
      absl::GetFlag(FLAGS_debug_softmax) && variant == "rbmm";
  if (debug_softmax) {
    attn0.SetName("attn0");
    outputs.push_back(attn0);
  }
  auto st = factory.AddSignature(inputs, outputs, "main");
  if (!st.ok()) return st;
  const std::string path = absl::GetFlag(FLAGS_tflite_path);
  st = factory.Save(path);
  if (!st.ok()) return st;
  std::cout << "serialized " << variant << " x" << blocks << " blocks ("
            << (mha ? "MHA-shaped" : "GQA") << ", G=" << g.groups
            << ") -> " << path << std::endl;

  // ---- Runners: CPU reference always; GPU if requested.
  auto env = ::litert::Environment::Create({});
  if (!env) return absl::InternalError("Environment::Create failed");

  auto cpu_options = ::litert::Options::Create();
  if (!cpu_options) return absl::InternalError("Options::Create failed");
  cpu_options->SetHardwareAccelerators(::litert::HwAccelerators::kCpu);
  auto cpu_runner_or = LitertDynamicRunner::Create(*env, path, *cpu_options);
  if (!cpu_runner_or.ok()) return cpu_runner_or.status();
  auto cpu_runner = std::move(*cpu_runner_or);

  const bool use_gpu = absl::GetFlag(FLAGS_accelerator) == "gpu";
  const bool feedback = variant == "cache_rbmm" &&
                        absl::GetFlag(FLAGS_multi_steps) > 0 &&
                        absl::GetFlag(FLAGS_feedback_loops);
  std::unique_ptr<LitertDynamicRunner> gpu_runner;
  // CPU feedback validation: a second, loop-registered CPU runner as the
  // measured side; cpu_runner stays the stateless per-step reference.
  std::unique_ptr<LitertDynamicRunner> cpu_looped;
  if (!use_gpu && feedback) {
    auto looped_options = ::litert::Options::Create();
    if (!looped_options) return absl::InternalError("Options::Create failed");
    looped_options->SetHardwareAccelerators(::litert::HwAccelerators::kCpu);
    auto looped_or = LitertDynamicRunner::Create(
        *env, path, *looped_options,
        {{"cache_k_in", "cache_k_new"}, {"cache_v_in", "cache_v_new"}});
    if (!looped_or.ok()) return looped_or.status();
    cpu_looped =
        std::make_unique<LitertDynamicRunner>(std::move(*looped_or));
  }
  if (use_gpu) {
    auto options = ::litert::Options::Create();
    if (!options) return absl::InternalError("Options::Create failed");
    options->SetHardwareAccelerators(::litert::HwAccelerators::kGpu);
    auto gpu_options = options->GetGpuOptions();
    if (gpu_options.HasValue()) {
      if (absl::GetFlag(FLAGS_gpu_precision) == "fp32") {
        gpu_options->SetPrecision(::litert::GpuOptions::Precision::kFp32);
      }
      const std::string storage = absl::GetFlag(FLAGS_gpu_buffer_storage);
      if (storage == "buffer") {
        gpu_options->SetBufferStorageType(
            ::litert::GpuOptions::BufferStorageType::kBuffer);
      } else if (storage == "texture2d") {
        gpu_options->SetBufferStorageType(
            ::litert::GpuOptions::BufferStorageType::kTexture2D);
      }
    }
    auto gpu_runner_or =
        feedback
            ? LitertDynamicRunner::Create(
                  *env, path, *options,
                  {{"cache_k_in", "cache_k_new"}, {"cache_v_in", "cache_v_new"}})
            : LitertDynamicRunner::Create(*env, path, *options);
    if (!gpu_runner_or.ok()) return gpu_runner_or.status();
    gpu_runner = std::make_unique<LitertDynamicRunner>(
        std::move(*gpu_runner_or));
  }
  LitertDynamicRunner& measured =
      use_gpu ? *gpu_runner : (cpu_looped ? *cpu_looped : cpu_runner);

  const HostData data = MakeHostData(g, absl::GetFlag(FLAGS_seed));
  float canary = absl::GetFlag(FLAGS_canary);
  if (variant == "rbmm" && absl::GetFlag(FLAGS_rbmm_mode) == "corrected" &&
      canary != 0.0f) {
    std::cout << "corrected mode requires zero tail cache slots; forcing "
                 "--canary=0"
              << std::endl;
    canary = 0.0f;
  }
  const bool align4 = absl::GetFlag(FLAGS_param_align4);
  const int runs = absl::GetFlag(FLAGS_runs);
  const int warmup = absl::GetFlag(FLAGS_warmup);

  std::vector<std::string> fill_strs =
      absl::StrSplit(absl::GetFlag(FLAGS_fills), ',', absl::SkipEmpty());

  auto read_output = [](LitertDynamicRunner& runner,
                        std::vector<float>& sink) -> absl::Status {
    auto out = runner.GetOutput("main", "out");
    if (!out.ok()) return out.status();
    auto buffer = out->GetBuffer();
    if (!buffer.ok()) return buffer.status();
    auto lock = buffer->Lock();
    const float* data = reinterpret_cast<const float*>(lock.data());
    sink.assign(data, data + lock.size() / sizeof(float));
    return absl::OkStatus();
  };

  const int multi_steps = absl::GetFlag(FLAGS_multi_steps);
  if (variant == "cache_rbmm" && multi_steps > 0) {
    // Chained decode: one runner instance, pos = 0..N-1. Each step's CPU
    // reference is computed statelessly from host-provided prefix rows; the
    // measured GPU runner gets zeroed host caches (see SetCacheStepInputs),
    // so per-step agreement proves cross-Run() cache accumulation.
    if (multi_steps > g.max_seq) {
      return absl::InvalidArgumentError("--multi_steps must be <= max_seq");
    }
    double worst_rel = 0.0;
    int worst_step = -1;
    std::vector<double> ms;
    ms.reserve(multi_steps);
    for (int pos = 0; pos < multi_steps; ++pos) {
      const int active = pos + 1;
      st = SetCacheStepInputs(cpu_runner, g, data, pos, active,
                              HostCacheMode::kReal);
      if (!st.ok()) return st;
      st = cpu_runner.Run("main");
      if (!st.ok()) return st;
      std::vector<float> ref;
      st = read_output(cpu_runner, ref);
      if (!st.ok()) return st;

      if (use_gpu || cpu_looped) {
        const std::string hc = absl::GetFlag(FLAGS_multi_host_cache);
        // Feedback mode: the swap chain owns the cache bindings, so never
        // re-set them. On GPU unwritten rows are never read (masked); the
        // CPU decomposition flows the whole buffer through DUS, so seed
        // zeros once at pos=0 there.
        const HostCacheMode measured_mode =
            feedback ? (cpu_looped && pos == 0 ? HostCacheMode::kZeros
                                               : HostCacheMode::kSkip)
            : hc == "real" ? HostCacheMode::kReal
            : hc == "once" ? (pos == 0 ? HostCacheMode::kZeros
                                       : HostCacheMode::kSkip)
                           : HostCacheMode::kZeros;
        st = SetCacheStepInputs(measured, g, data, pos, active, measured_mode);
        if (!st.ok()) return st;
        auto t0 = std::chrono::steady_clock::now();
        st = measured.Run("main");
        auto t1 = std::chrono::steady_clock::now();
        if (!st.ok()) return st;
        ms.push_back(
            std::chrono::duration<double, std::milli>(t1 - t0).count());
        std::vector<float> got;
        st = read_output(measured, got);
        if (!st.ok()) return st;
        double max_rel = 0.0;
        size_t worst_i = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
          double denom =
              std::max(1e-3, std::fabs(static_cast<double>(ref[i])));
          double rel = std::fabs(got[i] - ref[i]) / denom;
          if (rel > max_rel) {
            max_rel = rel;
            worst_i = i;
          }
        }
        if (max_rel > worst_rel) {
          worst_rel = max_rel;
          worst_step = pos;
        }
        if (max_rel > 1e-3 || pos < 3 || pos == multi_steps - 1 ||
            (pos + 1) % 64 == 0) {
          std::printf(
              "  step pos=%d active=%d max_rel=%.3g%s got[w]=%.6f ref[w]=%.6f "
              "ms=%.3f\n",
              pos, active, max_rel, max_rel > 1e-3 ? " MISMATCH" : "",
              got[worst_i], ref[worst_i], ms.back());
        }
      }
    }
    if (use_gpu || cpu_looped) {
      std::sort(ms.begin(), ms.end());
      const std::string mode_desc =
          absl::GetFlag(FLAGS_feedback_loops)
              ? "FEEDBACK LOOPS #8796"
              : absl::StrCat("host_cache=",
                             absl::GetFlag(FLAGS_multi_host_cache));
      std::printf(
          "multi_steps=%d (%s): worst max_rel=%.3g at "
          "pos=%d -> %s; per-step median_ms=%.3f min=%.3f max=%.3f\n",
          multi_steps, mode_desc.c_str(), worst_rel, worst_step,
          worst_rel <= 1e-3 ? "cache state ACCUMULATES across Run() calls"
                            : "MISMATCH — state did NOT accumulate",
          ms[ms.size() / 2], ms.front(), ms.back());
    } else {
      std::cout << "multi_steps on CPU exercises only the stateless "
                   "decomposition (host caches feed it); nothing to verify"
                << std::endl;
    }
    return absl::OkStatus();
  }

  for (const std::string& fs : fill_strs) {
    const int active = std::stoi(fs);

    // Reference output on CPU.
    st = SetCommonInputs(cpu_runner, g, variant, data, active, canary, align4);
    if (!st.ok()) return st;
    st = cpu_runner.Run("main");
    if (!st.ok()) return st;
    std::vector<float> ref_data;
    st = read_output(cpu_runner, ref_data);
    if (!st.ok()) return st;

    // Measured runner.
    double max_rel = 0.0;
    if (use_gpu) {
      st = SetCommonInputs(measured, g, variant, data, active, canary, align4);
      if (!st.ok()) return st;
      st = measured.Run("main");
      if (!st.ok()) return st;
      std::vector<float> got;
      st = read_output(measured, got);
      if (!st.ok()) return st;
      size_t worst = 0;
      for (size_t i = 0; i < ref_data.size(); ++i) {
        double denom = std::max(1e-3, std::fabs(static_cast<double>(
            ref_data[i])));
        double rel = std::fabs(got[i] - ref_data[i]) / denom;
        if (rel > max_rel) {
          max_rel = rel;
          worst = i;
        }
      }
      if (max_rel > 1e-3) {
        int bad = 0;
        for (size_t i = 0; i < ref_data.size(); ++i) {
          double denom = std::max(1e-3, std::fabs(static_cast<double>(
              ref_data[i])));
          if (std::fabs(got[i] - ref_data[i]) / denom > 1e-3) bad++;
        }
        std::printf(
            "  MISMATCH fill=%d: %d/%zu elems rel>1e-3; worst i=%zu (head=%zu "
            "dim=%zu) got=%.6f ref=%.6f; gpu[0..3]=[%.5f %.5f %.5f %.5f]\n",
            active, bad, ref_data.size(), worst,
            worst / static_cast<size_t>(g.head_dim),
            worst % static_cast<size_t>(g.head_dim), got[worst],
            ref_data[worst], got[0], got[1], got[2], got[3]);
      }
      if (debug_softmax) {
        auto a_or = measured.GetOutput("main", "attn0");
        if (a_or.ok()) {
          auto buffer = a_or->GetBuffer();
          if (buffer.ok()) {
            auto lock = buffer->Lock();
            const float* a = reinterpret_cast<const float*>(lock.data());
            double active_sum = 0.0, tail_sum = 0.0, tail_max = 0.0;
            for (int s = 0; s < g.max_seq; ++s) {
              if (s < active) {
                active_sum += a[s];
              } else {
                tail_sum += a[s];
                tail_max = std::max(tail_max, static_cast<double>(a[s]));
              }
            }
            std::printf(
                "  softmax row0 (measured): active_sum=%.6f tail_sum=%.6f "
                "tail_max=%.6g tail[active]=%.6g tail[last]=%.6g\n",
                active_sum, tail_sum, tail_max,
                static_cast<double>(active < g.max_seq ? a[active] : 0.f),
                static_cast<double>(a[g.max_seq - 1]));
          }
        }
      }
    }

    for (int i = 0; i < warmup; ++i) {
      st = measured.Run("main");
      if (!st.ok()) return st;
    }
    std::vector<double> ms;
    ms.reserve(runs);
    for (int i = 0; i < runs; ++i) {
      auto t0 = std::chrono::steady_clock::now();
      st = measured.Run("main");
      if (!st.ok()) return st;
      auto t1 = std::chrono::steady_clock::now();
      ms.push_back(
          std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    const double median = ms[ms.size() / 2];

    double mean_abs = 0.0;
    for (float x : ref_data) mean_abs += std::fabs(x);
    mean_abs /= ref_data.size();

    std::printf(
        "variant=%s accel=%s%s fill=%d blocks=%d median_ms=%.3f min=%.3f "
        "max=%.3f cpu_ref_mean_abs=%.6f sample=[%.5f %.5f %.5f %.5f]%s\n",
        variant.c_str(), use_gpu ? "gpu" : "cpu",
        mha ? " (mha)" : "", active, blocks, median, ms.front(), ms.back(),
        mean_abs, ref_data[0], ref_data[1], ref_data[2], ref_data[3],
        use_gpu ? absl::StrCat(" gpu_vs_cpu_max_rel=", max_rel).c_str() : "");

    const std::string dump = absl::GetFlag(FLAGS_dump_out);
    if (!dump.empty()) {
      const std::string p = absl::StrCat(dump, ".fill", active, ".bin");
      FILE* f = std::fopen(p.c_str(), "wb");
      if (f) {
        std::fwrite(ref_data.data(), sizeof(float), ref_data.size(), f);
        std::fclose(f);
        std::cout << "wrote CPU-ref output " << p << std::endl;
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  auto st = RunBench();
  if (!st.ok()) {
    std::cerr << "ERROR: " << st.message() << std::endl;
    return 1;
  }
  return 0;
}

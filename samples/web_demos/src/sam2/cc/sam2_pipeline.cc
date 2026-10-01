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

#include "sam2_pipeline.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "chain_graphs.h"
#include "host_plan.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"

namespace sam2_chain {
namespace {

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a) {
  return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
}

#define RETURN_IF_ERROR(expr)              \
  do {                                     \
    if (auto _st = (expr); !_st.ok()) return _st; \
  } while (0)
#define ASSIGN_OR_RETURN_IMPL(tmp, lhs, expr) \
  auto tmp = (expr);                          \
  if (!tmp.ok()) return tmp.status();         \
  lhs = std::move(*tmp)
// Free buffers kept per shape for reuse (3 objects x a few frames in flight).
constexpr size_t kMaxPooledPerShape = 32;

#define CONCAT_(a, b) a##b
#define CONCAT(a, b) CONCAT_(a, b)
#define ASSIGN_OR_RETURN(lhs, expr) \
  ASSIGN_OR_RETURN_IMPL(CONCAT(_so_, __LINE__), lhs, expr)

std::string DescKey(const HardwareBufferDescriptor& d) {
  return absl::StrCat(static_cast<int>(d.buffer_type), ":",
                      static_cast<int>(d.element_type), ":",
                      absl::StrJoin(d.shape, "x"), ":", d.PackedBytes());
}

}  // namespace

absl::Status WriteFloats(LitertBuffer& buffer, const std::vector<float>& data) {
  auto st = buffer.tensor_buffer().Write<float>(
      litert::Span<const float>(data.data(), data.size()));
  if (!st) return absl::InternalError(st.Error().Message());
  return absl::OkStatus();
}

absl::StatusOr<std::vector<float>> ReadFloats(LitertBuffer& buffer) {
  auto size = buffer.tensor_buffer().PackedSize();
  if (!size) return absl::InternalError(size.Error().Message());
  std::vector<float> out(*size / sizeof(float));
  auto st = buffer.tensor_buffer().Read<float>(
      litert::Span<float>(out.data(), out.size()));
  if (!st) return absl::InternalError(st.Error().Message());
  return out;
}

Sam2Pipeline::Sam2Pipeline(std::shared_ptr<litert::Environment> env,
                           std::shared_ptr<litert::CompiledModel> sam2,
                           HostConsts consts, ModelCompiler compile,
                           PipelineOptions options)
    : env_(std::move(env)),
      sam2_(std::move(sam2)),
      consts_(std::move(consts)),
      compile_(std::move(compile)),
      options_(std::move(options)),
      nmm_(options_.nmm) {}

absl::StatusOr<std::unique_ptr<Sam2Pipeline>> Sam2Pipeline::Create(
    std::shared_ptr<litert::Environment> env,
    std::shared_ptr<litert::CompiledModel> sam2_model, HostConsts consts,
    ModelCompiler compile, PipelineOptions options) {
  if (options.nmm != 2 && options.nmm != 7) {
    return absl::InvalidArgumentError("nmm must be 2 or 7");
  }
  std::unique_ptr<Sam2Pipeline> p(new Sam2Pipeline(
      std::move(env), std::move(sam2_model), std::move(consts),
      std::move(compile), std::move(options)));
  RETURN_IF_ERROR(p->Init());
  return p;
}

absl::StatusOr<std::shared_ptr<SignatureStage>> Sam2Pipeline::Sam2Stage(
    const std::string& name, const std::string& signature) {
  return SignatureStage::Create(name, sam2_, signature);
}

absl::StatusOr<std::shared_ptr<SignatureStage>> Sam2Pipeline::FrameStage(
    const std::string& name, const std::string& signature) {
  return SignatureStage::Create(name, frame_model_, signature);
}

absl::Status Sam2Pipeline::Init() {
  ASSIGN_OR_RETURN(encode_, Sam2Stage("encode", "encode"));
  encode_->SetEnvironment(env_);
  ASSIGN_OR_RETURN(auto pixels, encode_->GetInputDescriptor("pixels"));
  size_ = pixels.shape[1];
  hw_ = (size_ / 16) * (size_ / 16);

  // Descriptors of the per-frame buffers the pipeline owns, from track7 (the
  // prompt / track signatures share every step output shape).
  ASSIGN_OR_RETURN(auto t7, Sam2Stage("probe", TrackSignature(7)));
  ASSIGN_OR_RETURN(mem_desc_, t7->GetOutputDescriptor("mem"));
  ASSIGN_OR_RETURN(ptr_desc_, t7->GetOutputDescriptor("ptr"));
  ASSIGN_OR_RETURN(mask_desc_, t7->GetOutputDescriptor("low_mask"));
  ASSIGN_OR_RETURN(score_desc_, t7->GetOutputDescriptor("object_score"));
  ASSIGN_OR_RETURN(auto nomem_desc, t7->GetInputDescriptor("nomem"));
  ASSIGN_OR_RETURN(auto mem_in, t7->GetInputDescriptor(MemInput(0)));
  ASSIGN_OR_RETURN(auto ptr_in, t7->GetInputDescriptor(PtrInput(0)));

  // Constants: nomem flags, finite fillers for unused memory slots.
  ASSIGN_OR_RETURN(one_, AllocateLike(env_, nomem_desc));
  ASSIGN_OR_RETURN(zero_, AllocateLike(env_, nomem_desc));
  RETURN_IF_ERROR(WriteFloats(*one_, {1.f}));
  RETURN_IF_ERROR(WriteFloats(*zero_, {0.f}));
  ASSIGN_OR_RETURN(zero_mem_, AllocateLike(env_, mem_in));
  ASSIGN_OR_RETURN(zero_ptr_, AllocateLike(env_, ptr_in));
  RETURN_IF_ERROR(WriteFloats(
      *zero_mem_, std::vector<float>(static_cast<size_t>(hw_) * kMemCh, 0.f)));
  RETURN_IF_ERROR(WriteFloats(*zero_ptr_, std::vector<float>(kHidden, 0.f)));

  for (auto& e : encoded_) {
    ASSIGN_OR_RETURN(e.pix_raw, AllocateLike(env_, *encode_->GetOutputDescriptor("pix_raw")));
    ASSIGN_OR_RETURN(e.feat_s1, AllocateLike(env_, *encode_->GetOutputDescriptor("feat_s1")));
    ASSIGN_OR_RETURN(e.feat_s0, AllocateLike(env_, *encode_->GetOutputDescriptor("feat_s0")));
  }
  return absl::OkStatus();
}

absl::Status Sam2Pipeline::SetGeometry(int width, int height) {
  if (width <= 0 || height <= 0) {
    return absl::InvalidArgumentError("empty frame geometry");
  }
  geo_ = FrameGeometry::For(width, height);

  // Author the frame model for this geometry with the Tensor API.
  ModelFactory factory;
  RETURN_IF_ERROR(AddFrameSignatures(factory, geo_, size_));
  const std::string path = absl::StrCat(options_.scratch_dir, "/sam2_frame_",
                                        width, "x", height, ".tflite");
  RETURN_IF_ERROR(factory.Save(path));
  ASSIGN_OR_RETURN(frame_model_, compile_(path));

  // Reset state that depends on the geometry.
  step_chains_.clear();
  for (int k = 0; k < kMaxObjects; ++k) ClearObject(k);
  for (auto& e : encoded_) e.t = -1;

  // Encoder chain: preprocess -> encode.
  ASSIGN_OR_RETURN(auto pre, FrameStage("preprocess", "preprocess"));
  ASSIGN_OR_RETURN(auto frame_desc, pre->GetInputDescriptor("frame"));
  ASSIGN_OR_RETURN(frame_, AllocateLike(env_, frame_desc));
  {
    ModelChain::Builder b;
    b.WithEnvironment(env_).AddStage(pre).AddStage(encode_).Connect(
        "preprocess", "pixels", "encode", "pixels");
    ASSIGN_OR_RETURN(auto chain, b.Build());
    encoder_chain_ = std::make_unique<ModelChain>(std::move(chain));
    RETURN_IF_ERROR(encoder_chain_->SetInputBuffer("frame", frame_));
  }

  // Display chain: composite alone.
  ASSIGN_OR_RETURN(display_.composite, FrameStage("composite", "composite"));
  ASSIGN_OR_RETURN(auto out_desc, display_.composite->GetOutputDescriptor("rgb"));
  ASSIGN_OR_RETURN(output_, AllocateLike(env_, out_desc));
  ASSIGN_OR_RETURN(auto fx_desc, display_.composite->GetInputDescriptor("fx"));
  ASSIGN_OR_RETURN(fx_, AllocateLike(env_, fx_desc));
  ASSIGN_OR_RETURN(auto mask_in, display_.composite->GetInputDescriptor(MaskInput(0)));
  ASSIGN_OR_RETURN(empty_mask_, AllocateLike(env_, mask_in));
  RETURN_IF_ERROR(WriteFloats(
      *empty_mask_, std::vector<float>(mask_in.PackedBytes() / 4, -1024.f)));
  SetEffect(Effect::kOverlay, 3.f);
  {
    ModelChain::Builder b;
    b.WithEnvironment(env_).AddStage(display_.composite);
    ASSIGN_OR_RETURN(auto chain, b.Build());
    display_.chain = std::make_unique<ModelChain>(std::move(chain));
  }
  return absl::OkStatus();
}

void Sam2Pipeline::SetEffect(Effect effect, float stroke_px) {
  if (!fx_) return;
  auto fx = EffectParams(effect, stroke_px);
  (void)WriteFloats(*fx_, std::vector<float>(fx.begin(), fx.end()));
}

absl::StatusOr<std::shared_ptr<LitertBuffer>> Sam2Pipeline::pixels() const {
  if (!encoder_chain_) return absl::FailedPreconditionError("no geometry");
  ASSIGN_OR_RETURN(auto pre, encoder_chain_->GetStage("preprocess"));
  auto buf = pre->GetOutputBuffer("pixels");
  if (!buf) return absl::NotFoundError("pixels not allocated");
  return buf;
}

const Sam2Pipeline::EncodedFrame* Sam2Pipeline::Encoded(int t) const {
  for (const auto& e : encoded_) {
    if (e.t == t) return &e;
  }
  return nullptr;
}

absl::Status Sam2Pipeline::Encode(int t) {
  if (!encoder_chain_) return absl::FailedPreconditionError("SetGeometry first");
  if (Encoded(t)) return absl::OkStatus();
  const auto t0 = Clock::now();
  EncodedFrame& e = encoded_[next_encoded_];
  next_encoded_ ^= 1;
  e.t = -1;
  RETURN_IF_ERROR(encode_->SetOutputBuffer("pix_raw", e.pix_raw));
  RETURN_IF_ERROR(encode_->SetOutputBuffer("feat_s1", e.feat_s1));
  RETURN_IF_ERROR(encode_->SetOutputBuffer("feat_s0", e.feat_s0));
  RETURN_IF_ERROR(encoder_chain_->Execute());
  e.t = t;
  times_.encode_ms = Ms(t0);
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<LitertBuffer>> Sam2Pipeline::Alloc(
    const HardwareBufferDescriptor& desc) {
  auto& free = pool_[DescKey(desc)];
  if (!free.empty()) {
    auto b = std::move(free.back());
    free.pop_back();
    return b;
  }
  return AllocateLike(env_, desc);
}

void Sam2Pipeline::Release(std::shared_ptr<LitertBuffer> buffer) {
  if (!buffer) return;
  // Only recycle buffers nobody else holds (a chain may still be bound).
  if (buffer.use_count() > 1) return;
  auto type = buffer->BufferType();
  auto packed = buffer->PackedSize();
  auto tensor_type = buffer->tensor_buffer().TensorType();
  if (!type.ok() || !packed.ok() || !tensor_type) return;
  HardwareBufferDescriptor d;
  d.buffer_type = *type;
  d.element_type = tensor_type->ElementType();
  auto dims = tensor_type->Layout().Dimensions();
  d.shape.assign(dims.begin(), dims.end());
  d.size_bytes = *packed;
  // A small free list per shape covers steady-state reuse (tracking, camera);
  // beyond it the buffer is dropped, which frees it (e.g. a Reset of an
  // object with results for a whole clip).
  auto& free = pool_[DescKey(d)];
  if (free.size() < kMaxPooledPerShape) free.push_back(std::move(buffer));
}

void Sam2Pipeline::ClearObject(int object) {
  ObjectState& o = objects_[object];
  for (auto& [f, b] : o.mem) Release(std::move(b));
  for (auto& [f, b] : o.ptr) Release(std::move(b));
  for (auto& [f, r] : o.results) {
    Release(std::move(r.low_mask));
    Release(std::move(r.object_score));
    Release(std::move(r.iou));
  }
  o = ObjectState{};
}

void Sam2Pipeline::Evict(ObjectState& o, int t) {
  // Memory: the conditioning frame + the last 6 frames (nmm 7), pointers:
  // the last 15 — whatever nmm the user switches to next.
  for (auto it = o.mem.begin(); it != o.mem.end();) {
    if (it->first != o.cond_frame && it->first < t - 6) {
      Release(std::move(it->second));
      it = o.mem.erase(it);
    } else {
      ++it;
    }
  }
  for (auto it = o.ptr.begin(); it != o.ptr.end();) {
    if (it->first != o.cond_frame && it->first < t - (kNumPtrFrames - 1)) {
      Release(std::move(it->second));
      it = o.ptr.erase(it);
    } else {
      ++it;
    }
  }
  if (retention_ >= 0) {
    for (auto it = o.results.begin(); it != o.results.end();) {
      if (it->first < t - retention_) {
        Release(std::move(it->second.low_mask));
        Release(std::move(it->second.object_score));
        Release(std::move(it->second.iou));
        it = o.results.erase(it);
      } else {
        ++it;
      }
    }
  }
}

absl::Status Sam2Pipeline::BindEncoded(SignatureStage& stage, int t) {
  const EncodedFrame* e = Encoded(t);
  if (!e) {
    return absl::FailedPreconditionError(
        absl::StrCat("frame ", t, " is not encoded"));
  }
  RETURN_IF_ERROR(stage.SetInputBuffer("pix_raw", e->pix_raw));
  RETURN_IF_ERROR(stage.SetInputBuffer("feat_s1", e->feat_s1));
  return stage.SetInputBuffer("feat_s0", e->feat_s0);
}

// Fresh per-frame outputs: memory + pointer join the object's bank, the
// mask / scores are the frame's result (and composite's input).
absl::Status Sam2Pipeline::BindOutputs(SignatureStage& stage, int object,
                                       int t) {
  ObjectState& o = objects_[object];
  auto set = [&](std::map<int, std::shared_ptr<LitertBuffer>>& bank,
                 const HardwareBufferDescriptor& desc,
                 const char* port) -> absl::Status {
    auto old = bank.find(t);
    if (old != bank.end()) {
      Release(std::move(old->second));
      bank.erase(old);
    }
    ASSIGN_OR_RETURN(auto b, Alloc(desc));
    bank[t] = b;
    return stage.SetOutputBuffer(port, b);
  };
  RETURN_IF_ERROR(set(o.mem, mem_desc_, "mem"));
  RETURN_IF_ERROR(set(o.ptr, ptr_desc_, "ptr"));
  if (auto old = o.results.find(t); old != o.results.end()) {
    Release(std::move(old->second.low_mask));
    Release(std::move(old->second.object_score));
    Release(std::move(old->second.iou));
    o.results.erase(old);
  }
  FrameResult r;
  ASSIGN_OR_RETURN(r.low_mask, Alloc(mask_desc_));
  ASSIGN_OR_RETURN(r.object_score, Alloc(score_desc_));
  ASSIGN_OR_RETURN(r.iou, Alloc(score_desc_));
  RETURN_IF_ERROR(stage.SetOutputBuffer("low_mask", r.low_mask));
  RETURN_IF_ERROR(stage.SetOutputBuffer("object_score", r.object_score));
  RETURN_IF_ERROR(stage.SetOutputBuffer("iou", r.iou));
  o.results[t] = std::move(r);
  return absl::OkStatus();
}

absl::Status Sam2Pipeline::BindComposite(SignatureStage& composite, int t,
                                         const std::vector<int>& stepped) {
  RETURN_IF_ERROR(composite.SetInputBuffer("frame", frame_));
  RETURN_IF_ERROR(composite.SetInputBuffer("fx", fx_));
  RETURN_IF_ERROR(composite.SetOutputBuffer("rgb", output_));
  for (int k = 0; k < kMaxObjects; ++k) {
    // Objects stepped this run: their fresh low_mask (the chain edge).
    // Others: the stored result of this frame, or nothing.
    const FrameResult* r = Result(k, t);
    const bool stepped_k =
        std::find(stepped.begin(), stepped.end(), k) != stepped.end();
    (void)stepped_k;
    RETURN_IF_ERROR(composite.SetInputBuffer(
        MaskInput(k), r ? r->low_mask : empty_mask_));
  }
  return absl::OkStatus();
}

absl::StatusOr<Sam2Pipeline::Chain*> Sam2Pipeline::StepChain(
    const std::array<std::string, kMaxObjects>& steps) {
  const std::string key = absl::StrJoin(steps, ",");
  auto it = step_chains_.find(key);
  if (it != step_chains_.end()) return it->second.get();

  auto c = std::make_unique<Chain>();
  ModelChain::Builder b;
  b.WithEnvironment(env_);
  ASSIGN_OR_RETURN(c->composite, FrameStage("composite", "composite"));
  c->steps.resize(kMaxObjects);
  for (int k = 0; k < kMaxObjects; ++k) {
    if (steps[k].empty()) continue;
    const std::string name = absl::StrCat("obj", k, "_", steps[k]);
    ASSIGN_OR_RETURN(c->steps[k], Sam2Stage(name, steps[k]));
    b.AddStage(c->steps[k]);
    b.Connect(name, "low_mask", "composite", MaskInput(k));
  }
  b.AddStage(c->composite);
  ASSIGN_OR_RETURN(auto chain, b.Build());
  c->chain = std::make_unique<ModelChain>(std::move(chain));

  // Per-stage host-table inputs, written before each run.
  for (auto& s : c->steps) {
    if (!s) continue;
    for (const char* port : {"sparse", "slot_tpe", "ptr_pos", "key_mask"}) {
      auto desc = s->GetInputDescriptor(port);
      if (!desc.ok()) continue;  // prompt{k} has sparse, track{n} the rest
      ASSIGN_OR_RETURN(auto buf, AllocateLike(env_, *desc));
      RETURN_IF_ERROR(s->SetInputBuffer(port, buf));
    }
  }
  Chain* raw = c.get();
  step_chains_[key] = std::move(c);
  return raw;
}

absl::Status Sam2Pipeline::Prompt(int object, int t,
                                  const std::vector<Click>& clicks) {
  if (object < 0 || object >= kMaxObjects) {
    return absl::InvalidArgumentError("object index out of range");
  }
  if (clicks.empty() || clicks.size() > kMaxClicks) {
    return absl::InvalidArgumentError(
        absl::StrCat("1..", kMaxClicks, " clicks per prompt"));
  }
  const auto t0 = Clock::now();
  std::array<std::string, kMaxObjects> steps;
  steps[object] = PromptSignature(static_cast<int>(clicks.size()));
  ASSIGN_OR_RETURN(Chain * c, StepChain(steps));
  SignatureStage& s = *c->steps[object];

  ClearObject(object);  // this frame becomes the object's only prompt
  objects_[object].cond_frame = t;
  RETURN_IF_ERROR(BindEncoded(s, t));
  RETURN_IF_ERROR(s.SetInputBuffer("nomem", one_));
  RETURN_IF_ERROR(WriteFloats(*s.GetInputBuffer("sparse"),
                              PointsSparse(consts_, clicks, size_)));
  RETURN_IF_ERROR(BindOutputs(s, object, t));
  RETURN_IF_ERROR(BindComposite(*c->composite, t, {object}));
  RETURN_IF_ERROR(c->chain->Execute());
  times_.step_ms = Ms(t0);
  return absl::OkStatus();
}

std::vector<int> Sam2Pipeline::TrackedObjects(int t) const {
  std::vector<int> out;
  for (int k = 0; k < kMaxObjects; ++k) {
    if (objects_[k].cond_frame >= 0 && objects_[k].cond_frame < t) {
      out.push_back(k);
    }
  }
  return out;
}

absl::Status Sam2Pipeline::Track(int t) {
  const std::vector<int> objs = TrackedObjects(t);
  if (objs.empty()) return Composite(t);
  const auto t0 = Clock::now();
  std::array<std::string, kMaxObjects> steps;
  for (int k : objs) steps[k] = TrackSignature(nmm_);
  ASSIGN_OR_RETURN(Chain * c, StepChain(steps));

  for (int k : objs) {
    ObjectState& o = objects_[k];
    SignatureStage& s = *c->steps[k];
    const MemPlan plan = PlanMemory(
        consts_, o.cond_frame, t, nmm_, hw_,
        [&](int f) { return o.mem.count(f) > 0; },
        [&](int f) { return o.ptr.count(f) > 0; });
    RETURN_IF_ERROR(BindEncoded(s, t));
    RETURN_IF_ERROR(s.SetInputBuffer("nomem", zero_));
    for (int i = 0; i < nmm_; ++i) {
      RETURN_IF_ERROR(s.SetInputBuffer(
          MemInput(i), i < static_cast<int>(plan.slot_frames.size())
                           ? o.mem.at(plan.slot_frames[i])
                           : zero_mem_));
    }
    for (int i = 0; i < kNumPtrFrames; ++i) {
      RETURN_IF_ERROR(s.SetInputBuffer(
          PtrInput(i), i < static_cast<int>(plan.ptr_frames.size())
                           ? o.ptr.at(plan.ptr_frames[i])
                           : zero_ptr_));
    }
    RETURN_IF_ERROR(WriteFloats(*s.GetInputBuffer("slot_tpe"), plan.slot_tpe));
    RETURN_IF_ERROR(WriteFloats(*s.GetInputBuffer("ptr_pos"), plan.ptr_pos));
    RETURN_IF_ERROR(WriteFloats(*s.GetInputBuffer("key_mask"), plan.key_mask));
    RETURN_IF_ERROR(BindOutputs(s, k, t));
  }
  RETURN_IF_ERROR(BindComposite(*c->composite, t, objs));
  RETURN_IF_ERROR(c->chain->Execute());
  for (int k : objs) Evict(objects_[k], t);
  times_.step_ms = Ms(t0);
  return absl::OkStatus();
}

absl::Status Sam2Pipeline::Composite(int t) {
  if (!display_.chain) return absl::FailedPreconditionError("SetGeometry first");
  const auto t0 = Clock::now();
  RETURN_IF_ERROR(BindComposite(*display_.composite, t, {}));
  RETURN_IF_ERROR(display_.chain->Execute());
  times_.composite_ms = Ms(t0);
  return absl::OkStatus();
}

const FrameResult* Sam2Pipeline::Result(int object, int t) const {
  const auto& r = objects_[object].results;
  auto it = r.find(t);
  return it == r.end() ? nullptr : &it->second;
}

}  // namespace sam2_chain

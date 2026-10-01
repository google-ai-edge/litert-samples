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

// JavaScript API of the wasm build: the same C++ Sam2Pipeline (ModelChains of
// Tensor API models) that sam2_chain_main runs natively, over the
// LiteRT.js-backed runtime (litert_js_runtime.cc), so every stage runs on
// WebGPU and every tensor lives in a WebGPU buffer.
//
// Calls that run models or read buffers back suspend on JS promises (JSPI)
// and return promises in JavaScript.

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "chain_graphs.h"
#include "host_plan.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_weights.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video/sam2v_weights.h"
#include "sam2_pipeline.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"

int LiteRtJsBufferId(LiteRtTensorBuffer b);
extern "C" int lrtjs_read_many(const int* ids, const int* bytes, int count,
                               void* dst);

namespace sam2_chain {
namespace {

using emscripten::val;
namespace sam2 = ::litert::tensor::examples::sam2;

int BufferId(const std::shared_ptr<LitertBuffer>& b) {
  return b ? LiteRtJsBufferId(b->tensor_buffer().Get()) : 0;
}

val Floats(const std::vector<float>& v) {
  val out = val::global("Float32Array").new_(v.size());
  out.call<void>("set", val(emscripten::typed_memory_view(v.size(), v.data())));
  return out;
}

}  // namespace

class Sam2Chain {
 public:
  // Authors the SAM 2 model (encode, prompt1..8, track2/7) from the weights at
  // `weights_path` with the Tensor API and serializes it to `tflite_path`
  // (both in the in-memory filesystem). Returns false on error (lastError()).
  bool BuildSam2Model(std::string weights_path, int image_size,
                      std::string host_consts_path, std::string tflite_path) {
    auto consts = HostConsts::Load(host_consts_path);
    if (!consts.ok()) return Fail(consts.status());
    s2v::Sam2VideoConfig config;
    config.image.image_size = image_size;
    std::vector<sam2::WeightSpec> specs = sam2::GetWeightSpecs(config.image);
    std::vector<sam2::WeightSpec> vspecs = s2v::GetVideoWeightSpecs(config);
    specs.insert(specs.end(), vspecs.begin(), vspecs.end());
    WeightMap weights;
    if (auto st = sam2::LoadWeightSpecs(weights_path, specs, weights);
        !st.ok()) {
      return Fail(st);
    }
    ModelFactory factory;
    if (auto st = AddSam2Signatures(factory, config, weights,
                                    consts->track_sparse);
        !st.ok()) {
      return Fail(st);
    }
    if (auto st = factory.Save(tflite_path); !st.ok()) return Fail(st);
    return true;
  }

  // Compiles the SAM 2 model and creates the pipeline. async.
  bool Init(std::string sam2_tflite_path, std::string host_consts_path,
            int nmm) {
    auto consts = HostConsts::Load(host_consts_path);
    if (!consts.ok()) return Fail(consts.status());
    auto env = litert::Environment::Create({});
    if (!env) return Fail(absl::InternalError("Environment::Create failed"));
    env_ = std::make_shared<litert::Environment>(std::move(*env));
    auto compile = [env = env_](const std::string& path)
        -> absl::StatusOr<std::shared_ptr<litert::CompiledModel>> {
      auto options = litert::Options::Create();
      if (!options) return absl::InternalError("Options::Create failed");
      options->SetHardwareAccelerators(litert::HwAccelerators::kGpu);
      auto model = litert::CompiledModel::Create(*env, path, *options);
      if (!model) {
        return absl::InternalError(
            absl::StrCat("compile ", path, ": ", model.Error().Message()));
      }
      return std::make_shared<litert::CompiledModel>(std::move(*model));
    };
    auto sam2_model = compile(sam2_tflite_path);
    if (!sam2_model.ok()) return Fail(sam2_model.status());
    PipelineOptions options;
    options.nmm = nmm;
    options.scratch_dir = "/tmp";
    auto p = Sam2Pipeline::Create(env_, *sam2_model, *std::move(consts),
                                  compile, options);
    if (!p.ok()) return Fail(p.status());
    pipeline_ = std::move(*p);
    return true;
  }

  int ImageSize() const { return pipeline_ ? pipeline_->image_size() : 0; }
  void SetMemorySize(int nmm) { pipeline_->SetMemorySize(nmm); }
  void SetResultRetention(int frames) { pipeline_->SetResultRetention(frames); }
  void SetEffect(int effect, float stroke) {
    pipeline_->SetEffect(static_cast<Effect>(effect), stroke);
  }
  void ClearObject(int object) { pipeline_->ClearObject(object); }
  int CondFrame(int object) const { return pipeline_->cond_frame(object); }

  // Authors + compiles the frame model (preprocess + composite). async.
  bool SetGeometry(int width, int height) {
    return Check(pipeline_->SetGeometry(width, height));
  }
  int FrameStride() const { return pipeline_->geometry().stride; }
  int FrameBuffer() const { return BufferId(pipeline_->frame()); }
  int OutputBuffer() const { return BufferId(pipeline_->output()); }

  bool Encode(int t) { return Check(pipeline_->Encode(t)); }
  // clicks: [{x, y, label}] in model space (S x S).
  bool Prompt(int object, int t, val clicks) {
    std::vector<Click> cs;
    const int n = clicks["length"].as<int>();
    for (int i = 0; i < n; ++i) {
      val c = clicks[i];
      cs.push_back({c["x"].as<float>(), c["y"].as<float>(), c["label"].as<int>()});
    }
    return Check(pipeline_->Prompt(object, t, cs));
  }
  bool Track(int t) { return Check(pipeline_->Track(t)); }
  bool Composite(int t) { return Check(pipeline_->Composite(t)); }

  val TrackedObjects(int t) const {
    val out = val::array();
    for (int k : pipeline_->TrackedObjects(t)) out.call<void>("push", k);
    return out;
  }
  bool HasResult(int object, int t) const {
    return pipeline_->Result(object, t) != nullptr;
  }

  // Readbacks (async). null when absent / on error.
  val ReadMask(int object, int t) {
    const FrameResult* r = pipeline_->Result(object, t);
    return r ? Read(*r->low_mask) : val::null();
  }
  val ReadMasks(val slots, int t) {
    const int n = slots["length"].as<int>();
    std::vector<int> valid_i;
    std::vector<int> ids;
    std::vector<int> bytes;
    std::vector<size_t> counts;
    size_t total_floats = 0;
    for (int i = 0; i < n; ++i) {
      const int slot = slots[i].as<int>();
      const FrameResult* r = pipeline_->Result(slot, t);
      if (!r || !r->low_mask) continue;
      auto sz = r->low_mask->PackedSize();
      const int id = BufferId(r->low_mask);
      if (!sz.ok() || id <= 0) continue;
      valid_i.push_back(i);
      ids.push_back(id);
      bytes.push_back(static_cast<int>(*sz));
      counts.push_back(*sz / sizeof(float));
      total_floats += *sz / sizeof(float);
    }
    val res = val::array();
    for (int i = 0; i < n; ++i) res.call<void>("push", val::null());
    if (ids.empty()) return res;
    std::vector<float> buf(total_floats);
    if (!lrtjs_read_many(ids.data(), bytes.data(),
                         static_cast<int>(ids.size()), buf.data())) {
      return res;
    }
    size_t offset = 0;
    for (size_t j = 0; j < valid_i.size(); ++j) {
      val arr = val::global("Float32Array").new_(counts[j]);
      arr.call<void>("set", val(emscripten::typed_memory_view(
                                counts[j], buf.data() + offset)));
      res.set(valid_i[j], arr);
      offset += counts[j];
    }
    return res;
  }
  val ReadScores(int object, int t) {  // [object_score, iou]
    const FrameResult* r = pipeline_->Result(object, t);
    if (!r) return val::null();
    const int ids[2] = {BufferId(r->object_score), BufferId(r->iou)};
    const int bytes[2] = {sizeof(float), sizeof(float)};
    if (ids[0] > 0 && ids[1] > 0) {
      float out[2] = {0.f, 0.f};
      if (!lrtjs_read_many(ids, bytes, 2, out)) return val::null();
      return Floats({out[0], out[1]});
    }
    auto s = ReadFloats(*r->object_score);
    auto i = ReadFloats(*r->iou);
    if (!s.ok() || !i.ok()) return val::null();
    return Floats({(*s)[0], (*i)[0]});
  }
  val ReadPixels() {
    auto px = pipeline_->pixels();
    return px.ok() ? Read(**px) : val::null();
  }
  val ReadOutput() { return Read(*pipeline_->output()); }

  val Times() const {
    val o = val::object();
    const StageTimes& t = pipeline_->last_times();
    o.set("encode", t.encode_ms);
    o.set("step", t.step_ms);
    o.set("composite", t.composite_ms);
    return o;
  }
  std::string LastError() const { return error_; }

 private:
  bool Fail(const absl::Status& st) {
    error_ = std::string(st.message());
    return false;
  }
  bool Check(const absl::Status& st) { return st.ok() ? true : Fail(st); }
  val Read(LitertBuffer& b) {
    auto v = ReadFloats(b);
    if (!v.ok()) {
      Fail(v.status());
      return val::null();
    }
    return Floats(*v);
  }

  std::shared_ptr<litert::Environment> env_;
  std::unique_ptr<Sam2Pipeline> pipeline_;
  std::string error_;
};

}  // namespace sam2_chain

EMSCRIPTEN_BINDINGS(sam2_chain) {
  using emscripten::async;
  using sam2_chain::Sam2Chain;
  emscripten::class_<Sam2Chain>("Sam2Chain")
      .constructor<>()
      .function("buildSam2Model", &Sam2Chain::BuildSam2Model)
      .function("init", &Sam2Chain::Init, async())
      .function("imageSize", &Sam2Chain::ImageSize)
      .function("setMemorySize", &Sam2Chain::SetMemorySize)
      .function("setResultRetention", &Sam2Chain::SetResultRetention)
      .function("setEffect", &Sam2Chain::SetEffect)
      .function("clearObject", &Sam2Chain::ClearObject)
      .function("condFrame", &Sam2Chain::CondFrame)
      .function("setGeometry", &Sam2Chain::SetGeometry, async())
      .function("frameStride", &Sam2Chain::FrameStride)
      .function("frameBuffer", &Sam2Chain::FrameBuffer)
      .function("outputBuffer", &Sam2Chain::OutputBuffer)
      .function("encode", &Sam2Chain::Encode, async())
      .function("prompt", &Sam2Chain::Prompt, async())
      .function("track", &Sam2Chain::Track, async())
      .function("composite", &Sam2Chain::Composite, async())
      .function("trackedObjects", &Sam2Chain::TrackedObjects)
      .function("hasResult", &Sam2Chain::HasResult)
      .function("readMask", &Sam2Chain::ReadMask, async())
      .function("readMasks", &Sam2Chain::ReadMasks, async())
      .function("readScores", &Sam2Chain::ReadScores, async())
      .function("readPixels", &Sam2Chain::ReadPixels, async())
      .function("readOutput", &Sam2Chain::ReadOutput, async())
      .function("times", &Sam2Chain::Times)
      .function("lastError", &Sam2Chain::LastError);
  // Object slots compiled into the composite graph (the UI caps objects here).
  emscripten::function("maxObjects", +[] { return sam2_chain::kMaxObjects; });
}

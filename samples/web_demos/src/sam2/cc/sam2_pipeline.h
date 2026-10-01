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

// SAM 2 video segmentation as LiteRT ModelChains over Tensor API models.
//
//   encoder chain   preprocess ──pixels──▶ encode          (per frame)
//   step chains     prompt{k} ─┐                           (click frame)
//                   track{n} ──┼─low_mask──▶ composite     (tracked frame,
//                   track{n} ──┘                            one per object)
//   display chain   composite                              (playback / fx)
//
// All stages share two compiled models: the SAM 2 model and a per-geometry
// frame model (preprocess + composite) that SetGeometry authors with the
// Tensor API and compiles on the fly. Data stays in LitertBuffers end to end:
// the encoder outputs are cached for two frames and bound into the step
// stages; each object's memory bank is the set of mem / ptr buffers earlier
// steps wrote, bound slot by slot into track{n} (no copies); composite reads
// the RGBA frame the preprocess read and writes the display image.
//
// The same code runs natively (LiteRT CompiledModel on CPU / Metal / ...) and
// in the browser, compiled to wasm against a LiteRT runtime that executes on
// WebGPU through LiteRT.js (see wasm/).

#ifndef SAM2_WEB_TENSORAPI_CC_SAM2_PIPELINE_H_
#define SAM2_WEB_TENSORAPI_CC_SAM2_PIPELINE_H_

#include <array>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "chain_graphs.h"
#include "host_plan.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "signature_stage.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/runners/model_chain.h"

namespace sam2_chain {

using ::litert::tensor::ModelChain;

// Compiles a serialized .tflite (written to `path`) into a shared model.
using ModelCompiler =
    std::function<absl::StatusOr<std::shared_ptr<litert::CompiledModel>>(
        const std::string& path)>;

struct PipelineOptions {
  int nmm = 2;                  // memory slots: 2 or 7
  std::string scratch_dir = "/tmp";  // where the frame model is serialized
};

// One object's outputs on one frame (buffers stay on the accelerator).
struct FrameResult {
  std::shared_ptr<LitertBuffer> low_mask;      // [1, S/4, S/4] logits
  std::shared_ptr<LitertBuffer> object_score;  // [1, 1]
  std::shared_ptr<LitertBuffer> iou;           // [1, 1]
};

struct StageTimes {
  double encode_ms = 0, step_ms = 0, composite_ms = 0;
};

class Sam2Pipeline {
 public:
  static absl::StatusOr<std::unique_ptr<Sam2Pipeline>> Create(
      std::shared_ptr<litert::Environment> env,
      std::shared_ptr<litert::CompiledModel> sam2_model, HostConsts consts,
      ModelCompiler compile, PipelineOptions options = {});

  int image_size() const { return size_; }
  int nmm() const { return nmm_; }
  void SetMemorySize(int nmm) { nmm_ = nmm; }

  // Authors + compiles the frame model for a width x height video and
  // rebuilds the chains. Clears all objects.
  absl::Status SetGeometry(int width, int height);
  const FrameGeometry& geometry() const { return geo_; }

  // RGBA frame [1, H, stride, 4] in [0,1], read by preprocess and composite.
  // Fill it before Encode / Track / Composite of that frame.
  std::shared_ptr<LitertBuffer> frame() const { return frame_; }
  // Display image [1, H, W, 3] written by composite.
  std::shared_ptr<LitertBuffer> output() const { return output_; }
  // The last preprocess output [1, S, S, 3] (verification).
  absl::StatusOr<std::shared_ptr<LitertBuffer>> pixels() const;

  // preprocess + encode for frame t (no-op when t is among the 2 cached).
  absl::Status Encode(int t);
  // Conditions `object` on frame t with 1..kMaxClicks clicks (model space),
  // replacing its memory, then composites frame t. Needs Encode(t).
  absl::Status Prompt(int object, int t, const std::vector<Click>& clicks);
  // One track{nmm} step for every object prompted before t (one chain run),
  // then composite. Needs Encode(t).
  absl::Status Track(int t);
  // Composite frame t from the stored results (playback, effect change).
  absl::Status Composite(int t);

  void SetEffect(Effect effect, float stroke_px);
  void ClearObject(int object);
  // Keep per-frame results for at most `frames` frames behind the newest
  // (live camera); < 0 keeps everything (file tracking, scrubbing).
  void SetResultRetention(int frames) { retention_ = frames; }

  int cond_frame(int object) const { return objects_[object].cond_frame; }
  // Objects Track(t) steps.
  std::vector<int> TrackedObjects(int t) const;
  const FrameResult* Result(int object, int t) const;
  const StageTimes& last_times() const { return times_; }

 private:
  struct EncodedFrame {
    int t = -1;
    std::shared_ptr<LitertBuffer> pix_raw, feat_s1, feat_s0;
  };
  struct ObjectState {
    int cond_frame = -1;
    std::map<int, std::shared_ptr<LitertBuffer>> mem, ptr;
    std::map<int, FrameResult> results;
  };
  struct Chain {
    std::unique_ptr<ModelChain> chain;
    std::shared_ptr<SignatureStage> composite;
    std::vector<std::shared_ptr<SignatureStage>> steps;  // per object slot
  };

  Sam2Pipeline(std::shared_ptr<litert::Environment> env,
               std::shared_ptr<litert::CompiledModel> sam2, HostConsts consts,
               ModelCompiler compile, PipelineOptions options);
  absl::Status Init();

  absl::StatusOr<std::shared_ptr<SignatureStage>> Sam2Stage(
      const std::string& name, const std::string& signature);
  absl::StatusOr<std::shared_ptr<SignatureStage>> FrameStage(
      const std::string& name, const std::string& signature);
  // Chain of `steps` (signature per object slot, "" = none) -> composite.
  absl::StatusOr<Chain*> StepChain(
      const std::array<std::string, kMaxObjects>& steps);
  absl::Status BindComposite(SignatureStage& composite, int t,
                             const std::vector<int>& stepped);
  absl::Status BindEncoded(SignatureStage& stage, int t);
  absl::Status BindOutputs(SignatureStage& stage, int object, int t);
  absl::StatusOr<std::shared_ptr<LitertBuffer>> Alloc(
      const HardwareBufferDescriptor& desc);
  void Release(std::shared_ptr<LitertBuffer> buffer);
  void Evict(ObjectState& o, int t);
  const EncodedFrame* Encoded(int t) const;

  std::shared_ptr<litert::Environment> env_;
  std::shared_ptr<litert::CompiledModel> sam2_;
  std::shared_ptr<litert::CompiledModel> frame_model_;
  HostConsts consts_;
  ModelCompiler compile_;
  PipelineOptions options_;
  int size_ = 0, hw_ = 0, nmm_ = 2, retention_ = -1;
  FrameGeometry geo_;

  std::shared_ptr<SignatureStage> encode_;
  std::unique_ptr<ModelChain> encoder_chain_;
  Chain display_;
  absl::flat_hash_map<std::string, std::unique_ptr<Chain>> step_chains_;
  std::array<EncodedFrame, 2> encoded_;
  int next_encoded_ = 0;
  std::array<ObjectState, kMaxObjects> objects_;

  std::shared_ptr<LitertBuffer> frame_, output_, fx_;
  std::shared_ptr<LitertBuffer> one_, zero_, zero_mem_, zero_ptr_, empty_mask_;
  // Port descriptors of the per-frame outputs we allocate ourselves.
  HardwareBufferDescriptor mem_desc_, ptr_desc_, mask_desc_, score_desc_;
  std::map<std::string, std::vector<std::shared_ptr<LitertBuffer>>> pool_;
  StageTimes times_;
};

// Buffer I/O helpers (host <-> accelerator).
absl::Status WriteFloats(LitertBuffer& buffer, const std::vector<float>& data);
absl::StatusOr<std::vector<float>> ReadFloats(LitertBuffer& buffer);

}  // namespace sam2_chain

#endif  // SAM2_WEB_TENSORAPI_CC_SAM2_PIPELINE_H_

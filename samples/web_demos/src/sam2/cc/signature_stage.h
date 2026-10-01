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

// A ModelChain stage that runs ONE signature of a shared, already-compiled
// multi-signature model.
//
// CompiledModelStage owns its CompiledModel, so N stages over one
// multi-signature .tflite would compile (and upload the weights of) the model
// N times. The SAM 2 pipeline has up to ~20 stages over one model (encode,
// prompt1..8, track2/7 per object), so this stage shares a single
// CompiledModel and differs from CompiledModelStage only in ownership;
// descriptor discovery, buffer binding and Run are the same.

#ifndef SAM2_WEB_TENSORAPI_CC_SIGNATURE_STAGE_H_
#define SAM2_WEB_TENSORAPI_CC_SIGNATURE_STAGE_H_

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "tensor/runners/litert/litert_buffer.h"
#include "tensor/runners/model_chain.h"

namespace sam2_chain {

using ::litert::tensor::HardwareBufferDescriptor;
using ::litert::tensor::LitertBuffer;
using ::litert::tensor::ModelStage;

class SignatureStage : public ModelStage {
 public:
  static absl::StatusOr<std::shared_ptr<SignatureStage>> Create(
      std::string name, std::shared_ptr<litert::CompiledModel> model,
      absl::string_view signature_key);

  absl::string_view Name() const override { return name_; }
  const std::string& signature_key() const { return signature_key_; }
  std::vector<std::string> InputNames() const override { return input_names_; }
  std::vector<std::string> OutputNames() const override {
    return output_names_;
  }
  absl::StatusOr<HardwareBufferDescriptor> GetInputDescriptor(
      absl::string_view name) const override;
  absl::StatusOr<HardwareBufferDescriptor> GetOutputDescriptor(
      absl::string_view name) const override;
  absl::Status SetInputBuffer(absl::string_view name,
                              std::shared_ptr<LitertBuffer> buffer) override;
  absl::Status SetOutputBuffer(absl::string_view name,
                               std::shared_ptr<LitertBuffer> buffer) override;
  std::shared_ptr<LitertBuffer> GetInputBuffer(
      absl::string_view name) const override;
  std::shared_ptr<LitertBuffer> GetOutputBuffer(
      absl::string_view name) const override;
  void SetEnvironment(std::shared_ptr<litert::Environment> env) override {
    env_ = std::move(env);
  }
  absl::Status Run() override;

 private:
  SignatureStage(std::string name, std::shared_ptr<litert::CompiledModel> model,
                 std::string signature_key, size_t signature_index)
      : name_(std::move(name)),
        model_(std::move(model)),
        signature_key_(std::move(signature_key)),
        signature_index_(signature_index) {}
  absl::Status DiscoverDescriptors();

  std::shared_ptr<litert::Environment> env_;
  std::string name_;
  std::shared_ptr<litert::CompiledModel> model_;
  std::string signature_key_;
  size_t signature_index_;
  std::vector<std::string> input_names_, output_names_;
  absl::flat_hash_map<std::string, HardwareBufferDescriptor> input_desc_,
      output_desc_;
  absl::flat_hash_map<std::string, std::shared_ptr<LitertBuffer>> inputs_,
      outputs_;
};

// Allocates a buffer matching a stage port descriptor.
absl::StatusOr<std::shared_ptr<LitertBuffer>> AllocateLike(
    const std::shared_ptr<litert::Environment>& env,
    const HardwareBufferDescriptor& desc);

}  // namespace sam2_chain

#endif  // SAM2_WEB_TENSORAPI_CC_SIGNATURE_STAGE_H_

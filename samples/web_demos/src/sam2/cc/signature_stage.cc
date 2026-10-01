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

#include "signature_stage.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_tensor_buffer.h"
#include "litert/cc/litert_tensor_buffer_types.h"

namespace sam2_chain {
namespace {

// Same preference as ModelChain's CompiledModelStage: shareable device
// memory first, then host memory, else whatever the accelerator lists first.
litert::TensorBufferType Preferred(
    const std::vector<litert::TensorBufferType>& types) {
  for (auto want : {litert::TensorBufferType::kAhwb,
                    litert::TensorBufferType::kOpenClBuffer,
                    litert::TensorBufferType::kHostMemory}) {
    for (auto t : types) {
      if (t == want) return t;
    }
  }
  return types.empty() ? litert::TensorBufferType::kHostMemory : types.front();
}

}  // namespace

absl::StatusOr<std::shared_ptr<SignatureStage>> SignatureStage::Create(
    std::string name, std::shared_ptr<litert::CompiledModel> model,
    absl::string_view signature_key) {
  auto index = model->GetSignatureIndex(signature_key);
  if (!index) {
    return absl::NotFoundError(absl::StrCat("stage '", name,
                                            "': no signature '", signature_key,
                                            "': ", index.Error().Message()));
  }
  auto stage = std::shared_ptr<SignatureStage>(new SignatureStage(
      std::move(name), std::move(model), std::string(signature_key), *index));
  if (auto st = stage->DiscoverDescriptors(); !st.ok()) return st;
  return stage;
}

absl::Status SignatureStage::DiscoverDescriptors() {
  auto in_names = model_->GetSignatureInputNames(signature_index_);
  auto out_names = model_->GetSignatureOutputNames(signature_index_);
  if (!in_names || !out_names) {
    return absl::InternalError(
        absl::StrCat("stage '", name_, "': cannot read signature names"));
  }
  auto describe = [&](bool input, size_t i) -> absl::StatusOr<HardwareBufferDescriptor> {
    auto type = input ? model_->GetInputTensorType(signature_index_, i)
                      : model_->GetOutputTensorType(signature_index_, i);
    if (!type) {
      return absl::InternalError(absl::StrCat("stage '", name_,
                                              "': tensor type: ",
                                              type.Error().Message()));
    }
    HardwareBufferDescriptor desc;
    desc.element_type = type->ElementType();
    auto dims = type->Layout().Dimensions();
    desc.shape.assign(dims.begin(), dims.end());
    if (auto bytes = type->Bytes()) desc.size_bytes = *bytes;
    auto req = input ? model_->GetInputBufferRequirements(signature_index_, i)
                     : model_->GetOutputBufferRequirements(signature_index_, i);
    if (req) {
      if (auto sz = req->BufferSize(); sz && *sz > desc.size_bytes) {
        desc.size_bytes = *sz;
      }
      if (auto align = req->Alignment(); align && *align > 0) {
        desc.alignment = *align;
      }
      if (auto types = req->SupportedTypes(); types && !types->empty()) {
        desc.buffer_type = Preferred(
            std::vector<litert::TensorBufferType>(types->begin(), types->end()));
      }
    }
    return desc;
  };
  for (size_t i = 0; i < in_names->size(); ++i) {
    std::string n((*in_names)[i]);
    auto desc = describe(true, i);
    if (!desc.ok()) return desc.status();
    input_names_.push_back(n);
    input_desc_[n] = *desc;
  }
  for (size_t i = 0; i < out_names->size(); ++i) {
    std::string n((*out_names)[i]);
    auto desc = describe(false, i);
    if (!desc.ok()) return desc.status();
    output_names_.push_back(n);
    output_desc_[n] = *desc;
  }
  return absl::OkStatus();
}

absl::StatusOr<HardwareBufferDescriptor> SignatureStage::GetInputDescriptor(
    absl::string_view name) const {
  auto it = input_desc_.find(name);
  if (it == input_desc_.end()) {
    return absl::NotFoundError(
        absl::StrCat("stage '", name_, "' has no input '", name, "'"));
  }
  return it->second;
}

absl::StatusOr<HardwareBufferDescriptor> SignatureStage::GetOutputDescriptor(
    absl::string_view name) const {
  auto it = output_desc_.find(name);
  if (it == output_desc_.end()) {
    return absl::NotFoundError(
        absl::StrCat("stage '", name_, "' has no output '", name, "'"));
  }
  return it->second;
}

absl::Status SignatureStage::SetInputBuffer(
    absl::string_view name, std::shared_ptr<LitertBuffer> buffer) {
  if (!input_desc_.contains(name)) {
    return absl::NotFoundError(
        absl::StrCat("stage '", name_, "' has no input '", name, "'"));
  }
  inputs_.insert_or_assign(name, std::move(buffer));
  return absl::OkStatus();
}

absl::Status SignatureStage::SetOutputBuffer(
    absl::string_view name, std::shared_ptr<LitertBuffer> buffer) {
  if (!output_desc_.contains(name)) {
    return absl::NotFoundError(
        absl::StrCat("stage '", name_, "' has no output '", name, "'"));
  }
  outputs_.insert_or_assign(name, std::move(buffer));
  return absl::OkStatus();
}

std::shared_ptr<LitertBuffer> SignatureStage::GetInputBuffer(
    absl::string_view name) const {
  auto it = inputs_.find(name);
  return it == inputs_.end() ? nullptr : it->second;
}

std::shared_ptr<LitertBuffer> SignatureStage::GetOutputBuffer(
    absl::string_view name) const {
  auto it = outputs_.find(name);
  return it == outputs_.end() ? nullptr : it->second;
}

absl::Status SignatureStage::Run() {
  std::vector<litert::TensorBuffer> ins, outs;
  ins.reserve(input_names_.size());
  outs.reserve(output_names_.size());
  for (const auto& n : input_names_) {
    auto it = inputs_.find(n);
    if (it == inputs_.end() || !it->second) {
      return absl::FailedPreconditionError(
          absl::StrCat("stage '", name_, "': input '", n, "' is not bound"));
    }
    auto dup = it->second->tensor_buffer().Duplicate();
    if (!dup) return absl::InternalError(dup.Error().Message());
    ins.push_back(std::move(*dup));
  }
  for (const auto& n : output_names_) {
    auto it = outputs_.find(n);
    if (it == outputs_.end() || !it->second) {
      auto buf = AllocateLike(env_, output_desc_.at(n));
      if (!buf.ok()) return buf.status();
      it = outputs_.insert_or_assign(n, *buf).first;
    }
    auto dup = it->second->tensor_buffer().Duplicate();
    if (!dup) return absl::InternalError(dup.Error().Message());
    outs.push_back(std::move(*dup));
  }
  auto run = model_->Run(signature_index_, ins, outs);
  if (!run) {
    return absl::InternalError(
        absl::StrCat("stage '", name_, "' (", signature_key_,
                     "): ", run.Error().Message()));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::shared_ptr<LitertBuffer>> AllocateLike(
    const std::shared_ptr<litert::Environment>& env,
    const HardwareBufferDescriptor& desc) {
  return LitertBuffer::CreateManaged(env, desc.buffer_type,
                                     desc.ToRankedTensorType(),
                                     desc.PackedBytes());
}

}  // namespace sam2_chain

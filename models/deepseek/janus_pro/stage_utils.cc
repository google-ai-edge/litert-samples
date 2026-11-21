// Copyright 2026 The ODML Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "models/deepseek/janus_pro/stage_utils.h"

#include <cstddef>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_ranked_tensor_type.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert::omni::any2any {
namespace janus_pro {

absl::Status ValidateTensorBuffer(const TensorBuffer& buffer,
                                  ElementType expected_type,
                                  size_t expected_elements,
                                  absl::string_view label) {
  LITERT_ASSIGN_OR_RETURN(RankedTensorType tensor_type, buffer.TensorType());
  if (tensor_type.ElementType() != expected_type) {
    return absl::InvalidArgumentError(
        absl::StrCat(label, " has unexpected element type (expected ",
                     static_cast<int>(expected_type), ", got ",
                     static_cast<int>(tensor_type.ElementType()), ")."));
  }
  LITERT_ASSIGN_OR_RETURN(size_t num_elements,
                          tensor_type.Layout().NumElements());
  if (num_elements != expected_elements) {
    return absl::InvalidArgumentError(
        absl::StrCat(label, " element count (", num_elements,
                     ") does not match expected (", expected_elements, ")."));
  }
  return absl::OkStatus();
}

}  // namespace janus_pro
}  // namespace litert::omni::any2any

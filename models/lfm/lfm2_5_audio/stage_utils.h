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

#ifndef MODELS_LFM_LFM2_5_AUDIO_STAGE_UTILS_H_
#define MODELS_LFM_LFM2_5_AUDIO_STAGE_UTILS_H_

#include <cstddef>

#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "litert/cc/litert_element_type.h"
#include "litert/cc/litert_tensor_buffer.h"

namespace litert::omni::any2any {
namespace lfm2_audio {

// Validates that `buffer` has element type `expected_type` and exactly
// `expected_elements` total elements, returning an `absl::InvalidArgumentError`
// prefixed with `label` on mismatch.
absl::Status ValidateTensorBuffer(const TensorBuffer& buffer,
                                  ElementType expected_type,
                                  size_t expected_elements,
                                  absl::string_view label);

}  // namespace lfm2_audio
}  // namespace litert::omni::any2any

#endif  // MODELS_LFM_LFM2_5_AUDIO_STAGE_UTILS_H_

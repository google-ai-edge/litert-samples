/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// CPU-only version of the image segmentation sample that does not depend on
// OpenGL/EGL. This is intended for platforms where GLES compute shaders are
// unavailable (e.g. Windows desktop).
//
// All image pre/post-processing is performed using the pure-CPU helpers already
// provided by ImageUtils.

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "litert/c/litert_common.h"
#include "litert/cc/litert_common.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_macros.h"
#include "litert/cc/litert_model.h"
#include "compiled_model_api/image_segmentation/c++_segmentation/build_from_source/image_utils.h"
#include "compiled_model_api/image_segmentation/c++_segmentation/build_from_source/timing_utils.h"

int main(int argc, char* argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <model_path> <input_image_path> <output_image_path>"
              << std::endl;
    return 1;
  }

  const std::string model_path = argv[1];
  const std::string input_file = argv[2];
  const std::string output_file = argv[3];

  std::vector<RGBAColor> mask_colors = {
      {1.0f, 0.0f, 0.0f, 0.1f}, {0.0f, 1.0f, 0.0f, 0.1f},
      {0.0f, 0.0f, 1.0f, 0.1f}, {1.0f, 1.0f, 0.0f, 0.1f},
      {1.0f, 0.0f, 1.0f, 0.1f}, {0.0f, 1.0f, 1.0f, 0.1f}};

  // Initialize LiteRT environment
  LITERT_ASSIGN_OR_ABORT(auto env, litert::Environment::Create({}));

  // Compile the model for the CPU
  LITERT_ASSIGN_OR_ABORT(auto compiled_model,
                         litert::CompiledModel::Create(
                             env, model_path, litert::HwAccelerators::kCpu));

  // Create input and output buffers
  LITERT_ASSIGN_OR_ABORT(auto input_buffers,
                         compiled_model.CreateInputBuffers());
  LITERT_ASSIGN_OR_ABORT(auto output_buffers,
                         compiled_model.CreateOutputBuffers());

  // ================= PRE-PROCESSING (CPU) =================
  ProfilingTimestamps profiling_timestamps;
  profiling_timestamps.load_image_start_time = absl::Now();

  int width_orig = 0, height_orig = 0, channels_file = 0;
  const int loaded_channels = 3;
  auto img_data_cpu = ImageUtils::LoadImage(input_file, width_orig, height_orig,
                                            channels_file, loaded_channels);
  if (!img_data_cpu) {
    std::cerr << "Failed to load image file: " << input_file << std::endl;
    return 1;
  }

  profiling_timestamps.load_image_end_time =
      profiling_timestamps.e2e_start_time =
          profiling_timestamps.pre_process_start_time = absl::Now();

  const int preprocessed_width = 256;
  const int preprocessed_height = 256;

  std::vector<float> preprocessed_buffer_data =
      ImageUtils::PreprocessInputForSegmentationCpu(
          img_data_cpu, width_orig, height_orig, loaded_channels,
          preprocessed_width, preprocessed_height);
  ImageUtils::FreeImageData(img_data_cpu);

  // Write preprocessed data to the input tensor buffer
  LITERT_ABORT_IF_ERROR(
      input_buffers[0].Write(absl::MakeConstSpan(preprocessed_buffer_data)));

  profiling_timestamps.pre_process_end_time =
      profiling_timestamps.inference_start_time = absl::Now();

  // ================= INFERENCE =================
  LITERT_ABORT_IF_ERROR(compiled_model.Run(input_buffers, output_buffers));

  profiling_timestamps.inference_end_time =
      profiling_timestamps.post_process_start_time = absl::Now();

  // ================= POST-PROCESSING (CPU) =================
  std::vector<float> raw_output_data(preprocessed_width *
                                     preprocessed_height * 6);
  LITERT_ABORT_IF_ERROR(
      output_buffers[0].Read(absl::MakeSpan(raw_output_data)));

  // Deinterleave the 6 masks
  std::vector<std::vector<float>> masks = ImageUtils::DeinterleaveMasksCpu(
      raw_output_data.data(), preprocessed_width, preprocessed_height);

  // Reload the original image at its native resolution for blending.
  // We load as 3 channels (RGB) which is what ApplyColoredMasksCpu expects.
  int blend_w = 0, blend_h = 0, blend_ch_file = 0;
  const int blend_channels = 3;
  unsigned char* blend_img = ImageUtils::LoadImage(
      input_file, blend_w, blend_h, blend_ch_file, blend_channels);
  if (!blend_img) {
    std::cerr << "Failed to reload image for blending." << std::endl;
    return 1;
  }

  // The masks are at 256x256 but the original image may be a different size.
  // We need to resize the masks to the original image size before blending.
  // Simple nearest-neighbour resize for each mask.
  std::vector<std::vector<float>> resized_masks(6);
  for (int i = 0; i < 6; ++i) {
    resized_masks[i].resize(blend_w * blend_h);
    for (int y = 0; y < blend_h; ++y) {
      for (int x = 0; x < blend_w; ++x) {
        int src_x = x * preprocessed_width / blend_w;
        int src_y = y * preprocessed_height / blend_h;
        src_x = std::min(src_x, preprocessed_width - 1);
        src_y = std::min(src_y, preprocessed_height - 1);
        resized_masks[i][y * blend_w + x] =
            masks[i][src_y * preprocessed_width + src_x];
      }
    }
  }

  std::vector<unsigned char> blended_image = ImageUtils::ApplyColoredMasksCpu(
      blend_img, blend_w, blend_h, blend_channels, resized_masks, mask_colors);
  ImageUtils::FreeImageData(blend_img);

  profiling_timestamps.post_process_end_time =
      profiling_timestamps.e2e_end_time =
          profiling_timestamps.save_image_start_time = absl::Now();

  // Save the output image
  if (!ImageUtils::SaveImage(output_file, blend_w, blend_h, blend_channels,
                             blended_image.data())) {
    std::cerr << "Failed to save final blended image." << std::endl;
    return 1;
  }
  std::cout << "Successfully saved final blended image to " << output_file
            << std::endl;

  profiling_timestamps.save_image_end_time = absl::Now();
  PrintTiming(profiling_timestamps);

  return 0;
}

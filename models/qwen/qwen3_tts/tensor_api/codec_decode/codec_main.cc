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

// Codec decoder host reference: serialize the fixed-window decode graph,
// run production-style chunking (64-frame chunks + 25 frames left context),
// benchmark, and optionally dump the waveform for the numpy equivalence
// check (verify/numpy_codec_ref.py).
//
//   bazel build -c opt //models/qwen/qwen3_tts/tensor_api/codec_decode:codec_main
//   codec_main --weights=.../speech_tokenizer/model.safetensors \
//     --frames=128 --accelerator=cpu --dump_wav=/tmp/wav.f32
//
// Codes input: --codes_file (raw i32 [16, frames]) or deterministic
// synthetic codes. Real-time budget: 24000 samples/s.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/datatypes.h"
#include "models/qwen/qwen3_tts/tensor_api/codec_decode/codec_config.h"
#include "models/qwen/qwen3_tts/tensor_api/codec_decode/codec_graph.h"
#include "models/qwen/qwen3_tts/tensor_api/codec_decode/codec_weights.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, weights, "",
          "speech_tokenizer/model.safetensors path (required)");
ABSL_FLAG(int, frames, 128, "Total codec frames of the synthetic utterance");
ABSL_FLAG(int, runs, 8, "Timed chunk invocations (after 2 warmup)");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu|gpu");
ABSL_FLAG(std::string, gpu_precision, "fp16", "fp16|fp32");
ABSL_FLAG(std::string, gpu_buffer_storage, "default",
          "default|buffer|texture2d");
ABSL_FLAG(std::string, codes_file, "",
          "Raw i32 file [16, frames] (empty = deterministic synthetic codes)");
ABSL_FLAG(std::string, dump_wav, "",
          "If set, write the full decoded waveform as raw fp32");
ABSL_FLAG(std::string, tflite_path, "/tmp/codec_decode.tflite",
          "Where to write the serialized model");

namespace {

using ::litert::tensor::Create;
using ::litert::tensor::LitertDynamicRunner;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::Type;
namespace codec = ::litert::tensor::examples::codec;

absl::Status Run() {
  codec::CodecConfig config;
  const int frames = absl::GetFlag(FLAGS_frames);
  const int window = config.chunk_frames + config.context_frames;  // 89
  const int up = config.total_upsample();
  const std::string weights_path = absl::GetFlag(FLAGS_weights);
  if (weights_path.empty()) {
    return absl::InvalidArgumentError("--weights is required");
  }

  auto weights_or = codec::LoadCodecWeights(config, weights_path);
  if (!weights_or.ok()) return weights_or.status();
  std::cout << "weights: " << weights_path << " (" << weights_or->size()
            << " graph constants)" << std::endl;

  codec::TfTensor codes_in = codec::MakeCodesInput(config, window);
  codec::TfTensor wav_out =
      codec::BuildCodecDecode(config, codes_in, window, *weights_or);

  ModelFactory factory;
  {
    std::vector<::litert::tensor::TensorHandle> ins = {codes_in};
    std::vector<::litert::tensor::TensorHandle> outs = {wav_out};
    auto status = factory.AddSignature(ins, outs, "decode");
    if (!status.ok()) return status;
  }
  const std::string tflite_path = absl::GetFlag(FLAGS_tflite_path);
  auto save_status = factory.Save(tflite_path);
  if (!save_status.ok()) return save_status;
  std::cout << "Serialized: " << tflite_path << " (window " << window
            << " frames -> " << window * up << " samples)" << std::endl;

  auto env = ::litert::Environment::Create({});
  if (!env) return absl::InternalError("Environment::Create failed");
  auto options = ::litert::Options::Create();
  if (!options) return absl::InternalError("Options::Create failed");
  const bool use_gpu = absl::GetFlag(FLAGS_accelerator) == "gpu";
  options->SetHardwareAccelerators(use_gpu ? ::litert::HwAccelerators::kGpu
                                           : ::litert::HwAccelerators::kCpu);
  if (use_gpu) {
    auto gpu_options = options->GetGpuOptions();
    if (gpu_options.HasValue()) {
      const std::string precision = absl::GetFlag(FLAGS_gpu_precision);
      if (precision == "fp32") {
        gpu_options->SetPrecision(::litert::GpuOptions::Precision::kFp32);
      } else if (precision != "fp16") {
        return absl::InvalidArgumentError("--gpu_precision must be fp16|fp32");
      }
      const std::string storage = absl::GetFlag(FLAGS_gpu_buffer_storage);
      if (storage == "buffer") {
        gpu_options->SetBufferStorageType(
            ::litert::GpuOptions::BufferStorageType::kBuffer);
      } else if (storage == "texture2d") {
        gpu_options->SetBufferStorageType(
            ::litert::GpuOptions::BufferStorageType::kTexture2D);
      } else if (storage != "default") {
        return absl::InvalidArgumentError(
            "--gpu_buffer_storage must be default|buffer|texture2d");
      }
      std::cout << "gpu precision: " << precision << ", storage: " << storage
                << std::endl;
    }
  }
  auto runner_or = LitertDynamicRunner::Create(*env, tflite_path, *options);
  if (!runner_or.ok()) return runner_or.status();
  auto runner = std::move(*runner_or);

  // --- utterance codes [16, frames] ---
  std::vector<int32_t> codes(static_cast<size_t>(config.num_quantizers) *
                             frames);
  const std::string codes_file = absl::GetFlag(FLAGS_codes_file);
  if (codes_file.empty()) {
    unsigned state = 12345u;
    for (auto& c : codes) {
      state = state * 1664525u + 1013904223u;
      c = static_cast<int32_t>((state >> 8) % config.codebook_size);
    }
  } else {
    std::ifstream in(codes_file, std::ios::binary);
    if (!in) return absl::NotFoundError("codes_file: " + codes_file);
    in.read(reinterpret_cast<char*>(codes.data()),
            codes.size() * sizeof(int32_t));
    if (static_cast<size_t>(in.gcount()) != codes.size() * sizeof(int32_t)) {
      return absl::InvalidArgumentError("codes_file smaller than 16 x frames");
    }
  }

  // --- chunked decode: 64-frame chunks, 25 frames left context ---
  std::vector<float> wav;
  wav.reserve(static_cast<size_t>(frames) * up);
  std::vector<double> chunk_ms;
  int start = 0;
  while (start < frames) {
    int ctx = std::min(config.context_frames, start);
    int end = std::min(start + config.chunk_frames, frames);
    // Window layout: [ctx frames of context][end-start new frames][pad].
    std::vector<int32_t> win(static_cast<size_t>(config.num_quantizers) *
                                 window,
                             0);
    for (int q = 0; q < config.num_quantizers; ++q) {
      for (int t = 0; t < ctx + (end - start); ++t) {
        win[static_cast<size_t>(q) * window + t] =
            codes[static_cast<size_t>(q) * frames + (start - ctx) + t];
      }
    }
    auto st = runner.SetInput(
        "decode", "codes",
        Create("codes", Type::kI32, {config.num_quantizers, window},
               std::move(win)));
    if (!st.ok()) return st;

    auto t0 = std::chrono::steady_clock::now();
    st = runner.Run("decode");
    auto t1 = std::chrono::steady_clock::now();
    if (!st.ok()) return st;
    chunk_ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    auto out = runner.GetOutput("decode", "wav");
    if (!out.ok()) return out.status();
    auto buffer = out->GetBuffer();
    if (!buffer.ok()) return buffer.status();
    auto lock = buffer->Lock();
    const float* data = reinterpret_cast<const float*>(lock.data());
    // Crop the context and any right padding.
    wav.insert(wav.end(), data + static_cast<size_t>(ctx) * up,
               data + static_cast<size_t>(ctx + (end - start)) * up);
    start = end;
  }

  // --- steady-state benchmark on a full window ---
  const int runs = absl::GetFlag(FLAGS_runs);
  std::vector<double> bench_ms;
  for (int r = 0; r < runs + 2; ++r) {
    auto t0 = std::chrono::steady_clock::now();
    auto st = runner.Run("decode");
    auto t1 = std::chrono::steady_clock::now();
    if (!st.ok()) return st;
    if (r >= 2) {
      bench_ms.push_back(
          std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
  }
  std::sort(bench_ms.begin(), bench_ms.end());
  double median = bench_ms[bench_ms.size() / 2];
  double audio_ms = 1000.0 * config.chunk_frames * up / 24000.0;

  double sum_sq = 0.0;
  float peak = 0.0f;
  bool finite = true;
  for (float v : wav) {
    if (!std::isfinite(v)) finite = false;
    sum_sq += static_cast<double>(v) * v;
    peak = std::max(peak, std::fabs(v));
  }
  std::cout << "decoded " << frames << " frames -> " << wav.size()
            << " samples; RMS " << std::sqrt(sum_sq / wav.size()) << ", peak "
            << peak << (finite ? " (all finite)" : " (NON-FINITE!)")
            << std::endl;
  std::cout << "chunk window " << window << " frames: median " << median
            << " ms (" << bench_ms.size() << " runs) vs " << audio_ms
            << " ms of new audio => RTF " << (median / audio_ms) << std::endl;
  if (!finite) return absl::InternalError("non-finite waveform");

  const std::string dump = absl::GetFlag(FLAGS_dump_wav);
  if (!dump.empty()) {
    std::ofstream out(dump, std::ios::binary);
    if (!out) return absl::InternalError("cannot write " + dump);
    out.write(reinterpret_cast<const char*>(wav.data()),
              wav.size() * sizeof(float));
    std::cout << "wav dumped to " << dump << std::endl;
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = Run();
  if (!status.ok()) {
    std::cerr << "FAILED: " << status << std::endl;
    return 1;
  }
  return 0;
}

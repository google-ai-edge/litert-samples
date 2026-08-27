// SAM2.1 hiera-tiny image path — build, serialize, run, verify, bench.
//
// Usage (from the LiteRT repo root; GPU runs need cwd =
// litert/prebuilt/macos_arm64 so the Metal accelerator dylib resolves):
//   bazel build --config=macos_arm64 //models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_main
//   sam2_main --weights=.../sam2_tiny_512.safetensors \
//     --accelerator=gpu --gpu_precision=fp16 --gpu_buffer_storage=buffer
//
// Runs the same fixture as the reference apps: a white circle on black
// (radius size/4-8), one positive point at the center, ImageNet
// normalization. Prints iou_scores / object_score / best-mask foreground
// exactly like the mlx-swift app's SAM2MLXBENCH block, plus warm medians.
// --dump_dir writes every I/O tensor raw for the PyTorch parity check
// (verify/sam2_torch_ref.py).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/datatypes.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_config.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_graph.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_weights.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, weights, "",
          "sam2_tiny_512.safetensors path (empty = synthetic weights for a "
          "shape/route check)");
ABSL_FLAG(std::string, tflite_path, "/tmp/sam2_image.tflite",
          "Where to write the serialized two-signature model");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu|gpu");
ABSL_FLAG(std::string, gpu_precision, "fp16",
          "fp16|fp32 — GPU calculation precision (fp16 is the delegate "
          "default; fp32 for CPU-parity verification)");
ABSL_FLAG(std::string, gpu_buffer_storage, "default",
          "default|buffer|texture2d — GPU tensor storage (texture limits "
          "can silently push large graphs to CPU; probe with buffer)");
ABSL_FLAG(std::string, attention, "raw",
          "raw|sdpa|rbmm — plain ops, the odml.scaled_dot_product_attention "
          "composite, or the odml.runtime_bmm QK+AV pair (int32 control "
          "inputs bound at the full static length)");
ABSL_FLAG(std::string, hypernet, "raw",
          "raw|rbmm — the hypernetwork mask projection as a plain "
          "BatchMatMul or one dst-bounded odml.runtime_bmm");
ABSL_FLAG(std::string, norms, "raw",
          "raw|composite — plain ops or the odml.layer_norm composite");
ABSL_FLAG(std::string, upsampler, "transpose",
          "transpose|d2s — mask upsampler as TRANSPOSE_CONV or the exact "
          "4x(1x1)+DepthToSpace expansion (for delegates that reject "
          "TRANSPOSE_CONV, e.g. Mali ML Drift)");
ABSL_FLAG(int, runs, 20, "Timed iterations per stage");
ABSL_FLAG(int, warmup, 5, "Warmup iterations");
ABSL_FLAG(double, point_x, -1.0, "Prompt x in image space (-1 = center)");
ABSL_FLAG(double, point_y, -1.0, "Prompt y in image space (-1 = center)");
ABSL_FLAG(std::string, input_file, "",
          "Raw fp32 NHWC [1,512,512,3] normalized input (empty = the "
          "synthetic circle fixture)");
ABSL_FLAG(std::string, dump_dir, "",
          "If set, write pixels/image_embeddings/feat_s1/feat_s0/masks/"
          "iou_scores/object_score as raw fp32 for the parity check");
ABSL_FLAG(std::string, split_dir, "",
          "If set, ALSO serialize single-signature models "
          "sam2_encoder_512.tflite / sam2_decoder_512.tflite there (for "
          "harnesses that only run a model's first signature, e.g. the "
          "on-device gpu_test binary)");

namespace {

using ::litert::tensor::Create;
using ::litert::tensor::LitertDynamicRunner;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::Type;
namespace sam2 = ::litert::tensor::examples::sam2;

// The reference fixture: white circle (1.0) on black, radius size/4 - 8,
// ImageNet-normalized, NHWC.
std::vector<float> CircleInput(int size) {
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float stddev[3] = {0.229f, 0.224f, 0.225f};
  std::vector<float> out(static_cast<size_t>(size) * size * 3);
  const int cx = size / 2;
  const int cy = size / 2;
  const int r = size / 4 - 8;
  const int r2 = r * r;
  for (int y = 0; y < size; ++y) {
    for (int x = 0; x < size; ++x) {
      const int dx = x - cx;
      const int dy = y - cy;
      const float value = (dx * dx + dy * dy <= r2) ? 1.0f : 0.0f;
      size_t base = (static_cast<size_t>(y) * size + x) * 3;
      for (int ch = 0; ch < 3; ++ch) {
        out[base + ch] = (value - mean[ch]) / stddev[ch];
      }
    }
  }
  return out;
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

absl::Status ReadFloats(LitertDynamicRunner& runner, const std::string& sig,
                        const std::string& name, std::vector<float>& out) {
  auto t = runner.GetOutput(sig, name);
  if (!t.ok()) return t.status();
  auto buffer = t->GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  const float* data = reinterpret_cast<const float*>(lock.data());
  out.assign(data, data + lock.size() / sizeof(float));
  return absl::OkStatus();
}

absl::Status DumpFile(const std::string& dir, const std::string& name,
                      const std::vector<float>& data) {
  std::ofstream out(absl::StrCat(dir, "/", name, ".f32"), std::ios::binary);
  if (!out) return absl::InternalError(absl::StrCat("cannot write ", name));
  out.write(reinterpret_cast<const char*>(data.data()),
            data.size() * sizeof(float));
  return absl::OkStatus();
}

absl::Status Run() {
  sam2::Sam2Config config;
  const std::string attention = absl::GetFlag(FLAGS_attention);
  if (attention != "raw" && attention != "sdpa" && attention != "rbmm") {
    return absl::InvalidArgumentError("--attention must be raw|sdpa|rbmm");
  }
  config.use_sdpa_composite = attention == "sdpa";
  config.use_rbmm_attention = attention == "rbmm";
  const std::string hypernet = absl::GetFlag(FLAGS_hypernet);
  if (hypernet != "raw" && hypernet != "rbmm") {
    return absl::InvalidArgumentError("--hypernet must be raw|rbmm");
  }
  config.use_rbmm_hypernet = hypernet == "rbmm";
  const std::string norms = absl::GetFlag(FLAGS_norms);
  if (norms != "raw" && norms != "composite") {
    return absl::InvalidArgumentError("--norms must be raw|composite");
  }
  config.use_layer_norm_composite = norms == "composite";
  const std::string upsampler = absl::GetFlag(FLAGS_upsampler);
  if (upsampler != "transpose" && upsampler != "d2s") {
    return absl::InvalidArgumentError("--upsampler must be transpose|d2s");
  }
  config.use_d2s_upsampler = upsampler == "d2s";

  sam2::WeightMap weights;
  const std::string weights_path = absl::GetFlag(FLAGS_weights);
  if (weights_path.empty()) {
    weights = sam2::MakeSyntheticWeights(config, /*seed=*/42);
    std::cout << "weights: synthetic (seed 42) — shape/route check only"
              << std::endl;
  } else {
    auto weights_or = sam2::LoadCheckpointWeights(config, weights_path);
    if (!weights_or.ok()) return weights_or.status();
    weights = std::move(*weights_or);
    std::cout << "weights: " << weights_path << " (" << weights.size()
              << " tensors, fp16->fp32)" << std::endl;
  }
  if (config.use_sdpa_composite) {
    std::cout << "attention: odml.scaled_dot_product_attention composite"
              << std::endl;
  }
  if (config.use_rbmm_attention) {
    std::cout << "attention: odml.runtime_bmm QK+AV pair" << std::endl;
  }
  if (config.use_rbmm_hypernet) {
    std::cout << "hypernet: odml.runtime_bmm" << std::endl;
  }
  if (config.use_layer_norm_composite) {
    std::cout << "norms: odml.layer_norm composite" << std::endl;
  }

  sam2::EncoderInputs enc_in = sam2::MakeEncoderInputs(config);
  sam2::EncoderOutputs enc_out = sam2::BuildEncoder(config, enc_in, weights);
  auto enc_rbmm = sam2::TakeRbmmParams();
  sam2::DecoderInputs dec_in = sam2::MakeDecoderInputs(config);
  sam2::DecoderOutputs dec_out = sam2::BuildDecoder(config, dec_in, weights);
  auto dec_rbmm = sam2::TakeRbmmParams();

  ModelFactory factory;
  {
    std::vector<::litert::tensor::TensorHandle> ins, outs;
    for (auto& t : enc_in.AsList()) ins.push_back(t);
    for (auto& [s, t] : enc_rbmm) ins.push_back(t);
    for (auto& t : enc_out.AsList()) outs.push_back(t);
    auto status = factory.AddSignature(ins, outs, "encode_image");
    if (!status.ok()) return status;
  }
  {
    std::vector<::litert::tensor::TensorHandle> ins, outs;
    for (auto& t : dec_in.AsList()) ins.push_back(t);
    for (auto& [s, t] : dec_rbmm) ins.push_back(t);
    for (auto& t : dec_out.AsList()) outs.push_back(t);
    auto status = factory.AddSignature(ins, outs, "decode_mask");
    if (!status.ok()) return status;
  }
  const std::string tflite_path = absl::GetFlag(FLAGS_tflite_path);
  auto save_status = factory.Save(tflite_path);
  if (!save_status.ok()) return save_status;
  std::cout << "Serialized: " << tflite_path << std::endl;

  const std::string split_dir = absl::GetFlag(FLAGS_split_dir);
  if (!split_dir.empty()) {
    sam2::EncoderInputs enc_in2 = sam2::MakeEncoderInputs(config);
    sam2::EncoderOutputs enc_out2 = sam2::BuildEncoder(config, enc_in2, weights);
    auto enc_rbmm2 = sam2::TakeRbmmParams();
    ModelFactory enc_factory;
    std::vector<::litert::tensor::TensorHandle> ins, outs;
    for (auto& t : enc_in2.AsList()) ins.push_back(t);
    for (auto& [s, t] : enc_rbmm2) ins.push_back(t);
    for (auto& t : enc_out2.AsList()) outs.push_back(t);
    auto st1 = enc_factory.AddSignature(ins, outs, "encode_image");
    if (!st1.ok()) return st1;
    st1 = enc_factory.Save(split_dir + "/sam2_encoder_512.tflite");
    if (!st1.ok()) return st1;

    sam2::DecoderInputs dec_in2 = sam2::MakeDecoderInputs(config);
    sam2::DecoderOutputs dec_out2 = sam2::BuildDecoder(config, dec_in2, weights);
    auto dec_rbmm2 = sam2::TakeRbmmParams();
    ModelFactory dec_factory;
    ins.clear();
    outs.clear();
    for (auto& t : dec_in2.AsList()) ins.push_back(t);
    for (auto& [s, t] : dec_rbmm2) ins.push_back(t);
    for (auto& t : dec_out2.AsList()) outs.push_back(t);
    auto st2 = dec_factory.AddSignature(ins, outs, "decode_mask");
    if (!st2.ok()) return st2;
    st2 = dec_factory.Save(split_dir + "/sam2_decoder_512.tflite");
    if (!st2.ok()) return st2;
    std::cout << "Split models serialized to " << split_dir << std::endl;
  }

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

  // --- Stage input ---
  const int size = config.image_size;
  std::vector<float> pixels;
  const std::string input_file = absl::GetFlag(FLAGS_input_file);
  if (input_file.empty()) {
    pixels = CircleInput(size);
  } else {
    pixels.resize(static_cast<size_t>(size) * size * 3);
    std::ifstream in(input_file, std::ios::binary);
    if (!in) return absl::NotFoundError("input_file: " + input_file);
    in.read(reinterpret_cast<char*>(pixels.data()),
            pixels.size() * sizeof(float));
    if (static_cast<size_t>(in.gcount()) != pixels.size() * sizeof(float)) {
      return absl::InvalidArgumentError("input_file wrong size");
    }
  }
  auto set_pixels = [&]() -> absl::Status {
    return runner.SetInput(
        "encode_image", "pixels",
        Create("pixels", Type::kFP32, {1, size, size, 3},
               std::vector<float>(pixels)));
  };

  float px = static_cast<float>(absl::GetFlag(FLAGS_point_x));
  float py = static_cast<float>(absl::GetFlag(FLAGS_point_y));
  if (px < 0) px = size / 2.0f;
  if (py < 0) py = size / 2.0f;
  auto set_decoder_inputs = [&]() -> absl::Status {
    for (const std::string& name :
         {std::string("image_embeddings"), std::string("feat_s1"),
          std::string("feat_s0")}) {
      auto t = runner.GetOutput("encode_image", name);
      if (!t.ok()) return t.status();
      auto st = runner.SetInput("decode_mask", name, *t);
      if (!st.ok()) return st;
    }
    return runner.SetInput("decode_mask", "point_coords",
                           Create("point_coords", Type::kFP32, {1, 1, 2},
                                  std::vector<float>{px, py}));
  };

  // odml.runtime_bmm control inputs: seven int32 copies of the bound
  // length S per input (the proven attn_bench fill pattern; element 2 is
  // the one the kernels read). Set once — inputs persist across Run().
  auto set_rbmm_params =
      [&](const std::string& sig,
          const std::vector<std::pair<int, sam2::TfTensor>>& params)
      -> absl::Status {
    for (const auto& [s, t] : params) {
      const std::string name = absl::StrCat("rbmm_s", s);
      auto st = runner.SetInput(
          sig, name,
          Create(name, Type::kI32, {1, 1, 1, 7},
                 std::vector<int32_t>(7, s)));
      if (!st.ok()) return st;
    }
    return absl::OkStatus();
  };
  {
    auto st = set_rbmm_params("encode_image", enc_rbmm);
    if (!st.ok()) return st;
    st = set_rbmm_params("decode_mask", dec_rbmm);
    if (!st.ok()) return st;
  }

  // --- Warmup + correctness pass ---
  auto st = set_pixels();
  if (!st.ok()) return st;
  st = runner.Run("encode_image");
  if (!st.ok()) return st;
  st = set_decoder_inputs();
  if (!st.ok()) return st;
  st = runner.Run("decode_mask");
  if (!st.ok()) return st;

  const int warmup = absl::GetFlag(FLAGS_warmup);
  const int runs = absl::GetFlag(FLAGS_runs);
  for (int i = 0; i < warmup; ++i) {
    st = runner.Run("encode_image");
    if (!st.ok()) return st;
    st = runner.Run("decode_mask");
    if (!st.ok()) return st;
  }

  std::vector<double> enc_ms, dec_ms;
  for (int i = 0; i < runs; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    st = runner.Run("encode_image");
    if (!st.ok()) return st;
    enc_ms.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count());
  }
  for (int i = 0; i < runs; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    st = runner.Run("decode_mask");
    if (!st.ok()) return st;
    dec_ms.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count());
  }

  // --- Outputs ---
  std::vector<float> masks, iou, obj;
  st = ReadFloats(runner, "decode_mask", "masks", masks);
  if (!st.ok()) return st;
  st = ReadFloats(runner, "decode_mask", "iou_scores", iou);
  if (!st.ok()) return st;
  st = ReadFloats(runner, "decode_mask", "object_score", obj);
  if (!st.ok()) return st;

  const int mg = config.mask_grid();
  const int plane = mg * mg;
  int best = 0;
  for (int j = 1; j < 3; ++j) {
    if (iou[j] > iou[best]) best = j;
  }
  auto fg_count = [&](int m) {
    int fg = 0;
    for (int i = 0; i < plane; ++i) {
      if (masks[static_cast<size_t>(m) * plane + i] > 0.0f) ++fg;
    }
    return fg;
  };

  std::cout << absl::StrCat(
      "SAM2 hiera-tiny · LiteRT Tensor API (",
      use_gpu ? absl::StrCat("gpu ", absl::GetFlag(FLAGS_gpu_precision))
              : "cpu fp32",
      ", ", size, "x", size, ")\n", "enc_median=",
      absl::StrCat(Median(enc_ms)), "ms  dec_median=",
      absl::StrCat(Median(dec_ms)), "ms  (runs=", runs, ")\n", "iou_scores=[",
      iou[0], ", ", iou[1], ", ", iou[2], "]  object_score=", obj[0], "\n",
      "best mask = [", best, "]: fg=", fg_count(best), "/", plane,
      "   (mask[0] fg=", fg_count(0), "/", plane, ")\n");

  const std::string dump_dir = absl::GetFlag(FLAGS_dump_dir);
  if (!dump_dir.empty()) {
    st = DumpFile(dump_dir, "pixels", pixels);
    if (!st.ok()) return st;
    for (const std::string& name :
         {std::string("image_embeddings"), std::string("feat_s1"),
          std::string("feat_s0")}) {
      std::vector<float> data;
      st = ReadFloats(runner, "encode_image", name, data);
      if (!st.ok()) return st;
      st = DumpFile(dump_dir, name, data);
      if (!st.ok()) return st;
    }
    st = DumpFile(dump_dir, "masks", masks);
    if (!st.ok()) return st;
    st = DumpFile(dump_dir, "iou_scores", iou);
    if (!st.ok()) return st;
    st = DumpFile(dump_dir, "object_score", obj);
    if (!st.ok()) return st;
    std::cout << "dumped raw outputs to " << dump_dir << std::endl;
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = Run();
  if (!status.ok()) {
    std::cerr << "FAIL: " << status << std::endl;
    return 1;
  }
  return 0;
}

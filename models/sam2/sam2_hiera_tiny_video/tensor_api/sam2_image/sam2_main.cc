// SAM2.1 hiera-tiny image path — build, serialize, run, verify, bench.
//
// Usage (from the LiteRT repo root; GPU runs need cwd =
// litert/prebuilt/macos_arm64 so the Metal accelerator dylib resolves):
//   bazel build --config=macos_arm64 //models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image:sam2_main
//   sam2_main --weights=.../sam2_tiny_512.safetensors \
//     --accelerator=gpu --gpu_precision=fp16 --gpu_buffer_storage=buffer
//
// A frame runs through three graphs on the same Environment and accelerator:
//   1. Pre-processing (CreatePreprocessRunner, a CreateLambdaRunner graph):
//      the RGBA8888 camera frame [1,H,W,4] is sliced to RGB, cast to fp32,
//      resized to [1,512,512,3] and ImageNet-normalized, written straight
//      into encode_image's "pixels" input buffer.
//   2. The model (LitertDynamicRunner): the encode_image and decode_mask
//      signatures, serialized to --tflite_path.
//   3. Post-processing (CreatePostprocessRunner, a CreateLambdaRunner
//      graph): decode_mask's [1,3,128,128] mask logits, read straight from
//      its output buffer, are upsampled to the frame size and thresholded
//      at 0.
// Only the model is serialized; the pre/post graphs are authored with the
// same Tensor API ops and compiled in-process.
//
// By default the frame is the reference apps' fixture at 512x512: a white
// circle on black (radius size/4-8), one positive point at the center,
// ImageNet normalization (the pre-processing output is checked against the
// host-normalized CircleInput). Prints iou_scores / object_score /
// best-mask foreground exactly like the mlx-swift app's SAM2MLXBENCH block,
// plus warm medians. --dump_dir writes every I/O tensor raw for the PyTorch
// parity check (verify/sam2_torch_ref.py).

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
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_config.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_graph.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_weights.h"
#include "tensor/runners/litert/lambda_model_runner.h"
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
ABSL_FLAG(int, frame_width, 512,
          "Width of the RGBA8888 frame fed to the pre-processing graph");
ABSL_FLAG(int, frame_height, 512,
          "Height of the RGBA8888 frame fed to the pre-processing graph");
ABSL_FLAG(std::string, frame_file, "",
          "Raw RGBA8888 frame, frame_width*frame_height*4 bytes (empty = "
          "the synthetic circle fixture drawn at the frame size)");
ABSL_FLAG(double, point_x, -1.0, "Prompt x in frame pixels (-1 = center)");
ABSL_FLAG(double, point_y, -1.0, "Prompt y in frame pixels (-1 = center)");
ABSL_FLAG(std::string, input_file, "",
          "Raw fp32 NHWC [1,512,512,3] normalized input that bypasses the "
          "pre-processing graph (empty = pre-process the frame in-graph)");
ABSL_FLAG(std::string, dump_dir, "",
          "If set, write pixels/image_embeddings/feat_s1/feat_s0/masks/"
          "iou_scores/object_score as raw fp32 for the parity check, plus "
          "frame_masks (the post-processed [1,H,W,3] {0,1} masks)");
ABSL_FLAG(std::string, split_dir, "",
          "If set, ALSO serialize single-signature models "
          "sam2_encoder_512.tflite / sam2_decoder_512.tflite there (for "
          "harnesses that only run a model's first signature, e.g. the "
          "on-device gpu_test binary)");

namespace {

using ::litert::tensor::Create;
using ::litert::tensor::CreateLambdaRunner;
using ::litert::tensor::LitertDynamicRunner;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::OwningCpuBuffer;
using ::litert::tensor::TensorsMap;
using ::litert::tensor::Type;
namespace sam2 = ::litert::tensor::examples::sam2;

constexpr float kImageNetMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kImageNetStd[3] = {0.229f, 0.224f, 0.225f};

// The reference fixture: white circle (1.0) on black, radius size/4 - 8,
// ImageNet-normalized, NHWC. Also the host reference for the
// pre-processing graph's output.
std::vector<float> CircleInput(int size) {
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
        out[base + ch] = (value - kImageNetMean[ch]) / kImageNetStd[ch];
      }
    }
  }
  return out;
}

// The same fixture as an RGBA8888 camera frame: white (255) circle on
// black, radius min(width, height)/4 - 8, opaque. At 512x512 the
// pre-processing graph turns it into CircleInput(512).
std::vector<uint8_t> CircleFrame(int width, int height) {
  std::vector<uint8_t> out(static_cast<size_t>(width) * height * 4);
  const int cx = width / 2;
  const int cy = height / 2;
  const int r = std::min(width, height) / 4 - 8;
  const int r2 = r * r;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int dx = x - cx;
      const int dy = y - cy;
      const uint8_t value = (dx * dx + dy * dy <= r2) ? 255 : 0;
      uint8_t* rgba = &out[(static_cast<size_t>(y) * width + x) * 4];
      rgba[0] = rgba[1] = rgba[2] = value;
      rgba[3] = 255;
    }
  }
  return out;
}

// A [3] fp32 constant; broadcasts over the channel axis of an NHWC tensor.
sam2::TfTensor PerChannel(float c0, float c1, float c2) {
  return sam2::TfTensor(
      {.type = Type::kFP32,
       .shape = {3},
       .buffer = OwningCpuBuffer::Copy<Type::kFP32>({c0, c1, c2})});
}

// Pre-processing graph: RGBA8888 frame [1,H,W,4] -> slice RGB -> fp32 ->
// bilinear resize to [1,S,S,3] -> ImageNet normalization, with
// (x / 255 - mean) / std folded into one per-channel multiply-add.
auto CreatePreprocessRunner(::litert::Environment& env,
                            ::litert::Options& options, int frame_height,
                            int frame_width, int size) {
  return CreateLambdaRunner(
      env, options,
      {{"frame", sam2::TfTensor({.name = "frame",
                                 .type = Type::kU8,
                                 .shape = {1, frame_height, frame_width, 4}})}},
      [=](const TensorsMap& inputs) {
        sam2::TfTensor rgb = Slice(inputs.at("frame"), {0, 0, 0, 0},
                                   {1, frame_height, frame_width, 3});
        sam2::TfTensor resized =
            ResizeBilinear(Cast(rgb, Type::kFP32), {size, size},
                           /*align_corners=*/false,
                           /*half_pixel_centers=*/true);
        const float* mean = kImageNetMean;
        const float* stddev = kImageNetStd;
        sam2::TfTensor scale =
            PerChannel(1.0f / (255.0f * stddev[0]), 1.0f / (255.0f * stddev[1]),
                       1.0f / (255.0f * stddev[2]));
        sam2::TfTensor bias = PerChannel(
            -mean[0] / stddev[0], -mean[1] / stddev[1], -mean[2] / stddev[2]);
        return TensorsMap{{"pixels", Add(Mul(resized, scale), bias)}};
      });
}

// Post-processing graph: decode_mask's [1,3,M,M] logits -> NHWC ->
// bilinear upsample to the frame size (half-pixel centers, i.e. the
// reference's F.interpolate(..., align_corners=False)) -> threshold at 0
// -> {0,1} fp32 masks [1,H,W,3].
auto CreatePostprocessRunner(::litert::Environment& env,
                             ::litert::Options& options, int mask_grid,
                             int frame_height, int frame_width) {
  return CreateLambdaRunner(
      env, options,
      {{"masks", sam2::TfTensor({.name = "masks",
                                 .type = Type::kFP32,
                                 .shape = {1, 3, mask_grid, mask_grid}})}},
      [=](const TensorsMap& inputs) {
        sam2::TfTensor nhwc = Transpose(inputs.at("masks"), {0, 2, 3, 1});
        sam2::TfTensor logits =
            ResizeBilinear(nhwc, {frame_height, frame_width},
                           /*align_corners=*/false,
                           /*half_pixel_centers=*/true);
        sam2::TfTensor zero(
            {.type = Type::kFP32,
             .shape = {},
             .buffer = OwningCpuBuffer::Copy<Type::kFP32>({0.0f})});
        return TensorsMap{
            {"frame_masks", Cast(Greater(logits, zero), Type::kFP32)}};
      });
}

double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

absl::Status CopyFloats(absl::StatusOr<::litert::tensor::TensorHandle> t,
                        std::vector<float>& out) {
  if (!t.ok()) return t.status();
  auto buffer = t->GetBuffer();
  if (!buffer.ok()) return buffer.status();
  auto lock = buffer->Lock();
  const float* data = reinterpret_cast<const float*>(lock.data());
  out.assign(data, data + lock.size() / sizeof(float));
  return absl::OkStatus();
}

absl::Status ReadFloats(LitertDynamicRunner& runner, const std::string& sig,
                        const std::string& name, std::vector<float>& out) {
  return CopyFloats(runner.GetOutput(sig, name), out);
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
  sam2::DecoderInputs dec_in = sam2::MakeDecoderInputs(config);
  sam2::DecoderOutputs dec_out = sam2::BuildDecoder(config, dec_in, weights);

  ModelFactory factory;
  {
    std::vector<::litert::tensor::TensorHandle> ins, outs;
    for (auto& t : enc_in.AsList()) ins.push_back(t);
    for (auto& [s, t] : enc_out.rbmm_params) ins.push_back(t);
    for (auto& t : enc_out.AsList()) outs.push_back(t);
    auto status = factory.AddSignature(ins, outs, "encode_image");
    if (!status.ok()) return status;
  }
  {
    std::vector<::litert::tensor::TensorHandle> ins, outs;
    for (auto& t : dec_in.AsList()) ins.push_back(t);
    for (auto& [s, t] : dec_out.rbmm_params) ins.push_back(t);
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
    ModelFactory enc_factory;
    std::vector<::litert::tensor::TensorHandle> ins, outs;
    for (auto& t : enc_in2.AsList()) ins.push_back(t);
    for (auto& [s, t] : enc_out2.rbmm_params) ins.push_back(t);
    for (auto& t : enc_out2.AsList()) outs.push_back(t);
    auto st1 = enc_factory.AddSignature(ins, outs, "encode_image");
    if (!st1.ok()) return st1;
    st1 = enc_factory.Save(split_dir + "/sam2_encoder_512.tflite");
    if (!st1.ok()) return st1;

    sam2::DecoderInputs dec_in2 = sam2::MakeDecoderInputs(config);
    sam2::DecoderOutputs dec_out2 = sam2::BuildDecoder(config, dec_in2, weights);
    ModelFactory dec_factory;
    ins.clear();
    outs.clear();
    for (auto& t : dec_in2.AsList()) ins.push_back(t);
    for (auto& [s, t] : dec_out2.rbmm_params) ins.push_back(t);
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

  // --- Pre/post-processing graphs (same Environment and accelerator) ---
  const int size = config.image_size;
  const int mg = config.mask_grid();
  const std::string input_file = absl::GetFlag(FLAGS_input_file);
  const std::string frame_file = absl::GetFlag(FLAGS_frame_file);
  // --input_file feeds normalized pixels directly; the frame is then the
  // SxS model input itself.
  const bool preprocess = input_file.empty();
  const int frame_w = preprocess ? absl::GetFlag(FLAGS_frame_width) : size;
  const int frame_h = preprocess ? absl::GetFlag(FLAGS_frame_height) : size;
  if (frame_w <= 0 || frame_h <= 0) {
    return absl::InvalidArgumentError("frame size must be positive");
  }
  auto pre_runner =
      CreatePreprocessRunner(*env, *options, frame_h, frame_w, size);
  auto post_runner =
      CreatePostprocessRunner(*env, *options, mg, frame_h, frame_w);

  // --- Stage input ---
  std::vector<float> pixels;
  if (preprocess) {
    std::vector<uint8_t> frame;
    if (frame_file.empty()) {
      frame = CircleFrame(frame_w, frame_h);
    } else {
      frame.resize(static_cast<size_t>(frame_w) * frame_h * 4);
      std::ifstream in(frame_file, std::ios::binary);
      if (!in) return absl::NotFoundError("frame_file: " + frame_file);
      in.read(reinterpret_cast<char*>(frame.data()), frame.size());
      if (static_cast<size_t>(in.gcount()) != frame.size()) {
        return absl::InvalidArgumentError("frame_file wrong size");
      }
    }
    auto st = pre_runner.SetInput(
        "frame", Create("frame", Type::kU8, {1, frame_h, frame_w, 4},
                        OwningCpuBuffer::Copy<Type::kU8>(frame)));
    if (!st.ok()) return st;
    // Zero-copy hand-off: pre-processing writes encode_image's input buffer.
    auto pixels_in = runner.GetInput("encode_image", "pixels");
    if (!pixels_in.ok()) return pixels_in.status();
    st = pre_runner.SetOutput("pixels", *pixels_in);
    if (!st.ok()) return st;
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

  // The prompt is in frame pixels; decode_mask takes it in SxS model space.
  float px = static_cast<float>(absl::GetFlag(FLAGS_point_x));
  float py = static_cast<float>(absl::GetFlag(FLAGS_point_y));
  if (px < 0) px = frame_w / 2.0f;
  if (py < 0) py = frame_h / 2.0f;
  px *= static_cast<float>(size) / frame_w;
  py *= static_cast<float>(size) / frame_h;
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

  // Zero-copy hand-off: post-processing reads decode_mask's output buffer.
  {
    auto masks_out = runner.GetOutput("decode_mask", "masks");
    if (!masks_out.ok()) return masks_out.status();
    auto st = post_runner.SetInput("masks", *masks_out);
    if (!st.ok()) return st;
  }

  // odml.runtime_bmm control inputs: seven int32 copies of the bound
  // length S per input (the proven attn_bench fill pattern; element 2 is
  // the one the kernels read). Set once — inputs persist across Run().
  auto set_rbmm_params = [&](const std::string& sig,
                             const sam2::RbmmParams& params) -> absl::Status {
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
    auto st = set_rbmm_params("encode_image", enc_out.rbmm_params);
    if (!st.ok()) return st;
    st = set_rbmm_params("decode_mask", dec_out.rbmm_params);
    if (!st.ok()) return st;
  }

  // --- Warmup + correctness pass ---
  auto st = preprocess ? pre_runner.Run() : set_pixels();
  if (!st.ok()) return st;
  st = runner.Run("encode_image");
  if (!st.ok()) return st;
  st = set_decoder_inputs();
  if (!st.ok()) return st;
  st = runner.Run("decode_mask");
  if (!st.ok()) return st;
  st = post_runner.Run();
  if (!st.ok()) return st;

  const int warmup = absl::GetFlag(FLAGS_warmup);
  const int runs = absl::GetFlag(FLAGS_runs);
  for (int i = 0; i < warmup; ++i) {
    if (preprocess) {
      st = pre_runner.Run();
      if (!st.ok()) return st;
    }
    st = runner.Run("encode_image");
    if (!st.ok()) return st;
    st = runner.Run("decode_mask");
    if (!st.ok()) return st;
    st = post_runner.Run();
    if (!st.ok()) return st;
  }

  std::vector<double> pre_ms, enc_ms, dec_ms, post_ms;
  for (int i = 0; preprocess && i < runs; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    st = pre_runner.Run();
    if (!st.ok()) return st;
    pre_ms.push_back(std::chrono::duration<double, std::milli>(
                         std::chrono::steady_clock::now() - t0)
                         .count());
  }
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
  for (int i = 0; i < runs; ++i) {
    auto t0 = std::chrono::steady_clock::now();
    st = post_runner.Run();
    if (!st.ok()) return st;
    post_ms.push_back(std::chrono::duration<double, std::milli>(
                          std::chrono::steady_clock::now() - t0)
                          .count());
  }

  // --- Outputs ---
  std::vector<float> masks, iou, obj, frame_masks;
  st = ReadFloats(runner, "decode_mask", "masks", masks);
  if (!st.ok()) return st;
  st = ReadFloats(runner, "decode_mask", "iou_scores", iou);
  if (!st.ok()) return st;
  st = ReadFloats(runner, "decode_mask", "object_score", obj);
  if (!st.ok()) return st;
  st = CopyFloats(post_runner.GetOutput("frame_masks"), frame_masks);
  if (!st.ok()) return st;
  if (preprocess) {
    // What encode_image consumed, for the check below and --dump_dir.
    st = CopyFloats(pre_runner.GetOutput("pixels"), pixels);
    if (!st.ok()) return st;
  }

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
  // Foreground of post-processed mask m at frame resolution (NHWC).
  auto frame_fg_count = [&](int m) {
    int fg = 0;
    for (size_t i = m; i < frame_masks.size(); i += 3) {
      if (frame_masks[i] > 0.5f) ++fg;
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
  std::cout << absl::StrCat(
      "pre/post-processing graphs (frame ", frame_w, "x", frame_h,
      "): pre_median=",
      preprocess ? absl::StrCat(Median(pre_ms), "ms") : "n/a (--input_file)",
      "  post_median=", absl::StrCat(Median(post_ms)), "ms\n",
      "frame-res best mask = [", best, "]: fg=", frame_fg_count(best), "/",
      frame_w * frame_h, "\n");

  // At the default frame the pre-processing graph must reproduce the
  // host-normalized fixture (the 512 -> 512 resize is an identity).
  if (preprocess && frame_file.empty() && frame_w == size && frame_h == size) {
    const std::vector<float> ref = CircleInput(size);
    if (ref.size() != pixels.size()) {
      return absl::InternalError("pre-processing output has the wrong size");
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
      max_diff = std::max(max_diff, std::fabs(pixels[i] - ref[i]));
    }
    std::cout << "pre-processing graph vs host CircleInput: max|diff|="
              << max_diff << std::endl;
  }

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
    st = DumpFile(dump_dir, "frame_masks", frame_masks);
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

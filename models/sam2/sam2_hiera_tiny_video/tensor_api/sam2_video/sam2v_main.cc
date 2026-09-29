// SAM2.1 hiera-tiny video tracking — build, serialize, run the host loop,
// verify, bench.
//
// Usage (from the LiteRT repo root; GPU runs need cwd =
// litert/prebuilt/macos_arm64 so the Metal accelerator dylib resolves):
//   bazel build --config=macos_arm64 //models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video:sam2v_main
//   sam2v_main --weights=.../sam2_tiny_1024_video.safetensors \
//     --frames_file=frames.f32 --frames=10 --dump_dir=out \
//     [--accelerator=gpu --gpu_precision=fp16 --gpu_buffer_storage=buffer]
//
// The host loop is the numpy specification in verify/verify_video_1024.py
// (`track`): per frame it only calls the four graphs and does the bank
// bookkeeping, best-mask pick, no-object handling and the mask_for_mem
// construction. --dump_dir writes per-frame outputs for the parity check
// against the HF streaming reference.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
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
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_config.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_graph.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_weights.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video/sam2v_graph.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video/sam2v_weights.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, weights, "",
          "sam2_tiny_1024_video.safetensors path (empty = synthetic weights "
          "for a shape/route check)");
ABSL_FLAG(std::string, tflite_path, "/tmp/sam2_video.tflite",
          "Where to write the serialized five-signature model");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu|gpu");
ABSL_FLAG(std::string, gpu_precision, "fp16",
          "fp16|fp32 — GPU calculation precision (fp16 is the delegate "
          "default; fp32 for CPU-parity verification)");
ABSL_FLAG(std::string, gpu_buffer_storage, "default",
          "default|buffer|texture2d");
ABSL_FLAG(int, image_size, 1024,
          "Model input side: 1024 (native), 512 or 384 (the smaller sizes need "
          "a weight file with the resolution-dependent tables re-derived)");
ABSL_FLAG(std::string, attention, "raw",
          "raw|sdpa — image-encoder attention as raw ops or the "
          "odml.scaled_dot_product_attention composite (fused on delegates "
          "that support it; identical math on CPU)");
ABSL_FLAG(std::string, norms, "raw",
          "raw|composite — image-path LayerNorms as raw ops or the "
          "odml.layer_norm composite");
ABSL_FLAG(int, nmm, 7, "Memory-bank slots used by the loop (7 or 2)");
ABSL_FLAG(int, frames, 10, "Frames to track");
ABSL_FLAG(std::string, frames_file, "",
          "Raw fp32 NHWC [T,1024,1024,3] ImageNet-normalized clip (empty = "
          "the synthetic white-disk clip is required for parity; a static "
          "circle fixture is used as fallback for bench-only runs)");
ABSL_FLAG(double, click_x, 400.0, "Frame-0 click x in image space");
ABSL_FLAG(double, click_y, 512.0, "Frame-0 click y in image space");
ABSL_FLAG(std::string, dump_dir, "",
          "If set, write per-frame mask/obj/ptr/mem/pix_feat raw fp32");
ABSL_FLAG(std::string, path, "classic",
          "classic|fused — classic runs memcond/decode/memorize with the host "
          "post-processing in between; fused runs one step_prompt / step{nmm} "
          "signature per frame (post-processing in-graph)");
ABSL_FLAG(int, bench_loops, 1,
          "Repeat the whole clip N times and report warm per-stage medians "
          "(first pass excluded when N > 1)");

namespace {

using ::litert::tensor::Create;
using ::litert::tensor::LitertDynamicRunner;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::Type;
namespace sam2 = ::litert::tensor::examples::sam2;
namespace s2v = ::litert::tensor::examples::sam2_video;

// Resolution-dependent sizes, set from --image_size at the start of Run().
int kSize = 1024;
int kGrid = 64;                           // top-level feature grid
int kHw = kGrid * kGrid;                  // 4096
constexpr int kHidden = 256;
constexpr int kMemCh = 64;
int kMaskGrid = 256;                      // low-res mask side
int kPlane = kMaskGrid * kMaskGrid;
constexpr int kNptrFrames = 16;
constexpr int kPtrSplit = 4;
constexpr int kNptr = kNptrFrames * kPtrSplit;  // 64
// Additive key-mask fill. -30000 is fp16-safe on the Metal delegate (the
// reference uses -1e9, which overflows fp16); for softmax both give exactly
// zero weight in fp32, so parity is unaffected.
constexpr float kMaskNeg = -30000.0f;
constexpr float kNoObjScore = -1024.0f;
constexpr int kMaxClicks = 8;  // step_prompt{2..8} signatures
constexpr float kMemScale = 20.0f;
constexpr float kMemBias = -10.0f;
constexpr float kTwoPi = 6.283185307179586f;

double Median(std::vector<double> v) {
  if (v.empty()) return 0.0;
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

// Host tables read once from the weight map.
struct HostConsts {
  std::vector<float> gaussian;      // (2,128) row-major
  std::vector<float> point_embed1;  // (256)
  std::vector<float> not_a_point;   // (256)
  std::vector<float> track_sparse;  // (2,256)
  std::vector<float> mtpe;          // (7,64)
  std::vector<float> no_obj_ptr;    // (256)
  std::vector<float> tpos_w;        // (64,256)
  std::vector<float> tpos_b;        // (64)

  // 2-token sparse prompt: one positive click + the not-a-point pad
  // (matches the reference host and the HF _embed_points math).
  std::vector<float> ClickSparse(float x, float y) const {
    float xn = 2.0f * ((x + 0.5f) / kSize) - 1.0f;
    float yn = 2.0f * ((y + 0.5f) / kSize) - 1.0f;
    std::vector<float> out(2 * kHidden);
    for (int i = 0; i < 128; ++i) {
      float proj = kTwoPi * (xn * gaussian[i] + yn * gaussian[128 + i]);
      out[i] = std::sin(proj) + point_embed1[i];
      out[128 + i] = std::cos(proj) + point_embed1[128 + i];
    }
    for (int i = 0; i < kHidden; ++i) out[kHidden + i] = not_a_point[i];
    return out;
  }

  // tpos_proj(get_1d_sine_pe(t_diff / 15, 256)) — double accumulation like
  // the reference (float32 rounding only at the end).
  std::vector<float> PtrPos(int t_diff) const {
    double pe[kHidden];
    const double pos = static_cast<double>(t_diff) / (kNptrFrames - 1.0);
    for (int i = 0; i < 128; ++i) {
      double dim_t = std::pow(10000.0, 2.0 * (i / 2) / 128.0);
      pe[i] = std::sin(pos / dim_t);
      pe[128 + i] = std::cos(pos / dim_t);
    }
    std::vector<float> out(kMemCh);
    for (int r = 0; r < kMemCh; ++r) {
      double acc = static_cast<double>(tpos_b[r]);
      for (int c = 0; c < kHidden; ++c) {
        acc += static_cast<double>(tpos_w[static_cast<size_t>(r) * kHidden + c]) *
               pe[c];
      }
      out[r] = static_cast<float>(acc);
    }
    return out;
  }
};

std::vector<float> HostFloatsOf(const s2v::WeightMap& weights,
                                const std::string& name) {
  auto it = weights.find(name);
  if (it == weights.end()) return {};
  auto buffer = it->second.GetBuffer();
  if (!buffer.ok()) return {};
  auto lock = buffer->Lock();
  const float* data = reinterpret_cast<const float*>(lock.data());
  return std::vector<float>(data, data + lock.size() / sizeof(float));
}

// Bilinear kMaskGrid -> kSize (4x), align_corners=false (torch semantics).
void Upsample1024(const float* low, std::vector<float>& high) {
  const float scale = static_cast<float>(kMaskGrid) / kSize;  // 0.25
  for (int y = 0; y < kSize; ++y) {
    float sy = (y + 0.5f) * scale - 0.5f;
    int y0 = static_cast<int>(std::floor(sy));
    float fy = sy - y0;
    int y0c = std::clamp(y0, 0, kMaskGrid - 1);
    int y1c = std::clamp(y0 + 1, 0, kMaskGrid - 1);
    for (int x = 0; x < kSize; ++x) {
      float sx = (x + 0.5f) * scale - 0.5f;
      int x0 = static_cast<int>(std::floor(sx));
      float fx = sx - x0;
      int x0c = std::clamp(x0, 0, kMaskGrid - 1);
      int x1c = std::clamp(x0 + 1, 0, kMaskGrid - 1);
      float v00 = low[y0c * kMaskGrid + x0c];
      float v01 = low[y0c * kMaskGrid + x1c];
      float v10 = low[y1c * kMaskGrid + x0c];
      float v11 = low[y1c * kMaskGrid + x1c];
      high[static_cast<size_t>(y) * kSize + x] =
          (1 - fy) * ((1 - fx) * v00 + fx * v01) +
          fy * ((1 - fx) * v10 + fx * v11);
    }
  }
}

// Static circle fixture (bench fallback when no clip file is given).
std::vector<float> CircleFrame() {
  const float mean[3] = {0.485f, 0.456f, 0.406f};
  const float stddev[3] = {0.229f, 0.224f, 0.225f};
  std::vector<float> out(static_cast<size_t>(kSize) * kSize * 3);
  const int cx = kSize / 2, cy = kSize / 2, r = kSize / 4 - 8;
  for (int y = 0; y < kSize; ++y) {
    for (int x = 0; x < kSize; ++x) {
      const int dx = x - cx, dy = y - cy;
      const float value = (dx * dx + dy * dy <= r * r) ? 1.0f : 0.0f;
      size_t base = (static_cast<size_t>(y) * kSize + x) * 3;
      for (int ch = 0; ch < 3; ++ch) {
        out[base + ch] = (value - mean[ch]) / stddev[ch];
      }
    }
  }
  return out;
}

struct StageTimes {
  std::vector<double> encode, memcond, decode, memorize, host, step, e2e;
};

absl::Status Run() {
  const int nmm = absl::GetFlag(FLAGS_nmm);
  if (nmm != 7 && nmm != 2) {
    return absl::InvalidArgumentError("--nmm must be 7 or 2");
  }
  const int T = absl::GetFlag(FLAGS_frames);
  kSize = absl::GetFlag(FLAGS_image_size);
  if (kSize != 1024 && kSize != 512 && kSize != 384) {
    return absl::InvalidArgumentError("--image_size must be 1024, 512 or 384");
  }
  kGrid = kSize / 16;
  kHw = kGrid * kGrid;
  kMaskGrid = kSize / 4;
  kPlane = kMaskGrid * kMaskGrid;

  s2v::Sam2VideoConfig config;
  sam2::Sam2Config& img = config.image;
  img.image_size = kSize;
  const std::string attention = absl::GetFlag(FLAGS_attention);
  const std::string norms = absl::GetFlag(FLAGS_norms);
  if (attention != "raw" && attention != "sdpa") {
    return absl::InvalidArgumentError("--attention must be raw|sdpa");
  }
  if (norms != "raw" && norms != "composite") {
    return absl::InvalidArgumentError("--norms must be raw|composite");
  }
  img.use_sdpa_composite = attention == "sdpa";
  img.use_layer_norm_composite = norms == "composite";
  // The encoder emits the raw top-level feature map; the video decoder adds
  // the no-memory row itself, on the conditioning frame, via its nomem input.
  img.fold_no_mem_embed = false;

  // ---- Weights: image keys + video keys from the same file. ----
  s2v::WeightMap weights;
  const std::string weights_path = absl::GetFlag(FLAGS_weights);
  if (weights_path.empty()) {
    weights = sam2::MakeSyntheticWeights(img, /*seed=*/42);
    s2v::MakeSyntheticVideoWeights(config, /*seed=*/43, weights);
    std::cout << "weights: synthetic — shape/route check only" << std::endl;
  } else {
    auto weights_or = sam2::LoadCheckpointWeights(img, weights_path);
    if (!weights_or.ok()) return weights_or.status();
    weights = std::move(*weights_or);
    auto st = s2v::LoadVideoWeights(config, weights_path, weights);
    if (!st.ok()) return st;
    std::cout << "weights: " << weights_path << " (" << weights.size()
              << " tensors)" << std::endl;
  }

  HostConsts consts;
  consts.gaussian = HostFloatsOf(
      weights,
      "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix");
  consts.point_embed1 =
      HostFloatsOf(weights, "sam_prompt_encoder.point_embeddings.1.weight");
  consts.not_a_point =
      HostFloatsOf(weights, "sam_prompt_encoder.not_a_point_embed.weight");
  consts.track_sparse = HostFloatsOf(weights, "tables.track_sparse");
  consts.mtpe = HostFloatsOf(weights, "maskmem_tpos_enc");  // (7,1,1,64)
  consts.no_obj_ptr = HostFloatsOf(weights, "no_obj_ptr");
  consts.tpos_w = HostFloatsOf(weights, "obj_ptr_tpos_proj.weight");
  consts.tpos_b = HostFloatsOf(weights, "obj_ptr_tpos_proj.bias");

  // ---- Build the five signatures (shared weights, one flatbuffer). ----
  // One cache for the whole model: signatures share each baked constant.
  s2v::ConstCache cache;
  sam2::EncoderInputs enc_in = sam2::MakeEncoderInputs(img);
  sam2::EncoderOutputs enc_out = sam2::BuildEncoder(img, enc_in, weights);
  enc_out.image_embeddings.SetName("pix_raw");

  s2v::MemCondInputs mc7_in = s2v::MakeMemCondInputs(config, 7);
  s2v::TfTensor mc7_out = s2v::BuildMemCond(config, 7, mc7_in, weights, &cache);
  s2v::MemCondInputs mc2_in = s2v::MakeMemCondInputs(config, 2);
  s2v::TfTensor mc2_out = s2v::BuildMemCond(config, 2, mc2_in, weights, &cache);
  s2v::VideoDecoderInputs dec_in = s2v::MakeVideoDecoderInputs(config);
  s2v::VideoDecoderOutputs dec_out =
      s2v::BuildVideoDecoder(config, dec_in, weights, /*mask_mode=*/0,
                             &cache);
  s2v::MemorizeInputs mem_in = s2v::MakeMemorizeInputs(config);
  s2v::TfTensor mem_out = s2v::BuildMemorize(config, mem_in, weights, &cache);

  // Fused per-object steps (decode + in-graph host post + memorize, and for
  // tracked frames memory attention in front): no host round trip inside a
  // frame. The tracking sparse row is baked; nomem stays an input (see
  // BuildStep).
  auto const_f32 = [](std::vector<float> v, std::vector<int> shape) {
    return s2v::TfTensor({.type = Type::kFP32,
                          .shape = std::move(shape),
                          .buffer = ::litert::tensor::OwningCpuBuffer::Copy<
                              Type::kFP32>(v)});
  };
  s2v::VideoDecoderInputs sp_in = s2v::MakeVideoDecoderInputs(config);
  s2v::TfTensor sp_raw({.name = "pix_raw",
                        .type = Type::kFP32,
                        .shape = {1, img.embed_grid(), img.embed_grid(),
                                  img.d_model}});
  sp_in.pix_feat = sp_raw;
  s2v::StepOutputs sp_out =
      s2v::BuildStep(config, sp_in, sp_raw, weights, /*multimask=*/true,
                     /*binarize_mode=*/1, &cache);
  struct TrackStep {
    s2v::MemCondInputs mc;
    s2v::VideoDecoderInputs dec;
    s2v::StepOutputs out;
  };
  auto build_track_step = [&](int n) {
    s2v::MemCondInputs mc = s2v::MakeMemCondInputs(config, n);
    s2v::VideoDecoderInputs dec = s2v::MakeVideoDecoderInputs(config);
    dec.pix_feat = s2v::BuildMemCond(config, n, mc, weights, &cache);
    dec.sparse = const_f32(consts.track_sparse, {1, 2, kHidden});
    s2v::StepOutputs out =
        s2v::BuildStep(config, dec, mc.pix_raw, weights, /*multimask=*/true,
                       /*binarize_mode=*/0, &cache);
    return TrackStep{mc, dec, out};
  };
  TrackStep st7 = build_track_step(7);
  TrackStep st2 = build_track_step(2);

  ModelFactory factory;
  auto add_sig = [&](std::vector<s2v::TfTensor> ins,
                     std::vector<s2v::TfTensor> outs,
                     const std::string& name) -> absl::Status {
    std::vector<::litert::tensor::TensorHandle> in_handles, out_handles;
    for (auto& t : ins) in_handles.push_back(t);
    for (auto& t : outs) out_handles.push_back(t);
    return factory.AddSignature(in_handles, out_handles, name);
  };
  auto st = add_sig(enc_in.AsList(), enc_out.AsList(), "encode");
  if (!st.ok()) return st;
  st = add_sig(mc7_in.AsList(), {mc7_out}, "memcond7");
  if (!st.ok()) return st;
  st = add_sig(mc2_in.AsList(), {mc2_out}, "memcond2");
  if (!st.ok()) return st;
  st = add_sig(dec_in.AsList(), dec_out.AsList(), "decode");
  if (!st.ok()) return st;
  st = add_sig(mem_in.AsList(), {mem_out}, "memorize");
  if (!st.ok()) return st;
  st = add_sig({sp_raw, sp_in.feat_s1, sp_in.feat_s0, sp_in.sparse, sp_in.nomem},
               sp_out.AsList(), "step_prompt");
  if (!st.ok()) return st;
  // Multi-click prompts: step_prompt{k} takes k positive/negative points +
  // the not-a-point pad (sparse [1, k+1, 256]) — SAM's exact encoding for k
  // clicks, one signature per count (padding extra slots would not be
  // neutral in the decoder's token attention).
  for (int k = 2; k <= kMaxClicks; ++k) {
    s2v::VideoDecoderInputs in = s2v::MakeVideoDecoderInputs(config);
    s2v::TfTensor raw({.name = "pix_raw",
                       .type = Type::kFP32,
                       .shape = {1, img.embed_grid(), img.embed_grid(),
                                 img.d_model}});
    in.pix_feat = raw;
    in.sparse = s2v::TfTensor({.name = "sparse",
                               .type = Type::kFP32,
                               .shape = {1, k + 1, kHidden}});
    s2v::StepOutputs out =
        s2v::BuildStep(config, in, raw, weights, /*multimask=*/false,
                       /*binarize_mode=*/1, &cache);
    st = add_sig({raw, in.feat_s1, in.feat_s0, in.sparse, in.nomem},
                 out.AsList(), absl::StrCat("step_prompt", k));
    if (!st.ok()) return st;
  }
  for (auto* ts : {&st7, &st2}) {
    std::vector<s2v::TfTensor> ins = ts->mc.AsList();
    ins.push_back(ts->dec.feat_s1);
    ins.push_back(ts->dec.feat_s0);
    ins.push_back(ts->dec.nomem);
    st = add_sig(ins, ts->out.AsList(), ts == &st7 ? "step7" : "step2");
    if (!st.ok()) return st;
  }

  const std::string tflite_path = absl::GetFlag(FLAGS_tflite_path);
  auto save_status = factory.Save(tflite_path);
  if (!save_status.ok()) return save_status;
  std::cout << "Serialized: " << tflite_path << std::endl;

  // ---- Runner. ----
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

  // ---- Clip. ----
  const size_t frame_elems = static_cast<size_t>(kSize) * kSize * 3;
  std::vector<float> clip;
  const std::string frames_file = absl::GetFlag(FLAGS_frames_file);
  if (!frames_file.empty()) {
    clip.resize(frame_elems * T);
    std::ifstream in(frames_file, std::ios::binary);
    if (!in) return absl::NotFoundError("frames_file: " + frames_file);
    in.read(reinterpret_cast<char*>(clip.data()),
            clip.size() * sizeof(float));
    if (static_cast<size_t>(in.gcount()) != clip.size() * sizeof(float)) {
      return absl::InvalidArgumentError(
          "frames_file wrong size (need T*S*S*3 fp32, S = --image_size)");
    }
  } else {
    std::cout << "frames_file empty: static circle fixture (bench only, no "
                 "parity)" << std::endl;
    std::vector<float> f = CircleFrame();
    clip.resize(frame_elems * T);
    for (int t = 0; t < T; ++t) {
      std::copy(f.begin(), f.end(), clip.begin() + frame_elems * t);
    }
  }

  const std::string dump_dir = absl::GetFlag(FLAGS_dump_dir);
  const std::string memcond_sig = absl::StrCat("memcond", nmm);
  const std::string path = absl::GetFlag(FLAGS_path);
  if (path != "classic" && path != "fused") {
    return absl::InvalidArgumentError("--path must be classic|fused");
  }
  const bool fused = path == "fused";
  // The fixed-bank inputs feed memcond{N} (classic) or step{N} (fused).
  const std::string bank_sig = fused ? absl::StrCat("step", nmm) : memcond_sig;
  const int mem_len = nmm * kHw + kNptr;
  const int bench_loops = absl::GetFlag(FLAGS_bench_loops);

  StageTimes times;
  auto tick = [] { return std::chrono::steady_clock::now(); };
  auto ms = [](auto a, auto b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  for (int loop = 0; loop < bench_loops; ++loop) {
    const bool record = bench_loops == 1 || loop > 0;
    std::map<int, std::vector<float>> spatial_bank;  // t -> (4096,64)
    std::map<int, std::vector<float>> ptr_bank;      // t -> (256)
    const int cond_frame = 0;

    for (int t = 0; t < T; ++t) {
      auto t_start = tick();
      // 1. encode
      auto st2 = runner.SetInput(
          "encode", "pixels",
          Create("pixels", Type::kFP32, {1, kSize, kSize, 3},
                 std::vector<float>(clip.begin() + frame_elems * t,
                                    clip.begin() + frame_elems * (t + 1))));
      if (!st2.ok()) return st2;
      auto t0 = tick();
      st2 = runner.Run("encode");
      if (!st2.ok()) return st2;
      auto t1 = tick();
      if (record) times.encode.push_back(ms(t0, t1));

      const bool prompted = t == cond_frame;
      std::vector<float> pix_feat;  // token-major (4096,256) when tracked
      double memcond_ms = 0.0;
      if (!prompted) {
        // ---- assemble the fixed bank (reference `track` lines) ----
        std::vector<float> mem(static_cast<size_t>(nmm) * kHw * kMemCh, 0.0f);
        std::vector<float> tpe(static_cast<size_t>(nmm) * kMemCh, 0.0f);
        std::vector<float> km(mem_len, kMaskNeg);
        int slot = 0;
        auto put_slot = [&](const std::vector<float>& m, const float* row) {
          std::copy(m.begin(), m.end(),
                    mem.begin() + static_cast<size_t>(slot) * kHw * kMemCh);
          std::copy(row, row + kMemCh,
                    tpe.begin() + static_cast<size_t>(slot) * kMemCh);
          std::fill(km.begin() + static_cast<size_t>(slot) * kHw,
                    km.begin() + static_cast<size_t>(slot + 1) * kHw, 0.0f);
          ++slot;
        };
        put_slot(spatial_bank[cond_frame],
                 consts.mtpe.data() + 6 * kMemCh);  // cond frame row
        for (int off = nmm - 1; off >= 1; --off) {  // most distant first
          int pf = t - off;
          auto it = spatial_bank.find(pf);
          if (it != spatial_bank.end() && pf != cond_frame) {
            put_slot(it->second, consts.mtpe.data() + (off - 1) * kMemCh);
          }
        }
        std::vector<float> ptr_tok(static_cast<size_t>(kNptr) * kMemCh, 0.0f);
        std::vector<float> ptr_pos(static_cast<size_t>(kNptr) * kMemCh, 0.0f);
        std::vector<std::pair<int, const std::vector<float>*>> ptrs;
        ptrs.emplace_back(t - cond_frame, &ptr_bank[cond_frame]);
        for (int td = 1; td < kNptrFrames; ++td) {
          int pf = t - td;
          if (pf < 0) break;
          auto it = ptr_bank.find(pf);
          if (it != ptr_bank.end() && pf != cond_frame) {
            ptrs.emplace_back(td, &it->second);
          }
        }
        for (size_t i = 0; i < ptrs.size(); ++i) {
          std::vector<float> pos = consts.PtrPos(ptrs[i].first);
          const std::vector<float>& p = *ptrs[i].second;
          for (int j = 0; j < kPtrSplit; ++j) {
            size_t row = i * kPtrSplit + j;
            std::copy(p.begin() + j * kMemCh, p.begin() + (j + 1) * kMemCh,
                      ptr_tok.begin() + row * kMemCh);
            std::copy(pos.begin(), pos.end(), ptr_pos.begin() + row * kMemCh);
            km[static_cast<size_t>(nmm) * kHw + row] = 0.0f;
          }
        }

        auto rebind = runner.GetOutput("encode", "pix_raw");
        if (!rebind.ok()) return rebind.status();
        st2 = runner.SetInput(bank_sig, "pix_raw", *rebind);
        if (!st2.ok()) return st2;
        st2 = runner.SetInput(bank_sig, "mem_bank",
                              Create("mem_bank", Type::kFP32,
                                     {1, nmm, kHw, kMemCh}, std::move(mem)));
        if (!st2.ok()) return st2;
        st2 = runner.SetInput(bank_sig, "slot_tpe",
                              Create("slot_tpe", Type::kFP32,
                                     {1, nmm, 1, kMemCh}, std::move(tpe)));
        if (!st2.ok()) return st2;
        st2 = runner.SetInput(bank_sig, "ptr_tok",
                              Create("ptr_tok", Type::kFP32,
                                     {1, 1, kNptr, kMemCh},
                                     std::move(ptr_tok)));
        if (!st2.ok()) return st2;
        st2 = runner.SetInput(bank_sig, "ptr_pos",
                              Create("ptr_pos", Type::kFP32,
                                     {1, 1, kNptr, kMemCh},
                                     std::move(ptr_pos)));
        if (!st2.ok()) return st2;
        st2 = runner.SetInput(bank_sig, "key_mask",
                              Create("key_mask", Type::kFP32,
                                     {1, 1, 1, mem_len}, std::move(km)));
        if (!st2.ok()) return st2;
        if (!fused) {
          auto m0 = tick();
          st2 = runner.Run(memcond_sig);
          if (!st2.ok()) return st2;
          memcond_ms = ms(m0, tick());
          if (record) times.memcond.push_back(memcond_ms);
          st2 = ReadFloats(runner, memcond_sig, "pix_feat", pix_feat);
          if (!st2.ok()) return st2;
        }
      }

      if (fused) {
        // One signature: [memory attention] + decode + in-graph post +
        // memorize. The bank inputs (tracked frames) are already set above.
        const std::string sig = prompted ? "step_prompt" : bank_sig;
        auto pix = runner.GetOutput("encode", "pix_raw");
        if (!pix.ok()) return pix.status();
        if (prompted) {
          st2 = runner.SetInput(sig, "pix_raw", *pix);
          if (!st2.ok()) return st2;
          st2 = runner.SetInput(
              sig, "sparse",
              Create("sparse", Type::kFP32, {1, 2, kHidden},
                     consts.ClickSparse(
                         static_cast<float>(absl::GetFlag(FLAGS_click_x)),
                         static_cast<float>(absl::GetFlag(FLAGS_click_y)))));
          if (!st2.ok()) return st2;
        }
        for (const std::string& name :
             {std::string("feat_s1"), std::string("feat_s0")}) {
          auto rebind = runner.GetOutput("encode", name);
          if (!rebind.ok()) return rebind.status();
          st2 = runner.SetInput(sig, name, *rebind);
          if (!st2.ok()) return st2;
        }
        st2 = runner.SetInput(sig, "nomem",
                              Create("nomem", Type::kFP32, {1, 1, 1, 1},
                                     std::vector<float>{prompted ? 1.0f : 0.0f}));
        if (!st2.ok()) return st2;
        auto s0 = tick();
        st2 = runner.Run(sig);
        if (!st2.ok()) return st2;
        std::vector<float> low, obj, iou, mem_t, ptr_t;
        for (auto [name, out] :
             {std::pair<const char*, std::vector<float>*>{"low_mask", &low},
              {"object_score", &obj}, {"iou", &iou}, {"mem", &mem_t},
              {"ptr", &ptr_t}}) {
          st2 = ReadFloats(runner, sig, name, *out);
          if (!st2.ok()) return st2;
        }
        if (record) times.step.push_back(ms(s0, tick()));
        spatial_bank[t] = std::move(mem_t);
        ptr_bank[t] = std::move(ptr_t);
        if (record) times.e2e.push_back(ms(t_start, tick()));
        int fg = 0;
        for (float v : low) fg += v > 0.0f ? 1 : 0;
        std::cout << absl::StrCat("frame ", t, ": fg=", fg, " obj=", obj[0],
                                  " iou_best=", iou[0],
                                  prompted ? " (prompted)" : "", " [fused]")
                  << std::endl;
        if (!dump_dir.empty() && loop == 0) {
          std::string p = absl::StrCat("f", t < 10 ? "0" : "", t);
          for (auto [suffix, data] :
               {std::pair<const char*, const std::vector<float>*>{"_mask", &low},
                {"_ptr", &ptr_bank[t]}, {"_mem", &spatial_bank[t]}}) {
            st2 = DumpFile(dump_dir, p + suffix, *data);
            if (!st2.ok()) return st2;
          }
          st2 = DumpFile(dump_dir, p + "_obj", {obj[0]});
          if (!st2.ok()) return st2;
        }
        continue;
      }

      // 2. decode
      if (prompted) {
        auto rebind = runner.GetOutput("encode", "pix_raw");
        if (!rebind.ok()) return rebind.status();
        st2 = runner.SetInput("decode", "pix_feat_in", *rebind);
      } else {
        auto rebind = runner.GetOutput(memcond_sig, "pix_feat");
        if (!rebind.ok()) return rebind.status();
        st2 = runner.SetInput("decode", "pix_feat_in", *rebind);
      }
      if (!st2.ok()) return st2;
      for (const std::string& name :
           {std::string("feat_s1"), std::string("feat_s0")}) {
        auto rebind = runner.GetOutput("encode", name);
        if (!rebind.ok()) return rebind.status();
        st2 = runner.SetInput("decode", name, *rebind);
        if (!st2.ok()) return st2;
      }
      std::vector<float> sparse =
          prompted ? consts.ClickSparse(
                         static_cast<float>(absl::GetFlag(FLAGS_click_x)),
                         static_cast<float>(absl::GetFlag(FLAGS_click_y)))
                   : consts.track_sparse;
      st2 = runner.SetInput("decode", "sparse",
                            Create("sparse", Type::kFP32, {1, 2, kHidden},
                                   std::move(sparse)));
      if (!st2.ok()) return st2;
      st2 = runner.SetInput("decode", "nomem",
                            Create("nomem", Type::kFP32, {1, 1, 1, 1},
                                   std::vector<float>{prompted ? 1.0f : 0.0f}));
      if (!st2.ok()) return st2;
      auto d0 = tick();
      st2 = runner.Run("decode");
      if (!st2.ok()) return st2;
      if (record) times.decode.push_back(ms(d0, tick()));

      std::vector<float> masks, iou, ptr, obj;
      st2 = ReadFloats(runner, "decode", "masks", masks);
      if (!st2.ok()) return st2;
      st2 = ReadFloats(runner, "decode", "iou_scores", iou);
      if (!st2.ok()) return st2;
      st2 = ReadFloats(runner, "decode", "obj_ptr", ptr);
      if (!st2.ok()) return st2;
      st2 = ReadFloats(runner, "decode", "object_score", obj);
      if (!st2.ok()) return st2;

      // 3. host post: best of tokens 1..3, no-object handling, mask_for_mem.
      auto h0 = tick();
      int best = 1;
      for (int j = 2; j < 4; ++j) {
        if (iou[j] > iou[best]) best = j;
      }
      const bool appearing = obj[0] > 0.0f;
      std::vector<float> low(kPlane, kNoObjScore);
      if (appearing) {
        low.assign(masks.begin() + static_cast<size_t>(best) * kPlane,
                   masks.begin() + static_cast<size_t>(best + 1) * kPlane);
      }
      std::vector<float> obj_ptr =
          appearing ? std::vector<float>(
                          ptr.begin() + static_cast<size_t>(best) * kHidden,
                          ptr.begin() + static_cast<size_t>(best + 1) * kHidden)
                    : consts.no_obj_ptr;
      std::vector<float> high(static_cast<size_t>(kSize) * kSize);
      Upsample1024(low.data(), high);
      std::vector<float> mfm(high.size());
      if (prompted) {
        for (size_t i = 0; i < high.size(); ++i) {
          mfm[i] = (high[i] > 0.0f ? kMemScale : 0.0f) + kMemBias;
        }
      } else {
        for (size_t i = 0; i < high.size(); ++i) {
          mfm[i] = kMemScale / (1.0f + std::exp(-high[i])) + kMemBias;
        }
      }
      if (record) times.host.push_back(ms(h0, tick()));

      // 4. memorize
      auto rebind = runner.GetOutput("encode", "pix_raw");
      if (!rebind.ok()) return rebind.status();
      st2 = runner.SetInput("memorize", "pix_raw", *rebind);
      if (!st2.ok()) return st2;
      st2 = runner.SetInput("memorize", "mask_for_mem",
                            Create("mask_for_mem", Type::kFP32,
                                   {1, kSize, kSize, 1}, std::move(mfm)));
      if (!st2.ok()) return st2;
      st2 = runner.SetInput("memorize", "occ",
                            Create("occ", Type::kFP32, {1, 1, 1},
                                   std::vector<float>{appearing ? 0.0f : 1.0f}));
      if (!st2.ok()) return st2;
      auto z0 = tick();
      st2 = runner.Run("memorize");
      if (!st2.ok()) return st2;
      if (record) times.memorize.push_back(ms(z0, tick()));
      std::vector<float> mem_t;
      st2 = ReadFloats(runner, "memorize", "mem", mem_t);
      if (!st2.ok()) return st2;

      spatial_bank[t] = std::move(mem_t);
      ptr_bank[t] = obj_ptr;
      if (record) times.e2e.push_back(ms(t_start, tick()));

      int fg = 0;
      for (float v : low) fg += v > 0.0f ? 1 : 0;
      std::cout << absl::StrCat("frame ", t, ": fg=", fg, " obj=", obj[0],
                                " best=", best, " iou_best=", iou[best],
                                prompted ? " (prompted)" : "",
                                !prompted && bench_loops == 1
                                    ? absl::StrCat(" memcond_ms=", memcond_ms)
                                    : "")
                << std::endl;

      if (!dump_dir.empty() && loop == 0) {
        std::string p = absl::StrCat("f", t < 10 ? "0" : "", t);
        st2 = DumpFile(dump_dir, p + "_mask", low);
        if (!st2.ok()) return st2;
        st2 = DumpFile(dump_dir, p + "_obj", {obj[0]});
        if (!st2.ok()) return st2;
        st2 = DumpFile(dump_dir, p + "_ptr", ptr_bank[t]);
        if (!st2.ok()) return st2;
        st2 = DumpFile(dump_dir, p + "_mem", spatial_bank[t]);
        if (!st2.ok()) return st2;
        if (!prompted) {
          st2 = DumpFile(dump_dir, p + "_pixfeat", pix_feat);
          if (!st2.ok()) return st2;
        }
      }
    }
  }

  std::cout << absl::StrCat(
      "SAM2 video · LiteRT Tensor API (",
      use_gpu ? absl::StrCat("gpu ", absl::GetFlag(FLAGS_gpu_precision))
              : "cpu fp32",
      ", ", kSize, "x", kSize, ", nmm=", nmm, ", T=", T, ", loops=", bench_loops, ")\n",
      fused ? absl::StrCat("path=fused, medians ms: encode=", Median(times.encode),
                           " step=", Median(times.step))
            : absl::StrCat("path=classic, medians ms: encode=", Median(times.encode),
                           " memcond=", Median(times.memcond),
                           " decode=", Median(times.decode),
                           " memorize=", Median(times.memorize),
                           " host=", Median(times.host)),
      " e2e/frame=", Median(times.e2e), "\n");
  if (!dump_dir.empty()) {
    std::cout << "dumped per-frame outputs to " << dump_dir << std::endl;
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

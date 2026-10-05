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

// Native end-to-end run of the SAM 2 ModelChain pipeline.
//
// 1. Authors the SAM 2 model (encode, prompt1..8, track2/7) from the weights
//    with the Tensor API and serializes it (or loads a previous build).
// 2. Runs Sam2Pipeline over a raw RGBA clip: preprocess -> encode ->
//    prompt / track per object -> composite, every stage a ModelChain stage.
// 3. Dumps what the verifiers compare (tools/verify_chain.py): the
//    preprocessed frames, every object's low-res mask / scores per frame, and
//    composites.
//
//   sam2_chain_main --weights=.../sam2_tiny_384_video.safetensors \
//     --image_size=384 --host_consts=.../sam2_host_consts.safetensors \
//     --sam2_tflite=/tmp/sam2_chain_384.tflite --frames_rgba=clip.rgba \
//     --width=640 --height=360 --frames=24 \
//     --prompts='0@0:0.44,0.28,1;0.45,0.47,1|1@0:0.484,0.79,1' \
//     --dump_dir=/tmp/chain_dump

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "chain_graphs.h"
#include "host_plan.h"
#include "litert/cc/litert_compiled_model.h"
#include "litert/cc/litert_environment.h"
#include "litert/cc/litert_options.h"
#include "litert/cc/options/litert_gpu_options.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_weights.h"
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video/sam2v_weights.h"
#include "sam2_pipeline.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"

ABSL_FLAG(std::string, weights, "", "sam2_tiny_{S}_video.safetensors");
ABSL_FLAG(int, image_size, 384, "Model input side (384 | 512 | 1024)");
ABSL_FLAG(std::string, host_consts, "", "sam2_host_consts.safetensors");
ABSL_FLAG(std::string, sam2_tflite, "/tmp/sam2_chain.tflite",
          "Where the SAM 2 model is serialized");
ABSL_FLAG(bool, reuse_tflite, false,
          "Load --sam2_tflite if it exists instead of rebuilding it");
ABSL_FLAG(bool, build_only, false, "Serialize the SAM 2 model and exit");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu | gpu");
ABSL_FLAG(std::string, gpu_precision, "fp16", "fp16 | fp32");
ABSL_FLAG(std::string, frames_rgba, "", "Raw uint8 RGBA clip [T, H, W, 4]");
ABSL_FLAG(int, width, 0, "Clip width");
ABSL_FLAG(int, height, 0, "Clip height");
ABSL_FLAG(int, frames, 0, "Frames to process (0 = all in the file)");
ABSL_FLAG(std::string, prompts, "",
          "obj@frame:x,y,label;x,y,label|obj@frame:... — normalized frame "
          "coordinates, label 1 positive / 0 negative");
ABSL_FLAG(int, nmm, 7, "Memory slots: 2 | 7");
ABSL_FLAG(std::string, effect, "overlay", "overlay | spotlight | cutout");
ABSL_FLAG(double, stroke, 3.0, "Outline width, display px");
ABSL_FLAG(std::string, dump_dir, "", "Dump directory (empty = no dumps)");
ABSL_FLAG(std::string, dump_rgb, "0",
          "Frames whose composite is dumped (comma list, 'all')");

namespace {

using namespace sam2_chain;  // NOLINT
namespace sam2 = ::litert::tensor::examples::sam2;
using Clock = std::chrono::steady_clock;

struct Prompt {
  int object = 0, frame = 0;
  std::vector<std::array<float, 3>> points;  // normalized x, y, label
};

absl::StatusOr<std::vector<Prompt>> ParsePrompts(const std::string& spec) {
  std::vector<Prompt> out;
  if (spec.empty()) return out;
  for (absl::string_view item : absl::StrSplit(spec, '|')) {
    std::vector<std::string> head = absl::StrSplit(item, ':');
    std::vector<std::string> of = absl::StrSplit(head.at(0), '@');
    if (head.size() != 2 || of.size() != 2) {
      return absl::InvalidArgumentError(absl::StrCat("bad prompt: ", item));
    }
    Prompt p;
    if (!absl::SimpleAtoi(of[0], &p.object) ||
        !absl::SimpleAtoi(of[1], &p.frame)) {
      return absl::InvalidArgumentError(absl::StrCat("bad prompt: ", item));
    }
    for (absl::string_view pt : absl::StrSplit(head[1], ';')) {
      std::vector<std::string> v = absl::StrSplit(pt, ',');
      std::array<float, 3> xyz{};
      if (v.size() != 3 || !absl::SimpleAtof(v[0], &xyz[0]) ||
          !absl::SimpleAtof(v[1], &xyz[1]) || !absl::SimpleAtof(v[2], &xyz[2])) {
        return absl::InvalidArgumentError(absl::StrCat("bad point: ", pt));
      }
      p.points.push_back(xyz);
    }
    out.push_back(std::move(p));
  }
  return out;
}

absl::Status Dump(const std::string& path, const std::vector<float>& v) {
  std::ofstream f(path, std::ios::binary);
  if (!f) return absl::InternalError(absl::StrCat("cannot write ", path));
  f.write(reinterpret_cast<const char*>(v.data()), v.size() * sizeof(float));
  return absl::OkStatus();
}

absl::StatusOr<litert::Options> MakeOptions() {
  auto options = litert::Options::Create();
  if (!options) return absl::InternalError("Options::Create failed");
  const bool gpu = absl::GetFlag(FLAGS_accelerator) == "gpu";
  options->SetHardwareAccelerators(gpu ? litert::HwAccelerators::kGpu
                                       : litert::HwAccelerators::kCpu);
  if (gpu && absl::GetFlag(FLAGS_gpu_precision) == "fp32") {
    auto gpu_options = options->GetGpuOptions();
    if (gpu_options) {
      gpu_options->SetPrecision(litert::GpuOptions::Precision::kFp32);
    }
  }
  return std::move(*options);
}

absl::Status Run() {
  const int S = absl::GetFlag(FLAGS_image_size);
  auto consts = HostConsts::Load(absl::GetFlag(FLAGS_host_consts));
  if (!consts.ok()) return consts.status();

  // ---- 1. The SAM 2 model, authored with the Tensor API.
  const std::string tflite = absl::GetFlag(FLAGS_sam2_tflite);
  const bool reuse = absl::GetFlag(FLAGS_reuse_tflite) &&
                     std::ifstream(tflite).good();
  if (!reuse) {
    const auto t0 = Clock::now();
    s2v::Sam2VideoConfig config;
    config.image.image_size = S;
    std::vector<sam2::WeightSpec> specs = sam2::GetWeightSpecs(config.image);
    std::vector<sam2::WeightSpec> vspecs = s2v::GetVideoWeightSpecs(config);
    specs.insert(specs.end(), vspecs.begin(), vspecs.end());
    WeightMap weights;
    if (auto st =
            sam2::LoadWeightSpecs(absl::GetFlag(FLAGS_weights), specs, weights);
        !st.ok()) {
      return st;
    }
    ModelFactory factory;
    if (auto st = AddSam2Signatures(factory, config, weights,
                                    consts->track_sparse);
        !st.ok()) {
      return st;
    }
    if (auto st = factory.Save(tflite); !st.ok()) return st;
    std::cout << "built " << tflite << " in "
              << std::chrono::duration<double>(Clock::now() - t0).count()
              << " s" << std::endl;
  }
  if (absl::GetFlag(FLAGS_build_only)) return absl::OkStatus();

  // ---- 2. Runtime.
  auto env_or = litert::Environment::Create({});
  if (!env_or) return absl::InternalError("Environment::Create failed");
  auto env = std::make_shared<litert::Environment>(std::move(*env_or));
  auto compile = [&](const std::string& path)
      -> absl::StatusOr<std::shared_ptr<litert::CompiledModel>> {
    auto options = MakeOptions();
    if (!options.ok()) return options.status();
    auto model = litert::CompiledModel::Create(*env, path, *options);
    if (!model) {
      return absl::InternalError(absl::StrCat("compile ", path, ": ",
                                              model.Error().Message()));
    }
    return std::make_shared<litert::CompiledModel>(std::move(*model));
  };
  auto t0 = Clock::now();
  auto sam2_model = compile(tflite);
  if (!sam2_model.ok()) return sam2_model.status();
  std::cout << "compiled SAM 2 model (" << absl::GetFlag(FLAGS_accelerator)
            << ") in "
            << std::chrono::duration<double>(Clock::now() - t0).count()
            << " s" << std::endl;

  PipelineOptions popts;
  popts.nmm = absl::GetFlag(FLAGS_nmm);
  popts.scratch_dir = absl::GetFlag(FLAGS_dump_dir).empty()
                          ? "/tmp"
                          : absl::GetFlag(FLAGS_dump_dir);
  auto pipeline = Sam2Pipeline::Create(env, *sam2_model, *std::move(consts),
                                       compile, popts);
  if (!pipeline.ok()) return pipeline.status();
  Sam2Pipeline& p = **pipeline;

  // ---- 3. Clip.
  const int W = absl::GetFlag(FLAGS_width), H = absl::GetFlag(FLAGS_height);
  std::ifstream clip(absl::GetFlag(FLAGS_frames_rgba), std::ios::binary);
  if (!clip) return absl::NotFoundError("--frames_rgba");
  clip.seekg(0, std::ios::end);
  const size_t frame_bytes = static_cast<size_t>(W) * H * 4;
  int T = static_cast<int>(static_cast<size_t>(clip.tellg()) / frame_bytes);
  clip.seekg(0);
  if (absl::GetFlag(FLAGS_frames) > 0) T = std::min(T, absl::GetFlag(FLAGS_frames));
  if (auto st = p.SetGeometry(W, H); !st.ok()) return st;
  const std::string fx = absl::GetFlag(FLAGS_effect);
  p.SetEffect(fx == "spotlight" ? Effect::kSpotlight
              : fx == "cutout"  ? Effect::kCutout
                                : Effect::kOverlay,
              static_cast<float>(absl::GetFlag(FLAGS_stroke)));

  auto prompts = ParsePrompts(absl::GetFlag(FLAGS_prompts));
  if (!prompts.ok()) return prompts.status();
  const std::string dump = absl::GetFlag(FLAGS_dump_dir);
  const std::string rgb_spec = absl::GetFlag(FLAGS_dump_rgb);
  std::vector<int> rgb_frames;
  for (absl::string_view s : absl::StrSplit(rgb_spec, ',', absl::SkipEmpty())) {
    int f;
    if (absl::SimpleAtoi(s, &f)) rgb_frames.push_back(f);
  }

  const FrameGeometry& geo = p.geometry();
  std::vector<uint8_t> rgba(frame_bytes);
  std::vector<float> frame(static_cast<size_t>(H) * geo.stride * 4, 0.f);
  std::vector<double> enc_ms, step_ms, frame_ms;
  for (int t = 0; t < T; ++t) {
    clip.read(reinterpret_cast<char*>(rgba.data()), frame_bytes);
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W * 4; ++x) {
        frame[(static_cast<size_t>(y) * geo.stride) * 4 + x] =
            rgba[(static_cast<size_t>(y) * W) * 4 + x] / 255.f;
      }
    }
    const auto f0 = Clock::now();
    if (auto st = WriteFloats(*p.frame(), frame); !st.ok()) return st;
    if (auto st = p.Encode(t); !st.ok()) return st;
    enc_ms.push_back(p.last_times().encode_ms);
    for (const Prompt& pr : *prompts) {
      if (pr.frame != t) continue;
      std::vector<Click> clicks;
      for (const auto& pt : pr.points) {
        clicks.push_back({std::clamp(pt[0] * S, 0.f, S - 1.f),
                          std::clamp(pt[1] * S, 0.f, S - 1.f),
                          static_cast<int>(pt[2])});
      }
      if (auto st = p.Prompt(pr.object, t, clicks); !st.ok()) return st;
    }
    if (auto st = p.Track(t); !st.ok()) return st;
    step_ms.push_back(p.last_times().step_ms);
    frame_ms.push_back(std::chrono::duration<double, std::milli>(
                           Clock::now() - f0).count());

    std::vector<std::string> line;
    for (int k = 0; k < kMaxObjects; ++k) {
      const FrameResult* r = p.Result(k, t);
      if (!r) continue;
      auto mask = ReadFloats(*r->low_mask);
      auto score = ReadFloats(*r->object_score);
      auto iou = ReadFloats(*r->iou);
      if (!mask.ok() || !score.ok() || !iou.ok()) {
        return absl::InternalError("readback failed");
      }
      const int fg = static_cast<int>(
          std::count_if(mask->begin(), mask->end(), [](float v) { return v > 0; }));
      line.push_back(absl::StrCat("obj", k, " fg=", fg, " score=", (*score)[0]));
      if (!dump.empty()) {
        if (auto st = Dump(absl::StrCat(dump, "/mask_o", k, "_f", t, ".f32"), *mask);
            !st.ok()) {
          return st;
        }
        if (auto st = Dump(absl::StrCat(dump, "/score_o", k, "_f", t, ".f32"),
                           {(*score)[0], (*iou)[0]});
            !st.ok()) {
          return st;
        }
      }
    }
    if (!dump.empty()) {
      auto px = p.pixels();
      if (!px.ok()) return px.status();
      auto pixels = ReadFloats(**px);
      if (!pixels.ok()) return pixels.status();
      if (auto st = Dump(absl::StrCat(dump, "/pixels_f", t, ".f32"), *pixels);
          !st.ok()) {
        return st;
      }
      if (rgb_spec == "all" ||
          std::find(rgb_frames.begin(), rgb_frames.end(), t) != rgb_frames.end()) {
        auto rgb = ReadFloats(*p.output());
        if (!rgb.ok()) return rgb.status();
        if (auto st = Dump(absl::StrCat(dump, "/rgb_f", t, ".f32"), *rgb);
            !st.ok()) {
          return st;
        }
      }
    }
    std::cout << "frame " << t << ": " << absl::StrJoin(line, "  ") << std::endl;
  }
  auto median = [](std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  std::cout << "median ms: encode " << median(enc_ms) << "  step "
            << median(step_ms) << "  frame " << median(frame_ms) << std::endl;
  if (!dump.empty()) {
    std::ofstream meta(dump + "/meta.json");
    meta << "{\"size\": " << S << ", \"width\": " << W << ", \"height\": " << H
         << ", \"frames\": " << T << ", \"nmm\": " << p.nmm()
         << ", \"prompts\": \"" << absl::GetFlag(FLAGS_prompts)
         << "\", \"effect\": \"" << fx << "\", \"stroke\": "
         << absl::GetFlag(FLAGS_stroke) << "}\n";
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  if (auto st = Run(); !st.ok()) {
    std::cerr << "error: " << st << std::endl;
    return 1;
  }
  return 0;
}

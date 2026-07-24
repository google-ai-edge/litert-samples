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

// Diffusion-LM denoise-step scaffold — Stage 0 host reference loop.
//
// Builds the denoise-step signature with synthetic weights, then runs the
// full masked-diffusion schedule: start from a prompt + all-[MASK] tail,
// call the step signature until masked_count reaches zero. Everything per
// step (predict, confidence, top-k unmask, id update) is in-graph; the host
// only feeds ids back.
//
// Usage (from the litert-samples repository root):
//   bazel build //models/llada/llada_8b/tensor_api:diffusion_main
//   diffusion_main --layers=4 --seq_len=256 --unmask_k=8 --accelerator=cpu

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
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
#include "models/llada/llada_8b/tensor_api/diffusion_graph.h"
#include "models/llada/llada_8b/tensor_api/diffusion_weights.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"

ABSL_FLAG(int, layers, 4, "Backbone layers (28 = full 0.6B shape)");
ABSL_FLAG(bool, llada, false,
          "Use the LLaDA-8B-Base shape (d_model 4096, MHA 32/32, head 128, "
          "ffn 12288, rope 5e5, eps 1e-5, no QK norm, vocab 126464)");
ABSL_FLAG(int, vocab, 0,
          "Override vocab size (mask_id = vocab-1) for lightweight synthetic "
          "smokes; 0 = config default. With --weights, slices the first N "
          "rows of the embedding/head (equivalence runs).");
ABSL_FLAG(std::string, weights, "",
          "LLaDA-8B-Base checkpoint dir (sharded safetensors); empty = "
          "synthetic weights");
ABSL_FLAG(bool, trace_ids, false,
          "Print full token ids after every step (numpy equivalence)");
ABSL_FLAG(std::string, prompt_ids, "",
          "Comma-separated token ids for the prompt prefix (real tokenizer); "
          "overrides --prompt_len/LCG when set");
ABSL_FLAG(std::string, weight_quant, "none",
          "none|int4|int8 — blockwise-32 int4 / per-channel int8 RTN on the matmul weights "
          "(real weights only)");
ABSL_FLAG(int, seq_len, 256, "Generation window");
ABSL_FLAG(int, prompt_len, 32, "Unmasked prompt prefix length");
ABSL_FLAG(int, unmask_k, 8, "Tokens revealed per step");
ABSL_FLAG(int, warmup, 2, "Warmup steps excluded from stats");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu|gpu");
ABSL_FLAG(std::string, gpu_precision, "default",
          "default|fp16|fp32 - GPU compute precision (delegate default = "
          "fp16 on Apple silicon)");
ABSL_FLAG(std::string, norms, "raw",
          "raw|composite - emit RMS norms as raw ops or as the "
          "odml.rms_norm composite (fused ML Drift kernels)");
ABSL_FLAG(std::string, attention, "raw",
          "raw|sdpa|sdpa_nofold - emit the attention core as raw fold ops or "
          "as the odml.scaled_dot_product_attention composite (no mask; MHA "
          "= plain BSND, GQA = group-fold presented as batched MHA). "
          "sdpa_nofold skips the GQA fold — the Metal wrong-result repro");
ABSL_FLAG(std::string, tflite_path, "/tmp/diffusion_step.tflite",
          "Where to write the serialized model");
ABSL_FLAG(bool, reuse_tflite, false,
          "Load tflite_path directly and skip weight loading + graph build "
          "(for device runs without the checkpoint)");

namespace {

using ::litert::tensor::Create;
using ::litert::tensor::LitertDynamicRunner;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::Type;
namespace diffusion = ::litert::tensor::examples::diffusion;

void BuildRopeCosSin(int seq, int head_dim, float base,
                     std::vector<float>& cos_out, std::vector<float>& sin_out) {
  cos_out.resize(static_cast<size_t>(seq) * head_dim);
  sin_out.resize(static_cast<size_t>(seq) * head_dim);
  int half = head_dim / 2;
  for (int s = 0; s < seq; ++s) {
    for (int i = 0; i < half; ++i) {
      float freq = std::pow(base, -2.0f * static_cast<float>(i) / head_dim);
      float angle = static_cast<float>(s) * freq;
      cos_out[s * head_dim + i] = std::cos(angle);
      cos_out[s * head_dim + half + i] = std::cos(angle);
      sin_out[s * head_dim + i] = std::sin(angle);
      sin_out[s * head_dim + half + i] = std::sin(angle);
    }
  }
}

absl::Status RunDiffusion() {
  if (const std::string n = absl::GetFlag(FLAGS_norms);
      n != "raw" && n != "composite") {
    return absl::InvalidArgumentError("--norms must be raw|composite");
  }
  if (const std::string a = absl::GetFlag(FLAGS_attention);
      a != "raw" && a != "sdpa" && a != "sdpa_nofold" &&
      a != "sdpa_folddirect") {
    return absl::InvalidArgumentError(
        "--attention must be raw|sdpa|sdpa_nofold|sdpa_folddirect");
  }
  diffusion::DiffusionConfig config;
  if (absl::GetFlag(FLAGS_llada)) {
    // LLaDA-8B-Base (confirmed against the checkpoint config.json 2026-07-21).
    config.emb_dim = 4096;
    config.n_layers = 32;
    config.n_heads = 32;
    config.n_kv_groups = 32;  // MHA
    config.head_dim = 128;
    config.hidden_dim = 12288;
    config.rms_norm_eps = 1e-5f;
    config.rope_base = 500000.0f;
    config.use_qk_norm = false;
    config.vocab = 126464;
    config.mask_id = 126336;
  }
  // NOTE: this assignment used to sit inside the --llada branch, so
  // --norms=composite was silently ignored for the default (1024-dim GQA)
  // shape — the 7/22 "1024-dim neutral" data point was raw-vs-raw.
  config.use_rms_norm_composite = absl::GetFlag(FLAGS_norms) == "composite";
  if (config.use_rms_norm_composite) {
    std::cout << "norms: odml.rms_norm composite" << std::endl;
  }
  const std::string attention_flag = absl::GetFlag(FLAGS_attention);
  config.use_sdpa_composite = attention_flag != "raw";
  config.sdpa_skip_gqa_fold = attention_flag == "sdpa_nofold";
  config.sdpa_fold_direct = attention_flag == "sdpa_folddirect";
  if (config.use_sdpa_composite) {
    std::cout << "attention: odml.scaled_dot_product_attention composite "
                 "(no mask"
              << (config.sdpa_skip_gqa_fold ? ", GQA FOLD SKIPPED — repro mode"
                                            : "")
              << ")" << std::endl;
  }
  config.n_layers = absl::GetFlag(FLAGS_layers);
  config.seq_len = absl::GetFlag(FLAGS_seq_len);
  config.unmask_k = absl::GetFlag(FLAGS_unmask_k);
  if (int v = absl::GetFlag(FLAGS_vocab); v > 0) {
    config.vocab = v;
    config.mask_id = v - 1;
  }
  const int prompt_len = absl::GetFlag(FLAGS_prompt_len);
  const int warmup = absl::GetFlag(FLAGS_warmup);
  const std::string tflite_path = absl::GetFlag(FLAGS_tflite_path);

  std::cout << "arch: " << (absl::GetFlag(FLAGS_llada) ? "LLaDA-8B" : "Qwen3")
            << " d_model " << config.emb_dim << " heads " << config.n_heads
            << " kv " << config.n_kv_groups << " ffn " << config.hidden_dim
            << " qk_norm " << config.use_qk_norm << " rope "
            << config.rope_base << " vocab " << config.vocab << " mask_id "
            << config.mask_id << std::endl;

  diffusion::WeightMap weights;
  const bool reuse_tflite = absl::GetFlag(FLAGS_reuse_tflite);
  if (reuse_tflite) {
    std::cout << "reusing serialized model: " << tflite_path << std::endl;
  } else if (const std::string dir = absl::GetFlag(FLAGS_weights);
             !dir.empty()) {
    const std::string qm = absl::GetFlag(FLAGS_weight_quant);
    const auto quant_mode =
        qm == "int4" ? diffusion::WeightQuantMode::kInt4
        : qm == "int8" ? diffusion::WeightQuantMode::kInt8
                       : diffusion::WeightQuantMode::kNone;
    auto weights_or = diffusion::LoadLladaWeights(config, dir, quant_mode);
    if (!weights_or.ok()) return weights_or.status();
    weights = std::move(*weights_or);
    std::cout << "loaded " << weights.size() << " real tensors from " << dir
              << std::endl;
  } else {
    weights = diffusion::MakeSyntheticWeights(config, /*seed=*/42);
  }
  if (!reuse_tflite) {
    diffusion::DiffusionInputs step_in = diffusion::MakeDiffusionInputs(config);
    diffusion::DiffusionOutputs step_out =
        diffusion::BuildDenoiseStep(config, step_in, weights);

    ModelFactory factory;
    std::vector<::litert::tensor::TensorHandle> ins, outs;
    for (auto& t : step_in.AsList()) ins.push_back(t);
    for (auto& t : step_out.AsList()) outs.push_back(t);
    auto status = factory.AddSignature(ins, outs, "denoise");
    if (!status.ok()) return status;
    auto save_status = factory.Save(tflite_path);
    if (!save_status.ok()) return save_status;
    std::cout << "Serialized: " << tflite_path << std::endl;
  }

  auto env = ::litert::Environment::Create({});
  if (!env) return absl::InternalError("Environment::Create failed");
  auto options = ::litert::Options::Create();
  if (!options) return absl::InternalError("Options::Create failed");
  options->SetHardwareAccelerators(absl::GetFlag(FLAGS_accelerator) == "gpu"
                                       ? ::litert::HwAccelerators::kGpu
                                       : ::litert::HwAccelerators::kCpu);
  if (const std::string prec = absl::GetFlag(FLAGS_gpu_precision);
      prec != "default") {
    auto gpu_options = options->GetGpuOptions();
    if (gpu_options.HasValue()) {
      auto st = gpu_options->SetPrecision(
          prec == "fp32" ? ::litert::GpuOptions::Precision::kFp32
                         : ::litert::GpuOptions::Precision::kFp16);
      if (!st) return absl::InternalError("SetPrecision failed");
      std::cout << "gpu precision: " << prec << std::endl;
    }
  }
  auto runner_or = LitertDynamicRunner::Create(*env, tflite_path, *options);
  if (!runner_or.ok()) return runner_or.status();
  auto runner = std::move(*runner_or);

  // Static rope tables (positions never move — full-window bidirectional).
  std::vector<float> cos_v, sin_v;
  BuildRopeCosSin(config.seq_len, config.head_dim, config.rope_base, cos_v,
                  sin_v);
  auto st = runner.SetInput(
      "denoise", "rope_cos",
      Create("rope_cos", Type::kFP32, {1, 1, config.seq_len, config.head_dim},
             std::vector<float>(cos_v)));
  if (!st.ok()) return st;
  st = runner.SetInput(
      "denoise", "rope_sin",
      Create("rope_sin", Type::kFP32, {1, 1, config.seq_len, config.head_dim},
             std::vector<float>(sin_v)));
  if (!st.ok()) return st;

  // Initial ids: real-tokenizer prompt if given, else pseudo-random prefix;
  // [MASK] tail either way.
  std::vector<int32_t> ids(config.seq_len, config.mask_id);
  int effective_prompt_len = prompt_len;
  if (const std::string prompt_ids_flag = absl::GetFlag(FLAGS_prompt_ids);
      !prompt_ids_flag.empty()) {
    std::vector<int32_t> parsed;
    size_t pos = 0;
    while (pos < prompt_ids_flag.size()) {
      size_t comma = prompt_ids_flag.find(',', pos);
      if (comma == std::string::npos) comma = prompt_ids_flag.size();
      parsed.push_back(std::stoi(prompt_ids_flag.substr(pos, comma - pos)));
      pos = comma + 1;
    }
    if (static_cast<int>(parsed.size()) >= config.seq_len) {
      return absl::InvalidArgumentError("prompt_ids longer than seq_len");
    }
    effective_prompt_len = static_cast<int>(parsed.size());
    std::copy(parsed.begin(), parsed.end(), ids.begin());
  } else {
    unsigned state = 7u;
    for (int i = 0; i < prompt_len; ++i) {
      state = state * 1664525u + 1013904223u;
      ids[i] = static_cast<int32_t>(state % (config.vocab - 1));
    }
  }

  int masked = config.seq_len - effective_prompt_len;
  int max_steps = (masked + config.unmask_k - 1) / config.unmask_k + 4;
  std::vector<double> step_ms;
  int step = 0;
  auto wall0 = std::chrono::steady_clock::now();
  while (masked > 0 && step < max_steps) {
    st = runner.SetInput("denoise", "token_ids",
                         Create("token_ids", Type::kI32, {1, config.seq_len},
                                std::vector<int32_t>(ids)));
    if (!st.ok()) return st;

    auto t0 = std::chrono::steady_clock::now();
    st = runner.Run("denoise");
    auto t1 = std::chrono::steady_clock::now();
    if (!st.ok()) return st;
    if (step >= warmup) {
      step_ms.push_back(
          std::chrono::duration<double, std::milli>(t1 - t0).count());
    }

    auto ids_out = runner.GetOutput("denoise", "new_token_ids");
    if (!ids_out.ok()) return ids_out.status();
    auto buffer = ids_out->GetBuffer();
    if (!buffer.ok()) return buffer.status();
    {
      auto lock = buffer->Lock();
      const int32_t* data = reinterpret_cast<const int32_t*>(lock.data());
      ids.assign(data, data + config.seq_len);
    }
    if (absl::GetFlag(FLAGS_trace_ids)) {
      std::cout << "step " << step << " ids:";
      for (int32_t id : ids) std::cout << " " << id;
      std::cout << std::endl;
    }
    auto count_out = runner.GetOutput("denoise", "masked_count");
    if (!count_out.ok()) return count_out.status();
    auto count_buffer = count_out->GetBuffer();
    if (!count_buffer.ok()) return count_buffer.status();
    {
      auto lock = count_buffer->Lock();
      masked = reinterpret_cast<const int32_t*>(lock.data())[0];
    }
    ++step;
  }
  auto wall1 = std::chrono::steady_clock::now();

  if (step_ms.empty()) return absl::InternalError("no timed steps");
  std::sort(step_ms.begin(), step_ms.end());
  double median = step_ms[step_ms.size() / 2];
  double wall_s = std::chrono::duration<double>(wall1 - wall0).count();
  int generated = config.seq_len - effective_prompt_len - masked;
  std::cout << "layers=" << config.n_layers << " seq_len=" << config.seq_len
            << " unmask_k=" << config.unmask_k
            << " accelerator=" << absl::GetFlag(FLAGS_accelerator) << std::endl;
  std::cout << "steps=" << step << " (remaining masked=" << masked
            << "), median " << median << " ms/step" << std::endl;
  std::cout << "generated " << generated << " tokens in " << wall_s
            << " s => " << (generated / wall_s)
            << " tok/s effective (parallel decode, " << config.unmask_k
            << "/step in-graph)" << std::endl;
  std::cout << "first generated ids:";
  for (int i = effective_prompt_len;
       i < std::min(effective_prompt_len + 8, config.seq_len); ++i)
    std::cout << " " << ids[i];
  std::cout << std::endl;
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = RunDiffusion();
  if (!status.ok()) {
    std::cerr << "FAILED: " << status << std::endl;
    return 1;
  }
  return 0;
}

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

// Talker step-graph — Stage 2 host reference loop.
//
// One decode signature call = one full 12Hz frame (talker + in-graph greedy
// cb0 + unrolled 5-layer code predictor for the 15 sub-groups + next-frame
// feedback embedding). Per step the host only stages: the rebound
// frame_feedback_emb output, a text-track row, the cb0 bias vector, rope rows
// and the cache position — vs the production host loop which runs talker and
// folded-MTP as separate graphs and aggregates frame embeddings with numpy.
//
// Usage (from the litert-samples repository root):
//   bazel build //models/qwen/qwen3_tts/tensor_api/talker_step:talker_main
//   talker_main --layers=28 --weights=.../model.safetensors \
//     --accelerator=gpu --gpu_buffer_storage=buffer
//
// NOTES: greedy decode; suppression via the cb0_bias input (specials masked,
// EOS ignored in the benchmark loop). --weights loads the real bf16
// checkpoint; without it, synthetic weights under the real tensor names.
// On Metal use --gpu_buffer_storage=buffer for max_seq >= 1024 (texture
// storage silently falls back to CPU) and --gpu_precision=fp32 for
// CPU/numpy-parity verification (fp16 drift flips greedy codec ids).
// Equivalence check: verify/numpy_talker_ref.py
// (gen -> run with --prefill_emb_file/--dump_dir -> check).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
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
#include "models/qwen/qwen3_tts/tensor_api/talker_step/talker_config.h"
#include "models/qwen/qwen3_tts/tensor_api/talker_step/talker_graph.h"
#include "models/qwen/qwen3_tts/tensor_api/talker_step/talker_weights.h"
#include "tensor/runners/litert/litert_dynamic_runner.h"
#include "tensor/tensor.h"

ABSL_FLAG(int, layers, 4, "Talker layers (28 = full 0.6B shape)");
ABSL_FLAG(int, steps, 48, "Decode steps (frames) to run");
ABSL_FLAG(int, warmup, 3, "Warmup decode steps excluded from stats");
ABSL_FLAG(int, max_seq, 256, "Fixed KV cache length");
ABSL_FLAG(int, prefill_len, 64, "Static prefill length");
ABSL_FLAG(std::string, accelerator, "cpu", "cpu|gpu");
ABSL_FLAG(std::string, norms, "raw",
          "raw|composite - emit RMS norms as raw ops or as the "
          "odml.rms_norm composite (fused ML Drift kernels)");
ABSL_FLAG(std::string, attention, "raw",
          "raw|rbmm — backbone attention as raw ops or as the "
          "odml.runtime_bmm composite pair (avbound composition; value "
          "caches stored transposed, runtime control tensors added)");
ABSL_FLAG(std::string, gpu_precision, "fp16",
          "fp16|fp32 — GPU calculation precision (fp16 is the delegate "
          "default; fp32 for CPU-parity verification runs)");
ABSL_FLAG(std::string, gpu_buffer_storage, "default",
          "default|buffer|texture2d — GPU tensor storage (texture2d hits "
          "Metal texture size limits for long KV caches)");
ABSL_FLAG(std::string, gpu_serialization_dir, "",
          "If set, enable GPU program-cache serialization into this dir "
          "(pairs with --gpu_cache_key; cuts repeat kernel-compile cost)");
ABSL_FLAG(std::string, gpu_cache_key, "talker_step",
          "Model cache key identifying this graph inside "
          "--gpu_serialization_dir");
ABSL_FLAG(int, gpu_compile_threads, 0,
          ">0: threads for GPU kernel compilation (delegate default "
          "otherwise)");
ABSL_FLAG(bool, gpu_no_shader_opt, false,
          "Disable GPU shader optimization to trade run speed for compile "
          "time");
ABSL_FLAG(bool, copy_rebind, false,
          "Rebind KV caches by copying bytes into fresh input buffers "
          "(emulates a naive host loop) instead of zero-copy handle "
          "duplication; for measuring the rebind tax");
ABSL_FLAG(std::string, tflite_path, "/tmp/talker_step.tflite",
          "Where to write the serialized model");
ABSL_FLAG(std::string, weights, "",
          "model.safetensors path (empty = synthetic weights). Requires "
          "--layers=28 unless testing a truncated stack.");
ABSL_FLAG(std::string, weight_quant, "none",
          "none|int8|int4 — per-channel symmetric quantization of the 2-D "
          "matmul weights at load time (requires --weights)");
ABSL_FLAG(std::string, prequant_weights, "",
          "Companion safetensors with pre-quantized int4 matmul weights "
          "(<name> I8 + <name>.scale F32, e.g. gptq_quantize.py output); "
          "overrides --weight_quant for those weights (requires --weights)");
ABSL_FLAG(std::string, prefill_emb_file, "",
          "Raw fp32 file [prefill_len, 1024] for the prefill dual-track "
          "embeddings (empty = synthetic)");
ABSL_FLAG(std::string, text_track_file, "",
          "Raw fp32 file [steps, 1024] of per-step text-track rows (empty = "
          "constant 0.01 row)");
ABSL_FLAG(double, sample_temp, 0.0,
          "cb0 temperature sampling via Gumbel noise staged in the cb0_bias "
          "input: argmax(logits + bias + T*g) == sample from "
          "softmax((logits+bias)/T). 0 = greedy. No graph change involved.");
ABSL_FLAG(int, sample_seed, 1, "RNG seed for --sample_temp");
ABSL_FLAG(int, eos_id, -1,
          "cb0 EOS id (2150 for Qwen3-TTS). >= 0: EOS is unmasked in cb0_bias "
          "after min_new_tokens=2 frames (production semantics) and the loop "
          "stops when cb0 == eos_id; the EOS frame is not counted or dumped. "
          "-1 = benchmark mode (all specials suppressed, fixed steps)");
ABSL_FLAG(std::string, dump_dir, "",
          "If set, write frames.i32 ((1+steps) x 16), cb0_logits.f32 "
          "((1+steps) x talker_vocab) and feedback.f32 ((1+steps) x emb_dim) "
          "for the equivalence check");

namespace {

using ::litert::tensor::Create;
using ::litert::tensor::LitertDynamicRunner;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::Type;
namespace talker = ::litert::tensor::examples::talker;

constexpr float kMaskFloor = -30000.0f;  // fp16-safe

void BuildRopeCosSin(int pos, int seq, int head_dim, float base,
                     std::vector<float>& cos_out, std::vector<float>& sin_out) {
  cos_out.resize(static_cast<size_t>(seq) * head_dim);
  sin_out.resize(static_cast<size_t>(seq) * head_dim);
  int half = head_dim / 2;
  for (int s = 0; s < seq; ++s) {
    for (int i = 0; i < half; ++i) {
      float freq = std::pow(base, -2.0f * static_cast<float>(i) / head_dim);
      float angle = static_cast<float>(pos + s) * freq;
      cos_out[s * head_dim + i] = std::cos(angle);
      cos_out[s * head_dim + half + i] = std::cos(angle);
      sin_out[s * head_dim + i] = std::sin(angle);
      sin_out[s * head_dim + half + i] = std::sin(angle);
    }
  }
}

std::vector<float> BuildMask(int seq_q, int valid_len, int max_seq) {
  std::vector<float> mask(static_cast<size_t>(seq_q) * max_seq, kMaskFloor);
  for (int s = 0; s < seq_q; ++s) {
    int limit = std::min(valid_len + s + 1, max_seq);
    for (int j = 0; j < limit; ++j) mask[s * max_seq + j] = 0.0f;
  }
  return mask;
}

// Suppress non-codec ids so greedy cb0 stays in [0, codec_vocab) during the
// synthetic run (real loop: host also folds EOS gating + repetition
// penalties into this vector).
std::vector<float> BuildCb0Bias(const talker::TalkerConfig& config) {
  std::vector<float> bias(config.talker_vocab, 0.0f);
  for (int i = config.codec_vocab; i < config.talker_vocab; ++i)
    bias[i] = kMaskFloor;
  return bias;
}

absl::Status RunTalker() {
  talker::TalkerConfig config;
  config.n_layers = absl::GetFlag(FLAGS_layers);
  config.max_seq = absl::GetFlag(FLAGS_max_seq);
  config.prefill_len = absl::GetFlag(FLAGS_prefill_len);
  const std::string attention = absl::GetFlag(FLAGS_attention);
  if (attention != "raw" && attention != "rbmm") {
    return absl::InvalidArgumentError("--attention must be raw|rbmm");
  }
  config.use_runtime_bmm = attention == "rbmm";
  const std::string norms = absl::GetFlag(FLAGS_norms);
  if (norms != "raw" && norms != "composite") {
    return absl::InvalidArgumentError("--norms must be raw|composite");
  }
  config.use_rms_norm_composite = norms == "composite";
  if (config.use_rms_norm_composite) {
    std::cout << "norms: odml.rms_norm composite" << std::endl;
  }
  if (config.use_runtime_bmm) {
    std::cout << "attention: odml.runtime_bmm (avbound)" << std::endl;
  }
  const int steps = absl::GetFlag(FLAGS_steps);
  const int warmup = absl::GetFlag(FLAGS_warmup);
  const std::string tflite_path = absl::GetFlag(FLAGS_tflite_path);

  talker::WeightMap weights;
  const std::string weights_path = absl::GetFlag(FLAGS_weights);
  if (weights_path.empty()) {
    weights = talker::MakeSyntheticWeights(config, /*seed=*/42);
    std::cout << "weights: synthetic (seed 42)" << std::endl;
  } else {
    const std::string quant = absl::GetFlag(FLAGS_weight_quant);
    talker::WeightQuantMode quant_mode;
    if (quant == "none") {
      quant_mode = talker::WeightQuantMode::kNone;
    } else if (quant == "int8") {
      quant_mode = talker::WeightQuantMode::kInt8;
    } else if (quant == "int4") {
      quant_mode = talker::WeightQuantMode::kInt4;
    } else {
      return absl::InvalidArgumentError("--weight_quant must be none|int8|int4");
    }
    const std::string prequant = absl::GetFlag(FLAGS_prequant_weights);
    auto weights_or = talker::LoadCheckpointWeights(config, weights_path,
                                                    quant_mode, prequant);
    if (!weights_or.ok()) return weights_or.status();
    weights = std::move(*weights_or);
    std::cout << "weights: " << weights_path << " (" << weights.size()
              << " tensors, bf16->fp32, quant=" << quant;
    if (!prequant.empty()) std::cout << ", prequant=" << prequant;
    std::cout << ")" << std::endl;
  }

  talker::TalkerInputs prefill_in = talker::MakeTalkerInputs(config, false);
  talker::TalkerOutputs prefill_out =
      talker::BuildTalkerStep(config, prefill_in, weights, false);

  talker::TalkerInputs decode_in = talker::MakeTalkerInputs(config, true);
  talker::TalkerOutputs decode_out =
      talker::BuildTalkerStep(config, decode_in, weights, true);

  ModelFactory factory;
  {
    std::vector<::litert::tensor::TensorHandle> ins, outs;
    for (auto& t : prefill_in.AsList(false)) ins.push_back(t);
    for (auto& t : prefill_out.AsList()) outs.push_back(t);
    auto status = factory.AddSignature(ins, outs, "prefill");
    if (!status.ok()) return status;
  }
  {
    std::vector<::litert::tensor::TensorHandle> ins, outs;
    for (auto& t : decode_in.AsList(true)) ins.push_back(t);
    for (auto& t : decode_out.AsList()) outs.push_back(t);
    auto status = factory.AddSignature(ins, outs, "decode");
    if (!status.ok()) return status;
  }
  auto save_status = factory.Save(tflite_path);
  if (!save_status.ok()) return save_status;
  std::cout << "Serialized: " << tflite_path << std::endl;

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
      const std::string ser_dir = absl::GetFlag(FLAGS_gpu_serialization_dir);
      if (!ser_dir.empty()) {
        const std::string cache_key = absl::GetFlag(FLAGS_gpu_cache_key);
        gpu_options->SetSerializationDir(ser_dir.c_str());
        gpu_options->SetModelCacheKey(cache_key.c_str());
        gpu_options->SetSerializeProgramCache(true);
        std::cout << "gpu serialization: dir=" << ser_dir
                  << " key=" << cache_key << std::endl;
      }
      const int compile_threads = absl::GetFlag(FLAGS_gpu_compile_threads);
      if (compile_threads > 0) {
        gpu_options->SetNumThreadsToCompile(compile_threads);
        std::cout << "gpu compile threads: " << compile_threads << std::endl;
      }
      if (absl::GetFlag(FLAGS_gpu_no_shader_opt)) {
        gpu_options->DisableShaderOptimization(true);
        std::cout << "gpu shader optimization: disabled" << std::endl;
      }
    }
  }
  auto runner_or = LitertDynamicRunner::Create(*env, tflite_path, *options);
  if (!runner_or.ok()) return runner_or.status();
  auto runner = std::move(*runner_or);

  const int C = config.num_code_groups;
  const int hd = config.head_dim;
  const int eos_id = absl::GetFlag(FLAGS_eos_id);
  if (eos_id >= config.talker_vocab) {
    return absl::InvalidArgumentError("--eos_id out of talker vocab range");
  }
  std::vector<float> cos_v, sin_v;
  // EOS stays masked for the prefill frame and the first decode frame
  // (min_new_tokens=2), then set_common picks up the unmasked entry.
  std::vector<float> cb0_bias = BuildCb0Bias(config);
  // Per-call bias actually sent to the graph: base bias, plus T-scaled
  // Gumbel noise when sampling (argmax(l + T*g) samples softmax(l/T);
  // masked entries stay ~kMaskFloor).
  const double sample_temp = absl::GetFlag(FLAGS_sample_temp);
  std::mt19937 sample_rng(absl::GetFlag(FLAGS_sample_seed));
  std::uniform_real_distribution<float> sample_unif(1e-12f, 1.0f);
  std::vector<float> cb0_bias_call;
  auto stage_bias = [&]() {
    cb0_bias_call = cb0_bias;
    if (sample_temp > 0.0) {
      for (float& b : cb0_bias_call) {
        b += static_cast<float>(
            sample_temp * -std::log(-std::log(sample_unif(sample_rng))));
      }
    }
  };

  auto set_common = [&](const std::string& sig, int pos, int seq,
                        int valid_len) -> absl::Status {
    auto st = runner.SetInput(sig, "cb0_bias",
                              Create("cb0_bias", Type::kFP32,
                                     {1, config.talker_vocab},
                                     std::vector<float>(cb0_bias_call)));
    if (!st.ok()) return st;
    st = runner.SetInput(sig, "cache_position",
                         Create("cache_position", Type::kI32, {4},
                                std::vector<int32_t>{0, 0, pos, 0}));
    if (!st.ok()) return st;
    if (config.use_runtime_bmm) {
      // AV-side active length must cover every position any query row can
      // attend: BuildMask lets row s reach valid_len + s + 1 positions, so
      // the max over rows is valid_len + seq (prefill: 0 + 64; decode:
      // pos + 1).
      const int active = std::min(valid_len + seq, config.max_seq);
      st = runner.SetInput(sig, "runtime_param",
                           Create("runtime_param", Type::kI32, {1, 1, 1, 7},
                                  std::vector<int32_t>(7, active)));
      if (!st.ok()) return st;
      st = runner.SetInput(
          sig, "runtime_param_full",
          Create("runtime_param_full", Type::kI32, {1, 1, 1, 7},
                 std::vector<int32_t>(7, config.max_seq)));
      if (!st.ok()) return st;
      st = runner.SetInput(sig, "cache_position_vt",
                           Create("cache_position_vt", Type::kI32, {4},
                                  std::vector<int32_t>{0, 0, 0, pos}));
      if (!st.ok()) return st;
    }
    BuildRopeCosSin(pos, seq, hd, config.rope_base, cos_v, sin_v);
    st = runner.SetInput(sig, "rope_cos",
                         Create("rope_cos", Type::kFP32, {1, 1, seq, hd},
                                std::vector<float>(cos_v)));
    if (!st.ok()) return st;
    st = runner.SetInput(sig, "rope_sin",
                         Create("rope_sin", Type::kFP32, {1, 1, seq, hd},
                                std::vector<float>(sin_v)));
    if (!st.ok()) return st;
    return runner.SetInput(sig, "attention_mask",
                           Create("attention_mask", Type::kFP32,
                                  {1, 1, seq, config.max_seq},
                                  BuildMask(seq, valid_len, config.max_seq)));
  };

  // --- Prefill: dual-track prompt embeddings (file or synthetic) ---
  {
    std::vector<float> emb(static_cast<size_t>(config.prefill_len) *
                           config.emb_dim);
    const std::string emb_file = absl::GetFlag(FLAGS_prefill_emb_file);
    if (emb_file.empty()) {
      unsigned state = 7u;
      for (auto& v : emb) {
        state = state * 1664525u + 1013904223u;
        v = 0.1f * (static_cast<float>(state >> 8) /
                        static_cast<float>(1u << 24) * 2.0f -
                    1.0f);
      }
    } else {
      std::ifstream in(emb_file, std::ios::binary);
      if (!in) return absl::NotFoundError("prefill_emb_file: " + emb_file);
      in.read(reinterpret_cast<char*>(emb.data()),
              emb.size() * sizeof(float));
      if (static_cast<size_t>(in.gcount()) != emb.size() * sizeof(float)) {
        return absl::InvalidArgumentError(
            "prefill_emb_file smaller than prefill_len x emb_dim fp32");
      }
    }
    auto st = runner.SetInput(
        "prefill", "embedded_input",
        Create("embedded_input", Type::kFP32,
               {1, config.prefill_len, config.emb_dim}, std::move(emb)));
    if (!st.ok()) return st;
    stage_bias();
    st = set_common("prefill", 0, config.prefill_len, 0);
    if (!st.ok()) return st;
    for (int i = 0; i < config.n_layers; ++i) {
      size_t n = static_cast<size_t>(config.n_kv_groups) * config.max_seq * hd;
      st = runner.SetInput(
          "prefill", absl::StrCat("key_cache_", i),
          Create(absl::StrCat("key_cache_", i).c_str(), Type::kFP32,
                 {1, config.n_kv_groups, config.max_seq, hd},
                 std::vector<float>(n, 0.0f)));
      if (!st.ok()) return st;
      std::vector<int> v_shape =
          config.use_runtime_bmm
              ? std::vector<int>{1, config.n_kv_groups, hd, config.max_seq}
              : std::vector<int>{1, config.n_kv_groups, config.max_seq, hd};
      st = runner.SetInput(
          "prefill", absl::StrCat("value_cache_", i),
          Create(absl::StrCat("value_cache_", i).c_str(), Type::kFP32,
                 v_shape, std::vector<float>(n, 0.0f)));
      if (!st.ok()) return st;
    }
    st = runner.Run("prefill");
    if (!st.ok()) return st;
  }

  const bool copy_rebind = absl::GetFlag(FLAGS_copy_rebind);
  std::vector<double> rebind_ms;
  bool alias_checked = false;
  bool cache_aliased = false;
  auto copy_input = [&](const std::string& name,
                        const ::litert::tensor::TensorHandle& t)
      -> absl::Status {
    auto buffer = t.GetBuffer();
    if (!buffer.ok()) return buffer.status();
    auto lock = buffer->Lock();
    return runner.SetInput(
        "decode", name,
        absl::Span<const uint8_t>(
            reinterpret_cast<const uint8_t*>(lock.data()), lock.size()));
  };
  auto rebind = [&](const std::string& from_sig) -> absl::Status {
    auto r0 = std::chrono::steady_clock::now();
    for (int i = 0; i < config.n_layers; ++i) {
      auto k = runner.GetOutput(from_sig, absl::StrCat("key_cache_", i, "_new"));
      if (!k.ok()) return k.status();
      auto v =
          runner.GetOutput(from_sig, absl::StrCat("value_cache_", i, "_new"));
      if (!v.ok()) return v.status();
      absl::Status st;
      if (copy_rebind) {
        st = copy_input(absl::StrCat("key_cache_", i), *k);
        if (!st.ok()) return st;
        st = copy_input(absl::StrCat("value_cache_", i), *v);
      } else {
        st = runner.SetInput("decode", absl::StrCat("key_cache_", i), *k);
        if (!st.ok()) return st;
        st = runner.SetInput("decode", absl::StrCat("value_cache_", i), *v);
      }
      if (!st.ok()) return st;
    }
    auto fb = runner.GetOutput(from_sig, "frame_feedback_emb");
    if (!fb.ok()) return fb.status();
    auto st = copy_rebind ? copy_input("feedback_emb", *fb)
                          : runner.SetInput("decode", "feedback_emb", *fb);
    if (!st.ok()) return st;
    rebind_ms.push_back(std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - r0)
                            .count());
    // After the first decode->decode rebind, check whether the cache input
    // now aliases the cache output buffer (in-place DUS eligibility).
    if (!alias_checked && from_sig == "decode") {
      alias_checked = true;
      const void* p_in = nullptr;
      const void* p_out = nullptr;
      auto in_t = runner.GetInput("decode", "key_cache_0");
      auto out_t = runner.GetOutput("decode", "key_cache_0_new");
      if (in_t.ok() && out_t.ok()) {
        {
          auto b = in_t->GetBuffer();
          if (b.ok()) {
            auto lock = b->Lock();
            p_in = lock.data();
          }
        }
        {
          auto b = out_t->GetBuffer();
          if (b.ok()) {
            auto lock = b->Lock();
            p_out = lock.data();
          }
        }
      }
      cache_aliased = p_in != nullptr && p_in == p_out;
    }
    return absl::OkStatus();
  };

  auto read_frame = [&](const std::string& sig,
                        std::vector<int32_t>& ids) -> absl::Status {
    auto out = runner.GetOutput(sig, "codec_frame");
    if (!out.ok()) return out.status();
    auto buffer = out->GetBuffer();
    if (!buffer.ok()) return buffer.status();
    auto lock = buffer->Lock();
    const int32_t* data = reinterpret_cast<const int32_t*>(lock.data());
    ids.assign(data, data + C);
    return absl::OkStatus();
  };

  const std::string dump_dir = absl::GetFlag(FLAGS_dump_dir);
  std::vector<int32_t> dump_frames;
  std::vector<float> dump_logits;
  std::vector<float> dump_feedback;
  auto read_floats = [&](const std::string& sig, const std::string& name,
                         std::vector<float>& sink) -> absl::Status {
    auto out = runner.GetOutput(sig, name);
    if (!out.ok()) return out.status();
    auto buffer = out->GetBuffer();
    if (!buffer.ok()) return buffer.status();
    auto lock = buffer->Lock();
    const float* data = reinterpret_cast<const float*>(lock.data());
    sink.insert(sink.end(), data, data + lock.size() / sizeof(float));
    return absl::OkStatus();
  };
  auto dump_step = [&](const std::string& sig,
                       const std::vector<int32_t>& ids) -> absl::Status {
    if (dump_dir.empty()) return absl::OkStatus();
    dump_frames.insert(dump_frames.end(), ids.begin(), ids.end());
    auto st = read_floats(sig, "cb0_logits", dump_logits);
    if (!st.ok()) return st;
    return read_floats(sig, "frame_feedback_emb", dump_feedback);
  };

  std::vector<int32_t> frame(C, 0);
  auto st = read_frame("prefill", frame);
  if (!st.ok()) return st;
  st = dump_step("prefill", frame);
  if (!st.ok()) return st;
  st = rebind("prefill");
  if (!st.ok()) return st;

  // Text track: per-step rows from file, or a constant synthetic row.
  std::vector<float> text_track;
  {
    const std::string track_file = absl::GetFlag(FLAGS_text_track_file);
    if (track_file.empty()) {
      text_track.assign(static_cast<size_t>(steps) * config.emb_dim, 0.01f);
    } else {
      text_track.resize(static_cast<size_t>(steps) * config.emb_dim);
      std::ifstream in(track_file, std::ios::binary);
      if (!in) return absl::NotFoundError("text_track_file: " + track_file);
      in.read(reinterpret_cast<char*>(text_track.data()),
              text_track.size() * sizeof(float));
      if (static_cast<size_t>(in.gcount()) !=
          text_track.size() * sizeof(float)) {
        return absl::InvalidArgumentError(
            "text_track_file smaller than steps x emb_dim fp32");
      }
    }
  }

  std::vector<double> step_ms;
  step_ms.reserve(steps);
  bool eos_hit = false;
  int frames_generated = 1;  // the prefill frame
  int pos = config.prefill_len;
  for (int step = 0; step < steps && pos < config.max_seq; ++step, ++pos) {
    if (eos_id >= 0 && step == 1) cb0_bias[eos_id] = 0.0f;
    const float* row = text_track.data() +
                       static_cast<size_t>(step) * config.emb_dim;
    st = runner.SetInput(
        "decode", "text_track_emb",
        Create("text_track_emb", Type::kFP32, {1, 1, config.emb_dim},
               std::vector<float>(row, row + config.emb_dim)));
    if (!st.ok()) return st;
    stage_bias();
    st = set_common("decode", pos, 1, pos);
    if (!st.ok()) return st;

    auto t0 = std::chrono::steady_clock::now();
    st = runner.Run("decode");
    auto t1 = std::chrono::steady_clock::now();
    if (!st.ok()) return st;

    if (step >= warmup) {
      step_ms.push_back(
          std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    st = read_frame("decode", frame);
    if (!st.ok()) return st;
    if (eos_id >= 0 && frame[0] == eos_id) {
      eos_hit = true;
      std::cout << "EOS at decode step " << step << "; frames generated: "
                << frames_generated << std::endl;
      break;
    }
    ++frames_generated;
    st = dump_step("decode", frame);
    if (!st.ok()) return st;
    st = rebind("decode");
    if (!st.ok()) return st;
  }
  if (eos_id >= 0 && !eos_hit) {
    std::cout << "no EOS within " << steps << " steps; frames generated: "
              << frames_generated << std::endl;
  }

  if (step_ms.empty()) return absl::InternalError("no timed steps");
  std::sort(step_ms.begin(), step_ms.end());
  double median = step_ms[step_ms.size() / 2];
  double budget_ms = 1000.0 / config.frame_rate_hz;
  std::cout << "layers=" << config.n_layers << " max_seq=" << config.max_seq
            << " code_groups=" << C << " (cp " << config.cp_layers
            << "L in-graph) accelerator=" << absl::GetFlag(FLAGS_accelerator)
            << std::endl;
  std::cout << "decode steps timed: " << step_ms.size() << ", median " << median
            << " ms/frame (min " << step_ms.front() << ", max "
            << step_ms.back() << ")" << std::endl;
  if (!rebind_ms.empty()) {
    std::sort(rebind_ms.begin(), rebind_ms.end());
    std::cout << "cache rebind: median "
              << rebind_ms[rebind_ms.size() / 2] << " ms/frame (max "
              << rebind_ms.back() << ", mode="
              << (copy_rebind ? "copy" : "zero-copy")
              << ", decode in==out alias: "
              << (cache_aliased ? "YES (in-place DUS eligible)" : "no")
              << "); rebind is OUTSIDE the timed decode step" << std::endl;
  }
  std::cout << "RTF vs " << config.frame_rate_hz << "Hz budget (" << budget_ms
            << " ms): " << (median / budget_ms)
            << " (1 signature call per frame: talker + cb0 + 15-group code "
               "predictor + feedback aggregation in-graph)"
            << std::endl;
  std::cout << "last frame ids:";
  for (int32_t v : frame) std::cout << " " << v;
  std::cout << std::endl;

  // Numeric sanity on the last feedback embedding (catches fp blowups that
  // greedy ids can mask).
  {
    std::vector<float> fb;
    auto fst = read_floats("decode", "frame_feedback_emb", fb);
    if (!fst.ok()) return fst;
    double sum_sq = 0.0;
    bool finite = true;
    for (float v : fb) {
      if (!std::isfinite(v)) finite = false;
      sum_sq += static_cast<double>(v) * v;
    }
    std::cout << "last feedback emb: L2 " << std::sqrt(sum_sq)
              << (finite ? " (all finite)" : " (NON-FINITE VALUES!)")
              << std::endl;
    if (!finite) return absl::InternalError("non-finite feedback embedding");
  }

  if (!dump_dir.empty()) {
    auto write_raw = [&](const std::string& name, const void* data,
                         size_t bytes) -> absl::Status {
      std::string p = dump_dir + "/" + name;
      std::ofstream out(p, std::ios::binary);
      if (!out) return absl::InternalError("cannot write " + p);
      out.write(reinterpret_cast<const char*>(data), bytes);
      return absl::OkStatus();
    };
    st = write_raw("frames.i32", dump_frames.data(),
                   dump_frames.size() * sizeof(int32_t));
    if (!st.ok()) return st;
    st = write_raw("cb0_logits.f32", dump_logits.data(),
                   dump_logits.size() * sizeof(float));
    if (!st.ok()) return st;
    st = write_raw("feedback.f32", dump_feedback.data(),
                   dump_feedback.size() * sizeof(float));
    if (!st.ok()) return st;
    std::cout << "dumped " << dump_frames.size() / C << " frames to "
              << dump_dir << std::endl;
  }
  return absl::OkStatus();
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::Status status = RunTalker();
  if (!status.ok()) {
    std::cerr << "FAILED: " << status << std::endl;
    return 1;
  }
  return 0;
}

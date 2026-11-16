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

// Writes a Qwen2-family decoder (Qwen2.5-VL-72B-Instruct's text model) as a .tflite with the LiteRT Tensor
// API, in the signature layout LiteRT-LM's stock executor runs (the one litert-torch export_hf writes with an
// externalized embedder):
//   prefill_<N> : embeddings [1,N,D] f32, input_pos [N] i32, mask [1,1,N,C] f32,
//                 kv_cache_k_<i> [1,KV,C,hd] f32, kv_cache_v_<i> [1,KV,hd,C] f32 (V stored transposed)
//                 -> kv_cache_k_<i>, kv_cache_v_<i>
//   decode      : the same with N = 1 -> kv_cache_k_<i>, kv_cache_v_<i>, logits [1,1,V] f32
// RoPE cos / sin and the cache write position are computed in the graph from input_pos; the mask comes
// from the runtime. Weights come from tapi_pack.py (weights.bin, index.tsv, config.tsv): fp32
// norms and biases, int4 blockwise or int8 per-channel matmul weights, read through mmap and written once
// for all signatures.

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/flags/flag.h"  // from @com_google_absl
#include "absl/flags/parse.h"  // from @com_google_absl
#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/log/absl_log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/numbers.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_split.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/backends/tflite/tflite_flatbuffer_conversion.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

ABSL_FLAG(std::string, weights_dir, "",
          "Directory written by tapi_pack.py (weights.bin, index.tsv, config.tsv).");
ABSL_FLAG(std::string, out, "", "Output .tflite path.");
ABSL_FLAG(std::string, prefill, "128", "Prefill lengths, comma separated.");
ABSL_FLAG(int, cache, 4096, "KV cache length.");

namespace {

using ::litert::tensor::BlockwiseQuantization;
using ::litert::tensor::ModelFactory;
using ::litert::tensor::OwningCpuBuffer;
using ::litert::tensor::PerChannelAffineQuantization;
using ::litert::tensor::SpanCpuBuffer;
using ::litert::tensor::TensorHandle;
using ::litert::tensor::TfLiteMixinTag;
using ::litert::tensor::Type;
using T = ::litert::tensor::Tensor<TfLiteMixinTag>;

struct Config {
  int hidden = 0, layers = 0, heads = 0, kv_heads = 0, head_dim = 0, mlp = 0, vocab = 0;
  float eps = 0, rope_theta = 0;
};

std::vector<std::string> ReadLines(const std::string& path) {
  std::ifstream f(path);
  ABSL_CHECK(f.good()) << "cannot read " << path;
  std::vector<std::string> lines;
  for (std::string line; std::getline(f, line);) {
    if (!line.empty()) lines.push_back(line);
  }
  return lines;
}

Config ReadConfig(const std::string& path) {
  absl::flat_hash_map<std::string, std::string> kv;
  for (const std::string& line : ReadLines(path)) {
    std::vector<std::string> p = absl::StrSplit(line, '\t');
    ABSL_CHECK_EQ(p.size(), 2u) << line;
    kv[p[0]] = p[1];
  }
  auto i = [&](const char* k) {
    int v = 0;
    ABSL_CHECK(absl::SimpleAtoi(kv.at(k), &v)) << k;
    return v;
  };
  auto f = [&](const char* k) {
    double v = 0;
    ABSL_CHECK(absl::SimpleAtod(kv.at(k), &v)) << k;
    return static_cast<float>(v);
  };
  Config c;
  c.hidden = i("hidden");
  c.layers = i("layers");
  c.heads = i("heads");
  c.kv_heads = i("kv_heads");
  c.head_dim = i("head_dim");
  c.mlp = i("mlp");
  c.vocab = i("vocab");
  c.eps = f("eps");
  c.rope_theta = f("rope_theta");
  return c;
}

// The packed weights, mapped read-only. Each constant tensor is created once and reused by every signature,
// so ModelFactory stores its buffer once.
class Weights {
 public:
  explicit Weights(const std::string& dir) {
    const std::string bin = dir + "/weights.bin";
    int fd = open(bin.c_str(), O_RDONLY);
    ABSL_CHECK_GE(fd, 0) << "cannot open " << bin;
    struct stat st;
    ABSL_CHECK_EQ(fstat(fd, &st), 0);
    size_ = static_cast<size_t>(st.st_size);
    void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
    ABSL_CHECK(p != MAP_FAILED) << "mmap failed for " << bin;
    close(fd);
    base_ = static_cast<const std::byte*>(p);
    for (const std::string& line : ReadLines(dir + "/index.tsv")) {
      std::vector<std::string> f = absl::StrSplit(line, '\t');
      ABSL_CHECK_EQ(f.size(), 8u) << line;
      Entry e;
      e.kind = f[1];
      for (absl::string_view d : absl::StrSplit(f[2], ',')) {
        int v = 0;
        ABSL_CHECK(absl::SimpleAtoi(d, &v));
        e.shape.push_back(v);
      }
      ABSL_CHECK(absl::SimpleAtoi(f[3], &e.off));
      ABSL_CHECK(absl::SimpleAtoi(f[4], &e.bytes));
      ABSL_CHECK(absl::SimpleAtoi(f[5], &e.scale_off));
      ABSL_CHECK(absl::SimpleAtoi(f[6], &e.scale_count));
      ABSL_CHECK(absl::SimpleAtoi(f[7], &e.block));
      ABSL_CHECK_LE(e.off + e.bytes, size_) << f[0];
      ABSL_CHECK_LE(e.scale_off + e.scale_count * sizeof(float), size_) << f[0];
      entries_[f[0]] = e;
    }
  }

  ~Weights() {
    if (base_ != nullptr && size_ > 0) {
      munmap(const_cast<std::byte*>(base_), size_);
    }
  }

  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;

  const T& Get(const std::string& name) {
    auto it = cache_.find(name);
    if (it != cache_.end()) return it->second;
    auto eit = entries_.find(name);
    ABSL_CHECK(eit != entries_.end()) << "missing weight " << name;
    const Entry& e = eit->second;
    auto buffer = std::make_shared<SpanCpuBuffer>(base_ + e.off, e.bytes);
    T t;
    if (e.kind == "fp32") {
      t = T({.name = name, .type = Type::kFP32, .shape = e.shape, .buffer = buffer});
    } else if (e.kind == "bw4") {
      // Same fields as ai-edge-quantizer's blockwise int4: fp16 scales (written by ModelFactory from these
      // floats, already fp16-exact), no zero point, quantized_dimension 0.
      auto q = std::make_shared<BlockwiseQuantization>(Scales(e), std::vector<int64_t>{}, e.block, 0);
      t = T({.name = name, .type = Type::kI4, .shape = e.shape, .buffer = buffer, .quantization = q});
    } else if (e.kind == "pc8") {
      auto q = std::make_shared<PerChannelAffineQuantization>(
          Scales(e), std::vector<int64_t>(e.scale_count, 0), 0);
      t = T({.name = name, .type = Type::kI8, .shape = e.shape, .buffer = buffer, .quantization = q});
    } else {
      ABSL_LOG(FATAL) << "unknown kind " << e.kind << " for " << name;
    }
    return cache_.emplace(name, t).first->second;
  }

 private:
  struct Entry {
    std::string kind;
    std::vector<int> shape;
    uint64_t off = 0, bytes = 0, scale_off = 0, scale_count = 0;
    int block = 0;
  };

  std::vector<float> Scales(const Entry& e) const {
    std::vector<float> s(e.scale_count);
    std::memcpy(s.data(), base_ + e.scale_off, e.scale_count * sizeof(float));
    return s;
  }

  const std::byte* base_ = nullptr;
  size_t size_ = 0;
  absl::flat_hash_map<std::string, Entry> entries_;
  absl::flat_hash_map<std::string, T> cache_;
};

T ConstF(std::vector<float> v, std::vector<int> shape) {
  return T({.type = Type::kFP32, .shape = std::move(shape), .buffer = OwningCpuBuffer::Copy<Type::kFP32>(v)});
}

T ConstI(std::vector<int32_t> v, std::vector<int> shape) {
  return T({.type = Type::kI32, .shape = std::move(shape), .buffer = OwningCpuBuffer::Copy<Type::kI32>(v)});
}

T RmsNorm(const T& x, const T& w, float eps) {
  const int last = static_cast<int>(x.GetShape().size()) - 1;
  T ms = Mean(Mul(x, x), {last}, /*keep_dims=*/true);
  return Mul(Mul(x, Rsqrt(Add(ms, ConstF({eps}, {1})))), w);
}

// x: [1, n, seq, hd]; cos / sin: [1, 1, seq, hd].
T Rope(const T& x, const T& cos, const T& sin) {
  const auto& s = x.GetShape();
  const int half = s[3] / 2;
  T x1 = Slice(x, {0, 0, 0, 0}, {s[0], s[1], s[2], half});
  T x2 = Slice(x, {0, 0, 0, half}, {s[0], s[1], s[2], half});
  std::vector<T> parts = {Mul(x2, ConstF({-1.0f}, {1})), x1};
  T rotated = Concatenation(absl::MakeSpan(parts), 3);
  return Add(Mul(x, cos), Mul(rotated, sin));
}

struct Signature {
  std::vector<TensorHandle> inputs, outputs;
};

Signature Build(const Config& c, Weights& w, int seq, int cache, bool decode) {
  ABSL_CHECK_GT(c.kv_heads, 0);
  ABSL_CHECK_EQ(c.heads % c.kv_heads, 0);
  const int D = c.hidden, H = c.heads, KV = c.kv_heads, hd = c.head_dim, g = H / KV;
  T emb({.name = "embeddings", .type = Type::kFP32, .shape = {1, seq, D}});
  T pos({.name = "input_pos", .type = Type::kI32, .shape = {seq}});
  T mask({.name = "mask", .type = Type::kFP32, .shape = {1, 1, seq, cache}});
  std::vector<T> kin, vin;
  for (int i = 0; i < c.layers; ++i) {
    kin.push_back(T({.name = absl::StrCat("kv_cache_k_", i), .type = Type::kFP32, .shape = {1, KV, cache, hd}}));
    vin.push_back(T({.name = absl::StrCat("kv_cache_v_", i), .type = Type::kFP32, .shape = {1, KV, hd, cache}}));
  }

  // RoPE tables from the positions, in fp32 as HF: freqs = pos * inv_freq, emb = [freqs, freqs].
  std::vector<float> inv(hd / 2);
  for (int j = 0; j < hd / 2; ++j) {
    inv[j] = 1.0f / std::pow(c.rope_theta, static_cast<float>(2 * j) / static_cast<float>(hd));
  }
  T freqs = Mul(Reshape(Cast(pos, Type::kFP32), {seq, 1}), ConstF(inv, {1, hd / 2}));
  std::vector<T> ff = {freqs, freqs};
  T angles = Concatenation(absl::MakeSpan(ff), 1);
  T cos = Reshape(Cos(angles), {1, 1, seq, hd});
  T sin = Reshape(Sin(angles), {1, 1, seq, hd});

  // Cache write position = input_pos[0] (the runtime's positions are contiguous within a call).
  T pos0 = Slice(pos, {0}, {1});
  T zero = ConstI({0}, {1});
  std::vector<T> ks = {zero, zero, pos0, zero};
  std::vector<T> vs = {zero, zero, zero, pos0};
  T k_start = Concatenation(absl::MakeSpan(ks), 0);
  T v_start = Concatenation(absl::MakeSpan(vs), 0);
  const T scale = ConstF({1.0f / std::sqrt(static_cast<float>(hd))}, {1});

  T x = emb;
  std::vector<T> kout, vout;
  for (int l = 0; l < c.layers; ++l) {
    const std::string p = absl::StrCat("model.layers.", l, ".");
    T h = RmsNorm(x, w.Get(p + "input_layernorm.weight"), c.eps);
    T q = FullyConnected(h, w.Get(p + "self_attn.q_proj.weight"), w.Get(p + "self_attn.q_proj.bias"));
    T k = FullyConnected(h, w.Get(p + "self_attn.k_proj.weight"), w.Get(p + "self_attn.k_proj.bias"));
    T v = FullyConnected(h, w.Get(p + "self_attn.v_proj.weight"), w.Get(p + "self_attn.v_proj.bias"));
    q = Rope(Transpose(Reshape(q, {1, seq, H, hd}), {0, 2, 1, 3}), cos, sin);   // [1, H, seq, hd]
    k = Rope(Transpose(Reshape(k, {1, seq, KV, hd}), {0, 2, 1, 3}), cos, sin);  // [1, KV, seq, hd]
    v = Transpose(Reshape(v, {1, seq, KV, hd}), {0, 2, 3, 1});                  // [1, KV, hd, seq]
    T kc = DynamicUpdateSlice(kin[l], k, k_start);
    T vc = DynamicUpdateSlice(vin[l], v, v_start);
    kc.SetName(absl::StrCat("kv_cache_k_", l));
    vc.SetName(absl::StrCat("kv_cache_v_", l));
    kout.push_back(kc);
    vout.push_back(vc);

    // Grouped-query attention over the whole cache; the runtime's mask hides the unwritten slots.
    T scores = BatchMatMul(Reshape(q, {KV, g * seq, hd}), Reshape(kc, {KV, cache, hd}),
                           /*adj_x=*/false, /*adj_y=*/true);  // [KV, g*seq, cache]
    scores = Add(Mul(Reshape(scores, {1, H, seq, cache}), scale), mask);
    T probs = Reshape(Softmax(scores), {KV, g * seq, cache});
    T o = BatchMatMul(probs, Reshape(vc, {KV, hd, cache}), /*adj_x=*/false, /*adj_y=*/true);  // [KV, g*seq, hd]
    o = Reshape(Transpose(Reshape(o, {1, H, seq, hd}), {0, 2, 1, 3}), {1, seq, H * hd});
    x = Add(x, FullyConnected(o, w.Get(p + "self_attn.o_proj.weight")));

    T h2 = RmsNorm(x, w.Get(p + "post_attention_layernorm.weight"), c.eps);
    T gate = FullyConnected(h2, w.Get(p + "mlp.gate_proj.weight"));
    T up = FullyConnected(h2, w.Get(p + "mlp.up_proj.weight"));
    T act = Mul(Mul(gate, Logistic(gate)), up);
    x = Add(x, FullyConnected(act, w.Get(p + "mlp.down_proj.weight")));
  }

  Signature s;
  s.inputs = {emb, pos, mask};
  for (int l = 0; l < c.layers; ++l) {
    s.inputs.push_back(kin[l]);
    s.inputs.push_back(vin[l]);
  }
  for (int l = 0; l < c.layers; ++l) {
    s.outputs.push_back(kout[l]);
    s.outputs.push_back(vout[l]);
  }
  if (decode) {
    T logits = FullyConnected(RmsNorm(x, w.Get("model.norm.weight"), c.eps), w.Get("lm_head.weight"));  // [1, seq, V]
    logits.SetName("logits");
    s.outputs.push_back(logits);
  }
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  const std::string dir = absl::GetFlag(FLAGS_weights_dir);
  const std::string out = absl::GetFlag(FLAGS_out);
  ABSL_CHECK(!dir.empty() && !out.empty()) << "--weights_dir and --out are required";
  const int cache = absl::GetFlag(FLAGS_cache);
  const Config c = ReadConfig(dir + "/config.tsv");
  std::cout << "config: hidden " << c.hidden << " layers " << c.layers << " heads " << c.heads << "/" << c.kv_heads
            << " head_dim " << c.head_dim << " mlp " << c.mlp << " vocab " << c.vocab << " cache " << cache
            << std::endl;
  Weights w(dir);
  ModelFactory factory;
  for (absl::string_view p : absl::StrSplit(absl::GetFlag(FLAGS_prefill), ',')) {
    int n = 0;
    ABSL_CHECK(absl::SimpleAtoi(p, &n));
    Signature s = Build(c, w, n, cache, /*decode=*/false);
    absl::Status st = factory.AddSignature(s.inputs, s.outputs, absl::StrCat("prefill_", n));
    ABSL_CHECK(st.ok()) << st;
    std::cout << "signature prefill_" << n << std::endl;
  }
  Signature d = Build(c, w, 1, cache, /*decode=*/true);
  absl::Status st = factory.AddSignature(d.inputs, d.outputs, "decode");
  ABSL_CHECK(st.ok()) << st;
  std::cout << "signature decode" << std::endl;
  st = factory.Save(out);
  ABSL_CHECK(st.ok()) << st;
  std::cout << "WROTE " << out << std::endl;
  return 0;
}

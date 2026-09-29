// SAM2.1 hiera-tiny video path — graph construction. See sam2v_graph.h.

#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_video/sam2v_graph.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::sam2_video {

namespace {

constexpr float kTwoPi = 6.283185307179586f;

TfTensor ConstScalar(float value) {
  return TfTensor({.type = Type::kFP32,
                   .shape = {1},
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>({value})});
}

TfTensor ConstFloats(const std::vector<float>& values,
                     const std::vector<int>& shape,
                     const std::string& name = "") {
  return TfTensor({.name = name,
                   .type = Type::kFP32,
                   .shape = shape,
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>(values)});
}

const TfTensor& W(const WeightMap& weights, const std::string& name) {
  auto it = weights.find(name);
  ABSL_CHECK(it != weights.end()) << "missing weight: " << name;
  return it->second;
}

std::vector<float> HostFloats(const TfTensor& tensor) {
  auto buffer = tensor.GetBuffer();
  ABSL_CHECK(buffer.ok()) << "weight has no buffer";
  auto lock = buffer->Lock();
  const float* data = reinterpret_cast<const float*>(lock.data());
  return std::vector<float>(data, data + lock.size() / sizeof(float));
}

TfTensor PadHW(const TfTensor& x, int top, int bottom, int left, int right) {
  TfTensor paddings(
      {.type = Type::kI32,
       .shape = {4, 2},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(
           {0, 0, top, bottom, left, right, 0, 0})});
  return Pad(x, paddings);
}

// Last-axis LayerNorm with weight+bias (covers channels-first LayerNorm2d in
// NHWC). Raw decomposition, same form the image path verified.
TfTensor LayerNorm(const TfTensor& x, const TfTensor& weight,
                   const TfTensor& bias, float eps) {
  int last = static_cast<int>(x.GetShape().size()) - 1;
  TfTensor mean = Mean(x, {last}, /*keep_dims=*/true);
  TfTensor centered = Sub(x, mean);
  TfTensor var = Mean(Mul(centered, centered), {last}, /*keep_dims=*/true);
  TfTensor normed = Mul(centered, Rsqrt(Add(var, ConstScalar(eps))));
  return Add(Mul(normed, weight), bias);
}

TfTensor ScaleConst(ConstCache& cache, const TfTensor& t, float scale,
                    const std::string& name) {
  return cache.GetOrCreate(name + "@scaled", [&] {
    std::vector<float> vals = HostFloats(t);
    for (float& v : vals) v *= scale;
    return ConstFloats(vals, t.GetShape(), name);
  });
}

// Rank-4 BNSD attention core (batch dim always kept — the rank-3 form is
// silently mis-computed by the ML Drift delegate; image-path lesson).
// `q` is already pre-scaled by 1/sqrt(d_k).
TfTensor AttentionRaw(const TfTensor& q, const TfTensor& k, const TfTensor& v) {
  TfTensor scores = BatchMatMul(q, k, /*adj_x=*/false, /*adj_y=*/true);
  TfTensor attn = Softmax(scores);
  return BatchMatMul(attn, v);
}

// Multi-head attention over [B,N,C] token tensors (raw path only; the video
// decoder reuses the image decoder's verified shapes). `q` is pre-scaled.
TfTensor Mha(const TfTensor& q, const TfTensor& k, const TfTensor& v,
             int heads) {
  const auto& qs = q.GetShape();
  const auto& ks = k.GetShape();
  int b = qs[0];
  int nq = qs[1];
  int nk = ks[1];
  int c = qs[2];
  int hd = c / heads;
  TfTensor q4 = Transpose(Reshape(q, {b, nq, heads, hd}), {0, 2, 1, 3});
  TfTensor k4 = Transpose(Reshape(k, {b, nk, heads, hd}), {0, 2, 1, 3});
  TfTensor v4 = Transpose(Reshape(v, {b, nk, heads, hd}), {0, 2, 1, 3});
  TfTensor o = Transpose(AttentionRaw(q4, k4, v4), {0, 2, 1, 3});
  return Reshape(o, {b, nq, c});
}

TfTensor SamAttention(ConstCache& cache, const TfTensor& q_in,
                      const TfTensor& k_in, const TfTensor& v_in,
                      const std::string& prefix, const WeightMap& weights,
                      int heads) {
  const TfTensor& qw_t = W(weights, prefix + ".q_proj.weight");
  const TfTensor& qb_t = W(weights, prefix + ".q_proj.bias");
  const int c = qw_t.GetShape()[0];
  const int hd = c / heads;
  const float q_scale = 1.0f / std::sqrt(static_cast<float>(hd));

  TfTensor q =
      FullyConnected(q_in, ScaleConst(cache, qw_t, q_scale, prefix + ".q_proj.weight"),
                     ScaleConst(cache, qb_t, q_scale, prefix + ".q_proj.bias"));
  TfTensor k = FullyConnected(k_in, W(weights, prefix + ".k_proj.weight"),
                              W(weights, prefix + ".k_proj.bias"));
  TfTensor v = FullyConnected(v_in, W(weights, prefix + ".v_proj.weight"),
                              W(weights, prefix + ".v_proj.bias"));
  TfTensor out = Mha(q, k, v, heads);
  return FullyConnected(out, W(weights, prefix + ".out_proj.weight"),
                        W(weights, prefix + ".out_proj.bias"));
}

// 3-layer FeedForward (ReLU between layers; optional sigmoid).
TfTensor SamMlp3(const TfTensor& x, const std::string& prefix,
                 const WeightMap& weights, bool sigmoid_output) {
  TfTensor h = FullyConnected(x, W(weights, prefix + ".layers.0.weight"),
                              W(weights, prefix + ".layers.0.bias"), kActRelu);
  h = FullyConnected(h, W(weights, prefix + ".layers.1.weight"),
                     W(weights, prefix + ".layers.1.bias"), kActRelu);
  h = FullyConnected(h, W(weights, prefix + ".layers.2.weight"),
                     W(weights, prefix + ".layers.2.bias"));
  return sigmoid_output ? Logistic(h) : h;
}

// 2x transposed conv k=2/s=2 (VALID) — the image path's verified form.
TfTensor Upsample2x(const TfTensor& x, const TfTensor& weight,
                    const TfTensor& bias, int out_h, int out_w, int out_c) {
  return TransposeConv(weight, x, bias, {1, out_h, out_w, out_c},
                       kPaddingValid, 2, 2);
}

// Dense positional-encoding grid [1, G*G, 256] baked from the prompt
// encoder's Gaussian matrix (identical math to the image path).
TfTensor BakeDensePeGrid(const std::vector<float>& gaussian, int grid) {
  std::vector<float> out(static_cast<size_t>(grid) * grid * 256);
  for (int y = 0; y < grid; ++y) {
    for (int x = 0; x < grid; ++x) {
      float cx = 2.0f * ((x + 0.5f) / grid) - 1.0f;
      float cy = 2.0f * ((y + 0.5f) / grid) - 1.0f;
      size_t base = (static_cast<size_t>(y) * grid + x) * 256;
      for (int i = 0; i < 128; ++i) {
        float phase = kTwoPi * (cx * gaussian[i] + cy * gaussian[128 + i]);
        out[base + i] = std::sin(phase);
        out[base + 128 + i] = std::cos(phase);
      }
    }
  }
  return ConstFloats(out, {1, grid * grid, 256}, "dense_pe_grid");
}

// Swaps the two halves of the last axis: [x[..., d/2:], x[..., :d/2]].
// The negation of the first half is baked into the signed sin table.
TfTensor SwapHalf(const TfTensor& x) {
  const auto& s = x.GetShape();
  int d = s.back();
  std::vector<int> begin_hi(s.size(), 0);
  begin_hi.back() = d / 2;
  std::vector<int> size_lo(s.begin(), s.end());
  size_lo.back() = d / 2;
  std::vector<int> begin_lo(s.size(), 0);
  TfTensor hi = Slice(x, begin_hi, size_lo);
  TfTensor lo = Slice(x, begin_lo, size_lo);
  return Concatenation({hi, lo},
                       /*axis=*/static_cast<int>(s.size()) - 1);
}

// Rotate-half RoPE against sign-baked tables (`sin` has -sin on [:d/2]).
TfTensor Rope(const TfTensor& x, const TfTensor& cos, const TfTensor& sin) {
  return Add(Mul(x, cos), Mul(SwapHalf(x), sin));
}

// Exact step indicator in {0, 1} for x > 0 using only Relu, Mul, and Sub:
//   StepPos(x, scale) = 1 - Relu(1 - scale * Relu(x))
// Avoids Greater/GreaterEqual/Cast(bool->fp32) so fused step signatures stay
// 100% in a single WebGPU delegate partition without CPU fallback.
TfTensor StepPos(const TfTensor& x, float scale = 1e4f) {
  const TfTensor one = ConstScalar(1.0f);
  return Sub(one, Relu(Sub(one, Mul(Relu(x), ConstScalar(scale)))));
}

}  // namespace

const TfTensor& ConstCache::GetOrCreate(const std::string& key,
                                        const std::function<TfTensor()>& make) {
  auto it = entries_.find(key);
  if (it == entries_.end()) it = entries_.emplace(key, make()).first;
  return it->second;
}

MemCondInputs MakeMemCondInputs(const Sam2VideoConfig& config, int nmm) {
  const int hw = config.hw();
  const int g = config.image.embed_grid();
  MemCondInputs in;
  in.pix_raw = TfTensor({.name = "pix_raw",
                         .type = Type::kFP32,
                         .shape = {1, g, g, config.hidden}});
  in.mem_bank = TfTensor({.name = "mem_bank",
                          .type = Type::kFP32,
                          .shape = {1, nmm, hw, config.mem_ch}});
  in.slot_tpe = TfTensor({.name = "slot_tpe",
                          .type = Type::kFP32,
                          .shape = {1, nmm, 1, config.mem_ch}});
  in.ptr_tok = TfTensor({.name = "ptr_tok",
                         .type = Type::kFP32,
                         .shape = {1, 1, config.n_ptr(), config.mem_ch}});
  in.ptr_pos = TfTensor({.name = "ptr_pos",
                         .type = Type::kFP32,
                         .shape = {1, 1, config.n_ptr(), config.mem_ch}});
  in.key_mask = TfTensor({.name = "key_mask",
                          .type = Type::kFP32,
                          .shape = {1, 1, 1, config.mem_len(nmm)}});
  return in;
}

VideoDecoderInputs MakeVideoDecoderInputs(const Sam2VideoConfig& config) {
  const int g = config.image.embed_grid();
  const int mg = config.image.mask_grid();
  VideoDecoderInputs in;
  in.pix_feat = TfTensor({.name = "pix_feat_in",
                          .type = Type::kFP32,
                          .shape = {1, g, g, config.hidden}});
  in.feat_s1 = TfTensor({.name = "feat_s1",
                         .type = Type::kFP32,
                         .shape = {1, mg / 2, mg / 2, 64}});
  in.feat_s0 = TfTensor(
      {.name = "feat_s0", .type = Type::kFP32, .shape = {1, mg, mg, 32}});
  in.sparse = TfTensor(
      {.name = "sparse", .type = Type::kFP32, .shape = {1, 2, config.hidden}});
  in.nomem = TfTensor(
      {.name = "nomem", .type = Type::kFP32, .shape = {1, 1, 1, 1}});
  return in;
}

MemorizeInputs MakeMemorizeInputs(const Sam2VideoConfig& config) {
  const int g = config.image.embed_grid();
  const int s = config.image.image_size;
  MemorizeInputs in;
  in.pix_raw = TfTensor({.name = "pix_raw",
                         .type = Type::kFP32,
                         .shape = {1, g, g, config.hidden}});
  in.mask_for_mem = TfTensor({.name = "mask_for_mem",
                              .type = Type::kFP32,
                              .shape = {1, s, s, 1}});
  in.occ = TfTensor({.name = "occ", .type = Type::kFP32, .shape = {1, 1, 1}});
  return in;
}

TfTensor BuildMemCond(const Sam2VideoConfig& config, int nmm,
                      const MemCondInputs& inputs, const WeightMap& weights,
                      ConstCache* cache_in) {
  ConstCache local_cache;
  ConstCache& cache = cache_in ? *cache_in : local_cache;
  const int hw = config.hw();
  const int hd = config.hidden;
  const int mc = config.mem_ch;
  const int num_layers = config.ma_layers;
  const float scale = 1.0f / std::sqrt(static_cast<float>(hd));
  const float eps = config.ln_eps;
  const std::string ck = absl::StrCat("@", config.image.image_size);

  // Baked tables (per-resolution constants from the export).
  // Bake the rotate-half sign (-sin on [:hd/2]) into sin so runtime SwapHalf
  // needs no Neg op, and bake 1/sqrt(hd) into the query tables so attention
  // scores need no Mul op.
  const std::vector<float> raw_cos = HostFloats(W(weights, "tables.rope_cos"));
  const std::vector<float> raw_sin = HostFloats(W(weights, "tables.rope_sin"));
  std::vector<float> signed_sin = raw_sin;
  std::vector<float> scaled_cos(raw_cos.size());
  std::vector<float> scaled_sin(raw_sin.size());
  for (int t = 0; t < hw; ++t) {
    for (int c = 0; c < hd; ++c) {
      const size_t idx = static_cast<size_t>(t) * hd + c;
      if (c < hd / 2) signed_sin[idx] = -signed_sin[idx];
      scaled_cos[idx] = raw_cos[idx] * scale;
      scaled_sin[idx] = signed_sin[idx] * scale;
    }
  }

  TfTensor cos_q = cache.GetOrCreate("rope_cos_q" + ck, [&] {
    return ConstFloats(scaled_cos, {1, 1, hw, hd}, "rope_cos_q");
  });
  TfTensor sin_q = cache.GetOrCreate("rope_sin_q" + ck, [&] {
    return ConstFloats(scaled_sin, {1, 1, hw, hd}, "rope_sin_q");
  });
  TfTensor cos_k = cache.GetOrCreate("rope_cos_k" + ck, [&] {
    return ConstFloats(raw_cos, {1, 1, hw, hd}, "rope_cos_k");
  });
  TfTensor sin_k = cache.GetOrCreate("rope_sin_k" + ck, [&] {
    return ConstFloats(signed_sin, {1, 1, hw, hd}, "rope_sin_k");
  });
  TfTensor vpos = cache.GetOrCreate("vision_pos_scaled" + ck, [&] {
    return ConstFloats(HostFloats(W(weights, "tables.vision_pos_scaled")),
                       {1, 1, hw, hd}, "vision_pos_scaled");
  });
  TfTensor mem_pos = cache.GetOrCreate("mem_pos" + ck, [&] {
    return ConstFloats(HostFloats(W(weights, "tables.mem_pos")),
                       {1, 1, hw, mc}, "mem_pos");
  });

  // Queries: raw features to token-major + 0.1 * vision position encoding.
  TfTensor x = Add(Reshape(inputs.pix_raw, {1, 1, hw, hd}), vpos);

  // Memory tokens and their position rows: keep spatial [1,nmm,hw,mc] and
  // pointer [1,1,np,mc] separate for K so RoPE broadcasts contiguously over
  // [hw,hd] without slicing [1,1,len,hd].
  TfTensor spatial = Reshape(inputs.mem_bank, {1, 1, nmm * hw, mc});
  TfTensor memory = Concatenation({spatial, inputs.ptr_tok}, /*axis=*/2);
  TfTensor spatial_pos = Add(mem_pos, inputs.slot_tpe);  // [1,nmm,hw,mc]
  TfTensor keys_sp = Add(inputs.mem_bank, spatial_pos);  // [1,nmm,hw,mc]
  TfTensor keys_ptr = Add(inputs.ptr_tok, inputs.ptr_pos);  // [1,1,np,mc]

  for (int l = 0; l < num_layers; ++l) {
    const std::string p = absl::StrCat("memory_attention.layers.", l);
    // --- self attention (1/sqrt(hd) baked into cos_q/sin_q) ---
    TfTensor h = LayerNorm(x, W(weights, p + ".norm1.weight"),
                           W(weights, p + ".norm1.bias"), eps);
    TfTensor q = Rope(
        FullyConnected(h, W(weights, p + ".self_attn.q_proj.weight"),
                       W(weights, p + ".self_attn.q_proj.bias")),
        cos_q, sin_q);
    TfTensor k = Rope(
        FullyConnected(h, W(weights, p + ".self_attn.k_proj.weight"),
                       W(weights, p + ".self_attn.k_proj.bias")),
        cos_k, sin_k);
    TfTensor v = FullyConnected(h, W(weights, p + ".self_attn.v_proj.weight"),
                                W(weights, p + ".self_attn.v_proj.bias"));
    TfTensor sa = AttentionRaw(q, k, v);
    sa = FullyConnected(sa, W(weights, p + ".self_attn.out_proj.weight"),
                        W(weights, p + ".self_attn.out_proj.bias"));
    x = Add(x, sa);

    // --- cross attention to the memory (RoPE on q and spatial k only) ---
    h = LayerNorm(x, W(weights, p + ".norm2.weight"),
                  W(weights, p + ".norm2.bias"), eps);
    q = Rope(FullyConnected(h, W(weights, p + ".cross_attn_image.q_proj.weight"),
                            W(weights, p + ".cross_attn_image.q_proj.bias")),
             cos_q, sin_q);
    TfTensor k_sp =
        FullyConnected(keys_sp, W(weights, p + ".cross_attn_image.k_proj.weight"),
                       W(weights, p + ".cross_attn_image.k_proj.bias"));
    k_sp = Reshape(Rope(k_sp, cos_k, sin_k), {1, 1, nmm * hw, hd});
    TfTensor k_ptr =
        FullyConnected(keys_ptr, W(weights, p + ".cross_attn_image.k_proj.weight"),
                       W(weights, p + ".cross_attn_image.k_proj.bias"));
    k = Concatenation({k_sp, k_ptr}, /*axis=*/2);
    v = FullyConnected(memory, W(weights, p + ".cross_attn_image.v_proj.weight"),
                       W(weights, p + ".cross_attn_image.v_proj.bias"));
    TfTensor scores = BatchMatMul(q, k, /*adj_x=*/false, /*adj_y=*/true);
    scores = Add(scores, inputs.key_mask);
    TfTensor ca = BatchMatMul(Softmax(scores), v);
    ca = FullyConnected(ca, W(weights, p + ".cross_attn_image.out_proj.weight"),
                        W(weights, p + ".cross_attn_image.out_proj.bias"));
    x = Add(x, ca);

    // --- MLP (ReLU) ---
    h = LayerNorm(x, W(weights, p + ".norm3.weight"),
                  W(weights, p + ".norm3.bias"), eps);
    h = FullyConnected(h, W(weights, p + ".linear1.weight"),
                       W(weights, p + ".linear1.bias"), kActRelu);
    h = FullyConnected(h, W(weights, p + ".linear2.weight"),
                       W(weights, p + ".linear2.bias"));
    x = Add(x, h);
  }
  x = LayerNorm(x, W(weights, "memory_attention.norm.weight"),
                W(weights, "memory_attention.norm.bias"), eps);
  const int g = config.image.embed_grid();
  TfTensor out = Reshape(x, {1, g, g, hd});
  out.SetName("pix_feat");
  return out;
}

VideoDecoderOutputs BuildVideoDecoder(const Sam2VideoConfig& config,
                                      const VideoDecoderInputs& inputs,
                                      const WeightMap& weights,
                                      int mask_mode, ConstCache* cache_in) {
  ConstCache local_cache;
  ConstCache& cache = cache_in ? *cache_in : local_cache;
  const Sam2Config& img = config.image;
  const int eg = img.embed_grid();
  const int mg = img.mask_grid();
  const int n_img = eg * eg;
  const float eps = img.ln_eps_decoder;
  const std::string dec = "sam_mask_decoder";
  const std::string tr = dec + ".transformer";

  // Tokens: [obj_score | iou | mask x4 | sparse rows (k points + pad)].
  const std::string ck = absl::StrCat("@", img.image_size);
  TfTensor output_tokens = cache.GetOrCreate("output_tokens" + ck, [&] {
    std::vector<float> token_data;
    for (const char* name :
         {"sam_mask_decoder.obj_score_token.weight",
          "sam_mask_decoder.iou_token.weight",
          "sam_mask_decoder.mask_tokens.weight"}) {
      std::vector<float> rows = HostFloats(W(weights, name));
      token_data.insert(token_data.end(), rows.begin(), rows.end());
    }
    return ConstFloats(token_data, {1, 6, img.d_model}, "output_tokens");
  });
  TfTensor tokens = Concatenation({output_tokens, inputs.sparse}, /*axis=*/1);

  // Image side: pix_feat + (nomem * no_mem_embed + no_mask_dense); flatten;
  // baked dense PE. Combining the two [1,1,1,256] biases first avoids a second
  // full [1,eg,eg,256] Add pass over pix_feat.
  TfTensor no_mem = cache.GetOrCreate("no_mem_row" + ck, [&] {
    return ConstFloats(HostFloats(W(weights, "no_mem_embed")),
                       {1, 1, 1, img.d_model}, "no_mem_row");
  });
  std::vector<float> gaussian = HostFloats(W(
      weights, "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"));
  TfTensor no_mask_dense = cache.GetOrCreate("no_mask_dense" + ck, [&] {
    return ConstFloats(
        HostFloats(W(weights, "sam_prompt_encoder.no_mask_embed.weight")),
        {1, 1, 1, img.d_model}, "no_mask_dense");
  });
  TfTensor dense_bias = Add(Mul(no_mem, inputs.nomem), no_mask_dense);
  TfTensor src = Add(inputs.pix_feat, dense_bias);
  TfTensor keys = Reshape(src, {1, n_img, img.d_model});
  TfTensor kpe = cache.GetOrCreate("dense_pe_grid" + ck,
                             [&] { return BakeDensePeGrid(gaussian, eg); });

  // Two-way transformer (2 blocks + final token->image attention) — the
  // image path's verified construction, on 8 tokens.
  TfTensor queries = tokens;
  const TfTensor& qpe = tokens;
  for (int l = 0; l < 2; ++l) {
    std::string p = absl::StrCat(tr, ".layers.", l);
    if (l == 0) {
      queries = SamAttention(cache, queries, queries, queries, p + ".self_attn",
                             weights, 8);
    } else {
      TfTensor q = Add(queries, qpe);
      queries = Add(queries,
                    SamAttention(cache, q, q, queries, p + ".self_attn", weights, 8));
    }
    queries = LayerNorm(queries, W(weights, p + ".norm1.weight"),
                        W(weights, p + ".norm1.bias"), eps);

    TfTensor q = Add(queries, qpe);
    TfTensor k = Add(keys, kpe);
    queries = Add(queries, SamAttention(cache, q, k, keys,
                                        p + ".cross_attn_token_to_image",
                                        weights, 8));
    queries = LayerNorm(queries, W(weights, p + ".norm2.weight"),
                        W(weights, p + ".norm2.bias"), eps);

    TfTensor m = FullyConnected(queries, W(weights, p + ".mlp.layers.0.weight"),
                                W(weights, p + ".mlp.layers.0.bias"), kActRelu);
    m = FullyConnected(m, W(weights, p + ".mlp.layers.1.weight"),
                       W(weights, p + ".mlp.layers.1.bias"));
    queries = LayerNorm(Add(queries, m), W(weights, p + ".norm3.weight"),
                        W(weights, p + ".norm3.bias"), eps);

    q = Add(queries, qpe);
    keys = Add(keys, SamAttention(cache, k, q, queries,
                                  p + ".cross_attn_image_to_token", weights,
                                  8));
    keys = LayerNorm(keys, W(weights, p + ".norm4.weight"),
                     W(weights, p + ".norm4.bias"), eps);
  }
  TfTensor q_final = Add(queries, qpe);
  TfTensor k_final = Add(keys, kpe);
  queries = Add(queries, SamAttention(cache, q_final, k_final, keys,
                                      tr + ".final_attn_token_to_image",
                                      weights, 8));
  queries = LayerNorm(queries, W(weights, tr + ".norm_final_attn.weight"),
                      W(weights, tr + ".norm_final_attn.bias"), eps);

  // Mask upscaling with the high-res skips.
  TfTensor src_img = Reshape(keys, {1, eg, eg, img.d_model});
  TfTensor up = Upsample2x(src_img, W(weights, dec + ".output_upscaling_0.weight"),
                           W(weights, dec + ".output_upscaling_0.bias"),
                           2 * eg, 2 * eg, 64);
  up = Add(up, inputs.feat_s1);
  up = LayerNorm(up, W(weights, dec + ".output_upscaling_1.weight"),
                 W(weights, dec + ".output_upscaling_1.bias"), img.ln_eps_2d);
  up = Gelu(up);
  up = Upsample2x(up, W(weights, dec + ".output_upscaling_3.weight"),
                  W(weights, dec + ".output_upscaling_3.bias"), mg, mg, 32);
  up = Add(up, inputs.feat_s0);
  up = Gelu(up);  // [1, mg, mg, 32]

  // Heads:
  //   mask_mode == 0: all 4 masks [1,4,mg,mg], all 4 obj_ptr [1,4,256]
  //   mask_mode == 1 (multimask=true in BuildStep): masks 1..3 [1,3,mg,mg],
  //                   obj_ptr 1..3 [1,3,256] (token 0 is never selected)
  //   mask_mode == 2 (multimask=false in BuildStep): all 4 masks [1,4,mg,mg],
  //                   obj_ptr 0 only [1,1,256] (token 0's ptr is always used)
  TfTensor obj_tok = Slice(queries, {0, 0, 0}, {1, 1, img.d_model});
  TfTensor iou_tok = Slice(queries, {0, 1, 0}, {1, 1, img.d_model});
  const int first_mask = (mask_mode == 1) ? 1 : 0;
  const int num_masks = 4 - first_mask;
  std::vector<TfTensor> hyper;
  hyper.reserve(num_masks);
  for (int i = first_mask; i < 4; ++i) {
    TfTensor tok = Slice(queries, {0, 2 + i, 0}, {1, 1, img.d_model});
    hyper.push_back(SamMlp3(
        tok, absl::StrCat(dec, ".output_hypernetworks_mlps.", i), weights,
        /*sigmoid_output=*/false));  // [1,1,32]
  }
  TfTensor hyper_cat =
      num_masks == 3
          ? Concatenation({hyper[0], hyper[1], hyper[2]}, /*axis=*/1)
          : Concatenation({hyper[0], hyper[1], hyper[2], hyper[3]}, /*axis=*/1);
  TfTensor hyper_all = Reshape(hyper_cat, {1, 1, num_masks, 32});
  TfTensor up_flat = Reshape(up, {1, 1, mg * mg, 32});
  TfTensor masks = BatchMatMul(hyper_all, up_flat, /*adj_x=*/false,
                               /*adj_y=*/true);  // [1,1,num_masks,mg*mg]

  VideoDecoderOutputs out;
  out.masks = Reshape(masks, {1, num_masks, mg, mg});
  out.masks.SetName("masks");
  TfTensor iou4 = SamMlp3(iou_tok, dec + ".iou_prediction_head", weights,
                          /*sigmoid_output=*/true);  // [1,1,4]
  out.iou_scores = Reshape(iou4, {1, 4});
  out.iou_scores.SetName("iou_scores");
  TfTensor ptr_toks = (mask_mode == 1)
                          ? Slice(queries, {0, 3, 0}, {1, 3, img.d_model})
                      : (mask_mode == 2)
                          ? Slice(queries, {0, 2, 0}, {1, 1, img.d_model})
                          : Slice(queries, {0, 2, 0}, {1, 4, img.d_model});
  out.obj_ptr = SamMlp3(ptr_toks, "obj_ptr_proj", weights,
                        /*sigmoid_output=*/false);
  out.obj_ptr.SetName("obj_ptr");
  TfTensor obj = SamMlp3(obj_tok, dec + ".pred_obj_score_head", weights,
                         /*sigmoid_output=*/false);
  out.object_score = Reshape(obj, {1, 1});
  out.object_score.SetName("object_score");
  return out;
}

StepOutputs BuildStep(const Sam2VideoConfig& config,
                      const VideoDecoderInputs& dec_in,
                      const TfTensor& pix_raw, const WeightMap& weights,
                      bool multimask, int binarize_mode,
                      ConstCache* cache_in) {
  ConstCache local_cache;
  ConstCache& cache = cache_in ? *cache_in : local_cache;
  const int mg = config.image.mask_grid();
  const int plane = mg * mg;
  const int s = config.image.image_size;
  const int hd = config.hidden;
  const int mask_mode = multimask ? 1 : 2;
  VideoDecoderOutputs dec =
      BuildVideoDecoder(config, dec_in, weights, mask_mode, &cache);
  const TfTensor kOne = ConstScalar(1.0f);
  const TfTensor kTieEps = ConstScalar(1e-5f);

  // One-hot over mask tokens 1..3 where the FIRST max wins, using StepPos
  // (Relu/Mul/Sub) so there are no BOOL tensors or Cast ops.
  TfTensor s1 = Slice(dec.iou_scores, {0, 1}, {1, 1});
  TfTensor s2 = Slice(dec.iou_scores, {0, 2}, {1, 1});
  TfTensor s3 = Slice(dec.iou_scores, {0, 3}, {1, 1});
  TfTensor ge12 = StepPos(Add(Sub(s1, s2), kTieEps), 1e4f);
  TfTensor ge13 = StepPos(Add(Sub(s1, s3), kTieEps), 1e4f);
  TfTensor ge23 = StepPos(Add(Sub(s2, s3), kTieEps), 1e4f);
  TfTensor w1 = Mul(ge12, ge13);
  TfTensor w2 = Mul(Sub(kOne, w1), ge23);
  TfTensor w3 = Sub(Sub(kOne, w1), w2);

  TfTensor best_mask;
  TfTensor best_ptr;
  TfTensor best_iou;
  if (multimask) {
    TfTensor w123 =
        Reshape(Concatenation({w1, w2, w3}, /*axis=*/1), {1, 1, 1, 3});
    best_mask = Reshape(
        BatchMatMul(w123, Reshape(dec.masks, {1, 1, 3, plane}),
                    /*adj_x=*/false, /*adj_y=*/false),
        {1, 1, mg, mg});
    best_ptr = Reshape(
        BatchMatMul(w123, Reshape(dec.obj_ptr, {1, 1, 3, hd}),
                    /*adj_x=*/false, /*adj_y=*/false),
        {1, 1, hd});
    TfTensor iou13 = Slice(dec.iou_scores, {0, 1}, {1, 3});
    best_iou = Reshape(
        BatchMatMul(w123, Reshape(iou13, {1, 1, 3, 1}),
                    /*adj_x=*/false, /*adj_y=*/false),
        {1, 1});
  } else {
    // 2+ points: single-mask token 0 unless unstable (dynamic multimask via
    // stability). Mean over the last axis of a 4D tensor avoids fp16 overflow
    // and matches LayerNorm's 4D reduction layout on WebGPU.
    TfTensor m0 =
        Reshape(Slice(dec.masks, {0, 0, 0, 0}, {1, 1, mg, mg}),
                {1, 1, 1, plane});
    TfTensor gt_pos = StepPos(Add(m0, ConstScalar(-0.05f)), 1e4f);
    TfTensor gt_neg = StepPos(Add(m0, ConstScalar(0.05f)), 1e4f);
    TfTensor diff = Sub(gt_pos, Mul(gt_neg, ConstScalar(0.98f)));
    TfTensor mean_diff = Reshape(Mean(diff, {3}, /*keep_dims=*/true), {1, 1});
    TfTensor stable = StepPos(
        Add(Mul(mean_diff, ConstScalar(4096.0f)), ConstScalar(5e-4f)), 2e3f);
    TfTensor unstable = Sub(kOne, stable);
    TfTensor zero11 = cache.GetOrCreate("zero_1x1", [&] {
      return ConstFloats({0.0f}, {1, 1}, "zero_1x1");
    });
    TfTensor e0 = cache.GetOrCreate("e0_1x4", [&] {
      return ConstFloats({1.0f, 0.0f, 0.0f, 0.0f}, {1, 4}, "e0_1x4");
    });
    TfTensor w_best = Concatenation({zero11, w1, w2, w3}, /*axis=*/1);
    TfTensor w_single = Reshape(
        Add(Mul(e0, stable), Mul(w_best, unstable)), {1, 1, 1, 4});
    best_mask = Reshape(
        BatchMatMul(w_single, Reshape(dec.masks, {1, 1, 4, plane}),
                    /*adj_x=*/false, /*adj_y=*/false),
        {1, 1, mg, mg});
    best_iou = Reshape(
        BatchMatMul(w_single, Reshape(dec.iou_scores, {1, 1, 4, 1}),
                    /*adj_x=*/false, /*adj_y=*/false),
        {1, 1});
    best_ptr = dec.obj_ptr;  // [1, 1, hd] (always token 0)
  }

  // No-object handling: a = object_score > 0.
  TfTensor a = StepPos(dec.object_score, 1e4f);  // [1, 1]
  TfTensor not_a = Sub(kOne, a);
  TfTensor a4 = Reshape(a, {1, 1, 1, 1});
  TfTensor not_a4 = Reshape(not_a, {1, 1, 1, 1});
  TfTensor low = Add(Mul(best_mask, a4), Mul(not_a4, ConstScalar(-1024.0f)));
  TfTensor no_obj_ptr = cache.GetOrCreate("no_obj_ptr_row", [&] {
    return ConstFloats(HostFloats(W(weights, "no_obj_ptr")), {1, 1, hd},
                       "no_obj_ptr_row");
  });
  TfTensor ptr = Add(Mul(best_ptr, Reshape(a, {1, 1, 1})),
                     Mul(no_obj_ptr, Reshape(not_a, {1, 1, 1})));

  // Upsample low-res mask [1, 1, mg, mg] -> [1, mg, mg, 1] -> [1, s, s, 1]
  // via native ResizeBilinear (replaces 2x constant-weight FullyConnected +
  // 2x Transpose + 2x Reshape).
  TfTensor high = ResizeBilinear(Reshape(low, {1, mg, mg, 1}), {s, s},
                                 /*align_corners=*/false,
                                 /*half_pixel_centers=*/true);
  TfTensor mfm;
  if (binarize_mode == 1) {
    mfm = Sub(Mul(StepPos(high, 1e4f), ConstScalar(20.0f)), ConstScalar(10.0f));
  } else if (binarize_mode == 0) {
    mfm = Sub(Mul(Logistic(high), ConstScalar(20.0f)), ConstScalar(10.0f));
  } else {
    TfTensor prompted = dec_in.nomem;  // [1,1,1,1]
    TfTensor stepped = Mul(StepPos(high, 1e4f), ConstScalar(20.0f));
    TfTensor sig = Mul(Logistic(high), ConstScalar(20.0f));
    mfm = Sub(Add(Mul(prompted, stepped), Mul(Sub(kOne, prompted), sig)),
              ConstScalar(10.0f));
  }

  MemorizeInputs mi{.pix_raw = pix_raw, .mask_for_mem = mfm,
                    .occ = Reshape(not_a, {1, 1, 1})};
  StepOutputs out;
  out.mem = BuildMemorize(config, mi, weights, &cache);
  out.mem.SetName("mem");
  out.ptr = Reshape(ptr, {1, hd});
  out.ptr.SetName("ptr");
  out.low_mask = Reshape(low, {1, mg, mg});
  out.low_mask.SetName("low_mask");
  out.object_score = Reshape(dec.object_score, {1, 1});
  out.object_score.SetName("object_score");
  out.iou = Reshape(best_iou, {1, 1});
  out.iou.SetName("iou");
  return out;
}

TfTensor BuildMemorize(const Sam2VideoConfig& config,
                       const MemorizeInputs& inputs, const WeightMap& weights,
                       ConstCache* cache_in) {
  ConstCache local_cache;
  ConstCache& cache = cache_in ? *cache_in : local_cache;
  const int g = config.image.embed_grid();
  const int hw = config.hw();
  const float eps = config.ln_eps_2d;
  const std::string me = "memory_encoder";

  // Mask downsampler: 4x [pad1 + conv k3/s2 VALID + LN + GELU] (torch p=1
  // stride-2 sampling grid; TFLite SAME would shift it), then 1x1.
  TfTensor m = inputs.mask_for_mem;
  for (int i = 0; i < 4; ++i) {
    m = Conv2D(PadHW(m, 1, 1, 1, 1),
               W(weights, absl::StrCat(me, ".mask_downsampler.conv", i,
                                       ".weight")),
               W(weights, absl::StrCat(me, ".mask_downsampler.conv", i,
                                       ".bias")),
               /*stride_h=*/2, /*stride_w=*/2, kPaddingValid);
    m = LayerNorm(m,
                  W(weights, absl::StrCat(me, ".mask_downsampler.norm", i,
                                          ".weight")),
                  W(weights, absl::StrCat(me, ".mask_downsampler.norm", i,
                                          ".bias")),
                  eps);
    m = Gelu(m);
  }
  m = Conv2D(m, W(weights, me + ".mask_downsampler.conv4.weight"),
             W(weights, me + ".mask_downsampler.conv4.bias"), 1, 1,
             kPaddingValid);

  // Fuse with the projected pixel features.
  TfTensor f = Conv2D(inputs.pix_raw, W(weights, me + ".pix_feat_proj.weight"),
                      W(weights, me + ".pix_feat_proj.bias"), 1, 1,
                      kPaddingValid);
  f = Add(f, m);

  // 2 ConvNeXt fuser blocks: depthwise 7x7 (pad3 + VALID), LN, pw1, GELU,
  // pw2, gamma scale, residual. The depthwise filter is exported in the
  // TFLite [1,7,7,C] layout, like every other conv filter.
  for (int i = 0; i < 2; ++i) {
    const std::string p = absl::StrCat(me, ".fuser.", i);
    TfTensor t = DepthwiseConv2D(PadHW(f, 3, 3, 3, 3),
                                 W(weights, p + ".dwconv.weight"),
                                 W(weights, p + ".dwconv.bias"), 1, 1,
                                 kPaddingValid);
    t = LayerNorm(t, W(weights, p + ".norm.weight"),
                  W(weights, p + ".norm.bias"), eps);
    t = FullyConnected(t, W(weights, p + ".pwconv1.weight"),
                       W(weights, p + ".pwconv1.bias"));
    t = Gelu(t);
    t = FullyConnected(t, W(weights, p + ".pwconv2.weight"),
                       W(weights, p + ".pwconv2.bias"));
    t = Mul(t, W(weights, p + ".gamma"));
    f = Add(f, t);
  }
  f = Conv2D(f, W(weights, me + ".out_proj.weight"),
             W(weights, me + ".out_proj.bias"), 1, 1, kPaddingValid);
  (void)g;

  // Token-major + occ * occlusion embedding.
  TfTensor mem = Reshape(f, {1, hw, config.mem_ch});
  TfTensor no_obj = cache.GetOrCreate("no_obj_embed_row", [&] {
    return ConstFloats(HostFloats(W(weights, "no_obj_embed_spatial")),
                       {1, 1, config.mem_ch}, "no_obj_embed_row");
  });
  mem = Add(mem, Mul(no_obj, inputs.occ));
  mem.SetName("mem");
  return mem;
}

}  // namespace litert::tensor::examples::sam2_video

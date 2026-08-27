// SAM2.1 hiera-tiny image path — graph construction. See sam2_graph.h.

#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_graph.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensor/arithmetic.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::sam2 {

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

// Reads a loaded fp32 weight back to host floats (for build-time baking).
std::vector<float> HostFloats(const TfTensor& tensor) {
  auto buffer = tensor.GetBuffer();
  ABSL_CHECK(buffer.ok()) << "weight has no buffer";
  auto lock = buffer->Lock();
  const float* data = reinterpret_cast<const float*>(lock.data());
  return std::vector<float>(data, data + lock.size() / sizeof(float));
}

// NHWC zero padding on H and W.
TfTensor PadHW(const TfTensor& x, int top, int bottom, int left, int right) {
  TfTensor paddings(
      {.type = Type::kI32,
       .shape = {4, 2},
       .buffer = OwningCpuBuffer::Copy<Type::kI32>(
           {0, 0, top, bottom, left, right, 0, 0})});
  return Pad(x, paddings);
}

// Emission toggles (single-threaded graph construction, mirrored from the
// talker example's pattern).
bool g_layer_norm_composite = false;
bool g_sdpa_composite = false;
bool g_rbmm_attention = false;
bool g_rbmm_hypernet = false;

// odml.runtime_bmm control inputs for the CURRENT Build*() call, one per
// distinct bound length S. The ml_drift parser requires the control tensor
// to be a RUNTIME input (GetNumberOfRuntimeInputsForNode != 3 rejects the
// node outright), so these are graph inputs, not constants.
std::vector<std::pair<int, TfTensor>> g_rbmm_params;

TfTensor GetRbmmParam(int s) {
  for (const auto& [len, tensor] : g_rbmm_params) {
    if (len == s) return tensor;
  }
  TfTensor param({.name = absl::StrCat("rbmm_s", s),
                  .type = Type::kI32,
                  .shape = {1, 1, 1, 7}});
  g_rbmm_params.emplace_back(s, param);
  return param;
}

std::vector<uint8_t> RuntimeBmmAttributes(bool is_src) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() {
    fbb.Bool("is_global", true);
    fbb.Bool("is_src", is_src);
    fbb.Bool("rhs_cache_update", false);
  });
  fbb.Finish();
  return fbb.GetBuffer();
}

// The runtime_bmm contract (parser: transpose_right=true): LHS [B,H,M,K] x
// RHS [B,H,N,K] -> [B,H,M,N]; the batch dim merges into H on the delegate
// when B > 1 (window attention). Decompositions are plain BatchMatMuls that
// ignore the control tensor, so CPU execution equals the raw path.
TfTensor RuntimeBmm(const TfTensor& lhs, const TfTensor& rhs,
                    const TfTensor& param, bool is_src) {
  StableHLOCompositeOptions opts{
      .name = "odml.runtime_bmm",
      .composite_attributes = RuntimeBmmAttributes(is_src)};
  return StableHLOComposite(
      opts,
      [](TfTensor l, TfTensor r, TfTensor /*p*/) {
        return BatchMatMul(l, r, /*adj_x=*/false, /*adj_y=*/true);
      },
      lhs, rhs, param);
}

std::vector<uint8_t> EpsilonAttributes(float eps) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Float("epsilon", eps); });
  fbb.Finish();
  return fbb.GetBuffer();
}

// Last-axis LayerNorm with weight+bias. In NHWC this also covers the
// reference's LayerNorm2d (channelwise over NCHW == last axis here).
TfTensor LayerNormRaw(const TfTensor& x, const TfTensor& weight,
                      const TfTensor& bias, float eps) {
  int last = static_cast<int>(x.GetShape().size()) - 1;
  TfTensor mean = Mean(x, {last}, /*keep_dims=*/true);
  TfTensor centered = Sub(x, mean);
  TfTensor var = Mean(Mul(centered, centered), {last}, /*keep_dims=*/true);
  TfTensor normed = Mul(centered, Rsqrt(Add(var, ConstScalar(eps))));
  return Add(Mul(normed, weight), bias);
}

TfTensor LayerNorm(const TfTensor& x, const TfTensor& weight,
                   const TfTensor& bias, float eps) {
  if (!g_layer_norm_composite) return LayerNormRaw(x, weight, bias, eps);
  StableHLOCompositeOptions opts{.name = "odml.layer_norm",
                                 .composite_attributes =
                                     EpsilonAttributes(eps)};
  return StableHLOComposite(
      opts,
      [eps](TfTensor dx, TfTensor dw, TfTensor db) {
        return LayerNormRaw(dx, dw, db, eps);
      },
      x, weight, bias);
}

std::vector<uint8_t> SdpaAttributes(float scale) {
  flexbuffers::Builder fbb;
  fbb.Map([&]() { fbb.Float("scale", scale); });
  fbb.Finish();
  return fbb.GetBuffer();
}

// Rank-4 BNSD attention core: q [B,H,Nq,D], k/v [B,H,Nk,D] -> [B,H,Nq,D].
// Batch dim is always kept (the rank-3 form is silently mis-computed by the
// ML Drift GPU delegate — converted-path finding, decoder v2).
TfTensor AttentionRaw(const TfTensor& q, const TfTensor& k, const TfTensor& v,
                      float scale) {
  TfTensor scores = BatchMatMul(q, k, /*adj_x=*/false, /*adj_y=*/true);
  scores = Mul(scores, ConstScalar(scale));
  TfTensor attn = Softmax(scores);
  return BatchMatMul(attn, v);
}

// Multi-head attention over token tensors. q [B,Nq,C], k/v [B,Nk,C] with C
// = heads * head_dim -> [B,Nq,C]. Composite emission presents BSND operands
// ([B,N,H,D], the odml.scaled_dot_product_attention delegate contract); the
// decomposition transposes to BNSD and runs the same math, so CPU execution
// is identical either way.
TfTensor Mha(const TfTensor& q, const TfTensor& k, const TfTensor& v,
             int heads) {
  const auto& qs = q.GetShape();
  const auto& ks = k.GetShape();
  int b = qs[0];
  int nq = qs[1];
  int nk = ks[1];
  int c = qs[2];
  int hd = c / heads;
  float scale = 1.0f / std::sqrt(static_cast<float>(hd));

  TfTensor q4 = Reshape(q, {b, nq, heads, hd});
  TfTensor k4 = Reshape(k, {b, nk, heads, hd});
  TfTensor v4 = Reshape(v, {b, nk, heads, hd});

  TfTensor out;
  if (g_sdpa_composite) {
    StableHLOCompositeOptions opts{
        .name = "odml.scaled_dot_product_attention",
        .composite_attributes = SdpaAttributes(scale)};
    out = StableHLOComposite(
        opts,
        [scale](TfTensor dq, TfTensor dk, TfTensor dv) {
          TfTensor qt = Transpose(dq, {0, 2, 1, 3});
          TfTensor kt = Transpose(dk, {0, 2, 1, 3});
          TfTensor vt = Transpose(dv, {0, 2, 1, 3});
          TfTensor o = AttentionRaw(qt, kt, vt, scale);
          return Transpose(o, {0, 2, 1, 3});
        },
        q4, k4, v4);
  } else if (g_rbmm_attention) {
    // QK + AV odml.runtime_bmm pair with in-graph scale + softmax between.
    // Both sides bound at the full length (elem2 = nk): dst-bounded QK
    // writes every column and src-bounded AV reduces every position, so
    // the composition is complete computation — no stale-tail hazard at
    // full fill (that needs active < S).
    TfTensor qt = Transpose(q4, {0, 2, 1, 3});   // [B,H,M,D]
    TfTensor kt = Transpose(k4, {0, 2, 1, 3});   // [B,H,N,D]
    TfTensor param = GetRbmmParam(nk);
    TfTensor scores = RuntimeBmm(qt, kt, param, /*is_src=*/false);
    scores = Mul(scores, ConstScalar(scale));
    TfTensor attn = Softmax(scores);             // [B,H,M,N]
    TfTensor v4t = Transpose(v4, {0, 2, 3, 1});  // [B,H,D,N] (positions on
                                                 // channels — the AV layout)
    TfTensor o = RuntimeBmm(attn, v4t, param, /*is_src=*/true);  // [B,H,M,D]
    out = Transpose(o, {0, 2, 1, 3});
  } else {
    TfTensor qt = Transpose(q4, {0, 2, 1, 3});
    TfTensor kt = Transpose(k4, {0, 2, 1, 3});
    TfTensor vt = Transpose(v4, {0, 2, 1, 3});
    TfTensor o = AttentionRaw(qt, kt, vt, scale);
    out = Transpose(o, {0, 2, 1, 3});
  }
  return Reshape(out, {b, nq, c});
}

// <=4-D window partition: [1,H,W,C] -> [nH*nW, ws, ws, C] (+pad). Splits H
// into the batch, transposes, then splits W — this transposes row/col
// WITHIN each window, which WindowUnpartition exactly inverts; window
// attention (position already added) and the square symmetric query pooling
// are order-equivariant, so results are numerically identical to the 6-D
// reference form (proven at corr 1.0 in the converted-path work).
TfTensor WindowPartition(const TfTensor& x, int h, int w, int c, int ws,
                         int* n_h, int* n_w) {
  int pad_h = (ws - h % ws) % ws;
  int pad_w = (ws - w % ws) % ws;
  TfTensor t = x;
  if (pad_h > 0 || pad_w > 0) t = PadHW(t, 0, pad_h, 0, pad_w);
  int hp = h + pad_h;
  int wp = w + pad_w;
  *n_h = hp / ws;
  *n_w = wp / ws;
  t = Reshape(t, {*n_h, ws, wp, c});
  t = Transpose(t, {0, 2, 1, 3});  // [nH, Wp, ws, C]
  return Reshape(t, {(*n_h) * (*n_w), ws, ws, c});
}

// Exact inverse of WindowPartition at (possibly pooled) window size ws2,
// cropping any padding down to [1, h, w, C].
TfTensor WindowUnpartition(const TfTensor& windows, int n_h, int n_w, int ws2,
                           int h, int w, int c) {
  TfTensor t = Reshape(windows, {n_h, n_w * ws2, ws2, c});
  t = Transpose(t, {0, 2, 1, 3});  // [nH, ws2, nW*ws2, C]
  int hp = n_h * ws2;
  int wp = n_w * ws2;
  t = Reshape(t, {1, hp, wp, c});
  if (hp > h || wp > w) t = Slice(t, {0, 0, 0, 0}, {1, h, w, c});
  return t;
}

// One Hiera multi-scale block on an NHWC grid tensor.
TfTensor MultiScaleBlock(const Sam2Config& config,
                         const Sam2Config::BlockSpec& spec, const TfTensor& x,
                         const std::string& prefix, const WeightMap& weights) {
  const int dim = spec.dim;
  const int dim_out = spec.dim_out;
  const int h = spec.grid_in;

  TfTensor shortcut = x;
  TfTensor xn = LayerNorm(x, W(weights, prefix + ".norm1.weight"),
                          W(weights, prefix + ".norm1.bias"),
                          config.ln_eps_hiera);
  if (dim != dim_out) {
    shortcut = FullyConnected(xn, W(weights, prefix + ".proj.weight"),
                              W(weights, prefix + ".proj.bias"));
    if (spec.q_stride > 0) {
      shortcut = MaxPool2D(shortcut, spec.q_stride, spec.q_stride,
                           spec.q_stride, spec.q_stride, kPaddingValid);
    }
  }

  // Tokens for attention: windows into the batch axis, or the full grid.
  TfTensor tokens = xn;
  int n_h = 1;
  int n_w = 1;
  int batch = 1;
  int n = h * h;
  int win = spec.window;
  if (win > 0) {
    tokens = WindowPartition(tokens, h, h, dim, win, &n_h, &n_w);
    batch = n_h * n_w;
    n = win * win;
  }
  tokens = Reshape(tokens, {batch, n, dim});

  TfTensor qkv = FullyConnected(tokens, W(weights, prefix + ".attn.qkv.weight"),
                                W(weights, prefix + ".attn.qkv.bias"));
  TfTensor q = Slice(qkv, {0, 0, 0}, {batch, n, dim_out});
  TfTensor k = Slice(qkv, {0, 0, dim_out}, {batch, n, dim_out});
  TfTensor v = Slice(qkv, {0, 0, 2 * dim_out}, {batch, n, dim_out});

  int nq = n;
  if (spec.q_stride > 0) {
    int side = win > 0 ? win : h;
    q = Reshape(q, {batch, side, side, dim_out});
    q = MaxPool2D(q, spec.q_stride, spec.q_stride, spec.q_stride,
                  spec.q_stride, kPaddingValid);
    int side2 = side / spec.q_stride;
    nq = side2 * side2;
    q = Reshape(q, {batch, nq, dim_out});
  }

  TfTensor attn = Mha(q, k, v, spec.heads);
  attn = FullyConnected(attn, W(weights, prefix + ".attn.proj.weight"),
                        W(weights, prefix + ".attn.proj.bias"));

  int grid_out = spec.grid_out;
  if (win > 0) {
    int ws2 = spec.q_stride > 0 ? win / spec.q_stride : win;
    attn = Reshape(attn, {batch, ws2, ws2, dim_out});
    attn = WindowUnpartition(attn, n_h, n_w, ws2, grid_out, grid_out, dim_out);
  } else {
    attn = Reshape(attn, {1, grid_out, grid_out, dim_out});
  }

  TfTensor merged = Add(shortcut, attn);
  TfTensor m = LayerNorm(merged, W(weights, prefix + ".norm2.weight"),
                         W(weights, prefix + ".norm2.bias"),
                         config.ln_eps_hiera);
  m = FullyConnected(m, W(weights, prefix + ".mlp.layers.0.weight"),
                     W(weights, prefix + ".mlp.layers.0.bias"));
  m = Gelu(m);
  m = FullyConnected(m, W(weights, prefix + ".mlp.layers.1.weight"),
                     W(weights, prefix + ".mlp.layers.1.bias"));
  return Add(merged, m);
}

// SAM decoder attention wrapper: separate q/k/v/out projections (+bias),
// internal dim from the weight shapes (cross attention downsamples to 128).
TfTensor SamAttention(const TfTensor& q_in, const TfTensor& k_in,
                      const TfTensor& v_in, const std::string& prefix,
                      const WeightMap& weights, int heads) {
  TfTensor q = FullyConnected(q_in, W(weights, prefix + ".q_proj.weight"),
                              W(weights, prefix + ".q_proj.bias"));
  TfTensor k = FullyConnected(k_in, W(weights, prefix + ".k_proj.weight"),
                              W(weights, prefix + ".k_proj.bias"));
  TfTensor v = FullyConnected(v_in, W(weights, prefix + ".v_proj.weight"),
                              W(weights, prefix + ".v_proj.bias"));
  TfTensor out = Mha(q, k, v, heads);
  return FullyConnected(out, W(weights, prefix + ".out_proj.weight"),
                        W(weights, prefix + ".out_proj.bias"));
}

// 3-layer SamMLP (ReLU between layers; optional sigmoid on the output).
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

// 2x transposed conv (k=2, s=2, VALID). `transpose_conv` emits
// TRANSPOSE_CONV; otherwise the exact expansion: with k==s the taps never
// overlap, so out[2y+dy, 2x+dx, o] = sum_i x[y,x,i] * W[o,dy,dx,i] — four
// 1x1 projections interleaved by DepthToSpace (channel order (dy*2+dx)*O+o
// = the TFLite D2S contract). Same sums, bit-identical on CPU; for
// delegates that reject TRANSPOSE_CONV (Mali ML Drift).
TfTensor Upsample2x(const TfTensor& x, const TfTensor& weight,
                    const TfTensor& bias, int out_h, int out_w, int out_c,
                    bool transpose_conv) {
  if (transpose_conv) {
    return TransposeConv(weight, x, bias, {1, out_h, out_w, out_c},
                         kPaddingValid, 2, 2);
  }
  std::vector<float> w = HostFloats(weight);  // [O, 2, 2, I]
  const auto& ws = weight.GetShape();
  int o_ch = ws[0];
  int i_ch = ws[3];
  std::vector<TfTensor> taps;
  for (int dy = 0; dy < 2; ++dy) {
    for (int dx = 0; dx < 2; ++dx) {
      std::vector<float> tap(static_cast<size_t>(o_ch) * i_ch);
      for (int o = 0; o < o_ch; ++o) {
        for (int i = 0; i < i_ch; ++i) {
          tap[static_cast<size_t>(o) * i_ch + i] =
              w[((static_cast<size_t>(o) * 2 + dy) * 2 + dx) * i_ch + i];
        }
      }
      taps.push_back(FullyConnected(
          x, ConstFloats(tap, {o_ch, i_ch}), std::optional<TfTensor>()));
    }
  }
  TfTensor packed = Concatenation({taps[0], taps[1], taps[2], taps[3]},
                                  /*axis=*/3);  // [1,H,W,4*O]
  TfTensor up = DepthToSpace(packed, /*block_size=*/2);
  return Add(up, bias);
}

// Dense positional-encoding grid [1, G*G, 256] baked from the prompt
// encoder's Gaussian matrix (identical math to the reference denseGrid).
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

}  // namespace

std::vector<WeightSpec> GetWeightSpecs(const Sam2Config& config) {
  std::vector<WeightSpec> specs;
  auto add = [&](const std::string& name, const std::vector<int>& shape,
                 float scale = 0.02f) {
    specs.push_back({name, shape, scale});
  };

  // Trunk.
  add("trunk.patch_embed.proj.weight", {config.embed_dim, 7, 7, 3});
  add("trunk.patch_embed.proj.bias", {config.embed_dim});
  add("trunk.pos_embed_full",
      {1, config.token_grid(), config.token_grid(), config.embed_dim});
  auto blocks = config.Blocks();
  for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
    const auto& b = blocks[i];
    std::string p = absl::StrCat("trunk.blocks.", i);
    add(p + ".norm1.weight", {b.dim}, 1.0f);
    add(p + ".norm1.bias", {b.dim}, 0.0f);
    add(p + ".attn.qkv.weight", {3 * b.dim_out, b.dim});
    add(p + ".attn.qkv.bias", {3 * b.dim_out}, 0.0f);
    add(p + ".attn.proj.weight", {b.dim_out, b.dim_out});
    add(p + ".attn.proj.bias", {b.dim_out}, 0.0f);
    if (b.dim != b.dim_out) {
      add(p + ".proj.weight", {b.dim_out, b.dim});
      add(p + ".proj.bias", {b.dim_out}, 0.0f);
    }
    add(p + ".norm2.weight", {b.dim_out}, 1.0f);
    add(p + ".norm2.bias", {b.dim_out}, 0.0f);
    int hidden = b.dim_out * config.mlp_ratio;
    add(p + ".mlp.layers.0.weight", {hidden, b.dim_out});
    add(p + ".mlp.layers.0.bias", {hidden}, 0.0f);
    add(p + ".mlp.layers.1.weight", {b.dim_out, hidden});
    add(p + ".mlp.layers.1.bias", {b.dim_out}, 0.0f);
  }

  // Neck (convs.0 = deepest stage, per the reference ordering).
  std::vector<int> stage_ends = config.StageEnds();
  std::vector<int> backbone_channels;
  for (auto it = stage_ends.rbegin(); it != stage_ends.rend(); ++it) {
    backbone_channels.push_back(blocks[*it].dim_out);
  }
  for (int i = 0; i < static_cast<int>(backbone_channels.size()); ++i) {
    add(absl::StrCat("neck.convs.", i, ".weight"),
        {config.d_model, 1, 1, backbone_channels[i]});
    add(absl::StrCat("neck.convs.", i, ".bias"), {config.d_model}, 0.0f);
  }
  add("no_mem_embed", {1, 1, config.d_model}, 0.0f);

  // Prompt encoder (point path only).
  add("sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix",
      {2, 128}, 1.0f);
  for (int i = 0; i < 4; ++i) {
    add(absl::StrCat("sam_prompt_encoder.point_embeddings.", i, ".weight"),
        {1, config.d_model});
  }
  add("sam_prompt_encoder.not_a_point_embed.weight", {1, config.d_model});
  add("sam_prompt_encoder.no_mask_embed.weight", {1, config.d_model});

  // Mask decoder.
  add("sam_mask_decoder.conv_s0.weight", {32, 1, 1, config.d_model});
  add("sam_mask_decoder.conv_s0.bias", {32}, 0.0f);
  add("sam_mask_decoder.conv_s1.weight", {64, 1, 1, config.d_model});
  add("sam_mask_decoder.conv_s1.bias", {64}, 0.0f);
  add("sam_mask_decoder.iou_token.weight", {1, config.d_model});
  add("sam_mask_decoder.mask_tokens.weight", {4, config.d_model});
  add("sam_mask_decoder.obj_score_token.weight", {1, config.d_model});
  for (int l = 0; l < 2; ++l) {
    std::string p = absl::StrCat("sam_mask_decoder.transformer.layers.", l);
    for (const std::string& attn :
         {std::string(".self_attn"), std::string(".cross_attn_token_to_image"),
          std::string(".cross_attn_image_to_token")}) {
      int internal = attn == ".self_attn" ? 256 : 128;
      for (const std::string& proj : {std::string("q"), std::string("k"),
                                      std::string("v")}) {
        add(p + attn + "." + proj + "_proj.weight", {internal, 256});
        add(p + attn + "." + proj + "_proj.bias", {internal}, 0.0f);
      }
      add(p + attn + ".out_proj.weight", {256, internal});
      add(p + attn + ".out_proj.bias", {256}, 0.0f);
    }
    add(p + ".mlp.layers.0.weight", {2048, 256});
    add(p + ".mlp.layers.0.bias", {2048}, 0.0f);
    add(p + ".mlp.layers.1.weight", {256, 2048});
    add(p + ".mlp.layers.1.bias", {256}, 0.0f);
    for (int n = 1; n <= 4; ++n) {
      add(absl::StrCat(p, ".norm", n, ".weight"), {256}, 1.0f);
      add(absl::StrCat(p, ".norm", n, ".bias"), {256}, 0.0f);
    }
  }
  for (const std::string& proj :
       {std::string("q"), std::string("k"), std::string("v")}) {
    add("sam_mask_decoder.transformer.final_attn_token_to_image." + proj +
            "_proj.weight",
        {128, 256});
    add("sam_mask_decoder.transformer.final_attn_token_to_image." + proj +
            "_proj.bias",
        {128}, 0.0f);
  }
  add("sam_mask_decoder.transformer.final_attn_token_to_image.out_proj.weight",
      {256, 128});
  add("sam_mask_decoder.transformer.final_attn_token_to_image.out_proj.bias",
      {256}, 0.0f);
  add("sam_mask_decoder.transformer.norm_final_attn.weight", {256}, 1.0f);
  add("sam_mask_decoder.transformer.norm_final_attn.bias", {256}, 0.0f);
  add("sam_mask_decoder.output_upscaling_0.weight", {64, 2, 2, 256});
  add("sam_mask_decoder.output_upscaling_0.bias", {64}, 0.0f);
  add("sam_mask_decoder.output_upscaling_1.weight", {64}, 1.0f);
  add("sam_mask_decoder.output_upscaling_1.bias", {64}, 0.0f);
  add("sam_mask_decoder.output_upscaling_3.weight", {32, 2, 2, 64});
  add("sam_mask_decoder.output_upscaling_3.bias", {32}, 0.0f);
  for (int i = 1; i <= 3; ++i) {  // hypernetwork 0 feeds the dropped mask
    std::string p =
        absl::StrCat("sam_mask_decoder.output_hypernetworks_mlps.", i);
    add(p + ".layers.0.weight", {256, 256});
    add(p + ".layers.0.bias", {256}, 0.0f);
    add(p + ".layers.1.weight", {256, 256});
    add(p + ".layers.1.bias", {256}, 0.0f);
    add(p + ".layers.2.weight", {32, 256});
    add(p + ".layers.2.bias", {32}, 0.0f);
  }
  for (const std::string& head :
       {std::string("sam_mask_decoder.iou_prediction_head"),
        std::string("sam_mask_decoder.pred_obj_score_head")}) {
    int out = head == "sam_mask_decoder.iou_prediction_head" ? 4 : 1;
    add(head + ".layers.0.weight", {256, 256});
    add(head + ".layers.0.bias", {256}, 0.0f);
    add(head + ".layers.1.weight", {256, 256});
    add(head + ".layers.1.bias", {256}, 0.0f);
    add(head + ".layers.2.weight", {out, 256});
    add(head + ".layers.2.bias", {out}, 0.0f);
  }
  return specs;
}

WeightMap MakeSyntheticWeights(const Sam2Config& config, unsigned seed) {
  WeightMap weights;
  unsigned state = seed;
  for (const WeightSpec& spec : GetWeightSpecs(config)) {
    size_t count = 1;
    for (int d : spec.shape) count *= static_cast<size_t>(d);
    std::vector<float> data(count);
    for (size_t i = 0; i < count; ++i) {
      state = state * 1664525u + 1013904223u;
      data[i] = spec.init_scale * (static_cast<float>(state >> 8) /
                                       static_cast<float>(1u << 24) * 2.0f -
                                   1.0f);
    }
    weights[spec.name] = ConstFloats(data, spec.shape, spec.name);
  }
  return weights;
}

EncoderInputs MakeEncoderInputs(const Sam2Config& config) {
  EncoderInputs inputs;
  inputs.pixels =
      TfTensor({.name = "pixels",
                .type = Type::kFP32,
                .shape = {1, config.image_size, config.image_size, 3}});
  return inputs;
}

DecoderInputs MakeDecoderInputs(const Sam2Config& config) {
  DecoderInputs inputs;
  int eg = config.embed_grid();
  int mg = config.mask_grid();
  inputs.image_embeddings = TfTensor({.name = "image_embeddings",
                                      .type = Type::kFP32,
                                      .shape = {1, eg, eg, config.d_model}});
  inputs.feat_s1 = TfTensor({.name = "feat_s1",
                             .type = Type::kFP32,
                             .shape = {1, mg / 2, mg / 2, 64}});
  inputs.feat_s0 = TfTensor(
      {.name = "feat_s0", .type = Type::kFP32, .shape = {1, mg, mg, 32}});
  inputs.point_coords = TfTensor(
      {.name = "point_coords", .type = Type::kFP32, .shape = {1, 1, 2}});
  return inputs;
}

std::vector<std::pair<int, TfTensor>> TakeRbmmParams() {
  std::vector<std::pair<int, TfTensor>> params = std::move(g_rbmm_params);
  g_rbmm_params.clear();
  return params;
}

EncoderOutputs BuildEncoder(const Sam2Config& config,
                            const EncoderInputs& inputs,
                            const WeightMap& weights) {
  g_layer_norm_composite = config.use_layer_norm_composite;
  g_sdpa_composite = config.use_sdpa_composite;
  g_rbmm_attention = config.use_rbmm_attention;
  g_rbmm_hypernet = config.use_rbmm_hypernet;

  // Patch embed: explicit 3px pad + VALID 7x7/s4 (torch padding semantics —
  // TFLite SAME would pad 1+2 at stride 4 and shift the sampling grid).
  TfTensor x = PadHW(inputs.pixels, 3, 3, 3, 3);
  x = Conv2D(x, W(weights, "trunk.patch_embed.proj.weight"),
             W(weights, "trunk.patch_embed.proj.bias"), /*stride_h=*/4,
             /*stride_w=*/4, kPaddingValid);
  x = Add(x, W(weights, "trunk.pos_embed_full"));

  auto blocks = config.Blocks();
  auto stage_ends = config.StageEnds();
  std::vector<TfTensor> stage_outputs;
  for (int i = 0; i < static_cast<int>(blocks.size()); ++i) {
    x = MultiScaleBlock(config, blocks[i], x, absl::StrCat("trunk.blocks.", i),
                        weights);
    for (int e : stage_ends) {
      if (e == i) stage_outputs.push_back(x);
    }
  }

  // FPN neck: convs.0 pairs with the deepest stage. Only the second-deepest
  // level receives a top-down add (fpn_top_down_levels = {2,3}; the deepest
  // is its own start). scalp=1 drops the deepest output.
  int n_levels = static_cast<int>(stage_outputs.size());  // 4
  std::vector<TfTensor> laterals(n_levels);
  for (int i = 0; i < n_levels; ++i) {
    // stage_outputs is shallow->deep; convs index is deep->shallow.
    laterals[i] =
        Conv2D(stage_outputs[i],
               W(weights, absl::StrCat("neck.convs.", n_levels - 1 - i,
                                       ".weight")),
               W(weights, absl::StrCat("neck.convs.", n_levels - 1 - i,
                                       ".bias")),
               1, 1, kPaddingValid);
  }
  int eg = config.embed_grid();
  TfTensor size_const({.type = Type::kI32,
                       .shape = {2},
                       .buffer = OwningCpuBuffer::Copy<Type::kI32>({eg, eg})});
  TfTensor top_down = ResizeNearestNeighbor(laterals[3], size_const);
  TfTensor fpn2 = Add(laterals[2], top_down);

  EncoderOutputs outputs;
  // no_mem_embed folded in (the "no memory" image-only conditioning).
  // Re-baked at the broadcast shape: a graph Reshape of a constant is
  // rejected by the GPU delegate ("expected 1 runtime input").
  TfTensor no_mem = ConstFloats(HostFloats(W(weights, "no_mem_embed")),
                                {1, 1, 1, config.d_model}, "no_mem_row");
  outputs.image_embeddings = Add(fpn2, no_mem);
  outputs.image_embeddings.SetName("image_embeddings");
  outputs.feat_s1 =
      Conv2D(laterals[1], W(weights, "sam_mask_decoder.conv_s1.weight"),
             W(weights, "sam_mask_decoder.conv_s1.bias"), 1, 1, kPaddingValid);
  outputs.feat_s1.SetName("feat_s1");
  outputs.feat_s0 =
      Conv2D(laterals[0], W(weights, "sam_mask_decoder.conv_s0.weight"),
             W(weights, "sam_mask_decoder.conv_s0.bias"), 1, 1, kPaddingValid);
  outputs.feat_s0.SetName("feat_s0");
  return outputs;
}

DecoderOutputs BuildDecoder(const Sam2Config& config,
                            const DecoderInputs& inputs,
                            const WeightMap& weights) {
  g_layer_norm_composite = config.use_layer_norm_composite;
  g_sdpa_composite = config.use_sdpa_composite;
  g_rbmm_attention = config.use_rbmm_attention;
  g_rbmm_hypernet = config.use_rbmm_hypernet;

  const int eg = config.embed_grid();
  const int mg = config.mask_grid();
  const int n_img = eg * eg;
  const float eps = config.ln_eps_decoder;
  const std::string dec = "sam_mask_decoder";
  const std::string tr = dec + ".transformer";

  // --- Sparse prompt: one positive point + the baked not_a_point pad ---
  // (the reference embed_points with the label logic constant-folded for
  // the fixed [positive, pad] layout; matches upstream to float rounding).
  std::vector<float> gaussian = HostFloats(W(
      weights, "sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix"));
  // FC weight [128, 2]: out k = x * g[0][k] + y * g[1][k].
  std::vector<float> gauss_t(256);
  for (int k = 0; k < 128; ++k) {
    gauss_t[2 * k] = gaussian[k];
    gauss_t[2 * k + 1] = gaussian[128 + k];
  }
  TfTensor c = Mul(Add(inputs.point_coords, ConstScalar(0.5f)),
                   ConstScalar(1.0f / config.image_size));
  c = Add(Mul(c, ConstScalar(2.0f)), ConstScalar(-1.0f));
  TfTensor proj =
      FullyConnected(c, ConstFloats(gauss_t, {128, 2}, "pe_gaussian_t"),
                     std::optional<TfTensor>());
  proj = Mul(proj, ConstScalar(kTwoPi));
  TfTensor pe = Concatenation({Sin(proj), Cos(proj)}, /*axis=*/2);
  // Constant rows re-baked at rank 3 (constant-input Reshape is rejected
  // by the GPU delegate).
  TfTensor tok0 = Add(
      pe, ConstFloats(
              HostFloats(
                  W(weights, "sam_prompt_encoder.point_embeddings.1.weight")),
              {1, 1, config.d_model}, "point_embed_pos"));
  TfTensor tok1 = ConstFloats(
      HostFloats(W(weights, "sam_prompt_encoder.not_a_point_embed.weight")),
      {1, 1, config.d_model}, "not_a_point");
  TfTensor sparse = Concatenation({tok0, tok1}, /*axis=*/1);  // [1,2,256]

  // --- Tokens: [obj_score | iou | mask x4 | sparse x2] ---
  std::vector<float> token_data;
  for (const char* name :
       {"sam_mask_decoder.obj_score_token.weight",
        "sam_mask_decoder.iou_token.weight",
        "sam_mask_decoder.mask_tokens.weight"}) {
    std::vector<float> rows = HostFloats(W(weights, name));
    token_data.insert(token_data.end(), rows.begin(), rows.end());
  }
  TfTensor output_tokens =
      ConstFloats(token_data, {1, 6, config.d_model}, "output_tokens");
  TfTensor tokens = Concatenation({output_tokens, sparse}, /*axis=*/1);

  // --- Image side: += no-mask dense prompt row; flatten; baked dense PE ---
  std::vector<float> no_mask =
      HostFloats(W(weights, "sam_prompt_encoder.no_mask_embed.weight"));
  TfTensor src = Add(inputs.image_embeddings,
                     ConstFloats(no_mask, {1, 1, 1, config.d_model},
                                 "no_mask_dense"));
  TfTensor keys = Reshape(src, {1, n_img, config.d_model});
  TfTensor kpe = BakeDensePeGrid(gaussian, eg);

  // --- Two-way transformer (2 blocks + final token->image attention) ---
  TfTensor queries = tokens;
  const TfTensor& qpe = tokens;
  for (int l = 0; l < 2; ++l) {
    std::string p = absl::StrCat(tr, ".layers.", l);
    if (l == 0) {
      // skip_first_layer_pe: plain self-attention REPLACES the queries.
      queries = SamAttention(queries, queries, queries, p + ".self_attn",
                            weights, 8);
    } else {
      TfTensor q = Add(queries, qpe);
      queries = Add(queries,
                    SamAttention(q, q, queries, p + ".self_attn", weights, 8));
    }
    queries = LayerNorm(queries, W(weights, p + ".norm1.weight"),
                        W(weights, p + ".norm1.bias"), eps);

    TfTensor q = Add(queries, qpe);
    TfTensor k = Add(keys, kpe);
    queries = Add(queries, SamAttention(q, k, keys,
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
    k = Add(keys, kpe);
    keys = Add(keys, SamAttention(k, q, queries,
                                  p + ".cross_attn_image_to_token", weights,
                                  8));
    keys = LayerNorm(keys, W(weights, p + ".norm4.weight"),
                     W(weights, p + ".norm4.bias"), eps);
  }
  TfTensor q_final = Add(queries, qpe);
  TfTensor k_final = Add(keys, kpe);
  queries = Add(queries, SamAttention(q_final, k_final, keys,
                                      tr + ".final_attn_token_to_image",
                                      weights, 8));
  queries = LayerNorm(queries, W(weights, tr + ".norm_final_attn.weight"),
                      W(weights, tr + ".norm_final_attn.bias"), eps);

  // --- Mask upscaling (transposed convs; LayerNorm2d == last-axis LN in
  // NHWC) with the high-res skip features ---
  TfTensor src_img = Reshape(keys, {1, eg, eg, config.d_model});
  const bool tc = !config.use_d2s_upsampler;
  TfTensor up = Upsample2x(src_img, W(weights, dec + ".output_upscaling_0.weight"),
                           W(weights, dec + ".output_upscaling_0.bias"),
                           2 * eg, 2 * eg, 64, tc);
  up = Add(up, inputs.feat_s1);
  up = LayerNorm(up, W(weights, dec + ".output_upscaling_1.weight"),
                 W(weights, dec + ".output_upscaling_1.bias"),
                 config.ln_eps_2d);
  up = Gelu(up);
  up = Upsample2x(up, W(weights, dec + ".output_upscaling_3.weight"),
                  W(weights, dec + ".output_upscaling_3.bias"), mg, mg, 32, tc);
  up = Add(up, inputs.feat_s0);
  up = Gelu(up);  // [1, mg, mg, 32]

  // --- Heads ---
  TfTensor obj_tok = Slice(queries, {0, 0, 0}, {1, 1, config.d_model});
  TfTensor iou_tok = Slice(queries, {0, 1, 0}, {1, 1, config.d_model});
  std::vector<TfTensor> hyper;
  for (int i = 1; i <= 3; ++i) {
    TfTensor mask_tok =
        Slice(queries, {0, 2 + i, 0}, {1, 1, config.d_model});
    hyper.push_back(SamMlp3(
        mask_tok, absl::StrCat(dec, ".output_hypernetworks_mlps.", i), weights,
        /*sigmoid_output=*/false));  // [1,1,32]
  }
  TfTensor hyper_all = Concatenation({hyper[0], hyper[1], hyper[2]},
                                     /*axis=*/1);  // [1,3,32]
  TfTensor up_flat = Reshape(up, {1, mg * mg, 32});
  TfTensor masks;
  if (g_rbmm_hypernet) {
    // The activation x activation projection as one dst-bounded
    // runtime_bmm at rank 4 (elem2 = mg*mg = full width).
    TfTensor h4 = Reshape(hyper_all, {1, 1, 3, 32});
    TfTensor u4 = Reshape(up_flat, {1, 1, mg * mg, 32});
    TfTensor param = GetRbmmParam(mg * mg);
    TfTensor m4 = RuntimeBmm(h4, u4, param, /*is_src=*/false);
    masks = Reshape(m4, {1, 3, mg * mg});
  } else {
    masks = BatchMatMul(hyper_all, up_flat, /*adj_x=*/false,
                        /*adj_y=*/true);  // [1,3,mg*mg]
  }

  DecoderOutputs outputs;
  outputs.masks = Reshape(masks, {1, 3, mg, mg});
  outputs.masks.SetName("masks");
  TfTensor iou4 = SamMlp3(iou_tok, dec + ".iou_prediction_head", weights,
                          /*sigmoid_output=*/true);  // [1,1,4]
  outputs.iou_scores = Reshape(Slice(iou4, {0, 0, 1}, {1, 1, 3}), {1, 3});
  outputs.iou_scores.SetName("iou_scores");
  TfTensor obj = SamMlp3(obj_tok, dec + ".pred_obj_score_head", weights,
                         /*sigmoid_output=*/false);  // [1,1,1]
  outputs.object_score = Reshape(obj, {1, 1});
  outputs.object_score.SetName("object_score");
  return outputs;
}

}  // namespace litert::tensor::examples::sam2

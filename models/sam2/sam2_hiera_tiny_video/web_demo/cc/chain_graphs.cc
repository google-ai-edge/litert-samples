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

#include "chain_graphs.h"

#include <algorithm>
#include <array>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "models/sam2/sam2_hiera_tiny_video/tensor_api/sam2_image/sam2_graph.h"
#include "tensor/arithmetic.h"
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "tensor/buffer.h"
#include "tensor/datatypes.h"
#include "tensor/tensor.h"

namespace sam2_chain {
namespace {

using ::litert::tensor::OwningCpuBuffer;
using ::litert::tensor::TensorHandle;
using ::litert::tensor::Type;
namespace sam2 = ::litert::tensor::examples::sam2;

TfTensor Const(const std::vector<float>& values, std::vector<int> shape) {
  return TfTensor({.type = Type::kFP32,
                   .shape = std::move(shape),
                   .buffer = OwningCpuBuffer::Copy<Type::kFP32>(values)});
}
TfTensor Scalar(float v) { return Const({v}, {1}); }
TfTensor Input(const std::string& name, std::vector<int> shape) {
  return TfTensor(
      {.name = name, .type = Type::kFP32, .shape = std::move(shape)});
}
TfTensor Clamp01(const TfTensor& x) {
  return Minimum(Maximum(x, Scalar(0.f)), Scalar(1.f));
}

absl::Status AddSig(ModelFactory& factory, const std::vector<TfTensor>& ins,
                    const std::vector<TfTensor>& outs,
                    const std::string& name) {
  std::vector<TensorHandle> in_handles(ins.begin(), ins.end());
  std::vector<TensorHandle> out_handles(outs.begin(), outs.end());
  return factory.AddSignature(in_handles, out_handles, name);
}

std::vector<TfTensor> StepInputs(const s2v::VideoDecoderInputs& dec,
                                 const TfTensor& pix_raw) {
  return {pix_raw, dec.feat_s1, dec.feat_s0, dec.nomem};
}

}  // namespace

std::string PromptSignature(int clicks) { return absl::StrCat("prompt", clicks); }
std::string TrackSignature(int nmm) { return absl::StrCat("track", nmm); }
std::string MemInput(int slot) { return absl::StrCat("mem_", slot); }
std::string PtrInput(int slot) { return absl::StrCat("ptr_", slot); }
std::string MaskInput(int object) { return absl::StrCat("mask_", object); }

PoolFactors PreprocessPool(const FrameGeometry& geo, int image_size) {
  return {std::max(1, geo.height / image_size),
          std::max(1, geo.width / image_size)};
}

std::array<float, kFxCount> EffectParams(Effect effect, float stroke_px) {
  std::array<float, kFxCount> fx{};
  fx[kFxStroke] = stroke_px;
  switch (effect) {
    case Effect::kOverlay:
      fx[kFxFill] = 0.5f;
      fx[kFxRing] = stroke_px > 0 ? 0.95f : 0.f;
      break;
    case Effect::kSpotlight:
      fx[kFxMatte] = 1.f;
      fx[kFxSpot] = 1.f;
      break;
    case Effect::kCutout:
      fx[kFxMatte] = 1.f;
      break;
  }
  return fx;
}

absl::Status AddSam2Signatures(ModelFactory& factory,
                               const s2v::Sam2VideoConfig& video_config,
                               const WeightMap& weights,
                               const std::vector<float>& track_sparse) {
  // The encoder emits the raw top-level map; the decoder adds the no-memory
  // row itself on the conditioning frame (nomem input).
  s2v::Sam2VideoConfig config = video_config;
  config.image.fold_no_mem_embed = false;
  const sam2::Sam2Config& img = config.image;
  const int g = img.embed_grid();
  const int hw = config.hw();
  // One cache for the whole model: signatures share each baked constant.
  s2v::ConstCache cache;

  sam2::EncoderInputs enc_in = sam2::MakeEncoderInputs(img);
  sam2::EncoderOutputs enc_out = sam2::BuildEncoder(img, enc_in, weights);
  enc_out.image_embeddings.SetName("pix_raw");
  if (auto st = AddSig(factory, enc_in.AsList(), enc_out.AsList(), "encode");
      !st.ok()) {
    return st;
  }

  // prompt{k}: k positive / negative clicks + the not-a-point pad. SAM 2's
  // multimask rule: best of tokens 1..3 for one click, token 0 (unless
  // unstable) for 2+.
  for (int k = 1; k <= kMaxClicks; ++k) {
    s2v::VideoDecoderInputs in = s2v::MakeVideoDecoderInputs(config);
    TfTensor raw = Input("pix_raw", {1, g, g, img.d_model});
    in.pix_feat = raw;
    in.sparse = Input("sparse", {1, k + 1, kHidden});
    s2v::StepOutputs out =
        s2v::BuildStep(config, in, raw, weights, /*multimask=*/k == 1,
                       /*binarize_mode=*/1, &cache);
    std::vector<TfTensor> ins = StepInputs(in, raw);
    ins.push_back(in.sparse);
    if (auto st = AddSig(factory, ins, out.AsList(), PromptSignature(k));
        !st.ok()) {
      return st;
    }
  }

  // track{n}: memory slots / pointers as separate inputs, bank built in-graph.
  // Concatenate first along axis 1 and Reshape once (saves n+14 Reshape ops).
  for (int n : {2, 7}) {
    s2v::MemCondInputs mc = s2v::MakeMemCondInputs(config, n);
    std::vector<TfTensor> mems, ptrs;
    mems.reserve(n);
    ptrs.reserve(kNumPtrFrames);
    for (int k = 0; k < n; ++k) {
      mems.push_back(Input(MemInput(k), {1, hw, kMemCh}));
    }
    for (int k = 0; k < kNumPtrFrames; ++k) {
      ptrs.push_back(Input(PtrInput(k), {1, kHidden}));
    }
    mc.mem_bank = Reshape(Concatenation(absl::MakeSpan(mems), /*axis=*/1),
                          {1, n, hw, kMemCh});
    mc.ptr_tok = Reshape(Concatenation(absl::MakeSpan(ptrs), /*axis=*/1),
                         {1, 1, kNumPtrFrames * kPtrSplit, kMemCh});

    s2v::VideoDecoderInputs dec = s2v::MakeVideoDecoderInputs(config);
    dec.pix_feat = s2v::BuildMemCond(config, n, mc, weights, &cache);
    dec.sparse = Const(track_sparse, {1, 2, kHidden});
    s2v::StepOutputs out = s2v::BuildStep(config, dec, mc.pix_raw, weights,
                                          /*multimask=*/true,
                                          /*binarize_mode=*/0, &cache);

    std::vector<TfTensor> ins = StepInputs(dec, mc.pix_raw);
    ins.insert(ins.end(), mems.begin(), mems.end());
    ins.insert(ins.end(), ptrs.begin(), ptrs.end());
    ins.push_back(mc.slot_tpe);
    ins.push_back(mc.ptr_pos);
    ins.push_back(mc.key_mask);
    if (auto st = AddSig(factory, ins, out.AsList(), TrackSignature(n));
        !st.ok()) {
      return st;
    }
  }
  return absl::OkStatus();
}

absl::Status AddFrameSignatures(ModelFactory& factory,
                                const FrameGeometry& geo, int image_size) {
  const int H = geo.height, W = geo.width, S = image_size;
  const int mg = S / 4;

  // ---- preprocess
  {
    TfTensor frame = Input("frame", {1, H, geo.stride, 4});
    TfTensor rgb = Slice(frame, {0, 0, 0, 0}, {1, H, W, 3});
    const PoolFactors pool = PreprocessPool(geo, S);
    if (pool.ky > 1 || pool.kx > 1) {
      rgb = AveragePool2D(rgb, pool.ky, pool.kx, pool.ky, pool.kx,
                          ::litert::tensor::kPaddingValid);
    }
    TfTensor resized = ResizeBilinear(rgb, {S, S}, /*align_corners=*/false,
                                      /*half_pixel_centers=*/true);
    // (x - mean) / std with x in [0,1]: one multiply-add per channel.
    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float std_[3] = {0.229f, 0.224f, 0.225f};
    std::vector<float> scale(3), bias(3);
    for (int c = 0; c < 3; ++c) {
      scale[c] = 1.f / std_[c];
      bias[c] = -mean[c] / std_[c];
    }
    TfTensor pixels = Add(Mul(resized, Const(scale, {3})), Const(bias, {3}));
    pixels.SetName("pixels");
    if (auto st = AddSig(factory, {frame}, {pixels}, "preprocess"); !st.ok()) {
      return st;
    }
  }

  // ---- composite
  TfTensor frame = Input("frame", {1, H, geo.stride, 4});
  std::vector<TfTensor> masks, cols;
  for (int k = 0; k < kMaxObjects; ++k) {
    masks.push_back(Input(MaskInput(k), {1, mg, mg}));
    cols.push_back(Reshape(masks.back(), {1, mg, mg, 1}));
  }
  TfTensor fx = Input("fx", {1, 1, 1, kFxCount});
  auto param = [&](int i) { return Slice(fx, {0, 0, 0, i}, {1, 1, 1, 1}); };

  TfTensor rgb = Slice(frame, {0, 0, 0, 0}, {1, H, W, 3});
  // SAM 2's post-processing: bilinear, align_corners=False, straight to the
  // frame size (the model space is a squashed S x S).
  TfTensor logits = ResizeBilinear(Concatenation(absl::MakeSpan(cols), 3),
                                   {H, W}, false, true);  // [1,H,W,K]
  // Signed distance to the boundary in display px: d = l / |grad l|, with
  // central differences over an edge-replicated border (zero padding would
  // fake a gradient at the frame edge and draw an outline along it).
  std::vector<TfTensor> wcols = {Slice(logits, {0, 0, 0, 0}, {1, H, 1, kMaxObjects}),
                                 logits,
                                 Slice(logits, {0, 0, W - 1, 0}, {1, H, 1, kMaxObjects})};
  TfTensor padw = Concatenation(absl::MakeSpan(wcols), 2);  // [1,H,W+2,K]
  std::vector<TfTensor> hrows = {Slice(padw, {0, 0, 0, 0}, {1, 1, W + 2, kMaxObjects}),
                                 padw,
                                 Slice(padw, {0, H - 1, 0, 0}, {1, 1, W + 2, kMaxObjects})};
  TfTensor padded = Concatenation(absl::MakeSpan(hrows), 1);  // [1,H+2,W+2,K]
  std::vector<float> kx(9 * kMaxObjects, 0.f), ky(9 * kMaxObjects, 0.f);
  for (int c = 0; c < kMaxObjects; ++c) {
    kx[(1 * 3 + 0) * kMaxObjects + c] = -0.5f;  // [1,3,3,K] layout
    kx[(1 * 3 + 2) * kMaxObjects + c] = 0.5f;
    ky[(0 * 3 + 1) * kMaxObjects + c] = -0.5f;
    ky[(2 * 3 + 1) * kMaxObjects + c] = 0.5f;
  }
  const TfTensor zero_bias =
      Const(std::vector<float>(kMaxObjects, 0.f), {kMaxObjects});
  TfTensor gx = DepthwiseConv2D(padded, Const(kx, {1, 3, 3, kMaxObjects}),
                                zero_bias, 1, 1,
                                ::litert::tensor::kPaddingValid);
  TfTensor gy = DepthwiseConv2D(padded, Const(ky, {1, 3, 3, kMaxObjects}),
                                zero_bias, 1, 1,
                                ::litert::tensor::kPaddingValid);
  TfTensor grad = Sqrt(Add(Mul(gx, gx), Mul(gy, gy)));
  // Clamped to +-1000 px: an absent object (-1024 everywhere, zero gradient)
  // must not overflow fp16.
  TfTensor d = Minimum(
      Maximum(Div(logits, Maximum(grad, Scalar(1e-3f))), Scalar(-1000.f)),
      Scalar(1000.f));
  TfTensor cov = Clamp01(Add(d, Scalar(0.5f)));  // anti-aliased coverage

  // Overlay: per object a colour fill (fill * cov) under an outline ring of
  // `stroke` px centred on the boundary, composited premultiplied "over" in
  // object order — the WebGL renderer of the original demo, in-graph.
  TfTensor fill = param(kFxFill), ring = param(kFxRing);
  TfTensor half_stroke = Add(Mul(param(kFxStroke), Scalar(0.5f)), Scalar(0.5f));
  TfTensor sa_all = Mul(ring, Clamp01(Sub(half_stroke, Abs(d))));
  TfTensor fa_all = Mul(fill, cov);
  TfTensor fa_under_all = Mul(fa_all, Sub(Scalar(1.f), sa_all));
  TfTensor one_minus_a_all = Sub(Scalar(1.f), Add(sa_all, fa_under_all));
  TfTensor over = rgb;
  for (int k = 0; k < kMaxObjects; ++k) {
    TfTensor sa = Slice(sa_all, {0, 0, 0, k}, {1, H, W, 1});
    TfTensor fa_under = Slice(fa_under_all, {0, 0, 0, k}, {1, H, W, 1});
    TfTensor one_minus_a = Slice(one_minus_a_all, {0, 0, 0, k}, {1, H, W, 1});
    const Rgb& c = kPalette[k];
    TfTensor premul = Add(Mul(fa_under, Const({c[0], c[1], c[2]}, {3})), sa);
    over = Add(premul, Mul(over, one_minus_a));
  }

  // Matte: union coverage cuts the objects out over a dimmed grey frame
  // (spotlight: CSS grayscale(0.85) brightness(0.35)) or green (cutout).
  TfTensor u = ReduceMax(cov, {3}, /*keep_dims=*/true);  // [1,H,W,1]
  const float a = 1.f - 0.85f, b = 0.35f;
  const std::vector<float> grey = {
      b * (0.2126f + 0.7874f * a), b * (0.7152f - 0.7152f * a), b * (0.0722f - 0.0722f * a),
      b * (0.2126f - 0.2126f * a), b * (0.7152f + 0.2848f * a), b * (0.0722f - 0.0722f * a),
      b * (0.2126f - 0.2126f * a), b * (0.7152f - 0.7152f * a), b * (0.0722f + 0.9278f * a)};
  TfTensor dim = FullyConnected(rgb, Const(grey, {3, 3}));
  TfTensor spot = param(kFxSpot);
  TfTensor green =
      Const({kCutoutGreen[0], kCutoutGreen[1], kCutoutGreen[2]}, {3});
  TfTensor bg = Add(Mul(dim, spot), Mul(green, Sub(Scalar(1.f), spot)));
  TfTensor matte = Add(Mul(rgb, u), Mul(bg, Sub(Scalar(1.f), u)));

  TfTensor m = param(kFxMatte);
  TfTensor out = Add(Mul(matte, m), Mul(over, Sub(Scalar(1.f), m)));
  out.SetName("rgb");
  std::vector<TfTensor> ins = {frame};
  ins.insert(ins.end(), masks.begin(), masks.end());
  ins.push_back(fx);
  return AddSig(factory, ins, {out}, "composite");
}

}  // namespace sam2_chain

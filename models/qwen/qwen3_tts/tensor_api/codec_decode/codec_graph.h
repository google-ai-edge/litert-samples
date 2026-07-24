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

// Codec decoder step-graph (Tensor API, TfLite backend).
//
// One call decodes a fixed window of codec frames to waveform:
//   codes [16, T] kI32  ->  wav [1, T*1920] kFP32 in [-1, 1] at 24 kHz.
//
// Pipeline (mirrors qwen3tts_work/qtok12 modeling, all in-graph):
//   RVQ decode (16 x 1-D-indices Gather on precomputed EMA tables + two
//   1x1 output projections, semantic + acoustic sum)
//   -> causal pre_conv k3 (512 -> 1024)
//   -> pre_transformer: input_proj -> 8 x [x += scale_a*attn(rms(x));
//      x += scale_m*mlp(rms(x))] with rope(theta 1e4) and a baked
//      causal+sliding-window(72) additive mask -> norm -> output_proj
//   -> 2 x [TransposeConv k=s=2 + ConvNeXt block (causal dw k7, LayerNorm,
//      pw 1024->4096 GELU 4096->1024, gamma, residual)]
//   -> conv k7 (1024 -> 1536) -> 4 decoder blocks (SnakeBeta, causal
//      TransposeConv k=2r s=r, 3 residual units at dilations 1/3/9)
//      1536 -> 768 -> 384 -> 192 -> 96
//   -> SnakeBeta -> conv k7 (96 -> 1) -> clamp(-1, 1).
//
// The whole stack is causal, so the host runs production-style chunking:
// 64-frame chunks with 25 frames of left context, cropping ctx*1920 samples.
// Weights come preprocessed from LoadCodecWeights (TFLite filter layouts,
// EMA tables divided out, Snake exp() folded); see codec_weights.h.

#ifndef SPEECHLM_STAGE1_CODEC_DECODE_CODEC_GRAPH_H_
#define SPEECHLM_STAGE1_CODEC_DECODE_CODEC_GRAPH_H_

#include <string>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "tensor/backends/tflite/arithmetic_tflite.h"
#include "models/qwen/qwen3_tts/tensor_api/codec_decode/codec_config.h"
#include "tensor/tensor.h"

namespace litert::tensor::examples::codec {

using TfTensor = ::litert::tensor::Tensor<::litert::tensor::TfLiteMixinTag>;
using WeightMap = absl::flat_hash_map<std::string, TfTensor>;

// Input: codes [num_quantizers, frames] kI32, named "codes".
TfTensor MakeCodesInput(const CodecConfig& config, int frames);

// Output: wav [1, frames * 1920] kFP32, named "wav".
TfTensor BuildCodecDecode(const CodecConfig& config, const TfTensor& codes,
                          int frames, const WeightMap& weights);

}  // namespace litert::tensor::examples::codec

#endif  // SPEECHLM_STAGE1_CODEC_DECODE_CODEC_GRAPH_H_

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

// Qwen3-TTS-Tokenizer-12Hz codec DECODER config, from
// speech_tokenizer/config.json (decoder_config) of
// Qwen/Qwen3-TTS-12Hz-0.6B-Base.

#ifndef SPEECHLM_STAGE1_CODEC_DECODE_CODEC_CONFIG_H_
#define SPEECHLM_STAGE1_CODEC_DECODE_CODEC_CONFIG_H_

#include <vector>

namespace litert::tensor::examples::codec {

struct CodecConfig {
  // RVQ (SplitResidualVectorQuantizer): 1 semantic + 15 acoustic codebooks,
  // EMA codebooks (embedding = embedding_sum / clamp(cluster_usage, eps)).
  int num_quantizers = 16;
  int codebook_size = 2048;
  int vq_dim = 256;        // internal codebook dim (codebook_dim / 2)
  int codebook_dim = 512;  // RVQ output dim (after output_proj)
  float vq_eps = 1e-5f;

  // pre_transformer (8L, MHA, rope, sliding-window causal, LayerScale).
  int tf_hidden = 512;
  int tf_layers = 8;
  int tf_heads = 16;
  int tf_head_dim = 64;
  int tf_ffn = 1024;
  float tf_eps = 1e-5f;
  float rope_theta = 10000.0f;
  int sliding_window = 72;

  // Conv stack.
  int latent_dim = 1024;
  int decoder_dim = 1536;
  std::vector<int> upsampling_ratios = {2, 2};   // TransConv k=s + ConvNeXt
  std::vector<int> upsample_rates = {8, 5, 4, 3};  // decoder blocks

  // Host chunking (production: 64-frame chunks with 25 frames left context).
  int chunk_frames = 64;
  int context_frames = 25;

  int total_upsample() const {
    int p = 1;
    for (int r : upsample_rates) p *= r;
    for (int r : upsampling_ratios) p *= r;
    return p;  // 1920: 12.5 Hz frames -> 24 kHz samples
  }
  int block_in_dim(int i) const { return decoder_dim >> i; }
  int block_out_dim(int i) const { return decoder_dim >> (i + 1); }
  int output_dim() const {
    return decoder_dim >> static_cast<int>(upsample_rates.size());  // 96
  }
};

}  // namespace litert::tensor::examples::codec

#endif  // SPEECHLM_STAGE1_CODEC_DECODE_CODEC_CONFIG_H_

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

// Talker step-graph — Stage 2 config, aligned with the real
// Qwen/Qwen3-TTS-12Hz-0.6B-Base checkpoint (config.json talker_config +
// code_predictor_config + model.safetensors tensor shapes).

#ifndef SPEECHLM_STAGE1_TALKER_STEP_TALKER_CONFIG_H_
#define SPEECHLM_STAGE1_TALKER_STEP_TALKER_CONFIG_H_

namespace litert::tensor::examples::talker {

struct TalkerConfig {
  // Talker backbone (qwen3_tts_talker, 0.6B).
  int emb_dim = 1024;
  int n_layers = 28;
  int n_heads = 16;
  int n_kv_groups = 8;
  int head_dim = 128;
  int hidden_dim = 3072;
  float rms_norm_eps = 1e-6f;
  float rope_base = 1000000.0f;

  // Sequence / cache.
  int max_seq = 1024;
  int prefill_len = 64;

  // Codec token space: the talker predicts group 0 over 3072 ids (2048 codec
  // codes + specials, e.g. eos 2150); the code predictor fills groups 1..15
  // over 2048.
  int talker_vocab = 3072;
  int codec_vocab = 2048;
  int codec_eos_id = 2150;
  int num_code_groups = 16;

  // Code predictor (qwen3_tts_talker_code_predictor): 5-layer transformer,
  // same head geometry as the talker, per-group embeddings and lm_heads.
  int cp_layers = 5;

  // 1920-sample codec hop @ 24 kHz = 80 ms/frame; the "12Hz" model name
  // rounds down. RTF budgets must use 80 ms, not 83.3.
  double frame_rate_hz = 12.5;

  // Backbone attention via odml.runtime_bmm composites (the "avbound"
  // composition: QK unbounded so the fused mask epilogue covers the full
  // width, AV side runtime-bounded). Changes the value-cache layout to
  // [1, kv, head_dim, max_seq] and adds runtime control-tensor inputs.
  bool use_runtime_bmm = false;

  // Emit every RMS norm as the odml.rms_norm composite (epsilon attr,
  // constant 1-D scale input; decomposition = the raw MEAN/RSQRT/MUL math,
  // so CPU execution is unchanged). Fused kernels exist on Metal and CL.
  bool use_rms_norm_composite = false;

  int qkv_out_dim() const { return n_heads * head_dim; }
  int kv_out_dim() const { return n_kv_groups * head_dim; }
  int num_sub_groups() const { return num_code_groups - 1; }
  // Code predictor sequence: [talker hidden, cb0 embed, 14 sub embeds] — the
  // 15th sub code never feeds back in-frame.
  int cp_max_seq() const { return num_code_groups + 1; }
};

}  // namespace litert::tensor::examples::talker

#endif  // SPEECHLM_STAGE1_TALKER_STEP_TALKER_CONFIG_H_

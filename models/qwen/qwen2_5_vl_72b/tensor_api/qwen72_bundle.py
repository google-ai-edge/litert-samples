#!/usr/bin/env python3
# Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""qwen72_bundle.py --dec DIR --emb-path EMBEDDER.tflite --vis VISION_DIR --g 448 --tok tokenizer.json --out-dir DIR
                    [--name NAME] [--decode model.tflite] [--venc ...] [--vadp ...] [--dec-act fp32] [--max-tokens 4096]

Assembles the Qwen2.5-VL-72B-Instruct fast_vlm .litertlm:
  EMBEDDER       token_ids [1,1] -> [1,1,8192] (embedder_export.py, then int8 per channel)
  PREFILL_DECODE the Tensor API decoder (graph/qwen2_writer.cc): embeddings + per-layer K / V caches -> logits [1,1,152064]
  VISION_ENCODER image NHWC [1,G,G,3] in [0,1] -> [1,(G/28)^2,5120] (qwen25vl_vision.py)
  VISION_ADAPTER [1,N,5120] -> [1,N,8192]
  LlmMetadata    fast_vlm (image GxG); no start token (the template renders none); stop <|im_end|> 151645 and
                 <|endoftext|> 151643 (generation_config.json); structured ChatML prompt templates; the jinja =
                 qwen72_chat_template.jinja (its token ids equal the HF processor's on 8 conversations);
                 sampler TOP_P k 1 = greedy (the vendor's generation config samples at temperature 1e-6).
The tokenizer section is the model repo's tokenizer.json, byte for byte.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SOFT = "<image_soft_token>"
IM_END, EOT_TEXT = 151645, 151643


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(64 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dec", required=True)
    ap.add_argument("--emb-path", required=True)
    ap.add_argument("--vis", required=True)
    ap.add_argument("--g", type=int, default=448)
    ap.add_argument("--tok", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name")
    ap.add_argument("--decode", default="model.tflite")
    ap.add_argument("--venc")
    ap.add_argument("--vadp")
    ap.add_argument("--dec-act", default="fp32")
    # an explicit activation type for the vision sections (fp32 = what LiteRT-LM v0.19.0's GPU path already falls back
    # to when none is declared: SetCommonGpuOptions value_or(FLOAT32))
    ap.add_argument("--vis-act", default="fp32")
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--jinja", default=os.path.join(HERE, "qwen72_chat_template.jinja"))
    a = ap.parse_args()

    import litert_lm_builder as lb
    from litert_lm_builder.runtime.proto import llm_metadata_pb2, llm_model_type_pb2, sampler_params_pb2
    from tokenizers import Tokenizer

    venc = a.venc or f"vision_encoder_g{a.g}_fp32.tflite"
    vadp = a.vadp or f"vision_adapter_g{a.g}_fp32.tflite"
    parts = {"embedder": a.emb_path, "prefill_decode": os.path.join(a.dec, a.decode),
             "vision_encoder": os.path.join(a.vis, venc), "vision_adapter": os.path.join(a.vis, vadp)}
    for p in list(parts.values()) + [a.tok, a.jinja]:
        assert os.path.exists(p), p
    tk = Tokenizer.from_file(a.tok)
    for t, i in (("<|im_end|>", IM_END), ("<|endoftext|>", EOT_TEXT), ("<|im_start|>", 151644),
                 ("<|vision_start|>", 151652), ("<|vision_end|>", 151653), ("<|image_pad|>", 151655)):
        assert tk.token_to_id(t) == i, (t, tk.token_to_id(t), i)
    assert tk.token_to_id(SOFT) is None, "the soft token must not be a vocab entry"
    jinja = open(a.jinja).read()

    md = llm_metadata_pb2.LlmMetadata()
    md.max_num_tokens = a.max_tokens
    md.stop_tokens.add().token_ids.ids.append(IM_END)
    md.stop_tokens.add().token_ids.ids.append(EOT_TEXT)
    md.prompt_templates.user.prefix = "<|im_start|>user\n"
    md.prompt_templates.user.suffix = "<|im_end|>\n"
    md.prompt_templates.model.prefix = "<|im_start|>assistant\n"
    md.prompt_templates.model.suffix = "<|im_end|>\n"
    md.prompt_templates.system.prefix = "<|im_start|>system\n"
    md.prompt_templates.system.suffix = "<|im_end|>\n"
    md.jinja_prompt_template = jinja
    md.llm_model_type.CopyFrom(llm_model_type_pb2.LlmModelType(fast_vlm=llm_model_type_pb2.FastVlm()))
    md.llm_model_type.fast_vlm.image_tensor_height = a.g
    md.llm_model_type.fast_vlm.image_tensor_width = a.g
    md.sampler_params.type = sampler_params_pb2.SamplerParameters.TOP_P
    md.sampler_params.k = 1
    md.sampler_params.p = 1.0
    md.sampler_params.temperature = 1.0
    os.makedirs(a.out_dir, exist_ok=True)
    name = a.name or f"Qwen2.5-VL-72B-Instruct_g{a.g}.litertlm"
    md_path = os.path.join(a.out_dir, name + ".llm_metadata.pb")
    with open(md_path, "wb") as f:
        f.write(md.SerializeToString())
    with open(md_path + ".txt", "w") as f:
        f.write(str(md).replace(jinja, "<jinja: %d chars>" % len(jinja)))

    b = lb.LitertLmFileBuilder()
    b.add_system_metadata(lb.Metadata(key="Authors", value="", dtype=lb.DType.STRING))
    b.add_llm_metadata(md_path)
    b.add_hf_tokenizer(a.tok)
    b.add_tflite_model(parts["embedder"], "embedder")
    b.add_tflite_model(parts["prefill_decode"], "prefill_decode",
                       prefer_activation_type=a.dec_act or None)
    b.add_tflite_model(parts["vision_encoder"], "vision_encoder", prefer_activation_type=a.vis_act or None)
    b.add_tflite_model(parts["vision_adapter"], "vision_adapter", prefer_activation_type=a.vis_act or None)
    out = os.path.join(a.out_dir, name)
    with open(out, "wb") as f:
        b.build(f)
    rec = {"bundle": out, "bytes": os.path.getsize(out), "g": a.g, "max_tokens": a.max_tokens, "dec_act": a.dec_act, "vis_act": a.vis_act,
           "sections": {k: {"path": v, "bytes": os.path.getsize(v)} for k, v in parts.items()},
           "tokenizer_sha256": sha256(a.tok), "jinja_sha256": hashlib.sha256(jinja.encode()).hexdigest(),
           "stop": [IM_END, EOT_TEXT], "start": None}
    json.dump(rec, open(out + ".build.json", "w"), indent=1)
    print("BUNDLE_DONE", out, rec["bytes"], "B")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
"""lm_run.py BUNDLE --backend cpu|gpu [--vision-backend cpu|gpu] [--image PNG] --prompt TEXT [--system TEXT]
             [--extra JSON] [--max-tokens N] [--greedy] [--cache-dir DIR] [--out OUT.jsonl] [--tag TAG]

One run of a .litertlm through LiteRT-LM's Python API in this process (fresh Engine, benchmark on): the user message
is [image item, text item] (or the text alone), an optional system message goes in as the conversation preface,
--extra is the template's extra context (e.g. {"reasoning_strength": "low", "current_date": "2026-10-29"}).
--greedy = SamplerConfig(top_k=1) (the bundle's sampler otherwise). Prints and appends one JSON line: the reply
text, the wall clock, LiteRT-LM's BenchmarkInfo, the process's max RSS.
"""
import argparse
import json
import os
import resource
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bundle")
    ap.add_argument("--backend", default="cpu")
    ap.add_argument("--vision-backend")
    ap.add_argument("--image")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system")
    ap.add_argument("--extra")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--cache-dir")
    ap.add_argument("--out")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    import litert_lm

    be = {"cpu": litert_lm.interfaces.Backend.CPU, "gpu": litert_lm.interfaces.Backend.GPU}
    kw = {"backend": be[a.backend](), "enable_benchmark": True}
    if a.image:
        kw["vision_backend"] = be[a.vision_backend or a.backend]()
        kw["max_num_images"] = 1
    if a.cache_dir:
        kw["cache_dir"] = a.cache_dir
    t0 = time.time()
    engine = litert_lm.Engine(a.bundle, **kw)
    conv_kw = {"max_output_tokens": a.max_tokens}
    if a.system:
        conv_kw["messages"] = [{"role": "system", "content": a.system}]
    if a.extra:
        conv_kw["extra_context"] = json.loads(a.extra)
    if a.greedy:
        conv_kw["sampler_config"] = litert_lm.SamplerConfig(top_k=1)
    conv = engine.create_conversation(**conv_kw)
    t1 = time.time()
    content = ([{"type": "image", "path": os.path.abspath(a.image)}] if a.image else []) + \
              [{"type": "text", "text": a.prompt}]
    first, text = None, ""
    for chunk in conv.send_message_async({"role": "user", "content": content}):
        if first is None:
            first = time.time()
        c = chunk.get("content")
        if isinstance(c, list):
            for part in c:
                text += part.get("text", "")
        elif isinstance(c, str):
            text += c
    t2 = time.time()
    bi = conv.get_benchmark_info()
    rec = {"tag": a.tag, "bundle": os.path.basename(a.bundle), "backend": a.backend,
           "vision_backend": a.vision_backend or (a.backend if a.image else None), "image": a.image,
           "prompt": a.prompt, "system": a.system, "extra": a.extra, "greedy": a.greedy, "max_tokens": a.max_tokens,
           "engine_and_conversation_s": round(t1 - t0, 2), "wall_first_chunk_s": round(first - t1, 3) if first else None,
           "wall_total_s": round(t2 - t1, 3),
           "bench": {k: getattr(bi, k, None) for k in ("init_time_in_second", "time_to_first_token_in_second",
                                                       "last_prefill_token_count", "last_prefill_tokens_per_second",
                                                       "last_decode_token_count", "last_decode_tokens_per_second")},
           "max_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "text": text}
    line = json.dumps(rec, ensure_ascii=False)
    print(line, flush=True)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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
"""Exports the synthesized talker checkpoint to LiteRT prefill/decode graphs.

Produces (with externalized embedder, so both signatures take embeddings —
the host aggregates the 16 codebook embeddings plus text conditioning):
  * prefill_32 / prefill_128: embeddings + KV -> KV
  * decode: embeddings [1,1,1024] + KV -> logits [1,1,4096] + KV
    (logits[..., :3072] codec logits, logits[..., 3072:] last hidden state)

Usage (convert env):
    python export_talker.py                # fp32
    RECIPE=BOCTAV4 python export_talker.py # blockwise-32 OCTAV int4 weights
"""

import copy
import os

# BOCTAV4 = blockwise-32 OCTAV int4 + int8 EMBEDDING_LOOKUP. Channelwise
# int8/int4 (the tooling default) makes Qwen3-family decoders degenerate over
# long generations; blockwise granularity is required.
import ai_edge_quantizer.recipe as recipe_lib

RECIPE = os.environ.get('RECIPE', '')  # '' -> fp32
CACHE = int(os.environ.get('CACHE', '1024'))
PREFILL = [int(x) for x in os.environ.get('PREFILL', '32,128').split(',')]
OUT = os.environ.get(
    'OUT', 'out/talker-' + (RECIPE.lower() if RECIPE else 'fp32'))

_INT4 = recipe_lib.dynamic_wi4_afp32()[0]
_INT8_EMB = copy.deepcopy(_INT4)
_INT8_EMB['op_config']['weight_tensor_config']['num_bits'] = 8
_INT8_EMB['operation'] = 'EMBEDDING_LOOKUP'
_BLOCK_OCTAV4 = copy.deepcopy(_INT4)
_BLOCK_OCTAV4['algorithm_key'] = recipe_lib.AlgorithmName.OCTAV
_BLOCK_OCTAV4['op_config']['weight_tensor_config'][
    'granularity'] = 'BLOCKWISE_32'
recipe_lib.BOCTAV4 = lambda: [_BLOCK_OCTAV4, _INT8_EMB]

from litert_torch.generative.export_hf.export import export  # noqa: E402


def main() -> None:
    """Exports the talker prefill/decode graphs with the selected recipe."""
    os.makedirs(OUT, exist_ok=True)
    export(
        model='out/talker-llm',
        output_dir=OUT,
        quantization_recipe=RECIPE if RECIPE else None,
        externalize_embedder=True,
        single_token_embedder=True,
        cache_length=CACHE,
        prefill_lengths=PREFILL,
        bundle_litert_lm=False,
        keep_temporary_files=True,
        use_jinja_template=False,
        trust_remote_code=True,
    )
    print('files:', sorted(os.listdir(OUT)))


if __name__ == '__main__':
    main()

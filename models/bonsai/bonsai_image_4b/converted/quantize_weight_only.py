"""Weight-only quantization for the text encoder.

`add_weight_only_config` inserts an explicit DEQUANTIZE between the quantized
weight and the op, so the matmul runs in float and activations are never
quantized. Per ai-edge-quantizer's own docs, this generally preserves more
quality than dynamic-range quantization (no precision loss on activations) at
some latency cost -- on this model it is what recovers the conditioning
fidelity (see the sample README).

Usage: quantize_weight_only.py <src.tflite> <out.tflite> <bits> <block|0=channelwise> [regex]
"""
import os
import sys
import time

from ai_edge_quantizer import quantizer, recipe_manager
from ai_edge_quantizer.algorithm_manager import AlgorithmName
from ai_edge_quantizer.qtyping import QuantGranularity as G
from ai_edge_quantizer.qtyping import TFLOperationName as OP

src, out, bits, block = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
regex = sys.argv[5] if len(sys.argv) > 5 else ".*"
gran = {0: G.CHANNELWISE, 32: G.BLOCKWISE_32, 128: G.BLOCKWISE_128}[block]

rm = recipe_manager.RecipeManager()
rm.add_weight_only_config(regex=regex, operation_name=OP.FULLY_CONNECTED,
                          num_bits=bits, granularity=gran,
                          algorithm_key=AlgorithmName.MIN_MAX_UNIFORM_QUANT)
rm.add_weight_only_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP,
                          num_bits=8, granularity=G.CHANNELWISE)

t0 = time.time()
quantizer.Quantizer(src, rm.get_quantization_recipe()).quantize().export_model(out)
print(f"weight-only int{bits}/{'cw' if block == 0 else f'b{block}'} "
      f"{time.time()-t0:.0f}s -> {os.path.getsize(out)/2**30:.3f} GiB", flush=True)

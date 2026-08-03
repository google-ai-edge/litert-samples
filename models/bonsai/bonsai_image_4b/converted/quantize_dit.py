"""Quantize the exported Bonsai Image DiT: ternary block linears -> int4 blockwise,
everything else (modulation / embedders / time embed / proj_out, all NON-ternary)
-> int8 channelwise, mirroring the shipped LLM recipe's treatment of lm_head."""
import os, sys, time
from ai_edge_quantizer import quantizer, recipe_manager
from ai_edge_quantizer.algorithm_manager import AlgorithmName
from ai_edge_quantizer.qtyping import QuantGranularity as G
from ai_edge_quantizer.qtyping import TFLOperationName as OP

SRC = sys.argv[1] if len(sys.argv) > 1 else "dit_fp32.tflite"
BLOCK = int(os.environ.get("BLOCK", "32"))
OUT = os.path.basename(SRC).replace("_fp32.tflite", "") + f"_int4b{BLOCK}.tflite"
GRAN = {32: G.BLOCKWISE_32, 128: G.BLOCKWISE_128}[BLOCK]

rm = recipe_manager.RecipeManager()
# non-ternary tensors first; the ternary block linears override below
rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED,
                      num_bits=8, granularity=G.CHANNELWISE)
rm.add_dynamic_config(regex=".*TransformerBlock_.*", operation_name=OP.FULLY_CONNECTED,
                      num_bits=4, granularity=GRAN,
                      algorithm_key=AlgorithmName.MIN_MAX_UNIFORM_QUANT)

t0 = time.time()
qt = quantizer.Quantizer(SRC, rm.get_quantization_recipe())
res = qt.quantize()
print(f"quantize OK in {time.time()-t0:.0f}s", flush=True)
res.export_model(OUT)
print(f"b{BLOCK} -> {os.path.getsize(OUT)/2**30:.3f} GiB "
      f"(from {os.path.getsize(SRC)/2**30:.2f} GiB fp32)", flush=True)

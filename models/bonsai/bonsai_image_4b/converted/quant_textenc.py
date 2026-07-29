"""Text encoder is a stock fp16 Qwen3-4B (NOT ternary) -> normal int4 lane:
FULLY_CONNECTED int4 blockwise-128 (the Qwen3-4B recipe we ship), embedding int8."""
import os, time
from ai_edge_quantizer import quantizer, recipe_manager
from ai_edge_quantizer.qtyping import QuantGranularity as G
from ai_edge_quantizer.qtyping import TFLOperationName as OP
rm = recipe_manager.RecipeManager()
rm.add_dynamic_config(regex=".*", operation_name=OP.FULLY_CONNECTED,
                      num_bits=4, granularity=G.BLOCKWISE_128)
rm.add_dynamic_config(regex=".*", operation_name=OP.EMBEDDING_LOOKUP,
                      num_bits=8, granularity=G.CHANNELWISE)
t0=time.time()
quantizer.Quantizer("textenc_fp32.tflite", rm.get_quantization_recipe()).quantize().export_model("textenc_int4.tflite")
print(f"OK {time.time()-t0:.0f}s -> {os.path.getsize('textenc_int4.tflite')/2**30:.3f} GiB")

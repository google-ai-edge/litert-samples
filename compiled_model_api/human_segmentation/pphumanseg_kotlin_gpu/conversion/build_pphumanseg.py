"""Convert PP-HumanSeg (OpenCV Zoo, Apache-2.0) to a LiteRT CompiledModel-GPU tflite via onnx2tf.

    pip install onnx2tf onnx onnxsim tf_keras psutil ai_edge_litert
    python build_pphumanseg.py

Downloads the OpenCV-Zoo ONNX and runs onnx2tf (pure CNN -> GPU-compatible NHWC tflite, no patches).
Input  [1,192,192,3] NHWC, BGR, (x/255-0.5)/0.5.   Output [1,192,192,2] softmax (argmax -> human mask).
"""
import subprocess, shutil
from huggingface_hub import hf_hub_download
onnx = hf_hub_download("opencv/human_segmentation_pphumanseg", "human_segmentation_pphumanseg_2023mar.onnx")
shutil.copy(onnx, "pphs.onnx")
subprocess.run(["onnx2tf", "-i", "pphs.onnx", "-o", "tfout"], check=True)
shutil.copy("tfout/pphs_float32.tflite", "pphumanseg.tflite")
print("saved pphumanseg.tflite")

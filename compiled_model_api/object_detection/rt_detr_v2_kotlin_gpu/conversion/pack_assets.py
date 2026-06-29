# Package the converted RT-DETRv2-S graphs + host params into the Android module.
# Prereqs (run in ~/Downloads/meeting/rtdetr-work or any dir with the build scripts on PYTHONPATH):
#   python build_rtdetr_split.py fp16    # -> rtB_fp16.tflite  (Graph B, plain decoder)
#   python build_rtdetr_fix3.py  fp16    # -> rtA_fix3_fp16.tflite + host3_w.npz  (Graph A, the device fix)
# Then:
#   python pack_assets.py <work_dir> <module_dir>
# Copies the 2 tflites under the app's expected names and writes assets/host_params.bin + coco_labels.txt.
import sys, os, shutil
import numpy as np

work = sys.argv[1] if len(sys.argv) > 1 else "."
mod = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
assets = os.path.join(mod, "android", "app/src/main/assets") if os.path.isdir(os.path.join(mod, "android")) \
    else os.path.join(mod, "app/src/main/assets")
os.makedirs(assets, exist_ok=True)

# Models go to filesDir via install_to_device.sh (NOT committed) — just rename them in the work dir.
shutil.copy(os.path.join(work, "rtA_fix3_fp16.tflite"), os.path.join(work, "rtdetr_graphA_fp16.tflite"))
shutil.copy(os.path.join(work, "rtB_fp16.tflite"), os.path.join(work, "rtdetr_graphB_fp16.tflite"))
print("renamed -> rtdetr_graphA_fp16.tflite, rtdetr_graphB_fp16.tflite  (push with install_to_device.sh)")

# host_params.bin — layout MUST match RtDetr.kt init{}.
z = np.load(os.path.join(work, "host3_w.npz"))
order = ["eo_w", "eo_b", "eo_g", "eo_beta", "bb0w", "bb0b", "bb1w", "bb1b", "bb2w", "bb2b", "valid", "anchors"]
blob = np.concatenate([z[k].astype(np.float32).ravel() for k in order])
blob.astype("<f4").tofile(os.path.join(assets, "host_params.bin"))
print(f"wrote host_params.bin ({blob.nbytes} bytes, {blob.size} floats)")

# 80 contiguous COCO labels from the model config
import build_rtdetr_split as S
net = S.build_net(); id2 = net.config.id2label
with open(os.path.join(assets, "coco_labels.txt"), "w") as f:
    f.write("\n".join(id2[i] for i in range(80)) + "\n")
print("wrote coco_labels.txt (80 classes)")

"""Cycle 0: 3DDFA_V2 reference — detect face, regress 62 params, reconstruct 68 landmarks.
Run in ~/clipconv from tddfa-work/. Shims a pure-python NMS to skip the Cython FaceBoxes build."""
import os, sys, types
import numpy as np

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "3DDFA_V2")
sys.path.insert(0, REPO)
os.chdir(REPO)

# --- pure-python NMS shim so FaceBoxes runs without building cpu_nms.pyx ---
def _py_nms(dets, thresh):
    x1, y1, x2, y2, sc = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = sc.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1); h = np.maximum(0.0, yy2 - yy1 + 1)
        ovr = w * h / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep
_m = types.ModuleType("FaceBoxes.utils.nms.cpu_nms")
_m.cpu_nms = _py_nms
_m.cpu_soft_nms = lambda *a, **k: []
sys.modules["FaceBoxes.utils.nms.cpu_nms"] = _m

import cv2, yaml
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA

cfg = yaml.load(open("configs/mb1_120x120.yml"), Loader=yaml.SafeLoader)
tddfa = TDDFA(gpu_mode=False, **cfg)
fb = FaceBoxes()

IMG = sys.argv[1] if len(sys.argv) > 1 else "examples/inputs/emma.jpg"
img = cv2.imread(IMG)
boxes = fb(img)
print("faces detected:", len(boxes), "-> using first")
box = boxes[0]
param_lst, roi_box_lst = tddfa(img, [box])
param = param_lst[0]                                  # 62 (de-normalized already)
ver = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)[0]   # [3, 68]
print("param[:6]", np.round(param[:6], 3), "... shape", param.shape)
print("landmarks 68 x/y range:", ver[0].min(), ver[0].max(), "/", ver[1].min(), ver[1].max())

# save reference: the exact 120x120 input the regressor saw + 62 params + landmarks
sx, sy, ex, ey = tddfa.parse_roi_box_from_bbox(box) if hasattr(tddfa, "parse_roi_box_from_bbox") else roi_box_lst[0]
roi = roi_box_lst[0]
crop = cv2.resize(img[max(0,int(roi[1])):int(roi[3]), max(0,int(roi[0])):int(roi[2])], (120, 120))
OUT = os.path.join(os.path.dirname(REPO), "ref"); os.makedirs(OUT, exist_ok=True)
# rebuild the exact 120x120 normalized NCHW input the regressor saw (crop->resize->(x-127.5)/128)
from utils.functions import crop_img
img_crop = crop_img(img, roi)
inp_img = cv2.resize(img_crop, dsize=(120, 120), interpolation=cv2.INTER_LINEAR)
inp = ((inp_img.transpose(2, 0, 1).astype(np.float32) - 127.5) / 128.0)[None]   # [1,3,120,120]
np.savez(os.path.join(OUT, "ref_emma.npz"), param=param, ver=ver, roi=np.array(roi),
         box=np.array(box[:4]), inp=inp)
# also export BFM 68-kpt bases + param mean/std for the Kotlin recon
bfm = tddfa.bfm
np.savez(os.path.join(OUT, "recon_assets.npz"),
         u_base=bfm.u_base.reshape(-1), w_shp_base=bfm.w_shp_base, w_exp_base=bfm.w_exp_base,
         param_mean=tddfa.param_mean, param_std=tddfa.param_std)
print("u_base", bfm.u_base.shape, "w_shp_base", bfm.w_shp_base.shape, "w_exp_base", bfm.w_exp_base.shape)
print("saved refs to", OUT)

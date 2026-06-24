"""Prove the SSDlite raw-head -> boxes decode (the logic Kotlin will run) matches
stock torchvision, on the real FP16 tflite. Also export the anchors asset.

Pipeline mirrored exactly (torchvision SSD.postprocess_detections + BoxCoder):
  softmax(91) -> drop background -> per-class score_thresh -> topk -> per-class NMS -> top-K.
Head NCHW channel layout split as (A, K): for spatial (h,w), anchor a, class k ->
  cls channel = a*K + k ; box channel = a*4 + j  (the indexing the app uses).

    ~/clipconv/bin/python scripts/validate_ssdlite_decode.py
"""
import os, urllib.request
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import (
    ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights)
from torchvision.models.detection.image_list import ImageList
from ai_edge_litert.interpreter import Interpreter

LEVELS = [20, 10, 5, 3, 2, 1]
A, K = 6, 91                       # anchors/loc, classes (90 COCO + background)
SIZE = 320
# SSDLite320 normalizes with mean=std=0.5 -> (pixel/255)*2 - 1, range [-1,1]
# (NOT ImageNet). Resize = bilinear stretch to 320x320. Verified vs m.transform to 2e-7.
MEAN = np.array([0.5, 0.5, 0.5], np.float32)
STD = np.array([0.5, 0.5, 0.5], np.float32)
WX, WY, WW, WH = 10.0, 10.0, 5.0, 5.0
BBOX_CLIP = float(np.log(1000.0 / 16.0))
OUT_DIR = "out/ssdlite"
FP16 = f"{OUT_DIR}/ssdlite_320_4dtap_clean_fp16.tflite"


def softmax(x):
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def reshape_heads(raw12):
    """12 raw NCHW tensors [cls0,box0,...] -> (cls[3234,91], box[3234,4]) anchor-ordered.
    Mirrors torchvision view(N,A,K,H,W).permute(0,3,4,1,2).reshape(N,-1,K)."""
    cls_all, box_all = [], []
    for li, H in enumerate(LEVELS):
        W = H
        cls = raw12[2 * li][0].reshape(A, K, H, W).transpose(2, 3, 0, 1).reshape(H * W * A, K)
        box = raw12[2 * li + 1][0].reshape(A, 4, H, W).transpose(2, 3, 0, 1).reshape(H * W * A, 4)
        cls_all.append(cls); box_all.append(box)
    return np.concatenate(cls_all), np.concatenate(box_all)


def decode_boxes(box, anchors):
    """BoxCoder.decode_single: anchors xyxy(320) + deltas -> xyxy(320)."""
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + 0.5 * aw
    acy = anchors[:, 1] + 0.5 * ah
    dx, dy = box[:, 0] / WX, box[:, 1] / WY
    dw = np.minimum(box[:, 2] / WW, BBOX_CLIP)
    dh = np.minimum(box[:, 3] / WH, BBOX_CLIP)
    cx = dx * aw + acx
    cy = dy * ah + acy
    w = np.exp(dw) * aw
    h = np.exp(dh) * ah
    return np.stack([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], 1)


def nms(boxes, scores, iou_thresh):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]; keep.append(i)
        if order.size == 1:
            break
        x1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        ai = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        aj = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        iou = inter / (ai + aj - inter + 1e-9)
        order = order[1:][iou <= iou_thresh]
    return keep


def my_decode(cls_logits, box_reg, anchors, score_thresh, nms_thresh, topk, det_per_img):
    """Mirror SSD.postprocess_detections exactly (per-class threshold/topk/NMS)."""
    scores = softmax(cls_logits)
    boxes = decode_boxes(box_reg, anchors)
    boxes = np.clip(boxes, 0, SIZE)  # clip_boxes_to_image
    B, S, L = [], [], []
    for label in range(1, K):                 # skip background (0)
        sc = scores[:, label]
        m = sc > score_thresh
        if not m.any():
            continue
        sc = sc[m]; bx = boxes[m]
        if sc.size > topk:                    # keep topk
            top = sc.argsort()[::-1][:topk]
            sc = sc[top]; bx = bx[top]
        B.append(bx); S.append(sc); L.append(np.full(sc.shape, label))
    if not B:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, int)
    B, S, L = np.concatenate(B), np.concatenate(S), np.concatenate(L)
    keep = []
    for label in np.unique(L):                # batched_nms == per-class NMS
        idx = np.where(L == label)[0]
        k = nms(B[idx], S[idx], nms_thresh)
        keep.extend(idx[k])
    keep = np.array(keep)
    keep = keep[S[keep].argsort()[::-1][:det_per_img]]
    return B[keep], S[keep], L[keep]


def match_count(boxA, boxB, iou=0.99):
    """How many boxes in A have a >=iou twin in B (order-independent equality)."""
    if len(boxA) == 0 or len(boxB) == 0:
        return 0
    n = 0
    for a in boxA:
        x1 = np.maximum(a[0], boxB[:, 0]); y1 = np.maximum(a[1], boxB[:, 1])
        x2 = np.minimum(a[2], boxB[:, 2]); y2 = np.minimum(a[3], boxB[:, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        aa = (a[2] - a[0]) * (a[3] - a[1])
        ab = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])
        if (inter / (aa + ab - inter + 1e-9)).max() >= iou:
            n += 1
    return n


def main():
    m = ssdlite320_mobilenet_v3_large(
        weights=SSDLite320_MobileNet_V3_Large_Weights.DEFAULT).eval()
    st, nt, tk, dpi = m.score_thresh, m.nms_thresh, m.topk_candidates, m.detections_per_img
    print(f"params: score_thresh={st} nms_thresh={nt} topk={tk} det_per_img={dpi}")

    # anchors for 320x320 (xyxy, 320-space) -- the asset
    feats0 = [torch.zeros(1, 1, s, s) for s in LEVELS]
    il = ImageList(torch.zeros(1, 3, SIZE, SIZE), [(SIZE, SIZE)])
    anchors = m.anchor_generator(il, feats0)[0].numpy().astype(np.float32)
    assert anchors.shape == (3234, 4), anchors.shape

    # real image -> x: bilinear stretch to 320, normalize (pixel/255)*2-1, NCHW.
    # This is exactly what m.transform produces (matched to 2e-7) and what Kotlin runs.
    tmp = f"{OUT_DIR}/dog.jpg"
    if not os.path.exists(tmp):
        urllib.request.urlretrieve(
            "https://github.com/pytorch/hub/raw/master/images/dog.jpg", tmp)
    full = torch.from_numpy(
        np.asarray(Image.open(tmp).convert("RGB"), np.float32).transpose(2, 0, 1) / 255.0)[None]
    x = torch.nn.functional.interpolate(full, size=(SIZE, SIZE), mode="bilinear", align_corners=False)
    x = (x - torch.tensor(MEAN).view(1, 3, 1, 1)) / torch.tensor(STD).view(1, 3, 1, 1)

    # ---- torchvision reference (consume raw head outputs, same anchors) ----
    with torch.no_grad():
        feats = list(m.backbone(x).values())
        head = m.head(feats)                     # cls_logits[1,3234,91], bbox_regression[1,3234,4]
    ref = m.postprocess_detections(head, [torch.from_numpy(anchors)], [(SIZE, SIZE)])[0]
    rb, rs, rl = ref["boxes"].numpy(), ref["scores"].numpy(), ref["labels"].numpy()
    print(f"\ntorchvision ref: {len(rb)} detections, score range "
          f"[{rs.min():.3f},{rs.max():.3f}], labels {sorted(set(rl.tolist()))}")

    # ---- my decode from the SAME torch raw heads (validates indexing/decode) ----
    cls_t = head["cls_logits"][0].numpy()
    box_t = head["bbox_regression"][0].numpy()
    mb, ms, ml = my_decode(cls_t, box_t, anchors, st, nt, tk, dpi)
    print(f"my decode(torch heads): {len(mb)} detections; "
          f"box-match vs ref @IoU0.99: {match_count(mb, rb)}/{len(rb)}")

    # ---- end-to-end: FP16 TFLITE raw outputs -> my decode ----
    itp = Interpreter(model_path=FP16); itp.allocate_tensors()
    ins, outs = itp.get_input_details(), itp.get_output_details()
    itp.set_tensor(ins[0]["index"], x.numpy())
    itp.invoke()
    # map tflite outputs to (cls,box) per level by shape; rebuild ordered list
    by_shape = {tuple(o["shape"]): itp.get_tensor(o["index"]) for o in outs}
    raw12 = []
    for H in LEVELS:
        raw12.append(by_shape[(1, A * K, H, H)])   # cls
        raw12.append(by_shape[(1, A * 4, H, H)])   # box
    cls_f, box_f = reshape_heads(raw12)
    fb, fs, fl = my_decode(cls_f, box_f, anchors, st, nt, tk, dpi)
    print(f"my decode(FP16 tflite):  {len(fb)} detections; "
          f"box-match vs ref @IoU0.99: {match_count(fb, rb)}/{len(rb)}")

    # demo-threshold view (what the app shows by default)
    DEMO = 0.4
    fb2, fs2, fl2 = my_decode(cls_f, box_f, anchors, DEMO, nt, tk, 50)
    print(f"\n@demo thresh {DEMO}: {len(fb2)} boxes -> "
          f"{[(int(l), round(float(s), 2)) for s, l in zip(fs2, fl2)]}")

    # ---- export anchors asset (xyxy, 320-space, float32 little-endian) ----
    asset = f"{OUT_DIR}/ssdlite_anchors.bin"
    anchors.tofile(asset)
    print(f"\nanchors asset: {asset}  ({anchors.size} floats, "
          f"{os.path.getsize(asset)} bytes)  sum={anchors.sum():.4f}")


if __name__ == "__main__":
    main()

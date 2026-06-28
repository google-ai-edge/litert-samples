# LiteRT Feature Matching Sample (XFeat)

Android **local feature matching** sample on the LiteRT Compiled Model API (CPU / GPU). Pick two
images; the app extracts keypoints + descriptors with **[XFeat](https://github.com/verlab/accelerated_features)**
(Accelerated Features, Apache-2.0) and draws mutual-nearest-neighbor correspondences between them —
the basis of SLAM, AR, panorama stitching, and image registration.

## Overview
- Model `litert-community/xfeat-litert` (fetched at build time), FP16, **1.4 MB**, a **pure CNN**.
- **GPU-clean** (full `LITERT_CL` residency, **72/72 nodes, ~0.4 ms** on a Pixel 8a; GPU vs CPU
  corr 0.9999). The conversion re-authors `_unfold2d` → a one-hot space-to-depth conv and moves the
  input InstanceNorm host-side — see [`conversion/`](conversion).
- The net emits dense descriptors + keypoint logits + a reliability heatmap; **keypoint decode,
  descriptor sampling, and mutual-NN matching run in the app** (`XFeatHelper.kt`).

## Features
- Two-image picker; matched keypoints connected with colored lines
- **CPU / GPU** delegate toggle; match count + latency readout

## Build & run
Open `android/` in Android Studio (or `./gradlew :app:installDebug`). The model is fetched from
Hugging Face into `assets/` by `download_model.gradle` on first build.

## License
Sample Apache-2.0. XFeat is Apache-2.0; trained on public correspondence data (MegaDepth + synthetic).

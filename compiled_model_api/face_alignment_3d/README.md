# 3D Face Alignment (3DDFA_V2) — LiteRT CompiledModel, GPU

[3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (Guo et al., ECCV 2020, MIT) fits a **3D morphable
face model** to a photo: a MobileNetV1 regresses **62 3DMM parameters** (pose + 40 shape + 10
expression) on the LiteRT `CompiledModel` **GPU**, and the 68 3D face landmarks (and a dense mesh)
are reconstructed from the BFM bases host-side.

## On-device (Pixel 8a, Tensor G3 — verified)

| stage | in → out | where |
|---|---|---|
| face box | photo → box | android.media.FaceDetector (frontal) |
| 3DMM regressor | crop [1,3,120,120] → 62 params | **GPU** 6.3 MB fp16 |
| BFM reconstruction | 62 params → 68 3D landmarks | host |

fp16 tflite-vs-PyTorch 62-param corr 0.999999; reconstructed landmarks match to 0.02 px.

```
photo →[FaceDetector box → parse_roi → crop 120² (BGR, (x−127.5)/128)]→ [GPU MobileNetV1]→ 62 params
      → denorm → R,offset,α_shp,α_exp → BFM 68 verts → R·v+offset → similar_transform → 68 (x,y)
```

The regressor is a pure CNN — converts through litert-torch with no re-authoring, GPU-clean. Three
host-side details: the model was trained on **cv2 BGR** input; the BFM bases are **interleaved**
`[x0,y0,z0,x1,…]` (`reshape(3,-1, order='F')`); and `android.media.FaceDetector` needs an even width.

## Build & run

```bash
# get the model — from Hugging Face (litert-community/3DDFA-V2-LiteRT) or ./conversion/build_tddfa.py
# place tddfa_mb1_fp16.tflite + tddfa_*.bin in kotlin_cpu_gpu/android/app/src/main/assets/
cd kotlin_cpu_gpu/android
./gradlew :app:installDebug
```

Launch the app — it detects the bundled sample's face and draws the 68 landmarks; tap **Pick face
photo** for your own frontal-face photos.

Model: [`litert-community/3DDFA-V2-LiteRT`](https://huggingface.co/litert-community/3DDFA-V2-LiteRT).
Conversion in [`conversion/`](conversion/). Upstream:
[cleardusk/3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (MIT).

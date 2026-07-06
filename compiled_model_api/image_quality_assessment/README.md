# Image Quality Assessment (NIMA) — LiteRT CompiledModel, GPU

[NIMA (Neural Image Assessment)](https://github.com/idealo/image-quality-assessment) (idealo, Apache-2.0) scores a photo's quality on a **1-10** scale. Two MobileNet models — **aesthetic** (AVA) and **technical** (TID2013) — each predict a 10-bin score distribution; the score is the distribution mean. Both run fully on the LiteRT `CompiledModel` **GPU** (~6.4 MB each).

## On-device (Pixel 8a, Tensor G3 — verified)

| model | in → out | delegate |
|---|---|---|
| NIMA aesthetic | image [1,224,224,3] → dist [10] | **GPU** |
| NIMA technical | image [1,224,224,3] → dist [10] | **GPU** |

~173 ms for both models; tflite-vs-Keras score parity 0.999998 (aesthetic) / 0.999915 (technical).

```
image →[resize 224² · MobileNet /127.5−1]→ [GPU MobileNet]→ softmax dist[10] →[Σ i·pᵢ]→ score 1-10
```

NIMA is `MobileNet(224², pooling='avg')` → Dense(10, softmax) — a pure CNN, so it converts straight through `tf.lite` (fp16) with no re-authoring, and every op rides the GPU delegate. The 10-bin distribution is the graph output; the 1-10 mean is computed host-side.

## Build & run

```bash
# get the two tflites — from Hugging Face (litert-community/NIMA-LiteRT) or ./conversion/build_nima.py
# place nima_aesthetic_fp16.tflite + nima_technical_fp16.tflite in kotlin_cpu_gpu/android/app/src/main/assets/
cd kotlin_cpu_gpu/android
./gradlew :app:installDebug
```

Launch the app — it scores the bundled sample on start; tap **Pick image** for your own photos.

Model: [`litert-community/NIMA-LiteRT`](https://huggingface.co/litert-community/NIMA-LiteRT). Conversion in [`conversion/`](conversion/). Upstream: [idealo/image-quality-assessment](https://github.com/idealo/image-quality-assessment) (Apache-2.0).

# PP-HumanSeg — Human / portrait segmentation (LiteRT CompiledModel GPU)

Real-time **human (portrait) segmentation** running **fully on the LiteRT `CompiledModel`
GPU** delegate. [PP-HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/contrib/PP-HumanSeg)
(PaddleSeg) segments people from the background — the building block for video-call
background blur/replacement and portrait effects. Tiny (6 MB), ~36 ms/frame on a Pixel 8a.

- **Model:** [litert-community/PP-HumanSeg-LiteRT](https://huggingface.co/litert-community/PP-HumanSeg-LiteRT)
- **Weights:** [OpenCV Zoo / PaddleSeg](https://huggingface.co/opencv/human_segmentation_pphumanseg) · Apache-2.0
- **Input:** `[1, 192, 192, 3]` **NHWC**, **BGR**, `(x/255 - 0.5)/0.5`
- **Output:** `[1, 192, 192, 2]` NHWC softmax; argmax over the last dim → person mask (1 = person)

## How it works

PP-HumanSeg is a pure CNN, so the graph converts fully GPU-compatible (**128/128 nodes on
the delegate, 1 partition**; device corr 1.0 vs ONNX, ~36 ms) with **zero patches** —
converted with onnx2tf from the OpenCV-Zoo ONNX. Decode: `argmax` over the 2-class output
→ person mask; resize and composite (background blur/replace).

## Run

```bash
cd android
./gradlew :app:installDebug
```

The 6 MB `pphumanseg.tflite` is bundled in `app/src/main/assets/`. The sample replaces
the background of a bundled portrait with a studio color. Adapt `MainActivity.kt` to feed
live camera frames for a real-time background-replacement demo.

## Convert

See [`conversion/`](conversion/) — `build_pphumanseg.py` downloads the Apache-2.0 ONNX and
converts with onnx2tf.

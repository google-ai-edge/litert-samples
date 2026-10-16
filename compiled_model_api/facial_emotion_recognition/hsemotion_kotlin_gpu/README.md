# Facial Emotion Recognition with LiteRT — HSEmotion (EfficientNet-B0)

An Android sample that recognizes the **8 AffectNet emotions** (anger, contempt, disgust, fear, happiness, neutral, sadness, surprise) from a face, fully on the LiteRT `CompiledModel` GPU delegate. [HSEmotion](https://github.com/av-savchenko/face-emotion-recognition) (EmotiEffLib, Apache-2.0) is an EfficientNet-B0 fine-tuned on AffectNet.

## Model

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| HSEmotion EfficientNet-B0 | `image[1,3,224,224]` → `logits[1,8]` | GPU (342/342 nodes, 1 partition) |

fp16 weights, 8 MB, ~2 ms/inference. Converted with [litert-torch](https://github.com/google-ai-edge/litert) — see [`conversion/`](conversion/).

## Pipeline

```
photo --android.media.FaceDetector--> face crop
      --host: resize-224 + ImageNet-norm (NCHW)--> image[1,3,224,224]
                                    HSEmotion (GPU) -> logits[8] -> softmax -> emotions
```

The model expects a **tightly cropped face**, so `EmotionClassifier.kt` first locates a face with the built-in `android.media.FaceDetector` and crops to it (falling back to the whole image), then normalizes and runs the CompiledModel.

## Two conversion hurdles

1. **Old-timm pickle.** The released weights are a pickled model built with an old timm whose forward is broken under current timm (missing `conv_s2d`). The state dict is lifted into a fresh timm `tf_efficientnet_b0` (num_classes=8, `classifier.0.*` → `classifier.*`), which has a working forward — 358/360 tensors match by name+shape.
2. ⭐ **fp16 SqueezeExcite mean → NaN.** The SE block's global mean `x.mean((2,3))` over the 112×112 stem map is a single fp16 reduction whose partial sum overflows 65504 → the GPU delegate emits an **all-NaN** output (it reduces in fp16 even for an fp32 graph; desktop fp16 CPU is exact). Replaced by a **hierarchical mean** — repeated `avg_pool2d` over equal-size tiling windows (≤ 49 elements each) — mathematically identical, fp16-safe.

## Verification

- fp16 tflite vs fp32 PyTorch through the CompiledModel API: top-1 identical, logits corr 1.00000 (`conversion/validate_hsemotion.py`).
- On device (Pixel 8a): **342/342 nodes on the GPU delegate, 1 partition**, ~2 ms; device fp16 top-1 matches desktop fp32 (logits corr 0.99997). Bundled smiling-face sample → Happiness.

## Build & run

```bash
cd android
./gradlew :app:installDebug
```

The 8 MB `hsemotion_b0_fp16.tflite` is downloaded from Hugging Face ([`litert-community/HSEmotion-B0-LiteRT`](https://huggingface.co/litert-community/HSEmotion-B0-LiteRT)) into the app assets at build time (`app/download_model.gradle`); a sample face is bundled. Pick a face photo (or use the bundled sample) to see its emotion distribution.

## License

Model: Apache-2.0 (HSEmotion / EmotiEffLib).

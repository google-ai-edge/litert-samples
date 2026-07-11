# NIMA → LiteRT conversion

Converts [NIMA (idealo MobileNet)](https://github.com/idealo/image-quality-assessment) (Apache-2.0) into two fp16 tflite graphs. NIMA is `MobileNet(224², include_top=False, pooling='avg')` → Dense(10, softmax) — a pure CNN, so it converts straight through `tf.lite` with no re-authoring.

- `build_nima.py` — builds the aesthetic + technical models, loads the idealo Keras weights, exports fp16 tflite, and checks tflite-vs-Keras parity on a reference image.
- `ref_nima.py` — the reference scorer (build + load weights + score a photo → 1-10).

```bash
pip install tensorflow pillow           # TF 2.x; the Keras-2 .hdf5 weights load in Keras 3
git clone https://github.com/idealo/image-quality-assessment   # bundles the MobileNet weights
python build_nima.py                    # -> nima_{aesthetic,technical}_fp16.tflite
```

Preprocessing is MobileNet's `preprocess_input` (resize 224², RGB, `x/127.5 - 1`). Each model outputs a 10-bin score distribution; the quality score is its mean over 1..10 (done host-side in the app).

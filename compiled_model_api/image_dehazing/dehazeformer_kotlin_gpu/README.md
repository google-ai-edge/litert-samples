# DehazeFormer-MCT — Image dehazing (LiteRT CompiledModel GPU)

**Image dehazing** with the network fully on the LiteRT `CompiledModel` GPU delegate.
[DehazeFormer](https://github.com/IDKiro/DehazeFormer) (TIP 2023, MCT curve-mapping variant
trained on a mixed dataset for real-world haze) removes fog / haze / smoke and restores
contrast and color. ~255 ms/frame on a Pixel 8a.

- **Model:** [litert-community/DehazeFormer-MCT-LiteRT](https://huggingface.co/litert-community/DehazeFormer-MCT-LiteRT)
- **Weights:** author-hosted [IDKiro/DehazeFormer_Demo](https://huggingface.co/spaces/IDKiro/DehazeFormer_Demo) · MIT
- **Input:** `[1, 3, 256, 256]` NCHW, RGB in `[-1, 1]`
- **Output:** `[1, 72, 256, 256]` curve parameters (3 out × 3 in × 8 levels)

## How it works

The MCT design is mobile-ideal: the Swin-windowed-attention basenet always runs at 256×256;
the predicted per-pixel curves are applied to the **full-resolution** image host-side
(`Dehazer.applyCurves` — the exact official grid_sample mapping), so output resolution is
independent of the network. Fully GPU-resident: **2042/2042 nodes, 1 partition**, device
corr 0.999998, end-to-end vs the official pipeline corr 0.999997.

## Run

```bash
cd android
./install_to_device.sh <dir-with-dehazeformer_base.tflite>
./gradlew :app:installDebug
```

The sample dehazes a bundled hazy photo and shows the input and result. Adapt
`MainActivity.kt` to feed camera frames for a live dehazing viewfinder.

## Convert

See [`conversion/`](conversion/) — `build_dehaze.py` fetches the MIT weights from the
author's HF Space and converts with litert-torch; `validate_dehaze.py` checks GPU-op
cleanliness and CompiledModel parity.

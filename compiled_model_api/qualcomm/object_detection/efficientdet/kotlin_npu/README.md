# EfficientDet-Lite Android Sample

CameraX + LiteRT object detection using an EfficientDet-Lite TFLite model.

The app uses the backend ladder:

`NPU -> GPU -> CPU`

It logs the selected backend, tensor checks, and inference timings with tags like
`EfficientDetDetector`, `LiteRTDebugger`, and `EfficientDetPerf`.


## Build 

**Option 1: Android Studio (Recommended)**
1. Open the project in Android Studio
2. Wait for **Gradle sync** to finish
3. Connect your device (or use emulator)
4. Select your device in the top bar
5. Click **Run** (or press Shift + F10)

**Option 2: Command Line (Windows):**

```bat
set JAVA_HOME="C:\Program Files\Android\Android Studio\jbr"
./gradlew.bat :app:assembleDebug
```

**Option 3: Command Line (Linux/ macOS)**
```bash
export JAVA_HOME="/opt/android-studio/jbr"
./gradlew :app:assembleDebug
```

## Quantization

If you have an EfficientDet SavedModel and need to produce a TFLite model:

```bash
python scripts/quantize_efficientdet_tflite.py \
  --saved-model-dir build/efficientdet_saved_model \
  --full-int8 \
  --calib-dir /path/to/jpeg_calibration_images
```

To auto-download the Kaggle/TFHub EfficientDet-Lite SavedModel source:

```bash
python scripts/quantize_efficientdet_tflite.py --full-int8
```

To copy the converted model into app assets:

```bash
python scripts/quantize_efficientdet_tflite.py --full-int8 --copy-to-assets
```

Note: the Kaggle SavedModel path may require `--allow-select-tf-ops`, which
produces a Flex/Select-TF model. That is useful as a conversion fallback, but it
is not ideal for Qualcomm NPU benchmarking. Prefer the downloaded EfficientDet
Lite TFLite model for NPU testing.

## Download Model 

The app expects (already included with clone):

`app/src/main/assets/efficientdet_lite0_detection.tflite`

Download or refresh it with:

```bash
python scripts/download_efficientdet_lite.py --variant lite0
```

The included model is already quantized for byte input (`uint8`) and has mostly
`int8` tensors, so it is the preferred first model to test on Qualcomm NPU.

## Debugging

If the model is missing from the assets after all the steps, run the following:

```bash
curl -L "https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/default/1?lite-format=tflite" -o app/src/main/assets/efficientdet_lite0_detection.tflite
```

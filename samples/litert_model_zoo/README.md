# LiteRT Model Zoo — vision and audio models in one Android app

One Android app that runs 29 on-device tasks (21 vision, 8 audio) through the
[LiteRT](https://github.com/google-ai-edge/litert)
[CompiledModel API](https://ai.google.dev/edge/litert/android): pick a task, download its model,
choose a photo, a live camera frame, a recording, a WAV file or typed text, and read the result
together with the measured inference time and the backend it ran on.

What the app shows:

- **One runtime path for every model** — each task compiles its graph with `CompiledModel`
  (LiteRT 2.2.0), GPU by default; a few models keep part of their pipeline on CPU by design (the
  Backend column). If the GPU compile fails, the task recompiles on CPU and the result card says why.
- **Nothing bundled** — the 29 model sets (2.2 GB in total) download on demand from Hugging Face and
  resume after an interruption; every file is checked against its byte size and SHA-256 before use.
  The catalog pins each file to a repository revision, so a download always gets the verified file.
- **Every result carries its numbers** — inference time and backend on each result card; the About
  screen lists each model's license, upstream project and model card.

Tested on a Galaxy S26: all 29 tasks ran in the release build this sample was cut from; the sample
tree itself is built and unit-tested on the host. Other phones and GPUs are not verified yet. The app
builds for arm64-v8a on Android 8.0+ (minSdk 26). It connects only to Hugging Face (huggingface.co
and the download hosts it redirects to); camera, microphone and file inputs stay on the device.

## Tasks

The Galaxy S26 column is the median of 10 timed runs after warm-up, measured with an instrumented
build of the same model wrappers (512×512 image, 640×640 for OCR, 1–11 s audio clips, "Hello world."
for TTS), from engine entry to the last model-output readback. Source separation compiles its three graphs inside each call, so its time includes
compilation.

| Task | Model | Backend | Galaxy S26 | Download | License | Model files |
| :-- | :-- | :-- | --: | --: | :-- | :-- |
| Object Detection | RF-DETR Nano | GPU | 54 ms | 56 MB | [Apache-2.0](https://github.com/roboflow/rf-detr/blob/develop/LICENSE) | [litert-community/RF-DETR-Nano-LiteRT](https://huggingface.co/litert-community/RF-DETR-Nano-LiteRT) |
| Video Action Recognition | MoViNet-A0 stream | CPU | 3.6 ms | 15 MB | [Apache-2.0](https://github.com/Atze00/MoViNet-pytorch/blob/main/LICENSE) | [litert-community/MoViNet-A0-Stream-LiteRT](https://huggingface.co/litert-community/MoViNet-A0-Stream-LiteRT) |
| Semantic Segmentation | PIDNet-S | GPU | 31 ms | 31 MB | [MIT](https://github.com/XuJiacong/PIDNet/blob/main/LICENSE) | [litert-community/PIDNet-S-Cityscapes-LiteRT](https://huggingface.co/litert-community/PIDNet-S-Cityscapes-LiteRT) |
| Lane Detection | TwinLiteNet | GPU | 16 ms | 3.1 MB | [MIT](https://github.com/chequanghuy/TwinLiteNet/blob/main/LICENSE) | [litert-community/TwinLiteNet-LiteRT](https://huggingface.co/litert-community/TwinLiteNet-LiteRT) |
| Super-Resolution | EDSR-base x4 | GPU | 31 ms | 7.7 MB | [Apache-2.0](https://github.com/eugenesiow/super-image/blob/main/LICENSE) | [litert-community/EDSR-x4-LiteRT](https://huggingface.co/litert-community/EDSR-x4-LiteRT) |
| Image Dehazing | DehazeFormer-MCT | GPU | 136 ms | 17 MB | [MIT](https://github.com/IDKiro/DehazeFormer/blob/main/LICENSE) | [litert-community/DehazeFormer-MCT-LiteRT](https://huggingface.co/litert-community/DehazeFormer-MCT-LiteRT) |
| Face Liveness / Anti-Spoofing | Silent-Face MiniFASNetV2 | GPU | 1.9 ms | 1.9 MB | [Apache-2.0](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing/blob/master/LICENSE) | [litert-community/Silent-Face-Anti-Spoofing-LiteRT](https://huggingface.co/litert-community/Silent-Face-Anti-Spoofing-LiteRT) |
| Head Pose Estimation | 6DRepNet | GPU | 9.9 ms | 157 MB | [MIT](https://github.com/thohemp/6DRepNet/blob/master/LICENSE) | [litert-community/6DRepNet-HeadPose-LiteRT](https://huggingface.co/litert-community/6DRepNet-HeadPose-LiteRT) |
| Crowd Counting | DM-Count | GPU | 30 ms | 86 MB | [MIT](https://github.com/cvlab-stonybrook/DM-Count/blob/master/LICENSE) | [litert-community/DM-Count-Crowd-LiteRT](https://huggingface.co/litert-community/DM-Count-Crowd-LiteRT) |
| Instance Segmentation | RF-DETR-Seg Nano | GPU | 69 ms | 63 MB | [Apache-2.0](https://github.com/roboflow/rf-detr/blob/develop/LICENSE) | [litert-community/RF-DETR-Seg-Nano-LiteRT](https://huggingface.co/litert-community/RF-DETR-Seg-Nano-LiteRT) |
| Background Removal | ormbg | GPU | 90 ms | 176 MB | [Apache-2.0](https://huggingface.co/schirrmacher/ormbg) | [litert-community/ormbg-LiteRT](https://huggingface.co/litert-community/ormbg-LiteRT) |
| Portrait Matting | MODNet | GPU | 21 ms | 26 MB | [Apache-2.0](https://github.com/ZHKKKe/MODNet/blob/master/LICENSE) | [litert-community/MODNet-LiteRT](https://huggingface.co/litert-community/MODNet-LiteRT) |
| Dense Feature Visualization | DINOv2 ViT-S/14 | GPU | 61 ms | 45 MB | [Apache-2.0](https://github.com/facebookresearch/dinov2/blob/main/LICENSE) | [litert-community/DINOv2-ViT-S14-LiteRT](https://huggingface.co/litert-community/DINOv2-ViT-S14-LiteRT) |
| Image Matching | XFeat | GPU | 47 ms | 1.4 MB | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | [litert-community/xfeat-litert](https://huggingface.co/litert-community/xfeat-litert) |
| Image tagging | RAM++ | GPU + CPU | 533 ms | 807 MB | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | [litert-community/RAM-Plus-LiteRT](https://huggingface.co/litert-community/RAM-Plus-LiteRT) |
| Image quality | NIMA aesthetic | GPU | 1.7 ms | 6.4 MB | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | [litert-community/NIMA-LiteRT](https://huggingface.co/litert-community/NIMA-LiteRT) |
| Image Classification | Vision-RWKV-S | GPU | 70 ms | 48 MB | [Apache-2.0](https://github.com/OpenGVLab/Vision-RWKV/blob/master/LICENSE) | [litert-community/Vision-RWKV-S-LiteRT](https://huggingface.co/litert-community/Vision-RWKV-S-LiteRT) |
| Fine-Grained Classification | PlantNet-300K ResNet18 | GPU | 4.2 ms | 47 MB | [Apache-2.0](https://huggingface.co/cpoisson/plantnet300k-resnet18) | [litert-community/PlantNet-300K-ResNet18-LiteRT](https://huggingface.co/litert-community/PlantNet-300K-ResNet18-LiteRT) |
| OCR | PP-OCRv5 | GPU | 77 ms | 27 MB | [Apache-2.0](https://github.com/PaddlePaddle/PaddleOCR/blob/main/LICENSE) | [litert-community/PP-OCRv5-LiteRT](https://huggingface.co/litert-community/PP-OCRv5-LiteRT) |
| Super Resolution | Real-ESRGAN x4v3 | GPU | 912 ms | 3.5 MB | [BSD-3-Clause](https://github.com/xinntao/Real-ESRGAN/blob/master/LICENSE) | [litert-community/real-esrgan-x4v3-litert](https://huggingface.co/litert-community/real-esrgan-x4v3-litert) |
| Monocular Geometry Estimation | Depth Anything 3 Small | GPU | 243 ms | 55 MB | [Apache-2.0](https://github.com/ByteDance-Seed/Depth-Anything-3) | [litert-community/Depth-Anything-3-Small-LiteRT](https://huggingface.co/litert-community/Depth-Anything-3-Small-LiteRT) |
| Speech Recognition | Zipformer-small (CR-CTC) | GPU | 80 ms | 46 MB | [Apache-2.0](https://github.com/k2-fsa/icefall/blob/master/LICENSE) | [litert-community/Zipformer-medium-CR-CTC-LiteRT](https://huggingface.co/litert-community/Zipformer-medium-CR-CTC-LiteRT) |
| Text-to-Speech | Matcha-TTS | GPU + CPU | 580 ms | 94 MB | [MIT](https://github.com/shivammehta25/Matcha-TTS/blob/main/LICENSE) | [litert-community/Matcha-TTS](https://huggingface.co/litert-community/Matcha-TTS) |
| Audio Codec | DAC 16kHz | GPU | 91 ms | 149 MB | [MIT](https://github.com/descriptinc/descript-audio-codec/blob/main/LICENSE) | [litert-community/DAC-16kHz-LiteRT](https://huggingface.co/litert-community/DAC-16kHz-LiteRT) |
| Audio Classification | PANNs CNN14 | GPU | 116 ms | 162 MB | [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/) | [litert-community/PANNs-CNN14-AudioSet-LiteRT](https://huggingface.co/litert-community/PANNs-CNN14-AudioSet-LiteRT) |
| Pitch Detection | CREPE full | GPU | 90 ms | 45 MB | [MIT](https://github.com/marl/crepe/blob/master/LICENSE) | [litert-community/CREPE-pitch-LiteRT](https://huggingface.co/litert-community/CREPE-pitch-LiteRT) |
| Audio Source Separation | TIGER-DnR | GPU | 45.0 s | 48 MB | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | [litert-community/TIGER-DnR-LiteRT](https://huggingface.co/litert-community/TIGER-DnR-LiteRT) |
| Speech Enhancement | CMGAN | GPU | 384 ms | 4.2 MB | [MIT](https://github.com/ruizhecao96/CMGAN/blob/main/LICENSE) | [litert-community/CMGAN-LiteRT](https://huggingface.co/litert-community/CMGAN-LiteRT) |
| Music Transcription | Basic Pitch | GPU | 504 ms | 0.8 MB | [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | [litert-community/Basic-Pitch-LiteRT](https://huggingface.co/litert-community/Basic-Pitch-LiteRT) |

## Build & run

```bash
cd samples/litert_model_zoo/android
./gradlew :app:assembleDebug
```

Needs JDK 17 and Android SDK platform 36. Install
`app/build/outputs/apk/debug/app-debug.apk` with adb, or open `android/` in Android Studio and run
the `app` configuration. The first screen lists the tasks; a model downloads when you tap Download
and confirm its size. Downloaded files live in the app's private storage and can be deleted from the
Models tab.

## App structure

MVVM + Jetpack Compose (Material 3). Model calls run on single-thread executors, one in the view
model for photo and audio tasks and one inside the camera pipeline for live frames, because the
wrappers reuse native buffers; the UI collects a single `UiState`.

| File | Role |
| :-- | :-- |
| `MainViewModel.kt` | Downloads, task selection, camera and microphone sessions; runs the engines and owns `UiState`. |
| `data/ModelCatalog.kt` + `assets/models.json` | The 29 catalog rows: files (URL, bytes, SHA-256), license, upstream project, model card, backend. |
| `data/ModelStore.kt` | Resumable HTTPS download into private storage; byte-size and SHA-256 verification. |
| `common/CompiledModelRunner.kt` | Lifecycle wrapper over `CompiledModel` with pre-allocated tensor buffers (the same helper as `utilities/common`). |
| `image/`, `audio/`, `vision/` | Per-task engines: input conversion, GPU-to-CPU fallback, result rendering. |
| `models/<name>/` | One package per model: preprocessing, the inference call and the decoding math, unit-tested on the JVM. |
| `view/ModelZooScreen.kt` | Compose screens: Explore, task, Models, About. |

## Tests

```bash
./gradlew :app:testDebugUnitTest
```

118 JVM tests: catalog validation, download bookkeeping (partial files, resume, SHA-256
verification), per-model math against fixtures, GPU-to-CPU fallback and the permission flow.

## License

Code: Apache-2.0. Models: as listed above; the PANNs weights are CC-BY-4.0 and the About screen
carries the attribution. The `assets/` folder holds two generated DSP tables (an 80×257 mel filterbank
and a 400-sample Povey window) and the COCO label list for RF-DETR-Seg; no model weights.

# Image cut-out: WeVisOne-VL-2B on LiteRT (Compiled Model API)

Pick a photo and type what to cut out: the app finds that one thing and saves it with the background removed, as a transparent PNG, all on the phone. The same screen also detects, points at and counts what you name, and reads text.

<img src="https://huggingface.co/litert-community/WeVisOne-VL-2B/resolve/9f2f07d2678e6bb5d9212c5c20668059730a6a41/assets/demo.gif" width="300" alt="The app on a Galaxy S26: the kitten cut out of a photo of a kitten and a puppy"> <br> *On a Galaxy S26: a photo of a kitten and a puppy (by blende12 on Pixabay, CC0, from Wikimedia Commons), the words "the kitten", and the cut-out after 4.9 s (one run; the table below gives the median of several runs, 5.0 s).*

[WeVisOne-VL-2B](https://huggingface.co/tencent/WeVisOne-VL-2B) (Tencent, Apache-2.0) is a 2B vision-language model that answers with boxes, points and text, and marks each object it should cut out with an `<OBJ>` token that a small alpha-matte decoder turns into a mask. The app runs the whole path as LiteRT graphs with the Compiled Model API on the GPU: the vision encoder, the language model (greedy decoding with its KV cache) and the matte decoder. The matte decoder needs the language model's hidden state at the `<OBJ>` token, which is not part of what the LiteRT-LM API returns, so the app drives the three graphs itself through the Compiled Model API.

## Run it (Android)

1. Download six files of the [`cutout/`](https://huggingface.co/litert-community/WeVisOne-VL-2B/tree/main/cutout) folder of [litert-community/WeVisOne-VL-2B](https://huggingface.co/litert-community/WeVisOne-VL-2B) (3,257,599,697 bytes; the seventh file, the fp16 language model, is only for `FORM=fp16` or `both`, below). The `.litertlm` file at the top of that repository is the LiteRT-LM build of the model (boxes, points, counts and text, no cut-outs); this app does not use it. `pip install -U huggingface_hub` gives the `hf` command.

   ```bash
   export HF_HUB_DISABLE_XET=1
   hf download litert-community/WeVisOne-VL-2B \
     cutout/tokenizer.json cutout/wevisone_embed_tokens_fp16.bin \
     cutout/wevisone_vision_encoder_512_fp16.tflite cutout/wevisone_vision_adapter_512_fp16.tflite \
     cutout/wevisone_mask_decoder_gs16.tflite cutout/wevisone_decoder_b_int8.tflite \
     --revision 9f2f07d2678e6bb5d9212c5c20668059730a6a41 \
     --local-dir "$HOME/Downloads/WeVisOne-VL-2B-LiteRT"
   ```

2. With a phone connected over adb (USB debugging on; Gradle needs the Android SDK 35 and a JDK 17, or open `kotlin_gpu/android` in Android Studio), build and install the debug build, then copy the files into the app's private storage:

   ```bash
   cd kotlin_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh
   ```

   `install_to_device.sh` reads `~/Downloads/WeVisOne-VL-2B-LiteRT/cutout` unless you give it another directory. It checks every file's size before it copies anything and its SHA-256 on the phone after the copy, and it downloads three NASA example photos (`EXAMPLES=0` skips them). It copies with `run-as`, which needs the debug build. `FORM=fp16` installs the fp16 language model instead of the int8 one and `FORM=both` installs both; add `cutout/wevisone_decoder_b_fp16.tflite` to the download for them.

3. Open **WeVisOne Cut-out**. Choose a photo (**Photos**, **Camera**, or **Example**, which also fills in a task and words), pick a task, type the words, tap **Run**. Boxes, points and labels appear on the photo and a cut-out is tinted; tap the photo to see the cut-out alone on a checkerboard. **Save PNG** writes it with its transparency to `Pictures/WeVisOne`.

| Task | Words | The instruction the model gets (the authors' `build_prompt`) |
|---|---|---|
| Cut out | what to cut out: `the man on the left` | `Locate the instances that match the following description and provide the alpha mattes: the man on the left.` |
| Detect | labels, one per line or comma-separated: `person, hat` | `Locate all the instances that match the following description: person. hat.` |
| Find one | a description: `the yellow rectangle` | `Locate the instances that match the following description: the yellow rectangle.` |
| Point | labels: `airplane, windsock` | `point to: airplane. windsock` |
| Count (boxes / points) | what to count: `blue circles` | `How many blue circles in the image? Please count them with boxes.` |
| Read text | nothing | `detect all the word in the box format` |
| Find text | the text: `EXIT 12` | `Locate the following text in the image: EXIT 12` |

The app stops with a message, and loads nothing, when no photo is chosen, when a model file is missing or its SHA-256 differs, and when the phone does not have the free memory a step needs. When the answer names nothing, it says "No match". **Stop** ends a run between steps and between answer tokens.

## Speed and memory

| Phone | Photo + words → cut-out | Steps | Memory |
|---|---|---|---|
| Galaxy S26 (12 GB), GPU | **5.0 s** (3543 × 2362 photo, "the kitten"), 4.6 s (640 × 480 synthetic picture, "the green circle") | for the photo: vision encoder 0.64 s, adapter 0.04 s, prefill (295 tokens) 1.34 s, 12 answer tokens 1.08 s (90 ms a token), matte decoder 1.45 s, matte to photo size 0.09 s | free memory 7.2 GB before the first graph, 1.2 GB with all three graphs loaded; app peak 6.1 GB |
| Galaxy S26, first run after install | 35.0 s | building the graphs: vision 8.2 s, language model 19.9 s, matte decoder 1.7 s (the files' SHA-256, checked once per install, not included) | |
| Pixel 8a (8 GB) | does not run | the app stops before loading anything: 3.0 GB available, 4.1 GB needed | |

Galaxy S26 SM-S942Q (Android 16), LiteRT 2.3.0, 2026-10-26: the median of four runs after the first, every graph on the GPU (OpenCL), the language model with int8 weights and float32 arithmetic (`GPU (int8 weights, fp32)`, the app's default). Memory is in GB of 2^30 bytes, the unit of the app's messages. The menu's other language model forms, GPU (fp16, fp32 accumulation), GPU (fp32) and CPU (fp32), are there for comparison: on a 12 GB phone they are slower or do not fit.

## How it runs

| Step | Where | Notes |
|---|---|---|
| Photo → 512 × 512 and 1024 × 1024 | Kotlin | Pillow 12.2.0's bicubic resampler ported byte for byte (`PilBicubic.kt`): the vision encoder answers differently with other resamplers |
| Vision encoder (Qwen3-VL vision tower, 1024 patches) + adapter (the merger and the three DeepStack mergers side by side) | GPU, fp16 weights, fp32 | one graph each |
| Prompt | Kotlin | the fixed system turn `You are a helpful assistant.`, the image first, the instruction; byte-level BPE from the checkpoint's `tokenizer.json` (`QwenBpeTokenizer.kt`) |
| Token embeddings | Kotlin | rows of the checkpoint's table as float16 (`wevisone_embed_tokens_fp16.bin`, memory-mapped) |
| Language model (28 layers, 7 signatures: prefill 1024 / 256 / 64 / 16 / 4 / 1 and decode, KV cache 2048) | GPU, int8 weights (dynamic range, per channel), fp32 | the DeepStack features are added inside the graph; the M-RoPE positions are computed inside the graph from an image flag |
| `<OBJ>` hidden state | Kotlin | the decode step whose input token is `<OBJ>` returns it |
| Box encoding | Kotlin | Fourier features of x1, y1, 0.2 log w + 0.5, 0.2 log h + 0.5 up to 2^31 cycles: the float32 phase of torch, and the log values torch gives (a 28 KB table), since one float32 ulp of the box value moves the matte |
| Matte decoder (Deformable-DETR-style decoder over the vision features at 1024 × 1024) | GPU, fp16 | GroupNorm written as two-level means, its one-query attention as a plain projection (both exact) |
| Matte → photo size | Kotlin | sigmoid, bilinear (half-pixel) like `torch.nn.functional.interpolate` |

Graphs stay loaded while there is memory for them: before each step the app reads `MemAvailable` and closes the graphs the step does not use until the step fits, or stops with a message when it would not fit with all of them closed.

## Limits

- One photo and one instruction per run: no follow-up turns, the system turn is fixed, and a conversation of text alone is not what this app is for.
- The model sees the photo at 512 × 512, not at the authors' default resolution, so its answers can differ from the authors' package.
- Memory: the app needs about 4.1 GB of free memory before it starts. It runs on the 12 GB Galaxy S26; the 8 GB Pixel 8a stops with the message.
- The answers follow the authors' fp32 model on the requests checked below, not everywhere: lower-precision weights move a coordinate by one bin where the model's own top two choices are nearly tied.

## How the answers were checked

Against the authors' model in float32 (PyTorch, the same 512 × 512 view of the photo, greedy decoding), on a Galaxy S26 with this app, 17 requests:

- The prompt: the app's token ids equal the reference processor's on 17 of 17 requests. The JVM tests check the same on the 18 committed rows: `./gradlew :app:testDebugUnitTest -Pwevisone.models="$HOME/Downloads/WeVisOne-VL-2B-LiteRT/cutout"` (without the property, only the tests that need no model file run).
- The pixels: the photos the app decodes equal the reference pixels on 17 of 17 requests (one of them a JPEG).
- Cut-outs (six requests, among them the synthetic `the green circle` and a NASA photo's `the man on the left`): alpha MAE at most 0.0032 and IoU(alpha > 0.5) at least 0.984 against the reference matte. The box tokens are the reference's on two of the six: int8 weights move a box edge by one bin where the reference's top two coordinates are nearly tied.
- Boxes, points, counts and text (ten requests): 18 of 18 reference boxes matched at IoU ≥ 0.75, 9 of 9 points within 2 % of the side, 3 of 3 counts, character error rate 0 on the text.

## License

The sample's code is Apache-2.0. What it ports from WeVisOne-VL, Pillow and Hugging Face tokenizers, and the example images, are described in [`NOTICE`](kotlin_gpu/android/NOTICE); the license texts are in [`licenses/`](kotlin_gpu/android/licenses/). The model, WeVisOne-VL-2B, is Apache-2.0.

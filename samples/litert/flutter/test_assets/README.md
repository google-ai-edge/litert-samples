# Test fixtures

Inputs for the unit tests (`test/`), the integration tests (`integration_test/`) and the app's self-test.

## Photos

Two photos are from the [COCO 2017](https://cocodataset.org) validation set. The COCO Consortium does not own the
images: each one is a Flickr photo under the Creative Commons licence recorded in the COCO annotations, and it stays
under that licence here.

| File | COCO id | Title | Author | Flickr photo | Licence |
|---|---|---|---|---|---|
| `cats.jpg`, `yolo26n/cats_640x480_rgba.u8` | 39769 | "Cats and remote controllers" | DocChewbacca | [210383891](https://www.flickr.com/photos/st3f4n/210383891/) | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) |
| `showcase/coco_2592_pirate_mug.jpg` | 2592 | "Pirate Tea Break!" | Dave Crosby | [2864950455](https://www.flickr.com/photos/wikidave/2864950455/) | [CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/) |

## Drawn or generated

- `gate42.png`: a synthetic sign, drawn by `tool/make_gate42_fixture.py`.
- `showcase/sign_do_not_feed.jpg`, `showcase/sign_wifi.jpg`: drawn by `tool/make_showcase_signs.py`.
- `images/`: synthetic JPEGs for the image normalizer tests (a grey gradient; a red/blue block with an EXIF
  orientation).
- `litertlm/*.header.bin`: only the header (the first 1-2 KB, no weights) of two `.litertlm` files, a Gemma 4 E2B GPU
  build and a Qualcomm NPU build, for the header-parsing tests.

## Audio

Spoken questions, 16 kHz mono PCM16: `france_16k.pcm` (raw samples, "What is the capital of France?") and the `q_*.wav`
files here and in `showcase/` (44-byte WAV header). They are spoken by the app's own speech synthesizer,
Inflect-nano-v2 (Apache-2.0, built into the app), and made with `tool/make_question_audio.sh`: it synthesizes each
question at 24 kHz, resamples it to 16 kHz and adds 150 ms of silence at each end. The wording of every clip is the
list at the top of `integration_test/tools/make_question_audio_test.dart`; change it there and run the script again.

| File | Words | Duration |
|---|---|---|
| `france_16k.pcm` | "What is the capital of France?" | 2.08 s |
| `q_cats.wav`, `showcase/q_cats.wav` | "How many cats do you see?" | 2.15 s |
| `q_describe.wav`, `showcase/q_describe.wav` | "Describe the scene." | 1.57 s |
| `q_sign.wav`, `showcase/q_sign.wav` | "What does the sign say?" | 1.74 s |
| `q_describe_detail.wav` | "Describe this scene in detail, please." | 3.08 s |
| `showcase/q_see.wav` | "What do you see?" | 1.42 s |

## Golden data

- `kb_golden.json`: the knowledge-base retrieval golden set (on-topic and off-topic questions).
- `router_golden.json`: Demo 3's question router golden set.
- `skill_trigger_similarity.json`: measured EmbeddingGemma similarities of the skill trigger questions.
- `yolo26n/cats_golden.json`: the detector's strict-GPU fp32 detections on `yolo26n/cats_640x480_rgba.u8`.

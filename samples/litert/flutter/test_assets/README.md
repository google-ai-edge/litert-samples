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
- No `.litertlm` bytes: the app reads a `.litertlm` only through flutter_edge_ai_litertlm's
  `readLiteRtLmBundleInfo`, and the tests describe their files with hand-built `LiteRtLmBundleInfo`s
  (`test/support/fake_bundles.dart`).

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
| `showcase/q_mug.wav` | "Where did I leave my pirate mug?" | 2.48 s |
| `../assets/selftest/what_time_is_it.wav` (an app asset: the self-test's step 7b and the embedder's audio warm-up) | "What time is it?" | 1.37 s |

## Golden data

- `kb_golden.json`: the knowledge-base retrieval golden set (on-topic and off-topic questions). For its audio and
  Whisper paths `integration_test/kb_retrieval_test.dart` speaks the same questions with the app's own TTS into memory
  at start-up (`integration_test/support/question_audio.dart`, the code that makes the clips above); they are not
  shipped.
- `router_golden.json`: Demo 3's question router golden set.
- `skill_trigger_similarity.json`: measured EmbeddingGemma 2 similarities of the skill trigger questions against the
  knowledge base; `integration_test/kb_retrieval_test.dart` prints it again (`KB_SKILL_TRIGGERS`).
- `yolo26n/cats_golden.json`: the detector's strict-GPU fp32 detections on `yolo26n/cats_640x480_rgba.u8`.

## Windows hardware probe

`windows/lnl-cim.json` (an Intel Core Ultra 7 258V, Lunar Lake, with an Arc 140V and its NPU) and `windows/ptl-cim.json`
(an Intel Core Ultra X7 358H, Panther Lake, with an Arc B390 and its NPU), and `windows/vm-cim.json` (a Windows Server
2025 VM on Google Compute Engine with no display adapter): what Windows PowerShell 5.1 printed for the CIM and PnP
queries of the app's Windows hardware probe (`lib/data/services/hardware/windows_hardware_info_service.dart`) on those
machines, as `ConvertTo-Json` writes it (CRLF line ends, `&` escaped as a JSON `\u` escape). Read by
`test/data/services/hardware/windows_hardware_info_service_test.dart`.

Their `npu` arrays (`Get-PnpDevice -Class ComputeAccelerator -PresentOnly`) are also what the app's NPU check reads
on Windows (`lib/data/services/llm/npu_availability.dart`, tested in
`test/data/services/llm/npu_availability_test.dart`).

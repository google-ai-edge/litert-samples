# Receipt to JSON on Android with LiteRT-LM

## What this sample demonstrates

- A receipt's store, date, tax and total as JSON, on the phone: take or pick a photo of a receipt (or tap **Example**), tap **Extract**, and the JSON appears beside the photo, ready for **Copy JSON**. A US receipt's date (`10/18/25`, `11/21/2025`) comes back as `YYYY-MM-DD` (`2025-10-18`, `2025-11-21`), the form a spreadsheet or an expense tool sorts by. Reading runs on the phone, offline, once the model file is downloaded.
- [Xiaomi-OCR-0](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0) (Xiaomi, 0.8B parameters: a 12-layer vision encoder and a Qwen3.5 hybrid text decoder, trained from Qwen3.5-0.8B-Base for OCR), converted to a LiteRT-LM bundle, with its text decoder and vision encoder on the CPU, through the LiteRT-LM Kotlin API (`litertlm-android` 0.19.0). On a Galaxy S26 and a Pixel 11 Pro, the bundle's card measured a CPU request four to six times as fast as a GPU request. The bundle's card (the table below) says it "extracts named fields as JSON"; Xiaomi's card lists key information extraction (KIE) among its tasks.
- The model card's key-information-extraction prompt: `Extract key information in the image`, followed by a JSON schema with one empty string per key. The keys come from the editor, so a new field is a new word in the editor, not new code; `items[name, price]` asks for a list with one object per row.
- One `Engine` per read, made in advance. The model's text decoder keeps a recurrent state that a second conversation on the same `Engine` would continue from ([LiteRT-LM #3165](https://github.com/google-ai-edge/LiteRT-LM/issues/3165)), so the app makes a new `Engine` for every read, and makes the next one while you pick the next photo.
- The bundle reads one 768 × 1344 picture. LiteRT-LM resizes every picture to that size without keeping its aspect ratio, so the app prepares it as the model card's script does ([`ocr_xiaomi.py`](https://github.com/john-rocky/hf-to-litertlm/blob/main/xiaomiocr_work/ocr_xiaomi.py), `pad_to`; the same rules in Kotlin): the photo turned upright by its EXIF orientation, centred on a white canvas of the tall 4:7 shape, and resized with Pillow's bicubic filter. JVM tests check the app's pixels against hashes of Pillow's (written by [`tools/make_picture_rows.py`](android/tools/make_picture_rows.py)).

## The model

| Hugging Face repo | File | Size | License | Revision |
|---|---|---|---|---|
| [litert-community/Xiaomi-OCR-0](https://huggingface.co/litert-community/Xiaomi-OCR-0) | `Xiaomi-OCR-0_int8.litertlm` (int8 text decoder, fp16 vision encoder) | 1.31 GB | Apache-2.0 | `9012fe32` |

The app downloads the file on first use into its private storage and checks its size and SHA-256 against [`app/src/main/assets/models.json`](android/app/src/main/assets/models.json). The model file is not in the APK. The bundle is a conversion of [xiaomi-research/Xiaomi-OCR-0](https://huggingface.co/xiaomi-research/Xiaomi-OCR-0) (Apache-2.0); the bundle's card describes the conversion and its checks.

## Requirements

JDK 17 and Android SDK platform 36 to build. The app builds for arm64-v8a on Android 12+ (minSdk 31). The first start downloads 1.31 GB and needs twice that free in the app's storage. Before it loads the model, the app checks that about 3.2 GB of memory is free: over the reads of its checks, the free memory of a Pixel 11 Pro dropped by up to 2.6 GB (with the camera app open in one of them) and a Galaxy S26's by 2.4 GB. A phone with 8 GB of RAM was not tried.

## What it reads, and what it does not

The 12 bundled receipts (six stores, each as a flat scan and as a photo-like copy) with the default keys, on a Mac with the same file and runtime (LiteRT-LM 0.19.0, CPU): 11 came back with all four fields right. The miss was the farm stand's photo-like copy, whose store name came back without its spaces (`HILLTOPFARMSTAND`); the original weights in transformers made the same miss. On a Pixel 11 Pro through the app (**Example**, then **Extract**, for each of the 12), the same 11 came back right, with the same miss, and each answer was the same text as on the Mac.

From these and other receipts drawn the same way:

- A date printed with the month first and a day above 12 (`10/18/25`, `11/21/2025`) comes back as `YYYY-MM-DD`.
- A date whose day is 12 or less can come back with the day and the month swapped on a photo: `11/02/2025` came back as `2025-02-11` from a photo-like copy and as `2025-11-02` from the scan.
- A date in words (`Oct 30, 2025`) and a date printed as `2025-10-28` come back as printed.
- A list (`items[name, price]` with the total) on longer receipts came back with every row right on 3 of 12: on 7 of the others the only error was the letter O next to a digit read as a zero (`8OZ` as `80Z`), and on 2 the second line of a multi-quantity item (`2 @ 8.99`) came back as an item of its own. The total was right on all 12.
- Only month-first US dates were tried; a day-first date (`21/11/2025`) was not. Keys other than these four and `items[name, price]` were not measured.

## Architecture

```
a photo (system photo picker, camera, or a bundled example)
  -> turned upright by its EXIF orientation
  -> Pictures.padTo: centred on a white canvas of the tall 4:7 shape, resized to 768 x 1344
     (Pillow's bicubic resize)
  -> PNG + the key-information-extraction prompt with your keys
  -> Conversation.sendMessageAsync([picture, prompt]) -> the answer, streamed, beside the photo
  -> the JSON object in the answer -> Copy JSON
one Engine per read; the next Engine is made on the runtime thread as soon as a read ends
```

| File | What it holds |
|---|---|
| `OcrEngine.kt` | The LiteRT-LM `Engine` and `Conversation` of one read, the next one made in advance, the stream, Stop |
| `Pictures.kt`, `PilBicubic.kt` | The 768 × 1344 picture of the card's script: the white 4:7 canvas, Pillow 13.0.0's bicubic resize |
| `Kie.kt` | The model card's key-information-extraction prompt for your keys (lists included); the JSON object in the answer |
| `Photo.kt` | Decoding a photo as Pillow reads it (no color conversion), the EXIF turn, PNG for the runtime |
| `MemoryCheck.kt` | The free-memory check before an `Engine` is made |
| `ReceiptViewModel.kt`, `ReceiptScreen.kt` | The screen: the photo beside its JSON, the keys, the photo sources, Extract |
| `AutoRun.kt` | Debug builds: a plan of reads through the Extract button's code, one result line each |
| `data/` | The reader of `models.json`, the free-space rule and the download store (resume, SHA-256, HTTPS only), from [`voice_assistant`](../voice_assistant/) |

## Quickstart

```sh
cd samples/litert_lm/receipt_to_json/android
./gradlew :app:installDebug
```

Open the app and tap **Download (1.3 GB) and load**. Then tap **Example** (receipts drawn for this sample) or **Photo** / **Camera**, change the keys if you like, and tap **Extract**. The keys are separated by commas; each one becomes a key of the JSON the model writes, and `name[field, field]` becomes a list of objects with those fields.

## Development

**Side-load (development only).** A copy of the model file in the app's external files dir is hashed and, when its size and SHA-256 match `models.json`, imported instead of downloaded; the next **Download (1.3 GB) and load** logs `side-loaded <file> ... (sha256 verified)` under `adb logcat -s ReceiptToJsonDownload`. Open the app once first, so that it makes that dir:

```sh
P=com.google.ai.edge.examples.receipt_to_json
adb shell am start -n $P/.MainActivity
adb push Xiaomi-OCR-0_int8.litertlm /sdcard/Android/data/$P/files/
```

This worked on a Pixel 11 Pro (Android 17), where a copy made as the app (`run-as`) into that dir is refused. On a Galaxy S26 the copy was made as the app (a debug build):

```sh
adb push Xiaomi-OCR-0_int8.litertlm /data/local/tmp/
adb shell "run-as $P sh -c 'mkdir -p /sdcard/Android/data/$P/files && cp /data/local/tmp/Xiaomi-OCR-0_int8.litertlm /sdcard/Android/data/$P/files/'"
```

**Launch extras (debug builds only; a release build ignores them).** A running screen takes them again (`singleTop`). With any of them the screen shows over the keyguard and stays on.

| Extra | What it does |
|---|---|
| `--ez autoload true` | load the model now (the side-loaded copy, else the download) |
| `--es image <path>` | show this picture (a path in the app's external files dir), with no answer |
| `--es keys <keys>` | put these keys in the editor (comma-separated) |
| `--es plan <path>` | once the model is loaded, run the reads of this plan through the Extract button's code (`AutoRun.kt`) and write one result line each to `runs/<plan name>.jsonl` in the external files dir |

**Tests.** `./gradlew :app:testDebugUnitTest` checks the picture preparation against Pillow 13.0.0 and the card's script ([`tools/make_picture_rows.py`](android/tools/make_picture_rows.py) writes the expected SHA-256 of every picture: pictures taller and wider than 4:7, canvas sides half-way between two integers, the eight EXIF orientations), the prompt against the script's, the reading of the keys and of the answer's JSON, the catalog and the memory check.

**Example receipts.** [`tools/make_examples.py`](android/tools/make_examples.py) draws the bundled receipts (invented stores and items, DejaVu fonts) and writes `examples.json` with each receipt's fields: store, tax and total as printed, the date as `YYYY-MM-DD` and as printed.

## Licenses

See [NOTICE](android/NOTICE) and [licenses/](android/licenses/): the Pillow port (MIT-CMU).

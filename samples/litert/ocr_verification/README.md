# OCR verification: MiniCPM-OCR-Jev on LiteRT (Compiled Model API)

Take or choose a photo of printed text: the app reads it line by line with an on-device text recognizer, and gives each line a score from 0 to 1 that the text it read is correct. Lines at or above the threshold are marked **Accept**, the rest **Review**, so only the doubtful lines need a person. Edit a line's text and score it again. The photo and the text stay on the phone.

[MiniCPM-OCR-Jev](https://huggingface.co/openbmb/MiniCPM-OCR-Jev) (OpenBMB, Apache-2.0) is "a 2B OCR correctness decision model for text, formula, and table regions. Given a region image and a candidate OCR result, it returns a score from 0 to 1 — higher means the candidate is more likely correct. Use it to auto-accept reliable regions, route the rest to human review, or pick the best result among several OCR models." It reads the region of the photo and a picture of the candidate text drawn by its renderer. This app does all of it on the phone: [ML Kit Text Recognition v2](https://developers.google.com/ml-kit/vision/text-recognition/v2) finds the lines and reads them (the Latin model, bundled in the app), a WebView draws each line's text as the model's renderer does, Kotlin code prepares the inputs as the model's host does, and the converted graphs of [litert-community/MiniCPM-OCR-Decision-2B-LiteRT](https://huggingface.co/litert-community/MiniCPM-OCR-Decision-2B-LiteRT) run on the GPU through LiteRT's Compiled Model API.

## Run it (Android)

1. Download five files of [litert-community/MiniCPM-OCR-Decision-2B-LiteRT](https://huggingface.co/litert-community/MiniCPM-OCR-Decision-2B-LiteRT) (5,110,221,205 bytes): the tokenizer, the tables, the 400- and 576-patch picture towers and the 384-token decision graph. `pip install -U huggingface_hub` gives the `hf` command.

   ```bash
   export HF_HUB_DISABLE_XET=1
   hf download litert-community/MiniCPM-OCR-Decision-2B-LiteRT \
     tokenizer.json tables/tables.safetensors \
     tower_N400_fp16fc.tflite tower_N576_fp16fc.tflite decide_L384_f16sfc.tflite \
     --revision 672db2e312814fd9b56aa02d494755676140cfdb \
     --local-dir "$HOME/Downloads/MiniCPM-OCR-Decision-2B-LiteRT"
   ```

2. With a phone connected over adb (USB debugging on; Gradle needs the Android SDK 35 and a JDK 17, or open `kotlin_gpu/android` in Android Studio), build and install the debug build, then copy the files into the app's private storage:

   ```bash
   cd kotlin_gpu/android
   ./gradlew :app:installDebug
   ./install_to_device.sh
   ```

   `install_to_device.sh` reads `~/Downloads/MiniCPM-OCR-Decision-2B-LiteRT` unless you give it another directory. It checks every file's size before it copies anything and its SHA-256 on the phone after the copy. It copies with `run-as`, which needs the debug build. `MORE="decide_L512_f16sfc.tflite tower_N784_fp16fc.tflite"` (or any other decision file of up to 3,072 tokens or tower file of up to 3,136 patches from the repository, downloaded too) adds larger files: a line then takes the smallest file that holds it.

3. Open **OCR Check**. The first start checks the files' SHA-256 and builds the graphs on the GPU (32 s on a Galaxy S26). Then take a photo (**Camera**), choose one (**Photos**), or open one of three example pages (**Example**). Each line the recognizer reads gets a numbered box on the photo and a row below it: the text, the recognizer's own confidence (**OCR conf**), the line's kind, the score, and **Accept** or **Review** with the line's time on the phone. Move the slider to change the threshold (0.50 at start). Edit a line's text, or tap its kind (**Text ▾**) to set Formula or Table, and tap **Score again**.

The app stops with a message, and loads nothing, when a model file is missing or its SHA-256 differs, and when the phone does not have the free memory the graphs need. A line whose pictures or text need a larger graph file than the installed ones gets a message in its row instead of a score.

## Speed and memory

On a Galaxy S26 (SM-S942Q, Android 16, LiteRT 2.3.0), the first run after the phone rested (thermal status 0), on the homework example page, from the app's log:

| Step, for one line | Time on the Galaxy S26 |
|---|---:|
| The line's text drawn into a picture (WebView) | 359–514 ms (the first line 514 ms) |
| The inputs prepared (Kotlin: the crop, the pictures, the tokens, the embedding rows, the score) | 52–116 ms |
| Picture tower on the region (GPU) | 227–233 ms |
| Picture tower on the drawn text (GPU) | 198–218 ms |
| Decision graph (GPU) | 390–393 ms |
| **A line, as its row shows it** | **1.23–1.47 s** |

The recognizer reads the page once, before the lines are scored: 161 ms for this page. The three graph calls of a line add up to 818–842 ms; the model card gives 861.6 ms for a text region of similar length on this phone's GPU, from a test app on LiteRT 2.2.0 with inputs written by a Mac. The first start took 32 s, 28 s of it building the graphs (the decision graph 18 s, each tower 5 s).

Before it builds anything the app reads `MemAvailable` and needs the largest decision file, the two largest towers and 0.8 GiB: 4.6 GiB with the five files (the app's messages call 2^30 bytes a GB). After the first start, 1.9 GiB was available and the app's resident memory was 6.5 GiB.

## How it runs

| Step | Where | Notes |
|---|---|---|
| Photo → lines | ML Kit Text Recognition v2 (Latin, bundled) | each line's text, box and the recognizer's own confidence, top to bottom |
| Line → region picture | Kotlin | the line's box with a margin of a fifth of its height (at least 4 pixels) |
| Text → rendered picture | WebView | the model's renderer page (`RenderPage.kt`, a port of the source's `render_label.py` v8.5): Markdown with KaTeX 0.16.47 for math, DejaVu Sans for Latin text, at the source's 1600 × 1200 CSS-pixel viewport; drawn at 1 pixel per CSS pixel for text and tables and 3 for formulas, the white border trimmed to 16 pixels |
| Pictures → patches | Kotlin | the source's size rule (multiples of 32 pixels, 65,536 to 16,777,216 pixels), PyTorch's uint8 antialiased bicubic resize, 16 × 16 patches, position and rotary tables |
| Prompt | Kotlin | the source's prompt (its template's MD5 is checked at start); byte-level BPE from the checkpoint's `tokenizer.json` (`QwenBpeTokenizer.kt`) |
| Picture towers (Qwen3.5 vision encoder), one call per picture | GPU, fp32 | one file per size (400 to 3,136 patches); a picture takes the smallest installed file that holds it |
| Embeddings and M-RoPE tables | Kotlin | rows of the bf16 embedding table (`tables.safetensors`, memory-mapped), the towers' rows at the image positions |
| Decision graph (the Qwen3.5 language model without its head) | GPU, the default precision | one file per length (256 to 3,072 tokens); the request takes the smallest installed file that holds it |
| Score | Kotlin | the classification head (`score.weight`) on the last token, softmax, the probability of "correct" |

At most two tower files and one decision file stay built at once; a request that needs another closes the least recently used one first.

## How the scores were checked

On a Galaxy S26 on 2026-11-06, with the five files above and five larger ones of the same repository:

- **Inputs.** For 12 requests, every input the app prepares for the graphs (the picture arrays, the patch position and rotary tables, the token ids and the text rows of the embeddings) equals bit for bit what the source's preprocessing builds in Python on a Mac: the source's three examples (a text, a formula and a table region) with the source's rendered pictures, the same three with the app's own rendering, and the six lines of the homework example page as the app cut and drew them. The picture rows of the embeddings are the phone's own tower outputs.
- **Scores.** On the three examples with the source's rendered pictures the phone gives 0.001938, 0.968483 and 0.000912. The model card gives 0.00196, 0.96756 and 0.00087 for the same files on a Mac's GPU (Metal).
- **Rendering.** With the app's own pictures of the same three texts the phone gives 0.002481, 0.967710 and 0.000954, and the same graph files on the Mac's GPU, given those pictures, 0.002543, 0.967124 and 0.000917. The app's pictures of the text and the table have the source's size; the formula's is 1,527 × 114 pixels, the source's 1,539 × 114.

## Run it from a computer

The app takes runs from adb through the same code as the buttons, for scripted checks and screen recordings:

```bash
adb shell am start -n com.google.ai.edge.examples.ocr_verification/.MainActivity --es run example:homework.png
```

`--es run start` shows the opening screen; `example:<file>` opens an example page; `photo:<name>` opens a picture copied into the app's `files/inputs/`; `check:<case>` scores a region picture with a given rendered picture and then with the app's own rendering (a folder `files/check/<case>/` with `block.png`, `reference_render.png`, `text.txt` and `category.txt`) and writes every graph input as raw arrays. `--ez dump true` also writes each line's pictures and graph inputs. Every run writes its numbers to `files/runs/` and to the log (tag `OcrCheck`).

## Limits

- Printed text in Latin script: the recognizer is ML Kit's Latin model, and the example pages are English. The model's card lists English and Chinese as its languages.
- One line is one region. The kind of each line is Text until you change it; formulas and tables are drawn the way the source draws them only when their row is set to Formula or Table, and the recognizer reads a formula as plain text (a superscript as a plain digit).
- The renderer covers the Markdown an OCR line needs (paragraphs, line breaks, emphasis, code spans, links, escapes, headings, bullet lists, inline HTML); block quotes, code blocks, rules and Markdown tables are drawn as plain text. Tables are drawn from HTML (`<table>`), as the source's OCR writes them.
- The rendered picture is drawn by the phone's WebView, not by the source's Chromium: a picture can differ from the source's by a few pixels, and the score with it (see the checks above).
- Run on one phone, a Galaxy S26, with the five files and with five larger ones. Larger files need more free memory than the 4.6 GiB of the five.

## License

The sample's code is Apache-2.0. What it ports from MiniCPM-OCR, transformers, PyTorch, Python-Markdown and Hugging Face tokenizers, and what it bundles (KaTeX, the DejaVu fonts), are described in [`NOTICE`](kotlin_gpu/android/NOTICE); the license texts are in [`licenses/`](kotlin_gpu/android/licenses/). The model, MiniCPM-OCR-Jev, is Apache-2.0. ML Kit is used under the [ML Kit terms](https://developers.google.com/ml-kit/terms).

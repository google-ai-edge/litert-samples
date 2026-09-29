# Zero-shot text classification: Laya multilingual (mmBERT-base) on LiteRT (Compiled Model API)

This sample answers questions about a text that you define at run time, with no training step: pick one of several options (choice), place the text on an ordinal scale (score), or give a yes/no probability. Each question is one forward pass, on [LiteRT](https://github.com/google-ai-edge/litert), of the multilingual [Laya](https://huggingface.co/convaiinnovations/laya) checkpoint, an encoder built on [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base), converted in [litert-community/Laya-Multilingual-LiteRT](https://huggingface.co/litert-community/Laya-Multilingual-LiteRT). Tested with Japanese and English prompts.

The host code tokenizes the text, builds the prompt with one `<mask>` marker per option, looks up the embedding rows, reads the logits at the marker positions, and applies softmax with the calibration temperatures. The model itself is three files:

1. **Encoder graph** (251 MB): the mmBERT-base encoder and Laya's question heads, with fp16 weights. It takes embedding rows instead of token ids: with the token table looked up on the host, this fp16-weight encoder compiles for the GPU as one graph (all 1,779 ops on the GPU).
2. **Action head** (0.8 MB): a small graph that turns the pooled encoder output and the answer distribution into `act_probability`.
3. **Token table** (393 MB): the fp16 embedding row of every token, memory-mapped on the host.

## Quick start (Python, desktop)

```bash
pip install numpy transformers ai-edge-litert huggingface_hub
hf download litert-community/Laya-Multilingual-LiteRT laya_host.py laya_ml_s256_embeds_wfp16.tflite \
  laya_ml_act_head_fp32.tflite laya_ml_calibration.json tokenizer.json tokenizer_config.json \
  token_embeddings_fp16.bin token_embeddings.json --revision 058cbf34ed2c6a2854ec60e9534c229d08740387 --local-dir laya
cd laya
python - <<'EOF'
from laya_host import LayaHost
q = {"intent": {"type": "choice", "instructions": "What does the customer need?",
                "criteria": {"refund": "money returned", "help": "technical help"}}}
with LayaHost(".", "laya_ml_s256_embeds_wfp16.tflite", "laya_ml_act_head_fp32.tflite", 256, 256,
              "laya_ml_calibration.json", "token_embeddings_fp16.bin") as host:
    print(host.predict("I was charged twice for the same order. Please refund me.", q)["answers"])
EOF
```

`laya_host.py` is the Python host from the model card (CPU). The script prints `{'intent': {'type': 'choice', 'choice': 'refund', 'probabilities': {'refund': 0.9956, 'help': 0.0044}, 'confidence': 0.9591, 'action': {'act_probability': 1.0}}}`.

## Android app (`kotlin_cpu_gpu/android`)

```bash
cd kotlin_cpu_gpu/android
./gradlew :app:installDebug        # or open in Android Studio
```

The app is a Kotlin port of the same host code (tokenizer, prompt builder, table lookup, decoder) with a Compose UI; both graphs run through the Compiled Model API on the GPU (FP32) or the CPU. When the model files are missing, the app shows **Download (679 MB)**. It fetches the seven files from Hugging Face at a pinned revision into its private storage and checks each one against the SHA-256 in `app/src/main/assets/model_manifest.json`. Keep the app open while it downloads: closing it pauses the download, and the next launch offers **Resume download** from the bytes on disk. Uninstalling the app or clearing its storage deletes the files. Then pick a question preset (email triage, support intent, moderation), load the Japanese or English example or type your own text, choose GPU or CPU, and tap Run. `./gradlew :app:connectedDebugAndroidTest` runs the on-device check against the 201 published reference rows.

## Performance

| Galaxy S26, LiteRT 2.2.0, release build | GPU (FP32) | CPU |
|---|---|---|
| Launch to Ready | 2.3 s | 1.5 s |
| Five questions (the email preset, 529 tokens) | 0.29–0.31 s | 0.49–0.50 s |

Launch to Ready includes loading the tokenizer and the model, compiling, and a warm-up pass. In the on-device check (debug build), the two graphs take 51 ms per question on the GPU (median of 200 rows). The debug build from `installDebug` took 3.1–3.3 s to Ready and 0.34–0.36 s for five questions on the GPU. The on-device check matches the official laya 0.3.4 answers on both accelerators: the same top answer on all 81 choice and score questions of the 201 rows, with a maximum probability difference of 0.0014.

The prompt window is 256 tokens, and longer text is cut on the right. The model card also has a 512-token graph; this app does not use it.

## Android app on the NPU (`kotlin_npu/android_jit`)

The same app with the encoder on the Qualcomm NPU as a third accelerator choice. LiteRT compiles the graph for the NPU on the phone and caches it. The NPU runtime comes from the LiteRT release and the Qualcomm AI Runtime, and the app is installed as a bundle for the phone's Snapdragon generation; [`kotlin_npu/android_jit/README.md`](kotlin_npu/android_jit/README.md) has the steps and the numbers.

## Conversion

The graphs were exported with `litert-torch` 0.9.3. The encoder graph stores its fully connected weights as fp16, written by `ai-edge-quantizer` 0.8.0 (FLOAT_CASTING); the action head stays fp32. The host contract (`HOST_CONTRACT.md`) and the Python host are on the [model card](https://huggingface.co/litert-community/Laya-Multilingual-LiteRT).

## License

The sample code is Apache-2.0. Laya's code, checkpoint and tokenizer are Apache-2.0 (Convai Innovations), and mmBERT-base is MIT (Johns Hopkins CLSP). The Kotlin host ports Laya's host logic; attribution and license texts are in [`NOTICE`](kotlin_cpu_gpu/android/NOTICE) and [`licenses/`](kotlin_cpu_gpu/android/licenses/).

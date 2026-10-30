# Component licenses and attribution

The sample's own license is [Apache-2.0](../../../../../../LICENSE). The third-party license texts below are byte-for-byte copies of the files at the sources listed; [NOTICE](../NOTICE) describes what the sample ports from each component.

| Component | License copy | Source |
|---|---|---|
| WeVisOne-VL package (prompts, answer parsing, matte steps, box encoding) | [WeVisOne-VL-APACHE-2.0.txt](WeVisOne-VL-APACHE-2.0.txt) | [`LICENSE` at commit 40f735b8](https://github.com/Tencent/WeVisOne-VL/blob/40f735b8daed437ece982db9718e9c9856f52a79/LICENSE) |
| Pillow 12.2.0 (bicubic resampling, `PilBicubic.kt`) | [Pillow-MIT-CMU.txt](Pillow-MIT-CMU.txt) | [`LICENSE` at 12.2.0](https://github.com/python-pillow/Pillow/blob/12.2.0/LICENSE) |
| Hugging Face tokenizers (the `tokenizer.json` contract) | [Tokenizers-APACHE-2.0.txt](Tokenizers-APACHE-2.0.txt) | [`LICENSE` at v0.23.2](https://github.com/huggingface/tokenizers/blob/v0.23.2/LICENSE) |
| Hugging Face transformers (the `tokenizer.json` contract) | [Transformers-APACHE-2.0.txt](Transformers-APACHE-2.0.txt) | [`LICENSE` at v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/LICENSE) |

The model files are not part of this tree. WeVisOne-VL-2B is Apache-2.0, with Qwen3-VL-2B-Instruct (Apache-2.0) listed as a third-party component in its [LICENSE](https://huggingface.co/tencent/WeVisOne-VL-2B/blob/3dad9ce8bcb6fbe0aea945bf8860bd2cfc3663bf/LICENSE); the converted files are in [litert-community/WeVisOne-VL-2B](https://huggingface.co/litert-community/WeVisOne-VL-2B).

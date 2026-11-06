# Component licenses and attribution

The sample's own license is [Apache-2.0](../../../../../../LICENSE). The third-party license texts below are byte-for-byte copies of the files at the sources listed; [NOTICE](../NOTICE) describes what the sample ports from or bundles of each component.

| Component | License copy | Source |
|---|---|---|
| MiniCPM-OCR (the renderer's page, the score path's prompt and token checks) | [MiniCPM-OCR-APACHE-2.0.txt](MiniCPM-OCR-APACHE-2.0.txt) | [`LICENSE` at commit 8177040f](https://github.com/OpenBMB/MiniCPM-OCR/blob/8177040f0d6c824ccf87bc18b938b6a0f91b6d0e/LICENSE) |
| Hugging Face transformers (the picture preprocessing, position and rotary tables; the `tokenizer.json` contract) | [Transformers-APACHE-2.0.txt](Transformers-APACHE-2.0.txt) | [`LICENSE` at v5.17.0](https://github.com/huggingface/transformers/blob/v5.17.0/LICENSE) |
| Hugging Face tokenizers (the `tokenizer.json` contract) | [Tokenizers-APACHE-2.0.txt](Tokenizers-APACHE-2.0.txt) | [`LICENSE` at v0.23.2](https://github.com/huggingface/tokenizers/blob/v0.23.2/LICENSE) |
| PyTorch (uint8 antialiased bicubic resize) | [PyTorch-BSD-3-Clause.txt](PyTorch-BSD-3-Clause.txt) | [`LICENSE` at v2.14.1](https://github.com/pytorch/pytorch/blob/v2.14.1/LICENSE) |
| Python-Markdown 3.10.3 (the HTML the renderer's Markdown step writes) | [Python-Markdown-BSD-3-Clause.txt](Python-Markdown-BSD-3-Clause.txt) | [`LICENSE.md` at 3.10.3](https://github.com/Python-Markdown/markdown/blob/3.10.3/LICENSE.md) |
| KaTeX 0.16.47 (bundled in `app/src/main/assets/render/katex/`) | [KaTeX-MIT.txt](KaTeX-MIT.txt) | [`LICENSE` at v0.16.47](https://github.com/KaTeX/KaTeX/blob/v0.16.47/LICENSE) |
| DejaVu fonts 2.37 (DejaVu Sans and DejaVu Sans Bold bundled in `app/src/main/assets/render/fonts/`) | [DejaVu-Fonts-LICENSE.txt](DejaVu-Fonts-LICENSE.txt) | [`LICENSE` at version_2_37](https://github.com/dejavu-fonts/dejavu-fonts/blob/version_2_37/LICENSE) |

The model files are not part of this tree. MiniCPM-OCR-Jev is Apache-2.0 ([model card](https://huggingface.co/openbmb/MiniCPM-OCR-Jev)), trained from Qwen3.5-2B (Apache-2.0); the converted files are in [litert-community/MiniCPM-OCR-Decision-2B-LiteRT](https://huggingface.co/litert-community/MiniCPM-OCR-Decision-2B-LiteRT). The text recognizer is ML Kit Text Recognition v2 (`com.google.mlkit:text-recognition`), used under the [ML Kit terms](https://developers.google.com/ml-kit/terms).

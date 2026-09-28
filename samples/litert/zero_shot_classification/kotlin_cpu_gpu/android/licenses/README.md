# Component licenses and attribution

The sample's own license is [Apache-2.0](../../../../../../LICENSE). Third-party license texts
below are retained with their source declarations; [NOTICE](../NOTICE) describes
the port and model-conversion changes.

| Component | License copy | Source |
|---|---|---|
| Laya code, checkpoint, and tokenizer | [Laya-APACHE-2.0.txt](Laya-APACHE-2.0.txt) | Laya source LICENSE at commit 42626c348753fbb17572a813127df2278a1ec527; [pinned model declaration](https://huggingface.co/convaiinnovations/laya/blob/1c5edc17a7acd8701df6fc341c0d179f1c62c982/README.md) |
| ModernBERT | [ModernBERT-APACHE-2.0.txt](ModernBERT-APACHE-2.0.txt) | [Upstream LICENSE at c6d9423](https://github.com/AnswerDotAI/ModernBERT/blob/c6d942312f1b0b24d423628b8e477a3e97c7038f/LICENSE) |
| mmBERT-base | [mmBERT-MIT.txt](mmBERT-MIT.txt) | [Pinned model-card MIT declaration and author attribution](https://huggingface.co/jhu-clsp/mmBERT-base/blob/c5955035435e2bf121cde7f3c8863ef52ff35d82/README.md); [standard MIT text](https://opensource.org/license/mit) |
| Transformers 5.17.0 | [Transformers-APACHE-2.0.txt](Transformers-APACHE-2.0.txt) | License retained from the installed 5.17.0 distribution; [upstream project](https://github.com/huggingface/transformers) |
| tokenizers 0.23.2 | [Tokenizers-APACHE-2.0.txt](Tokenizers-APACHE-2.0.txt) | [Upstream LICENSE at v0.23.2](https://github.com/huggingface/tokenizers/blob/v0.23.2/LICENSE) |

The ModernBERT license retains its received `Copyright 2022 MosaicML Examples authors`
notice. The pinned mmBERT model repository declares MIT without a separate LICENSE/NOTICE
or copyright notice. Its standard permission text and model-card author attribution are
included; no copyright holder or year is invented.

The checkpoint is Laya's trained multilingual checkpoint, not a newly downloaded mmBERT
or ModernBERT weight file. The independent upstream source revisions above identify
license/attribution evidence. The unchanged calibration JSON retains its dataset provenance;
raw calibration corpora are not packaged with this sample.

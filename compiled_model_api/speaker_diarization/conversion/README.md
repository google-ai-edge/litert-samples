# Diarization models → LiteRT / ONNX conversion

`build_diar.py` converts both models with parity checks at every stage:

- **WeSpeaker ResNet34** ([pyannote/wespeaker-voxceleb-resnet34-LM](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM)) → GPU-native LiteRT fp16 via [litert-torch](https://github.com/google-ai-edge/LiteRT-Torch). Input = CMN'd kaldi fbank `[1, 500, 80]` (the waveform front-end runs host-side). A pure CNN — zero re-authoring except the statistics-pooling standard deviation (down-scaled unbiased variance so the fp16 sum cannot overflow). torch-vs-torch cosine 1.0, fp16 tflite cosine 1.0.
- **pyannote segmentation-3.0** ([pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)) → ONNX (opset 17) for onnxruntime CPU (the BiLSTM has no mobile-GPU kernel). ONNX-vs-PyTorch corr 1.0, per-frame argmax agreement 100%.

`gen_assets.py` exports the kaldi mel banks + hamming window as app assets and proves the plain-array fbank algorithm (the one ported to Kotlin) matches `torchaudio.compliance.kaldi.fbank` (corr 1.0, max|d| 4e-4 in the log domain).

```bash
hf download pyannote/wespeaker-voxceleb-resnet34-LM --local-dir wespeaker
hf download pyannote/segmentation-3.0 --local-dir seg30   # gated: accept conditions on HF first
python build_diar.py
python gen_assets.py
```

# PANNs CNN14 → LiteRT conversion

Produces the `cnn14_audioset_fp16.tflite` graph used by the Android sample, from
[PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) **CNN14** (`Cnn14_mAP=0.431`), with
[litert-torch](https://github.com/google-ai-edge/litert).

## Environment

```bash
pip install litert-torch ai-edge-litert ai-edge-quantizer
pip install torch numpy librosa torchlibrosa
# Model defs (models.py -> panns_models.py, pytorch_utils.py) from
#   github.com/qiuqiangkong/audioset_tagging_cnn
# Checkpoint Cnn14_mAP=0.431.pth (CC-BY-4.0) from
#   https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth
# AudioSet labels (class_labels_indices.csv) from the same repo's metadata/.
# On macOS, `import _stub_propack` first (it guards a scipy/_propack dlopen + an inspect crash).
```

## Run

```bash
python build_panns.py     # converts the CNN body to fp16, op-check + parity, exports mel_basis.bin + fixtures
```

Emits `cnn14_audioset_fp16.tflite` (the GPU CNN body) and `mel_basis.bin` (the mel filterbank for the
host log-mel). Push the tflite with `../kotlin_cpu_gpu/android/install_to_device.sh`, and copy
`mel_basis.bin` + `audioset_labels.txt` into the app's `assets/`.

## How it converts — and why the log-mel is host-side

PANNs builds its spectrogram with **torchlibrosa**, whose STFT is a **DFT-as-Conv1d** — so there is **no
FFT op** and the whole raw-audio→tags graph is almost GPU-clean. The only GPU blocker is the STFT centering
**reflect-pad** (one `GATHER_ND`); switching it to zero-pad (`pad_mode='constant'`) removes it at corr
**1.0** (only the first/last frame differ). So a fully-in-graph raw-waveform model *looks* achievable.

**But** the converted spectral front-end is unusable two ways:

1. litert-torch lowers the giant 1024-tap DFT-conv **numerically wrong even in fp32** (tflite-vs-torch corr
   ≈ 0.19).
2. The power spectrum `|STFT|²` reaches ~1e6, which **overflows fp16 on Mali → NaN**. It is not rescalable
   (log-mel is scale-nonlinear; the `bn0` that follows won't absorb the constant offset).

So the spectral front-end is computed **host-side** (the Whisper/Kokoro pattern), and only the CNN body
rides the GPU:

| part | placement | rewrite / note |
|---|---|---|
| log-mel | CPU (Kotlin `MelSpectrogram`) | reflect-pad center, periodic Hann, 1024-pt FFT, power, `librosa.filters.mel` (slaney) matmul, `10·log10(max(·,1e-10))`. Host-vs-torch corr **1.000000**, max\|d\| 0.0017. Mel basis shipped as `mel_basis.bin` [64,513]. |
| CNN14 body | **GPU** | `bn0` + 6 conv blocks + mean/max time-pool + 2 FC + sigmoid. Pure CNN — corr **1.000000** in fp32 **and** fp16, op-check banned NONE / >4D 0, one delegatable graph (45/45 LITERT_CL on Pixel 8a). |

## A general finding

**torchlibrosa / STFT-as-Conv1d audio models: do the log-mel host-side.** The STFT lowers to convolution
(no FFT op), so the op-check *looks* clean, but the converted front-end is both numerically wrong (the long
DFT-conv) and fp16-unsafe (the `|STFT|²` magnitude). Keep the spectral front-end on the CPU and put only the
CNN body on the GPU.

## Files

| File | What |
|---|---|
| `build_panns.py` | the conversion: CNN body → fp16 tflite, op-check + parity, host-mel validation, mel/fixtures export. |
| `_stub_propack.py` | macOS import guard (scipy `_propack` dlopen, `inspect.getsourcefile`). |

Upstream: [qiuqiangkong/audioset_tagging_cnn](https://github.com/qiuqiangkong/audioset_tagging_cnn) (Apache-2.0
code, CC-BY-4.0 weights).

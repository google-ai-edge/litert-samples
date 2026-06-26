# Text-to-Speech with LiteRT — Matcha-TTS (on-device, FFT-free)

An Android sample that runs [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
end-to-end on device with the LiteRT `CompiledModel` API: type text, synthesize it, and play
it back. No network, no espeak.

This is the **FFT-free** TTS path: Matcha-TTS pairs a conditional flow-matching (CFM) acoustic
model with a **HiFi-GAN time-domain vocoder**, so there is **no FFT/iSTFT anywhere** in the
synthesis path — which is what lets the heavy graphs run on the GPU delegate (spectral vocoders
need an FFT kernel the GPU delegate does not provide). 22.05 kHz, LJSpeech voice.

## Models

| Graph | In → Out | Delegate (Pixel 8a) |
| :-- | :-- | :--: |
| Text encoder | emb[1,256,192] + mask[1,1,256] → mu[1,80,256], logw[1,1,256] | GPU |
| CFM decoder (×N ODE steps) | x,mu[1,80,512] + t_sin[1,160] + mask[1,1,512] → v[1,80,512] | CPU¹ |
| HiFi-GAN vocoder | mel[1,80,512] → wav[1,1,131072] | GPU |
| English G2P (DeepPhonemizer) | text[1,96] → logits[1,96,64] | CPU |

All four graphs are loaded with `CompiledModel.create(...)`; the accelerator (`GPU`/`CPU`) is
chosen per graph. fp16 weights. Converted with [litert-torch](https://github.com/google-ai-edge/litert)
— per-graph tflite-vs-torch corr **1.000000**, end-to-end waveform corr ≥0.99. See
[`conversion/`](conversion/).

¹ The CFM decoder runs on the **CompiledModel CPU** delegate. It converts GPU-clean and is exact
on CPU, but on the Pixel 8a the Mali ML Drift GPU delegate **mis-fuses the decoder's transformer
blocks at large activation magnitude** — the residual collapses (output corr 0.006 vs CPU) when
the block is fused into the full decoder graph, while the *same block as a standalone graph runs
correctly on the GPU* (corr 0.984). So it is a graph-fusion bug, not an unsupported op (every op —
GroupNorm-4D, Mish, SnakeBeta, ZeroStuffConvT1d, the manual masked attention — verified correct on
the GPU). The GPU vocoder dominates wall time, so the pipeline stays realtime (**RTF ~0.8**).
`conversion/probe_tx_standalone.py` is the minimal repro.

## Pipeline

```
text --G2P(CPU dict+neural)--> phoneme ids
     --host: embed + intersperse + pad-->     text_encoder(GPU) -> mu, logw
     --host: durations + length-regulator-->  mu_y[1,80,T]
     --host: Euler ODE loop (N steps)-->        decoder(CPU) x N -> v
     --host: denormalize-->                     vocoder(GPU)     -> waveform
```

Fixed shapes (256 phonemes, 512 mel frames ≈ 5.9 s); a runtime float mask makes padded positions
a no-op, so one compiled graph handles any length. `MatchaSynthesizer.kt` does the host
orchestration (embedding, duration/length-regulator, sinusoidal time-embed, the Euler ODE loop,
denormalize); `MatchaG2P.kt` does the text→phoneme conversion.

### G2P (espeak-free)

Matcha-LJSpeech is trained on espeak en-us IPA, but espeak is GPL. The clean replacement is a
275k-entry espeak-IPA dictionary (from [OpenPhonemizer](https://github.com/NeuralVox/OpenPhonemizer),
Clear BSD) as primary + [DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer) (MIT) on
LiteRT CPU for out-of-dictionary words. Output IPA maps 1:1 onto the keithito 178-symbol set.

## Build & run

```bash
cd kotlin_cpu_gpu/android
./gradlew :app:installDebug
# the four .tflite graphs are pushed to the app's filesDir (too big to bundle):
./install_to_device.sh <dir-with-the-tflites>
```

Get the `.tflite` files from Hugging Face
([`mlboydaisuke/Matcha-TTS-LiteRT`](https://huggingface.co/mlboydaisuke/Matcha-TTS-LiteRT)) or
build them with [`conversion/`](conversion/). The host tables (`emb.bin`, `g2p_dict.txt.gz`,
`config.json`, `g2p_meta.json`) are bundled in the app assets. The first launch shows
"model not found" until the install script has run.

## License

Model: MIT (Matcha-TTS / HiFi-GAN). G2P: Clear BSD (OpenPhonemizer) + MIT (DeepPhonemizer).

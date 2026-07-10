# Dia2-1B — two-speaker dialogue TTS with LiteRT `CompiledModel`

This sample runs [Dia2-1B](https://huggingface.co/nari-labs/Dia2-1B) (Nari Labs, Apache-2.0) fully
on-device. You type a two-speaker script and the app speaks it:

```
[S1] Hello, how are you today? [S2] I'm great, thanks for asking.
```

Dia2 is a Moshi-style **RQ-Transformer**. Once per 12.5 Hz frame a 30-layer *temporal* transformer
emits a word-timing action plus [Mimi](https://huggingface.co/kyutai/mimi) codebook 0; a 3-layer
*depformer* then autoregressively fills the remaining 31 codebooks for that same frame. Mimi's 32
quantizers decode the codes to 24 kHz audio.

| | |
|---|---|
| Model | Dia2-1B (Apache-2.0) |
| Backbone | 30-layer temporal transformer (GQA 16Q/8KV, RoPE) + 3-layer depformer |
| Codec | kyutai/mimi, 32 quantizers |
| Sample rate | 24 kHz mono |
| Accelerator | CPU (fp32) |
| Weights | [mlboydaisuke/Dia2-1B-LiteRT](https://huggingface.co/mlboydaisuke/Dia2-1B-LiteRT) |

## Device placement

Every graph runs on the **CPU as fp32**, because fp16 collapses these deep stacks on ARM. The KV
caches, RoPE, the 34-channel embedding sum, the depformer's projections and all sampling live in
Kotlin; the graphs are pure step functions.

**Correction (2026-07-10): the GPU delegate is not the obstacle.** An earlier version of this sample
said it rejects the language models' KV-step `FULLY_CONNECTED` weight shapes. That rejection is real
on LiteRT 2.1.3 and **fixed in 2.1.5**. The depformer's own compile failure was in *our* graph: a
rank-5 reshape inside the fused-QKV authoring, above ML Drift's maximum tensor rank of 4. Slicing the
last dimension into thirds instead gives **237/237 nodes delegated at 4–7 ms/stage**; it then
miscomputed at both default and FP32 precision, which is the known BMM + broadcast-`ADD` bug, and
pre-expanding the attention mask host-side from `[1,1,1,D]` to `[1,NH,1,D]` brings it to **corr
1.000000** against the desktop CPU reference.

**We then moved the depformer onto the GPU and measured it.** All three weight sets delegate
**237/237 nodes** and the audio is bit-identical to the CPU path (4288/4288 codebook tokens equal,
waveform correlation 1.000000, RMS difference 0.000000, 133 frames, fixed seed). It is still not
worth shipping, because it is not faster. Per-graph timing on a Pixel 8a over 3906 depformer calls:

| | KV upload | small inputs | `run()` | output readback | depformer total |
|---|---|---|---|---|---|
| CPU | 0.45 s | 0.28 s | 75.8 s | 0.58 s | **76.4 s** |
| GPU | 2.0 s | 2.2 s | 17.1 s | 61.2 s | **78.3 s** |

`CompiledModel.run()` only *enqueues* — the GPU work is paid inside the first `readFloat()`, so
timing `run()` alone suggests a 4.4× win that does not exist. Per call the GPU costs 21.1 ms against
the CPU's 19.7 ms: a 3-layer, single-token step graph is too small to amortise dispatch and
synchronisation. The 30-layer temporal transformer is the graph worth moving, but at 3.0 GB fp32 it
does not fit in GPU memory as exported.

| Graph | Input | Output |
|---|---|---|
| `dia2_temporal_fp32` | emb `[1,1,1024]`, RoPE cos/sin, mask `[1,1,1,257]`, packed K/V `[1,240,256,128]` | hidden, action `[2]`, cb0 `[2050]`, new K/V |
| `dia2_depformer_wi{0,1,2}` | dep_in `[1,1,1024]`, RoPE, mask `[1,1,1,32]`, packed K/V `[1,24,31,128]` | hidden, new K/V |
| `dia2_mimi_dequant` | codes `[1,32,1]` | latent `[1,512,1]` |
| `dia2_mimi_decode_t256` | latent `[1,512,256]` | audio `[1,1,491520]` |

## Three things that are easy to get wrong

**1. Both text streams carry real word tokens.** Channels 0 and 1 are *not* new-word/pad markers.
On a new word the main stream emits the word's first text token while the second stream emits
`NEW_WORD`; during the padding frames that follow, the main stream drains the rest of the word and
the second stream drains a two-word lookahead. Feeding markers instead produces fluent, confident,
completely wrong speech.

**2. Undo the delay pattern before decoding.** Codebook `cb` lags the aligned timeline by 16 frames
(cb0) or 18 frames (the rest). Codes are stored at `audioCodes[cb][t+1]`, so
`aligned[cb][τ] = audioCodes[cb][delay[cb] + τ]`, and the result is `max(delay)` frames shorter.
Skipping this yields muffled, unintelligible audio.

**3. Mimi decode must be a single pass.** The decode path is upsample → **causal** decoder
transformer → SEANet, so its receptive field is unbounded. Decoding disjoint windows starts each one
with no history and costs ~13% relative error (correlation 0.991 against a full-sequence PyTorch
decode). The exported graph spans 256 frames; the unused tail stays zeroed and causality makes it
exact — correlation **0.999999**.

## The speaker is sampled, so the voice prompt is baked in

With no voice prefix Dia2 **samples the speaker identity**, and the voice changes on every run
(median F0 wanders over a ~120 Hz range). Classifier-free guidance does *not* fix that: over 8
matched seeds the F0 spread is 144 Hz at `cfg_scale = 1.0` and 134 Hz at `2.0`. What guidance buys
is steadier output levels.

The model's own remedy is a **voice prefix**, which normally needs Whisper word timings and a Mimi
*encoder* — both host-only. `conversion/bake_prefix.py` therefore precomputes the prompt into a
13 kB JSON (aligned Mimi codes, `new_word_steps`, prefix word entries) that ships in `assets/`. On
device only the warm-up runs: the temporal transformer replays the prompt to prime both KV caches —
no Mimi encoder, no sampling, no depformer. The generated speakers then track their prompts
(measured on device: S1 214 Hz / S2 114 Hz against prompts of 247 Hz / 88 Hz; without a prefix S2
never drops below 214 Hz).

Classifier-free guidance (`cfg_scale = 2.0`, Dia2's default) is implemented faithfully and is
subtle: the guided logits `uncond + scale·(cond − uncond)` only **select** the top-k candidate set,
while the draw is a temperature softmax over the **conditional** logits restricted to that set. It
therefore needs a second, unconditional branch (text forced to `zero`/`pad`, the same audio codes,
its own KV cache) — hence two temporal runs and two depformer runs per frame.

## Build and run

```bash
cd conversion
export DIA2_OUT=out
python build_dia2_temporal.py
python build_dia2_depformer.py
MIMI_WORK=/path/to/mimi python build_dia2_mimi_decode.py
python export_dia2_assets.py
python bake_prefix.py

cd ../kotlin_cpu_gpu/android
./install_to_device.sh ../../conversion/out     # ~4.1 GB via adb
./gradlew installDebug
```

The first launch fails with "Load failed" until `install_to_device.sh` has been run.

## Performance and memory

On a Pixel 8a a 4-second utterance takes ~190 s: 71 warm-up frames (temporal only, ×2 guidance
branches) plus ~67 generated frames, each running 2 temporal steps and 2×31 depformer stages. This
sample is correctness-first, not real-time.

The process peaks at **~4.6 GB RSS** (3.0 GB fp32 temporal graph + 3×164 MB depformer + ~400 MB of
baked fp32 tables and KV caches) and settles around 3.2 GB. On an 8 GB phone that leaves little head
room — close other apps, or the kernel's low-memory killer may take the process mid-run.

## Validation

Every ported component was checked against the reference PyTorch implementation on host before it
reached the device:

| Component | Check | Result |
|---|---|---|
| Word-pacing state machine | vs recorded reference frame stream | 0/60 mismatches |
| Tokenizer + script parsing | vs reference entries | 10/10 exact |
| Depformer 31-stage KV glue | vs PyTorch depformer | corr 1.0000, argmax 31/31 |
| Mimi decode | vs PyTorch full-sequence decode | corr 0.999999 |
| Voice-prefix warm-up | vs reference warm-up + generation | 0 mismatches (71 + 63 frames) |

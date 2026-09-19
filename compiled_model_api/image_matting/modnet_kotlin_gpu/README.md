# MODNet — Trimap-free portrait matting (LiteRT CompiledModel GPU)

Real-time **portrait matting** running **fully on the LiteRT `CompiledModel` GPU** delegate. [MODNet](https://arxiv.org/abs/2011.11961) (AAAI 2022) predicts a **soft alpha matte** for a person — no trimap, no green screen — for background blur/replace (video calls, virtual backgrounds). ~79 ms/frame on a Pixel 8a.

- **Model:** [litert-community/MODNet-LiteRT](https://huggingface.co/litert-community/MODNet-LiteRT) · 26 MB
- **Weights:** [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) · Apache-2.0
- **Input:** `[1, 3, 512, 512]` NCHW, RGB, normalized to `[-1, 1]` (`(x/255 - 0.5)/0.5`)
- **Output:** `[1, 1, 512, 512]` soft alpha matte in `[0, 1]` (composite `fg·α + bg·(1-α)`)

## How it works

MODNet is a pure CNN (MobileNetV2 low-res branch + high-res + fusion branches) with `align_corners=False` interpolation. Two re-authoring patches make it a fully GPU-compatible graph (**0 tensors of rank > 4, 0 GPU-incompatible ops**):

1. **SE block `Linear` → `1×1 conv`** — the stock squeeze-excite `pool → Linear → view(b,c,1,1) → x*w` confuses the NCHW↔NHWC layout; 1×1 convs on the pooled tensor are identical and NCHW-clean.
2. **fp16-safe hierarchical-mean `InstanceNorm`** — MODNet's IBNorm runs `InstanceNorm2d` over up to 512×512 spatial; on the Mali GPU (fp16) the variance `sum(dd²)` overflows (≫ 65504) and the matte degrades (halos, blotchy interior, corr 0.94). Computing the mean via a cascade of `/2` average-pools (magnitude-bounded, exact for power-of-2) + `dd·rsqrt(mean(dd²)+eps)` restores GPU corr **0.99994**.

CPU-exact vs PyTorch (corr 0.99999999999); device Mali GPU corr 0.99994.

## Run

```bash
# 1. Get the model (build with ../conversion or download from Hugging Face)
#    then push it into the app's private storage:
cd android
./install_to_device.sh <dir-with-modnet.tflite>

# 2. Build & run
./gradlew :app:installDebug
```

The sample mattes a bundled portrait and displays the foreground composited over a green-screen background. Adapt `MainActivity.kt` to feed live camera frames and a blurred/replaced background for a real-time virtual-background demo.

## Convert

See [`conversion/`](conversion/) — `build_modnet.py` loads the trained MODNet weights and converts with litert-torch.

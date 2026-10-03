# MoViNet-A0 stream — conversion

Re-authors [Atze00/MoViNet-pytorch](https://github.com/Atze00/MoViNet-pytorch) (causal streaming, Kinetics-600) into a single-frame, 4D-only functional forward with explicit recurrent-state I/O, verifies parity against the original model, and converts it to a LiteRT `CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch litert-torch fvcore einops
git clone https://github.com/Atze00/MoViNet-pytorch.git
```

## Run

```bash
MOVINET_PYTORCH=./MoViNet-pytorch python build_movinet.py
# -> movinet_a0_stream.tflite  (15 MB, 47 inputs / 28 outputs, all float32)
```

`stream_model.py` holds the 4D re-authoring (`MoViNetA0Stream`): the temporal depthwise convs become a per-channel weighted sum of the buffered frames; the streaming average pools become `avg = (running_sum + mean) * inv_count`; and the tf-`same` residual average pool is reformulated as `count_include_pad=True` + a constant boundary-correction mask so it lowers to `AVERAGE_POOL_2D + MUL`.

The stream-buffer shift register and the pool running-sum accumulation are done host-side by the app (`../android/.../MoViNet.kt`); the graph consumes the recurrent state but only emits fresh tensors, which keeps the streaming numerically stable on the Mali GPU delegate.

`build_movinet.py` verifies that streaming the flat 4D module frame-by-frame (with the same host-side accumulation the app does) matches the original PyTorch model's full-clip forward (max abs diff ~2e-6).

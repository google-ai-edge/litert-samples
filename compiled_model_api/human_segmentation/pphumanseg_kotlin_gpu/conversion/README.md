# PP-HumanSeg — conversion

Converts [PP-HumanSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.9/contrib/PP-HumanSeg)
(PaddleSeg, via OpenCV Zoo, Apache-2.0) to a LiteRT `CompiledModel`-GPU `.tflite` with onnx2tf.

## Setup

```bash
pip install onnx2tf onnx onnxsim tf_keras psutil ai_edge_litert huggingface_hub
```

## Run

```bash
python build_pphumanseg.py    # downloads the OpenCV-Zoo Apache-2.0 ONNX, runs onnx2tf
# -> pphumanseg.tflite  (6 MB, [1,192,192,3] NHWC -> [1,192,192,2])
```

PP-HumanSeg is a pure CNN (Conv/Relu/Add/Resize/Softmax), so onnx2tf produces a fully
GPU-compatible NHWC tflite with zero patches: 128/128 nodes on the delegate, device
corr 1.0 vs ONNX. Input BGR `(x/255-0.5)/0.5`; `argmax` the 2-class output for the mask.

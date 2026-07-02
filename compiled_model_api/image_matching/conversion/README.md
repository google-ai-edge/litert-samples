# XFeat → LiteRT conversion

`convert_xfeat.py` converts [verlab/accelerated_features](https://github.com/verlab/accelerated_features)
XFeat (Apache-2.0) to a GPU-native LiteRT `.tflite` with litert-torch (fp16, 1.4 MB), with two
numerically-equivalent re-authorings: the input InstanceNorm moves host-side (fp16
spatial-reduction overflow on the GPU delegate), and `_unfold2d` space-to-depth becomes an exact
one-hot `Conv2d(1, 64, k=8, s=8)` (the unfold otherwise emits >4-D tensors / GATHER_ND).
Verified: op-check banned NONE / ≤4-D, tflite-vs-PyTorch corr 0.9999, Pixel 8a 72/72 LITERT_CL.

# Imagenet LiteRT end-to-end sample

This sample shows how to convert an ImageNet model from PyTorch to LiteRT format and
then run classification with the LiteRT Python package.

We recommend using `uv` to run this sample. Install `uv` by following:
https://docs.astral.sh/uv/getting-started/installation/

While you may skip the model conversion step and get a converted model from a
model hub instead, the model might be trained with different image 
preprocessing steps(or parameters) which might be slightly incompatible with 
this script. You might need to adjust the preprocessing code. 

## Run the sample

1) Convert a pretrained model to LiteRT (TFLite):

```bash
uv run main.py convert --arch mobilenet_v2
```

This writes `mobilenet_v2.tflite` to the current directory. If you choose a
different architecture, the default output name matches it (for example,
`resnet18.tflite`).

Supported architectures include:
`mobilenet_v2`, `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`,
`efficientnet_b0`-`efficientnet_b7`, `efficientnet_v2_s`, `efficientnet_v2_m`,
`efficientnet_v2_l`.

Input size for conversion is inferred from the selected torchvision weights, so
no batch/height/width flags are required.

2) Run classification on an image:

```bash
uv run main.py --image /path/to/something.jpg
```

If you converted a different architecture, pass the model path explicitly:

```bash
uv run main.py --model resnet18.tflite --image /path/to/something.jpg
```

The label files `imagenet_lsvrc_2015_synsets.txt` and `imagenet_metadata.txt` are
read from the current directory by default. The two files were copied from 
https://github.com/tensorflow/examples repo.

# Imagenet LiteRT end-to-end sample

This sample demonstrates how to convert an ImageNet model from PyTorch to LiteRT (TFLite) format and run classification using the LiteRT Python package.

While you can often download converted models from a model hub, they may have been trained with preprocessing steps (or parameters) incompatible with this script. This sample ensures the preprocessing logic matches the model conversion.

### Why convert the model yourself?

You can certainly use pre-converted `.tflite` models downloaded from model hubs (like Kaggle or TensorFlow Hub), but be aware that **preprocessing logic is tightly coupled to the model weights**.  Different training pipelines use different normalization constants (mean/std), resize methods (Bilinear vs Bicubic), or crop sizes. If you use this script with a generic downloaded model, you may see poor accuracy if those parameters don't match. Some pre-converted models even do not have this information.

This sample ensures accuracy by converting the model *and* defining the exact preprocessing steps (found in `main.py`) that match the PyTorch `torchvision` weights. If you bring your own model, simply check its documentation and update the `_load_image` or `_pick_preprocess_config` functions in `main.py` to match.

## Prerequisites

### 1. Install `uv`
We recommend using `uv` to run this sample.
[Install uv following the official guide](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Download Label Files
The script requires ImageNet label metadata to map model outputs to human-readable names. Run the following commands in the project root to download the required files:

```bash
curl -sSL -o imagenet_lsvrc_2015_synsets.txt [https://raw.githubusercontent.com/tensorflow/models/refs/heads/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt](https://raw.githubusercontent.com/tensorflow/models/refs/heads/master/research/slim/datasets/imagenet_lsvrc_2015_synsets.txt)
curl -sSL -o imagenet_metadata.txt [https://raw.githubusercontent.com/tensorflow/models/refs/heads/master/research/slim/datasets/imagenet_metadata.txt](https://raw.githubusercontent.com/tensorflow/models/refs/heads/master/research/slim/datasets/imagenet_metadata.txt)
```


## Run the sample

1) Convert a pretrained PyTorch model to LiteRT (TFLite):

```bash
uv run main.py convert --arch mobilenet_v2
```

This will generate mobilenet_v2.tflite in the current directory.

Supported architectures include:
`mobilenet_v2`, `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`,
`efficientnet_b0` through `efficientnet_b7`, `efficientnet_v2_s`, `efficientnet_v2_m`,
`efficientnet_v2_l`.

If you choose a different architecture, the default output name matches it (for example,
`resnet18.tflite`).

2) Run Classification

```bash
uv run main.py --model resnet18.tflite --image /path/to/something.jpg
```

### Options

You can customize the execution with the following flags:
  - --model: Path to the .tflite file (default: mobilenet_v2.tflite).
  - --image: Path to the input image (required).
  - --labels: Path to the synsets file (default: imagenet_lsvrc_2015_synsets.txt).
  - --metadata: Path to the metadata file (default: imagenet_metadata.txt).
  - --top_k: Number of top results to display (default: 5).

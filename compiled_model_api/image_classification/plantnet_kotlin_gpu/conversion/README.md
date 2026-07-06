# PlantNet-300K ResNet18 — conversion

Converts the [PlantNet-300K](https://github.com/plantnet/PlantNet-300K) (NeurIPS 2021)
ResNet18 (1081-species plant classifier, Apache-2.0 weights) to a LiteRT
`CompiledModel`-GPU `.tflite` with litert-torch.

## Setup

```bash
pip install torch torchvision litert-torch huggingface_hub
```

## Run

```bash
python build_plantnet.py
# -> plantnet.tflite  (47 MB, [1,3,224,224] -> [1,1081])
```

Plain torchvision ResNet18 (pure CNN). One GPU re-authoring patch (baked into the
graph → 37/37 nodes on the delegate, 1 partition): the ResNet stem
`MaxPool2d(padding=1)` lowers to a PADV2 with `-inf` padding (`PADV2: src has wrong
size` on the Mali delegate), replaced by an explicit 0-pad + unpadded maxpool — exact
since the maxpool input is post-ReLU (≥ 0).

Weights load from `cpoisson/plantnet300k-resnet18` (Apache-2.0). Input is RGB,
ImageNet-normalized, NCHW; output is 1081 species logits. Class index `i` → the `i`-th
species when the PlantNet-300K species-id strings are sorted (torchvision
`ImageFolder` order); names from `plantnet300K_species_id_2_name.json`.

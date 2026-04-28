# EfficientDet-Lite assets

`efficientdet_lite0_detection.tflite` is the TensorFlow EfficientDet-Lite0 COCO object detection model downloaded from:

https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/detection/default/1?lite-format=tflite

The analyzer expects a 320x320 RGB input and EfficientDet detection outputs:

- boxes: `[1, N, 4]` as `ymin, xmin, ymax, xmax`
- classes: `[1, N]`
- scores: `[1, N]`
- detection count: `[1]`

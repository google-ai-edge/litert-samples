Detector fixtures for the unit tests:

- `cats_640x480_rgba.u8`: raw RGBA8888, 640×480, COCO val2017 `000000039769.jpg` (`../cats.jpg`) decoded by PIL
  (SHA-256 prefix `9e0d42666864d10e`).
- `cats_golden.json`: the strict-GPU fp32 detections of `yolo26n_fp16_rawhead.tflite` for that frame (decode threshold
  0.25).

COCO `000000039769.jpg` is "Cats and remote controllers" by DocChewbacca, Flickr photo
[210383891](https://www.flickr.com/photos/st3f4n/210383891/), under
[CC BY-SA 2.0](https://creativecommons.org/licenses/by-sa/2.0/); the raw frame stays under that licence (all image
credits: `../README.md`). The model file is never committed: `tool/fetch_models.sh` fetches it.

# LiteRT Image Segmentation with DIS Model Demo

### Overview

This is an app that allows you to take a picture or open an image from the gallery in order to segment out the main objects within the image. 

This sample should be run on a physical Android device.

This sample requires a model converted from PyTorch to the LiteRT format. This has been tested through [this online sample]([url](https://github.com/google-ai-edge/models-samples/blob/main/convert_pytorch/DIS_segmentation_and_quantization.ipynb)).

## Build the demo using Android Studio

### Prerequisites

* The **[Android Studio](https://developer.android.com/studio/index.html)**
  IDE. This sample has been tested on Android Studio Dolphin.

* A physical Android device with a minimum OS version of SDK 24 (Android 7.0 -
  Nougat) with developer mode enabled. The process of enabling developer mode
  may vary by device. You may also use an Android emulator with more limited
  functionality.

### Building

* Open Android Studio. From the Welcome screen, select Open an existing
  Android Studio project.

* From the Open File or Project window that appears, navigate to and select
  the litert-samples/examples/image_segmentation_DIS/android directory. Click OK. You may
  be asked if you trust the project. Select Trust.

* Copy the model file created by the PyTorch conversion script into the _assets_ directory and ensure the name matches the file name used in the _helper_ class.

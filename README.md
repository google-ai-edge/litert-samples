# **Google AI Edge LiteRT Samples**

This repository contains official sample applications and code examples for **LiteRT** (formerly known as TensorFlow Lite), Google's high-performance on-device machine learning framework.

The samples are organized into two main versions to demonstrate different API paradigms:

* **`v1/`**: Samples using the standard **Interpreter API** (Classic TensorFlow Lite).  
* **`v2/`**: Samples using the new **LiteRT CompiledModel API** (Optimized for Accelerators/NPU).

**Note:** For Generative AI and Large Language Models (LLMs), please refer to the [LiteRT-LM repository](https://github.com/google-ai-edge/LiteRT-LM).

## **📂 Repository Structure**

### **1\. `v1/` (Standard Interpreter API)**

This folder contains the classic samples that use the `Interpreter` class. These are best for general-purpose inference on CPU or standard GPU delegates.

* **Key Features:**  
  * Standard `.tflite` model execution.  
  * Broad compatibility across all Android/iOS versions.  
  * Legacy Task Library usage.  
* **Available Samples:**  
  * **Image Classification**: Recognize objects in images/video.  
  * **Object Detection**: Locate and label multiple objects.  
  * **Image Segmentation**: Separate objects from the background.  
  * **Audio Classification**: Identify audio events.  
  * **Digit Classification**: Handwritten digit recognition (MNIST).  
* **Platforms:** Android (Kotlin/Java), iOS (Swift/Objective-C), Python (Raspberry Pi/Linux).

### **2\. `v2/` (CompiledModel API)**

This folder contains samples using the **LiteRT CompiledModel API**. This new API is designed to maximize performance by compiling models ahead-of-time (AOT) or just-in-time (JIT) for specific hardware accelerators, particularly **NPUs** (Neural Processing Units).

* **Key Features:**  
  * **Hardware Acceleration**: Specialized for NPU execution.  
  * **Async Execution**: Improved performance for complex pipelines.  
  * **Buffer Management**: efficient input/output handling.  
* **Available Samples:**  
  * **NPU AOT**: Ahead-of-Time compilation examples.  
  * **NPU JIT**: Just-in-Time compilation examples.  
* **Platforms:** Primarily Android (Kotlin/C++).

## **🛠️ Getting Started**

### **Prerequisites**

* **Android**: Android Studio (latest stable version).  
* **iOS**: Xcode (latest version).  
* **Python**: Python 3.9+ and `pip install ai-edge-litert`.

### **Running a Sample**

#### **For V1 Samples (Standard)**

1. Navigate to `v1/` directory.  
2. Open the project in Android Studio or Xcode.  
3. Build and run on your device.

#### **For V2 Samples (High Performance / NPU)**

1. Navigate to the `v2/` directory.  
2. Ensure you have a device with a supported NPU (e.g., modern Pixel, Samsung, or devices with MediaTek/Qualcomm chips).  
3. Follow the specific setup instructions in the sub-folder to enable the specialized hardware delegates.

## **📚 Documentation**

* **LiteRT Overview**: [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)  
* **CompiledModel API Guide**: [LiteRT for Android](https://ai.google.dev/edge/litert/android)  
* **Model Conversion**: [Convert models to LiteRT](https://ai.google.dev/edge/litert/models/convert)

## **🤝 Contributing**

Contributions are welcome\!

1. Read [CONTRIBUTING.md](https://www.google.com/search?q=CONTRIBUTING.md).  
2. Fork the repo and create a branch.  
3. Submit a Pull Request.

## **📄 License**

Apache License 2.0. See [LICENSE](https://www.google.com/search?q=LICENSE) for details.

---

*Disclaimer: This is a sample repository maintained by Google. It is provided "as is" without warranty of any kind.*

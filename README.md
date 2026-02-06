# **Google AI Edge LiteRT Samples**

This repository contains official sample applications and code examples for **LiteRT** (formerly known as TensorFlow Lite), Google's high-performance on-device machine learning framework.

The samples are organized into two main versions (`interpreter_api/` and `compiled_model_api/`) to demonstrate different API paradigms.

**Note:** For Generative AI and Large Language Models (LLMs), please refer to the [LiteRT-LM repository](https://github.com/google-ai-edge/LiteRT-LM).

## **📂 Repository Structure**

### **1\. `compiled_model_api/`**

This folder contains samples using the **LiteRT CompiledModel API**. This new API is designed for advanced GPU/NPU acceleration, delivering superior ML & GenAI performance.

* **Key Features:**  
  * **Hardware Acceleration**: Specialized for GPU/NPU execution.  
  * **Async Execution**: Improved performance for complex pipelines.  
  * **Buffer Management**: efficient input/output handling.  
* **Available Samples:**  
  * **NPU AOT**: Ahead-of-Time compilation examples.  
  * **NPU JIT**: Just-in-Time compilation examples.  
* **Platforms:** Primarily Android (Kotlin/C++).

### **2\. `interpreter_api/`**

This folder contains the CPU samples that use the **Interpreter API**.

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

## **🛠️ Getting Started**

### **Prerequisites**

* **Android**: Android Studio (latest stable version).  
* **iOS**: Xcode (latest version).  
* **Python**: Python 3.9+ and `pip install ai-edge-litert`.

### **Running a Sample**

#### **For Samples Using Compiled Model API**

1. Navigate to the `compiled_model_api/` directory.  
2. Ensure you have a device with a supported NPU (e.g., modern Pixel, Samsung, or devices with MediaTek/Qualcomm chips).  
3. Follow the specific setup instructions in the sub-folder to enable the specialized hardware delegates.

#### **For Samples Using Interpreter API**

1. Navigate to `interpreter_api/` directory.  
2. Open the project in Android Studio or Xcode.  
3. Build and run on your device.

## **📚 Documentation**

* **LiteRT Overview**: [ai.google.dev/edge/litert](https://ai.google.dev/edge/litert)  
* **CompiledModel API Guide**: [LiteRT for Android](https://ai.google.dev/edge/litert/android)  
* **Model Conversion**: [Convert models to LiteRT](https://ai.google.dev/edge/litert/conversion/overview)

## **🤝 Contributing**

Contributions are welcome\!

1. Read [CONTRIBUTING.md](https://github.com/google-ai-edge/litert-samples/blob/main/CONTRIBUTING.md).  
2. Fork the repo and create a branch.  
3. Submit a Pull Request.

## **📄 License**

Apache License 2.0. See [LICENSE](https://github.com/google-ai-edge/litert-samples/blob/main/LICENSE) for details.

---

*Disclaimer: This is a sample repository maintained by Google. It is provided "as is" without warranty of any kind.*

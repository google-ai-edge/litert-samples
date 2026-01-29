# LiteRT Digit Classifier Sample

This directory contains an Android sample demonstrating how to use LiteRT (Google's new runtime for TensorFlow Lite) to classify handwritten digits.

## Overview

The Digit Classifier sample allows users to draw digits (0-9) on the screen and uses a machine learning model to classify the drawn digit in real-time. It demonstrates the use of the LiteRT Compiled Model API with support for hardware acceleration.

## Available Implementations

### kotlin_cpu_gpu

A standard implementation utilizing the Compiled Model API with support for CPU and GPU delegates.

**Features:**
-   **Canvas Drawing**: Interactive canvas to draw digits.
-   **Hardware Acceleration**: Switch between CPU and GPU delegates at runtime.
-   **Real-time Inference**: Classifies the drawing as you draw (or upon updates).
-   **Jetpack Compose**: Modern Android UI toolkit.

### Screenshots

| CPU Execution | GPU Execution |
| :---: | :---: |
| ![CPU](../../CPU.jpg) | ![GPU](../../GPU.jpg) |

## Technical Details

### Model Architecture
-   **Task**: Single digit classification (MNIST-style).
-   **Input**: 28x28x3 RGB image (preprocessed from drawing canvas).
-   **Output**: Probability score for 10 classes (digits 0-9).
-   **Model format**: TensorFlow Lite (`.tflite`).

### Key Dependencies
-   **LiteRT** (`com.google.ai.edge.litert`)
-   **Jetpack Compose** (UI)
-   **Kotlin Coroutines** (Async operations)

### Architecture Components
-   **`DigitClassificationHelper`**: Handles model initialization, hardware delegate selection (CPU/GPU), and inference execution.
-   **`MainActivity`**: Setup of the main screen and UI components.
-   **`Board` Composable**: Handles touch input for drawing and renders the strokes.
-   **`MainViewModel`**: Manages UI state and communicates between the UI and the Helper.

## Getting Started

1.  Open `kotlin_cpu_gpu/android` in Android Studio.
2.  Build and run the application on an Android device.
3.  Draw a digit on the white board area.
4.  Observe the prediction and confidence score.
5.  Use the toggle switch to change between CPU and GPU acceleration.

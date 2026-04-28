package com.example.efficientdet_lite

import android.graphics.RectF

data class Detection(
    val box: RectF,
    val label: String,
    val confidence: Float,
)

data class FrameMetadata(
    val sourceWidth: Int,
    val sourceHeight: Int,
    val inputSize: Int,
    val scale: Float,
    val padX: Float,
    val padY: Float,
)

data class PreprocessedFrame(
    val input: ByteArray,
    val metadata: FrameMetadata,
)

data class EfficientDetFrameResult(
    val detections: List<Detection> = emptyList(),
    val frameWidth: Int = 0,
    val frameHeight: Int = 0,
    val backend: String = "",
)

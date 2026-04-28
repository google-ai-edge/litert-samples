package com.example.efficientdet_lite

import android.graphics.RectF
import android.util.Log
import kotlin.math.roundToInt

class EfficientDetPostprocessor(
    private val labels: List<String>,
    private val confidenceThreshold: Float = 0.35f,
) {
    fun process(outputs: List<FloatArray>, metadata: FrameMetadata): List<Detection> {
        if (outputs.size < 4) {
            Log.w(TAG, "Expected EfficientDet outputs boxes/classes/scores/count, got ${outputs.size}")
            return emptyList()
        }

        val count = outputs.firstOrNull { it.size == 1 }?.firstOrNull()?.roundToInt()
            ?: outputs.filter { it.size > 1 }.minOf { it.size }
        val boxes = outputs.firstOrNull { it.size >= count * 4 && looksLikeBoxes(it, count) }
            ?: outputs[0]
        val vectors = outputs.filter { it !== boxes && it.size >= count }
        val scores = vectors.firstOrNull { looksLikeScores(it, count) } ?: outputs.getOrNull(2) ?: return emptyList()
        val classes = vectors.firstOrNull { it !== scores } ?: outputs.getOrNull(1) ?: return emptyList()
        val detectionCount = minOf(count, boxes.size / 4, scores.size, classes.size)

        Log.d(
            TAG,
            "Postprocess tensors: count=$detectionCount boxes=${boxes.size} classes=${classes.size} scores=${scores.size}",
        )

        val detections = ArrayList<Detection>()
        for (index in 0 until detectionCount) {
            val score = scores[index]
            if (score < confidenceThreshold) continue

            val boxOffset = index * 4
            val rawYmin = boxes[boxOffset]
            val rawXmin = boxes[boxOffset + 1]
            val rawYmax = boxes[boxOffset + 2]
            val rawXmax = boxes[boxOffset + 3]
            val boxesAreNormalized = rawYmin <= 1.5f && rawXmin <= 1.5f && rawYmax <= 1.5f && rawXmax <= 1.5f

            val inputLeft: Float
            val inputTop: Float
            val inputRight: Float
            val inputBottom: Float
            if (boxesAreNormalized) {
                inputLeft = rawXmin.coerceIn(0f, 1f) * metadata.inputSize
                inputTop = rawYmin.coerceIn(0f, 1f) * metadata.inputSize
                inputRight = rawXmax.coerceIn(0f, 1f) * metadata.inputSize
                inputBottom = rawYmax.coerceIn(0f, 1f) * metadata.inputSize
            } else {
                inputLeft = rawXmin.coerceIn(0f, metadata.inputSize.toFloat())
                inputTop = rawYmin.coerceIn(0f, metadata.inputSize.toFloat())
                inputRight = rawXmax.coerceIn(0f, metadata.inputSize.toFloat())
                inputBottom = rawYmax.coerceIn(0f, metadata.inputSize.toFloat())
            }

            val left = ((inputLeft - metadata.padX) / metadata.scale).coerceIn(0f, metadata.sourceWidth.toFloat())
            val top = ((inputTop - metadata.padY) / metadata.scale).coerceIn(0f, metadata.sourceHeight.toFloat())
            val right = ((inputRight - metadata.padX) / metadata.scale).coerceIn(0f, metadata.sourceWidth.toFloat())
            val bottom = ((inputBottom - metadata.padY) / metadata.scale).coerceIn(0f, metadata.sourceHeight.toFloat())
            if (right <= left || bottom <= top) continue

            val classIndex = classes[index].roundToInt()
            val label = labels.getOrElse(classIndex) { "class $classIndex" }
            if (label == "N/A") continue
            detections += Detection(
                box = RectF(left, top, right, bottom),
                label = label,
                confidence = score,
            )
        }
        return detections
    }

    private fun looksLikeBoxes(values: FloatArray, count: Int): Boolean {
        if (values.size < count * 4) return false
        val sampleCount = minOf(values.size, count * 4, 200)
        return values.take(sampleCount).count { it in -0.05f..1.05f } >= sampleCount * 0.9f
    }

    private fun looksLikeScores(values: FloatArray, count: Int): Boolean {
        if (values.size < count) return false
        val sampleCount = minOf(values.size, count, 100)
        return values.take(sampleCount).all { it in 0f..1f } &&
            values.take(sampleCount).any { it > 0f && it < 1f }
    }

    private companion object {
        const val TAG = "EfficientDetPostprocessor"
    }
}

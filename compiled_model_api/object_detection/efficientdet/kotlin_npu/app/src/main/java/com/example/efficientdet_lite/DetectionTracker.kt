package com.example.efficientdet_lite

import android.graphics.RectF
import kotlin.math.max
import kotlin.math.min

class DetectionTracker(
    private val iouThreshold: Float = 0.3f,
    private val smoothingAlpha: Float = 0.5f,
    private val minHitsToConfirm: Int = 2,
    private val maxMissedFrames: Int = 4,
) {
    private data class TrackedItem(
        val label: String,
        val confidence: Float,
        val box: RectF,
        val hits: Int,
        val missedFrames: Int,
    )

    private var items: List<TrackedItem> = emptyList()

    fun reset() {
        items = emptyList()
    }

    fun update(detections: List<Detection>): List<Detection> {
        val matched = BooleanArray(detections.size)
        val updated = mutableListOf<TrackedItem>()

        for (item in items) {
            var bestIou = iouThreshold
            var bestIndex = -1
            for (index in detections.indices) {
                if (matched[index] || detections[index].label != item.label) continue
                val iou = iou(item.box, detections[index].box)
                if (iou > bestIou) {
                    bestIou = iou
                    bestIndex = index
                }
            }
            if (bestIndex >= 0) {
                matched[bestIndex] = true
                val detection = detections[bestIndex]
                updated += item.copy(
                    confidence = detection.confidence,
                    box = lerp(item.box, detection.box, smoothingAlpha),
                    hits = item.hits + 1,
                    missedFrames = 0,
                )
            } else if (item.missedFrames < maxMissedFrames) {
                updated += item.copy(missedFrames = item.missedFrames + 1)
            }
        }

        for (index in detections.indices) {
            if (!matched[index]) {
                val detection = detections[index]
                updated += TrackedItem(
                    label = detection.label,
                    confidence = detection.confidence,
                    box = RectF(detection.box),
                    hits = 1,
                    missedFrames = 0,
                )
            }
        }

        items = updated
        return updated
            .filter { it.hits >= minHitsToConfirm }
            .map { Detection(box = RectF(it.box), label = it.label, confidence = it.confidence) }
    }

    private fun lerp(from: RectF, to: RectF, alpha: Float) = RectF(
        from.left + (to.left - from.left) * alpha,
        from.top + (to.top - from.top) * alpha,
        from.right + (to.right - from.right) * alpha,
        from.bottom + (to.bottom - from.bottom) * alpha,
    )

    private fun iou(a: RectF, b: RectF): Float {
        val interLeft = max(a.left, b.left)
        val interTop = max(a.top, b.top)
        val interRight = min(a.right, b.right)
        val interBottom = min(a.bottom, b.bottom)
        val interArea = max(0f, interRight - interLeft) * max(0f, interBottom - interTop)
        if (interArea == 0f) return 0f
        val aArea = a.width() * a.height()
        val bArea = b.width() * b.height()
        return interArea / (aArea + bArea - interArea)
    }
}

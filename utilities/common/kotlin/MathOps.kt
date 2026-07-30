/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package __MODULE_PACKAGE__
// Vendored from common/kotlin/MathOps.kt — edit the canonical and run tools/sync_common.py --apply.

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min

/** Small post-processing math shared across model heads. */
object MathOps {

    private const val IOU_EPS = 1e-6f

    /** Numerically stable sigmoid. */
    fun sigmoid(x: Float): Float = 1f / (1f + exp(-x))

    /** In-place, max-subtracted softmax over [logits]. Returns the same array. */
    fun softmaxInPlace(logits: FloatArray): FloatArray {
        var maxLogit = Float.NEGATIVE_INFINITY
        for (v in logits) {
            maxLogit = max(maxLogit, v)
        }
        var sum = 0f
        for (i in logits.indices) {
            logits[i] = exp(logits[i] - maxLogit)
            sum += logits[i]
        }
        for (i in logits.indices) {
            logits[i] /= sum
        }
        return logits
    }

    /** Index of the largest value, or -1 for an empty array. */
    fun argmax(values: FloatArray): Int {
        var best = -1
        var bestValue = Float.NEGATIVE_INFINITY
        for (i in values.indices) {
            if (values[i] > bestValue) {
                bestValue = values[i]
                best = i
            }
        }
        return best
    }

    /** Intersection-over-union of two xyxy boxes. */
    fun iou(
        ax1: Float, ay1: Float, ax2: Float, ay2: Float,
        bx1: Float, by1: Float, bx2: Float, by2: Float,
    ): Float {
        val interWidth = max(0f, min(ax2, bx2) - max(ax1, bx1))
        val interHeight = max(0f, min(ay2, by2) - max(ay1, by1))
        val inter = interWidth * interHeight
        val areaA = (ax2 - ax1) * (ay2 - ay1)
        val areaB = (bx2 - bx1) * (by2 - by1)
        return inter / (areaA + areaB - inter + IOU_EPS)
    }

    /**
     * Greedy non-maximum suppression.
     *
     * @param boxes flat xyxy boxes, `boxes[4 * i .. 4 * i + 3]` for detection i
     * @param scores one score per detection
     * @return indices of the kept detections, sorted by descending score
     */
    fun nms(
        boxes: FloatArray,
        scores: FloatArray,
        iouThreshold: Float,
        scoreThreshold: Float = 0f,
    ): IntArray {
        val order = scores.indices
            .filter { scores[it] >= scoreThreshold }
            .sortedByDescending { scores[it] }
        val kept = mutableListOf<Int>()
        val suppressed = BooleanArray(scores.size)
        for (position in order.indices) {
            val i = order[position]
            if (suppressed[i]) {
                continue
            }
            kept.add(i)
            for (lower in position + 1 until order.size) {
                val j = order[lower]
                if (suppressed[j]) {
                    continue
                }
                val overlap = iou(
                    boxes[4 * i], boxes[4 * i + 1], boxes[4 * i + 2], boxes[4 * i + 3],
                    boxes[4 * j], boxes[4 * j + 1], boxes[4 * j + 2], boxes[4 * j + 3],
                )
                if (overlap > iouThreshold) {
                    suppressed[j] = true
                }
            }
        }
        return kept.toIntArray()
    }
}

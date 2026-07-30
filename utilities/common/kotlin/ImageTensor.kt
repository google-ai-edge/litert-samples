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
// Vendored from common/kotlin/ImageTensor.kt — edit the canonical and run tools/sync_common.py --apply.

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint

/**
 * Bitmap → normalized float tensor conversion with zero per-frame allocation.
 *
 * Configure once with the model's input geometry and normalization, then call [load]
 * per frame. The returned array is owned by this instance and overwritten by the next
 * call. Model-specific parameters (mean/std, layout, channel order) are constructor
 * arguments — do not edit a vendored copy to change them.
 *
 * Normalization applied per channel c: `value = (pixel[c] / 255 - mean[c]) / std[c]`
 * (skip the `/ 255` with `scaleTo01 = false` for raw 0–255 models such as YOLOX).
 */
class ImageTensor(
    private val width: Int,
    private val height: Int,
    private val mean: FloatArray = floatArrayOf(0f, 0f, 0f),
    private val std: FloatArray = floatArrayOf(1f, 1f, 1f),
    private val layout: Layout = Layout.NCHW,
    private val channelOrder: ChannelOrder = ChannelOrder.RGB,
    private val scaleTo01: Boolean = true,
) {
    /** Tensor memory layout: channels-first (PyTorch exports) or channels-last. */
    enum class Layout { NCHW, NHWC }

    /** Channel order expected by the model ([mean]/[std] are indexed in this order). */
    enum class ChannelOrder { RGB, BGR }

    /** Aspect-ratio policy applied by [load]. */
    enum class Fit { STRETCH, LETTERBOX }

    /**
     * Geometry of the last [load]: `model_x = source_x * scaleX + padX` (and same for y).
     * Use it to map detections back to source-image coordinates.
     */
    data class Mapping(val scaleX: Float, val scaleY: Float, val padX: Float, val padY: Float)

    companion object {
        val IMAGENET_MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        val IMAGENET_STD = floatArrayOf(0.229f, 0.224f, 0.225f)
    }

    /** Destination tensor, reused across [load] calls. */
    val floats = FloatArray(3 * width * height)

    private val pixels = IntArray(width * height)
    private val scaled = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(scaled)
    private val matrix = Matrix()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    /** Geometry of the last [load]. */
    var mapping = Mapping(1f, 1f, 0f, 0f)
        private set

    /**
     * Converts [bitmap] into the configured float layout and returns [floats].
     *
     * @param padColor letterbox padding color (e.g. `0xFF727272.toInt()` for YOLOX gray 114)
     */
    fun load(bitmap: Bitmap, fit: Fit = Fit.STRETCH, padColor: Int = Color.BLACK): FloatArray {
        when (fit) {
            Fit.STRETCH -> {
                val scaleX = width.toFloat() / bitmap.width
                val scaleY = height.toFloat() / bitmap.height
                matrix.setScale(scaleX, scaleY)
                mapping = Mapping(scaleX, scaleY, 0f, 0f)
            }
            Fit.LETTERBOX -> {
                val scale = minOf(width.toFloat() / bitmap.width, height.toFloat() / bitmap.height)
                val padX = (width - bitmap.width * scale) / 2f
                val padY = (height - bitmap.height * scale) / 2f
                canvas.drawColor(padColor)
                matrix.setScale(scale, scale)
                matrix.postTranslate(padX, padY)
                mapping = Mapping(scale, scale, padX, padY)
            }
        }
        canvas.drawBitmap(bitmap, matrix, paint)
        scaled.getPixels(pixels, 0, width, 0, 0, width, height)

        val plane = width * height
        for (i in 0 until plane) {
            val p = pixels[i]
            var c0 = ((p shr 16) and 0xFF).toFloat()
            var c1 = ((p shr 8) and 0xFF).toFloat()
            var c2 = (p and 0xFF).toFloat()
            if (channelOrder == ChannelOrder.BGR) {
                val swap = c0
                c0 = c2
                c2 = swap
            }
            if (scaleTo01) {
                c0 /= 255f
                c1 /= 255f
                c2 /= 255f
            }
            c0 = (c0 - mean[0]) / std[0]
            c1 = (c1 - mean[1]) / std[1]
            c2 = (c2 - mean[2]) / std[2]
            if (layout == Layout.NCHW) {
                floats[i] = c0
                floats[plane + i] = c1
                floats[2 * plane + i] = c2
            } else {
                val j = i * 3
                floats[j] = c0
                floats[j + 1] = c1
                floats[j + 2] = c2
            }
        }
        return floats
    }

    /** Releases the internal scratch bitmap. */
    fun release() {
        if (!scaled.isRecycled) {
            scaled.recycle()
        }
    }
}

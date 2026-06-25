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

package com.google.aiedge.examples.backgroundremoval

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.os.SystemClock
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

/**
 * Runs salient-object / background removal with U^2-Net on the LiteRT Compiled Model API.
 *
 * U^2-Net (a nested U-structure, pure CNN) is converted with litert-torch and runs entirely on the
 * CPU or GPU delegate. It maps an image to a single-channel saliency mask in [0, 1]; the foreground
 * is composited onto a transparent background to produce the cut-out result.
 */
class BackgroundRemovalHelper(
    private val context: Context,
    private var options: Options = Options(),
) {
    class Options(
        /** Model variant to run. */
        var model: Model = DEFAULT_MODEL,
        /** The delegate for running computationally intensive operations. */
        var delegate: AcceleratorEnum = DEFAULT_DELEGATE,
    )

    companion object {
        private const val TAG = "BackgroundRemoval"

        private const val INPUT_SIZE = 320

        // U^2-Net normalization: pixel/255, divide by the per-image max, then ImageNet mean/std.
        private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
        private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

        val DEFAULT_MODEL = Model.U2Net
        val DEFAULT_DELEGATE = AcceleratorEnum.GPU

        fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator {
            return when (acceleratorEnum) {
                AcceleratorEnum.CPU -> Accelerator.CPU
                AcceleratorEnum.GPU -> Accelerator.GPU
            }
        }
    }

    val segmentation: SharedFlow<SegmentationResult>
        get() = _segmentation
    private val _segmentation = MutableSharedFlow<SegmentationResult>(
        extraBufferCapacity = 64, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    private var model: CompiledModel? = null
    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

    // Reusable preprocessing buffers (allocated once).
    private val inputFloats = FloatArray(3 * INPUT_SIZE * INPUT_SIZE)
    private val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)
    private val squareBitmap = Bitmap.createBitmap(INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
    private val resizeMatrix = Matrix()
    private val resizePaint = Paint(Paint.FILTER_BITMAP_FLAG)

    /** Create a [CompiledModel] for the selected [Model] on the selected [AcceleratorEnum]. */
    suspend fun initSegmenter() {
        cleanup()
        try {
            withContext(singleThreadDispatcher) {
                model = CompiledModel.create(
                    context.assets,
                    options.model.fileName,
                    CompiledModel.Options(toAccelerator(options.delegate)),
                    null
                )
                Log.i(TAG, "Created CompiledModel from ${options.model.fileName} on ${options.delegate}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Failed to create CompiledModel from ${options.model.fileName}: ${e.message}")
            _error.emit(e)
        }
    }

    suspend fun cleanup() {
        withContext(singleThreadDispatcher) {
            model?.close()
            model = null
        }
    }

    fun setOptions(options: Options) {
        this.options = options
    }

    /** Segment [bitmap], composite the foreground onto transparency, and emit the cut-out. */
    suspend fun segment(bitmap: Bitmap, rotationDegrees: Int) {
        try {
            withContext(singleThreadDispatcher) {
                val currentModel = model ?: return@withContext
                val startTime = SystemClock.uptimeMillis()

                val upright = if (rotationDegrees % 360 != 0) rotate(bitmap, rotationDegrees) else bitmap
                preprocess(upright)

                val inputBuffers = currentModel.createInputBuffers()
                val outputBuffers = currentModel.createOutputBuffers()
                inputBuffers[0].writeFloat(inputFloats)
                currentModel.run(inputBuffers, outputBuffers)
                val maskRaw = outputBuffers[0].readFloat() // [INPUT_SIZE*INPUT_SIZE], sigmoid [0,1]
                inputBuffers.forEach { it.close() }
                outputBuffers.forEach { it.close() }

                val cutout = compositeCutout(upright, maskRaw)
                val inferenceTime = SystemClock.uptimeMillis() - startTime
                if (isActive) {
                    _segmentation.emit(SegmentationResult(cutout, inferenceTime))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Background removal error: ${e.message}")
            _error.emit(e)
        }
    }

    /** Resize to 320x320 and write planar NCHW floats (/max then ImageNet normalize). */
    private fun preprocess(bitmap: Bitmap) {
        resizeMatrix.setScale(
            INPUT_SIZE.toFloat() / bitmap.width,
            INPUT_SIZE.toFloat() / bitmap.height
        )
        val canvas = Canvas(squareBitmap)
        canvas.drawColor(Color.BLACK)
        canvas.drawBitmap(bitmap, resizeMatrix, resizePaint)
        squareBitmap.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)

        val plane = INPUT_SIZE * INPUT_SIZE
        var maxV = 1e-6f
        val r = FloatArray(plane); val g = FloatArray(plane); val b = FloatArray(plane)
        for (i in pixels.indices) {
            val p = pixels[i]
            val rv = ((p shr 16) and 0xFF) / 255f
            val gv = ((p shr 8) and 0xFF) / 255f
            val bv = (p and 0xFF) / 255f
            r[i] = rv; g[i] = gv; b[i] = bv
            if (rv > maxV) maxV = rv; if (gv > maxV) maxV = gv; if (bv > maxV) maxV = bv
        }
        for (i in 0 until plane) {
            inputFloats[i] = (r[i] / maxV - MEAN[0]) / STD[0]
            inputFloats[plane + i] = (g[i] / maxV - MEAN[1]) / STD[1]
            inputFloats[2 * plane + i] = (b[i] / maxV - MEAN[2]) / STD[2]
        }
    }

    /** Foreground (original RGB) over transparency, using the upscaled mask as alpha. */
    private fun compositeCutout(bitmap: Bitmap, maskRaw: FloatArray): Bitmap {
        // Normalize mask to [0,1].
        var lo = Float.MAX_VALUE; var hi = -Float.MAX_VALUE
        for (v in maskRaw) { if (v < lo) lo = v; if (v > hi) hi = v }
        val range = if (hi > lo) hi - lo else 1f

        val maskArgb = IntArray(INPUT_SIZE * INPUT_SIZE)
        for (i in maskArgb.indices) {
            val a = (((maskRaw[i] - lo) / range) * 255f).toInt().coerceIn(0, 255)
            maskArgb[i] = a shl 24
        }
        val small = Bitmap.createBitmap(maskArgb, INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val ow = bitmap.width; val oh = bitmap.height
        val up = Bitmap.createScaledBitmap(small, ow, oh, true)
        small.recycle()

        val maskPix = IntArray(ow * oh); up.getPixels(maskPix, 0, ow, 0, 0, ow, oh); up.recycle()
        val srcPix = IntArray(ow * oh); bitmap.getPixels(srcPix, 0, ow, 0, 0, ow, oh)
        val out = IntArray(ow * oh)
        for (i in out.indices) {
            val alpha = (maskPix[i] ushr 24) and 0xFF
            out[i] = (alpha shl 24) or (srcPix[i] and 0x00FFFFFF)
        }
        return Bitmap.createBitmap(out, ow, oh, Bitmap.Config.ARGB_8888)
    }

    private fun rotate(bitmap: Bitmap, degrees: Int): Bitmap {
        val m = Matrix().apply { postRotate(degrees.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, m, true)
    }

    enum class AcceleratorEnum {
        CPU, GPU
    }

    enum class Model(val fileName: String) {
        U2Net("u2net_fp16.tflite")
    }

    data class SegmentationResult(
        val cutout: Bitmap?, val inferenceTime: Long
    )
}

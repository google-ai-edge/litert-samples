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

package com.google.aiedge.examples.portraitmatting

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
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
 * Runs trimap-free portrait matting with MODNet on the LiteRT Compiled Model API.
 *
 * MODNet (Ke et al., AAAI 2022) is a pure CNN (MobileNetV2 backbone + conv decoder, no attention),
 * converted with litert-torch and running entirely on the CPU or GPU delegate. It maps an RGB image
 * to a single-channel soft alpha matte in [0, 1] (1 = person, 0 = background); the foreground is
 * composited onto a transparent background using that matte as alpha, preserving hair-level detail.
 *
 * The [0,1] -> [-1,1] input normalization is baked into the graph, so the app feeds a plain [0,1]
 * RGB image (NHWC, interleaved) and reads the final matte directly.
 */
class PortraitMattingHelper(
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
        private const val TAG = "PortraitMatting"

        private const val INPUT_SIZE = 512

        // Backdrop the cut-out is composited onto (matches the UI's Box background). The compositing
        // is done here in code into an OPAQUE bitmap, so the soft alpha matte is blended exactly
        // once — drawing a transparent bitmap and letting Compose alpha-composite it over the
        // background instead produces a dark fringe at the hair boundary (premultiplied-alpha /
        // resample mismatch).
        private const val BACKDROP_R = 0xE6
        private const val BACKDROP_G = 0xE6
        private const val BACKDROP_B = 0xE6

        val DEFAULT_MODEL = Model.Modnet
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

    // Reusable preprocessing buffers (allocated once). Input is NHWC interleaved [0,1] RGB.
    private val inputFloats = FloatArray(INPUT_SIZE * INPUT_SIZE * 3)
    private val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)

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

    /** Matte [bitmap], composite the foreground onto transparency, and emit the cut-out. */
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
                val matteRaw = outputBuffers[0].readFloat() // [INPUT_SIZE*INPUT_SIZE], alpha [0,1]
                inputBuffers.forEach { it.close() }
                outputBuffers.forEach { it.close() }

                val cutout = compositeCutout(upright, matteRaw)
                val inferenceTime = SystemClock.uptimeMillis() - startTime
                if (isActive) {
                    _segmentation.emit(SegmentationResult(cutout, inferenceTime))
                }
            }
        } catch (e: Exception) {
            Log.e(TAG, "Portrait matting error: ${e.message}")
            _error.emit(e)
        }
    }

    /** Resize to 512x512 and write NHWC interleaved [0,1] RGB floats (normalization is in-graph). */
    private fun preprocess(bitmap: Bitmap) {
        // Antialiased downscale to 512x512. Android's bilinear sampling does NOT area-average, so a
        // single large downscale (e.g. a 4000px photo straight to 512) aliases — and the noisy input
        // makes MODNet emit a corrupted matte (a dark ring / speckle hugging the hair). Strategy:
        //   1) halve (each 2x step box-averages 4 px) until <= 1024,
        //   2) scale to 1024x1024 (an upscale or a <=2x step — no aliasing),
        //   3) one final 2x step to 512x512, which IS an exact 2x box average.
        // That final box step gives a clean, alias-free input for ANY source size (a plain
        // source->512 bilinear leaves visible boundary artifacts even at ~1.4x).
        var work = bitmap
        var scaled = false
        while (work.width > INPUT_SIZE * 2 && work.height > INPUT_SIZE * 2) {
            val half = Bitmap.createScaledBitmap(work, work.width / 2, work.height / 2, true)
            if (scaled) work.recycle()
            work = half
            scaled = true
        }
        val big = Bitmap.createScaledBitmap(work, INPUT_SIZE * 2, INPUT_SIZE * 2, true)
        if (scaled) work.recycle()
        val square = Bitmap.createScaledBitmap(big, INPUT_SIZE, INPUT_SIZE, true)
        big.recycle()
        square.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE)
        square.recycle()

        for (i in pixels.indices) {
            val p = pixels[i]
            inputFloats[i * 3] = ((p shr 16) and 0xFF) / 255f
            inputFloats[i * 3 + 1] = ((p shr 8) and 0xFF) / 255f
            inputFloats[i * 3 + 2] = (p and 0xFF) / 255f
        }
    }

    /**
     * Composite the foreground (original RGB) over the [BACKDROP] using the upscaled soft matte as
     * alpha, into an OPAQUE bitmap: `out = fg*a + bg*(1-a)`. MODNet's true [0,1] matte is used
     * directly (no min-max stretch) to keep the soft hair edges. Doing the blend here — instead of
     * returning a transparent bitmap for Compose to alpha-composite — is what keeps the boundary
     * clean (the alpha-rendered path leaves a dark fringe around the hair).
     */
    private fun compositeCutout(bitmap: Bitmap, matteRaw: FloatArray): Bitmap {
        val matteArgb = IntArray(INPUT_SIZE * INPUT_SIZE)
        for (i in matteArgb.indices) {
            val a = (matteRaw[i] * 255f).toInt().coerceIn(0, 255)
            matteArgb[i] = a shl 24
        }
        val small = Bitmap.createBitmap(matteArgb, INPUT_SIZE, INPUT_SIZE, Bitmap.Config.ARGB_8888)
        val ow = bitmap.width; val oh = bitmap.height
        val up = Bitmap.createScaledBitmap(small, ow, oh, true)
        small.recycle()

        val mattePix = IntArray(ow * oh); up.getPixels(mattePix, 0, ow, 0, 0, ow, oh); up.recycle()
        val srcPix = IntArray(ow * oh); bitmap.getPixels(srcPix, 0, ow, 0, 0, ow, oh)
        val out = IntArray(ow * oh)
        for (i in out.indices) {
            val a = ((mattePix[i] ushr 24) and 0xFF) / 255f
            val s = srcPix[i]
            val r = (((s shr 16) and 0xFF) * a + BACKDROP_R * (1f - a)).toInt().coerceIn(0, 255)
            val g = (((s shr 8) and 0xFF) * a + BACKDROP_G * (1f - a)).toInt().coerceIn(0, 255)
            val b = ((s and 0xFF) * a + BACKDROP_B * (1f - a)).toInt().coerceIn(0, 255)
            out[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
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
        Modnet("modnet_512_fp16.tflite")
    }

    data class SegmentationResult(
        val cutout: Bitmap?, val inferenceTime: Long
    )
}

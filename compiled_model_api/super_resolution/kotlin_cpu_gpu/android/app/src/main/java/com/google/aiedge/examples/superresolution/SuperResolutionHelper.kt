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

package com.google.aiedge.examples.superresolution

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
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
 * Real-ESRGAN ×4 super-resolution on LiteRT CompiledModel (CPU / GPU).
 *
 * Model = Real-ESRGAN realesr-general-x4v3 (SRVGGNetCompact, BSD-3) re-authored GPU-clean via
 * litert_torch: PReLU -> relu(x)-a*relu(-x), PixelShuffle -> one-hot ConvTranspose -> ZeroStuffConvT.
 * Full LITERT_CL residency on the Pixel 8a (211/211 nodes, ~1 ms).
 *
 *   input : images [1, 128, 128, 3]  NHWC, RGB, 0-1 float (no normalization)
 *   output: [1, 512, 512, 3]         NHWC, RGB, 0-1 float (clamp), ×4 upscale
 *
 * Larger images are tiled into 128×128 patches; this sample resizes the picked image to one tile.
 */
class SuperResolutionHelper(
    private val context: Context,
    private var options: Options = Options(),
) {
    class Options(
        var model: Model = DEFAULT_MODEL,
        var delegate: AcceleratorEnum = DEFAULT_DELEGATE,
    )

    companion object {
        private const val TAG = "SuperResolution"
        val DEFAULT_MODEL = Model.RealEsrganX4v3
        val DEFAULT_DELEGATE = AcceleratorEnum.GPU
        const val IN = 128
        const val SCALE = 4
        const val OUT = IN * SCALE // 512

        fun toAccelerator(a: AcceleratorEnum): Accelerator = when (a) {
            AcceleratorEnum.CPU -> Accelerator.CPU
            AcceleratorEnum.GPU -> Accelerator.GPU
        }
    }

    val result: SharedFlow<Result>
        get() = _result
    private val _result = MutableSharedFlow<Result>(
        extraBufferCapacity = 8, onBufferOverflow = BufferOverflow.DROP_OLDEST
    )

    val error: SharedFlow<Throwable?>
        get() = _error
    private val _error = MutableSharedFlow<Throwable?>()

    private var model: CompiledModel? = null
    private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")
    private val inputFloats = FloatArray(IN * IN * 3)
    private val inPixels = IntArray(IN * IN)
    private val outPixels = IntArray(OUT * OUT)

    suspend fun initModel() {
        cleanup()
        try {
            withContext(singleThreadDispatcher) {
                model = CompiledModel.create(
                    context.assets, options.model.fileName,
                    CompiledModel.Options(toAccelerator(options.delegate)), null,
                )
                Log.i(TAG, "Created CompiledModel ${options.model.fileName} on ${options.delegate}")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Create CompiledModel failed: ${e.message}")
            _error.emit(e)
        }
    }

    suspend fun cleanup() {
        withContext(singleThreadDispatcher) {
            model?.close()
            model = null
        }
    }

    fun setOptions(options: Options) { this.options = options }

    /** Resize [bitmap] to one 128×128 tile, ×4 super-resolve, emit the 512×512 result + bicubic baseline. */
    suspend fun superResolve(bitmap: Bitmap) {
        try {
            withContext(singleThreadDispatcher) {
                val currentModel = model ?: return@withContext
                val tile = Bitmap.createScaledBitmap(bitmap, IN, IN, true)
                tile.getPixels(inPixels, 0, IN, 0, 0, IN, IN)
                var idx = 0
                for (p in inPixels) {
                    inputFloats[idx++] = Color.red(p) / 255f
                    inputFloats[idx++] = Color.green(p) / 255f
                    inputFloats[idx++] = Color.blue(p) / 255f
                }

                val start = SystemClock.uptimeMillis()
                val inputBuffers = currentModel.createInputBuffers()
                val outputBuffers = currentModel.createOutputBuffers()
                inputBuffers[0].writeFloat(inputFloats)
                currentModel.run(inputBuffers, outputBuffers)
                val out = outputBuffers[0].readFloat() // [512*512*3] NHWC 0-1
                inputBuffers.forEach { it.close() }
                outputBuffers.forEach { it.close() }
                val inferenceTime = SystemClock.uptimeMillis() - start

                var o = 0
                for (i in outPixels.indices) {
                    val r = (out[o++].coerceIn(0f, 1f) * 255f).toInt()
                    val g = (out[o++].coerceIn(0f, 1f) * 255f).toInt()
                    val b = (out[o++].coerceIn(0f, 1f) * 255f).toInt()
                    outPixels[i] = Color.argb(255, r, g, b)
                }
                val sr = Bitmap.createBitmap(outPixels, OUT, OUT, Bitmap.Config.ARGB_8888)
                val bicubic = Bitmap.createScaledBitmap(tile, OUT, OUT, true) // before (bilinear ×4)
                if (isActive) _result.emit(Result(sr, bicubic, inferenceTime))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Super-resolution error: ${e.message}")
            _error.emit(e)
        }
    }

    enum class AcceleratorEnum { CPU, GPU }

    enum class Model(val fileName: String) {
        RealEsrganX4v3("realesr_general_x4v3.tflite"),
    }

    data class Result(val superResolved: Bitmap, val baseline: Bitmap, val inferenceTime: Long)
}

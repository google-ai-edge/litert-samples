/*
 * Copyright 2025 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.depth_estimation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.os.SystemClock
import android.util.Log
import androidx.core.graphics.scale
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.BufferOverflow
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext

/**
 * Runs MiDaS_small monocular depth estimation with the LiteRT CompiledModel API.
 *
 * Model: midas_small_256_fp16.tflite
 *  - input : 1 x 256 x 256 x 3 float32, RGB, ImageNet-normalized (NHWC, interleaved)
 *  - output: 1 x 256 x 256 float32, inverse depth (relative; larger = nearer)
 */
class DepthEstimationHelper(private val context: Context) {

  val depth: SharedFlow<DepthResult>
    get() = _depth

  private val _depth =
    MutableSharedFlow<DepthResult>(
      extraBufferCapacity = 64,
      onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

  val error: SharedFlow<Throwable?>
    get() = _error

  private val _error = MutableSharedFlow<Throwable?>()

  private var estimator: Estimator? = null
  private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

  /** Create a CompiledModel for the chosen accelerator. */
  suspend fun initEstimator(acceleratorEnum: AcceleratorEnum = AcceleratorEnum.GPU) {
    cleanup()
    try {
      withContext(singleThreadDispatcher) {
        val model =
          CompiledModel.create(
            context.assets,
            MODEL_FILE,
            CompiledModel.Options(toAccelerator(acceleratorEnum)),
            null,
          )
        estimator = Estimator(model)
        Log.d(TAG, "Created a depth estimator ($acceleratorEnum)")
      }
    } catch (e: Exception) {
      Log.i(TAG, "Create LiteRT from $MODEL_FILE failed: ${e.message}")
      _error.emit(e)
    }
  }

  suspend fun cleanup() {
    try {
      withContext(singleThreadDispatcher) {
        estimator?.cleanup()
        estimator = null
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error during cleanup: ${e.message}")
    }
  }

  suspend fun estimate(bitmap: Bitmap) {
    try {
      withContext(singleThreadDispatcher) {
        estimator?.estimate(bitmap)?.let { if (isActive) _depth.emit(it) }
      }
    } catch (e: Exception) {
      Log.i(TAG, "Depth estimate error: ${e.message}")
      _error.emit(e)
    }
  }

  private class Estimator(private val model: CompiledModel) {
    private val inputBuffers = model.createInputBuffers()
    private val outputBuffers = model.createOutputBuffers()

    fun cleanup() {
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
      model.close()
    }

    fun estimate(bitmap: Bitmap): DepthResult {
      val image = bitmap.scale(SIZE, SIZE, true)
      val input = preprocess(image)

      val start = SystemClock.uptimeMillis()
      inputBuffers[0].writeFloat(input)
      model.run(inputBuffers, outputBuffers)
      val depth = outputBuffers[0].readFloat() // SIZE * SIZE inverse depth
      val inferenceTime = SystemClock.uptimeMillis() - start

      // Colorize at model resolution, then restore the source aspect ratio so the
      // depth map lines up with the original image instead of being stretched (the
      // input was squashed to a square for inference).
      val overlay =
        DepthColorMap.toBitmap(depth, SIZE, SIZE).restoreAspectRatio(bitmap.width, bitmap.height)
      return DepthResult(overlay, inferenceTime)
    }

    /** Resize the square depth map back to the source aspect ratio (capped on the long side). */
    private fun Bitmap.restoreAspectRatio(srcWidth: Int, srcHeight: Int): Bitmap {
      if (srcWidth <= 0 || srcHeight <= 0) return this
      val ratio = (MAX_OVERLAY_SIDE.toFloat() / maxOf(srcWidth, srcHeight)).coerceAtMost(1f)
      val w = (srcWidth * ratio).toInt().coerceAtLeast(1)
      val h = (srcHeight * ratio).toInt().coerceAtLeast(1)
      return scale(w, h, true)
    }

    /** Resize-> ImageNet normalize -> interleaved NHWC RGB float32. */
    private fun preprocess(image: Bitmap): FloatArray {
      val numPixels = SIZE * SIZE
      val pixels = IntArray(numPixels)
      val out = FloatArray(numPixels * 3)
      image.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
      for (i in 0 until numPixels) {
        val p = pixels[i]
        val base = i * 3
        out[base + 0] = (Color.red(p) / 255f - MEAN[0]) / STD[0]
        out[base + 1] = (Color.green(p) / 255f - MEAN[1]) / STD[1]
        out[base + 2] = (Color.blue(p) / 255f - MEAN[2]) / STD[2]
      }
      return out
    }
  }

  data class DepthResult(val overlay: Bitmap, val inferenceTime: Long)

  enum class AcceleratorEnum {
    CPU,
    GPU,
  }

  private companion object {
    const val TAG = "DepthEstimation"
    const val MODEL_FILE = "midas_small_256_fp16.tflite"
    const val SIZE = 256
    const val MAX_OVERLAY_SIDE = 1024
    val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    val STD = floatArrayOf(0.229f, 0.224f, 0.225f)

    fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator =
      when (acceleratorEnum) {
        AcceleratorEnum.CPU -> Accelerator.CPU
        AcceleratorEnum.GPU -> Accelerator.GPU
      }
  }
}

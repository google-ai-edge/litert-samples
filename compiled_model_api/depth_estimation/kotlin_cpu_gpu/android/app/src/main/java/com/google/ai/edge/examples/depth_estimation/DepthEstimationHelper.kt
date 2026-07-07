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

package com.google.ai.edge.examples.depth_estimation

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
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
 * Runs monocular depth estimation with the LiteRT CompiledModel API. Two models are selectable:
 *
 *  - **MiDaS-small** (`midas_small_256_fp16.tflite`): 256x256 NHWC input (image stretched to square),
 *    256x256 inverse-depth output, `inferno` colormap. Fast and light.
 *  - **Depth Anything 3 — Small** (`da3_small_gpu_fp16.tflite`): 504x896 NCHW input (image letterboxed
 *    into the native portrait aspect), 896x504 metric-ish depth output, disparity + Spectral colormap.
 *    State-of-the-art (2025), heavier. Both run fully on the LiteRT GPU delegate.
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

  /** Create a CompiledModel for the chosen model and accelerator. */
  suspend fun initEstimator(
    model: Model = Model.MidasSmall,
    acceleratorEnum: AcceleratorEnum = AcceleratorEnum.GPU,
  ) {
    cleanup()
    try {
      withContext(singleThreadDispatcher) {
        val compiledModel =
          CompiledModel.create(
            context.assets,
            model.fileName,
            CompiledModel.Options(toAccelerator(acceleratorEnum)),
            null,
          )
        estimator = Estimator(compiledModel, model)
        Log.i(TAG, "Created a depth estimator ($model, $acceleratorEnum)")
      }
    } catch (e: Exception) {
      Log.e(TAG, "Create LiteRT from ${model.fileName} failed: ${e.message}")
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
      Log.e(TAG, "Depth estimate error: ${e.message}")
      _error.emit(e)
    }
  }

  private class Estimator(private val model: CompiledModel, private val spec: Model) {
    private val inputBuffers = model.createInputBuffers()
    private val outputBuffers = model.createOutputBuffers()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

    fun cleanup() {
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
      model.close()
    }

    fun estimate(bitmap: Bitmap): DepthResult =
      when (spec) {
        Model.MidasSmall -> estimateMidas(bitmap)
        Model.DepthAnything3 -> estimateDepthAnything3(bitmap)
      }

    /** MiDaS: stretch to a square, NHWC input, inverse-depth -> inferno, restore source aspect. */
    private fun estimateMidas(bitmap: Bitmap): DepthResult {
      val w = spec.inputW
      val image = bitmap.scale(w, w, true)
      val pixels = IntArray(w * w)
      image.getPixels(pixels, 0, w, 0, 0, w, w)
      val input = FloatArray(w * w * 3)
      for (i in pixels.indices) {
        val p = pixels[i]
        val base = i * 3
        input[base + 0] = (Color.red(p) / 255f - MEAN[0]) / STD[0]
        input[base + 1] = (Color.green(p) / 255f - MEAN[1]) / STD[1]
        input[base + 2] = (Color.blue(p) / 255f - MEAN[2]) / STD[2]
      }

      val start = SystemClock.uptimeMillis()
      inputBuffers[0].writeFloat(input)
      model.run(inputBuffers, outputBuffers)
      val depth = outputBuffers[0].readFloat() // w * w inverse depth
      val inferenceTime = SystemClock.uptimeMillis() - start

      val overlay =
        DepthColorMap.inverseDepthInferno(depth, w, w).restoreAspectRatio(bitmap.width, bitmap.height)
      return DepthResult(overlay, inferenceTime)
    }

    /**
     * Depth Anything 3: letterbox into the native portrait rectangle (no distortion), NCHW input,
     * depth -> disparity (1/depth) -> robust percentile normalize -> Spectral, cropped to content.
     */
    private fun estimateDepthAnything3(bitmap: Bitmap): DepthResult {
      val w = spec.inputW
      val h = spec.inputH
      val canvasBmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
      val canvas = Canvas(canvasBmp)
      canvas.drawColor(Color.BLACK)
      val srcAspect = bitmap.width.toFloat() / bitmap.height
      val dstAspect = w.toFloat() / h
      val dst =
        if (srcAspect > dstAspect) { // wider than the model rect -> fit width, pad top/bottom
          val fitH = w / srcAspect
          val y = (h - fitH) * 0.5f
          RectF(0f, y, w.toFloat(), y + fitH)
        } else { // taller -> fit height, pad left/right
          val fitW = h * srcAspect
          val x = (w - fitW) * 0.5f
          RectF(x, 0f, x + fitW, h.toFloat())
        }
      canvas.drawBitmap(bitmap, null, dst, paint)
      val pixels = IntArray(w * h)
      canvasBmp.getPixels(pixels, 0, w, 0, 0, w, h)
      canvasBmp.recycle()

      val plane = w * h
      val input = FloatArray(plane * 3)
      for (i in pixels.indices) {
        val p = pixels[i]
        input[i] = (Color.red(p) / 255f - MEAN[0]) / STD[0]
        input[plane + i] = (Color.green(p) / 255f - MEAN[1]) / STD[1]
        input[2 * plane + i] = (Color.blue(p) / 255f - MEAN[2]) / STD[2]
      }

      val start = SystemClock.uptimeMillis()
      inputBuffers[0].writeFloat(input)
      model.run(inputBuffers, outputBuffers)
      val depth = outputBuffers[0].readFloat() // h * w depth
      val inferenceTime = SystemClock.uptimeMillis() - start

      val overlay =
        DepthColorMap.disparitySpectral(
          depth,
          w,
          h,
          dst.left.toInt().coerceIn(0, w),
          dst.top.toInt().coerceIn(0, h),
          dst.right.toInt().coerceIn(0, w),
          dst.bottom.toInt().coerceIn(0, h),
        )
      return DepthResult(overlay, inferenceTime)
    }

    /** Resize a depth map to the source aspect ratio (capped on the long side). */
    private fun Bitmap.restoreAspectRatio(srcWidth: Int, srcHeight: Int): Bitmap {
      if (srcWidth <= 0 || srcHeight <= 0) return this
      val ratio = (MAX_OVERLAY_SIDE.toFloat() / maxOf(srcWidth, srcHeight)).coerceAtMost(1f)
      val outW = (srcWidth * ratio).toInt().coerceAtLeast(1)
      val outH = (srcHeight * ratio).toInt().coerceAtLeast(1)
      return scale(outW, outH, true)
    }
  }

  data class DepthResult(val overlay: Bitmap, val inferenceTime: Long)

  enum class Model(
    val displayName: String,
    val fileName: String,
    val inputW: Int,
    val inputH: Int,
  ) {
    MidasSmall("MiDaS-small", "midas_small_256_fp16.tflite", 256, 256),
    DepthAnything3("Depth Anything 3", "da3_small_gpu_fp16.tflite", 504, 896),
  }

  enum class AcceleratorEnum {
    CPU,
    GPU,
  }

  private companion object {
    const val TAG = "DepthEstimation"
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

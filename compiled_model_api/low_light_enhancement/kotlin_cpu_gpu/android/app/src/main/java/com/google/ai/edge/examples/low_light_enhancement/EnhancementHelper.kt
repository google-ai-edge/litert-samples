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

package com.google.ai.edge.examples.low_light_enhancement

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
 * Runs Zero-DCE low-light image enhancement with the LiteRT CompiledModel API.
 *
 * Model: zerodce_512_fp16.tflite
 *  - input : 1 x 512 x 512 x 3 float32, RGB, range [0,1] (NHWC, interleaved)
 *  - output: 1 x 512 x 512 x 3 float32, RGB, range [0,1] (the enhanced image)
 *
 * Zero-DCE estimates per-pixel tone curves and applies them iteratively; the whole
 * pipeline is baked into the graph, so the model maps a dark image straight to the
 * brightened one. No ImageNet normalization — the input is just pixels / 255.
 */
class EnhancementHelper(private val context: Context) {

  val enhanced: SharedFlow<EnhanceResult>
    get() = _enhanced

  private val _enhanced =
    MutableSharedFlow<EnhanceResult>(
      extraBufferCapacity = 64,
      onBufferOverflow = BufferOverflow.DROP_OLDEST,
    )

  val error: SharedFlow<Throwable?>
    get() = _error

  private val _error = MutableSharedFlow<Throwable?>()

  private var enhancer: Enhancer? = null
  private val singleThreadDispatcher = Dispatchers.IO.limitedParallelism(1, "ModelDispatcher")

  /** Create a CompiledModel for the chosen accelerator. */
  suspend fun initEnhancer(acceleratorEnum: AcceleratorEnum = AcceleratorEnum.GPU) {
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
        enhancer = Enhancer(model)
        Log.d(TAG, "Created an enhancer ($acceleratorEnum)")
      }
    } catch (e: Exception) {
      Log.i(TAG, "Create LiteRT from $MODEL_FILE failed: ${e.message}")
      _error.emit(e)
    }
  }

  suspend fun cleanup() {
    try {
      withContext(singleThreadDispatcher) {
        enhancer?.cleanup()
        enhancer = null
      }
    } catch (e: Exception) {
      Log.e(TAG, "Error during cleanup: ${e.message}")
    }
  }

  suspend fun enhance(bitmap: Bitmap) {
    try {
      withContext(singleThreadDispatcher) {
        enhancer?.enhance(bitmap)?.let { if (isActive) _enhanced.emit(it) }
      }
    } catch (e: Exception) {
      Log.i(TAG, "Enhance error: ${e.message}")
      _error.emit(e)
    }
  }

  private class Enhancer(private val model: CompiledModel) {
    private val inputBuffers = model.createInputBuffers()
    private val outputBuffers = model.createOutputBuffers()

    fun cleanup() {
      inputBuffers.forEach { it.close() }
      outputBuffers.forEach { it.close() }
      model.close()
    }

    fun enhance(bitmap: Bitmap): EnhanceResult {
      val image = bitmap.scale(SIZE, SIZE, true)
      val input = preprocess(image)

      val start = SystemClock.uptimeMillis()
      inputBuffers[0].writeFloat(input)
      model.run(inputBuffers, outputBuffers)
      val out = outputBuffers[0].readFloat() // SIZE * SIZE * 3 interleaved RGB in [0,1]
      val inferenceTime = SystemClock.uptimeMillis() - start

      val enhanced = postprocess(out).restoreAspectRatio(bitmap.width, bitmap.height)
      return EnhanceResult(enhanced, inferenceTime)
    }

    /** Resize -> interleaved NHWC RGB float32 in [0,1] (no mean/std normalization). */
    private fun preprocess(image: Bitmap): FloatArray {
      val numPixels = SIZE * SIZE
      val pixels = IntArray(numPixels)
      val out = FloatArray(numPixels * 3)
      image.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
      for (i in 0 until numPixels) {
        val p = pixels[i]
        val base = i * 3
        out[base + 0] = Color.red(p) / 255f
        out[base + 1] = Color.green(p) / 255f
        out[base + 2] = Color.blue(p) / 255f
      }
      return out
    }

    /** Interleaved RGB [0,1] -> ARGB bitmap. */
    private fun postprocess(out: FloatArray): Bitmap {
      val numPixels = SIZE * SIZE
      val pixels = IntArray(numPixels)
      for (i in 0 until numPixels) {
        val base = i * 3
        val r = (out[base + 0] * 255f).toInt().coerceIn(0, 255)
        val g = (out[base + 1] * 255f).toInt().coerceIn(0, 255)
        val b = (out[base + 2] * 255f).toInt().coerceIn(0, 255)
        pixels[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
      }
      return Bitmap.createBitmap(pixels, SIZE, SIZE, Bitmap.Config.ARGB_8888)
    }

    /** Resize the square output back to the source aspect ratio (capped on the long side). */
    private fun Bitmap.restoreAspectRatio(srcWidth: Int, srcHeight: Int): Bitmap {
      if (srcWidth <= 0 || srcHeight <= 0) return this
      val ratio = (MAX_OUTPUT_SIDE.toFloat() / maxOf(srcWidth, srcHeight)).coerceAtMost(1f)
      val w = (srcWidth * ratio).toInt().coerceAtLeast(1)
      val h = (srcHeight * ratio).toInt().coerceAtLeast(1)
      return scale(w, h, true)
    }
  }

  data class EnhanceResult(val overlay: Bitmap, val inferenceTime: Long)

  enum class AcceleratorEnum {
    CPU,
    GPU,
  }

  private companion object {
    const val TAG = "LowLightEnhancement"
    const val MODEL_FILE = "zerodce_512_fp16.tflite"
    const val SIZE = 512
    const val MAX_OUTPUT_SIDE = 1024

    fun toAccelerator(acceleratorEnum: AcceleratorEnum): Accelerator =
      when (acceleratorEnum) {
        AcceleratorEnum.CPU -> Accelerator.CPU
        AcceleratorEnum.GPU -> Accelerator.GPU
      }
  }
}

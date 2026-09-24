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

package com.google.ai.edge.examples.model_zoo.models.realesrgan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

/**
 * Real-ESRGAN x4 upscaler using LiteRT CompiledModel (GPU). Input: [1, 128, 128, 3], Output:
 * [1, 512, 512, 3]. Processes large images by tiling with overlap.
 */
class RealEsrganUpscaler(
  context: Context,
  modelFile: File,
  accelerator: Accelerator = Accelerator.GPU,
) : AutoCloseable {

  companion object {
    private const val TAG = "RealESRGAN"
    private const val TILE_SIZE = 128
    private const val OUTPUT_SCALE = 4
    private const val OUTPUT_TILE = TILE_SIZE * OUTPUT_SCALE // 512
    private const val OVERLAP = 16 // tile overlap to hide seams

    // I/O adapter for the HF artifact; the original per-pixel decode below is unchanged.
    internal fun outputFromNchw(planar: FloatArray): FloatArray {
      require(planar.size % 3 == 0)
      val plane = planar.size / 3
      return FloatArray(planar.size).also { interleaved ->
        for (i in 0 until plane) {
          interleaved[i * 3] = planar[i]
          interleaved[i * 3 + 1] = planar[plane + i]
          interleaved[i * 3 + 2] = planar[2 * plane + i]
        }
      }
    }

    internal fun fillNhwcPixels(output: FloatArray, outPixels: IntArray) {
      for (i in outPixels.indices) {
        val r = (output[i * 3] * 255f).toInt().coerceIn(0, 255)
        val g = (output[i * 3 + 1] * 255f).toInt().coerceIn(0, 255)
        val b = (output[i * 3 + 2] * 255f).toInt().coerceIn(0, 255)
        outPixels[i] = 0xFF000000.toInt() or (r shl 16) or (g shl 8) or b
      }
    }
  }

  private val compiledModel: CompiledModel
  private val inputBuffers: List<TensorBuffer>

  // Pre-allocated buffers for single tile
  private val inputFloats = FloatArray(TILE_SIZE * TILE_SIZE * 3)
  private val inputPixels = IntArray(TILE_SIZE * TILE_SIZE)
  private val tileBitmap = Bitmap.createBitmap(TILE_SIZE, TILE_SIZE, Bitmap.Config.ARGB_8888)
  private val scaleMatrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    var createdModel: CompiledModel? = null
    var createdInputs: List<TensorBuffer>? = null
    try {
      Log.i(TAG, "Loading model: $modelFile")
      val options = CompiledModel.Options(accelerator)
      if (accelerator == Accelerator.GPU)
        try {
          options.gpuOptions =
            CompiledModel.GpuOptions(
              null,
              null,
              null,
              CompiledModel.GpuOptions.Precision.FP32,
              null,
              null,
              null,
              null,
              null,
              null,
              null,
              null,
              null,
              null,
              null,
            )
        } catch (_: Exception) {}
      compiledModel =
        CompiledModel.create(modelFile.absolutePath, options, null).also { createdModel = it }
      Log.i(TAG, "$accelerator FP32 compiled OK")
      inputBuffers = compiledModel.createInputBuffers().also { createdInputs = it }
      Log.i(TAG, "Model ready: ${TILE_SIZE}x${TILE_SIZE} -> ${OUTPUT_TILE}x${OUTPUT_TILE}")
    } catch (failure: Throwable) {
      createdInputs?.forEach { runCatching { it.close() } }
      runCatching { createdModel?.close() }
      runCatching { if (!tileBitmap.isRecycled) tileBitmap.recycle() }
      throw failure
    }
  }

  /** Upscale a single 128x128 tile. Returns 512x512 bitmap. */
  fun upscaleTile(input: Bitmap): Bitmap {
    // Resize input to tile size
    val canvas = Canvas(tileBitmap)
    scaleMatrix.setScale(TILE_SIZE.toFloat() / input.width, TILE_SIZE.toFloat() / input.height)
    canvas.drawBitmap(input, scaleMatrix, paint)

    // Normalize to 0-1
    tileBitmap.getPixels(inputPixels, 0, TILE_SIZE, 0, 0, TILE_SIZE, TILE_SIZE)
    var idx = 0
    for (pixel in inputPixels) {
      inputFloats[idx++] = Color.red(pixel) / 255f
      inputFloats[idx++] = Color.green(pixel) / 255f
      inputFloats[idx++] = Color.blue(pixel) / 255f
    }
    inputBuffers[0].writeFloat(inputFloats)

    val resultBuffers = compiledModel.run(inputBuffers)
    val output =
      try {
        outputFromNchw(resultBuffers[0].readFloat())
      } finally {
        resultBuffers.forEach { it.close() }
      }

    // Convert output to bitmap
    val outPixels = IntArray(OUTPUT_TILE * OUTPUT_TILE)
    fillNhwcPixels(output, outPixels)

    val result = Bitmap.createBitmap(OUTPUT_TILE, OUTPUT_TILE, Bitmap.Config.ARGB_8888)
    result.setPixels(outPixels, 0, OUTPUT_TILE, 0, 0, OUTPUT_TILE, OUTPUT_TILE)
    return result
  }

  /** Upscale an image of any size using tiling with overlap. Returns a 4x upscaled bitmap. */
  fun upscale(input: Bitmap, onProgress: ((Float) -> Unit)? = null): Bitmap {
    val w = input.width
    val h = input.height
    val outW = w * OUTPUT_SCALE
    val outH = h * OUTPUT_SCALE
    val result = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
    val resultCanvas = Canvas(result)

    val step = TILE_SIZE - OVERLAP
    val tilesX = (w + step - 1) / step
    val tilesY = (h + step - 1) / step
    val totalTiles = tilesX * tilesY
    var tilesDone = 0

    Log.i(TAG, "Upscaling ${w}x${h} -> ${outW}x${outH} ($totalTiles tiles)")

    for (ty in 0 until tilesY) {
      for (tx in 0 until tilesX) {
        val srcX = (tx * step).coerceAtMost(w - TILE_SIZE).coerceAtLeast(0)
        val srcY = (ty * step).coerceAtMost(h - TILE_SIZE).coerceAtLeast(0)

        // Handle images smaller than tile size
        val cropW = TILE_SIZE.coerceAtMost(w - srcX).coerceAtMost(w)
        val cropH = TILE_SIZE.coerceAtMost(h - srcY).coerceAtMost(h)

        val tile =
          if (cropW == TILE_SIZE && cropH == TILE_SIZE) {
            Bitmap.createBitmap(input, srcX, srcY, TILE_SIZE, TILE_SIZE)
          } else {
            // Pad small tiles
            val padded = Bitmap.createBitmap(TILE_SIZE, TILE_SIZE, Bitmap.Config.ARGB_8888)
            Canvas(padded).drawBitmap(input, (-srcX).toFloat(), (-srcY).toFloat(), null)
            padded
          }

        val upscaled = upscaleTile(tile)
        if (tile !== input) tile.recycle()

        // Draw to result at 4x position
        val dstX = srcX * OUTPUT_SCALE
        val dstY = srcY * OUTPUT_SCALE
        resultCanvas.drawBitmap(upscaled, dstX.toFloat(), dstY.toFloat(), null)
        upscaled.recycle()

        tilesDone++
        onProgress?.invoke(tilesDone.toFloat() / totalTiles)
      }
    }

    Log.i(TAG, "Upscaling complete: ${outW}x${outH}")
    return result
  }

  override fun close() {
    inputBuffers.forEach { it.close() }
    compiledModel.close()
    tileBitmap.recycle()
  }
}

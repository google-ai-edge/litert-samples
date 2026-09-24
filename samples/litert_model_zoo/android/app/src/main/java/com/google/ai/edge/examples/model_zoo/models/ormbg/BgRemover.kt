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

package com.google.ai.edge.examples.model_zoo.models.ormbg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

/**
 * ormbg open background removal on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 1024, 1024] NCHW, RGB, x/255. Output: [1, 1, 1024, 1024] alpha matte in [0,1]
 * (min-max normalized per frame).
 *
 * ISNet (RSU / U²-Net-style blocks) — a pure CNN, fully GPU-compatible with one defensive patch
 * (align_corners=False on the bilinear upsamples). ~10 ms/frame.
 */
class BgRemover(
  context: Context,
  modelFile: File,
  private val accelerator: Accelerator = Accelerator.GPU,
) : AutoCloseable {

  companion object {
    private const val TAG = "ormbg"
    const val SIZE = 1024
    const val OUT = 256 // downscaled matte returned to the UI (fast compositing)

    internal fun normalizeMatte(
      full: FloatArray,
      SIZE: Int = BgRemover.SIZE,
      OUT: Int = BgRemover.OUT,
    ): FloatArray {
      var mn = Float.MAX_VALUE
      var mx = -Float.MAX_VALUE
      for (v in full) {
        if (v < mn) mn = v
        if (v > mx) mx = v
      }
      val inv = 1f / (mx - mn + 1e-6f)
      val out = FloatArray(OUT * OUT)
      val step = SIZE / OUT
      for (y in 0 until OUT) {
        val sy = y * step
        for (x in 0 until OUT) {
          out[y * OUT + x] = (full[sy * SIZE + x * step] - mn) * inv
        }
      }
      return out
    }
  }

  private var env: Environment? = null
  private val model: CompiledModel

  /** Wall time of the one-off compile/load, which is where the NPU separates itself. */
  var loadMs: Long = 0
    private set

  /**
   * Time for the model alone — run plus the readback that forces it to finish. This is the figure
   * the published benchmarks quote. [matte] also returns a whole-frame time, which additionally
   * carries resize, NCHW packing and matte normalization; those are identical on both accelerators
   * and would dilute the comparison.
   */
  var modelMs: Long = 0
    private set

  private val inBufs: List<TensorBuffer>
  private val outBufs: List<TensorBuffer>

  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    var createdModel: CompiledModel? = null
    var createdInputs: List<TensorBuffer>? = null
    var createdOutputs: List<TensorBuffer>? = null
    try {
      val t0 = System.nanoTime()
      val options = CompiledModel.Options(accelerator)
      if (accelerator == Accelerator.NPU) {
        // The NPU needs the dispatch library directory explicitly: LiteRT only warns
        // when it is missing and then runs without the NPU. The same directory also
        // becomes ADSP_LIBRARY_PATH, which is how the Hexagon skel is found.
        env =
          Environment.create(
            context,
            mapOf(
              Environment.Option.DispatchLibraryDir to context.applicationInfo.nativeLibraryDir,
              // On-device (JIT) compilation needs the compiler plugin as well.
              // Without it the model silently runs on CPU and still returns a number.
              Environment.Option.CompilerPluginLibraryDir to
                context.applicationInfo.nativeLibraryDir,
            ),
          )
        options.qualcommOptions =
          CompiledModel.QualcommOptions(
            htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST
          )
      }
      model = CompiledModel.create(modelFile.absolutePath, options, env).also { createdModel = it }
      inBufs = model.createInputBuffers().also { createdInputs = it }
      outBufs = model.createOutputBuffers().also { createdOutputs = it }
      loadMs = (System.nanoTime() - t0) / 1_000_000
      Log.i(TAG, "$accelerator ready in ${loadMs}ms — ${inBufs.size} in / ${outBufs.size} out")
    } catch (failure: Throwable) {
      createdOutputs?.forEach { runCatching { it.close() } }
      createdInputs?.forEach { runCatching { it.close() } }
      runCatching { createdModel?.close() }
      runCatching { env?.close() }
      runCatching { if (!resized.isRecycled) resized.recycle() }
      throw failure
    }
  }

  /** Returns an [OUT]×[OUT] alpha matte (0..1, min-max normalized) + time (ms). */
  fun matte(bitmap: Bitmap): Pair<FloatArray, Long> {
    val t = System.nanoTime()
    Canvas(resized)
      .drawBitmap(
        bitmap,
        matrix.apply { setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height) },
        paint,
      )
    resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
    val plane = SIZE * SIZE
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = ((p shr 16) and 0xFF) / 255f
      inputFloats[plane + i] = ((p shr 8) and 0xFF) / 255f
      inputFloats[2 * plane + i] = (p and 0xFF) / 255f
    }
    inBufs[0].writeFloat(inputFloats)
    val tm = System.nanoTime()
    model.run(inBufs, outBufs)
    // run() only enqueues; reading the output is what waits for the compute.
    val full = outBufs[0].readFloat() // [1024*1024]
    modelMs = (System.nanoTime() - tm) / 1_000_000

    // min-max normalize then downsample to OUT×OUT (nearest) for fast UI compositing
    val out = normalizeMatte(full)

    return out to ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    inBufs.forEach { it.close() }
    outBufs.forEach { it.close() }
    model.close()
    env?.close()
    if (!resized.isRecycled) resized.recycle()
  }
}

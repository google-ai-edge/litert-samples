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

package com.google.ai.edge.examples.model_zoo.models.da3

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import java.io.File

/**
 * Depth Anything 3 (DA3-SMALL mono) depth predictor: single CompiledModel GPU inference. The model
 * is fixed to the native portrait aspect (896x504). Arbitrary images are letterboxed (aspect
 * preserved, padded) into 896x504, and the depth is cropped back to the content region. Input
 * [1,3,896,504] NCHW, ImageNet-normalized; output [1,1,896,504] depth.
 */
class DA3Predictor(context: Context, modelDir: File, accelerator: Accelerator = Accelerator.GPU) :
  AutoCloseable {

  companion object {
    private const val TAG = "DA3"
    private const val MODEL_FILE = "da3_small_gpu_fp16.tflite"
    const val MODEL_H = 896
    const val MODEL_W = 504
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  private val compiledModel: CompiledModel
  private val inputBuffers: List<TensorBuffer>
  private val inputFloats = FloatArray(3 * MODEL_H * MODEL_W)
  private val pixels = IntArray(MODEL_H * MODEL_W)
  private val canvasBmp = Bitmap.createBitmap(MODEL_W, MODEL_H, Bitmap.Config.ARGB_8888)
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  var acceleratorName = ""
    private set

  init {
    compiledModel =
      try {
        CompiledModel.create(
          File(modelDir, MODEL_FILE).absolutePath,
          CompiledModel.Options(accelerator),
          null,
        )
      } catch (failure: Exception) {
        canvasBmp.recycle()
        throw failure
      }
    acceleratorName = if (accelerator == Accelerator.GPU) "GPU" else "CPU"
    inputBuffers =
      try {
        compiledModel.createInputBuffers()
      } catch (failure: Exception) {
        compiledModel.close()
        canvasBmp.recycle()
        throw failure
      }
  }

  fun predict(src: Bitmap): DA3Result {
    val t = System.nanoTime()

    // letterbox into MODEL_W x MODEL_H, preserving aspect (no distortion)
    val canvas = Canvas(canvasBmp)
    canvas.drawColor(Color.BLACK)
    val srcAspect = src.width.toFloat() / src.height
    val dstAspect = MODEL_W.toFloat() / MODEL_H
    val dst =
      if (srcAspect > dstAspect) { // wider than model → fit width
        val fitH = MODEL_W / srcAspect
        val y = (MODEL_H - fitH) * 0.5f
        RectF(0f, y, MODEL_W.toFloat(), y + fitH)
      } else { // taller → fit height
        val fitW = MODEL_H * srcAspect
        val x = (MODEL_W - fitW) * 0.5f
        RectF(x, 0f, x + fitW, MODEL_H.toFloat())
      }
    canvas.drawBitmap(src, null, dst, paint)
    canvasBmp.getPixels(pixels, 0, MODEL_W, 0, 0, MODEL_W, MODEL_H)

    val plane = MODEL_H * MODEL_W
    for (i in pixels.indices) {
      val p = pixels[i]
      inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
      inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
      inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
    }
    inputBuffers[0].writeFloat(inputFloats)

    val out = compiledModel.run(inputBuffers)
    val depth =
      try {
        out[0].readFloat()
      } finally {
        out.forEach { it.close() }
      }

    val ms = (System.nanoTime() - t) / 1_000_000
    Log.i(TAG, "Inference: ${ms}ms ($acceleratorName)")
    val cl = dst.left.toInt().coerceIn(0, MODEL_W)
    val ct = dst.top.toInt().coerceIn(0, MODEL_H)
    val cr = dst.right.toInt().coerceIn(0, MODEL_W)
    val cb = dst.bottom.toInt().coerceIn(0, MODEL_H)
    return DA3Result(depth, MODEL_W, MODEL_H, cl, ct, cr, cb, ms, acceleratorName)
  }

  override fun close() {
    inputBuffers.forEach { it.close() }
    compiledModel.close()
    canvasBmp.recycle()
  }
}

data class DA3Result(
  val depth: FloatArray,
  val width: Int,
  val height: Int,
  val cropL: Int,
  val cropT: Int,
  val cropR: Int,
  val cropB: Int,
  val inferenceMs: Long,
  val accelerator: String,
) {
  /**
   * Depth visualization matching the official DA3 `visualize.py`: disparity = 1/depth → robust
   * 2nd/98th-percentile normalization → invert → **Spectral** colormap (256-entry LUT). Cropped to
   * the content region. The percentile clip (not full min-max) is what keeps it clean instead of
   * washed out by outliers. (For plain grayscale: `n = (disp-lo)/range; gray = n*255`.)
   */
  fun depthBitmap(): Bitmap {
    val cw = (cropR - cropL).coerceAtLeast(1)
    val ch = (cropB - cropT).coerceAtLeast(1)
    val px = depthPixels()
    return Bitmap.createBitmap(cw, ch, Bitmap.Config.ARGB_8888).also {
      it.setPixels(px, 0, cw, 0, 0, cw, ch)
    }
  }

  /** Disparity/percentile/colormap block, exposed for a JVM fixture. */
  internal fun depthPixels(): IntArray {
    val cw = (cropR - cropL).coerceAtLeast(1)
    val ch = (cropB - cropT).coerceAtLeast(1)
    val disp = FloatArray(cw * ch)
    var k = 0
    for (y in cropT until cropB) for (x in cropL until cropR) {
      val d = depth[y * width + x]
      disp[k++] = if (d > 0f) 1f / d else 0f // 1/depth = disparity (near = large)
    }
    val sorted = disp.copyOf()
    sorted.sort() // 2nd / 98th percentile
    val lo = sorted[(0.02f * (sorted.size - 1)).toInt()]
    val hi = sorted[(0.98f * (sorted.size - 1)).toInt()]
    val range = if (hi > lo) hi - lo else 1e-6f
    val px = IntArray(cw * ch)
    for (i in disp.indices) {
      val n = 1f - ((disp[i] - lo) / range).coerceIn(0f, 1f) // normalize + invert (official)
      px[i] = SPECTRAL[(n * 255f).toInt().coerceIn(0, 255)] // Spectral colormap (matches DA3)
    }
    return px
  }

  companion object {
    // matplotlib "Spectral" colormap, 256 RGB entries (the DA3 default depth colormap)
    private const val LUT =
      "9e0142a00343a20643a40844a70b44a90d45ab0f45ad1246af1446b11747b41947b61b48b81e48ba2049bc2249be254a" +
        "c1274ac32a4bc52c4bc72e4cc9314ccb334dcd364dd0384ed23a4ed43d4fd63f4fd7414ed8434ed9444dda464ddc484c" +
        "dd4a4cde4c4bdf4e4be1504be2514ae3534ae45549e55749e75948e85b48e95c47ea5e47eb6046ed6246ee6445ef6645" +
        "f06744f26944f36b43f46d43f47044f57245f57547f57748f67a49f67c4af67f4bf7814cf7844ef8864ff88950f88c51" +
        "f98e52f99153f99355fa9656fa9857fa9b58fb9d59fba05bfba35cfca55dfca85efcaa5ffdad60fdaf62fdb163fdb365" +
        "fdb567fdb768fdb96afdbb6cfdbd6dfdbf6ffdc171fdc372fdc574fdc776fec877feca79fecc7bfece7cfed07efed27f" +
        "fed481fed683fed884feda86fedc88fede89fee08bfee18dfee28ffee491fee593fee695fee797fee999feea9bfeeb9d" +
        "feec9ffeeda1feefa3fff0a6fff1a8fff2aafff3acfff5aefff6b0fff7b2fff8b4fffab6fffbb8fffcbafffdbcfffebe" +
        "ffffbefefebdfdfebbfcfebafbfdb8fafdb7f9fcb5f8fcb4f7fcb2f6fbb0f5fbaff4faadf3faacf2faaaf1f9a9f0f9a7" +
        "eff9a6eef8a4edf8a3ecf7a1ebf7a0eaf79ee9f69de8f69be7f59ae6f598e4f498e1f399dff299ddf19adaf09ad8ef9b" +
        "d6ee9bd3ed9cd1ed9ccfec9dcdeb9dcaea9ec8e99ec6e89fc3e79fc1e6a0bfe5a0bce4a0bae3a1b8e2a1b5e1a2b3e0a2" +
        "b1dfa3aedea3acdda4aadca4a7dba4a4daa4a2d9a49fd8a49cd7a499d6a497d5a494d4a491d3a48fd2a48cd1a489d0a4" +
        "86cfa584cea581cda57ecca57ccaa579c9a576c8a574c7a571c6a56ec5a56bc4a569c3a566c2a564c0a662bda760bba8" +
        "5eb9a95cb7aa5ab4ab58b2ac56b0ad54aead52abae50a9af4ea7b04ba4b149a2b247a0b3459eb4439bb54199b63f97b7" +
        "3d95b83b92b93990ba378ebb358bbc3389bd3387bc3585bb3682ba3880b93a7eb83b7cb73d79b63f77b54175b44273b3" +
        "4471b2466eb1486cb0496aaf4b68ae4d65ad4e63ac5061aa525fa9545ca8555aa75758a65956a55b53a45c51a35e4fa2"
    private val SPECTRAL =
      IntArray(256) { i ->
        val r = LUT.substring(i * 6, i * 6 + 2).toInt(16)
        val g = LUT.substring(i * 6 + 2, i * 6 + 4).toInt(16)
        val b = LUT.substring(i * 6 + 4, i * 6 + 6).toInt(16)
        (0xFF shl 24) or (r shl 16) or (g shl 8) or b
      }
  }
}

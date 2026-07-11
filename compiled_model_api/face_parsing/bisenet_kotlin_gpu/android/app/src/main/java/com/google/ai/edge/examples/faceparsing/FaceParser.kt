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

package com.google.ai.edge.examples.faceparsing

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer

/**
 * BiSeNet face parsing on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 512, 512]  NCHW, RGB, ImageNet-normalized.
 * Output: [1, 19, 512, 512]  CelebAMask-HQ class logits.
 *
 * BiSeNet (ResNet18 backbone) is a pure CNN; three re-authoring patches make the
 * graph fully GPU-compatible (see conversion/): align_corners=False, global avg-pool
 * -> mean reduce, and a zero-pad maxpool (the -inf-pad PADV2 is rejected by Mali).
 */
class FaceParser(modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "FaceParsing"
    const val SIZE = 512
    const val N_CLASS = 19
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  private val model = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()

  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val labelPixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val labelBitmap = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out")
  }

  /** Parse the face. Returns a 512x512 CelebAMask-colored label bitmap + time (ms). */
  fun parse(bitmap: Bitmap): Pair<Bitmap, Long> {
    val t = System.nanoTime()
    Canvas(resized).drawBitmap(
      bitmap,
      matrix.apply { setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height) },
      paint)
    resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
    val plane = SIZE * SIZE
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
      inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
      inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    val logits = outBufs[0].readFloat()   // [19, 512, 512]

    for (i in 0 until plane) {
      var best = 0
      var bestV = logits[i]
      for (c in 1 until N_CLASS) {
        val v = logits[c * plane + i]
        if (v > bestV) {
            bestV = v
            best = c
        }
      }
      labelPixels[i] = CelebAMaskPalette.COLORS[best]
    }
    labelBitmap.setPixels(labelPixels, 0, SIZE, 0, 0, SIZE, SIZE)
    return labelBitmap to ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) {
      resized.recycle()
    }
    if (!labelBitmap.isRecycled) {
      labelBitmap.recycle()
    }
  }
}

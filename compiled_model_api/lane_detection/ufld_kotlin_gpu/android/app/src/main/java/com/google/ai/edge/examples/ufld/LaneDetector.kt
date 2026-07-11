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

package com.google.ai.edge.examples.ufld

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import kotlin.math.exp

/** A detected lane point in normalized image coords (0..1). */
data class LanePoint(val lane: Int, val x: Float, val y: Float)

/**
 * Ultra-Fast-Lane-Detection (ResNet18, CULane) on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 288, 800] NCHW, RGB, x/255 then ImageNet-normalized.
 * Output: [1, 201, 18, 4]  (griding+1, row_anchors, lanes) row-wise class logits.
 *
 * Pure CNN (ResNet18 + row-classification head) — fully on the GPU delegate; the
 * grid→lane decode (softmax + expectation per row) runs host-side.
 */
class LaneDetector(modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "UFLD"
    const val W = 800
    const val H = 288
    const val GRIDING = 200
    const val ROWS = 18
    const val LANES = 4
    private val ROW_ANCHOR = intArrayOf(
      121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287)
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  private val model = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()

  private val inputFloats = FloatArray(3 * H * W)
  private val pixels = IntArray(W * H)
  private val resized = Bitmap.createBitmap(W, H, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init { Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out") }

  fun detect(bitmap: Bitmap): Pair<List<LanePoint>, Long> {
    val t = System.nanoTime()
    Canvas(resized).drawBitmap(
      bitmap,
      matrix.apply { setScale(W.toFloat() / bitmap.width, H.toFloat() / bitmap.height) },
      paint)
    resized.getPixels(pixels, 0, W, 0, 0, W, H)
    val plane = W * H
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = (((p shr 16) and 0xFF) / 255f - MEAN[0]) / STD[0]
      inputFloats[plane + i] = (((p shr 8) and 0xFF) / 255f - MEAN[1]) / STD[1]
      inputFloats[2 * plane + i] = ((p and 0xFF) / 255f - MEAN[2]) / STD[2]
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    val o = outBufs[0].readFloat()   // [201*18*4], index = ((g*ROWS)+row)*LANES + lane

    val pts = ArrayList<LanePoint>()
    val colStep = (W - 1f) / (GRIDING - 1)
    for (lane in 0 until LANES) {
      for (row in 0 until ROWS) {
        var maxV = -Float.MAX_VALUE
        var maxIdx = 0
        for (g in 0..GRIDING) {
          val v = o[((g * ROWS) + row) * LANES + lane]
          if (v > maxV) {
              maxV = v
              maxIdx = g
          }
        }
        if (maxIdx == GRIDING) continue   // no lane at this row
        var sum = 0f
        var wsum = 0f
        for (g in 0 until GRIDING) {
          val e = exp(o[((g * ROWS) + row) * LANES + lane] - maxV)
          sum += e
          wsum += e * (g + 1)
        }
        val col = wsum / sum
        pts.add(LanePoint(lane, ((col - 1) * colStep) / W, ROW_ANCHOR[row].toFloat() / H))
      }
    }
    return pts to ((System.nanoTime() - t) / 1_000_000)
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) {
      resized.recycle()
    }
  }
}

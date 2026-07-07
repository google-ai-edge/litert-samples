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

package com.google.ai.edge.examples.sixdrepnet

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Matrix
import android.graphics.Paint
import android.util.Log
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.TensorBuffer
import kotlin.math.atan2
import kotlin.math.sqrt

/** Head pose in degrees. */
data class HeadPose(val yaw: Float, val pitch: Float, val roll: Float)

/**
 * 6DRepNet head pose estimation on LiteRT CompiledModel (GPU).
 *
 * Input : [1, 3, 224, 224] NCHW, RGB, ImageNet-normalized (a face crop).
 * Output: [1, 6] continuous 6D rotation -> Gram-Schmidt -> yaw/pitch/roll (host-side).
 *
 * RepVGG-B1g2 (deploy) — pure CNN, fully on the GPU delegate (zero patches). ~21 ms/frame.
 */
class HeadPoseEstimator(modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "6DRepNet"
    const val SIZE = 224
    private val MEAN = floatArrayOf(0.485f, 0.456f, 0.406f)
    private val STD = floatArrayOf(0.229f, 0.224f, 0.225f)
  }

  private val model = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()

  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init { Log.i(TAG, "GPU compiled OK — ${inBufs.size} in / ${outBufs.size} out") }

  fun estimate(faceCrop: Bitmap): Pair<HeadPose, Long> {
    val t = System.nanoTime()
    Canvas(resized).drawBitmap(
      faceCrop, matrix.apply { setScale(SIZE.toFloat() / faceCrop.width, SIZE.toFloat() / faceCrop.height) }, paint)
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
    val v = outBufs[0].readFloat()   // 6D

    val x = normalize(v[0], v[1], v[2])
    var z = cross(x[0], x[1], x[2], v[3], v[4], v[5]); z = normalize(z[0], z[1], z[2])
    val y = cross(z[0], z[1], z[2], x[0], x[1], x[2])
    val r00 = x[0]; val r10 = x[1]; val r20 = x[2]; val r21 = y[2]; val r22 = z[2]
    val sy = sqrt(r00 * r00 + r10 * r10)
    val pitch = Math.toDegrees(atan2(r21, r22).toDouble()).toFloat()
    val yaw = Math.toDegrees(atan2(-r20, sy).toDouble()).toFloat()
    val roll = Math.toDegrees(atan2(r10, r00).toDouble()).toFloat()
    return HeadPose(yaw, pitch, roll) to ((System.nanoTime() - t) / 1_000_000)
  }

  private fun normalize(a: Float, b: Float, c: Float): FloatArray {
    val n = sqrt(a * a + b * b + c * c) + 1e-8f
    return floatArrayOf(a / n, b / n, c / n)
  }

  private fun cross(a0: Float, a1: Float, a2: Float, b0: Float, b1: Float, b2: Float): FloatArray =
    floatArrayOf(a1 * b2 - a2 * b1, a2 * b0 - a0 * b2, a0 * b1 - a1 * b0)

  override fun close() {
    model.close()
    if (!resized.isRecycled) resized.recycle()
  }
}

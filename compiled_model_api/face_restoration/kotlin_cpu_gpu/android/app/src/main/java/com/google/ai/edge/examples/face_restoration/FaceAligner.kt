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

package com.google.ai.edge.examples.face_restoration

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import kotlin.math.abs

/**
 * FFHQ face alignment: warp the detected face to the standard 512x512 template GFPGAN was trained on
 * (eyes/nose/mouth at fixed positions). Without this, GFPGAN's StyleGAN prior mangles the mouth on
 * off-template crops. Uses the 5 YuNet landmarks and a least-squares similarity transform.
 */
object FaceAligner {

  // facexlib's 512x512 5-point template: [right eye, left eye, nose, right mouth, left mouth]
  // (image-left to image-right), matching YuNet's landmark order.
  private val TEMPLATE = floatArrayOf(
    192.98138f, 239.94708f,
    318.90277f, 240.19366f,
    256.63416f, 314.01935f,
    201.26117f, 371.41043f,
    313.08905f, 371.15118f,
  )

  /** Warp [src] so its [landmarks] (in src pixel coords, 5×(x,y)) land on the template. -> 512x512. */
  fun align(src: Bitmap, landmarks: FloatArray): Bitmap {
    val m = similarity(landmarks, TEMPLATE)
    val out = Bitmap.createBitmap(512, 512, Bitmap.Config.ARGB_8888)
    val matrix = Matrix().apply { setValues(floatArrayOf(m[0], m[1], m[2], m[3], m[4], m[5], 0f, 0f, 1f)) }
    Canvas(out).apply {
      drawColor(Color.rgb(135, 133, 132)) // facexlib neutral fill for out-of-image border
      drawBitmap(src, matrix, Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG))
    }
    return out
  }

  /**
   * Least-squares 2D similarity (rotation + uniform scale + translation) mapping src->dst.
   * Returns the 2x3 affine [a, -b, tx, b, a, ty]. Solves the 4-param normal equations (linear in
   * a,b,tx,ty) — no SVD needed.
   */
  private fun similarity(src: FloatArray, dst: FloatArray): FloatArray {
    val n = src.size / 2
    var sd = 0.0
    var sx = 0.0
    var sy = 0.0
    var c0 = 0.0
    var c1 = 0.0
    var c2 = 0.0
    var c3 = 0.0
    for (i in 0 until n) {
      val x = src[2 * i].toDouble()
      val y = src[2 * i + 1].toDouble()
      val bx = dst[2 * i].toDouble()
      val by = dst[2 * i + 1].toDouble()
      sd += x * x + y * y
      sx += x
      sy += y
      c0 += x * bx + y * by
      c1 += x * by - y * bx
      c2 += bx
      c3 += by
    }
    // Normal matrix A^T A (from rows [x,-y,1,0] and [y,x,0,1]):
    val a = arrayOf(
      doubleArrayOf(sd, 0.0, sx, sy),
      doubleArrayOf(0.0, sd, -sy, sx),
      doubleArrayOf(sx, -sy, n.toDouble(), 0.0),
      doubleArrayOf(sy, sx, 0.0, n.toDouble()),
    )
    val b = doubleArrayOf(c0, c1, c2, c3)
    val p = solve4(a, b) // [a, b, tx, ty]
    return floatArrayOf(
      p[0].toFloat(), (-p[1]).toFloat(), p[2].toFloat(),
      p[1].toFloat(), p[0].toFloat(), p[3].toFloat(),
    )
  }

  /** Gaussian elimination with partial pivoting for a 4x4 system. */
  private fun solve4(a: Array<DoubleArray>, b: DoubleArray): DoubleArray {
    val n = 4
    for (col in 0 until n) {
      var piv = col
      for (r in col + 1 until n) if (abs(a[r][col]) > abs(a[piv][col])) piv = r
      val tmp = a[col]
      a[col] = a[piv]
      a[piv] = tmp
      val tb = b[col]
      b[col] = b[piv]
      b[piv] = tb
      val d = a[col][col]
      for (r in 0 until n) {
        if (r == col) continue
        val f = a[r][col] / d
        for (c in col until n) a[r][c] -= f * a[col][c]
        b[r] -= f * b[col]
      }
    }
    return DoubleArray(n) { b[it] / a[it][it] }
  }
}

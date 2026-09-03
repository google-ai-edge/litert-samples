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

package com.google.ai.edge.examples.face_alignment_3d

import android.content.Context
import android.graphics.Bitmap
import android.media.FaceDetector
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * 3DDFA_V2 3D dense face alignment on the LiteRT CompiledModel GPU. A face box (from Android's
 * built-in FaceDetector) is cropped to 120², a MobileNetV1 regresses 62 3DMM parameters on the
 * GPU, and the 68 3D landmarks are reconstructed host-side from the BFM bases
 * (u + w_shp·α_shp + w_exp·α_exp), posed by R·v + offset and mapped back to image coordinates.
 */
class TddfaLandmarks(private val ctx: Context) {

  companion object {
      const val SIZE = 120
      const val NP = 62
      const val NK = 68
  }

  private val model = CompiledModel.create(
    ctx.assets, "tddfa_mb1_fp16.tflite", CompiledModel.Options(Accelerator.GPU), null)
  private val inBuf = model.createInputBuffers()
  private val outBuf = model.createOutputBuffers()

  private fun asset(n: String): FloatArray {
    val b = ctx.assets.open(n).readBytes()
    val bb = ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(b.size / 4) { bb.float }
  }
  private val uBase = asset("tddfa_u_base.bin")
  private val wShp = asset("tddfa_w_shp_base.bin")
  private val wExp = asset("tddfa_w_exp_base.bin")
  private val pMean = asset("tddfa_param_mean.bin")
  private val pStd = asset("tddfa_param_std.bin")

  /** Face box via android.media.FaceDetector (frontal faces), or null. Returns [l,t,r,b]. */
  fun detectFace(bm: Bitmap): FloatArray? {
    val w0 = bm.width - (bm.width % 2)                       // FaceDetector needs an even width
    val scaled = if (w0 == bm.width) bm else Bitmap.createScaledBitmap(bm, w0, bm.height, true)
    val rgb565 = scaled.copy(Bitmap.Config.RGB_565, false)
    val faces = arrayOfNulls<FaceDetector.Face>(1)
    val n = try {
      FaceDetector(rgb565.width, rgb565.height, 1).findFaces(rgb565, faces)
    } catch (e: Throwable) {
      0
    }
    rgb565.recycle()
    if (n == 0 || faces[0] == null) return null
    val f = faces[0]!!
    val mid = android.graphics.PointF()
    f.getMidPoint(mid)
    val w = f.eyesDistance() * 2.2f
    return floatArrayOf(mid.x - w / 2f, mid.y - w * 0.6f, mid.x + w / 2f, mid.y + w * 0.9f)
  }

  /** 3DDFA parse_roi_box_from_bbox: expand + recenter to a square ROI. */
  private fun roiBox(b: FloatArray): FloatArray {
    val (l, t, r, bo) = b
    val old = (r - l + bo - t) / 2f
    val cx = r - (r - l) / 2f
    val cy = bo - (bo - t) / 2f + old * 0.14f
    val size = old * 1.58f
    return floatArrayOf(
      cx - size / 2f, cy - size / 2f, cx - size / 2f + size, cy - size / 2f + size)
  }

  /** Crop the ROI to 120² NCHW BGR (the model was trained on cv2 BGR), (x-127.5)/128, 0-pad OOB. */
  private fun cropInput(bm: Bitmap, roi: FloatArray): FloatArray {
    val (sx, sy, ex, ey) = roi
    val px = IntArray(bm.width * bm.height)
    bm.getPixels(px, 0, bm.width, 0, 0, bm.width, bm.height)
    val out = FloatArray(3 * SIZE * SIZE)
    val plane = SIZE * SIZE
    for (oy in 0 until SIZE) for (ox in 0 until SIZE) {
      val srcX = (sx + (ox + 0.5f) * (ex - sx) / SIZE).toInt()
      val srcY = (sy + (oy + 0.5f) * (ey - sy) / SIZE).toInt()
      var r = 0
      var g = 0
      var b = 0
      if (srcX in 0 until bm.width && srcY in 0 until bm.height) {
        val p = px[srcY * bm.width + srcX]
        r = p shr 16 and 0xFF
        g = p shr 8 and 0xFF
        b = p and 0xFF
      }
      val i = oy * SIZE + ox
      out[i] = (b - 127.5f) / 128f
      out[plane + i] = (g - 127.5f) / 128f
      out[2 * plane + i] = (r - 127.5f) / 128f
    }
    return out
  }

  /** 68 landmarks as (x,y) pairs in image coordinates, or null if no face. */
  fun landmarks(bm: Bitmap): FloatArray? {
    val box = detectFace(bm) ?: return null
    val roi = roiBox(box)
    inBuf[0].writeFloat(cropInput(bm, roi))
    model.run(inBuf, outBuf)
    val out = outBuf[0].readFloat()
    val p = FloatArray(NP) { out[it] * pStd[it] + pMean[it] }
    val R = Array(3) { i -> FloatArray(3) { j -> p[i * 4 + j] } }
    val off = FloatArray(3) { p[it * 4 + 3] }
    val shp = FloatArray(204)
    for (rI in 0 until 204) {
      var v = uBase[rI]
      for (j in 0 until 40) {
        v += wShp[rI * 40 + j] * p[12 + j]
      }
      for (k in 0 until 10) {
        v += wExp[rI * 10 + k] * p[52 + k]
      }
      shp[rI] = v
    }
    val (sx, sy, ex, ey) = roi
    val scaleX = (ex - sx) / SIZE
    val scaleY = (ey - sy) / SIZE
    val lm = FloatArray(NK * 2)
    for (n in 0 until NK) {
      // interleaved [x0,y0,z0,x1,...] — the reference reconstructs with reshape(3,-1, order='F')
      val vx = shp[3 * n]
      val vy = shp[3 * n + 1]
      val vz = shp[3 * n + 2]
      var x = R[0][0] * vx + R[0][1] * vy + R[0][2] * vz + off[0]
      var y = R[1][0] * vx + R[1][1] * vy + R[1][2] * vz + off[1]
      x -= 1f
      y = SIZE - y
      lm[n * 2] = x * scaleX + sx
      lm[n * 2 + 1] = y * scaleY + sy
    }
    return lm
  }

  fun close() = model.close()
  private operator fun FloatArray.component1() = this[0]
  private operator fun FloatArray.component2() = this[1]
  private operator fun FloatArray.component3() = this[2]
  private operator fun FloatArray.component4() = this[3]
}

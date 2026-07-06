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

package com.google.ai.edge.examples.yolact

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

data class Instance(val cls: Int, val score: Float, val x1: Float, val y1: Float,
                    val x2: Float, val y2: Float, val mask: BooleanArray)

object Palette {
  private val C = IntArray(80) { Color.HSVToColor(floatArrayOf((it * 47 % 360).toFloat(), 0.75f, 1f)) }
  fun color(cls: Int) = C[cls % 80]
}

/**
 * YOLACT-ResNet50 real-time instance segmentation on LiteRT CompiledModel (GPU).
 *
 * The GPU graph (ResNet50 + FPN + protonet + heads) emits raw outputs — loc, conf,
 * mask coefficients, prototype masks — and the decode (SSD box-decode vs the baked
 * priors, per-class NMS, linear-combination masks) runs host-side. The whole network
 * runs on the GPU delegate (base YOLACT is a pure CNN).
 */
class YolactSegmenter(context: Context, modelPath: String) : AutoCloseable {

  companion object {
    private const val TAG = "YOLACT"
    const val SIZE = 550
    const val N = 19248; const val NC = 81; const val K = 32; const val PS = 138
    private val MEAN = floatArrayOf(103.94f, 116.78f, 123.68f)   // BGR
    private val STD = floatArrayOf(57.38f, 57.12f, 58.40f)
    private const val SCORE = 0.3f; private const val IOU = 0.5f; private const val MAX_DET = 40
  }

  private val model = CompiledModel.create(modelPath, CompiledModel.Options(Accelerator.GPU), null)
  private val inBufs: List<TensorBuffer> = model.createInputBuffers()
  private val outBufs: List<TensorBuffer> = model.createOutputBuffers()
  private val priors: FloatArray
  private var iLoc = 0; private var iConf = 1; private var iMask = 2; private var iProto = 3

  private val inputFloats = FloatArray(3 * SIZE * SIZE)
  private val pixels = IntArray(SIZE * SIZE)
  private val resized = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
  private val matrix = Matrix()
  private val paint = Paint(Paint.FILTER_BITMAP_FLAG)

  init {
    for (i in outBufs.indices) when (outBufs[i].readFloat().size) {
      N * 4 -> iLoc = i; N * NC -> iConf = i; N * K -> iMask = i; PS * PS * K -> iProto = i
    }
    val b = context.assets.open("priors.bin").use { it.readBytes() }
    val fb = java.nio.ByteBuffer.wrap(b).order(java.nio.ByteOrder.LITTLE_ENDIAN).asFloatBuffer()
    priors = FloatArray(fb.limit()).also { fb.get(it) }
    Log.i(TAG, "GPU compiled OK — loc=$iLoc conf=$iConf mask=$iMask proto=$iProto, priors=${priors.size / 4}")
  }

  fun segment(bitmap: Bitmap): Pair<List<Instance>, Long> {
    val t = System.nanoTime()
    Canvas(resized).drawBitmap(
      bitmap, matrix.apply { setScale(SIZE.toFloat() / bitmap.width, SIZE.toFloat() / bitmap.height) }, paint)
    resized.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
    val plane = SIZE * SIZE
    for (i in 0 until plane) {
      val p = pixels[i]
      inputFloats[i] = ((p and 0xFF) - MEAN[0]) / STD[0]
      inputFloats[plane + i] = (((p shr 8) and 0xFF) - MEAN[1]) / STD[1]
      inputFloats[2 * plane + i] = (((p shr 16) and 0xFF) - MEAN[2]) / STD[2]
    }
    inBufs[0].writeFloat(inputFloats)
    model.run(inBufs, outBufs)
    val loc = outBufs[iLoc].readFloat(); val conf = outBufs[iConf].readFloat()
    val mask = outBufs[iMask].readFloat(); val proto = outBufs[iProto].readFloat()

    data class Cand(val a: Int, val cls: Int, val sc: Float, val box: FloatArray)
    val cands = ArrayList<Cand>()
    for (a in 0 until N) {
      var best = 1; var bv = conf[a * NC + 1]
      for (c in 2 until NC) { val v = conf[a * NC + c]; if (v > bv) { bv = v; best = c } }
      if (bv < SCORE) continue
      val pcx = priors[a * 4]; val pcy = priors[a * 4 + 1]; val pw = priors[a * 4 + 2]; val ph = priors[a * 4 + 3]
      val cx = pcx + loc[a * 4] * 0.1f * pw; val cy = pcy + loc[a * 4 + 1] * 0.1f * ph
      val w = pw * Math.exp((loc[a * 4 + 2] * 0.2f).toDouble()).toFloat()
      val h = ph * Math.exp((loc[a * 4 + 3] * 0.2f).toDouble()).toFloat()
      cands.add(Cand(a, best - 1, bv, floatArrayOf(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    }
    cands.sortByDescending { it.sc }
    val kept = ArrayList<Cand>(); val removed = BooleanArray(cands.size)
    for (i in cands.indices) {
      if (removed[i]) continue
      kept.add(cands[i]); if (kept.size >= MAX_DET) break
      for (j in i + 1 until cands.size)
        if (!removed[j] && cands[j].cls == cands[i].cls && iou(cands[i].box, cands[j].box) > IOU) removed[j] = true
    }
    val out = ArrayList<Instance>(kept.size)
    for (d in kept) {
      val coeff = FloatArray(K) { mask[d.a * K + it] }
      val m = BooleanArray(SIZE * SIZE)
      val px1 = (d.box[0] * SIZE).toInt().coerceIn(0, SIZE); val py1 = (d.box[1] * SIZE).toInt().coerceIn(0, SIZE)
      val px2 = (d.box[2] * SIZE).toInt().coerceIn(0, SIZE); val py2 = (d.box[3] * SIZE).toInt().coerceIn(0, SIZE)
      for (yy in py1 until py2) {
        val protoY = yy * PS / SIZE
        for (xx in px1 until px2) {
          val base = (protoY * PS + xx * PS / SIZE) * K
          var s = 0f; for (k in 0 until K) s += proto[base + k] * coeff[k]
          if (s > 0f) m[yy * SIZE + xx] = true
        }
      }
      out.add(Instance(d.cls, d.sc, d.box[0], d.box[1], d.box[2], d.box[3], m))
    }
    return out to ((System.nanoTime() - t) / 1_000_000)
  }

  private fun iou(a: FloatArray, b: FloatArray): Float {
    val x1 = maxOf(a[0], b[0]); val y1 = maxOf(a[1], b[1]); val x2 = minOf(a[2], b[2]); val y2 = minOf(a[3], b[3])
    val inter = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
    val ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return if (ua <= 0f) 0f else inter / ua
  }

  override fun close() {
    model.close()
    if (!resized.isRecycled) resized.recycle()
  }
}

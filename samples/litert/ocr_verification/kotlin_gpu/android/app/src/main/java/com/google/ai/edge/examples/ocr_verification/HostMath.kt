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

package com.google.ai.edge.examples.ocr_verification

import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * The scoring host's arithmetic, ported from the Python host of
 * litert-community/MiniCPM-OCR-Decision-2B-LiteRT (`host/decide_host/picture.py` and
 * `formulas.py`), which follows the source model's preprocessing (openbmb/MiniCPM-OCR-Jev,
 * Qwen3.5). Every float operation keeps the Python host's order and precision, so the same
 * picture and text give the same graph inputs: the unit tests and the app's check mode compare
 * them with the Python host.
 */
object HostMath {
  const val FACTOR = 32
  const val MIN_PIXELS = 65_536L
  const val MAX_PIXELS = 16_777_216L
  const val PATCH = 16
  const val MERGE = 2
  const val TEMPORAL = 2
  const val PATCH_DIM = 3 * TEMPORAL * PATCH * PATCH
  const val POS_DIM = 1024
  const val ROT_DIM = 64
  const val HIDDEN = 2048
  const val SIDE = 48
  const val MAX_ASPECT = 199.0

  const val IMAGE_PAD = 248_056
  const val VIDEO_PAD = 248_057
  const val POOLED = 271

  /** The fixed instruction of the source's prompt (the model was trained with it, in Chinese). */
  const val INSTRUCTION =
    "\u7b2c\u4e00\u5f20\u56fe\u662f\u539f\u56fe\uff0c\u7b2c\u4e8c\u5f20\u56fe\u662f\u4e0a\u8ff0 OCR " +
      "\u6587\u672c\u6e32\u67d3\u5f97\u5230\u7684\u56fe\u7247\u3002\n" +
      "\u8bf7\u5224\u65ad\u8be5 OCR \u6587\u672c\u53ca\u5176\u5bf9\u5e94\u6e32\u67d3\u56fe\u662f\u5426" +
      "\u4e0e\u539f\u56fe\u5185\u5bb9\u4e00\u81f4\u3002\n" +
      "\u8fd9\u662f\u4e00\u4e2a\u4e8c\u5206\u7c7b\u4efb\u52a1\uff1a\u4e00\u81f4\u6807\u4e3a 1\uff0c" +
      "\u4e0d\u4e00\u81f4\u6807\u4e3a 0\u3002"
  const val ASSISTANT_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

  /** The text rotary's inverse frequencies: numpy's float32 values (`np.power` in float32). */
  private val TEXT_INV_FREQ =
    floats(
      0x3f800000, 0x3f1ab32b, 0x3ebaf81b, 0x3e61f835, 0x3e088d77, 0x3da50956, 0x3d47763f,
      0x3cf11177, 0x3c91ad39, 0x3c301052, 0x3bd4ca15, 0x3b80967d, 0x3b1b690d, 0x3abbd3ed,
      0x3a6301e2, 0x3a092e02, 0x39a5cb60, 0x394860c1, 0x38f22ce2, 0x3892587e, 0x3830df52,
      0x37d5c441, 0x37812dab, 0x371c1fc4, 0x36bcb0c1, 0x36640cc6, 0x3609cf4a, 0x35a68e4c,
      0x35494c57, 0x34f3499d, 0x3493048e, 0x3431af44,
    )

  /** The picture rotary's inverse frequencies (theta 10000, 32 dimensions), numpy's float32. */
  private val VISION_INV_FREQ =
    floats(
      0x3f800000, 0x3f0ff59a, 0x3ea1e89b, 0x3e361887, 0x3dcccccd, 0x3d6655c2, 0x3d0186e3,
      0x3c91ad39, 0x3c23d70a, 0x3bb8449c, 0x3b4f3e38, 0x3ae91528, 0x3a83126f, 0x3a136a16,
      0x39a5cb60, 0x393a7753,
    )

  private fun floats(vararg bits: Long) = FloatArray(bits.size) { Float.fromBits(bits[it].toInt()) }

  /** An 8-bit RGB picture, rows top to bottom, R G B per pixel. */
  class Rgb(val bytes: ByteArray, val width: Int, val height: Int) {
    init {
      require(bytes.size == width * height * 3) { "RGB bytes do not match $width x $height" }
    }
  }

  /** One picture as the tower reads it: n patches of a gh x gw patch grid. */
  class Picture(
    val gh: Int,
    val gw: Int,
    /** [n, 1536] */
    val patches: FloatArray,
    /** [n, 1024] */
    val pos: FloatArray,
    /** [n, 64] */
    val cos: FloatArray,
    /** [n, 64] */
    val sin: FloatArray,
    /** The resized 8-bit picture the patches were cut from. */
    val resized: Rgb,
  ) {
    val n: Int
      get() = gh * gw

    /** Merged tokens of the picture in the prompt. */
    val tokens: Int
      get() = n / 4
  }

  // ---------------------------------------------------------------- picture

  /**
   * The source's pad of an extreme aspect ratio: above 199, the picture is centred on a white
   * canvas.
   */
  fun padAspect(rgb: Rgb): Rgb {
    val w = rgb.width
    val h = rgb.height
    if (max(w, h).toDouble() / min(w, h) <= MAX_ASPECT) return rgb
    val nw: Int
    val nh: Int
    if (w >= h) {
      nw = w
      nh = max(h, (w / MAX_ASPECT + 0.999999).toInt())
    } else {
      nw = max(w, (h / MAX_ASPECT + 0.999999).toInt())
      nh = h
    }
    val out = ByteArray(nw * nh * 3) { 0xFF.toByte() }
    val x0 = (nw - w) / 2
    val y0 = (nh - h) / 2
    for (y in 0 until h) {
      System.arraycopy(rgb.bytes, y * w * 3, out, ((y0 + y) * nw + x0) * 3, w * 3)
    }
    return Rgb(out, nw, nh)
  }

  /** qwen2_vl smart_resize with factor 32 and the checkpoint's pixel budget: [height, width]. */
  fun smartResize(height: Int, width: Int): IntArray {
    require(max(height, width).toDouble() / min(height, width) <= 200.0) { "aspect ratio over 200" }
    var hBar = Math.rint(height.toDouble() / FACTOR).toInt() * FACTOR
    var wBar = Math.rint(width.toDouble() / FACTOR).toInt() * FACTOR
    if (hBar.toLong() * wBar > MAX_PIXELS) {
      val beta = sqrt(height.toDouble() * width / MAX_PIXELS)
      hBar = max(FACTOR, floor(height / beta / FACTOR).toInt() * FACTOR)
      wBar = max(FACTOR, floor(width / beta / FACTOR).toInt() * FACTOR)
    } else if (hBar.toLong() * wBar < MIN_PIXELS) {
      val beta = sqrt(MIN_PIXELS.toDouble() / (height.toDouble() * width))
      hBar = ceil(height * beta / FACTOR).toInt() * FACTOR
      wBar = ceil(width * beta / FACTOR).toInt() * FACTOR
    }
    return intArrayOf(hBar, wBar)
  }

  private class AxisWeights(
    val xmin: IntArray,
    val w16: Array<IntArray>,
    val k: Int,
    val precision: Int,
  )

  private fun keysCubic(value: Double): Double {
    val a = -0.5
    val x = Math.abs(value)
    return when {
      x < 1.0 -> ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
      x < 2.0 -> (((x - 5.0) * x + 8.0) * x - 4.0) * a
      else -> 0.0
    }
  }

  /**
   * PyTorch's uint8 antialiased bicubic weights for one axis (aten UpsampleKernel.cpp, a port of
   * Pillow's Resample.c with int16 weights), as the Python host computes them.
   */
  private fun axisWeights(inSize: Int, outSize: Int): AxisWeights {
    val scale = inSize.toDouble() / outSize
    val fscale = max(scale, 1.0)
    val support = 2.0 * fscale
    val invscale = 1.0 / fscale
    val k = ceil(support).toInt() * 2 + 1
    val xmin = IntArray(outSize)
    val w = Array(outSize) { DoubleArray(k) }
    var maxW = Double.NEGATIVE_INFINITY
    for (i in 0 until outSize) {
      val center = scale * (i + 0.5)
      val lo = max((center - support + 0.5).toLong(), 0L)
      val hi = min((center + support + 0.5).toLong(), inSize.toLong())
      val size = (hi - lo).coerceIn(0L, k.toLong()).toInt()
      xmin[i] = lo.toInt()
      val row = w[i]
      for (j in 0 until k) {
        row[j] = if (j < size) keysCubic((j + lo.toDouble() - center + 0.5) * invscale) else 0.0
      }
      // float64 running sum in index order, as the kernel
      var total = 0.0
      for (j in 0 until k) total += row[j]
      if (total != 0.0) {
        for (j in 0 until k) row[j] = row[j] / total
      }
      for (j in 0 until k) {
        if (row[j] > maxW) maxW = row[j]
      }
    }
    var p = 0
    while (p < 22) {
      if ((0.5 + maxW * (1L shl (p + 1))).toLong() >= (1L shl 15)) break
      p++
    }
    val scaleP = (1L shl p).toDouble()
    val w16 =
      Array(outSize) { i ->
        IntArray(k) { j ->
          val v = w[i][j] * scaleP
          (if (v < 0) v - 0.5 else v + 0.5).toLong().toShort().toInt()
        }
      }
    return AxisWeights(xmin, w16, k, p)
  }

  private fun resizeWidth(src: ByteArray, height: Int, inW: Int, outW: Int): ByteArray {
    val aw = axisWeights(inW, outW)
    val out = ByteArray(height * outW * 3)
    val half = 1 shl (aw.precision - 1)
    for (y in 0 until height) {
      val rowIn = y * inW * 3
      val rowOut = y * outW * 3
      for (i in 0 until outW) {
        val ws = aw.w16[i]
        val x0 = aw.xmin[i]
        var r = half
        var g = half
        var b = half
        for (j in 0 until aw.k) {
          val wj = ws[j]
          if (wj == 0) continue
          val o = rowIn + min(x0 + j, inW - 1) * 3
          r += (src[o].toInt() and 0xFF) * wj
          g += (src[o + 1].toInt() and 0xFF) * wj
          b += (src[o + 2].toInt() and 0xFF) * wj
        }
        val o = rowOut + i * 3
        out[o] = clip8(r shr aw.precision)
        out[o + 1] = clip8(g shr aw.precision)
        out[o + 2] = clip8(b shr aw.precision)
      }
    }
    return out
  }

  private fun resizeHeight(src: ByteArray, inH: Int, width: Int, outH: Int): ByteArray {
    val aw = axisWeights(inH, outH)
    val out = ByteArray(outH * width * 3)
    val half = 1 shl (aw.precision - 1)
    val acc = IntArray(width * 3)
    for (i in 0 until outH) {
      java.util.Arrays.fill(acc, half)
      val ws = aw.w16[i]
      val y0 = aw.xmin[i]
      for (j in 0 until aw.k) {
        val wj = ws[j]
        if (wj == 0) continue
        val rowIn = min(y0 + j, inH - 1) * width * 3
        for (x in 0 until width * 3) {
          acc[x] += (src[rowIn + x].toInt() and 0xFF) * wj
        }
      }
      val rowOut = i * width * 3
      for (x in 0 until width * 3) {
        out[rowOut + x] = clip8(acc[x] shr aw.precision)
      }
    }
    return out
  }

  private fun clip8(v: Int): Byte = (if (v < 0) 0 else if (v > 255) 255 else v).toByte()

  /** uint8 RGB -> uint8 RGB, PyTorch's uint8 antialiased bicubic, width first, then height. */
  fun resize(rgb: Rgb, height: Int, width: Int): Rgb {
    var bytes = rgb.bytes
    var w = rgb.width
    if (w != width) {
      bytes = resizeWidth(bytes, rgb.height, w, width)
      w = width
    }
    if (rgb.height != height) {
      bytes = resizeHeight(bytes, rgb.height, w, height)
    }
    return Rgb(bytes, width, height)
  }

  /**
   * uint8 [H, W, 3] with H and W multiples of 32 -> float32 [n, 1536] in the processor's order:
   * merge-block-major patches, each patch twice along the temporal axis, (u8 - 127.5) / 127.5.
   */
  fun patches(rgb: Rgb): FloatArray {
    val w = rgb.width
    val gh = rgb.height / PATCH
    val gw = w / PATCH
    val out = FloatArray(gh * gw * PATCH_DIM)
    var idx = 0
    for (bh in 0 until gh / MERGE) for (bw in 0 until gw / MERGE) {
      for (m1 in 0 until MERGE) for (m2 in 0 until MERGE) {
        val pr = bh * MERGE + m1
        val pc = bw * MERGE + m2
        for (c in 0 until 3) for (t in 0 until TEMPORAL) {
          for (py in 0 until PATCH) {
            val rowBase = ((pr * PATCH + py) * w + pc * PATCH) * 3 + c
            for (px in 0 until PATCH) {
              val u = rgb.bytes[rowBase + px * 3].toInt() and 0xFF
              out[idx++] = (u.toFloat() - 127.5f) / 127.5f
            }
          }
        }
      }
    }
    return out
  }

  /** (row, col) of each patch in merge-block-major order. */
  private inline fun forEachPatch(gh: Int, gw: Int, body: (index: Int, row: Int, col: Int) -> Unit) {
    var i = 0
    for (bh in 0 until gh / MERGE) for (bw in 0 until gw / MERGE) {
      for (m1 in 0 until MERGE) for (m2 in 0 until MERGE) {
        body(i++, bh * MERGE + m1, bw * MERGE + m2)
      }
    }
  }

  /**
   * cos, sin float32 [n, 64]: emb = cat(rot, rot), rot = (row, col) times the inverse frequencies;
   * the float32 angle's cosine in float64, rounded to float32.
   */
  fun visionRotary(gh: Int, gw: Int): Pair<FloatArray, FloatArray> {
    val n = gh * gw
    val cos = FloatArray(n * ROT_DIM)
    val sin = FloatArray(n * ROT_DIM)
    val half = VISION_INV_FREQ.size
    val rot = FloatArray(2 * half)
    forEachPatch(gh, gw) { i, row, col ->
      for (f in 0 until half) {
        rot[f] = row.toFloat() * VISION_INV_FREQ[f]
        rot[half + f] = col.toFloat() * VISION_INV_FREQ[f]
      }
      val base = i * ROT_DIM
      for (d in 0 until ROT_DIM) {
        val a = rot[d % (2 * half)].toDouble()
        cos[base + d] = StrictMath.cos(a).toFloat()
        sin[base + d] = StrictMath.sin(a).toFloat()
      }
    }
    return cos to sin
  }

  /**
   * torch.linspace(0, end, steps) in float32: i * step on the first half, end - (steps - 1 - i) *
   * step on the second.
   */
  private fun linspace(end: Double, steps: Int): FloatArray {
    if (steps == 1) return floatArrayOf(0f)
    val step = (end / (steps - 1)).toFloat().toDouble()
    val half = steps / 2
    return FloatArray(steps) { i ->
      (if (i < half) i.toDouble() * step else end - (steps - 1 - i).toDouble() * step).toFloat()
    }
  }

  /**
   * The bilinear position rows of the 48 x 48 position table for one picture, float32 [n, 1024]:
   * (table[idx] * w) summed over the four corners in order.
   */
  fun posRows(table: FloatArray, gh: Int, gw: Int): FloatArray {
    val hg = linspace((SIDE - 1).toDouble(), gh)
    val wg = linspace((SIDE - 1).toDouble(), gw)
    val hf = IntArray(gh) { hg[it].toInt() }
    val wf = IntArray(gw) { wg[it].toInt() }
    val hc = IntArray(gh) { min(hf[it] + 1, SIDE - 1) }
    val wc = IntArray(gw) { min(wf[it] + 1, SIDE - 1) }
    val hr = FloatArray(gh) { hg[it] - hf[it].toFloat() }
    val wr = FloatArray(gw) { wg[it] - wf[it].toFloat() }
    val out = FloatArray(gh * gw * POS_DIM)
    val one = 1f
    forEachPatch(gh, gw) { i, y, x ->
      val t0 = (hf[y] * SIDE + wf[x]) * POS_DIM
      val t1 = (hf[y] * SIDE + wc[x]) * POS_DIM
      val t2 = (hc[y] * SIDE + wf[x]) * POS_DIM
      val t3 = (hc[y] * SIDE + wc[x]) * POS_DIM
      val w0 = (one - hr[y]) * (one - wr[x])
      val w1 = (one - hr[y]) * wr[x]
      val w2 = hr[y] * (one - wr[x])
      val w3 = hr[y] * wr[x]
      val base = i * POS_DIM
      for (d in 0 until POS_DIM) {
        var o = table[t0 + d] * w0
        o += table[t1 + d] * w1
        o += table[t2 + d] * w2
        o += table[t3 + d] * w3
        out[base + d] = o
      }
    }
    return out
  }

  /** The whole picture path: pad, resize, patches, position rows and rotary tables. */
  fun picture(rgb: Rgb, posTable: FloatArray): Picture {
    val padded = padAspect(rgb)
    val hw = smartResize(padded.height, padded.width)
    val resized = resize(padded, hw[0], hw[1])
    val gh = hw[0] / PATCH
    val gw = hw[1] / PATCH
    val (cos, sin) = visionRotary(gh, gw)
    return Picture(gh, gw, patches(resized), posRows(posTable, gh, gw), cos, sin, resized)
  }

  // ---------------------------------------------------------------- text side

  /**
   * The source's prompt with the two picture placeholders expanded (m = merged tokens of each
   * picture): the original picture, the OCR text, the rendered picture, the instruction.
   */
  fun buildPrompt(text: String, m1: Int, m2: Int): String {
    val sb = StringBuilder(64 + 13 * (m1 + m2) + text.length + 256)
    sb.append("<|im_start|>user\n<|vision_start|>")
    repeat(m1) { sb.append("<|image_pad|>") }
    sb.append("<|vision_end|>\nOCR\u6587\u672c\uff1a\n").append(text).append("\n<|vision_start|>")
    repeat(m2) { sb.append("<|image_pad|>") }
    sb.append("<|vision_end|>\n").append(INSTRUCTION).append(ASSISTANT_SUFFIX)
    return sb.toString()
  }

  /**
   * The M-RoPE position planes [3][T] of one record (images only): text runs count up on all
   * three planes; a run of image pads takes t = current, h and w over its merged grid, then the
   * current position moves by max(H, W).
   */
  fun mropePlanes(ids: IntArray, grids: List<IntArray>): Array<LongArray> {
    val t = ids.size
    val pos = Array(3) { LongArray(t) }
    var i = 0
    var cur = 0L
    var g = 0
    while (i < t) {
      val img = ids[i] == IMAGE_PAD
      var j = i
      while (j < t && (ids[j] == IMAGE_PAD) == img) j++
      val n = j - i
      if (img) {
        val hh = grids[g][0] / MERGE
        val ww = grids[g][1] / MERGE
        g++
        check(hh * ww == n) { "image run of $n tokens for a $hh x $ww grid" }
        for (k in 0 until n) {
          pos[0][i + k] = cur
          pos[1][i + k] = cur + k / ww
          pos[2][i + k] = cur + k % ww
        }
        cur += max(hh, ww)
      } else {
        for (k in 0 until n) {
          val v = cur + k
          pos[0][i + k] = v
          pos[1][i + k] = v
          pos[2][i + k] = v
        }
        cur += n
      }
      i = j
    }
    return pos
  }

  /** [3][L]: the record's planes, then last + 1, + 2, ... on all planes (pads, never read). */
  fun paddedPlanes(pos: Array<LongArray>, length: Int): Array<LongArray> {
    val n = pos[0].size
    var top = Long.MIN_VALUE
    for (p in pos) for (v in p) top = max(top, v)
    return Array(3) { plane ->
      LongArray(length) { k -> if (k < n) pos[plane][k] else top + 1 + (k - n) }
    }
  }

  /**
   * cos, sin float32 [L, 64] of the interleaved M-RoPE (sections 11, 11, 10): the float32 angle's
   * cosine in float64, rounded to float32.
   */
  fun mropeTables(pos: Array<LongArray>): Pair<FloatArray, FloatArray> {
    val length = pos[0].size
    val half = TEXT_INV_FREQ.size
    val cos = FloatArray(length * ROT_DIM)
    val sin = FloatArray(length * ROT_DIM)
    val a = FloatArray(half)
    for (t in 0 until length) {
      val p0 = pos[0][t].toFloat()
      val p1 = pos[1][t].toFloat()
      val p2 = pos[2][t].toFloat()
      for (f in 0 until half) a[f] = p0 * TEXT_INV_FREQ[f]
      var f = 1
      while (f < 33) {
        a[f] = p1 * TEXT_INV_FREQ[f]
        f += 3
      }
      f = 2
      while (f < 30) {
        a[f] = p2 * TEXT_INV_FREQ[f]
        f += 3
      }
      val base = t * ROT_DIM
      for (d in 0 until ROT_DIM) {
        val v = a[d % half].toDouble()
        cos[base + d] = StrictMath.cos(v).toFloat()
        sin[base + d] = StrictMath.sin(v).toFloat()
      }
    }
    return cos to sin
  }

  /** logits = W hidden (no bias), prob = softmax: (logits, prob_1). */
  fun head(hidden: FloatArray, offset: Int, scoreWeight: FloatArray): Pair<FloatArray, Double> {
    val logits = FloatArray(2)
    for (c in 0 until 2) {
      var s = 0.0
      val base = c * HIDDEN
      for (d in 0 until HIDDEN) s += scoreWeight[base + d].toDouble() * hidden[offset + d]
      logits[c] = s.toFloat()
    }
    val m = max(logits[0], logits[1]).toDouble()
    val e0 = Math.exp(logits[0] - m)
    val e1 = Math.exp(logits[1] - m)
    return logits to e1 / (e0 + e1)
  }
}

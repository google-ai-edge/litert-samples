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

// Port of WeVisOne-VL's BoxFourierEncoder up to its MLP (Tencent, Apache-2.0; see NOTICE).
package com.google.ai.edge.examples.image_cutout

import java.io.DataInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin

/**
 * The mask decoder's box encoding on the host: `BoxFourierEncoder` of wevisone_vl/mask_decoder.py
 * up to its MLP (which stays in the graph):
 * [pe(x1) | pe(y1) | pe(0.2 log w + 0.5) | pe(0.2 log h + 0.5)], 256 floats, with pe(v) =
 * [sin(2 pi v 2^k) | cos(2 pi v 2^k)], k = 0..31, in float32.
 *
 * The phases reach 2 pi x 2^31, where one ulp of v moves the phase by hundreds of radians and the
 * alpha matte by up to 0.009 MAE, so v must be the float32 value torch computes: the phase is
 * float32(2 pi) * v (rounded once) times 2^k (exact), and 0.2 log(w) + 0.5 is read from [table],
 * the value torch 2.11 gives for every box width the 1001 coordinate bins allow (3,493 widths;
 * torch's float32 log differs from the correctly rounded one on 29 of them). sin and cos of the
 * float32 phase are taken in double and rounded (within one float32 ulp of torch's float32 sin /
 * cos).
 */
class BoxFeatures(private val widths: FloatArray, private val encoded: FloatArray) {
  init {
    require(widths.size == encoded.size && widths.isNotEmpty()) {
      "box table: ${widths.size} / ${encoded.size}"
    }
    for (i in 1 until widths.size) {
      require(widths[i - 1] < widths[i]) {
        "box table widths not ascending at $i"
      }
    }
  }

  val tableSize: Int
    get() = widths.size

  /** 0.2 log(w) + 0.5 as torch computes it for a width w = x2 - x1 of two bins (float32). */
  fun encodeSize(w: Float): Float {
    val i = java.util.Arrays.binarySearch(widths, w)
    if (i >= 0) return encoded[i]
    // Not a difference of two bins (cannot come from a parsed box): the float32 formula.
    return 0.2f * ln(w.coerceAtLeast(1e-6f).toDouble()).toFloat() + 0.5f
  }

  /** [box] = normalized x1, y1, x2, y2 (n / 1000 as float32); returns the 256 features. */
  fun features(box: FloatArray): FloatArray {
    require(box.size == 4)
    val out = FloatArray(4 * PE)
    fourier(box[0], out, 0)
    fourier(box[1], out, PE)
    fourier(encodeSize(box[2] - box[0]), out, 2 * PE)
    fourier(encodeSize(box[3] - box[1]), out, 3 * PE)
    return out
  }

  companion object {
    const val FREQS = 32
    const val PE = 2 * FREQS
    private val TWO_PI = (2.0 * PI).toFloat()

    /** pe(v) into out[at until at + 64]: sin of the 32 phases, then cos. */
    fun fourier(v: Float, out: FloatArray, at: Int) {
      val base = TWO_PI * v
      var scale = 1f
      for (k in 0 until FREQS) {
        val phase = (base * scale).toDouble()
        out[at + k] = sin(phase).toFloat()
        out[at + FREQS + k] = cos(phase).toFloat()
        scale *= 2f
      }
    }

    /**
     * The table file: int32 n, n float32 widths (ascending), n float32 encoded values;
     * little-endian.
     */
    fun read(input: InputStream): BoxFeatures {
      val bytes = DataInputStream(input).use { it.readBytes() }
      val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
      val n = buf.int
      require(n in 1..100_000 && bytes.size == 4 + 8 * n) { "box table: n=$n, ${bytes.size} bytes" }
      val widths = FloatArray(n) { buf.float }
      val encoded = FloatArray(n) { buf.float }
      return BoxFeatures(widths, encoded)
    }

    /** A box of the model's bins as the package parses it: n / 1000.0, then float32. */
    fun normalized(bins: IntArray): FloatArray = FloatArray(4) { (bins[it] / 1000.0).toFloat() }
  }
}

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

package com.google.ai.edge.examples.model_zoo.models.dac

import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

/**
 * Residual Vector Quantizer for DAC — the only non-GPU part of the codec (it has an argmin/argmax
 * over the codebook + int indices, which the GPU delegate rejects). Pure-CPU, ~1 ms.
 *
 * encode: continuous latent z[1,HID,T] -> codes[NQ,T] (per codebook: in_proj 1x1 -> L2-normalize ->
 * cosine-argmax -> codebook lookup -> out_proj 1x1, subtract from residual). decode: codes[NQ,T] ->
 * z_q[1,HID,T] (codebook lookup -> out_proj -> sum; == DAC from_codes).
 *
 * Weights come from `dac_rvq.bin` (float32 LE). Per codebook i, contiguous: codebook[SIZE*DIM],
 * in_proj W[DIM*HID], in_proj b[DIM], out_proj W[HID*DIM], out_proj b[HID]. Validated bit-exact vs
 * torchvision DAC (codes match 100%, z_q corr 1.0).
 */
class DacRVQ(bytes: ByteArray) {

  companion object {
    const val NQ = 12 // codebooks
    const val DIM = 8 // codebook_dim
    const val SIZE = 1024 // codebook_size
    const val HID = 1024 // hidden_size
  }

  private val cb = Array(NQ) { FloatArray(SIZE * DIM) } // raw codebook [SIZE][DIM]
  private val cbn = Array(NQ) { FloatArray(SIZE * DIM) } // L2-normalized codebook (for cosine)
  private val wIn = Array(NQ) { FloatArray(DIM * HID) } // [DIM][HID]
  private val bIn = Array(NQ) { FloatArray(DIM) }
  private val wOut = Array(NQ) { FloatArray(HID * DIM) } // [HID][DIM]
  private val bOut = Array(NQ) { FloatArray(HID) }

  init {
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    fun fill(a: FloatArray) {
      for (i in a.indices) a[i] = buf.float
    }
    for (i in 0 until NQ) {
      fill(cb[i])
      fill(wIn[i])
      fill(bIn[i])
      fill(wOut[i])
      fill(bOut[i])
      // precompute L2-normalized codebook rows
      for (k in 0 until SIZE) {
        var n = 0f
        for (d in 0 until DIM) {
          val v = cb[i][k * DIM + d]
          n += v * v
        }
        n = sqrt(n).coerceAtLeast(1e-12f)
        for (d in 0 until DIM) cbn[i][k * DIM + d] = cb[i][k * DIM + d] / n
      }
    }
  }

  /** z [HID*T] (channel-major: c*T + t) -> codes [NQ*T]. */
  fun encode(z: FloatArray, t: Int): IntArray {
    val residual = z.copyOf()
    val codes = IntArray(NQ * t)
    val proj = FloatArray(DIM)
    for (i in 0 until NQ) {
      for (tt in 0 until t) {
        // in_proj + L2-normalize the DIM-vector at time tt
        var pn = 0f
        for (d in 0 until DIM) {
          var s = bIn[i][d]
          val wrow = d * HID
          for (c in 0 until HID) s += wIn[i][wrow + c] * residual[c * t + tt]
          proj[d] = s
          pn += s * s
        }
        pn = sqrt(pn).coerceAtLeast(1e-12f)
        for (d in 0 until DIM) proj[d] /= pn
        // cosine-argmax over codebook
        var best = -1e30f
        var code = 0
        for (k in 0 until SIZE) {
          var dot = 0f
          val cr = k * DIM
          for (d in 0 until DIM) dot += proj[d] * cbn[i][cr + d]
          if (dot > best) {
            best = dot
            code = k
          }
        }
        codes[i * t + tt] = code
        // out_proj(raw codebook[code]) ; subtract from residual
        val cr = code * DIM
        for (c in 0 until HID) {
          var s = bOut[i][c]
          val wrow = c * DIM
          for (d in 0 until DIM) s += wOut[i][wrow + d] * cb[i][cr + d]
          residual[c * t + tt] -= s
        }
      }
    }
    return codes
  }

  /** codes [NQ*T] -> z_q [HID*T] (channel-major). */
  fun decode(codes: IntArray, t: Int): FloatArray {
    val zq = FloatArray(HID * t)
    for (i in 0 until NQ) {
      for (tt in 0 until t) {
        val cr = codes[i * t + tt] * DIM
        for (c in 0 until HID) {
          var s = bOut[i][c]
          val wrow = c * DIM
          for (d in 0 until DIM) s += wOut[i][wrow + d] * cb[i][cr + d]
          zq[c * t + tt] += s
        }
      }
    }
    return zq
  }
}

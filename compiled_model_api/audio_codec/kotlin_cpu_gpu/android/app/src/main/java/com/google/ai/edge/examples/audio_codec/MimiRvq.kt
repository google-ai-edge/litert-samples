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

package com.google.ai.edge.examples.audio_codec

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Mimi split Residual Vector Quantizer — the only non-GPU part of the codec (Euclidean argmin over
 * the codebook + int indices, which the Mali GPU delegate rejects: int64 CAST + EMBEDDING_LOOKUP).
 * Pure CPU. Split = 1 semantic + 31 acoustic codebooks; each group has ONE shared input_proj /
 * output_proj (Conv1d 1x1, no bias) and quantizes the embedding independently in a residual loop.
 *
 *   encode: emb[1,HID,T] -> codes[NQ,T]
 *           semantic : in_proj -> Euclidean-argmin (1 codebook)
 *           acoustic : in_proj -> Euclidean-argmin x31 with residual subtraction
 *   decode: codes[NQ,T] -> emb[1,HID,T]
 *           (sum codebook lookups in DIM space, then out_proj per group)
 *
 * Weights from `mimi_rvq.bin` (float32 LE, contiguous):
 *   sem_Win[DIM*HID], aco_Win[DIM*HID], sem_Wout[HID*DIM], aco_Wout[HID*DIM],
 *   sem_CB[SIZE*DIM], aco_CB[0..30][SIZE*DIM]
 * Validated vs transformers Mimi: encode codes match (torch latent) / decode bit-exact.
 */
class MimiRvq(binPath: String) {

    companion object {
        const val NQ = 32       // total codebooks (1 semantic + 31 acoustic)
        const val N_SEM = 1
        const val N_ACO = 31
        const val DIM = 256     // codebook_dim / vq hidden
        const val SIZE = 2048   // codebook_size
        const val HID = 512     // hidden_size
    }

    private val semWin = FloatArray(DIM * HID)
    private val acoWin = FloatArray(DIM * HID)
    private val semWout = FloatArray(HID * DIM)
    private val acoWout = FloatArray(HID * DIM)
    private val semCB = Array(N_SEM) { FloatArray(SIZE * DIM) }
    private val acoCB = Array(N_ACO) { FloatArray(SIZE * DIM) }
    // ||c||^2 per row (Euclidean shortcut).
    private val semCBnorm = Array(N_SEM) { FloatArray(SIZE) }
    private val acoCBnorm = Array(N_ACO) { FloatArray(SIZE) }

    init {
        val bytes = File(binPath).readBytes()
        val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        fun fill(a: FloatArray) { for (i in a.indices) a[i] = buf.float }
        fill(semWin)
        fill(acoWin)
        fill(semWout)
        fill(acoWout)
        for (i in 0 until N_SEM) {
            fill(semCB[i])
        }
        for (i in 0 until N_ACO) {
            fill(acoCB[i])
        }
        for (i in 0 until N_SEM) {
            sqNorms(semCB[i], semCBnorm[i])
        }
        for (i in 0 until N_ACO) {
            sqNorms(acoCB[i], acoCBnorm[i])
        }
    }

    private fun sqNorms(cb: FloatArray, out: FloatArray) {
        for (k in 0 until SIZE) {
            var n = 0f
            val r = k * DIM
            for (d in 0 until DIM) {
                val v = cb[r + d]
                n += v * v
            }
            out[k] = n
        }
    }

    /** in_proj: W[DIM*HID] applied to emb[c*T+t] -> proj[DIM] at time t. */
    private fun project(w: FloatArray, emb: FloatArray, t: Int, T: Int, proj: FloatArray) {
        for (d in 0 until DIM) {
            var s = 0f
            val wr = d * HID
            for (c in 0 until HID) {
                s += w[wr + c] * emb[c * T + t]
            }
            proj[d] = s
        }
    }

    /** argmin_k ||proj - cb[k]||^2 == argmin_k (||cb[k]||^2 - 2 proj.cb[k]). */
    private fun nearest(proj: FloatArray, cb: FloatArray, cbn: FloatArray): Int {
        var best = Float.MAX_VALUE
        var code = 0
        for (k in 0 until SIZE) {
            var dot = 0f
            val r = k * DIM
            for (d in 0 until DIM) {
                dot += proj[d] * cb[r + d]
            }
            val dist = cbn[k] - 2f * dot
            if (dist < best) {
                best = dist
                code = k
            }
        }
        return code
    }

    /** emb[HID*T] (channel-major c*T+t) -> codes[NQ*T]. */
    fun encode(emb: FloatArray, T: Int): IntArray {
        val codes = IntArray(NQ * T)
        val proj = FloatArray(DIM)
        for (t in 0 until T) {
            project(semWin, emb, t, T, proj)
            codes[t] = nearest(proj, semCB[0], semCBnorm[0])
        }
        val res = FloatArray(DIM)
        for (t in 0 until T) {
            project(acoWin, emb, t, T, res)
            for (i in 0 until N_ACO) {
                val code = nearest(res, acoCB[i], acoCBnorm[i])
                codes[(N_SEM + i) * T + t] = code
                val r = code * DIM
                for (d in 0 until DIM) {
                    res[d] -= acoCB[i][r + d]
                }
            }
        }
        return codes
    }

    /** codes[NQ*T] -> emb[HID*T] (channel-major). out_proj per group, then sum. */
    fun decode(codes: IntArray, T: Int): FloatArray {
        val emb = FloatArray(HID * T)
        val qSem = FloatArray(DIM)
        val qAco = FloatArray(DIM)
        for (t in 0 until T) {
            val sr = codes[t] * DIM
            for (d in 0 until DIM) {
                qSem[d] = semCB[0][sr + d]
            }
            for (d in 0 until DIM) {
                qAco[d] = 0f
            }
            for (i in 0 until N_ACO) {
                val r = codes[(N_SEM + i) * T + t] * DIM
                for (d in 0 until DIM) {
                    qAco[d] += acoCB[i][r + d]
                }
            }
            for (c in 0 until HID) {
                var s = 0f
                val wr = c * DIM
                for (d in 0 until DIM) {
                    s += semWout[wr + d] * qSem[d] + acoWout[wr + d] * qAco[d]
                }
                emb[c * T + t] = s
            }
        }
        return emb
    }
}

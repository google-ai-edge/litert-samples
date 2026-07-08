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

package com.google.ai.edge.examples.speech_enhancement

import android.content.Context
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import java.io.Closeable
import java.io.File
import kotlin.math.cos
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.min
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * CMGAN speech enhancement (noise suppression) on the LiteRT CompiledModel GPU — fully GPU.
 *
 * 2 s 16 kHz chunks (reflect-padded on the host) run through one 1.83 M-param conformer graph
 * (~20 ms on a Pixel 8a); the graph contains the STFT (DFT-as-conv) and the mag^0.3 power
 * compression, and emits the enhanced COMPRESSED complex spectrogram [1, 1, 321, 201] (real,
 * imag). The host un-compresses (mag^(1/0.3)) and runs the inverse STFT (n_fft 400 / hop 100,
 * periodic hamming) + overlap-add across chunks.
 */
class NoiseSuppressor(ctx: Context) : Closeable {

    companion object {
        const val SR = 16000
        const val NFFT = 400
        const val HOP = 100
        const val ENC = NFFT / 2 + 1          // 201
        const val CHUNK = 2 * SR              // 32000 samples -> 321 frames
        const val FRAMES = CHUNK / HOP + 1    // 321
        const val PAD = NFFT / 2              // 200 reflect pad each side
        const val HOP_CHUNK = CHUNK - 4000    // 0.25 s crossfade between chunks
    }

    private val model: CompiledModel = run {
        val f = File(ctx.filesDir, "cmgan_fp16.tflite")
        check(f.exists()) { "Model not found: ${f.name}. Run scripts/install_to_device.sh first." }
        CompiledModel.create(f.absolutePath, CompiledModel.Options(Accelerator.GPU), null)
    }
    private val inBuf = model.createInputBuffers()
    private val outBuf = model.createOutputBuffers()

    // inverse real-DFT synthesis tables (n_fft=400 is not a power of two; direct synthesis)
    private val cosT = Array(ENC) { k ->
        FloatArray(NFFT) { n -> cos(2.0 * Math.PI * k * n / NFFT).toFloat() }
    }
    private val sinT = Array(ENC) { k ->
        FloatArray(NFFT) { n -> sin(2.0 * Math.PI * k * n / NFFT).toFloat() }
    }
    private val ham = FloatArray(NFFT) { (0.54 - 0.46 * cos(2.0 * Math.PI * it / NFFT)).toFloat() }

    /** Enhance a mono 16 kHz track. Returns the denoised track (same length). */
    fun enhance(pcm: FloatArray, onProgress: (Int, Int) -> Unit): FloatArray {
        // whole-utterance RMS normalization (evaluation.py): x' = x * c, output / c
        var p = 0.0
        for (v in pcm) p += v * v
        val c = sqrt(pcm.size / (p + 1e-12)).toFloat()

        val out = FloatArray(pcm.size)
        val weight = FloatArray(pcm.size)
        val nChunks =
            if (pcm.size <= CHUNK) 1
            else 1 + ((pcm.size - CHUNK) + HOP_CHUNK - 1) / HOP_CHUNK
        for (ci in 0 until nChunks) {
            onProgress(ci + 1, nChunks)
            val start = ci * HOP_CHUNK
            val x = FloatArray(CHUNK + 2 * PAD)
            for (i in 0 until CHUNK) {
                val q = start + i
                x[PAD + i] = if (q < pcm.size) pcm[q] * c else 0f
            }
            for (j in 1..PAD) {                                   // reflect pad
                x[PAD - j] = x[PAD + j]
                x[PAD + CHUNK - 1 + j] = x[PAD + CHUNK - 1 - j]
            }
            inBuf[0].writeFloat(x)
            model.run(inBuf, outBuf)
            val er = outBuf[0].readFloat()                        // [321 * 201] compressed real
            val ei = outBuf[1].readFloat()
            val seg = istft(er, ei)                               // [CHUNK]
            val n = min(CHUNK, pcm.size - start)
            for (i in 0 until n) {                                // equal-weight overlap-add
                out[start + i] += seg[i] / c
                weight[start + i] += 1f
            }
        }
        for (i in out.indices) if (weight[i] > 1f) out[i] /= weight[i]
        return out
    }

    /** Un-compress (mag^0.3 -> mag) + inverse STFT + hamming overlap-add, trim center pad. */
    private fun istft(er: FloatArray, ei: FloatArray): FloatArray {
        val padded = FloatArray((FRAMES - 1) * HOP + NFFT)
        val wsum = FloatArray(padded.size)
        val re = FloatArray(ENC)
        val im = FloatArray(ENC)
        val frame = FloatArray(NFFT)
        for (t in 0 until FRAMES) {
            for (k in 0 until ENC) {
                val r = er[t * ENC + k]
                val i = ei[t * ENC + k]
                val m2 = r * r + i * i
                // mag_c^(1/0.3) scaling: multiply (r, i) by m2^((1/0.3 - 1)/2) = m2^(7/6)
                val s = if (m2 > 1e-12f) exp((7.0 / 6.0) * ln(m2.toDouble())).toFloat() else 0f
                re[k] = r * s
                im[k] = i * s
            }
            // inverse one-sided real DFT (direct synthesis; N=400 is not radix-2)
            for (n in 0 until NFFT) {
                var acc = re[0] * cosT[0][n]
                for (k in 1 until ENC - 1) acc += 2f * (re[k] * cosT[k][n] - im[k] * sinT[k][n])
                acc += re[ENC - 1] * cosT[ENC - 1][n] - im[ENC - 1] * sinT[ENC - 1][n]
                frame[n] = acc / NFFT
            }
            val base = t * HOP
            for (n in 0 until NFFT) {
                padded[base + n] += frame[n] * ham[n]
                wsum[base + n] += ham[n] * ham[n]
            }
        }
        val outSeg = FloatArray(CHUNK)
        for (i in 0 until CHUNK) {
            val w = wsum[PAD + i]
            outSeg[i] = if (w > 1e-8f) padded[PAD + i] / w else 0f
        }
        return outSeg
    }

    override fun close() {
        inBuf.forEach { it.close() }
        outBuf.forEach { it.close() }
        model.close()
    }
}

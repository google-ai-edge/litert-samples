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

package com.google.ai.edge.examples.audio_source_separation

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.sin

/**
 * Host-side inverse STFT (win 2048, hop 512, periodic Hann), mirroring torch.istft with
 * center=True: overlap-add of windowed IFFT frames, normalized by the squared-window envelope,
 * then the n_fft/2 center padding is trimmed. The forward STFT runs INSIDE the GPU graph
 * (DFT-as-conv); only this inverse runs on the host.
 */
object Istft {
    const val WIN = 2048
    const val HOP = 512
    const val ENC = WIN / 2 + 1   // 1025 one-sided bins

    private val hann = FloatArray(WIN) { (0.5 - 0.5 * cos(2.0 * PI * it / WIN)).toFloat() }
    private val cosTab = FloatArray(WIN / 2) { cos(2.0 * PI * it / WIN).toFloat() }
    private val sinTab = FloatArray(WIN / 2) { sin(2.0 * PI * it / WIN).toFloat() }

    /**
     * @param real, imag one-sided spectrogram, layout [ENC][frames] (f-major, as the model emits)
     * @param frames     number of STFT frames
     * @param outLen     final sample count (center padding of WIN/2 is trimmed from both ends)
     */
    fun run(real: FloatArray, imag: FloatArray, frames: Int, outLen: Int): FloatArray {
        val padded = FloatArray((frames - 1) * HOP + WIN)
        val wsum = FloatArray(padded.size)
        val re = FloatArray(WIN)
        val im = FloatArray(WIN)
        for (t in 0 until frames) {
            // rebuild the full conjugate-symmetric spectrum for this frame
            for (f in 0 until ENC) {
                re[f] = real[f * frames + t]
                im[f] = imag[f * frames + t]
            }
            for (f in 1 until WIN / 2) {
                re[WIN - f] = re[f]
                im[WIN - f] = -im[f]
            }
            ifft(re, im)
            val base = t * HOP
            for (n in 0 until WIN) {
                padded[base + n] += re[n] * hann[n]
                wsum[base + n] += hann[n] * hann[n]
            }
        }
        val out = FloatArray(outLen)
        val off = WIN / 2
        for (i in 0 until outLen) {
            val w = wsum[off + i]
            out[i] = if (w > 1e-8f) padded[off + i] / w else 0f
        }
        return out
    }

    /** In-place radix-2 complex IFFT of size WIN (2048). */
    private fun ifft(re: FloatArray, im: FloatArray) {
        val n = WIN
        // conjugate -> forward FFT -> conjugate, / n
        for (i in 0 until n) im[i] = -im[i]
        fft(re, im)
        val inv = 1f / n
        for (i in 0 until n) {
            re[i] *= inv
            im[i] = -im[i] * inv
        }
    }

    /** Iterative in-place radix-2 complex FFT of size WIN. */
    private fun fft(re: FloatArray, im: FloatArray) {
        val n = WIN
        // bit-reversal permutation
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                var tmp = re[i]
                re[i] = re[j]
                re[j] = tmp
                tmp = im[i]
                im[i] = im[j]
                im[j] = tmp
            }
            var m = n shr 1
            while (m in 1..j) {
                j -= m
                m = m shr 1
            }
            j += m
        }
        var len = 2
        while (len <= n) {
            val half = len shr 1
            val step = n / len
            var i = 0
            while (i < n) {
                var k = 0
                for (jj in i until i + half) {
                    val wr = cosTab[k]
                    val wi = -sinTab[k]
                    val xr = re[jj + half] * wr - im[jj + half] * wi
                    val xi = re[jj + half] * wi + im[jj + half] * wr
                    re[jj + half] = re[jj] - xr
                    im[jj + half] = im[jj] - xi
                    re[jj] += xr
                    im[jj] += xi
                    k += step
                }
                i += len
            }
            len = len shl 1
        }
    }
}

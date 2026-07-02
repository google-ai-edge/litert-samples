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

package com.google.ai.edge.examples.speaker_diarization

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln
import kotlin.math.max

/**
 * Kaldi-compatible 80-mel log filterbank (torchaudio.compliance.kaldi.fbank with dither=0,
 * hamming, 25 ms / 10 ms, snip_edges) + cepstral mean normalization — the WeSpeaker front-end.
 * Mel banks and the hamming window are precomputed assets; verified against torchaudio
 * (max|d| 4e-4 in the log domain, corr 1.0).
 */
class Fbank(ctx: Context) {

    companion object {
        const val SR = 16000
        const val WIN = 400
        const val HOP = 160
        const val NFFT = 512
        const val NBIN = 257
        const val NMEL = 80
        private const val EPS = 1.1920928955078125e-07f
        private const val PREEMPH = 0.97f
    }

    private val mel = loadFloats(ctx, "mel80_257.bin", NMEL * NBIN)      // [80][257] row-major
    private val ham = loadFloats(ctx, "hamming400.bin", WIN)
    private val cosT = FloatArray(NFFT / 2) { kotlin.math.cos(2.0 * Math.PI * it / NFFT).toFloat() }
    private val sinT = FloatArray(NFFT / 2) { kotlin.math.sin(2.0 * Math.PI * it / NFFT).toFloat() }

    private fun loadFloats(ctx: Context, name: String, n: Int): FloatArray {
        val b = ctx.assets.open(name).readBytes()
        check(b.size == n * 4) { "$name: ${b.size} bytes, expected ${n * 4}" }
        val out = FloatArray(n)
        ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
        return out
    }

    /** pcm [-1,1] -> log-mel [frames][80], then CMN (subtract per-mel mean over frames). */
    fun compute(pcm: FloatArray, cmn: Boolean = true): Array<FloatArray> {
        val nFrames = 1 + (pcm.size - WIN) / HOP
        val out = Array(nFrames) { FloatArray(NMEL) }
        val re = FloatArray(NFFT)
        val im = FloatArray(NFFT)
        val power = FloatArray(NBIN)
        for (t in 0 until nFrames) {
            // frame -> x32768, remove DC, pre-emphasis (replicate pad), window, zero-pad
            var mean = 0f
            val base = t * HOP
            for (i in 0 until WIN) mean += pcm[base + i]
            mean = mean * 32768f / WIN
            var prev = pcm[base] * 32768f - mean
            for (i in 0 until WIN) {
                val cur = pcm[base + i] * 32768f - mean
                re[i] = (cur - PREEMPH * prev) * ham[i]
                im[i] = 0f
                prev = cur
            }
            for (i in WIN until NFFT) { re[i] = 0f; im[i] = 0f }
            fft(re, im)
            for (k in 0 until NBIN) power[k] = re[k] * re[k] + im[k] * im[k]
            val row = out[t]
            for (m in 0 until NMEL) {
                var acc = 0f
                val off = m * NBIN
                for (k in 0 until NBIN) acc += mel[off + k] * power[k]
                row[m] = ln(max(acc, EPS))
            }
        }
        if (cmn) {
            val mu = FloatArray(NMEL)
            for (t in 0 until nFrames) for (m in 0 until NMEL) mu[m] += out[t][m]
            for (m in 0 until NMEL) mu[m] /= nFrames
            for (t in 0 until nFrames) for (m in 0 until NMEL) out[t][m] -= mu[m]
        }
        return out
    }

    /** Iterative in-place radix-2 complex FFT of size NFFT (512). */
    private fun fft(re: FloatArray, im: FloatArray) {
        val n = NFFT
        var j = 0
        for (i in 0 until n - 1) {
            if (i < j) {
                var t = re[i]; re[i] = re[j]; re[j] = t
                t = im[i]; im[i] = im[j]; im[j] = t
            }
            var m = n shr 1
            while (m in 1..j) { j -= m; m = m shr 1 }
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
                    val wr = cosT[k]
                    val wi = -sinT[k]
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

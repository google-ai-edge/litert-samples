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

package com.google.ai.edge.examples.parakeet

import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

/**
 * Log-mel features matching NeMo's AudioToMelSpectrogramPreprocessor for parakeet-tdt_ctc-110m, computed
 * on the host (FFT is GPU-banned). Pipeline, verified to reproduce the reference transcript: preemphasis
 * (0.97) -> center zero-pad (n_fft/2) -> frame -> hann(400) padded to 512 -> |rFFT512|^2 (mag_power=2) ->
 * mel filterbank (slaney) -> log(x + 2^-24) -> per-feature normalize (over time, std with Bessel n-1
 * correction + 1e-5).
 *
 * @param fb mel filterbank, row-major [N_MEL * N_BIN] (assets/mel_fb.bin, the model's exact buffer).
 */
class MelSpectrogram(private val fb: FloatArray) {

  companion object {
    const val SR = 16000
    const val N_FFT = 512
    const val WIN = 400
    const val HOP = 160
    const val N_MEL = 80
    const val N_BIN = 257 // n_fft/2 + 1
    const val PREEMPH = 0.97f
    const val LOG_GUARD = 5.9604645e-8f // 2^-24
    const val NORM_EPS = 1e-5f

    /** Subsampled length T' after the dw_striding factor-8 front-end (3x floor((L-1)/2)+1). */
    fun subLen(melFrames: Int): Int {
      var l = melFrames
      repeat(3) { l = (l - 1) / 2 + 1 }
      return l
    }
  }

  /** hann(400) symmetric (periodic=False), centered in a 512 frame with 56 zeros each side. */
  private val window =
    FloatArray(N_FFT).also { w ->
      val off = (N_FFT - WIN) / 2
      for (n in 0 until WIN) w[off + n] = (0.5 - 0.5 * cos(2.0 * Math.PI * n / (WIN - 1))).toFloat()
    }
  private val fft = Fft(N_FFT)

  data class Result(val logmel: FloatArray, val frames: Int) // logmel is [N_MEL * frames], bin-major

  /** @param audio mono PCM in [-1,1] at 16 kHz. */
  fun compute(audio: FloatArray): Result {
    val n = audio.size
    val x = FloatArray(n)
    if (n > 0) x[0] = audio[0]
    for (t in 1 until n) x[t] = audio[t] - PREEMPH * audio[t - 1]

    val nfr = n / HOP + 1 // = NeMo get_seq_len (center pad cancels)
    val logmel = FloatArray(N_MEL * nfr)
    val frame = FloatArray(N_FFT)
    val re = FloatArray(N_FFT)
    val im = FloatArray(N_FFT)
    val power = FloatArray(N_BIN)
    val pad = N_FFT / 2

    for (t in 0 until nfr) {
      val start = t * HOP - pad
      for (k in 0 until N_FFT) {
        val idx = start + k
        frame[k] = if (idx in 0 until n) x[idx] * window[k] else 0f
      }
      fft.transform(frame, re, im)
      for (b in 0 until N_BIN) power[b] = re[b] * re[b] + im[b] * im[b]
      for (m in 0 until N_MEL) {
        var acc = 0f
        val base = m * N_BIN
        for (b in 0 until N_BIN) acc += fb[base + b] * power[b]
        logmel[m * nfr + t] = ln(acc + LOG_GUARD)
      }
    }

    // per-feature (per mel bin) normalization over time
    for (m in 0 until N_MEL) {
      val base = m * nfr
      var mean = 0f
      for (t in 0 until nfr) mean += logmel[base + t]
      mean /= nfr
      var v = 0f
      for (t in 0 until nfr) {
        val d = logmel[base + t] - mean
        v += d * d
      }
      val inv = 1f / (sqrt(v / (nfr - 1)) + NORM_EPS)
      for (t in 0 until nfr) logmel[base + t] = (logmel[base + t] - mean) * inv
    }
    return Result(logmel, nfr)
  }

  /** In-place iterative radix-2 Cooley-Tukey FFT (N must be a power of two). */
  private class Fft(private val n: Int) {
    private val rev = IntArray(n)
    private val cosT = FloatArray(n / 2)
    private val sinT = FloatArray(n / 2)

    init {
      var log = 0
      while ((1 shl log) < n) log++
      for (i in 0 until n) {
        var x = i
        var r = 0
        for (b in 0 until log) {
          r = (r shl 1) or (x and 1)
          x = x shr 1
        }
        rev[i] = r
      }
      for (k in 0 until n / 2) {
        val a = -2.0 * Math.PI * k / n
        cosT[k] = cos(a).toFloat()
        sinT[k] = Math.sin(a).toFloat()
      }
    }

    /** real -> (re, im), full complex spectrum (first n/2+1 bins are the useful ones). */
    fun transform(real: FloatArray, re: FloatArray, im: FloatArray) {
      for (i in 0 until n) {
        re[rev[i]] = real[i]
        im[rev[i]] = 0f
      }
      var len = 2
      while (len <= n) {
        val half = len / 2
        val step = n / len
        var i = 0
        while (i < n) {
          var k = 0
          for (j in 0 until half) {
            val wr = cosT[k]
            val wi = sinT[k]
            val a = i + j
            val b = a + half
            val vr = re[b] * wr - im[b] * wi
            val vi = re[b] * wi + im[b] * wr
            re[b] = re[a] - vr
            im[b] = im[a] - vi
            re[a] += vr
            im[a] += vi
            k += step
          }
          i += len
        }
        len = len shl 1
      }
    }
  }
}

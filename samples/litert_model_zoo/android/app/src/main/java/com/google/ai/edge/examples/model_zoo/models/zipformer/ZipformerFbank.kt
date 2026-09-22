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

package com.google.ai.edge.examples.model_zoo.models.zipformer

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.ln
import kotlin.math.max

/**
 * Kaldi-compatible 80-mel log filterbank matching `torchaudio.compliance.kaldi.fbank` with
 * dither=0, snip_edges=false (reflect padding), povey window, high_freq=-400 — the icefall
 * Zipformer front-end. Input PCM stays in [-1, 1] (do NOT scale to int16 range) and there is no
 * CMN. Mel banks and the povey window are precomputed assets; the port was verified against
 * torchaudio (max|diff| 0.0026 in the log domain, corr 1.0).
 */
class ZipformerFbank
internal constructor(private val mel: FloatArray, private val win: FloatArray) {
  constructor(
    ctx: Context
  ) : this(loadFloats(ctx, "mel80_257.bin", NMEL * NBIN), loadFloats(ctx, "povey400.bin", WIN))

  companion object {
    const val SR = 16000
    const val WIN = 400
    const val HOP = 160
    const val NFFT = 512
    const val NBIN = 257
    const val NMEL = 80
    const val LOG_PAD = -23.025851f // ln(1e-10), the icefall padding value
    private const val EPS = 1.1920928955078125e-07f
    private const val PREEMPH = 0.97f
    private const val PAD = WIN / 2 - HOP / 2 // 120, snip_edges=false reflect margin

    private fun loadFloats(ctx: Context, name: String, n: Int): FloatArray {
      val b = ctx.assets.open(name).readBytes()
      check(b.size == n * 4) { "$name: ${b.size} bytes, expected ${n * 4}" }
      val out = FloatArray(n)
      ByteBuffer.wrap(b).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out)
      return out
    }
  }

  private val cosT = FloatArray(NFFT / 2) { kotlin.math.cos(2.0 * Math.PI * it / NFFT).toFloat() }
  private val sinT = FloatArray(NFFT / 2) { kotlin.math.sin(2.0 * Math.PI * it / NFFT).toFloat() }

  /** Frame count for n samples (snip_edges=false). */
  fun frames(n: Int): Int = (n + HOP / 2) / HOP

  /** pcm [-1,1] -> log-mel [frames][80]. */
  fun compute(pcm: FloatArray): Array<FloatArray> {
    val n = pcm.size
    val nFrames = frames(n)
    val out = Array(nFrames) { FloatArray(NMEL) }
    val re = FloatArray(NFFT)
    val im = FloatArray(NFFT)
    val frame = FloatArray(WIN)
    val power = FloatArray(NBIN)

    // reflect-padded sample fetch: q<0 -> pcm[-q-1], q>=n -> pcm[2n-1-q]
    fun sample(p: Int): Float {
      val q = p - PAD
      return when {
        q < 0 -> pcm[-q - 1]
        q < n -> pcm[q]
        else -> pcm[2 * n - 1 - q]
      }
    }

    for (t in 0 until nFrames) {
      val base = t * HOP
      var mean = 0f
      for (i in 0 until WIN) {
        frame[i] = sample(base + i)
        mean += frame[i]
      }
      mean /= WIN
      // remove DC, pre-emphasis (replicate pad), povey window, zero-pad to NFFT
      var prev = frame[0] - mean
      for (i in 0 until WIN) {
        val cur = frame[i] - mean
        re[i] = (cur - PREEMPH * prev) * win[i]
        im[i] = 0f
        prev = cur
      }
      for (i in WIN until NFFT) {
        re[i] = 0f
        im[i] = 0f
      }
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
    return out
  }

  /** Iterative in-place radix-2 complex FFT of size NFFT (512). */
  private fun fft(re: FloatArray, im: FloatArray) {
    val n = NFFT
    var j = 0
    for (i in 0 until n - 1) {
      if (i < j) {
        var t = re[i]
        re[i] = re[j]
        re[j] = t
        t = im[i]
        im[i] = im[j]
        im[j] = t
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

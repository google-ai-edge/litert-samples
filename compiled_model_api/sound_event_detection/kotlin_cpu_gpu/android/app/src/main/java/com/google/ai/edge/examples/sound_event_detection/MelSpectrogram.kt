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

package com.google.ai.edge.examples.sound_event_detection

import android.content.Context
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.sin

/**
 * Host-side log-mel front-end for PANNs CNN14, computed to match torchlibrosa exactly:
 *   Spectrogram(n_fft=1024, hop=320, win='hann', center=True, pad_mode='reflect', power=2)
 *   + LogmelFilterBank(sr=32000, n_mels=64, fmin=50, fmax=14000, ref=1.0, amin=1e-10, top_db=None)
 *   => log_mel = 10 * log10(max(mel_power, 1e-10))
 *
 * Done on the CPU (not the GPU graph) on purpose: the power spectrum |STFT|^2 reaches ~1e6, which
 * overflows fp16 on the Mali delegate (-> NaN). The downstream CNN (logmel -> 527 tags) is the part
 * that runs on the LiteRT CompiledModel GPU.
 *
 * Output is the [T=1001, 64] log-mel in frame-major (t*64 + mel) order, which is exactly the
 * [1, 1, 1001, 64] layout the CNN tflite expects.
 */
class MelSpectrogram(context: Context) {

  companion object {
    const val SAMPLE_RATE = 32000
    const val N_FFT = 1024
    const val HOP = 320
    const val N_MELS = 64
    const val CLIP_SAMPLES = 320000           // 10 s @ 32 kHz (PANNs canonical window)
    const val PAD = N_FFT / 2                  // 512 (center=True)
    const val N_FREQS = N_FFT / 2 + 1          // 513
    const val N_FRAMES = 1 + CLIP_SAMPLES / HOP    // 1001
  }

  // librosa/torchlibrosa mel basis [N_MELS, N_FREQS], mel-major (asset exported from melW.T)
  private val melBasis: FloatArray

  init {
    val bytes = context.assets.open("mel_basis.bin").use { it.readBytes() }
    val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    melBasis = FloatArray(bytes.size / 4)
    buf.asFloatBuffer().get(melBasis)
    require(melBasis.size == N_MELS * N_FREQS) {
      "mel_basis.bin has ${melBasis.size} floats, expected ${N_MELS * N_FREQS}"
    }
  }

  private val window = FloatArray(N_FFT) { i -> (0.5 - 0.5 * cos(2.0 * PI * i / N_FFT)).toFloat() }  // periodic Hann

  // Precomputed twiddle factors W_N^k = exp(-2πi k / N_FFT), k = 0..N_FFT/2-1 (reused across all
  // frames/stages so the FFT does no per-butterfly trig — the dominant cost otherwise).
  private val twCos = FloatArray(N_FFT / 2) { k -> cos(-2.0 * PI * k / N_FFT).toFloat() }
  private val twSin = FloatArray(N_FFT / 2) { k -> sin(-2.0 * PI * k / N_FFT).toFloat() }

  /**
   * Compute the log-mel spectrogram from a CLIP_SAMPLES-long 32 kHz mono clip. Pads/truncates.
   * @return FloatArray of length N_FRAMES * N_MELS in (frame, mel) row-major order.
   */
  fun compute(samples: FloatArray): FloatArray {
    val x = FloatArray(CLIP_SAMPLES)
    System.arraycopy(samples, 0, x, 0, minOf(samples.size, CLIP_SAMPLES))

    // center=True reflect padding of width PAD on both sides
    val padded = FloatArray(CLIP_SAMPLES + 2 * PAD)
    System.arraycopy(x, 0, padded, PAD, CLIP_SAMPLES)
    for (j in 0 until PAD) {
      padded[j] = x[PAD - j]                                    // left reflect: x[512], x[511], ..., x[1]
      padded[PAD + CLIP_SAMPLES + j] = x[CLIP_SAMPLES - 2 - j]  // right reflect
    }

    val out = FloatArray(N_FRAMES * N_MELS)
    val re = FloatArray(N_FFT)
    val im = FloatArray(N_FFT)
    val power = FloatArray(N_FREQS)

    for (t in 0 until N_FRAMES) {
      val off = t * HOP
      for (i in 0 until N_FFT) {
          re[i] = padded[off + i] * window[i]
          im[i] = 0f
      }
      fft(re, im, N_FFT)
      for (f in 0 until N_FREQS) {
        power[f] = re[f] * re[f] + im[f] * im[f]
      }

      val base = t * N_MELS
      for (mel in 0 until N_MELS) {
        var sum = 0f
        val wbase = mel * N_FREQS
        for (f in 0 until N_FREQS) {
          sum += melBasis[wbase + f] * power[f]
        }
        out[base + mel] = (10.0 * log10(maxOf(sum, 1e-10f).toDouble())).toFloat()
      }
    }
    return out
  }

  /** Cooley-Tukey radix-2 in-place FFT (N is a power of two). */
  private fun fft(real: FloatArray, imag: FloatArray, n: Int) {
    var j = 0
    for (i in 0 until n - 1) {
      if (i < j) {
        var tmp = real[i]
        real[i] = real[j]
        real[j] = tmp
        tmp = imag[i]
        imag[i] = imag[j]
        imag[j] = tmp
      }
      var k = n / 2
      while (k <= j) {
          j -= k
          k /= 2
      }
      j += k
    }
    var step = 2
    while (step <= n) {
      val half = step / 2
      val twStride = n / step               // twiddle index for `pair` = pair * twStride
      for (group in 0 until n step step) {
        for (pair in 0 until half) {
          val tw = pair * twStride
          val wr = twCos[tw]
          val wi = twSin[tw]
          val i1 = group + pair
          val i2 = i1 + half
          val tr = wr * real[i2] - wi * imag[i2]
          val ti = wr * imag[i2] + wi * real[i2]
          real[i2] = real[i1] - tr
          imag[i2] = imag[i1] - ti
          real[i1] = real[i1] + tr
          imag[i1] = imag[i1] + ti
        }
      }
      step *= 2
    }
  }
}

/*
 * Copyright 2026 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.asr

import com.jlibrosa.audio.JLibrosa
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sqrt

class MelSpectroProcessor(
  private val samplingRate: Int,
  private val config: LogMelSpectroConfig,
  // MelSpectrogramGenerator is a function to convert a waveform to a mel spectrogram. Testing can
  // be done by mocking this function.
  private val generateMelSpectrogram: (FloatArray, Int, Int, Int, Int) -> Array<FloatArray> =
    ::defaultGenerateMelSpectrogram,
) : AudioPreprocessor {

  override fun close() {}

  override fun process(rawSpeech: FloatArray): FloatArray {
    if (rawSpeech.isEmpty()) return FloatArray(0)
    val audio =
      if (config.preemphasis > 0.0f) {
        applyPreemphasis(rawSpeech, config.preemphasis)
      } else {
        rawSpeech
      }

    val paddedNFFT = getSmallestPowerOfTwoGreaterOrEqualTo(config.nFFT)
    val melSpec =
      generateMelSpectrogram(audio, samplingRate, paddedNFFT, config.nMels, config.hopLength)
    val validFrames = rawSpeech.size / config.hopLength

    // Normalize the mel spectrogram based on the normType.
    if (config.normType == "whisper") {
      var maxVal = Float.NEGATIVE_INFINITY
      for (m in melSpec.indices) {
        for (f in 0 until validFrames) {
          // Whisper normalization constants are typically based on log10. Since we are using ln,
          // we scale down mel spectrogram values by ln(10).
          melSpec[m][f] = melSpec[m][f] / LN_10
          if (melSpec[m][f] > maxVal) maxVal = melSpec[m][f]
        }
      }

      val clipVal = maxVal - 8.0f
      for (m in melSpec.indices) {
        for (f in 0 until validFrames) {
          melSpec[m][f] = (max(melSpec[m][f], clipVal) + 4.0f) / 4.0f
        }
      }
    } else {
      val means = computeMean(melSpec, validFrames)
      val stds = computeStd(melSpec, means, validFrames)
      for (m in melSpec.indices) {
        val mean = means[m]
        val std = stds[m] + EPSILON
        for (f in 0 until validFrames) {
          melSpec[m][f] = (melSpec[m][f] - mean) / std
        }
      }
    }

    val nFrames = if (config.nFrames > 0) config.nFrames else melSpec[0].size
    val result = FloatArray(nFrames * config.nMels)
    for (m in melSpec.indices) {
      for (f in melSpec[m].indices) {
        if (f >= nFrames) break
        val value = if (f < validFrames) melSpec[m][f] else 0.0f
        if (config.transpose) {
          result[f * config.nMels + m] = value
        } else {
          result[m * nFrames + f] = value
        }
      }
    }

    return result
  }

  private fun applyPreemphasis(audio: FloatArray, coeff: Float): FloatArray {
    if (audio.isEmpty()) return FloatArray(0)
    val result = FloatArray(audio.size)
    result[0] = audio[0]
    for (i in 1 until audio.size) {
      result[i] = audio[i] - coeff * audio[i - 1]
    }
    return result
  }

  /** Computes mean over TIME, respecting valid frames */
  private fun computeMean(melSpec: Array<FloatArray>, nFrames: Int): FloatArray {
    val result = FloatArray(melSpec.size)
    for (m in melSpec.indices) {
      var sum = 0.0f
      for (f in 0 until nFrames) {
        sum += melSpec[m][f]
      }
      result[m] = sum / nFrames
    }
    return result
  }

  /** Computes sample variance over TIME, respecting valid frames */
  private fun computeStd(melSpec: Array<FloatArray>, mean: FloatArray, nFrames: Int): FloatArray {
    val result = FloatArray(melSpec.size)
    for (m in melSpec.indices) {
      var sumSq = 0.0f
      for (f in 0 until nFrames) {
        val diff = melSpec[m][f] - mean[m]
        sumSq += diff * diff
      }
      result[m] = sqrt(sumSq / max(1, nFrames - 1)) // Sample variance (N-1)
    }
    return result
  }

  private companion object {
    val LOG_ZERO_GUARD_VALUE = 2.0f.pow(-24) // 2**-24 \approx 5.96e-8
    const val EPSILON = 1e-5f
    val LN_10 = ln(10.0f)

    val jLibrosa by lazy { JLibrosa() }

    fun defaultGenerateMelSpectrogram(
      waveform: FloatArray,
      samplingRate: Int,
      nFFT: Int,
      nMels: Int,
      hopLength: Int,
    ): Array<FloatArray> {
      // JLibrosa's generateMelSpectroGram uses librosa's default parameters.
      // In librosa, the default normalization for mel filters is 'slaney'.
      // See: https://librosa.org/doc/latest/generated/librosa.filters.mel.html
      // melSpec is [mel_bins][frames]
      val melSpec = jLibrosa.generateMelSpectroGram(waveform, samplingRate, nFFT, nMels, hopLength)

      for (m in melSpec.indices) {
        for (f in melSpec[m].indices) {
          melSpec[m][f] = ln(melSpec[m][f] + LOG_ZERO_GUARD_VALUE)
        }
      }

      return melSpec
    }

    fun getSmallestPowerOfTwoGreaterOrEqualTo(n: Int): Int {
      if (n > 0 && (n and (n - 1)) == 0) {
        return n
      }

      var p = 1
      while (p < n) {
        p = p shl 1
      }
      return p
    }
  }
}

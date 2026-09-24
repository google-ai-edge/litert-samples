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

package com.google.ai.edge.examples.model_zoo.audio

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Read uncompressed PCM16 or float32 WAV input and normalize to 16 kHz mono. */
object WavAudio {
  data class MonoAudio(val samples: FloatArray, val sampleRate: Int)

  fun read16k(bytes: ByteArray): FloatArray {
    val audio = readMono(bytes)
    return if (audio.sampleRate != 16000) resample(audio.samples, audio.sampleRate, 16000)
    else audio.samples
  }

  /** Preserve the source sample rate for task-specific audio frontends. */
  fun readMono(bytes: ByteArray): MonoAudio {
    require(bytes.size >= 12 && tag(bytes, 0) == "RIFF" && tag(bytes, 8) == "WAVE") {
      "Choose an uncompressed PCM16 or float32 WAV file."
    }
    val data = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    var cursor = 12
    var format = 0
    var channels = 0
    var sampleRate = 0
    var bits = 0
    var blockAlign = 0
    var samplesOffset = -1
    var samplesBytes = 0
    while (cursor + 8 <= bytes.size) {
      val name = tag(bytes, cursor)
      val length = data.getInt(cursor + 4).toLong() and 0xffffffffL
      val start = cursor + 8
      require(length <= bytes.size.toLong() - start) { "The WAV file is incomplete." }
      if (name == "fmt ") {
        require(length >= 16) { "The WAV format header is incomplete." }
        format = data.getShort(start).toInt() and 0xffff
        channels = data.getShort(start + 2).toInt() and 0xffff
        sampleRate = data.getInt(start + 4)
        blockAlign = data.getShort(start + 12).toInt() and 0xffff
        bits = data.getShort(start + 14).toInt() and 0xffff
      } else if (name == "data") {
        samplesOffset = start
        samplesBytes = length.toInt()
      }
      cursor = (start.toLong() + length + length % 2).coerceAtMost(bytes.size.toLong()).toInt()
    }
    require(channels in 1..8 && sampleRate in 8000..192000 && samplesOffset >= 0) {
      "The WAV file has no supported audio stream."
    }
    require((format == 1 && bits == 16) || (format == 3 && bits == 32)) {
      "Choose a WAV encoded as PCM16 or float32."
    }
    require(blockAlign == channels * (bits / 8) && samplesBytes % blockAlign == 0) {
      "The WAV audio frames are incomplete."
    }
    val sampleCount = samplesBytes / (bits / 8)
    require(sampleCount > 0) { "The WAV file is empty." }
    data.position(samplesOffset)
    val pcmSamples =
      FloatArray(sampleCount) { if (format == 1) data.short.toFloat() / 32768f else data.float }
    require(pcmSamples.all { it.isFinite() }) { "The WAV file contains invalid samples." }
    val monoSamples =
      if (channels > 1) {
        FloatArray(pcmSamples.size / channels) { i ->
          var sum = 0f
          for (ch in 0 until channels) sum += pcmSamples[i * channels + ch]
          sum / channels
        }
      } else {
        pcmSamples
      }
    return MonoAudio(monoSamples, sampleRate)
  }

  private fun tag(bytes: ByteArray, offset: Int): String =
    String(bytes, offset, 4, Charsets.US_ASCII)

  internal fun resample(input: FloatArray, fromRate: Int, toRate: Int): FloatArray {
    if (fromRate == toRate) return input
    val ratio = fromRate.toDouble() / toRate
    val outputLen = (input.size / ratio).toInt()
    val output = FloatArray(outputLen)
    for (i in 0 until outputLen) {
      val srcPos = i * ratio
      val srcIdx = srcPos.toInt()
      val frac = (srcPos - srcIdx).toFloat()
      output[i] =
        if (srcIdx + 1 < input.size) {
          input[srcIdx] * (1f - frac) + input[srcIdx + 1] * frac
        } else {
          input[srcIdx.coerceAtMost(input.size - 1)]
        }
    }
    return output
  }
}

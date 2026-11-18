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

package com.google.ai.edge.examples.voice_to_text

import java.nio.ByteBuffer
import java.nio.ByteOrder

/** A WAV file the app cannot use; the message says why. */
class WavException(message: String) : IllegalArgumentException(message)

/**
 * WAV files in memory. The model takes 16 kHz mono audio; the app sends each recording as a 32-bit
 * float WAV (format tag 3), which LiteRT-LM reads as it is. 16-bit samples become floats as x /
 * 32768, exactly, so a 16-bit file gives the model the same numbers either way.
 */
object Wav {
  const val SAMPLE_RATE = 16_000

  private const val PCM = 1
  private const val IEEE_FLOAT = 3
  private const val EXTENSIBLE = 0xFFFE

  /** [samples] in [-1, 1] as a 16 kHz mono float32 WAV. */
  fun encodeFloat(samples: FloatArray, sampleRate: Int = SAMPLE_RATE): ByteArray {
    val dataBytes = samples.size * 4
    val buffer = ByteBuffer.allocate(44 + dataBytes).order(ByteOrder.LITTLE_ENDIAN)
    buffer.put("RIFF".toByteArray(Charsets.US_ASCII))
    buffer.putInt(36 + dataBytes)
    buffer.put("WAVE".toByteArray(Charsets.US_ASCII))
    buffer.put("fmt ".toByteArray(Charsets.US_ASCII))
    buffer.putInt(16)
    buffer.putShort(IEEE_FLOAT.toShort())
    buffer.putShort(1) // channels
    buffer.putInt(sampleRate)
    buffer.putInt(sampleRate * 4) // byte rate
    buffer.putShort(4) // block align
    buffer.putShort(32) // bits per sample
    buffer.put("data".toByteArray(Charsets.US_ASCII))
    buffer.putInt(dataBytes)
    for (s in samples) buffer.putFloat(s)
    return buffer.array()
  }

  /** 16-bit PCM samples as floats in [-1, 1). */
  fun toFloat(pcm: ShortArray, count: Int = pcm.size): FloatArray =
    FloatArray(count) { pcm[it] / 32768f }

  /**
   * The samples of a 16 kHz mono WAV with 16-bit PCM or 32-bit float data (also in the extensible
   * format). Used by the debug plan runs, which feed files into the path a recording takes.
   */
  fun decode(bytes: ByteArray): FloatArray {
    val b = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    if (bytes.size < 12 || tag(bytes, 0) != "RIFF" || tag(bytes, 8) != "WAVE") {
      throw WavException("not a WAV file")
    }
    var pos = 12
    var format = -1
    var channels = 0
    var rate = 0
    var bits = 0
    while (pos + 8 <= bytes.size) {
      val id = tag(bytes, pos)
      val size = b.getInt(pos + 4).toLong() and 0xFFFFFFFFL
      val body = pos + 8
      if (body + size > bytes.size) {
        throw WavException("the $id chunk runs past the end of the file")
      }
      when (id) {
        "fmt " -> {
          if (size < 16) throw WavException("the fmt chunk is too short")
          format = b.getShort(body).toInt() and 0xFFFF
          channels = b.getShort(body + 2).toInt()
          rate = b.getInt(body + 4)
          bits = b.getShort(body + 14).toInt()
          if (format == EXTENSIBLE) {
            if (size < 40) throw WavException("the extensible fmt chunk is too short")
            format = b.getShort(body + 24).toInt() and 0xFFFF
          }
        }
        "data" -> {
          if (format < 0) throw WavException("the data chunk comes before the fmt chunk")
          if (channels != 1 || rate != SAMPLE_RATE) {
            throw WavException("$channels channel(s) at $rate Hz; the model takes 16 kHz mono")
          }
          val n = size.toInt()
          return when {
            format == PCM && bits == 16 -> FloatArray(n / 2) { b.getShort(body + 2 * it) / 32768f }
            format == IEEE_FLOAT && bits == 32 -> FloatArray(n / 4) { b.getFloat(body + 4 * it) }
            else -> throw WavException("format $format with $bits bits; use 16-bit PCM or 32-bit float")
          }
        }
      }
      pos = body + size.toInt() + (size.toInt() and 1)
    }
    throw WavException("no data chunk")
  }

  private fun tag(bytes: ByteArray, at: Int) = String(bytes, at, 4, Charsets.US_ASCII)
}

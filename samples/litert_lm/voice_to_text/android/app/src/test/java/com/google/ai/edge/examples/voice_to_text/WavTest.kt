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
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class WavTest {
  /** A 16 kHz mono WAV with [format] (1 = PCM, 3 = float) and the given data bytes. */
  private fun wav(format: Int, bits: Int, data: ByteArray, rate: Int = 16_000, channels: Int = 1): ByteArray {
    val b = ByteBuffer.allocate(44 + data.size).order(ByteOrder.LITTLE_ENDIAN)
    b.put("RIFF".toByteArray()).putInt(36 + data.size).put("WAVE".toByteArray())
    b.put("fmt ".toByteArray()).putInt(16).putShort(format.toShort()).putShort(channels.toShort())
    b.putInt(rate).putInt(rate * channels * bits / 8).putShort((channels * bits / 8).toShort())
    b.putShort(bits.toShort())
    b.put("data".toByteArray()).putInt(data.size).put(data)
    return b.array()
  }

  @Test
  fun floatRoundTripIsExact() {
    val samples = floatArrayOf(0f, 0.5f, -0.25f, 1f, -1f, 1.2345e-5f)
    val bytes = Wav.encodeFloat(samples)
    assertEquals(44 + samples.size * 4, bytes.size)
    assertEquals(3, ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).getShort(20).toInt())
    assertArrayEquals(samples, Wav.decode(bytes), 0f)
  }

  @Test
  fun pcm16BecomesXOver32768() {
    val shorts = shortArrayOf(0, 1, -1, 32767, -32768, 1234)
    val data = ByteBuffer.allocate(shorts.size * 2).order(ByteOrder.LITTLE_ENDIAN)
    shorts.forEach { data.putShort(it) }
    val decoded = Wav.decode(wav(1, 16, data.array()))
    assertArrayEquals(FloatArray(shorts.size) { shorts[it] / 32768f }, decoded, 0f)
    // The recorder's conversion gives the same floats.
    assertArrayEquals(Wav.toFloat(shorts), decoded, 0f)
  }

  @Test
  fun skipsOtherChunksBeforeData() {
    val samples = floatArrayOf(0.1f, -0.2f)
    val plain = Wav.encodeFloat(samples)
    // Insert a LIST chunk (odd size, padded) between fmt and data.
    val list = byteArrayOf('L'.code.toByte(), 'I'.code.toByte(), 'S'.code.toByte(), 'T'.code.toByte(), 3, 0, 0, 0, 1, 2, 3, 0)
    val withList = plain.copyOfRange(0, 36) + list + plain.copyOfRange(36, plain.size)
    assertArrayEquals(samples, Wav.decode(withList), 0f)
  }

  @Test
  fun refusesStereoOtherRatesAndOtherFormats() {
    for ((bytes, word) in
      listOf(
        wav(1, 16, ByteArray(8), channels = 2) to "16 kHz mono",
        wav(1, 16, ByteArray(8), rate = 44_100) to "16 kHz mono",
        wav(1, 24, ByteArray(9)) to "16-bit PCM or 32-bit float",
        "not a wav".toByteArray() to "not a WAV",
      )) {
      try {
        Wav.decode(bytes)
        fail("accepted: $word")
      } catch (e: WavException) {
        assertTrue(e.message!!, e.message!!.contains(word))
      }
    }
  }
}

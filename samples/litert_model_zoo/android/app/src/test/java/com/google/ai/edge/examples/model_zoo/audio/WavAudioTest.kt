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
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Test

class WavAudioTest {
  @Test
  fun preservesOriginalRateForMusicFrontends() {
    val audio = WavAudio.readMono(pcm16Wav(shortArrayOf(16384, 16384, -16384, -16384), 2, 22050))
    assertEquals(22050, audio.sampleRate)
    assertArrayEquals(floatArrayOf(0.5f, -0.5f), audio.samples, 0f)
  }

  @Test
  fun readsSignedPcm16AndDownmixesStereo() {
    val wav = pcm16Wav(shortArrayOf(32767, -32768, 16384, 16384), 2, 16000)
    assertArrayEquals(floatArrayOf(-1f / 65536f, 0.5f), WavAudio.read16k(wav), 0f)
  }

  @Test
  fun resamplesRampUsingZooInterpolation() {
    assertArrayEquals(
      floatArrayOf(0f, 0.5f, 1f, 0.5f, 0f, 0f),
      WavAudio.resample(floatArrayOf(0f, 1f, 0f), 8000, 16000),
      0f,
    )
  }

  @Test(expected = IllegalArgumentException::class)
  fun rejectsTruncatedWaveData() {
    WavAudio.read16k(pcm16Wav(shortArrayOf(0, 1), 1, 16000).dropLast(1).toByteArray())
  }

  private fun pcm16Wav(samples: ShortArray, channels: Int, sampleRate: Int): ByteArray {
    val bytes = samples.size * 2
    val b = ByteBuffer.allocate(44 + bytes).order(ByteOrder.LITTLE_ENDIAN)
    b.put("RIFF".toByteArray()).putInt(36 + bytes).put("WAVE".toByteArray())
    b.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(channels.toShort())
    b.putInt(sampleRate)
      .putInt(sampleRate * channels * 2)
      .putShort((channels * 2).toShort())
      .putShort(16)
    b.put("data".toByteArray()).putInt(bytes)
    samples.forEach { b.putShort(it) }
    return b.array()
  }
}

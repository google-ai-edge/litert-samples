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

import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Rule
import org.junit.Test
import org.junit.rules.TemporaryFolder

class SpeechOutputCacheTest {
  @get:Rule val temporary = TemporaryFolder()

  @Test
  fun floatWaveHeaderAndSamplesRoundTripWithoutQuantization() {
    val samples = floatArrayOf(0f, -0f, 0.12345679f, -0.875f, 1.25f, -1.25f)
    val bytes =
      ByteArrayOutputStream().also { FloatWaveWriter.write(samples, 22050, it) }.toByteArray()
    val data = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
    assertEquals("RIFF", String(bytes, 0, 4, Charsets.US_ASCII))
    assertEquals(bytes.size - 8, data.getInt(4))
    assertEquals(18, data.getInt(16))
    assertEquals(3, data.getShort(20).toInt())
    assertEquals(1, data.getShort(22).toInt())
    assertEquals(22050, data.getInt(24))
    assertEquals(22050 * 4, data.getInt(28))
    assertEquals(4, data.getShort(32).toInt())
    assertEquals(32, data.getShort(34).toInt())
    assertEquals("fact", String(bytes, 38, 4, Charsets.US_ASCII))
    assertEquals(samples.size, data.getInt(46))
    assertEquals("data", String(bytes, 50, 4, Charsets.US_ASCII))
    assertEquals(samples.size * 4, data.getInt(54))
    val decoded = WavAudio.readMono(bytes)
    assertEquals(22050, decoded.sampleRate)
    assertArrayEquals(
      samples.map(Float::toRawBits).toIntArray(),
      decoded.samples.map(Float::toRawBits).toIntArray(),
    )
  }

  @Test
  fun streamedWavePreservesSamplesAcrossBufferBoundaries() {
    val samples = FloatArray(2401) { (it - 1200) / 1200f }
    val bytes =
      ByteArrayOutputStream().also { FloatWaveWriter.write(samples, 48000, it) }.toByteArray()
    assertArrayEquals(samples, WavAudio.readMono(bytes).samples, 0f)
    assertEquals(58 + samples.size * 4, bytes.size)
  }

  @Test
  fun cacheReplacesLastOutputAndExportsExactlyThoseBytes() {
    val directory = temporary.newFolder()
    val cache = SpeechOutputCache(directory)
    cache.save(floatArrayOf(0.5f), 16000)
    val latest = floatArrayOf(-0.3f, 0.7f)
    cache.save(latest, 22050)
    val exported = ByteArrayOutputStream().also(cache::copyTo).toByteArray()
    assertArrayEquals(directory.resolve("last-speech.wav").readBytes(), exported)
    assertArrayEquals(latest, WavAudio.readMono(exported).samples, 0f)
    assertEquals(listOf("last-speech.wav"), directory.list()?.toList())
  }

  @Test
  fun invalidOutputLeavesPreviousSuccessfulCacheIntact() {
    val directory = temporary.newFolder()
    val cache = SpeechOutputCache(directory)
    cache.save(floatArrayOf(0.25f), 22050)
    val before = directory.resolve("last-speech.wav").readBytes()
    assertTrue(runCatching { cache.save(floatArrayOf(Float.NaN), 22050) }.isFailure)
    assertArrayEquals(before, directory.resolve("last-speech.wav").readBytes())
    assertEquals(listOf("last-speech.wav"), directory.list()?.toList())
  }
}

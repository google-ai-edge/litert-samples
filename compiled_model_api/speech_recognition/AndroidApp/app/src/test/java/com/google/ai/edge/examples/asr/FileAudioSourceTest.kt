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

import java.io.ByteArrayInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds
import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class FileAudioSourceTest {

  @Test
  fun getAudioData_parsesWavCorrectly() {
    val sampleRate = 16000
    // 0.1 second of audio = 1600 samples
    val samples = ShortArray(1600) { i -> i.toShort() }
    val wavBytes = createWavBytes(samples, sampleRate)
    val inputStream = ByteArrayInputStream(wavBytes)

    val audioSource =
      FileAudioSource(
        inputStream = inputStream,
        samplingRate = sampleRate,
        interval = 100.milliseconds,
        overlap = 0.milliseconds,
      )

    val audioData = audioSource.getAudioData().map { it.clone() }.toList()

    assertEquals(1, audioData.size)
    assertEquals(1600, audioData[0].size)

    // Check first and last samples to ensure conversion is correct
    // MAX_PCM_VALUE = 32768.0f
    assertEquals((0f / 32768f).toDouble(), audioData[0][0].toDouble(), 1e-6)
    assertEquals((1599f / 32768f).toDouble(), audioData[0][1599].toDouble(), 1e-6)
  }

  @Test
  fun getAudioData_handlesOverlap() {
    val sampleRate = 16000
    // 0.2 seconds of audio = 3200 samples
    // interval = 0.1s (1600 samples), overlap = 0.05s (800 samples)
    // First chunk: 0 to 1600
    // Second chunk: 800 to 2400
    // Third chunk: 1600 to 3200
    val samples = ShortArray(3200) { i -> i.toShort() }
    val wavBytes = createWavBytes(samples, sampleRate)
    val inputStream = ByteArrayInputStream(wavBytes)

    val audioSource =
      FileAudioSource(
        inputStream = inputStream,
        samplingRate = sampleRate,
        interval = 100.milliseconds,
        overlap = 50.milliseconds,
      )

    val audioData = audioSource.getAudioData().map { it.clone() }.toList()

    assertEquals(3, audioData.size)

    // Chunk 1: [0, 1600)
    assertEquals(1600, audioData[0].size)
    assertEquals((0f / 32768f).toDouble(), audioData[0][0].toDouble(), 1e-6)
    assertEquals((1599f / 32768f).toDouble(), audioData[0][1599].toDouble(), 1e-6)

    // Chunk 2: [800, 2400)
    assertEquals(1600, audioData[1].size)
    assertEquals((800f / 32768f).toDouble(), audioData[1][0].toDouble(), 1e-6)
    assertEquals((2399f / 32768f).toDouble(), audioData[1][1599].toDouble(), 1e-6)

    // Chunk 3: [1600, 3200)
    assertEquals(1600, audioData[2].size)
    assertEquals((1600f / 32768f).toDouble(), audioData[2][0].toDouble(), 1e-6)
    assertEquals((3199f / 32768f).toDouble(), audioData[2][1599].toDouble(), 1e-6)
  }

  @Test
  fun getAudioData_shortAudio_returnsWhatItCan() {
    val sampleRate = 16000
    // 0.05 seconds of audio = 800 samples
    // interval = 0.1s (1600 samples)
    val samples = ShortArray(800) { i -> i.toShort() }
    val wavBytes = createWavBytes(samples, sampleRate)
    val inputStream = ByteArrayInputStream(wavBytes)

    val audioSource =
      FileAudioSource(
        inputStream = inputStream,
        samplingRate = sampleRate,
        interval = 100.milliseconds,
        overlap = 0.milliseconds,
      )

    val audioData = audioSource.getAudioData().map { it.clone() }.toList()

    // It should yield the available samples pre-allocated to full interval, with remaining slots
    // padded with zeros.
    assertEquals(1, audioData.size)
    assertEquals(1600, audioData[0].size)
    assertEquals((0f / 32768f).toDouble(), audioData[0][0].toDouble(), 1e-6)
    assertEquals((499f / 32768f).toDouble(), audioData[0][499].toDouble(), 1e-6)
    assertEquals((799f / 32768f).toDouble(), audioData[0][799].toDouble(), 1e-6)
    org.junit.Assert.assertTrue(audioData[0].slice(800 until 1600).all { it == 0f })
  }

  @Test
  fun constructor_throwsOnInvalidIntervalOverlap() {
    assertThrows(IllegalArgumentException::class.java) {
      FileAudioSource(
        inputStream = ByteArrayInputStream(ByteArray(0)),
        samplingRate = 16000,
        interval = 1.seconds,
        overlap = 1.seconds,
      )
    }
  }

  @Test
  fun getAudioData_throwsOnInvalidHeader() {
    val inputStream = ByteArrayInputStream("NOTAWAVFILE".toByteArray())
    val audioSource =
      FileAudioSource(
        inputStream = inputStream,
        samplingRate = 16000,
        interval = 1.seconds,
        overlap = 0.seconds,
      )

    val exception =
      assertThrows(IllegalArgumentException::class.java) { audioSource.getAudioData().toList() }
    assertEquals("Invalid WAV file: Too short", exception.message)
  }

  @Test
  fun getAudioData_throwsOnMismatchedSampleRate() {
    val samples = ShortArray(1600)
    val wavBytes = createWavBytes(samples, sampleRate = 44100)
    val inputStream = ByteArrayInputStream(wavBytes)

    val audioSource =
      FileAudioSource(
        inputStream = inputStream,
        samplingRate = 16000,
        interval = 1.seconds,
        overlap = 0.seconds,
      )

    val exception =
      assertThrows(IllegalArgumentException::class.java) { audioSource.getAudioData().toList() }
    // "Unsupported sample rate: expected 16000, but got 44100"
    assertEquals("Unsupported sample rate: expected 16000, but got 44100", exception.message)
  }

  private fun <T : Throwable> assertThrows(expectedType: Class<T>, executable: () -> Unit): T {
    try {
      executable()
    } catch (actualException: Throwable) {
      if (expectedType.isInstance(actualException)) {
        return actualException as T
      } else {
        throw AssertionError(
          "Expected ${expectedType.name} but caught ${actualException::class.java.name}",
          actualException,
        )
      }
    }
    throw AssertionError("Expected ${expectedType.name} to be thrown, but nothing was thrown")
  }

  private fun createWavBytes(samples: ShortArray, sampleRate: Int = 16000): ByteArray {
    val dataSize = samples.size * 2
    val buffer = ByteBuffer.allocate(44 + dataSize).order(ByteOrder.LITTLE_ENDIAN)

    // RIFF header
    buffer.put("RIFF".toByteArray())
    buffer.putInt(36 + dataSize)
    buffer.put("WAVE".toByteArray())

    // fmt chunk
    buffer.put("fmt ".toByteArray())
    buffer.putInt(16) // chunk size
    buffer.putShort(1.toShort()) // PCM
    buffer.putShort(1.toShort()) // Mono
    buffer.putInt(sampleRate)
    buffer.putInt(sampleRate * 2) // Byte rate
    buffer.putShort(2.toShort()) // Block align
    buffer.putShort(16.toShort()) // Bits per sample

    // data chunk
    buffer.put("data".toByteArray())
    buffer.putInt(dataSize)
    for (sample in samples) {
      buffer.putShort(sample)
    }

    return buffer.array()
  }
}

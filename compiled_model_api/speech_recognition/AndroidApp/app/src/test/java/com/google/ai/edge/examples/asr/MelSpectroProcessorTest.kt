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

import kotlin.math.PI
import kotlin.math.sin
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class MelSpectroProcessorTest {

  @Test
  fun process_emptyAudio_returnsEmptyArray() {
    val processor = MelSpectroProcessor(16000, LogMelSpectroConfig(nFFT = 512))
    val result = processor.process(FloatArray(0))
    assertEquals(0, result.size)
  }

  @Test
  fun process_dummyAudio_returnsCorrectShape() {
    val processor = MelSpectroProcessor(16000, LogMelSpectroConfig(nFFT = 512))

    // Generate a simple 1 second 440Hz sine wave as a dummy float array
    val sampleRate = 16000
    val durationSeconds = 1
    val audio =
      FloatArray(sampleRate * durationSeconds) { i ->
        sin(2.0 * PI * 440.0 * i / sampleRate).toFloat()
      }

    val result = processor.process(audio)
    assertTrue(result.isNotEmpty())
    // Because result is flattened (frames * featureSize), we expect its size to be a multiple of
    // featureSize.
    val featureSize = 80
    assertEquals(0, result.size % featureSize)

    // Ensure values are within a reasonable floating point range and not all zero
    val hasNonZero = result.any { it != 0.0f }
    assertTrue("Expected non-zero values in the processed output", hasNonZero)
  }

  @Test
  fun process_withMockMelSpectrogram_normalizesCorrectly() {
    // 3 frames, featureSize=2.
    // validFrames will be audio.size / hopLength = 3/160.
    // If audio size is 3*160 = 480, validFrames = 3.
    // for feature 0: mean=(1+2+3)/3=2, std=sqrt(((1-2)^2+(2-2)^2+(3-2)^2)/(3-1)) = sqrt((1+0+1)/2)
    val nFrames = 3
    val nMels = 80
    val hopLength = 160
    val audio = FloatArray(nFrames * hopLength)
    val melSpec = Array(nMels) { FloatArray(nFrames) }
    // melSpec[0] = [1,2,3], melSpec[1] = [4,5,6], others are 0.
    melSpec[0][0] = 1.0f
    melSpec[0][1] = 2.0f
    melSpec[0][2] = 3.0f
    melSpec[1][0] = 4.0f
    melSpec[1][1] = 5.0f
    melSpec[1][2] = 6.0f

    // For m=0: mean=2, std=1. Normalized: -1, 0, 1
    // For m=1: mean=5, std=1. Normalized: -1, 0, 1
    // For m>1: mean=0, std=0. Normalized: 0, 0, 0
    val processor =
      MelSpectroProcessor(
        samplingRate = 16000,
        config = LogMelSpectroConfig(nFFT = 512, transpose = true),
        generateMelSpectrogram = { _, _, _, _, _ -> melSpec },
      )
    val result = processor.process(audio)

    val expected = FloatArray(nFrames * nMels)
    // Frame 0
    expected[0] = -1.0f
    expected[1] = -1.0f
    // Frame 1
    expected[80] = 0.0f
    expected[81] = 0.0f
    // Frame 2
    expected[160] = 1.0f
    expected[161] = 1.0f

    assertArrayEquals(expected, result, 1.01e-5f)
  }

  @Test
  fun process_withWhisperNormalization_clipsAndScalesCorrectly() {
    val nFrames = 3
    val nMels = 80
    val hopLength = 160
    val audio = FloatArray(nFrames * hopLength)
    val melSpec = Array(nMels) { FloatArray(nFrames) }
    melSpec[0][0] = 0.0f
    melSpec[0][1] = 1.0f
    melSpec[0][2] = 2.0f

    val processor =
      MelSpectroProcessor(
        samplingRate = 16000,
        config = LogMelSpectroConfig(nFFT = 512, normType = "whisper"),
        generateMelSpectrogram = { _, _, _, _, _ -> melSpec },
      )
    val result = processor.process(audio)

    val expected = FloatArray(nFrames * nMels) { 1.0f }
    // Mock outputs (0.0, 1.0, 2.0) simulate ln outputs from generator.
    // They are divided by LN_10 in process() to translate to log10.
    // For input 1.0f: (1.0f / ln(10.0f) + 4.0f) / 4.0f = 1.1085736f
    // For input 2.0f: (2.0f / ln(10.0f) + 4.0f) / 4.0f = 1.2171472f
    expected[1] = 1.1085736f
    expected[2] = 1.2171472f

    assertArrayEquals(expected, result, 1.01e-5f)
  }

  @Test
  fun process_withNonPowerOfTwoNFFT_padsCorrectly() {
    var capturedNFFT = 0
    val processor =
      MelSpectroProcessor(
        samplingRate = 16000,
        config = LogMelSpectroConfig(nFFT = 400),
        generateMelSpectrogram = { _, _, nfft, _, _ ->
          capturedNFFT = nfft
          Array(80) { FloatArray(1) }
        },
      )
    processor.process(FloatArray(160))
    assertEquals(512, capturedNFFT)
  }

  @Test
  fun process_padsToExpectedNumberOfFrames() {
    val nFrames = 3
    val hopLength = 160
    val nFramesInConfig = 10
    val audio = FloatArray(nFrames * hopLength)
    val melSpec = Array(80) { FloatArray(nFrames) }

    // Using defaults (transpose = false, normType = "standard")
    val config = LogMelSpectroConfig(nFFT = 512, nFrames = nFramesInConfig)
    val processor =
      MelSpectroProcessor(
        samplingRate = 16000,
        config = config,
        generateMelSpectrogram = { _, _, _, _, _ -> melSpec },
      )
    val result = processor.process(audio)

    // Expected size: nFramesInConfig (10) * nMels (80) = 800
    assertEquals(800, result.size)

    // With transpose = false, layout is [mels][frames], stride is nFramesInConfig (10).
    // Mel 0, frame 9 is at index 9. Should be padded with 0.
    assertEquals(0.0f, result[9])

    // Mel 1, frame 9 is at index 19. Should be padded with 0.
    assertEquals(0.0f, result[19])
  }
}

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

import kotlin.time.Duration.Companion.seconds
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class MicrophoneAudioSourceTest {

  /** A mock AudioReader that provides precise control over read results. */
  class MockAudioReader(private val responses: List<ReadResponse>) : AudioReader {
    private var responseIndex = 0

    override fun close() {}

    override fun read(audioData: ShortArray, offsetInShorts: Int, sizeInShorts: Int): Int {
      if (responseIndex >= responses.size) return 0

      val response = responses[responseIndex++]
      if (response.result > 0 && response.data != null) {
        // Ensure we don't exceed the requested size or our mock data size
        val toCopy = minOf(response.result, response.data.size, sizeInShorts)
        System.arraycopy(response.data, 0, audioData, offsetInShorts, toCopy)
        return toCopy
      }
      return response.result
    }
  }

  data class ReadResponse(val result: Int, val data: ShortArray? = null)

  @Test
  fun testGetAudioData_yieldsCorrectChunks() {
    val stepSamples = 8000
    val responses =
      listOf(
        ReadResponse(stepSamples, ShortArray(stepSamples) { 1000 }),
        ReadResponse(stepSamples, ShortArray(stepSamples) { 2000 }),
        ReadResponse(stepSamples, ShortArray(stepSamples) { 3000 }),
        ReadResponse(0),
      )

    val source = createSource(responses, overlap = 0.5.seconds)
    val results = source.getAudioData().map { it.clone() }.toList()

    assertEquals(3, results.size)
    assertEquals(24000, results[0].size)
    assertTrue(results[0].take(16000).all { it == 0f })
    assertTrue(results[0].takeLast(8000).all { it == (1000f / 32768f) })
    assertEquals(24000, results[1].size)
    assertTrue(results[1].take(8000).all { it == 0f })
    assertTrue(results[1].slice(8000 until 16000).all { it == (1000f / 32768f) })
    assertTrue(results[1].takeLast(8000).all { it == (2000f / 32768f) })
    assertEquals(24000, results[2].size)
    assertTrue(results[2].slice(0 until 8000).all { it == (1000f / 32768f) })
    assertTrue(results[2].slice(8000 until 16000).all { it == (2000f / 32768f) })
    assertTrue(results[2].takeLast(8000).all { it == (3000f / 32768f) })
  }

  @Test
  fun testSilenceDetection_zerosOutSilentChunks() {
    val stepSamples = 16000
    val responses =
      listOf(
        ReadResponse(stepSamples, ShortArray(stepSamples) { 1000 }), // Loud
        ReadResponse(stepSamples, ShortArray(stepSamples) { 10 }), // Silent
        ReadResponse(0),
      )

    val source = createSource(responses, overlap = 0.seconds)
    val results = source.getAudioData().map { it.clone() }.toList()

    assertEquals(2, results.size)
    assertEquals(32000, results[0].size)
    assertTrue(results[0].take(16000).all { it == 0f })
    assertTrue(results[0].takeLast(16000).all { it == (1000f / 32768f) })
    assertEquals(32000, results[1].size)
    assertTrue(results[1].take(16000).all { it == (1000f / 32768f) })
    assertTrue(results[1].takeLast(16000).all { it == 0f })
  }

  @Test
  fun testError_deadObject_terminatesSequence() {
    val stepSamples = 8000
    val responses =
      listOf(
        ReadResponse(stepSamples, ShortArray(stepSamples) { 1000 }),
        ReadResponse(-6), // AudioRecord.ERROR_DEAD_OBJECT
      )

    val source = createSource(responses)
    val results = source.getAudioData().toList()

    assertEquals(1, results.size)
  }

  @Test
  fun testError_invalidOperation_terminatesSequence() {
    val responses =
      listOf(
        ReadResponse(-3) // AudioRecord.ERROR_INVALID_OPERATION
      )

    val source = createSource(responses)
    val results = source.getAudioData().toList()

    assertTrue(results.isEmpty())
  }

  @Test
  fun testError_genericNegative_continues() {
    val stepSamples = 8000
    val responses =
      listOf(
        ReadResponse(-1), // Generic ERROR, should be ignored
        ReadResponse(stepSamples, ShortArray(stepSamples) { 5000 }),
        ReadResponse(0),
      )

    val source = createSource(responses)
    val results = source.getAudioData().map { it.clone() }.toList()

    assertEquals(1, results.size)
    assertEquals(24000, results[0].size)
    assertTrue(results[0].take(16000).all { it == 0f })
    assertTrue(results[0].takeLast(8000).all { it == (5000f / 32768f) })
  }

  @Test
  fun testPartialReads_accumulateBeforeYield() {
    val interval = 0.5.seconds
    val overlap = 0.seconds
    val stepSamples = 8000

    val responses =
      listOf(
        ReadResponse(4000, ShortArray(4000) { 1000 }), // Half a step
        ReadResponse(4000, ShortArray(4000) { 2000 }), // Second half
        ReadResponse(0), // End of data
      )

    val source = createSource(responses, interval = interval, overlap = overlap)
    val results = source.getAudioData().map { it.clone() }.toList()

    assertEquals(1, results.size)
    assertEquals(16000, results[0].size)
    assertTrue(results[0].take(8000).all { it == 0f })
    assertTrue(results[0].slice(8000 until 12000).all { it == (1000f / 32768f) })
    assertTrue(results[0].slice(12000 until 16000).all { it == (2000f / 32768f) })
  }

  private fun createSource(
    responses: List<ReadResponse>,
    samplingRate: Int = 16000,
    interval: kotlin.time.Duration = 1.seconds,
    overlap: kotlin.time.Duration = 0.5.seconds,
  ) =
    MicrophoneAudioSource(
      samplingRate = samplingRate,
      interval = interval,
      overlap = overlap,
      getAudioReader = { _ -> MockAudioReader(responses) },
    )

  @Test(expected = IllegalArgumentException::class)
  fun testInit_invalidOverlap_throwsException() {
    createSource(emptyList(), interval = 1.seconds, overlap = 1.seconds)
  }

  @Test(expected = IllegalStateException::class)
  fun testInit_audioRecordStateUninitialized_throwsException() {
    val source =
      MicrophoneAudioSource(
        samplingRate = 16000,
        interval = 1.seconds,
        overlap = 0.5.seconds,
        getAudioReader = { _ -> throw IllegalStateException("AudioRecord initialization failed") },
      )
    source.getAudioData().toList()
  }

  @Test
  fun testLiveAudioRecording_yieldsOverlappedChunks() {
    val stepSamples = 8000 // 0.5 seconds at 16kHz sampling rate
    val responses =
      listOf(
        ReadResponse(stepSamples, ShortArray(stepSamples) { 1000 }),
        ReadResponse(stepSamples, ShortArray(stepSamples) { 2000 }),
        ReadResponse(stepSamples, ShortArray(stepSamples) { 3000 }),
        ReadResponse(0),
      )

    // As overlap is 0.5 seconds, full size of responses will be 1s long, i.e. 16000 samples.
    val source = createSource(responses, overlap = 0.5.seconds)
    val results = source.getAudioData().map { it.clone() }.toList()

    assertEquals(3, results.size)

    assertEquals(24000, results[0].size)
    assertTrue(results[0].take(16000).all { it == 0f })
    assertTrue(results[0].takeLast(8000).all { it == (1000f / 32768f) })

    assertEquals(24000, results[1].size)
    assertTrue(results[1].take(8000).all { it == 0f })
    assertTrue(results[1].slice(8000 until 16000).all { it == (1000f / 32768f) })
    assertTrue(results[1].takeLast(8000).all { it == (2000f / 32768f) })

    assertEquals(24000, results[2].size)
    assertTrue(results[2].slice(0 until 8000).all { it == (1000f / 32768f) })
    assertTrue(results[2].slice(8000 until 16000).all { it == (2000f / 32768f) })
    assertTrue(results[2].slice(16000 until 24000).all { it == (3000f / 32768f) })
  }

  @Test
  fun testClose_succeedsWithoutException() {
    val source = createSource(emptyList())
    source.close()
  }

  @Test
  fun testError_badValueTransient_continuesAndRecovers() {
    val stepSamples = 8000
    val responses =
      listOf(
        ReadResponse(-2), // AudioRecord.ERROR_BAD_VALUE
        ReadResponse(stepSamples, ShortArray(stepSamples) { 4000 }),
        ReadResponse(0),
      )

    val source = createSource(responses)
    val results = source.getAudioData().map { it.clone() }.toList()

    assertEquals(1, results.size)
    assertEquals(24000, results[0].size)
    assertTrue(results[0].take(16000).all { it == 0f })
    assertTrue(results[0].takeLast(8000).all { it == (4000f / 32768f) })
  }

  @Test
  fun testIntervalNotMultipleOfStep_accumulatesCorrectly() {
    val interval = 1.2.seconds // 19200 samples at 16kHz
    val overlap = 0.5.seconds // 8000 samples at 16kHz
    val stepSamples = 11200 // 19200 - 8000

    val responses =
      listOf(
        ReadResponse(stepSamples, ShortArray(stepSamples) { 1000 }),
        ReadResponse(stepSamples, ShortArray(stepSamples) { 2000 }),
        ReadResponse(stepSamples, ShortArray(stepSamples) { 3000 }),
        ReadResponse(0),
      )

    val source = createSource(responses, interval = interval, overlap = overlap)
    val results = source.getAudioData().map { it.clone() }.toList()

    assertEquals(3, results.size)
    assertEquals(30400, results[0].size)
    assertTrue(results[0].take(19200).all { it == 0f })
    assertTrue(results[0].takeLast(11200).all { it == (1000f / 32768f) })

    assertEquals(30400, results[1].size)
    assertTrue(results[1].take(8000).all { it == 0f })
    assertTrue(results[1].slice(8000 until 19200).all { it == (1000f / 32768f) })
    assertTrue(results[1].takeLast(11200).all { it == (2000f / 32768f) })

    assertEquals(30400, results[2].size)
    assertTrue(results[2].slice(0 until 8000).all { it == (1000f / 32768f) })
    assertTrue(results[2].slice(8000 until 19200).all { it == (2000f / 32768f) })
    assertTrue(results[2].slice(19200 until 30400).all { it == (3000f / 32768f) })
  }
}

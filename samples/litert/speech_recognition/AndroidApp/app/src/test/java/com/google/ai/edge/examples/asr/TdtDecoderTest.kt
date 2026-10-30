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
import kotlin.math.PI
import kotlin.math.sin
import kotlin.time.Duration.Companion.seconds
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class TdtDecoderTest {

  /** A token and a duration the decoder network predicts at one position of its logits. */
  private data class Prediction(val tokenId: Int, val duration: Int)

  /** One run of the network: the signature, the token IDs, the input states buffers used. */
  private data class Run(val signature: String, val tokenIds: List<Int>, val statesIndex: Int)

  private class LoopDidNotEndException : RuntimeException()

  /**
   * Stands in for the decode and decode_1 signatures. Every run returns logits for every time index
   * and token slot, all 0 except a 1 at the token and at the duration [predict] gives for that run,
   * time index and slot, so a loop reading a wrong position reads another prediction. Records the
   * runs, and throws after [maxRuns] of them (the loop has not ended).
   */
  private class FakeNetwork(
    val maxTimeIndex: Int,
    val numOfTokenIds: Int,
    val predict: (run: Int, timeIndex: Int, slot: Int) -> Prediction,
    val maxRuns: Int = 100_000,
  ) : TdtDecoder.Network {
    val runs = ArrayList<Run>()
    var inputStatesBuffersIndex = 0
    // Refilled by every run: the loops read the logits of a run before the next run.
    private val decodeLogits = FloatArray(maxTimeIndex * numOfTokenIds * NUM_LOGITS_PER_TOKEN)
    private val decode1Logits = FloatArray(maxTimeIndex * NUM_LOGITS_PER_TOKEN)

    override fun run(signature: String, tokenIds: IntArray): FloatArray {
      if (runs.size >= maxRuns) throw LoopDidNotEndException()
      val run = runs.size
      runs.add(Run(signature, tokenIds.toList(), inputStatesBuffersIndex))
      val isDecode1 = signature == TdtDecoder.DECODE_1_SIGNATURE
      val numOfSlots = if (isDecode1) 1 else numOfTokenIds
      val logits = if (isDecode1) decode1Logits else decodeLogits
      logits.fill(0.0f)
      for (timeIndex in 0 until maxTimeIndex) {
        for (slot in 0 until numOfSlots) {
          val prediction = predict(run, timeIndex, slot)
          val start = (timeIndex * numOfSlots + slot) * NUM_LOGITS_PER_TOKEN
          logits[start + prediction.tokenId] = 1.0f
          logits[start + VOCAB_SIZE + prediction.duration] = 1.0f
        }
      }
      return logits
    }

    override fun adoptStates() {
      inputStatesBuffersIndex = 1 - inputStatesBuffersIndex
    }
  }

  /**
   * The decoding loop of TdtDecoder.decode before it moved into TdtDecoder.greedyDecode (upstream
   * e3be1469), line for line, with its writeInt / run / readFloat calls replaced by
   * [FakeNetwork.run] and its state buffers swap by [FakeNetwork.adoptStates]: every time index of
   * the window, no cap on the tokens at one time index.
   */
  private fun previousDecode(
    network: FakeNetwork,
    hasDecode1: Boolean,
    maxTimeIndex: Int = network.maxTimeIndex,
    numOfTokenIds: Int = network.numOfTokenIds,
    numOfLogitsPerToken: Int = NUM_LOGITS_PER_TOKEN,
    blankTokenId: Int = BLANK_TOKEN_ID,
    decodeStartTokenId: Int = BLANK_TOKEN_ID,
  ): List<Pair<Int, Int>> = sequence {
    var inferenceSignature = LiteRtRunner.DECODE_SIGNATURE
    var numOfInferenceTokenIds = numOfTokenIds
    var inferenceTokenIds = IntArray(numOfTokenIds) { 0 }
    inferenceTokenIds[0] = decodeStartTokenId

    var tokenIndex: Int = 0
    var timeIndex: Int = 0
    while (timeIndex < maxTimeIndex) {
      val logits = network.run(inferenceSignature, inferenceTokenIds)
      val startIndexInCurrentTimeIndex = timeIndex * numOfInferenceTokenIds * numOfLogitsPerToken
      // Do argmax among the logits of the current token.
      val startIndexOfTokenId = startIndexInCurrentTimeIndex + tokenIndex * numOfLogitsPerToken
      val endIndexOfDuration = startIndexInCurrentTimeIndex + (tokenIndex + 1) * numOfLogitsPerToken
      val endIndexOfTokenId = endIndexOfDuration - TdtDecoder.NUM_DURATIONS
      val tokenId =
        ((startIndexOfTokenId until endIndexOfTokenId).maxByOrNull { logits[it] }
          ?: startIndexOfTokenId) - startIndexOfTokenId
      if (tokenId != blankTokenId) {
        yield(Pair(tokenId, timeIndex))
        if (numOfInferenceTokenIds > 1) {
          tokenIndex++
          if (tokenIndex < numOfInferenceTokenIds) {
            // Still room in the stateless token array.
          } else if (hasDecode1) { // Switch to decode_1 for stateful decoding.
            inferenceSignature = TdtDecoder.DECODE_1_SIGNATURE
            numOfInferenceTokenIds = 1
            inferenceTokenIds = IntArray(1) { 0 }
            tokenIndex = 0
          } else { // Stateless decoding reached the max token size.
            break
          }
        }
        inferenceTokenIds[tokenIndex] = tokenId
      }

      val duration =
        ((endIndexOfTokenId until endIndexOfDuration).maxByOrNull { logits[it] }
          ?: endIndexOfTokenId) - endIndexOfTokenId
      timeIndex += if (duration == 0 && tokenId == blankTokenId) 1 else duration

      if (numOfInferenceTokenIds == 1 && tokenId != blankTokenId) {
        network.adoptStates()
      }
    }

    yield(Pair(SpeechRecognizer.END_OF_SEQUENCE, maxTimeIndex))
  }
    .toList()

  private fun greedyDecode(
    network: FakeNetwork,
    hasDecode1: Boolean,
    timeLimit: Int = network.maxTimeIndex,
  ): List<Pair<Int, Int>> =
    TdtDecoder.greedyDecode(
        network,
        network.maxTimeIndex,
        timeLimit,
        network.numOfTokenIds,
        NUM_LOGITS_PER_TOKEN,
        blankTokenId = BLANK_TOKEN_ID,
        decodeStartTokenId = BLANK_TOKEN_ID,
        hasDecode1 = hasDecode1,
      )
      .toList()

  /**
   * Random predictions, fixed by [seed], different at every run, time index and slot: blank with
   * probability [pBlank], duration 0 with probability [pZeroDuration].
   */
  private fun randomPredictions(seed: Int, pBlank: Double, pZeroDuration: Double) =
    { run: Int, timeIndex: Int, slot: Int ->
      val h = hash(seed, run, timeIndex, slot)
      val tokenId =
        if ((h and 0xFFFF) / 65536.0 < pBlank) BLANK_TOKEN_ID
        else ((h ushr 16) and 0xFFFF).toInt() % BLANK_TOKEN_ID
      val duration =
        if (((h ushr 32) and 0xFFFF) / 65536.0 < pZeroDuration) 0
        else 1 + ((h ushr 48) and 0xFFFF).toInt() % (TdtDecoder.NUM_DURATIONS - 1)
      Prediction(tokenId, duration)
    }

  /** A 64-bit hash of four ints (SplitMix64's finalizer after each one). */
  private fun hash(a: Int, b: Int, c: Int, d: Int): Long {
    var h = 0x2545F4914F6CDD1DL
    for (x in intArrayOf(a, b, c, d)) {
      h = (h xor x.toLong()) * -0x40a7b892e31b1a47L
      h = (h xor (h ushr 27)) * -0x6b2fb644ecceee15L
      h = h xor (h ushr 31)
    }
    return h
  }

  /** The same prediction at every position of a run, by run number. */
  private fun byRun(predict: (run: Int) -> Prediction) = { run: Int, _: Int, _: Int ->
    predict(run)
  }

  /** Whether a decoding emitted [TdtDecoder.MAX_SYMBOLS_PER_STEP] tokens at one time index. */
  private fun reachesTheCap(tokens: List<Pair<Int, Int>>) =
    tokens.groupingBy { it.second }.eachCount().values.any { it >= TdtDecoder.MAX_SYMBOLS_PER_STEP }

  /** One decoding of the random comparisons. */
  private class RandomCase(
    val maxTimeIndex: Int,
    val numOfTokenIds: Int,
    val hasDecode1: Boolean,
    val predict: (run: Int, timeIndex: Int, slot: Int) -> Prediction,
  )

  /** 1440 random decodings over window lengths, token array sizes and prediction mixes. */
  private fun randomCases() = sequence {
    for (maxTimeIndex in listOf(1, 7, 63, 375)) {
      for (numOfTokenIds in listOf(1, 2, 4)) {
        for (hasDecode1 in listOf(true, false)) {
          for ((pBlank, pZeroDuration) in listOf(0.5 to 0.3, 0.8 to 0.5, 0.2 to 0.6)) {
            for (seed in 0 until if (maxTimeIndex == 375) 5 else 25) {
              val predict = randomPredictions(seed, pBlank, pZeroDuration)
              yield(RandomCase(maxTimeIndex, numOfTokenIds, hasDecode1, predict))
            }
          }
        }
      }
    }
  }

  // The whole window and the cap not reached: the loop is the previous loop.

  @Test
  fun greedyDecode_wholeWindowBelowTheCap_sameTokensAndRunsAsThePreviousLoop() {
    var compared = 0
    var tokens = 0
    var decode1Runs = 0
    for (case in randomCases()) {
      val previousNetwork = FakeNetwork(case.maxTimeIndex, case.numOfTokenIds, case.predict)
      val expected = previousDecode(previousNetwork, case.hasDecode1)
      if (!reachesTheCap(expected)) {
        val network = FakeNetwork(case.maxTimeIndex, case.numOfTokenIds, case.predict)
        assertEquals(expected, greedyDecode(network, case.hasDecode1))
        assertEquals(previousNetwork.runs, network.runs)
        assertEquals(previousNetwork.inputStatesBuffersIndex, network.inputStatesBuffersIndex)
        compared++
        tokens += expected.size - 1
        decode1Runs += network.runs.count { it.signature == TdtDecoder.DECODE_1_SIGNATURE }
      }
    }
    // Most of the 1440 decodings are compared, and they reach both signatures and emit tokens.
    assertTrue("compared $compared of 1440", compared > 1300)
    assertTrue(tokens > 2_000)
    assertTrue(decode1Runs > 300)
  }

  @Test
  fun getTimeLimit_allSamples_isTheWholeWindow() {
    assertEquals(375, TdtDecoder.getTimeLimit(SpeechRecognizer.ALL_SAMPLES, HOP_LENGTH, 375))
    assertEquals(63, TdtDecoder.getTimeLimit(SpeechRecognizer.ALL_SAMPLES, HOP_LENGTH, 63))
  }

  // The stop: nothing at or past the time limit is read, the end marker keeps maxTimeIndex.

  @Test
  fun greedyDecode_stopsBeforeTheTimeLimit() {
    val network = FakeNetwork(375, 4, byRun { Prediction(1, 1) }) // A token at every time index.
    assertEquals(
      (0 until 129).map { Pair(1, it) } + Pair(SpeechRecognizer.END_OF_SEQUENCE, 375),
      greedyDecode(network, hasDecode1 = true, timeLimit = 129),
    )
    assertEquals(129, network.runs.size)
  }

  @Test
  fun greedyDecode_aDurationPastTheTimeLimitEndsTheWindow() {
    val network = FakeNetwork(375, 4, byRun { Prediction(2, 4) })
    assertEquals(
      listOf(Pair(2, 0), Pair(2, 4), Pair(2, 8), Pair(SpeechRecognizer.END_OF_SEQUENCE, 375)),
      greedyDecode(network, hasDecode1 = true, timeLimit = 10),
    )
    assertEquals(3, network.runs.size)
  }

  @Test
  fun greedyDecode_readsNoTimeIndexPastTheTimeLimit() {
    // Random predictions before the limit and a non-blank token at every position from the limit
    // on, so that reading any of those positions would emit a token at or past the limit. Below the
    // cap, the tokens and runs are the previous loop's up to the limit.
    var compared = 0
    for (case in randomCases()) {
      val timeLimit = (case.maxTimeIndex + 1) / 2
      val predictUpToTheLimit = { run: Int, timeIndex: Int, slot: Int ->
        if (timeIndex < timeLimit) case.predict(run, timeIndex, slot)
        else Prediction(POISON_TOKEN_ID, 1)
      }
      val previousNetwork = FakeNetwork(case.maxTimeIndex, case.numOfTokenIds, case.predict)
      val previous = previousDecode(previousNetwork, case.hasDecode1)
      val network = FakeNetwork(case.maxTimeIndex, case.numOfTokenIds, predictUpToTheLimit)
      val tokens = greedyDecode(network, case.hasDecode1, timeLimit)
      assertEquals(Pair(SpeechRecognizer.END_OF_SEQUENCE, case.maxTimeIndex), tokens.last())
      assertTrue(tokens.dropLast(1).all { it.second < timeLimit })
      if (!reachesTheCap(previous)) {
        assertEquals(previous.filter { it.second < timeLimit }, tokens.dropLast(1))
        assertEquals(previousNetwork.runs.subList(0, network.runs.size), network.runs)
        compared++
      }
    }
    assertTrue("compared $compared of 1440", compared > 1300)
  }

  @Test
  fun getTimeLimit_fourTimeIndicesPastTheAudio() {
    // 30 s window: 375 time indices, 3000 feature frames of 160 samples, 8 frames per time index.
    assertEquals(375, TdtDecoder.getTimeLimit(480_000, HOP_LENGTH, 375))
    assertEquals(375, TdtDecoder.getTimeLimit(473_760, HOP_LENGTH, 375)) // 2961 frames
    assertEquals(374, TdtDecoder.getTimeLimit(473_600, HOP_LENGTH, 375)) // 2960 frames
    assertEquals(129, TdtDecoder.getTimeLimit(160_000, HOP_LENGTH, 375)) // 10 s
    assertEquals(130, TdtDecoder.getTimeLimit(160_160, HOP_LENGTH, 375))
    assertEquals(5, TdtDecoder.getTimeLimit(160, HOP_LENGTH, 375))
    assertEquals(4, TdtDecoder.getTimeLimit(159, HOP_LENGTH, 375))
    assertEquals(4, TdtDecoder.getTimeLimit(0, HOP_LENGTH, 375))
    // 5 s window: 63 time indices; a full window decodes all of them.
    assertEquals(63, TdtDecoder.getTimeLimit(80_000, HOP_LENGTH, 63))
    assertEquals(54, TdtDecoder.getTimeLimit(64_000, HOP_LENGTH, 63)) // 4 s
  }

  @Test
  fun getNumOfValidSamples_endsAtTheLastNonZeroSample() {
    assertEquals(0, getNumOfValidSamples(FloatArray(0)))
    assertEquals(0, getNumOfValidSamples(FloatArray(4)))
    assertEquals(1, getNumOfValidSamples(floatArrayOf(0.5f, 0.0f, 0.0f)))
    assertEquals(2, getNumOfValidSamples(floatArrayOf(0.0f, -0.25f)))
    assertEquals(2, getNumOfValidSamples(floatArrayOf(0.1f, 0.2f, -0.0f)))
    assertEquals(3, getNumOfValidSamples(floatArrayOf(0.1f, 0.0f, 0.2f)))
  }

  @Test
  fun getNumOfValidSamples_ofTheLastChunkOfAFile() {
    // A 10 s file in 30 s windows: one window, the audio and then zeros.
    val chunks = fileChunks(sine(seconds = 10.0), windowSeconds = 30)
    assertEquals(1, chunks.size)
    assertEquals(160_000, getNumOfValidSamples(chunks[0]))
    assertEquals(129, TdtDecoder.getTimeLimit(getNumOfValidSamples(chunks[0]), HOP_LENGTH, 375))
    // A 7 s file in 5 s windows overlapping by 2 s: a full window, then 2 s of audio after the
    // overlap.
    val chunks5s = fileChunks(sine(seconds = 7.0), windowSeconds = 5)
    assertEquals(2, chunks5s.size)
    assertEquals(80_000, getNumOfValidSamples(chunks5s[0]))
    assertEquals(64_000, getNumOfValidSamples(chunks5s[1]))
  }

  // The cap: at most MAX_SYMBOLS_PER_STEP non-blank tokens at one time index, then the next one.

  @Test
  fun greedyDecode_movesOnAfterTenTokensAtOneTimeIndex() {
    // Time indices 0 and 1 blank; then 25 non-blank tokens with duration 0; then blanks.
    val predict = byRun { run ->
      if (run < 2 || run >= 27) Prediction(BLANK_TOKEN_ID, 1) else Prediction(3, 0)
    }
    val network = FakeNetwork(6, 4, predict)
    assertEquals(
      List(10) { Pair(3, 2) } +
        List(10) { Pair(3, 3) } +
        List(5) { Pair(3, 4) } +
        Pair(SpeechRecognizer.END_OF_SEQUENCE, 6),
      greedyDecode(network, hasDecode1 = true),
    )
    // 2 blanks, 25 tokens, a blank at time index 4, one at time index 5.
    assertEquals(2 + 25 + 1 + 1, network.runs.size)
    // The first 4 tokens fill the stateless token array, the rest go through decode_1.
    assertEquals(2 + 4, network.runs.count { it.signature == LiteRtRunner.DECODE_SIGNATURE })
  }

  @Test
  fun greedyDecode_countsTheTokensOfEachTimeIndex() {
    // Time index 2: 9 tokens with duration 0, then a 10th with duration 1 (the loop moves on by
    // itself); time index 3: 10 tokens with duration 0 (the cap moves it on); then blanks.
    val network =
      FakeNetwork(
        6,
        4,
        byRun { run ->
          when {
            run < 2 -> Prediction(BLANK_TOKEN_ID, 1)
            run < 11 -> Prediction(3, 0)
            run == 11 -> Prediction(3, 1)
            run < 22 -> Prediction(4, 0)
            else -> Prediction(BLANK_TOKEN_ID, 1)
          }
        },
      )
    assertEquals(
      List(10) { Pair(3, 2) } + List(10) { Pair(4, 3) } + Pair(SpeechRecognizer.END_OF_SEQUENCE, 6),
      greedyDecode(network, hasDecode1 = true),
    )
    assertEquals(2 + 20 + 2, network.runs.size)
  }

  @Test
  fun greedyDecode_endsWhenTheModelKeepsPredictingAZeroDurationToken() {
    // After two blanks, a non-blank token with duration 0 at every step: the previous loop never
    // leaves time index 2.
    val predict = byRun { run -> if (run < 2) Prediction(BLANK_TOKEN_ID, 1) else Prediction(3, 0) }
    try {
      previousDecode(FakeNetwork(10, 4, predict, maxRuns = 10_000), hasDecode1 = true)
      fail("the previous loop ended")
    } catch (e: LoopDidNotEndException) {
      // Expected: 10 000 runs at time index 2.
    }
    for ((numOfTokenIds, hasDecode1) in listOf(4 to true, 1 to false)) {
      val network = FakeNetwork(10, numOfTokenIds, predict)
      val tokens = greedyDecode(network, hasDecode1)
      // Time indices 2 to 9, 10 tokens each.
      assertEquals((2 until 10).flatMap { t -> List(10) { Pair(3, t) } }, tokens.dropLast(1))
      assertEquals(2 + 80, network.runs.size)
    }
  }

  // Inputs at the edges: a silent window, a clip shorter than a second.

  @Test
  fun decode_silentWindow_decodesTheMarginOnly() {
    val chunks = fileChunks(ShortArray(32_000), windowSeconds = 30) // 2 s of zeros
    assertEquals(1, chunks.size)
    val timeLimit = TdtDecoder.getTimeLimit(getNumOfValidSamples(chunks[0]), HOP_LENGTH, 375)
    assertEquals(4, timeLimit)
    val predict = randomPredictions(seed = 7, pBlank = 0.5, pZeroDuration = 0.3)
    val tokens = greedyDecode(FakeNetwork(375, 4, predict), hasDecode1 = true, timeLimit)
    assertEquals(Pair(SpeechRecognizer.END_OF_SEQUENCE, 375), tokens.last())
    assertTrue(tokens.dropLast(1).all { it.second < 4 })
  }

  @Test
  fun decode_clipShorterThanOneSecond() {
    val chunks = fileChunks(sine(seconds = 0.5), windowSeconds = 30)
    assertEquals(1, chunks.size)
    assertEquals(8_000, getNumOfValidSamples(chunks[0]))
    val timeLimit = TdtDecoder.getTimeLimit(getNumOfValidSamples(chunks[0]), HOP_LENGTH, 375)
    assertEquals(11, timeLimit) // 50 feature frames: 7 time indices, then 4 more.
    val predict = randomPredictions(seed = 11, pBlank = 0.5, pZeroDuration = 0.3)
    val tokens = greedyDecode(FakeNetwork(375, 4, predict), hasDecode1 = true, timeLimit)
    assertEquals(Pair(SpeechRecognizer.END_OF_SEQUENCE, 375), tokens.last())
    assertTrue(tokens.dropLast(1).all { it.second < 11 })
  }

  /** A 440 Hz tone of [seconds] at 16 kHz, as 16-bit PCM. */
  private fun sine(seconds: Double) =
    ShortArray((SAMPLING_RATE * seconds).toInt()) { i ->
      (8000 * sin(2 * PI * 440 * i / SAMPLING_RATE)).toInt().toShort()
    }

  /** The chunks FileAudioSource reads from a WAV of [samples], as MainActivity reads a file. */
  private fun fileChunks(samples: ShortArray, windowSeconds: Int): List<FloatArray> {
    val buffer = ByteBuffer.allocate(44 + samples.size * 2).order(ByteOrder.LITTLE_ENDIAN)
    buffer.put("RIFF".toByteArray()).putInt(36 + samples.size * 2).put("WAVE".toByteArray())
    buffer.put("fmt ".toByteArray()).putInt(16).putShort(1).putShort(1)
    buffer.putInt(SAMPLING_RATE).putInt(SAMPLING_RATE * 2).putShort(2).putShort(16)
    buffer.put("data".toByteArray()).putInt(samples.size * 2)
    for (sample in samples) buffer.putShort(sample)
    return FileAudioSource(
        ByteArrayInputStream(buffer.array()),
        SAMPLING_RATE,
        windowSeconds.seconds,
        2.seconds,
      )
      .getAudioData()
      .map { it.clone() }
      .toList()
  }

  private companion object {
    // A small vocabulary with the blank token last, as in the parakeet exports.
    const val VOCAB_SIZE = 9
    const val BLANK_TOKEN_ID = VOCAB_SIZE - 1
    const val POISON_TOKEN_ID = 7
    const val NUM_LOGITS_PER_TOKEN = VOCAB_SIZE + TdtDecoder.NUM_DURATIONS

    const val SAMPLING_RATE = 16000
    const val HOP_LENGTH = 160
  }
}

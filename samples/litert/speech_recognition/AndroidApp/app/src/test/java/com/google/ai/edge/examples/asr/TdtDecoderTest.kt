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

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
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
   * [FakeNetwork.run] and its state buffers swap by [FakeNetwork.adoptStates].
   */
  private fun previousDecode(
    network: FakeNetwork,
    hasDecode1: Boolean,
    maxTimeIndex: Int = network.maxTimeIndex,
    numOfTokenIds: Int = network.numOfTokenIds,
    numOfLogitsPerToken: Int = NUM_LOGITS_PER_TOKEN,
    blankTokenId: Int = BLANK_TOKEN_ID,
    decodeStartTokenId: Int = BLANK_TOKEN_ID,
  ): List<Pair<Int, Int>> =
    sequence {
        var inferenceSignature = LiteRtRunner.DECODE_SIGNATURE
        var numOfInferenceTokenIds = numOfTokenIds
        var inferenceTokenIds = IntArray(numOfTokenIds) { 0 }
        inferenceTokenIds[0] = decodeStartTokenId

        var tokenIndex: Int = 0
        var timeIndex: Int = 0
        while (timeIndex < maxTimeIndex) {
          val logits = network.run(inferenceSignature, inferenceTokenIds)
          val startIndexInCurrentTimeIndex =
            timeIndex * numOfInferenceTokenIds * numOfLogitsPerToken
          // Do argmax among the logits of the current token.
          val startIndexOfTokenId = startIndexInCurrentTimeIndex + tokenIndex * numOfLogitsPerToken
          val endIndexOfDuration =
            startIndexInCurrentTimeIndex + (tokenIndex + 1) * numOfLogitsPerToken
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

  private fun greedyDecode(network: FakeNetwork, hasDecode1: Boolean): List<Pair<Int, Int>> =
    TdtDecoder.greedyDecode(
        network,
        network.maxTimeIndex,
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

  @Test
  fun greedyDecode_sameTokensAndRunsAsThePreviousLoop() {
    var cases = 0
    var tokens = 0
    var decode1Runs = 0
    for (maxTimeIndex in listOf(1, 7, 63, 375)) {
      for (numOfTokenIds in listOf(1, 2, 4)) {
        for (hasDecode1 in listOf(true, false)) {
          for ((pBlank, pZeroDuration) in listOf(0.5 to 0.3, 0.8 to 0.5, 0.2 to 0.6)) {
            for (seed in 0 until if (maxTimeIndex == 375) 5 else 25) {
              val predict = randomPredictions(seed, pBlank, pZeroDuration)
              val previousNetwork = FakeNetwork(maxTimeIndex, numOfTokenIds, predict)
              val expected = previousDecode(previousNetwork, hasDecode1)
              val network = FakeNetwork(maxTimeIndex, numOfTokenIds, predict)
              assertEquals(expected, greedyDecode(network, hasDecode1))
              assertEquals(previousNetwork.runs, network.runs)
              assertEquals(previousNetwork.inputStatesBuffersIndex, network.inputStatesBuffersIndex)
              cases++
              tokens += expected.size - 1
              decode1Runs += network.runs.count { it.signature == TdtDecoder.DECODE_1_SIGNATURE }
            }
          }
        }
      }
    }
    // The cases reach both signatures and emit tokens: the comparison is not empty.
    assertEquals(3 * 3 * 2 * 3 * 25 + 3 * 2 * 3 * 5, cases)
    assertTrue(tokens > 2_000)
    assertTrue(decode1Runs > 300)
  }

  private companion object {
    // A small vocabulary with the blank token last, as in the parakeet exports.
    const val VOCAB_SIZE = 9
    const val BLANK_TOKEN_ID = VOCAB_SIZE - 1
    const val NUM_LOGITS_PER_TOKEN = VOCAB_SIZE + TdtDecoder.NUM_DURATIONS
  }
}

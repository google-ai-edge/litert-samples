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

import android.content.Context
import android.os.Build
import androidx.annotation.VisibleForTesting
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.BuiltinNpuAcceleratorProvider
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment
import com.google.ai.edge.litert.TensorBuffer
import com.google.ai.edge.litert.TensorType

/** Token Duration Transducer (TDT) decoder. */
class TdtDecoder(private val compiledModel: CompiledModel, private val modelConfig: ModelConfig) :
  LiteRtRunner.Decoder {
  private val inputBuffers = compiledModel.createInputBuffers(DECODE_SIGNATURE)
  private val outputBuffers = compiledModel.createOutputBuffers(DECODE_SIGNATURE)

  private val maxTimeIndex =
    compiledModel.getInputTensorType(inputBufferName(0), DECODE_SIGNATURE).numElements /
      NUM_FEATURES
  private val numOfTokenIds =
    compiledModel.getInputTensorType(inputBufferName(1), DECODE_SIGNATURE).numElements
  private val numOfLogitsPerToken =
    compiledModel.getOutputTensorType(outputBufferName(0), DECODE_SIGNATURE).numElements /
      numOfTokenIds /
      maxTimeIndex

  private val blankTokenId = modelConfig.decodeStartTokenId

  private val inputBuffersOfDecode1 =
    try {
      compiledModel.createInputBuffers(DECODE_1_SIGNATURE)
    } catch (e: Exception) {
      null
    }
  private val hasDecode1 = inputBuffersOfDecode1 != null
  private val outputBuffersOfDecode1 =
    if (hasDecode1) compiledModel.createOutputBuffers(DECODE_1_SIGNATURE) else null

  // Use input/output buffers of decode (not decode_1) as a hack to avoid copying or re-assigning
  // when switching from decode to decode_1.
  private val statesBuffers =
    listOf(listOf(inputBuffers[2], inputBuffers[3]), listOf(outputBuffers[1], outputBuffers[2]))
  private var inputStatesBuffersIndex = 0
  private val numOfStates =
    compiledModel.getInputTensorType(inputBufferName(2), DECODE_SIGNATURE).numElements

  override fun decode(encodeOutputBuffers: List<TensorBuffer>): Sequence<Pair<Int, Int>> =
    sequence {
      // Clear the states as initial states are all zeros.
      val zeroStates = FloatArray(numOfStates) { 0.0f }
      statesBuffers[inputStatesBuffersIndex].forEach { it.writeFloat(zeroStates) }

      yieldAll(
        greedyDecode(
          network(encodeOutputBuffers),
          maxTimeIndex,
          numOfTokenIds,
          numOfLogitsPerToken,
          blankTokenId,
          modelConfig.decodeStartTokenId,
          hasDecode1,
        )
      )
    }

  /** The decode and decode_1 signatures, run on the encoder output of one window. */
  private fun network(encodeOutputBuffers: List<TensorBuffer>) =
    object : Network {
      override fun run(signature: String, tokenIds: IntArray): FloatArray {
        val isDecode1 = signature == DECODE_1_SIGNATURE
        val tokenIdsBuffer = if (isDecode1) inputBuffersOfDecode1!![1] else inputBuffers[1]
        val logitsBuffer = if (isDecode1) outputBuffersOfDecode1!![0] else outputBuffers[0]
        tokenIdsBuffer.writeInt(tokenIds)

        compiledModel.run(
          getInputBuffers(encodeOutputBuffers, tokenIdsBuffer),
          getOutputBuffers(logitsBuffer),
          signature,
        )

        return logitsBuffer.readFloat()
      }

      override fun adoptStates() {
        // Swap the input and output state buffers.
        inputStatesBuffersIndex = 1 - inputStatesBuffersIndex
      }
    }

  private fun getInputBuffers(
    encodeOutputBuffers: List<TensorBuffer>,
    tokenIdsBuffer: TensorBuffer,
  ) =
    buildList<TensorBuffer> {
      // Avoid copy. Build a list of TensorBuffers directly from the encoder output buffers.
      addAll(encodeOutputBuffers)
      add(tokenIdsBuffer)
      addAll(statesBuffers[inputStatesBuffersIndex])
    }

  private fun getOutputBuffers(logitsBuffer: TensorBuffer) =
    listOf(logitsBuffer) + statesBuffers[1 - inputStatesBuffersIndex]

  /** The decode and decode_1 signatures as [greedyDecode] runs them. */
  @VisibleForTesting
  internal interface Network {
    /** Runs [signature] on [tokenIds] and returns its logits for every time index and token. */
    fun run(signature: String, tokenIds: IntArray): FloatArray

    /** Makes the LSTM states the last [run] wrote the input states of the next [run]. */
    fun adoptStates()
  }

  companion object {
    private const val DECODE_SIGNATURE = LiteRtRunner.DECODE_SIGNATURE
    @VisibleForTesting internal const val DECODE_1_SIGNATURE = "decode_1"

    // Hardcoded because CompileModel doesn't expose shape info.
    private const val NUM_FEATURES = 1024
    @VisibleForTesting internal const val NUM_DURATIONS = 5

    private fun inputBufferName(index: Int) = LiteRtRunner.inputBufferName(index)

    private fun outputBufferName(index: Int) = LiteRtRunner.outputBufferName(index)

    /**
     * Greedy TDT decoding of one window: yields (token ID, time index) for every non-blank token,
     * then (END_OF_SEQUENCE, [maxTimeIndex]).
     */
    @VisibleForTesting
    internal fun greedyDecode(
      network: Network,
      maxTimeIndex: Int,
      numOfTokenIds: Int,
      numOfLogitsPerToken: Int,
      blankTokenId: Int,
      decodeStartTokenId: Int,
      hasDecode1: Boolean,
    ): Sequence<Pair<Int, Int>> = sequence {
      // Start inference with decode signature for better quality in sequence start as stateless
      // decoding can keep context better than decode_1, stateful decoding.
      var inferenceSignature = DECODE_SIGNATURE
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
        val endIndexOfDuration =
          startIndexInCurrentTimeIndex + (tokenIndex + 1) * numOfLogitsPerToken
        val endIndexOfTokenId = endIndexOfDuration - NUM_DURATIONS
        val tokenId =
          ((startIndexOfTokenId until endIndexOfTokenId).maxByOrNull { logits[it] }
            ?: startIndexOfTokenId) - startIndexOfTokenId
        if (tokenId != blankTokenId) {
          yield(Pair(tokenId, timeIndex))
          if (numOfInferenceTokenIds > 1) {
            tokenIndex++
            if (tokenIndex < numOfInferenceTokenIds) {
              // Still room in the stateless token array: keep filling it, so the
              // LSTM states handed to decode_1 below are computed from real
              // tokens only (an unfilled zero slot skews them).
            } else if (hasDecode1) { // Switch to decode_1 for stateful decoding.
              inferenceSignature = DECODE_1_SIGNATURE
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
          // Stateful RNN decoder: adopt the new LSTM states — but only on a non-blank emission.
          // In RNN-T greedy decoding the prediction network advances only when
          // a token is emitted; adopting the state on blank steps re-consumes
          // the last token once per blank and degrades the transcript.
          network.adoptStates()
        }
      }

      yield(Pair(SpeechRecognizer.END_OF_SEQUENCE, maxTimeIndex))
    }
  }
}

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

      // Start inference with decode signature for better quality in sequence start as stateless
      // decoding can keep context better than decode_1, stateful decoding.
      var inferenceSignature = DECODE_SIGNATURE
      var numOfInferenceTokenIds = numOfTokenIds
      var inferenceTokenIdsBuffer = inputBuffers[1]
      var inferenceTokenIds = IntArray(numOfTokenIds) { 0 }
      inferenceTokenIds[0] = modelConfig.decodeStartTokenId
      var inferenceLogitsBuffer = outputBuffers[0]

      var tokenIndex: Int = 0
      var timeIndex: Int = 0
      while (timeIndex < maxTimeIndex) {
        inferenceTokenIdsBuffer.writeInt(inferenceTokenIds)

        compiledModel.run(
          getInputBuffers(encodeOutputBuffers, inferenceTokenIdsBuffer),
          getOutputBuffers(inferenceLogitsBuffer),
          inferenceSignature,
        )

        val logits = inferenceLogitsBuffer.readFloat()
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
            if (tokenIndex < numOfInferenceTokenIds - 1) {
              // No-op.
            } else if (hasDecode1) { // Switch to decode_1 for stateful decoding.
              inferenceSignature = DECODE_1_SIGNATURE
              numOfInferenceTokenIds = 1
              inferenceTokenIdsBuffer = inputBuffersOfDecode1!![1]
              inferenceTokenIds = IntArray(1) { 0 }
              inferenceLogitsBuffer = outputBuffersOfDecode1!![0]
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

        if (numOfInferenceTokenIds == 1) {
          // Stateful RNN decoder: Swap input and output state buffers for the next decode.
          inputStatesBuffersIndex = 1 - inputStatesBuffersIndex
        }
      }

      yield(Pair(SpeechRecognizer.END_OF_SEQUENCE, maxTimeIndex))
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

  private companion object {
    const val DECODE_SIGNATURE = LiteRtRunner.DECODE_SIGNATURE
    const val DECODE_1_SIGNATURE = "decode_1"

    const val NUM_FEATURES = 1024 // Hardcoded because CompileModel doesn't expose shape info.
    const val NUM_DURATIONS = 5

    private fun inputBufferName(index: Int) = LiteRtRunner.inputBufferName(index)

    private fun outputBufferName(index: Int) = LiteRtRunner.outputBufferName(index)
  }
}

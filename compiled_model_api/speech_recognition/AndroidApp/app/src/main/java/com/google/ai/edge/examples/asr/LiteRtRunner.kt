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

/** Extension for the number of elements in a tensor buffer with the given tensor type. */
internal val TensorType.numElements: Int
  get() = layout!!.dimensions.fold(1, Int::times)

class LiteRtRunner(
  context: Context,
  private val modelConfig: ModelConfig,
  accelerator: Accelerator,
  decoderFactory: (CompiledModel, ModelConfig) -> Decoder,
) : SpeechRecognizer {

  /** Interface for decoding the ASR encoder output into token IDs. */
  interface Decoder {
    /**
     * Decodes the encoder output buffers and returns the token IDs and their relative timestamps
     * (if available) as a sequence.
     *
     * @param encodeOutputBuffers The list of encoder output buffers to decode.
     * @return The list of token IDs and their timestamps (if available) as a sequence.
     */
    fun decode(encodeOutputBuffers: List<TensorBuffer>): Sequence<Pair<Int, Int>>
  }

  private val compiledModel: CompiledModel =
    {
      // Always include CPU as a fallback.
      val accelerators = setOf(Accelerator.CPU, accelerator)

      val options = CompiledModel.Options(accelerators)
      if (accelerator == Accelerator.GPU) {
        // Force float32 and constant tensors sharing for GPU inference.
        options.gpuOptions =
          CompiledModel.GpuOptions(
            precision =
              if (modelConfig.gpuEnforceFloat32) {
                CompiledModel.GpuOptions.Precision.FP32
              } else {
                CompiledModel.GpuOptions.Precision.DEFAULT
              },
            constantTensorSharing = modelConfig.gpuShareConstantTensors,
          )
      } else if (accelerator == Accelerator.NPU) {
        options.qualcommOptions =
          CompiledModel.QualcommOptions(
            htpPerformanceMode =
              if (modelConfig.npuHighPerformance) {
                CompiledModel.QualcommOptions.HtpPerformanceMode.HIGH_PERFORMANCE
              } else {
                CompiledModel.QualcommOptions.HtpPerformanceMode.DEFAULT
              }
          )
      }

      val env =
        if (accelerator == Accelerator.NPU) {
          Environment.create(BuiltinNpuAcceleratorProvider(context))
        } else {
          Environment.create()
        }

      val modelPath =
        if (accelerator == Accelerator.NPU) modelConfig.npuModelPath else modelConfig.modelPath
      if (existsInAssets(context, modelPath)) {
        CompiledModel.create(context.assets, modelPath, options, env)
      } else {
        val unused = openOrDownloadFile(context, modelPath, getRemoteUrl(accelerator, modelConfig))
        CompiledModel.create("${context.filesDir.path}/${modelPath}", options, env)
      }
    }()

  private val decoder: Decoder = decoderFactory(compiledModel, modelConfig)
  private val inputBuffers = compiledModel.createInputBuffers()
  private val outputBuffers = compiledModel.createOutputBuffers()

  private val expectedNumOfFloats =
    compiledModel.getInputTensorType(inputBufferName(0), ENCODE_SIGNATURE).numElements

  override fun close() {
    compiledModel.close()
  }

  override fun recognize(features: FloatArray): Sequence<Pair<Int, Int>> {
    val paddedFeatures = padOrTruncateFeatures(features, expectedNumOfFloats)
    inputBuffers[0].writeFloat(paddedFeatures)

    compiledModel.run(inputBuffers, outputBuffers)

    return decoder.decode(outputBuffers)
  }

  /** Helper class for decoding the ASR encoder output by looping over the decoder inputs. */
  class DefaultDecoder(
    private val compiledModel: CompiledModel,
    private val modelConfig: ModelConfig,
  ) : Decoder {
    private val inputBuffers = compiledModel.createInputBuffers(DECODE_SIGNATURE)
    private val tokenIdsBuffer = inputBuffers[inputBuffers.size - 2]
    private val maskBuffer = inputBuffers[inputBuffers.size - 1]
    private val outputBuffers = compiledModel.createOutputBuffers(DECODE_SIGNATURE)

    private val numOfTokenIds =
      compiledModel
        .getInputTensorType(inputBufferName(inputBuffers.size - 2), DECODE_SIGNATURE)
        .numElements
    private val numOfLogitsPerToken =
      compiledModel.getOutputTensorType(outputBufferName(0), DECODE_SIGNATURE).numElements /
        numOfTokenIds

    private val causalMask: FloatArray =
      {
        var maskBufferName = inputBufferName(inputBuffers.size - 1)
        val numOfMaskEntries =
          compiledModel.getInputTensorType(maskBufferName, DECODE_SIGNATURE).numElements
        val n = kotlin.math.sqrt(numOfMaskEntries.toDouble()).toInt()
        val mask = FloatArray(numOfMaskEntries) { MASKED_OUT_FLOAT_VALUE }
        for (r in 0 until n) {
          for (c in 0..r) {
            mask[r * n + c] = MASKED_IN_FLOAT_VALUE
          }
        }
        mask
      }()

    override fun decode(encodeOutputBuffers: List<TensorBuffer>): Sequence<Pair<Int, Int>> =
      sequence {
        val decodeInputBuffers =
          buildList<TensorBuffer> {
            // Avoid copy. Build a list of TensorBuffers directly from the encoder output buffers.
            addAll(encodeOutputBuffers)
            // Add the decoder input buffers at the end of the list.
            add(tokenIdsBuffer)
            add(maskBuffer)
          }
        maskBuffer.writeFloat(causalMask)

        val tokenIds = IntArray(numOfTokenIds) { 0 }
        tokenIds[0] = modelConfig.decodeStartTokenId

        var seenSkipUntilTokenId = modelConfig.decodeSkipUntilTokenId < 0
        for (i in 0 until numOfTokenIds - 1) {
          tokenIdsBuffer.writeInt(tokenIds)

          compiledModel.run(decodeInputBuffers, outputBuffers, DECODE_SIGNATURE)

          val logits = outputBuffers[0].readFloat()
          // Do argmax among the logits of the current token.
          val startIndex = i * numOfLogitsPerToken
          val endIndex = (i + 1) * numOfLogitsPerToken
          val tokenId =
            ((startIndex until endIndex).maxByOrNull { logits[it] } ?: startIndex) - startIndex
          if (tokenId == modelConfig.decodeStopTokenId) {
            break
          }

          if (seenSkipUntilTokenId) {
            yield(Pair(tokenId, SpeechRecognizer.NO_TIMESTAMP))
          } else if (tokenId == modelConfig.decodeSkipUntilTokenId) {
            seenSkipUntilTokenId = true
          }

          tokenIds[i + 1] = tokenId
        }

        yield(Pair(SpeechRecognizer.END_OF_SEQUENCE, SpeechRecognizer.NO_TIMESTAMP))
      }
  }

  companion object {
    private const val INPUT_PREFIX = "args_"
    private const val OUTPUT_PREFIX = "output_"

    private const val ENCODE_SIGNATURE = "encode"
    internal const val DECODE_SIGNATURE = "decode"
    @VisibleForTesting internal const val MASKED_IN_FLOAT_VALUE = 0.0f
    @VisibleForTesting internal const val MASKED_OUT_FLOAT_VALUE = -0.7f * Float.MAX_VALUE

    internal fun inputBufferName(index: Int) = "${INPUT_PREFIX}${index}"

    internal fun outputBufferName(index: Int) = "${OUTPUT_PREFIX}${index}"

    @VisibleForTesting
    internal fun padOrTruncateFeatures(features: FloatArray, expectedNumOfFloats: Int): FloatArray =
      when {
        features.size == expectedNumOfFloats -> features
        features.size < expectedNumOfFloats -> features.copyOf(expectedNumOfFloats) // Pad zeros.
        else -> features.sliceArray((features.size - expectedNumOfFloats) until features.size)
      }

    @VisibleForTesting
    internal fun getRemoteUrl(
      accelerator: Accelerator,
      modelConfig: ModelConfig,
      socModel: String = Build.SOC_MODEL ?: "",
    ): String =
      if (accelerator == Accelerator.NPU) {
        val urlPattern = modelConfig.npuModelRemoteUrlPattern
        if (urlPattern.contains("<target>")) {
          urlPattern.replace("<target>", getNpuTarget(socModel))
        } else {
          urlPattern
        }
      } else {
        modelConfig.modelRemoteUrl
      }

    /** Returns the NPU target string based on the device's build properties. */
    private fun getNpuTarget(socModel: String): String =
      if (socModel.startsWith("Tensor")) {
        "Google_${socModel.trim().replace(" ", "_")}" // e.g. "Google_Tensor_G5"
      } else if (socModel.startsWith("SM8")) {
        "Qualcomm_${socModel}"
      } else {
        throw IllegalArgumentException("Unsupported NPU target: ${socModel}")
      }
  }
}

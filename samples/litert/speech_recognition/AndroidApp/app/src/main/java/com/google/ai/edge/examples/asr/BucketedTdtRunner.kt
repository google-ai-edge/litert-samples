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
import java.io.File

/**
 * Runs a Parakeet TDT model exported as separate graphs, as in litert-community/parakeet-ultra and
 * litert-community/Phonon-2: a log-mel front end for each window length, an encoder for each window
 * length (one file per length, or one file with one signature per length), and a one-step decoder
 * with `predict` and `joint` signatures. Greedy TDT decoding runs on the host.
 *
 * It takes raw 16 kHz audio (no AudioPreprocessor), runs the smallest loaded window length that
 * holds the valid samples, and returns token IDs with their encoder frame indices as timestamps.
 */
class BucketedTdtRunner(
  context: Context,
  private val modelConfig: ModelConfig,
  accelerator: Accelerator,
) : SpeechRecognizer {

  /**
   * The mel front end and the encoder of one window length, with their buffers by name. Buffers are
   * addressed by name throughout: a signature's buffer list follows its subgraph's tensor order,
   * which is not the order of the names.
   */
  private class Bucket(
    val mel: CompiledModel,
    val encoder: CompiledModel,
    val encoderSignature: String,
  ) {
    /** The mel graph's audio length: the most valid samples this window length holds. */
    val numSamples = mel.getInputTensorType("audio", SERVING_SIGNATURE).numElements
    val melInputs =
      listOf("audio", "n_valid").associateWith { mel.createInputBuffer(it, SERVING_SIGNATURE) }
    val melOutputs =
      listOf("features", "frame_mask", "mel_mask").associateWith {
        mel.createOutputBuffer(it, SERVING_SIGNATURE)
      }
    val encoderInputs =
      listOf("features", "mel_mask").associateWith {
        encoder.createInputBuffer(it, encoderSignature)
      }
    val encoderOutputs = mapOf("enc" to encoder.createOutputBuffer("enc", encoderSignature))
  }

  /** Every graph this runner loaded; one encoder may serve several window lengths. */
  private val models = mutableListOf<CompiledModel>()
  private val buckets: List<Bucket>
  private val decoder: CompiledModel

  init {
    if (accelerator == Accelerator.NPU) {
      // Only an encoder compiled ahead of time for the phone's chip runs on the NPU: compiling the
      // float16 encoder on the device takes more memory than a phone app gets.
      if (modelConfig.npuModelRemoteUrlPattern.isEmpty()) {
        throw UnsupportedOperationException("no NPU model yet, pick CPU or GPU")
      }
      if (Build.SOC_MODEL !in modelConfig.npuSocModels) {
        throw UnsupportedOperationException(
          "the NPU model is compiled for ${modelConfig.npuSocModels.joinToString()} only, " +
            "pick CPU or GPU"
        )
      }
      // LiteRT looks for its NPU runtime in the app's native library folder. Without it (the
      // default build, see app/build.gradle.kts) the NPU model still loads: LiteRT 2.3.0 only logs
      // "No usable Dispatch runtime found".
      val nativeLibraries = File(context.applicationInfo.nativeLibraryDir).list().orEmpty()
      if (nativeLibraries.none { it.startsWith("libLiteRtDispatch_") }) {
        throw UnsupportedOperationException(
          "this build has no NPU libraries, see app/build.gradle.kts"
        )
      }
    }
    val env =
      if (accelerator == Accelerator.NPU) {
        Environment.create(BuiltinNpuAcceleratorProvider(context))
      } else {
        Environment.create()
      }
    val cpuOptions = CompiledModel.Options(setOf(Accelerator.CPU))
    val encoderUrlPattern =
      when (accelerator) {
        Accelerator.GPU -> modelConfig.gpuModelRemoteUrl
        Accelerator.NPU -> modelConfig.npuModelRemoteUrlPattern
        else -> modelConfig.modelRemoteUrl
      }
    // One encoder file with a signature per window length, such as `enc10`, or one file per length.
    val oneEncoderFile = modelConfig.encoderSignature.contains(SECONDS)

    try {
      val folder = modelConfig.tokenizerPath.substringBeforeLast("/")
      // With one file per length, only the longest window's graphs load off the CPU: the three
      // float16 encoders together do not fit every phone's GPU. One encoder file stores the weights
      // once for all lengths. The shorter graphs only skip the work on padding.
      val windowSeconds =
        if (accelerator == Accelerator.CPU || oneEncoderFile) modelConfig.windowSeconds.sorted()
        else listOf(modelConfig.windowSeconds.max())
      val encoderOptions = encoderOptions(accelerator, oneEncoderFile)
      val sharedEncoder =
        if (oneEncoderFile) {
          loadModel(context, folder, encoderUrlPattern, "", encoderOptions, env)
        } else {
          null
        }
      buckets =
        windowSeconds.map { seconds ->
          val name = windowName(seconds)
          val mel = loadModel(context, folder, modelConfig.melModelRemoteUrl, name, cpuOptions, env)
          val encoder =
            sharedEncoder
              ?: loadModel(context, folder, encoderUrlPattern, name, encoderOptions, env)
          Bucket(mel, encoder, modelConfig.encoderSignature.replace(SECONDS, name))
        }
      decoder = loadModel(context, folder, modelConfig.decoderModelRemoteUrl, "", cpuOptions, env)
    } catch (e: Exception) {
      // Do not leak the graphs that loaded before the failure.
      models.forEach { it.close() }
      throw e
    }
  }

  // predict: inputs c, h, token, outputs c_out, dec_out, h_out; joint: inputs dec_out, enc_frame,
  // output logits. joint reads predict's dec_out buffer directly.
  private val tokenBuffer = decoder.createInputBuffer("token", PREDICT_SIGNATURE)
  private val decoderOutputBuffer = decoder.createOutputBuffer("dec_out", PREDICT_SIGNATURE)
  private val encoderFrameBuffer = decoder.createInputBuffer("enc_frame", JOINT_SIGNATURE)
  private val jointInputs =
    mapOf("dec_out" to decoderOutputBuffer, "enc_frame" to encoderFrameBuffer)
  private val jointOutputs =
    mapOf("logits" to decoder.createOutputBuffer("logits", JOINT_SIGNATURE))

  // Avoid copying the LSTM states: the output state buffers of one predict call are the input state
  // buffers of the next one.
  private val statesBuffers =
    listOf(
      listOf("c", "h").map { decoder.createInputBuffer(it, PREDICT_SIGNATURE) },
      listOf("c_out", "h_out").map { decoder.createOutputBuffer(it, PREDICT_SIGNATURE) },
    )
  private val predictInputs =
    statesBuffers.map { (c, h) -> mapOf("c" to c, "h" to h, "token" to tokenBuffer) }
  private val predictOutputs =
    statesBuffers.map { (c, h) ->
      mapOf("c_out" to c, "dec_out" to decoderOutputBuffer, "h_out" to h)
    }
  private var inputStatesBuffersIndex = 0
  private val zeroStates = FloatArray(NUM_STATES) { 0.0f }

  override fun close() {
    models.forEach { it.close() }
  }

  override fun recognize(features: FloatArray): Sequence<Pair<Int, Int>> = sequence {
    val numValidSamples = getNumOfValidSamples(features)
    if (numValidSamples >= MIN_NUM_SAMPLES) {
      val bucket =
        buckets.firstOrNull { numValidSamples <= it.numSamples } ?: buckets.last()
      val numSamples = minOf(numValidSamples, bucket.numSamples)
      bucket.melInputs.getValue("audio").writeFloat(features.copyOf(bucket.numSamples))
      bucket.melInputs.getValue("n_valid").writeInt(intArrayOf(numSamples))
      bucket.mel.run(bucket.melInputs, bucket.melOutputs, SERVING_SIGNATURE)

      // The two graphs may run on different accelerators, so copy rather than share the buffers.
      for ((name, buffer) in bucket.encoderInputs) {
        buffer.writeFloat(bucket.melOutputs.getValue(name).readFloat())
      }
      bucket.encoder.run(bucket.encoderInputs, bucket.encoderOutputs, bucket.encoderSignature)

      val encoded = bucket.encoderOutputs.getValue("enc").readFloat()
      val numValidFrames = bucket.melOutputs.getValue("frame_mask").readFloat().sum().toInt()
      yieldAll(decodeGreedy(encoded, numValidFrames))
    }

    yield(Pair(SpeechRecognizer.END_OF_SEQUENCE, getNumOfFrames(features.size)))
  }

  /** Greedy TDT decoding of one window, as in the model card's "How it works". */
  private fun decodeGreedy(encoded: FloatArray, numValidFrames: Int): List<Pair<Int, Int>> {
    val tokens = mutableListOf<Pair<Int, Int>>()

    // Start from blank with zero states.
    statesBuffers[inputStatesBuffersIndex].forEach { it.writeFloat(zeroStates) }
    predict(modelConfig.decodeStartTokenId)

    var timeIndex = 0
    var numSteps = 0
    var numTokensAtTimeIndex = 0
    while (timeIndex < numValidFrames && numSteps < MAX_STEPS) {
      numSteps++
      encoderFrameBuffer.writeFloat(
        encoded.copyOfRange(timeIndex * ENCODER_DIM, (timeIndex + 1) * ENCODER_DIM)
      )
      decoder.run(jointInputs, jointOutputs, JOINT_SIGNATURE)

      val logits = jointOutputs.getValue("logits").readFloat()
      val tokenId = argmax(logits, 0, VOCAB_SIZE)
      var duration = argmax(logits, VOCAB_SIZE, VOCAB_SIZE + NUM_DURATIONS) - VOCAB_SIZE
      if (tokenId == modelConfig.decodeStartTokenId) {
        // A blank moves at least one frame.
        if (duration == 0) duration = 1
      } else {
        tokens.add(Pair(tokenId, timeIndex))
        if (tokenId == modelConfig.decodeStopTokenId) break
        predict(tokenId)
      }

      // Move on after MAX_TOKENS_PER_TIME_INDEX tokens at one time index.
      if (duration == 0 && ++numTokensAtTimeIndex >= MAX_TOKENS_PER_TIME_INDEX) duration = 1
      if (duration > 0) numTokensAtTimeIndex = 0
      timeIndex += duration
    }

    return tokens
  }

  private fun predict(tokenId: Int) {
    tokenBuffer.writeInt(intArrayOf(tokenId))
    decoder.run(
      predictInputs[inputStatesBuffersIndex],
      predictOutputs[1 - inputStatesBuffersIndex],
      PREDICT_SIGNATURE,
    )
    inputStatesBuffersIndex = 1 - inputStatesBuffersIndex
  }

  /** Downloads the file of [remoteUrlPattern] for the window length [name] once, and loads it. */
  private fun loadModel(
    context: Context,
    folder: String,
    remoteUrlPattern: String,
    name: String,
    options: CompiledModel.Options,
    env: Environment,
  ): CompiledModel {
    val remoteUrl = remoteUrlPattern.replace(SECONDS, name)
    val localPath = "$folder/${remoteUrl.substringAfterLast("/")}"
    openOrDownloadFile(context, localPath, remoteUrl).close()
    return CompiledModel.create("${context.filesDir.path}/$localPath", options, env).also {
      models.add(it)
    }
  }

  companion object {
    /** The window length in a file name or a signature name, such as `<seconds>s.tflite`. */
    private const val SECONDS = "<seconds>"
    private const val SERVING_SIGNATURE = "serving_default"
    private const val PREDICT_SIGNATURE = "predict"
    private const val JOINT_SIGNATURE = "joint"

    /** Audio samples per encoder frame: a mel hop of 160 samples, subsampled 8 times. */
    private const val SAMPLES_PER_FRAME = 160 * 8
    /** A window under two mel frames has no text. */
    private const val MIN_NUM_SAMPLES = 320

    private const val ENCODER_DIM = 640
    private const val NUM_STATES = 2 * 640 // 2 LSTM layers, hidden size 640.
    private const val VOCAB_SIZE = 8193 // 8192 tokens and blank.
    private const val NUM_DURATIONS = 5 // 0 to 4 frames.
    private const val MAX_TOKENS_PER_TIME_INDEX = 10
    private const val MAX_STEPS = 3750

    private fun encoderOptions(
      accelerator: Accelerator,
      oneEncoderFile: Boolean,
    ): CompiledModel.Options {
      val options = CompiledModel.Options(setOf(Accelerator.CPU, accelerator))
      if (accelerator == Accelerator.GPU) {
        // The float16 encoders rescale their LayerNorm inputs so that float16 arithmetic does
        // not overflow. The window lengths of one encoder file read the same weights.
        options.gpuOptions =
          CompiledModel.GpuOptions(
            precision = CompiledModel.GpuOptions.Precision.FP16,
            constantTensorSharing = if (oneEncoderFile) true else null,
          )
      } else if (accelerator == Accelerator.NPU) {
        // The burst mode, as in the model cards' NPU runs.
        options.qualcommOptions =
          CompiledModel.QualcommOptions(
            htpPerformanceMode = CompiledModel.QualcommOptions.HtpPerformanceMode.BURST
          )
      }
      return options
    }

    /** Returns a window length as the model files name it: "10" for 10 s, "7.5" for 7.5 s. */
    @VisibleForTesting
    internal fun windowName(seconds: Float): String =
      if (seconds % 1 == 0f) seconds.toInt().toString() else seconds.toString()

    /** Returns the number of samples up to the last non-zero one; the rest is padding. */
    @VisibleForTesting
    internal fun getNumOfValidSamples(samples: FloatArray): Int =
      samples.indexOfLast { it != 0.0f } + 1

    /** Returns the number of encoder frames of a window, the timestamp at its end. */
    @VisibleForTesting
    internal fun getNumOfFrames(numSamples: Int): Int =
      (numSamples + SAMPLES_PER_FRAME - 1) / SAMPLES_PER_FRAME

    @VisibleForTesting
    internal fun argmax(values: FloatArray, from: Int, until: Int): Int =
      (from until until).maxByOrNull { values[it] } ?: from
  }
}

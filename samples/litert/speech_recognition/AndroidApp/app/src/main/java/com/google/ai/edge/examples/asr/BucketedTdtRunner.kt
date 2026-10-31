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
import androidx.annotation.VisibleForTesting
import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.CompiledModel
import com.google.ai.edge.litert.Environment

/**
 * Runs a Parakeet TDT model exported as separate graphs, as in litert-community/parakeet-ultra: a
 * log-mel front end and an encoder for each window length (10, 20 and 30 s), and a one-step decoder
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
  private class Bucket(val numSamples: Int, val mel: CompiledModel, val encoder: CompiledModel) {
    val melInputs =
      listOf("audio", "n_valid").associateWith { mel.createInputBuffer(it, SERVING_SIGNATURE) }
    val melOutputs =
      listOf("features", "frame_mask", "mel_mask").associateWith {
        mel.createOutputBuffer(it, SERVING_SIGNATURE)
      }
    val encoderInputs =
      listOf("features", "mel_mask").associateWith {
        encoder.createInputBuffer(it, SERVING_SIGNATURE)
      }
    val encoderOutputs = mapOf("enc" to encoder.createOutputBuffer("enc", SERVING_SIGNATURE))

    fun close() {
      mel.close()
      encoder.close()
    }
  }

  private val buckets: List<Bucket>
  private val decoder: CompiledModel

  init {
    // There is no NPU model yet: compiling the float16 encoder on the device takes more memory
    // than a phone app gets.
    if (accelerator == Accelerator.NPU) {
      throw UnsupportedOperationException("no NPU model yet, pick CPU or GPU")
    }
    val env = Environment.create()
    val cpuOptions = CompiledModel.Options(setOf(Accelerator.CPU))
    val encoderUrlPattern =
      if (accelerator == Accelerator.CPU) modelConfig.modelRemoteUrl
      else modelConfig.gpuModelRemoteUrl

    val loaded = mutableListOf<CompiledModel>()
    try {
      val folder = modelConfig.tokenizerPath.substringBeforeLast("/")
      // On the GPU only the longest window's graphs load: the three float16 encoders together do
      // not fit every phone's GPU. The shorter graphs only skip the work on padding.
      val windowSeconds =
        if (accelerator == Accelerator.CPU) modelConfig.windowSeconds.sorted()
        else listOf(modelConfig.windowSeconds.max())
      buckets =
        windowSeconds.map { seconds ->
          val mel =
            createModel(context, folder, modelConfig.melModelRemoteUrl, seconds, cpuOptions, env)
          loaded.add(mel)
          val encoder =
            createModel(
              context,
              folder,
              encoderUrlPattern,
              seconds,
              encoderOptions(accelerator),
              env,
            )
          loaded.add(encoder)
          Bucket(seconds * SAMPLING_RATE, mel, encoder)
        }
      decoder =
        createModel(
          context,
          folder,
          modelConfig.decoderModelRemoteUrl,
          seconds = 0,
          cpuOptions,
          env,
        )
    } catch (e: Exception) {
      // Do not leak the graphs that loaded before the failure.
      loaded.forEach { it.close() }
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
    buckets.forEach { it.close() }
    decoder.close()
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
      bucket.encoder.run(bucket.encoderInputs, bucket.encoderOutputs, SERVING_SIGNATURE)

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

  companion object {
    private const val SERVING_SIGNATURE = "serving_default"
    private const val PREDICT_SIGNATURE = "predict"
    private const val JOINT_SIGNATURE = "joint"

    private const val SAMPLING_RATE = 16000
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

    private fun createModel(
      context: Context,
      folder: String,
      remoteUrlPattern: String,
      seconds: Int,
      options: CompiledModel.Options,
      env: Environment,
    ): CompiledModel {
      val remoteUrl = remoteUrlPattern.replace("<seconds>", seconds.toString())
      val localPath = "$folder/${remoteUrl.substringAfterLast("/")}"
      openOrDownloadFile(context, localPath, remoteUrl).close()
      return CompiledModel.create("${context.filesDir.path}/$localPath", options, env)
    }

    private fun encoderOptions(accelerator: Accelerator): CompiledModel.Options {
      val options = CompiledModel.Options(setOf(Accelerator.CPU, accelerator))
      if (accelerator == Accelerator.GPU) {
        // The float16 encoders rescale their LayerNorm inputs so that float16 arithmetic does
        // not overflow.
        options.gpuOptions =
          CompiledModel.GpuOptions(precision = CompiledModel.GpuOptions.Precision.FP16)
      }
      return options
    }

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

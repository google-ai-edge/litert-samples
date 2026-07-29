/*
 * Copyright 2024 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.ai.edge.examples.asr

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/** Represents the configuration for Log-Mel Spectrogram. */
@Serializable
data class LogMelSpectroConfig(
  val nFFT: Int,
  val nMels: Int = 80,
  val hopLength: Int = 160,
  val nFrames: Int = 0,
  val transpose: Boolean = false,
  val preemphasis: Float = 0.0f,
  val normType: String = "standard",
)

/** Represents the configuration for a specific model. */
data class ModelConfig(
  val modelPath: String,
  val modelRemoteUrl: String,
  val npuModelPath: String,
  val npuModelRemoteUrlPattern: String,
  val tokenizerPath: String,
  val tokenizerRemoteUrl: String,
  val inputMilliseconds: Int,
  val logMelSpectro: LogMelSpectroConfig?,
  val hasDecoder: Boolean,
  val decodeStartTokenId: Int,
  val decodeStopTokenId: Int,
  val decodeSkipUntilTokenId: Int,
  val gpuEnforceFloat32: Boolean,
  val gpuShareConstantTensors: Boolean,
  val npuHighPerformance: Boolean,
)

@Serializable
private data class ModelConfigJson(
  val modelRemoteUrl: String = "",
  val npuModelRemoteUrlPattern: String = "",
  val tokenizerUrl: String = "",
  val inputMilliseconds: Int = -1,
  val logMelSpectro: LogMelSpectroConfig? = null,
  val decodeStartTokenId: Int = -1,
  val decodeStopTokenId: Int = -1,
  val decodeSkipUntilTokenId: Int = -1,
  val gpuEnforceFloat32: Boolean = true,
  val gpuShareConstantTensors: Boolean = true,
  val npuHighPerformance: Boolean = true,
)

/** Manager for model metadata. */
class ModelMetadataManager(jsonString: String) {
  private val json = Json { ignoreUnknownKeys = false }
  private val configs: Map<String, ModelConfigJson> = json.decodeFromString(jsonString)

  /** Returns the list of available model names. */
  fun getAvailableModels(): List<String> = configs.keys.toList()

  /**
   * Returns a [ModelConfig] for the given [modelName].
   *
   * @param modelName The key corresponding to the desired model configuration in the JSON.
   * @return A [ModelConfig] instance.
   */
  fun getModelConfig(modelName: String): ModelConfig {
    val config = configs[modelName] ?: throw NoSuchElementException("Model $modelName not found")
    return ModelConfig(
      modelPath = modelName + "/" + MODEL_PATH,
      modelRemoteUrl = config.modelRemoteUrl,
      npuModelPath = modelName + "/" + NPU_MODEL_PATH,
      npuModelRemoteUrlPattern = config.npuModelRemoteUrlPattern,
      tokenizerPath = modelName + "/" + TOKENIZER_PATH,
      tokenizerRemoteUrl = config.tokenizerUrl,
      inputMilliseconds = config.inputMilliseconds,
      logMelSpectro = config.logMelSpectro,
      hasDecoder = config.decodeStartTokenId >= 0 && config.decodeStopTokenId >= 0,
      decodeStartTokenId = config.decodeStartTokenId,
      decodeStopTokenId = config.decodeStopTokenId,
      decodeSkipUntilTokenId = config.decodeSkipUntilTokenId,
      gpuEnforceFloat32 = config.gpuEnforceFloat32,
      gpuShareConstantTensors = config.gpuShareConstantTensors,
      npuHighPerformance = config.npuHighPerformance,
    )
  }

  private companion object {
    const val MODEL_PATH = "model.tflite"
    const val NPU_MODEL_PATH = "model_npu.tflite"
    const val TOKENIZER_PATH = "tokenizer.json"
  }
}

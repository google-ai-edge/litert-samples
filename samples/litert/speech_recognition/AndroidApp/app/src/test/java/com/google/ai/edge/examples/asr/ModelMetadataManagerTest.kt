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

import org.junit.Assert.assertEquals
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class ModelMetadataManagerTest {

  @Test
  fun testParseMetadata() {
    val jsonString =
      """
      {
        "parakeet-ctc-0.6b": {
          "tokenizerUrl": "https://example.com/tokenizer.json",
          "modelRemoteUrl": "https://example.com/model.tflite",
          "npuModelRemoteUrlPattern": "https://example.com/model_<target>.tflite",
          "logMelSpectro": {
            "nFFT": 512,
            "transpose": true,
            "nMels": 128,
            "hopLength": 200,
            "nFrames": 1000
          }
        },
        "moonshine-tiny": {
          "tokenizerUrl": "https://example.com/another-tokenizer.json",
          "decodeStartTokenId": 1,
          "decodeStopTokenId": 2,
          "decodeSkipUntilTokenId": 3
        }
      }
    """
    val manager = ModelMetadataManager(jsonString)

    val ctcConfig = manager.getModelConfig("parakeet-ctc-0.6b")
    assertEquals("parakeet-ctc-0.6b/model.tflite", ctcConfig.modelPath)
    assertEquals("parakeet-ctc-0.6b/model_npu.tflite", ctcConfig.npuModelPath)
    assertEquals("parakeet-ctc-0.6b/tokenizer.json", ctcConfig.tokenizerPath)
    assertEquals("https://example.com/tokenizer.json", ctcConfig.tokenizerRemoteUrl)
    assertEquals("https://example.com/model.tflite", ctcConfig.modelRemoteUrl)
    assertEquals("https://example.com/model_<target>.tflite", ctcConfig.npuModelRemoteUrlPattern)

    assertEquals(512, ctcConfig.logMelSpectro?.nFFT)
    assertEquals(true, ctcConfig.logMelSpectro?.transpose)
    assertEquals(128, ctcConfig.logMelSpectro?.nMels)
    assertEquals(200, ctcConfig.logMelSpectro?.hopLength)
    assertEquals(1000, ctcConfig.logMelSpectro?.nFrames)
    assertEquals(false, ctcConfig.hasDecoder)
    assertEquals(-1, ctcConfig.decodeStartTokenId)
    assertEquals(-1, ctcConfig.decodeStopTokenId)
    assertEquals(-1, ctcConfig.decodeSkipUntilTokenId)

    val moonshineConfig = manager.getModelConfig("moonshine-tiny")
    assertEquals("moonshine-tiny/model.tflite", moonshineConfig.modelPath)
    assertEquals("https://example.com/another-tokenizer.json", moonshineConfig.tokenizerRemoteUrl)
    assertEquals(null, moonshineConfig.logMelSpectro)
    assertEquals(true, moonshineConfig.hasDecoder)
    assertEquals(1, moonshineConfig.decodeStartTokenId)
    assertEquals(2, moonshineConfig.decodeStopTokenId)
    assertEquals(3, moonshineConfig.decodeSkipUntilTokenId)
  }

  @Test(expected = NoSuchElementException::class)
  fun testGetModelConfigMissingModel() {
    val jsonString =
      """
      {
        "parakeet-ctc-0.6b": {
          "tokenizerUrl": "https://example.com/tokenizer.json"
        }
      }
    """
    val manager = ModelMetadataManager(jsonString)
    val unused = manager.getModelConfig("non-existent-model")
  }

  @Test
  fun testGetAvailableModels() {
    val jsonString =
      """
      {
        "parakeet-ctc-0.6b": {
          "tokenizerUrl": "https://example.com/tokenizer.json"
        },
        "moonshine-tiny": {
          "tokenizerUrl": "https://example.com/another-tokenizer.json"
        }
      }
    """
    val manager = ModelMetadataManager(jsonString)
    val models = manager.getAvailableModels()
    assertEquals(setOf("parakeet-ctc-0.6b", "moonshine-tiny"), models.toSet())
  }

  @Test(expected = Exception::class)
  fun testParseMetadataWithUnknownKey() {
    val jsonString =
      """
      {
        "parakeet-ctc-0.6b": {
          "tokenizerUrl": "https://example.com/tokenizer.json",
          "unknownKey": "value"
        }
      }
    """
    val unused = ModelMetadataManager(jsonString)
  }

  @Test
  fun parseMetadata_withFloatMaskAndDefaultBooleanFlags_correctlyPopulatesFlags() {
    val jsonString =
      """
      {
        "parakeet-ctc-new": {
          "tokenizerUrl": "https://example.com/tokenizer.json",
          "modelRemoteUrl": "https://example.com/model.tflite",
          "npuModelRemoteUrlPattern": "https://example.com/model_<target>.tflite"
        }
      }
    """
    val manager = ModelMetadataManager(jsonString)
    val config = manager.getModelConfig("parakeet-ctc-new")

    assertEquals(true, config.gpuEnforceFloat32)
    assertEquals(true, config.gpuShareConstantTensors)
    assertEquals(true, config.npuHighPerformance)
  }

  @Test
  fun parseMetadata_withOverriddenBooleanFlags_correctlyPopulatesFlags() {
    val jsonString =
      """
      {
        "parakeet-ctc-overridden": {
          "tokenizerUrl": "https://example.com/tokenizer.json",
          "modelRemoteUrl": "https://example.com/model.tflite",
          "npuModelRemoteUrlPattern": "https://example.com/model_<target>.tflite",
          "gpuEnforceFloat32": false,
          "gpuShareConstantTensors": false,
          "npuHighPerformance": false
        }
      }
    """
    val manager = ModelMetadataManager(jsonString)
    val config = manager.getModelConfig("parakeet-ctc-overridden")

    assertEquals(false, config.gpuEnforceFloat32)
    assertEquals(false, config.gpuShareConstantTensors)
    assertEquals(false, config.npuHighPerformance)
  }
}

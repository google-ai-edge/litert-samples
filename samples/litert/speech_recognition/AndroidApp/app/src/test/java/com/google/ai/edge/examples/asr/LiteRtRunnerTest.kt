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

import com.google.ai.edge.litert.Accelerator
import com.google.ai.edge.litert.TensorType
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.junit.runners.JUnit4

@RunWith(JUnit4::class)
class LiteRtRunnerTest {

  @Test(expected = IllegalArgumentException::class)
  fun getRemoteUrl_npu_withPattern_throwsOnUnsupportedDevice() {
    val config =
      ModelConfig(
        modelPath = "path",
        modelRemoteUrl = "model_url",
        npuModelPath = "npu_path",
        npuModelRemoteUrlPattern = "https://example.com/<target>.tflite",
        tokenizerPath = "tok_path",
        tokenizerRemoteUrl = "tok_url",
        inputMilliseconds = -1,
        logMelSpectro = null,
        hasDecoder = false,
        decodeStartTokenId = -1,
        decodeStopTokenId = -1,
        decodeSkipUntilTokenId = -1,
        gpuEnforceFloat32 = true,
        gpuShareConstantTensors = true,
        npuHighPerformance = true,
      )
    LiteRtRunner.getRemoteUrl(Accelerator.NPU, config, "Unsupported_soc_model")
  }

  @Test
  fun getRemoteUrl_npu_withPattern_returnsCorrectUrl() {
    val config =
      ModelConfig(
        modelPath = "path",
        modelRemoteUrl = "model_url",
        npuModelPath = "npu_path",
        npuModelRemoteUrlPattern = "https://example.com/<target>.tflite",
        tokenizerPath = "tok_path",
        tokenizerRemoteUrl = "tok_url",
        inputMilliseconds = -1,
        logMelSpectro = null,
        hasDecoder = false,
        decodeStartTokenId = -1,
        decodeStopTokenId = -1,
        decodeSkipUntilTokenId = -1,
        gpuEnforceFloat32 = true,
        gpuShareConstantTensors = true,
        npuHighPerformance = true,
      )
    val url = LiteRtRunner.getRemoteUrl(Accelerator.NPU, config, "SM8650")
    assertEquals("https://example.com/Qualcomm_SM8650.tflite", url)
  }

  @Test
  fun getRemoteUrl_npu_withoutPattern_returnsPattern() {
    val config =
      ModelConfig(
        modelPath = "path",
        modelRemoteUrl = "model_url",
        npuModelPath = "npu_path",
        npuModelRemoteUrlPattern = "https://example.com/fixed.tflite",
        tokenizerPath = "tok_path",
        tokenizerRemoteUrl = "tok_url",
        inputMilliseconds = -1,
        logMelSpectro = null,
        hasDecoder = false,
        decodeStartTokenId = -1,
        decodeStopTokenId = -1,
        decodeSkipUntilTokenId = -1,
        gpuEnforceFloat32 = true,
        gpuShareConstantTensors = true,
        npuHighPerformance = true,
      )
    val url = LiteRtRunner.getRemoteUrl(Accelerator.NPU, config)
    assertEquals("https://example.com/fixed.tflite", url)
  }

  @Test
  fun getRemoteUrl_cpu_returnsModelRemoteUrl() {
    val config =
      ModelConfig(
        modelPath = "path",
        modelRemoteUrl = "https://example.com/model.tflite",
        npuModelRemoteUrlPattern = "npu_model_url",
        npuModelPath = "npu_path",
        tokenizerPath = "tok_path",
        tokenizerRemoteUrl = "tok_url",
        inputMilliseconds = -1,
        logMelSpectro = null,
        hasDecoder = false,
        decodeStartTokenId = -1,
        decodeStopTokenId = -1,
        decodeSkipUntilTokenId = -1,
        gpuEnforceFloat32 = true,
        gpuShareConstantTensors = true,
        npuHighPerformance = true,
      )
    val url = LiteRtRunner.getRemoteUrl(Accelerator.CPU, config)
    assertEquals("https://example.com/model.tflite", url)
  }

  @Test
  fun padOrTruncateFeatures_exactSize_returnsSameArray() {
    val features = floatArrayOf(1.0f, 2.0f, 3.0f)
    val result = LiteRtRunner.padOrTruncateFeatures(features, 3)
    assertTrue(features === result)
  }

  @Test
  fun padOrTruncateFeatures_needsPadding_postPadsWithZeros() {
    val features = floatArrayOf(1.0f, 2.0f, 3.0f)
    val result = LiteRtRunner.padOrTruncateFeatures(features, 5)
    assertEquals(5, result.size)
    // Front contains the original data
    assertTrue(result[0] == 1.0f)
    assertTrue(result[1] == 2.0f)
    assertTrue(result[2] == 3.0f)
    // Tail is padded with zeros
    assertTrue(result[3] == 0.0f)
    assertTrue(result[4] == 0.0f)
  }

  @Test
  fun padOrTruncateFeatures_needsTruncation_tailTruncates() {
    val features = floatArrayOf(1.0f, 2.0f, 3.0f, 4.0f, 5.0f)
    val result = LiteRtRunner.padOrTruncateFeatures(features, 3)
    assertEquals(3, result.size)
    // Slices the end (tail), throwing away the front
    assertTrue(result[0] == 3.0f)
    assertTrue(result[1] == 4.0f)
    assertTrue(result[2] == 5.0f)
  }

  @Test
  fun numElements_scalar_returnsOne() {
    val tensorType = TensorType(TensorType.ElementType.FLOAT, TensorType.Layout(emptyList()))
    assertEquals(1, tensorType.numElements)
  }

  @Test
  fun numElements_oneDimension_returnsDimensionSize() {
    val tensorType = TensorType(TensorType.ElementType.INT, TensorType.Layout(listOf(5)))
    assertEquals(5, tensorType.numElements)
  }

  @Test
  fun numElements_multipleDimensions_returnsProductOfDimensions() {
    val tensorType = TensorType(TensorType.ElementType.FLOAT, TensorType.Layout(listOf(2, 3, 4)))
    assertEquals(24, tensorType.numElements)
  }

  @Test
  fun numElements_zeroDimension_returnsZero() {
    val tensorType = TensorType(TensorType.ElementType.FLOAT, TensorType.Layout(listOf(2, 0, 4)))
    assertEquals(0, tensorType.numElements)
  }
}

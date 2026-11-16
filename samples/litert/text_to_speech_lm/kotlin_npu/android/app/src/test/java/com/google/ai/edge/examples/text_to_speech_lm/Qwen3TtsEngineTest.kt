/*
 * Copyright 2026 The Google AI Edge Authors. All Rights Reserved.
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

package com.google.ai.edge.examples.text_to_speech_lm

import com.google.ai.edge.litert.Accelerator
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class Qwen3TtsEngineTest {
  @Test
  fun toAccelerator_mapsAllDelegatesIncludingNpu() {
    assertEquals(
      Accelerator.CPU,
      Qwen3TtsEngine.toAccelerator(Qwen3TtsEngine.AcceleratorEnum.CPU),
    )
    assertEquals(
      Accelerator.GPU,
      Qwen3TtsEngine.toAccelerator(Qwen3TtsEngine.AcceleratorEnum.GPU),
    )
    assertEquals(
      Accelerator.NPU,
      Qwen3TtsEngine.toAccelerator(Qwen3TtsEngine.AcceleratorEnum.NPU),
    )
  }

  @Test
  fun languageIds_containsSupportedLanguages() {
    assertTrue(Qwen3TtsEngine.LANGUAGE_IDS.containsKey("english"))
    assertEquals(2050, Qwen3TtsEngine.LANGUAGE_IDS["english"])
  }

  @Test
  fun effectiveDelegate_routesNpuAndGpuSafely() {
    val npu = Qwen3TtsEngine.AcceleratorEnum.NPU
    val gpu = Qwen3TtsEngine.AcceleratorEnum.GPU
    val cpu = Qwen3TtsEngine.AcceleratorEnum.CPU

    assertEquals(cpu, Qwen3TtsEngine.effectiveDelegate("talker_int4.tflite", npu))
    assertEquals(cpu, Qwen3TtsEngine.effectiveDelegate("talker_int4.tflite", gpu))
    assertEquals(cpu, Qwen3TtsEngine.effectiveDelegate("mtp_folded_int8.tflite", npu))
    assertEquals(cpu, Qwen3TtsEngine.effectiveDelegate("mtp_folded_int8.tflite", gpu))
    assertEquals(cpu, Qwen3TtsEngine.effectiveDelegate("mtp_fp32.tflite", npu))

    assertEquals(npu, Qwen3TtsEngine.effectiveDelegate("codec_partA.tflite", npu))
    assertEquals(gpu, Qwen3TtsEngine.effectiveDelegate("codec_partA.tflite", gpu))

    assertEquals(cpu, Qwen3TtsEngine.effectiveDelegate("codec_partB.tflite", npu))
    assertEquals(gpu, Qwen3TtsEngine.effectiveDelegate("codec_partB.tflite", gpu))
    assertEquals(cpu, Qwen3TtsEngine.effectiveDelegate("codec_decoder_fp32.tflite", npu))
    assertEquals(gpu, Qwen3TtsEngine.effectiveDelegate("codec_decoder_fp32.tflite", gpu))
  }
}

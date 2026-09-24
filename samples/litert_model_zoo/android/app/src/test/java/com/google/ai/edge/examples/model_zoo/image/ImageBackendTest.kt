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

package com.google.ai.edge.examples.model_zoo.image

import com.google.ai.edge.litert.Accelerator
import org.junit.Assert.*
import org.junit.Test

/** Exercises the production factory used by image tasks, RF-DETR and Zipformer without JNI. */
class ImageBackendTest {
  @Test
  fun failedGpuCreationReturnsCpuRunnerAndExactReason() {
    val calls = mutableListOf<Accelerator>()
    val cpuRunner = Any()
    val failure = IllegalStateException("GPU compile: unsupported op RELU_0_TO_1")
    var logged: Pair<String, Exception>? = null
    val loaded =
      compileImageBackend("gpu", "test", { reason, error -> logged = reason to error }) {
        accelerator ->
        calls += accelerator
        if (accelerator == Accelerator.GPU) throw failure
        cpuRunner
      }
    assertEquals(listOf(Accelerator.GPU, Accelerator.CPU), calls)
    assertSame(cpuRunner, loaded.runner)
    assertEquals("CPU", loaded.backend)
    assertEquals(failure.message, loaded.fallbackReason)
    assertEquals(failure.message, logged?.first)
    assertSame(failure, logged?.second)
  }

  @Test
  fun mixedGpuFailureReportsCpuAndRetainsReason() {
    val loaded =
      compileImageBackend("mixed", "test", { _, _ -> }) {
        if (it == Accelerator.GPU) throw IllegalArgumentException("mixed graph cannot compile")
        "cpu runner"
      }
    assertEquals("cpu runner", loaded.runner)
    assertEquals("CPU", loaded.backend)
    assertEquals("mixed graph cannot compile", loaded.fallbackReason)
  }

  @Test
  fun cpuRequestNeverAttemptsGpuAndSuccessfulMixedRequestStaysMixed() {
    val calls = mutableListOf<Accelerator>()
    val cpu =
      compileImageBackend("cpu", "test", { _, _ -> fail("No failure expected") }) {
        calls += it
        "runner"
      }
    assertEquals(listOf(Accelerator.CPU), calls)
    assertEquals("CPU", cpu.backend)
    assertNull(cpu.fallbackReason)
    val mixed = compileImageBackend("mixed", "test") { "runner" }
    assertEquals("GPU + CPU", mixed.backend)
    assertNull(mixed.fallbackReason)
  }

  @Test
  fun cpuCreationFailurePropagatesInsteadOfClaimingFallbackSuccess() {
    val cpuFailure = IllegalStateException("CPU compile also failed")
    try {
      compileImageBackend("gpu", "test", { _, _ -> }) {
        if (it == Accelerator.GPU) throw IllegalArgumentException("GPU failed")
        throw cpuFailure
      }
      fail("CPU failure must propagate")
    } catch (actual: IllegalStateException) {
      assertSame(cpuFailure, actual)
    }
  }
}

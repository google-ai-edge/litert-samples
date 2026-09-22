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

package com.google.ai.edge.examples.model_zoo.audio

import com.google.ai.edge.litert.Accelerator
import org.junit.Assert.*
import org.junit.Test

class AudioCompilerTest {
  @Test
  fun gpuFailureReturnsCpuRunnerAndExactGraphReason() {
    val calls = mutableListOf<Accelerator>()
    val failure = IllegalStateException("unsupported GPU graph")
    var logged: Triple<String, String, Exception>? = null
    val compiler =
      AudioCompiler(
        "gpu",
        logFailure = { graph, reason, error -> logged = Triple(graph, reason, error) },
      )
    val runner = Any()
    val result =
      compiler.create("encoder") {
        calls += it
        if (it == Accelerator.GPU) throw failure
        runner
      }
    assertSame(runner, result)
    assertEquals(listOf(Accelerator.GPU, Accelerator.CPU), calls)
    assertEquals("CPU", compiler.snapshot.actual)
    assertEquals(
      "encoder: java.lang.IllegalStateException: unsupported GPU graph",
      compiler.snapshot.fallbackReason,
    )
    assertEquals("encoder", logged?.first)
    assertEquals("java.lang.IllegalStateException: unsupported GPU graph", logged?.second)
    assertSame(failure, logged?.third)
  }

  @Test
  fun failedGraphRemembersCpuChoiceButOtherGraphCanStillUseGpu() {
    val compiler = AudioCompiler("gpu", logFailure = { _, _, _ -> })
    compiler.create("stem") {
      if (it == Accelerator.GPU) throw IllegalArgumentException("compile error")
      "cpu"
    }
    val calls = mutableListOf<Accelerator>()
    compiler.create("stem") {
      calls += it
      "cpu"
    }
    compiler.create("next") {
      calls += it
      "gpu"
    }
    assertEquals(listOf(Accelerator.CPU, Accelerator.GPU), calls)
    assertEquals("GPU + CPU", compiler.snapshot.actual)
    assertEquals(
      "stem: java.lang.IllegalArgumentException: compile error",
      compiler.snapshot.fallbackReason,
    )
    assertTrue(compiler.snapshot.details.contains("stem=CPU"))
    assertTrue(compiler.snapshot.details.contains("next=GPU"))
  }

  @Test
  fun cpuRequestNeverAttemptsGpu() {
    val compiler = AudioCompiler("cpu", logFailure = { _, _, _ -> fail("No failure expected") })
    val calls = mutableListOf<Accelerator>()
    compiler.create("graph") {
      calls += it
      "cpu"
    }
    assertEquals(listOf(Accelerator.CPU), calls)
    assertEquals("CPU", compiler.snapshot.actual)
    assertNull(compiler.snapshot.fallbackReason)
  }

  @Test
  fun failedCpuCreationPropagates() {
    val compiler = AudioCompiler("gpu", logFailure = { _, _, _ -> })
    val cpuFailure = IllegalStateException("CPU failed")
    try {
      compiler.create("graph") {
        if (it == Accelerator.GPU) throw IllegalArgumentException("GPU failed")
        throw cpuFailure
      }
      fail("CPU failure must propagate")
    } catch (actual: IllegalStateException) {
      assertSame(cpuFailure, actual)
    }
  }
}

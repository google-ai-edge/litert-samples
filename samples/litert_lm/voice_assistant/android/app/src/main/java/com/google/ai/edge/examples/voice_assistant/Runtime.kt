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

// Adapted from john-rocky/hfmodels-android (commit 3086d647):
// litert/src/main/kotlin/io/github/johnrocky/hfmodels/litert/LiteRtDecisionModel.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant

import com.google.ai.edge.litert.Environment
import java.util.concurrent.Callable
import java.util.concurrent.ExecutionException
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import kotlinx.coroutines.asCoroutineDispatcher

/**
 * The one thread every LiteRT call of the transcriber and the speaker runs on (compile, run,
 * close), and the process-wide [Environment], created once on that thread and kept.
 */
internal object Runtime {
  private const val THREAD = "voice-assistant-litert"

  val executor: ExecutorService =
    Executors.newSingleThreadExecutor { r -> Thread(r, THREAD).apply { isDaemon = true } }
  val dispatcher = executor.asCoroutineDispatcher()

  @Volatile private var env: Environment? = null

  /** On the LiteRT thread. */
  fun environment(): Environment = env ?: Environment.create().also { env = it }

  /** Runs [block] on the LiteRT thread and waits for it; its exception is thrown unchanged. */
  fun <T> call(block: () -> T): T {
    if (Thread.currentThread().name == THREAD) {
      return block()
    }
    try {
      return executor.submit(Callable { block() }).get()
    } catch (e: ExecutionException) {
      throw e.cause ?: e
    }
  }
}

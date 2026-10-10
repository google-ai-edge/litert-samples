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

package com.google.ai.edge.examples.voice_assistant

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext

/**
 * Makes a native object with [create] on [dispatcher] and hands it to the caller, or closes it
 * with [close] when the caller was cancelled while [create] ran: a native call is not
 * interrupted, so it finishes either way.
 *
 * The nesting is what keeps the object. `withContext(dispatcher + NonCancellable)` still loses
 * it: the return to the caller's dispatcher can be cancelled, so a caller cancelled meanwhile
 * gets a CancellationException and the object is made and never closed (NativeHandoffTest).
 * Here the return happens inside NonCancellable, and the caller's cancellation is checked after
 * it, with the object in hand.
 */
internal suspend fun <T> handOver(
  dispatcher: CoroutineDispatcher,
  create: () -> T,
  close: suspend (T) -> Unit,
): T {
  currentCoroutineContext().ensureActive()
  val made = withContext(NonCancellable) { withContext(dispatcher) { create() } }
  try {
    currentCoroutineContext().ensureActive()
  } catch (e: CancellationException) {
    withContext(NonCancellable) { close(made) }
    throw e
  }
  return made
}

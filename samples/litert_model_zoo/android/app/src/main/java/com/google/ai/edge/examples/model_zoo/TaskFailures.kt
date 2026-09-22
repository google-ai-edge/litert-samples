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

package com.google.ai.edge.examples.model_zoo

import kotlinx.coroutines.CancellationException

/** Java/Kotlin/native-wrapper errors are recoverable UI messages; coroutine cancellation is not. */
object TaskFailures {
  fun message(failure: Throwable): String {
    if (failure is CancellationException) throw failure
    return if (failure is OutOfMemoryError)
      "Not enough memory to load or run this model. Close other tasks and try again."
    else failure.message?.takeIf { it.isNotBlank() } ?: failure.javaClass.simpleName
  }
}

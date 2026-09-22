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
import org.junit.Assert.*
import org.junit.Test

class TaskFailuresTest {
  @Test
  fun memoryFailureBecomesAnActionableMessage() {
    val message = TaskFailures.message(OutOfMemoryError("test fixture"))
    assertTrue(message.contains("Not enough memory"))
    assertTrue(message.contains("try again"))
  }

  @Test
  fun NativeWrapperJavaErrorRetainsItsCauseMessage() {
    assertEquals(
      "GPU compiler rejected graph",
      TaskFailures.message(LinkageError("GPU compiler rejected graph")),
    )
  }

  @Test
  fun failureWithoutMessageRemainsVisible() {
    assertEquals("IllegalStateException", TaskFailures.message(IllegalStateException()))
  }

  @Test(expected = CancellationException::class)
  fun navigationCancellationIsNotTurnedIntoAUserError() {
    TaskFailures.message(CancellationException("screen closed"))
  }
}

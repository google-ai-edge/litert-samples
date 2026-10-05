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

package com.google.ai.edge.examples.voice_assistant.loop

import android.content.ContextWrapper
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertThrows
import org.junit.Test

/** The tools' whole numbers: the model sees an exception's message as "Error: <message>". */
class ToolArgsTest {
  @Test
  fun sevenArrivesAsANumberOrAsText() {
    val args = mapOf<String, Any?>("a" to 7, "b" to 7.0, "c" to "7", "d" to " 7 ")
    assertEquals(listOf(7, 7, 7, 7), listOf("a", "b", "c", "d").map { ToolArgs.int(args, it) })
  }

  @Test
  fun aNumberThatIsNotWholeIsNotCutShort() {
    // These were 1, 0, 0 and Int.MAX_VALUE.
    val args = mapOf<String, Any?>("a" to 1.5, "b" to -0.5, "c" to "NaN", "d" to "Infinity")
    for ((key, v) in args) {
      val e = assertThrows(IllegalArgumentException::class.java) { ToolArgs.int(args, key) }
      assertEquals("$key '$v' must be a whole number", e.message)
    }
    val tooBig = assertThrows(IllegalArgumentException::class.java) {
      ToolArgs.int(mapOf("a" to 1e10), "a")
    }
    assertEquals("a '1.0E10' is out of range", tooBig.message)
    val missing = assertThrows(IllegalArgumentException::class.java) {
      ToolArgs.int(emptyMap(), "a")
    }
    assertEquals("missing a", missing.message)
  }

  @Test
  fun theTimerAndTheAlarmRefuseAPartOfAMinuteOrAnHour() {
    // Before, a 1-minute timer started, and hour -0.5 passed the 0-23 check as 00.
    val context = ContextWrapper(null)
    val timer =
      assertThrows(IllegalArgumentException::class.java) {
        runBlocking { TimerTool(context).call(mapOf("minutes" to 1.5, "label" to "Tea")) }
      }
    assertEquals("minutes '1.5' must be a whole number", timer.message)
    val alarm =
      assertThrows(IllegalArgumentException::class.java) {
        runBlocking { AlarmTool(context).call(mapOf("hour" to -0.5, "minute" to 0)) }
      }
    assertEquals("hour '-0.5' must be a whole number", alarm.message)
  }
}

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
import java.time.Instant
import java.util.TimeZone
import kotlinx.coroutines.runBlocking
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * What the calendar tools read from the model and what they send back; the provider itself needs
 * a phone. The model sees an exception's message as "Error: <message>".
 */
class CalendarToolTest {
  // The dates are read before the Context is touched, so a bare ContextWrapper will do.
  private val calendar = CalendarTool(ContextWrapper(null))

  private fun utc(text: String) = Instant.parse(text).toEpochMilli()

  @Test
  fun aTimeTheToolWouldMisreadGoesBackToTheModelWithTheFormat() {
    // A lenient format read these as 2026-10-08 01:00, 2026-03-02 10:00 and 09:30.
    for (bad in listOf("2026-10-05 73:00", "2026-02-30 10:00", "2026-10-05 9:30 PM")) {
      val args = mapOf("title" to "Lunch", "start" to bad, "end" to "2026-10-05 23:00")
      val e =
        assertThrows(IllegalArgumentException::class.java) {
          runBlocking { calendar.add.call(args) }
        }
      assertEquals("bad time '$bad' (use YYYY-MM-DD HH:MM)", e.message)
    }
    val e =
      assertThrows(IllegalArgumentException::class.java) {
        runBlocking { calendar.read.call(mapOf("date" to "2026-02-30")) }
      }
    assertEquals("bad date '2026-02-30' (use YYYY-MM-DD)", e.message)
  }

  @Test
  fun aTimeIsReadWithASpaceOrATAndWithOrWithoutSeconds() {
    val ten = calendar.parseMinute("2026-10-05 10:00")
    assertEquals(ten, calendar.parseMinute("2026-10-05T10:00"))
    assertEquals(ten, calendar.parseMinute(" 2026-10-05T10:00:00 "))
    assertEquals(ten, calendar.parseMinute("2026-10-05 10:00:00.000"))
    assertEquals(ten - 30 * 60_000L, calendar.parseMinute("2026-10-05 9:30"))
  }

  @Test
  fun aDayRunsFromItsMidnightToTheNextOnTheDaysTheClocksChange() {
    val newYork = TimeZone.getTimeZone("America/New_York")
    // 25 hours: 00:00 EDT to 00:00 EST (begin + 24 h ended at 23:00 and missed the last hour).
    val fallBack = utc("2026-11-01T04:00:00Z") to utc("2026-11-02T05:00:00Z")
    assertEquals(fallBack, calendar.dayWindow("2026-11-01", newYork))
    // 23 hours: 00:00 EST to 00:00 EDT (begin + 24 h ran into 01:00 of the next day).
    val springForward = utc("2026-03-08T05:00:00Z") to utc("2026-03-09T04:00:00Z")
    assertEquals(springForward, calendar.dayWindow("2026-03-08", newYork))
    val (begin, end) = calendar.dayWindow("2026-10-05", newYork)!!
    assertEquals(24 * 3_600_000L, end - begin)
    assertNull(calendar.dayWindow("tomorrow", newYork))
  }

  @Test
  fun theLocationIsOptionalAndAnEventWithoutOneIsSaidWithoutAt() {
    val declared = calendar.add.functionMap()["parameters"] as Map<*, *>
    assertEquals(listOf("title", "start", "end"), declared["required"])
    val start = calendar.parseMinute("2026-10-05 12:00")
    val end = calendar.parseMinute("2026-10-05 13:00")
    val said = "Event 'Lunch' added: 2026-10-05 12:00 to 2026-10-05 13:00"
    assertEquals(said, calendar.receipt("Lunch", start, end, ""))
    assertEquals("$said at Cafe", calendar.receipt("Lunch", start, end, "Cafe"))
  }

  @Test
  fun aMissingCalendarPermissionIsOnePlainSentence() {
    val denial =
      "Permission Denial: reading com.android.providers.calendar.CalendarProvider2 uri " +
        "content://com.android.calendar/instances/when/1/2 from pid=1, uid=2 requires " +
        "android.permission.READ_CALENDAR, or grantUriPermission()"
    val result = calendar.granted { throw SecurityException(denial) }
    assertEquals("Error: the calendar permission is not granted", result)
    // Said as one sentence; Android's own text would be cut at every dot of its names.
    assertEquals(1, SentenceSplitter.split(actionReceipt(result)).size)
    assertTrue(SentenceSplitter.split(actionReceipt("Error: $denial")).size > 5)
  }
}

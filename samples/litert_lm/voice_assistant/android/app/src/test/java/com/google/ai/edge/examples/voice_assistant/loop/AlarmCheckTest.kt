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
// voice/src/test/kotlin/io/github/johnrocky/hfmodels/voice/AlarmCheckTest.kt
// Its descriptor, model store and client layers are replaced by this sample's catalog and store.

package com.google.ai.edge.examples.voice_assistant.loop

import java.util.Calendar
import java.util.TimeZone
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

/** AlarmTool's reading of Android's next alarm clock; the tool itself needs a phone. */
class AlarmCheckTest {
  private val tokyo = TimeZone.getTimeZone("Asia/Tokyo")

  private fun at(y: Int, mo: Int, d: Int, h: Int, mi: Int, s: Int = 0): Long =
    Calendar.getInstance(tokyo)
      .apply {
        clear()
        set(y, mo - 1, d, h, mi, s)
      }
      .timeInMillis

  @Test
  fun theRequestedTimeIsItsNextOccurrence() {
    // 17:12 on Saturday: 07:30 is tomorrow morning, 21:30 tonight; the minute in progress is
    // tomorrow's.
    val now = at(2026, 10, 3, 17, 12, 20)
    assertEquals(at(2026, 10, 4, 7, 30), AlarmCheck.nextOccurrence(now, 7, 30, tokyo))
    assertEquals(at(2026, 10, 3, 21, 30), AlarmCheck.nextOccurrence(now, 21, 30, tokyo))
    assertEquals(at(2026, 10, 4, 17, 12), AlarmCheck.nextOccurrence(now, 17, 12, tokyo))
  }

  @Test
  fun androidsNextAlarmConfirmsItOrHidesIt() {
    val expected = at(2026, 10, 4, 7, 30)
    // What dumpsys alarm reported after the S26's turn: 2026-10-04 07:30:00.000.
    assertTrue(AlarmCheck.matches(expected, expected))
    assertFalse(AlarmCheck.matches(null, expected))
    assertFalse(AlarmCheck.matches(at(2026, 10, 4, 7, 31), expected))
    // An earlier alarm (06:00) stays next whatever the Clock app did; a later one or none does
    // not.
    assertTrue(AlarmCheck.hidden(at(2026, 10, 4, 6, 0), expected))
    assertFalse(AlarmCheck.hidden(at(2026, 10, 4, 8, 0), expected))
    assertFalse(AlarmCheck.hidden(null, expected))
  }

  @Test
  fun anAlarmAlreadyAtThatMinuteIsNotTakenForTheNewOne() {
    // Android reports one next alarm: the request cannot be told apart from it.
    val expected = at(2026, 10, 4, 6, 45)
    assertEquals(
      "Alarm requested for 06:45 (Wake Up). Android's next alarm was already Sun 06:45, " +
        "so this one could not be confirmed.",
      AlarmCheck.unconfirmed(expected, expected, 6, 45, "Wake Up", tokyo),
    )
  }

  @Test
  fun anEarlierAlarmIsNamedAndALaterOneHidesNothing() {
    val expected = at(2026, 10, 4, 7, 30)
    assertEquals(
      "Alarm requested for 07:30 (Wake Up). Android's next alarm is Sun 06:45, " +
        "so this one could not be confirmed.",
      AlarmCheck.unconfirmed(at(2026, 10, 4, 6, 45), expected, 7, 30, "Wake Up", tokyo),
    )
    assertNull(AlarmCheck.unconfirmed(at(2026, 10, 4, 8, 0), expected, 7, 30, "Wake Up", tokyo))
    assertNull(AlarmCheck.unconfirmed(null, expected, 7, 30, "Wake Up", tokyo))
  }
}

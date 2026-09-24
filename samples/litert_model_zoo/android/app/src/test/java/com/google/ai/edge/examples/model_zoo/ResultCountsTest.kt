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

import java.util.Locale
import org.junit.Assert.assertEquals
import org.junit.Test

class ResultCountsTest {
  @Test
  fun objectCountsUseEnglishWithAJapaneseDeviceLocale() {
    withJapaneseLocale {
      assertEquals("0 objects detected", ResultCounts.objects(0))
      assertEquals("1 object detected", ResultCounts.objects(1))
      assertEquals("2 objects detected", ResultCounts.objects(2))
    }
  }

  @Test
  fun allOtherCountsUseEnglishWithAJapaneseDeviceLocale() {
    withJapaneseLocale {
      assertEquals("1 box", ResultCounts.boxes(1))
      assertEquals("2 boxes", ResultCounts.boxes(2))
      assertEquals("1 tag", ResultCounts.tags(1))
      assertEquals("2 tags", ResultCounts.tags(2))
      assertEquals("1 note", ResultCounts.notes(1))
      assertEquals("2 notes", ResultCounts.notes(2))
      assertEquals("1 match", ResultCounts.matches(1))
      assertEquals("2 matches", ResultCounts.matches(2))
    }
  }

  private fun withJapaneseLocale(check: () -> Unit) {
    val previous = Locale.getDefault()
    try {
      Locale.setDefault(Locale.JAPAN)
      check()
    } finally {
      Locale.setDefault(previous)
    }
  }

  @Test
  fun tagsUseTheSingularOnlyForOne() {
    assertEquals("0 tags", ResultCounts.tags(0))
    assertEquals("1 tag", ResultCounts.tags(1))
    assertEquals("2 tags", ResultCounts.tags(2))
  }

  @Test
  fun boxesUseTheSingularOnlyForOne() {
    assertEquals("0 boxes", ResultCounts.boxes(0))
    assertEquals("1 box", ResultCounts.boxes(1))
    assertEquals("2 boxes", ResultCounts.boxes(2))
  }

  @Test
  fun notesUseTheSingularOnlyForOne() {
    assertEquals("0 notes", ResultCounts.notes(0))
    assertEquals("1 note", ResultCounts.notes(1))
    assertEquals("2 notes", ResultCounts.notes(2))
  }
}

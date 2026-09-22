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

package com.google.ai.edge.examples.model_zoo.models.basicpitch

import org.junit.Assert.assertEquals
import org.junit.Test

class BasicPitchMathTest {
  @Test
  fun onsetStartsNoteFramePosteriorSustainsAndMinimumDurationFilters() {
    val note = Array(6) { FloatArray(88) }
    val onset = Array(6) { FloatArray(88) }
    for (frame in 1..4) note[frame][48] = if (frame == 3) 0.9f else 0.6f
    onset[1][48] = 0.7f
    note[0][0] = 0.8f
    onset[0][0] = 0.9f
    val events = Transcriber.decode(note, onset)
    assertEquals(1, events.size)
    assertEquals(69, events[0].midi)
    assertEquals(Transcriber.FRAME_SEC, events[0].startSec, 0.0)
    assertEquals(5 * Transcriber.FRAME_SEC, events[0].endSec, 0.0)
    assertEquals(0.9f, events[0].amplitude, 0f)
  }
}

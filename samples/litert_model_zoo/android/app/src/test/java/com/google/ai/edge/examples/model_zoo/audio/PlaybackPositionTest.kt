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

import org.junit.Assert.assertEquals
import org.junit.Test

class PlaybackPositionTest {
  @Test
  fun frameHeadDeterminesElapsedInsteadOfWallClock() {
    val position = PlaybackPosition.fromPlaybackHead(11025, 131072, 22050)
    assertEquals(0.5f, position.elapsedSeconds, 0f)
    assertEquals(5.944308f, position.totalSeconds, 0.000001f)
  }

  @Test
  fun endOfOutputAndUnsignedHeadCannotExceedTotal() {
    val finished = PlaybackPosition.fromPlaybackHead(48000, 48000, 48000)
    assertEquals(1f, finished.elapsedSeconds, 0f)
    assertEquals(finished.totalSeconds, finished.elapsedSeconds, 0f)
    assertEquals(1f, PlaybackPosition.fromPlaybackHead(-1, 48000, 48000).elapsedSeconds, 0f)
  }

  @Test
  fun emptyPositionStartsAtZero() {
    assertEquals(PlaybackPosition(0f, 0f), PlaybackPosition.fromPlaybackHead(0, 0, 16000))
  }
}

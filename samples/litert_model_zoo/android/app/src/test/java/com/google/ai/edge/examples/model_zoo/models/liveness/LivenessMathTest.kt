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

package com.google.ai.edge.examples.model_zoo.models.liveness

import org.junit.Assert.*
import org.junit.Test

class LivenessMathTest {
  @Test
  fun liveClassIsIndexOneAndWinsTiesAsInZoo() {
    assertEquals(true to .8f, LivenessDetector.decode(floatArrayOf(.1f, .8f, .1f)))
    assertEquals(false to .2f, LivenessDetector.decode(floatArrayOf(.1f, .2f, .7f)))
    assertEquals(true to .5f, LivenessDetector.decode(floatArrayOf(.5f, .5f, 0f)))
  }
}

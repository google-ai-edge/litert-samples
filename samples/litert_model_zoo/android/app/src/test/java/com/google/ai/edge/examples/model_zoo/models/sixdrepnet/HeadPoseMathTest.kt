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

package com.google.ai.edge.examples.model_zoo.models.sixdrepnet

import org.junit.Assert.*
import org.junit.Test

class HeadPoseMathTest {
  @Test
  fun gramSchmidtMapsIdentityAndQuarterTurnToDegrees() {
    val identity = HeadPoseEstimator.decode(floatArrayOf(2f, 0f, 0f, 0f, 3f, 0f))
    assertEquals(0f, identity.yaw, 1e-5f)
    assertEquals(0f, identity.pitch, 1e-5f)
    assertEquals(0f, identity.roll, 1e-5f)
    val roll = HeadPoseEstimator.decode(floatArrayOf(0f, 1f, 0f, -1f, 0f, 0f))
    assertEquals(90f, roll.roll, 1e-5f)
    assertEquals(0f, roll.yaw, 1e-5f)
  }
}

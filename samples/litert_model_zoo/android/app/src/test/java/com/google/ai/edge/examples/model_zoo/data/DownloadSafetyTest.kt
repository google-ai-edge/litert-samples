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

package com.google.ai.edge.examples.model_zoo.data

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class DownloadSafetyTest {
  @Test
  fun storageRequiresTwiceFullSizeAtTheBoundaryEvenForAResume() {
    assertEquals(200L, DownloadSafety.requiredFreeBytes(100L))
    assertFalse(DownloadSafety.hasSpace(100L, 199L))
    assertTrue(DownloadSafety.hasSpace(100L, 200L))
  }

  @Test
  fun storageAccountsForConcurrentDownloadReservations() {
    assertFalse(DownloadSafety.hasSpace(100L, 249L, reservedBytes = 50L))
    assertTrue(DownloadSafety.hasSpace(100L, 250L, reservedBytes = 50L))
    assertFalse(DownloadSafety.hasSpace(100L, 49L, reservedBytes = 50L))
  }

  @Test(expected = IllegalArgumentException::class)
  fun invalidModelSizeCannotOverflowStorageRequirement() {
    DownloadSafety.requiredFreeBytes(Long.MAX_VALUE)
  }
}

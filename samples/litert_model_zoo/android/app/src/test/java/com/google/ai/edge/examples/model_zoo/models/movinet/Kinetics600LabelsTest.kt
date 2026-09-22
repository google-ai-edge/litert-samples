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

package com.google.ai.edge.examples.model_zoo.models.movinet

import java.security.MessageDigest
import org.junit.Assert.*
import org.junit.Test

class Kinetics600LabelsTest {
  @Test
  fun labelsMatchTheApprovedZooClassIndexOrder() {
    assertEquals(600, Kinetics600Labels.NAMES.size)
    assertEquals(600, Kinetics600Labels.NAMES.toSet().size)
    assertEquals("abseiling", Kinetics600Labels.NAMES.first())
    val digest = MessageDigest.getInstance("SHA-256")
      .digest(Kinetics600Labels.NAMES.joinToString("\n").toByteArray(Charsets.UTF_8))
      .joinToString("") { "%02x".format(it.toInt() and 255) }
    assertEquals("5d9a560f98fcfed08f6129dd498f97e043edf729efb3c02dd96b482d2970f9c0", digest)
  }
}

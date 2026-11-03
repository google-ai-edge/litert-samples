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

package com.google.ai.edge.examples.image_cutout

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * The host code that needs no model file: the fp16 table read and the instructions the app builds.
 */
class HostMathTest {
  @Test
  fun halfToFloatAllValues() {
    // every finite fp16 value against its definition (-1)^s * 2^(e - 15) * (1 + m / 1024),
    // subnormals m * 2^-24
    for (bits in 0 until 0x10000) {
      val s = if (bits and 0x8000 != 0) -1.0 else 1.0
      val e = (bits ushr 10) and 0x1F
      val m = bits and 0x3FF
      if (e == 0x1F) continue
      val want =
        if (e == 0) {
          s * m * Math.pow(2.0, -24.0)
        } else {
          s * Math.pow(2.0, (e - 15).toDouble()) * (1 + m / 1024.0)
        }
      assertEquals("bits $bits", want.toFloat(), Half.toFloat(bits), 0f)
    }
    assertTrue(Half.toFloat(0x7C00) == Float.POSITIVE_INFINITY && Half.toFloat(0x7E00).isNaN())
  }

  @Test
  fun promptsMatchFixtureInstructions() {
    // build_prompt for the app's tasks on the fixture queries (the public rows' instructions)
    assertEquals(
      "Locate all the instances that match the following description: " +
        "red rectangle. blue circle. green circle.",
      Prompts.instruction(Task.DETECT, "red rectangle, blue circle, green circle"),
    )
    assertEquals(
      "Locate the instances that match the following description and provide the alpha mattes: " +
        "the green circle.",
      Prompts.instruction(Task.CUT_OUT, "the green circle"),
    )
    assertEquals(
      "point to: airplane. windsock",
      Prompts.instruction(Task.POINT, "airplane, windsock"),
    )
    assertEquals(
      "How many red rectangles in the image?",
      Prompts.instruction(Task.COUNT_POINTS, "red rectangles"),
    )
    assertEquals(
      "How many blue circles in the image? Please count them with boxes.",
      Prompts.instruction(Task.COUNT_BOXES, "blue circles"),
    )
    assertEquals("detect all the word in the box format", Prompts.instruction(Task.READ_TEXT, ""))
    assertEquals(
      "Locate the following text in the image: EXIT 12",
      Prompts.instruction(Task.FIND_TEXT, "EXIT 12"),
    )
    assertEquals(
      "Locate the instances that match the following description: the yellow rectangle.",
      Prompts.instruction(Task.FIND, "the yellow rectangle"),
    )
  }
}

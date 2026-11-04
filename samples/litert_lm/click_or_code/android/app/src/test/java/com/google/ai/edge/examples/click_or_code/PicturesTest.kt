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

package com.google.ai.edge.examples.click_or_code

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The app's picture preparation against Pillow 13.0.0 (tools/make_picture_rows.py writes the
 * expected rows).
 */
class PicturesTest {
  private val rows =
    JSONObject(javaClass.classLoader!!.getResource("picture_rows.json")!!.readText())

  /** The pattern of tools/make_picture_rows.py: r = 7x + 3y, g = x*x + 5y, b = x xor y (mod 256). */
  private fun pattern(w: Int, h: Int): Rgb {
    val bytes = ByteArray(w * h * 3)
    for (y in 0 until h) {
      for (x in 0 until w) {
        val i = (y * w + x) * 3
        bytes[i] = ((7 * x + 3 * y) % 256).toByte()
        bytes[i + 1] = ((x.toLong() * x + 5 * y) % 256).toInt().toByte()
        bytes[i + 2] = ((x xor y) % 256).toByte()
      }
    }
    return Rgb(w, h, bytes)
  }

  @Test
  fun stretchMatchesPillow() {
    val stretch = rows.getJSONArray("stretch")
    for (i in 0 until stretch.length()) {
      val r = stretch.getJSONObject(i)
      val out = Pictures.stretch(pattern(r.getInt("w"), r.getInt("h")))
      assertEquals(1920, out.width)
      assertEquals(1088, out.height)
      assertEquals("${r.getInt("w")}x${r.getInt("h")}", r.getString("sha256"), out.sha256())
    }
  }

  @Test
  fun orientMatchesExifTranspose() {
    val orient = rows.getJSONArray("orient")
    for (i in 0 until orient.length()) {
      val r = orient.getJSONObject(i)
      val out = pattern(r.getInt("w"), r.getInt("h")).orient(r.getInt("orientation"))
      assertEquals("orientation ${r.getInt("orientation")}", r.getString("sha256"), out.sha256())
    }
  }
}

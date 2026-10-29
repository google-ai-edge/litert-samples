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

package com.google.ai.edge.examples.math_to_latex

import org.json.JSONObject
import org.junit.Assert.assertArrayEquals
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Test

/**
 * The app's picture preparation against Pillow 13.0.0 and the model card's page script, pixel for
 * pixel: the rows (`picture_rows.json`) come from tools/make_picture_rows.py.
 */
class PicturesTest {
  private val rows =
    JSONObject(javaClass.classLoader!!.getResource("picture_rows.json")!!.readText())
      .getJSONArray("rows")

  private fun rowsOf(op: String) =
    (0 until rows.length()).map { rows.getJSONObject(it) }.filter { it.getString("op") == op }

  /** The generator's test picture, byte for byte. */
  private fun pattern(w: Int, h: Int, seed: Int): Rgb {
    val out = ByteArray(w * h * 3)
    var state = seed.toLong() and 0x7FFFFFFF
    var i = 0
    for (y in 0 until h) {
      for (x in 0 until w) {
        state = (state * 1103515245L + 12345L) and 0x7FFFFFFF
        val noise = ((state shr 16) and 0x3F).toInt()
        val stroke = if ((x / 7 + y / 5) % 11 == 0) 170 else 0
        for (c in 0 until 3) {
          val v = (x * (c + 1) * 3 + y * (3 - c) * 2 + noise) % 256
          out[i++] = maxOf(0, v - stroke).toByte()
        }
      }
    }
    return Rgb(w, h, out)
  }

  @Test
  fun resizeMatchesPillow() {
    val cases = rowsOf("resize")
    assertEquals(8, cases.size)
    for (r in cases) {
      val img = pattern(r.getInt("w"), r.getInt("h"), r.getInt("seed"))
      val got = img.resize(r.getInt("ow"), r.getInt("oh")).sha256()
      assertEquals("resize ${r.getInt("w")}x${r.getInt("h")}", r.getString("sha256"), got)
    }
  }

  @Test
  fun padSquareMatchesThePageScript() {
    for (r in rowsOf("pad_square")) {
      val got = Pictures.padSquare(pattern(r.getInt("w"), r.getInt("h"), r.getInt("seed"))).sha256()
      assertEquals("pad_square ${r.getInt("w")}x${r.getInt("h")}", r.getString("sha256"), got)
    }
  }

  @Test
  fun prepareMatchesThePageScript() {
    for (r in rowsOf("prepare")) {
      val p = Pictures.prepare(pattern(r.getInt("w"), r.getInt("h"), r.getInt("seed")))
      val want = r.getJSONArray("sha256").let { a -> (0 until a.length()).map { a.getString(it) } }
      assertEquals(
        "pictures of ${r.getInt("w")}x${r.getInt("h")}",
        want,
        p.pictures.map { it.sha256() },
      )
      if (r.isNull("slice_cols")) {
        assertNull(p.sliceCols)
        assertEquals(emptyMap<String, Any>(), p.extraContext)
      } else {
        assertEquals(r.getInt("slice_cols"), p.sliceCols)
        assertEquals(mapOf("slice_cols" to r.getInt("slice_cols")), p.extraContext)
      }
    }
  }

  @Test
  fun gridMatchesTheProcessor() {
    for (r in rowsOf("grid")) {
      val got = Pictures.slicedGrid(r.getInt("w"), r.getInt("h"))
      if (r.isNull("grid")) {
        assertNull("grid of ${r.getInt("w")}x${r.getInt("h")}", got)
      } else {
        val g = r.getJSONArray("grid")
        assertArrayEquals(
          "grid of ${r.getInt("w")}x${r.getInt("h")}",
          intArrayOf(g.getInt(0), g.getInt(1)),
          got,
        )
      }
    }
  }

  @Test
  fun orientMatchesExifTranspose() {
    val cases = rowsOf("orient")
    assertEquals(8, cases.size)
    for (r in cases) {
      val got =
        pattern(r.getInt("w"), r.getInt("h"), r.getInt("seed")).orient(r.getInt("orientation"))
      val o = r.getInt("orientation")
      assertEquals("width, orientation $o", r.getInt("out_w"), got.width)
      assertEquals("height, orientation $o", r.getInt("out_h"), got.height)
      assertEquals("pixels, orientation $o", r.getString("sha256"), got.sha256())
    }
  }

  @Test
  fun cropKeepsTheBox() {
    val img = pattern(10, 8, 3)
    val c = img.crop(2, 1, 7, 6)
    assertEquals(5, c.width)
    assertEquals(5, c.height)
    for (y in 0 until 5) {
      for (x in 0 until 5) {
        for (k in 0 until 3) {
          assertEquals(img.bytes[((y + 1) * 10 + x + 2) * 3 + k], c.bytes[(y * 5 + x) * 3 + k])
        }
      }
    }
  }

  @Test(expected = IllegalArgumentException::class)
  fun cropOutsideThePictureFails() {
    pattern(10, 8, 3).crop(5, 5, 11, 8)
  }
}

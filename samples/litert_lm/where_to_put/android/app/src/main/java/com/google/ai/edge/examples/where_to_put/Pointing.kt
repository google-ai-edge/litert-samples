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

package com.google.ai.edge.examples.where_to_put

/**
 * A point in the picture as the model gives it: x and y from 0 to 1000 across the whole picture (x to the right, y
 * down), whatever the picture's size; [how] names the form it was read from.
 */
data class Spot(val x: Double, val y: Double, val how: String) {
  /** Inside the picture (the edges included). */
  val inPicture: Boolean get() = x in 0.0..1000.0 && y in 0.0..1000.0
}

/** The request sent to the model and the point read back from its answer. Plain Kotlin: no Android classes. */
object Pointing {
  /**
   * The model's own pointing request (LFM3-VL's documentation) with the thing to point to replaced by the free spot
   * for the user's request, which goes in as typed.
   */
  fun instruction(request: String): String =
    "Point to the free spot where the item should be placed for the request \"" + request.trim() + "\". " +
      "Answer in JSON as a list of points: [{\"point_2d\": [x, y], \"label\": \"<short label>\"}, ...], " +
      "with x and y as integers normalized to 0-1000."

  private val END_TOKENS = listOf("<|im_end|>", "<|endoftext|>", "<|pad|>", "</s>")
  private const val THINK_CLOSE = "</think>"
  private const val NUM = "-?[0-9]+(?:\\.[0-9]+)?"
  private val POINT_2D = Regex("\"point_2d\"\\s*:\\s*\\[\\s*($NUM)\\s*,\\s*($NUM)\\s*\\]")
  private val BBOX_2D = Regex("\"bbox_2d\"\\s*:\\s*\\[\\s*($NUM)\\s*,\\s*($NUM)\\s*,\\s*($NUM)\\s*,\\s*($NUM)\\s*\\]")
  private val GROUP = Regex("[\\[(]([^\\[\\]()]*)[\\])]")
  private val NUMBER = Regex(NUM)

  /** The answer: the text after the last thinking close, without end-of-turn tokens. */
  fun answerPart(text: String): String {
    var t = text
    for (e in END_TOKENS) t = t.replace(e, "")
    val close = t.lastIndexOf(THINK_CLOSE)
    if (close >= 0) t = t.substring(close + THINK_CLOSE.length)
    return t.trim()
  }

  /**
   * The first point in [answer]: a `"point_2d": [x, y]`; else the centre of a `"bbox_2d": [x1, y1, x2, y2]`; else the
   * first bracket or parenthesis group holding only numbers: two numbers are a point, four a box (its centre). The
   * last rule also reads a JSON the model left broken, and a `"point_2d"` given four numbers. Null when none is found.
   */
  fun read(answer: String): Spot? {
    firstPoint2d(answer)?.let { return it }
    BBOX_2D.find(answer)?.let { m ->
      val v = (1..4).map { m.groupValues[it].toDouble() }
      return Spot((v[0] + v[2]) / 2, (v[1] + v[3]) / 2, "bbox_2d centre")
    }
    for (g in GROUP.findAll(answer)) {
      val parts = g.groupValues[1].split(",").map { it.trim() }
      if (parts.isEmpty() || !parts.all { NUMBER.matches(it) }) continue
      val v = parts.map { it.toDouble() }
      if (v.size == 2) return Spot(v[0], v[1], "bare pair")
      if (v.size == 4) return Spot((v[0] + v[2]) / 2, (v[1] + v[3]) / 2, "bare box centre")
    }
    return null
  }

  /**
   * The first whole `"point_2d": [x, y]` in [answer], or null. While the answer streams in, the first one found in the
   * part that has arrived is also the first one in the whole answer (a match cannot hold another `"point_2d"`), so the
   * app can mark it before the answer ends and [read] gives the same point at the end.
   */
  fun firstPoint2d(answer: String): Spot? =
    POINT_2D.find(answer)?.let { m -> Spot(m.groupValues[1].toDouble(), m.groupValues[2].toDouble(), "point_2d") }

  /**
   * The pixel of a 1000 x 1000 region mask that holds the point (pixel (i, j) covers x in [i, i + 1) and y in
   * [j, j + 1); a point on the far edge, 1000, falls in the last pixel). Only for a point [Spot.inPicture].
   */
  fun maskPixel(spot: Spot): Pair<Int, Int> =
    Pair(minOf(MASK_SIDE - 1, spot.x.toInt()), minOf(MASK_SIDE - 1, spot.y.toInt()))

  const val MASK_SIDE = 1000
}

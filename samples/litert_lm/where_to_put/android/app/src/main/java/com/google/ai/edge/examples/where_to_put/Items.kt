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

import android.graphics.Path
import android.graphics.RectF

/** One shape of a drawing, in the drawing's units: filled, or with [stroke] > 0 a line of that width. */
class Layer(val color: Int, val path: String, val stroke: Float = 0f)

/**
 * A flat drawing of an item, drawn by the app (no image files). Its units run [w] x [h] with y down; ([spotX],
 * [spotY]) is the point on the surface under the item's middle, which goes on the model's point, and [footW] is the
 * width of what stands on the surface (the ring under the item follows it). [width] is the drawing's width in
 * thousandths of the photo's width, for a photo that does not say how big the item is there.
 */
class ItemDrawing(
  val name: String,
  val w: Float,
  val h: Float,
  val spotX: Float,
  val spotY: Float,
  val footW: Float,
  val width: Int,
  val layers: List<Layer>,
) {
  /** The layers as paths, built once. */
  val paths: List<Path> by lazy { layers.map { Items.path(it.path) } }
}

/** The items the app can draw, and which one a request names. */
object Items {
  private fun c(argb: Long) = argb.toInt()

  val MUG = ItemDrawing(
    name = "mug", w = 100f, h = 100f, spotX = 44f, spotY = 96f, footW = 64f, width = 110,
    layers = listOf(
      Layer(c(0xFFA8232D), "M 74 30 C 102 26 102 80 74 76 L 74 66 C 89 68 89 38 74 42 Z"),
      Layer(c(0xFFD9343F), "M 12 16 L 12 92 C 12 101.3 76 101.3 76 92 L 76 16 Z"),
      Layer(c(0xFFA8232D), "M 58 16 L 76 16 L 76 92 C 76 96 69 98.5 58 98.9 Z"),
      Layer(c(0x66FFFFFF), "M 19 26 L 24 26 L 24 84 L 19 84 Z"),
      Layer(c(0xFFEE8A90), "E 44 16 32 7"),
      Layer(c(0xFF4A2A1A), "E 44 16.6 28 5.4"),
      Layer(c(0xFF6B4430), "E 36 15.6 9 1.8"),
    ),
  )

  val BOOK = ItemDrawing(
    name = "book", w = 100f, h = 42f, spotX = 50f, spotY = 30f, footW = 92f, width = 200,
    layers = listOf(
      Layer(c(0xFF2E64C8), "M 14 6 L 86 6 L 96 28 L 4 28 Z"),
      Layer(c(0xFFE8EEF9), "M 24 11 L 76 11 L 78 15 L 22 15 Z"),
      Layer(c(0xFFE8EEF9), "M 30 18 L 70 18 L 71 20.5 L 29 20.5 Z"),
      Layer(c(0xFF1F4A99), "M 4 28 L 96 28 L 96 30.5 L 4 30.5 Z"),
      Layer(c(0xFFF3EBD8), "M 5 30.5 L 95 30.5 L 95 38 L 5 38 Z"),
      Layer(c(0xFFCFC3A8), "M 6 33 L 94 33 M 6 35.5 L 94 35.5", stroke = 0.8f),
      Layer(c(0xFF1F4A99), "M 4 38 L 96 38 L 96 41 L 4 41 Z"),
    ),
  )

  val VASE = ItemDrawing(
    name = "vase", w = 70f, h = 120f, spotX = 35f, spotY = 118f, footW = 30f, width = 120,
    layers = listOf(
      Layer(c(0xFF3F8F3A), "M 35 62 Q 34 40 24 24 M 35 62 Q 36 36 38 14 M 35 62 Q 40 42 52 26", stroke = 2.2f),
      Layer(c(0xFF4FA34A), "M 34 50 Q 25 43 20 47 Q 27 54 34 50 Z"),
      Layer(c(0xFF4FA34A), "M 37 46 Q 46 38 51 42 Q 44 50 37 46 Z"),
      Layer(c(0xFFF2C230), "E 24 22 7 7"),
      Layer(c(0xFFC77A12), "E 24 22 2.6 2.6"),
      Layer(c(0xFFE8578A), "E 38 12 7.5 7.5"),
      Layer(c(0xFFA8325A), "E 38 12 2.8 2.8"),
      Layer(c(0xFFF08A3C), "E 52 24 6.5 6.5"),
      Layer(c(0xFFB2541A), "E 52 24 2.4 2.4"),
      Layer(c(0xFF1F8A8A), "M 26 60 L 44 60 L 44 66 C 60 76 62 104 50 118 L 20 118 C 8 104 10 76 26 66 Z"),
      Layer(c(0xFF156B6B), "M 44 66 C 60 76 62 104 50 118 L 42 118 C 50 104 50 78 40 68 Z"),
      Layer(c(0x55FFFFFF), "M 22 80 C 19 90 20 102 25 110 L 27.5 110 C 23.5 100 23.5 90 25.5 80 Z"),
      Layer(c(0xFF2BA3A3), "E 35 60 11 3"),
      Layer(c(0xFF0F4F4F), "E 35 60 8 2"),
    ),
  )

  val BAG = ItemDrawing(
    name = "bag", w = 100f, h = 96f, spotX = 50f, spotY = 94f, footW = 88f, width = 220,
    layers = listOf(
      Layer(c(0xFF6B3A1E), "M 30 42 C 30 4 70 4 70 42", stroke = 4.5f),
      Layer(c(0xFFC97B3B), "M 14 40 L 86 40 L 94 88 C 94 93 90 95 86 95 L 14 95 C 10 95 6 93 6 88 Z"),
      Layer(c(0xFFA8612A), "M 76 40 L 86 40 L 94 88 C 94 93 90 95 86 95 L 80 95 Z"),
      Layer(c(0xFFA8612A), "M 14 40 L 86 40 L 87.5 50 L 12.5 50 Z"),
      Layer(c(0xFFE9C46A), "E 50 52 5 4"),
    ),
  )

  val PAN = ItemDrawing(
    name = "pan", w = 150f, h = 54f, spotX = 52f, spotY = 34f, footW = 92f, width = 300,
    layers = listOf(
      Layer(c(0xFF2B2B2B), "M 94 24 L 145 13 C 150 12 152 18.5 147 20 L 97 33 Z"),
      Layer(c(0xFF6E7781), "E 52 33 46 18"),
      Layer(c(0xFFC3CAD2), "E 52 28 46 17"),
      Layer(c(0xFF3B4048), "E 52 29 40 13.5"),
      Layer(c(0xFF59606A), "E 40 25 14 3.5"),
    ),
  )

  val SUGAR_BOWL = ItemDrawing(
    name = "sugar bowl", w = 80f, h = 78f, spotX = 40f, spotY = 77f, footW = 40f, width = 110,
    layers = listOf(
      Layer(c(0xFFE4E8EE), "M 18 46 C 4 44 4 62 20 60 L 20 56 C 10 57 10 49 18 50 Z"),
      Layer(c(0xFFE4E8EE), "M 62 46 C 76 44 76 62 60 60 L 60 56 C 70 57 70 49 62 50 Z"),
      Layer(c(0xFFD9DEE5), "M 30 72 L 50 72 L 52 78 L 28 78 Z"),
      Layer(c(0xFFF7F9FB), "M 14 40 C 14 70 26 76 40 76 C 54 76 66 70 66 40 Z"),
      Layer(c(0xFFDDE2E9), "M 52 40 L 66 40 C 66 70 54 76 40 76 C 52 72 56 60 52 40 Z"),
      Layer(c(0xFF2F5FB3), "M 15.5 52 C 18 60 28 63 40 63 C 52 63 62 60 64.5 52 L 64 56 C 60 64 52 67 40 67 C 28 67 20 64 16 56 Z"),
      Layer(c(0xFFF7F9FB), "M 12 40 C 14 24 66 24 68 40 Z"),
      Layer(c(0xFFDDE2E9), "E 40 40 28 4"),
      Layer(c(0xFF2F5FB3), "E 40 23 5 4"),
    ),
  )

  /** The words that name each drawing (a cup is drawn as the mug); "sugar bowl" before anything a "bowl" could hit. */
  private val NAMES = listOf(
    "sugar bowls?" to SUGAR_BOWL,
    "mugs?|cups?|teacups?" to MUG,
    "books?|notebooks?" to BOOK,
    "vases?" to VASE,
    "bags?|handbags?|totes?" to BAG,
    "pans?|frying pans?|skillets?" to PAN,
  )
  private val WORD = Regex("\\b(" + NAMES.joinToString("|") { it.first } + ")\\b", RegexOption.IGNORE_CASE)

  /**
   * The drawing for the item [request] names: the first word in it that names a drawing ("Put the mug in front of the
   * books." -> the mug). Null when no word does; the app then marks the point with a ring.
   */
  fun forRequest(request: String): ItemDrawing? {
    val word = WORD.find(request)?.value ?: return null
    return NAMES.first { Regex("^(" + it.first + ")$", RegexOption.IGNORE_CASE).matches(word) }.second
  }

  /**
   * A path from the drawings' short form: `M x y` (move), `L x y` (line), `Q x1 y1 x y` and `C x1 y1 x2 y2 x y`
   * (curves), `Z` (close), and `E cx cy rx ry` (an ellipse), absolute coordinates separated by spaces.
   */
  fun path(spec: String): Path {
    val t = spec.trim().split(Regex("[\\s,]+"))
    val p = Path()
    var i = 0
    fun n() = t[i++].toFloat()
    while (i < t.size) {
      when (val cmd = t[i++]) {
        "M" -> p.moveTo(n(), n())
        "L" -> p.lineTo(n(), n())
        "Q" -> p.quadTo(n(), n(), n(), n())
        "C" -> p.cubicTo(n(), n(), n(), n(), n(), n())
        "Z" -> p.close()
        "E" -> {
          val cx = n(); val cy = n(); val rx = n(); val ry = n()
          p.addOval(RectF(cx - rx, cy - ry, cx + rx, cy + ry), Path.Direction.CW)
        }
        else -> error("unknown path command $cmd in $spec")
      }
    }
    return p
  }
}

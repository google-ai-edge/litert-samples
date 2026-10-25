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

// Port of WeVisOne-VL's parse_seg_boxes and parse_predictions (Tencent, Apache-2.0; see NOTICE).
package com.google.ai.edge.examples.image_cutout

/** One shape of an answer in the model's bins (0..1000 on each axis). */
data class Shape(val label: String, val kind: String, val bins: IntArray) {
  val isBox: Boolean
    get() = kind == "box"

  val isPoint: Boolean
    get() = kind == "point"
}

/**
 * The model's answer text read back into shapes, with the package's own rules: [segBoxes] is
 * `engine.parse_seg_boxes` (the boxes right before `<OBJ>`, which the mask decoder cuts out) and
 * [shapes] is `visualization.parse_predictions` (every `<ref>label</ref><box|point|polygon>...`
 * group, degenerate boxes dropped).
 */
class Answer(raw: String) {
  /** The answer up to the end-of-turn token. */
  val text: String = raw.substringBefore("<|im_end|>").substringBefore("<|endoftext|>")

  val segBoxes: List<IntArray> = parseSegBoxes(raw)
  val shapes: List<Shape> = parseShapes(text)

  /** The "Total number: N." a count answer ends with, or null. */
  val statedCount: Int? =
    Regex("Total number:\\s*(\\d+)").findAll(text).lastOrNull()?.groupValues?.get(1)?.toInt()

  /** True when the answer names no box, point or polygon (e.g. `<box>none</box>`). */
  val isEmpty: Boolean
    get() = shapes.isEmpty() && segBoxes.isEmpty()

  /**
   * The answer without the coordinate tokens, for the screen: "<ref>EXIT</ref><box>...</box>" ->
   * "EXIT".
   */
  fun labels(): List<String> = shapes.map { it.label }.distinct()

  companion object {
    private val BOX_RE =
      Regex("<box>(.*?)(?:</box>|$)", setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL))
    private val COORD_RE = Regex("<(\\d{1,4})>")
    private val SEG_RE = Regex("<OBJ>", RegexOption.IGNORE_CASE)
    private val REF_RE =
      Regex(
        "<ref>(.*?)</ref>\\s*<(box|point|polygon)>(.*?)(?:</(?:box|point|polygon)>|<ref>|$)",
        setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL),
      )

    /** engine.parse_seg_boxes, in bins: the last four numbers before each <OBJ> inside a <box>. */
    fun parseSegBoxes(raw: String): List<IntArray> {
      val text = raw.substringBefore("<|im_end|>")
      val boxes = ArrayList<IntArray>()
      for (m in BOX_RE.findAll(text)) {
        val parts = SEG_RE.split(m.groupValues[1])
        for (prefix in parts.dropLast(1)) {
          val numbers = COORD_RE.findAll(prefix).map { it.groupValues[1].toInt() }.toList()
          if (numbers.size < 4) continue
          val (x1, y1, x2, y2) = numbers.takeLast(4)
          if (listOf(x1, y1, x2, y2).all { it in 0..1000 } && x2 > x1 && y2 > y1) {
            boxes.add(intArrayOf(x1, y1, x2, y2))
          }
        }
      }
      return boxes
    }

    /** visualization.parse_predictions in bins (labels stripped, an empty label is "object"). */
    fun parseShapes(text: String): List<Shape> {
      val out = ArrayList<Shape>()
      for (m in REF_RE.findAll(text)) {
        val label = m.groupValues[1].trim().ifEmpty { "object" }
        val kind = m.groupValues[2].lowercase()
        val size = if (kind == "point") 2 else 4
        for (segment in m.groupValues[3].split(",")) {
          val values = COORD_RE.findAll(segment).map { it.groupValues[1].toInt() }.toList()
          var offset = 0
          while (offset + size <= values.size) {
            val chunk = values.subList(offset, offset + size).toIntArray()
            offset += size
            if (size == 4 && (chunk[2] <= chunk[0] || chunk[3] <= chunk[1])) continue
            out.add(Shape(label, if (kind == "polygon") "box" else kind, chunk))
          }
        }
      }
      return out
    }
  }
}

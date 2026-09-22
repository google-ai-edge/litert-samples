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

package com.google.ai.edge.examples.model_zoo.image

import kotlin.math.abs

/** Display formatting only. Text and box coordinates returned by the recognizer are unchanged. */
data class OcrDisplayBox(
  val left: Int,
  val top: Int,
  val right: Int,
  val bottom: Int,
  val text: String,
) {
  val centerY: Float
    get() = (top + bottom) / 2f

  val height: Int
    get() = (bottom - top).coerceAtLeast(1)
}

object OcrReadingOrder {
  fun lines(boxes: List<OcrDisplayBox>): List<String> {
    val rows = mutableListOf<MutableList<OcrDisplayBox>>()
    for (box in boxes.sortedWith(compareBy({ it.centerY }, { it.left }))) {
      val row =
        rows
          .filter { candidates ->
            val center = candidates.map { it.centerY }.average().toFloat()
            val height = candidates.map { it.height }.average().toFloat()
            abs(box.centerY - center) <= minOf(box.height.toFloat(), height) / 2f
          }
          .minByOrNull { candidates -> abs(box.centerY - candidates.map { it.centerY }.average()) }
      if (row == null) rows.add(mutableListOf(box)) else row.add(box)
    }
    return rows
      .sortedBy { row -> row.map { it.centerY }.average() }
      .map { row -> row.sortedBy { it.left }.joinToString(" ") { it.text } }
  }
}

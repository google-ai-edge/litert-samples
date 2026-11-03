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

import java.io.File
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test

/**
 * The prompt the app builds equals the reference's prompt ids (the HF processor) on the committed
 * rows, and the reference answer ids decode to the reference's answer text, which the app reads
 * back into the same boxes and shapes as the package's parsers. Needs the checkpoint's
 * tokenizer.json: `-Pwevisone.models=<dir with the model files>`; skipped without it.
 */
class TokenizerTest {
  private val moduleRoot = File(System.getProperty("wevisone.moduleRoot") ?: ".")

  private fun tokenizer(): QwenBpeTokenizer? =
    System.getProperty("wevisone.models")
      ?.let { File(it, ModelFiles.TOKENIZER.name) }
      ?.takeIf { it.isFile }
      ?.let { QwenBpeTokenizer.fromFile(it) }

  @Test
  fun committedRows() {
    val tok = tokenizer()
    assumeTrue("needs -Pwevisone.models=<dir with tokenizer.json>", tok != null)
    val rows =
      JSONObject(File(moduleRoot, "app/src/test/resources/tok_rows_public.json").readText())
        .getJSONArray("rows")
    var promptOk = 0
    var decodeOk = 0
    var parseOk = 0
    for (i in 0 until rows.length()) {
      val r = rows.getJSONObject(i)
      val want = r.getJSONArray("prompt_ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
      val got =
        tok!!.encode(Prompts.BEFORE_IMAGE) +
          IntArray(WeVisOneEngine.N_IMAGE) { WeVisOneEngine.IMAGE_PAD } +
          tok.encode(Prompts.afterImage(r.getString("instruction")))
      if (got.contentEquals(want)) {
        promptOk++
      } else {
        println("PROMPT DIFF ${r.getString("id")}: ${got.toList()} vs ${want.toList()}")
      }
      val newIds = r.getJSONArray("new_ids").let { a -> IntArray(a.length()) { a.getInt(it) } }
      val text = tok.decode(newIds)
      if (text == r.getString("response")) {
        decodeOk++
      } else {
        println("DECODE DIFF ${r.getString("id")}: $text")
      }
      val answer = Answer(text)
      val seg =
        r.getJSONArray("seg_boxes").let { a ->
          (0 until a.length()).map { k ->
            a.getJSONArray(k).let { b -> List(b.length()) { b.getInt(it) } }
          }
        }
      val shapes =
        r.getJSONArray("shapes").let { a ->
          (0 until a.length()).map { k ->
            val s = a.getJSONObject(k)
            val kind = s.getString("kind").let { if (it == "polygon") "box" else it }
            Triple(
              s.getString("label"),
              kind,
              s.getJSONArray("bins").let { b -> List(b.length()) { b.getInt(it) } },
            )
          }
        }
      val mineSeg = answer.segBoxes.map { it.toList() }
      val mineShapes = answer.shapes.map { Triple(it.label, it.kind, it.bins.toList()) }
      // the package groups shapes by label (dict order = first appearance); compare as such
      if (mineSeg == seg && mineShapes.groupBy { it.first } == shapes.groupBy { it.first }) {
        parseOk++
      } else {
        println("PARSE DIFF ${r.getString("id")}: $mineSeg / $mineShapes vs $seg / $shapes")
      }
    }
    println(
      "TOKENIZER: prompt ids $promptOk/${rows.length()}, " +
        "answers decode $decodeOk/${rows.length()}, parsed $parseOk/${rows.length()}"
    )
    assertEquals("prompt ids", rows.length(), promptOk)
    assertEquals("decode", rows.length(), decodeOk)
    assertEquals("parse", rows.length(), parseOk)
  }

  @Test
  fun tokenizerCounts() {
    val tok = tokenizer()
    assumeTrue("needs -Pwevisone.models=<dir with tokenizer.json>", tok != null)
    println("TOKENIZER vocab ${tok!!.vocabSize} merges ${tok.mergeCount} added ${tok.addedCount}")
    assertEquals(151_643, tok.vocabSize)
    assertEquals(1_036, tok.addedCount)
    assertEquals(152_678, tok.addedId("<OBJ>"))
  }
}

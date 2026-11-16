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

package com.google.ai.edge.examples.ocr_verification

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Test

/**
 * The page fragments against the source renderer's (render_html_public.json: the source's
 * prepare_render_content with Python-Markdown 3.10.3 on the strings there).
 */
class RenderPageTest {
  @Test
  fun fragmentsMatchTheSourceRenderer() {
    val doc = JSONObject(javaClass.classLoader!!.getResource("render_html_public.json")!!.readText())
    val cases = doc.getJSONArray("cases")
    val failures = ArrayList<String>()
    for (i in 0 until cases.length()) {
      val c = cases.getJSONObject(i)
      val category = RenderPage.Category.valueOf(c.getString("category"))
      val got = RenderPage.content(c.getString("text"), category)
      if (got != c.getString("html")) {
        failures.add("${c.getString("text")}\n  want ${c.getString("html")}\n  got  $got")
      }
    }
    println("RENDER_HTML ${cases.length() - failures.size}/${cases.length()} equal")
    failures.forEach { println("RENDER_HTML_DIFF $it") }
    assertEquals(failures.joinToString("\n"), 0, failures.size)
  }
}

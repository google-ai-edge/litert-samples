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

package com.google.ai.edge.examples.receipt_to_json

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test

class KieTest {
  private val rows = JSONObject(javaClass.classLoader!!.getResource("picture_rows.json")!!.readText())

  @Test
  fun promptIsTheCardScriptsPrompt() {
    val keys = rows.getJSONArray("kie_keys").let { a -> List(a.length()) { Kie.Key(a.getString(it)) } }
    assertEquals(rows.getString("kie_prompt"), Kie.prompt(keys))
  }

  @Test
  fun defaultKeysGiveTheCardScriptsPrompt() {
    val keys =
      rows.getJSONArray("kie_default_keys").let { a -> List(a.length()) { Kie.Key(a.getString(it)) } }
    assertEquals(keys, Kie.DEFAULT_KEYS)
    assertEquals(rows.getString("kie_default_prompt"), Kie.prompt(Kie.DEFAULT_KEYS))
    assertEquals(Kie.DEFAULT_KEYS, Kie.parseKeys(Kie.DEFAULT_KEYS.joinToString(", ")))
  }

  @Test
  fun aListKeyGivesTheJsonDumpsSchema() {
    val keys = Kie.parseKeys("items[name, price], total")
    assertEquals(listOf(Kie.Key("items", listOf("name", "price")), Kie.Key("total")), keys)
    assertEquals(rows.getString("kie_schema_prompt"), Kie.prompt(keys))
    assertEquals(keys, Kie.parseKeys(keys.joinToString(", ")))
  }

  @Test
  fun keysAreSplitTrimmedAndDeduplicated() {
    assertEquals(
      listOf(Kie.Key("store_name"), Kie.Key("date"), Kie.Key("total")),
      Kie.parseKeys(" store_name, date,\ntotal, date ,"),
    )
    assertEquals(
      listOf(Kie.Key("items", listOf("name", "qty", "price")), Kie.Key("total")),
      Kie.parseKeys("items[ name, qty,\nprice, ], total"),
    )
    assertEquals(emptyList<Kie.Key>(), Kie.parseKeys(" , \n"))
    for ((text, word) in
      listOf(
        "total, \"tip\"" to "quote",
        "items[name, \"price\"]" to "quote",
        "items[name, price" to "pair",
        "items]name" to "pair",
        "items[a[b]]" to "pair",
        "items[]" to "fields",
        "[name, price]" to "name",
        "items[name] x" to "name[field",
      )) {
      try {
        Kie.parseKeys(text)
        fail("accepted: $text")
      } catch (e: IllegalArgumentException) {
        assertTrue("$text -> ${e.message}", e.message!!.contains(word))
      }
    }
  }

  @Test
  fun firstJsonObjectSkipsBracesInStrings() {
    val answer = "```json\n{\"store_name\": \"A {B}\", \"total\": \"9.99\"}\n```\nmore {x}"
    assertEquals("{\"store_name\": \"A {B}\", \"total\": \"9.99\"}", Kie.firstJsonObject(answer))
    assertNull(Kie.firstJsonObject("no object here"))
    assertNull(Kie.firstJsonObject("{\"open\": \"never closed\""))
  }

  @Test
  fun keysFoundAndRowsReadTheAnswer() {
    val answer =
      "{\n  \"items\": [{\"name\": \"FLEA COMB\", \"price\": \"6.49\"}, " +
        "{\"name\": \"CATNIP 1OZ\", \"price\": \"3.99\"}],\n  \"total\": \"10.48\"\n}"
    val listKeys = Kie.parseKeys("items[name, price], total")
    assertEquals(2, Kie.keysFound(answer, listKeys))
    assertEquals(listOf("items" to 2), Kie.rows(answer, listKeys))
    // a list key answered with a value is not found
    assertEquals(1, Kie.keysFound("{\"items\": \"none\", \"total\": \"0\"}", listKeys))
    assertEquals(2, Kie.keysFound(answer, Kie.parseKeys("items, total, date")))
    assertEquals(1, Kie.keysFound(answer, Kie.DEFAULT_KEYS))
    assertEquals(emptyList<Pair<String, Int>>(), Kie.rows(answer, Kie.DEFAULT_KEYS))
    assertNull(Kie.keysFound("{not json}", listKeys))
    assertEquals(emptyList<Pair<String, Int>>(), Kie.rows("{not json}", listKeys))
  }
}

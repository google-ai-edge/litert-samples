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

import org.json.JSONArray
import org.json.JSONObject

/**
 * Key information extraction as the model card asks for it ("Task prompts": `Extract key
 * information in the image`, followed by a JSON schema), and the reading of the answer.
 */
object Kie {
  const val HEAD = "Extract key information in the image"

  /**
   * One key of the schema: a value (`total`), or a list with one object per row and [fields] in
   * each (`items[name, price]`).
   */
  data class Key(val name: String, val fields: List<String> = emptyList()) {
    val isList: Boolean
      get() = fields.isNotEmpty()

    override fun toString() = if (isList) "$name[${fields.joinToString(", ")}]" else name
  }

  /** The keys the app starts with; the editor changes them. */
  val DEFAULT_KEYS = listOf(Key("store_name"), Key("date"), Key("tax"), Key("total"))

  /**
   * The prompt for [keys]: the schema with an empty string for every value, written as Python's
   * `json.dumps(schema, indent=4)` writes it (for keys without fields, the text of
   * `ocr_xiaomi.py --task kie`).
   */
  fun prompt(keys: List<Key>): String {
    val body =
      "{\n" +
        keys.joinToString(",\n") { k ->
          if (!k.isList) {
            "    \"${k.name}\": \"\""
          } else {
            "    \"${k.name}\": [\n        {\n" +
              k.fields.joinToString(",\n") { "            \"$it\": \"\"" } +
              "\n        }\n    ]"
          }
        } +
        "\n}"
    return "$HEAD\n\nPlease output the key information in JSON format according to the following " +
      "schema:\n$body"
  }

  /**
   * The keys in the editor's text: separated by commas or new lines (not those inside brackets),
   * trimmed, blanks and repeated names dropped; `name[field, field]` is a list key. A key with a
   * quote or a backslash would break the schema, so it is refused, as are brackets that do not
   * pair up.
   */
  fun parseKeys(text: String): List<Key> {
    val parts = mutableListOf<String>()
    val part = StringBuilder()
    var depth = 0
    for (c in text) {
      when {
        c == '[' -> depth++
        c == ']' -> depth--
      }
      require(depth in 0..1) { "Brackets do not pair up: $text" }
      if ((c == ',' || c == '\n') && depth == 0) {
        parts.add(part.toString())
        part.clear()
      } else {
        part.append(c)
      }
    }
    require(depth == 0) { "Brackets do not pair up: $text" }
    parts.add(part.toString())
    val keys =
      parts
        .map { it.trim() }
        .filter { it.isNotEmpty() }
        .map { p ->
          val open = p.indexOf('[')
          val key =
            if (open < 0) {
              Key(p)
            } else {
              require(p.endsWith("]")) { "Write a list key as name[field, field]: $p" }
              val fields =
                p.substring(open + 1, p.length - 1)
                  .split(',')
                  .map { it.trim() }
                  .filter { it.isNotEmpty() }
                  .distinct()
              require(fields.isNotEmpty()) {
                "A list key needs its fields, for example items[name, price]: $p"
              }
              Key(p.substring(0, open).trim(), fields)
            }
          require(key.name.isNotEmpty()) { "A list key needs a name before the bracket: $p" }
          (listOf(key.name) + key.fields)
            .firstOrNull { k -> k.any { it == '"' || it == '\\' } }
            ?.let { throw IllegalArgumentException("A key cannot hold a quote or a backslash: $it") }
          key
        }
    return keys.distinctBy { it.name }
  }

  /** The text of the first `{...}` object in [answer] (strings and escapes respected), or null. */
  fun firstJsonObject(answer: String): String? {
    val start = answer.indexOf('{')
    if (start < 0) return null
    var depth = 0
    var inString = false
    var escaped = false
    for (i in start until answer.length) {
      val c = answer[i]
      if (inString) {
        when {
          escaped -> escaped = false
          c == '\\' -> escaped = true
          c == '"' -> inString = false
        }
      } else {
        when (c) {
          '"' -> inString = true
          '{' -> depth++
          '}' -> {
            depth--
            if (depth == 0) return answer.substring(start, i + 1)
          }
        }
      }
    }
    return null
  }

  /**
   * How many of [keys] the answer's JSON object holds (a list key only with a list), or null when
   * it holds no object that parses.
   */
  fun keysFound(answer: String, keys: List<Key>): Int? {
    val obj = answerObject(answer) ?: return null
    return keys.count { k -> obj.has(k.name) && (!k.isList || obj.opt(k.name) is JSONArray) }
  }

  /** The number of rows the answer gives each list key it holds, in the order of [keys]. */
  fun rows(answer: String, keys: List<Key>): List<Pair<String, Int>> {
    val obj = answerObject(answer) ?: return emptyList()
    return keys.filter { it.isList }.mapNotNull { k ->
      (obj.opt(k.name) as? JSONArray)?.let { k.name to it.length() }
    }
  }

  private fun answerObject(answer: String): JSONObject? {
    val text = firstJsonObject(answer) ?: return null
    return runCatching { JSONObject(text) }.getOrNull()
  }
}
